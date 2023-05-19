"""Implement a DICOM router"""
from __future__ import annotations
import asyncio, logging
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy, copy
from typing import (
    Optional,
    Tuple,
    Dict,
    List,
    Union,
    Iterable,
    AsyncIterator,
    AsyncContextManager,
    Any,
)
from contextlib import asynccontextmanager

import janus
from pydicom import Dataset

from .lazyset import LazySet, FrozenLazySet
from .query import (
    QueryLevel,
    QueryResult,
    DataNode,
    get_uid,
    minimal_copy,
)
from .filt import Filter
from .reports.net_report import IncomingErrorType, DicomOpReport
from .route import (
    Route,
    StaticRoute,
    DynamicRoute,
    NoValidTransferMethodError,
    StaticRoute,
)
from .reports.xfer_report import DynamicTransferReport
from .store.base import DataBucket, DataRepo, TransferMethod


log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Entry in a SendAssociationCache"""

    ctx_mgr: AsyncContextManager["janus._AsyncQueueProxy[Dataset]"]
    send_q: "janus._AsyncQueueProxy[Dataset]"
    op_report: DicomOpReport
    last_use: datetime


# TODO: Make generic association caching in `net` module supporting
#       query/move/send. Could then use that everywhere, and use it to
#       manage max association limits on any node.
class SendAssociationCache:
    def __init__(self, timeout: float = 30.0):
        """Keeps cache of recent associations"""
        self._timeout = timeout
        self._cache: Dict[DataBucket[Any, Any], _CacheEntry] = {}

    @property
    def next_timeout(self) -> float:
        """Number of seconds until the next cache entry will timeout"""
        next_timeout = self._timeout
        now = datetime.now()
        for cache_entry in self._cache.values():
            td = now - cache_entry.last_use
            timeout = max(self._timeout - td.total_seconds(), 0)
            if timeout < next_timeout:
                next_timeout = timeout
        return next_timeout

    async def send(
        self, ds: Dataset, dest: DataBucket[Any, Any]
    ) -> Optional[DicomOpReport]:
        """Send a data set to dests, utilizing the cache of active associations"""
        res = None
        cache_entry = self._cache.get(dest, None)
        if cache_entry is None:
            op_report = dest.get_empty_send_report()
            res = op_report
            ctx_mgr = dest.send(op_report)
            send_q = await ctx_mgr.__aenter__()
            cache_entry = _CacheEntry(ctx_mgr, send_q, op_report, datetime.now())
            self._cache[dest] = cache_entry
        else:
            cache_entry.last_use = datetime.now()
            send_q = cache_entry.send_q
        await send_q.put(ds)
        return res

    async def update_cache(self) -> Dict[DataBucket[Any, Any], DicomOpReport]:
        """Close associations that haven't been used in a while

        Returns reports for all closed associations.
        """
        curr_time = datetime.now()
        reports = {}
        for dest, cache_entry in self._cache.items():
            age = curr_time - cache_entry.last_use
            if age.total_seconds() > self._timeout:
                await cache_entry.ctx_mgr.__aexit__(None, None, None)
                reports[dest] = cache_entry.op_report
        for dest in reports:
            del self._cache[dest]
        return reports

    async def empty_cache(self) -> Dict[DataBucket[Any, Any], DicomOpReport]:
        """Close all associations

        Returns dict of dest/op_report for all closed associations.
        """
        reports = {}
        for dest, cache_entry in self._cache.items():
            await cache_entry.ctx_mgr.__aexit__(None, None, None)
            reports[dest] = cache_entry.op_report
        self._cache.clear()
        return reports


class InsufficientElemsError(Exception):
    """We don't have the required DICOM elements for the operation"""


class Router:
    """Work with multiple dynamic/static routes"""

    def __init__(self, routes: Iterable[Route], assoc_cache_time: int = 20):
        self._routes = tuple(routes)
        self._assoc_cache_time = assoc_cache_time
        self._static: List[StaticRoute] = []
        self._dynamic: List[DynamicRoute] = []
        self._route_level = QueryLevel.PATIENT
        req_elems: LazySet[str] = LazySet()

        self._all_proxy = True
        for route in routes:
            if isinstance(route, DynamicRoute):
                self._dynamic.append(route)
                req_elems |= route.required_elems
                self._route_level = max(self._route_level, route.route_level)
                if route.dest_methods is not None:
                    for methods in route.dest_methods.values():
                        if TransferMethod.PROXY not in methods:
                            self._all_proxy = False
            elif isinstance(route, StaticRoute):
                self._static.append(route)
                if TransferMethod.PROXY not in route.methods:
                    self._all_proxy = False
            else:
                raise ValueError("Unrecognized route type")
        self._required_elems = FrozenLazySet(req_elems)
        if len(self._dynamic) == 0:
            self._route_level = QueryLevel.STUDY
        elif not self.can_pre_route and not self.can_dyn_route:
            raise NoValidTransferMethodError()

    @property
    def required_elems(self) -> FrozenLazySet[str]:
        """All required DICOM elements for making routing decisions"""
        return self._required_elems

    @property
    def has_dynamic_routes(self) -> bool:
        return len(self._dynamic) != 0

    @property
    def can_pre_route(self) -> bool:
        return self._route_level != QueryLevel.IMAGE

    @property
    def can_dyn_route(self) -> bool:
        return self._all_proxy

    def get_filter_dest_map(
        self, ds: Dataset
    ) -> Dict[Optional[Filter], Tuple[DataBucket[Any, Any], ...]]:
        """Get dict mapping filters to lists of destinations"""
        selected: Dict[Optional[Filter], List[DataBucket[Any, Any]]] = {}
        for route in self._routes:
            dests = route.get_dests(ds)
            if not dests:
                continue
            filt = route.filt
            if filt not in selected:
                selected[filt] = list(dests)
            else:
                selected[filt] += dests
        return {k: tuple(v) for k, v in selected.items()}

    async def pre_route(
        self,
        src: DataRepo[Any, Any, Any, Any],
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
    ) -> Dict[Tuple[StaticRoute, ...], QueryResult]:
        """Pre-calculate any dynamic routing for data on `src`

        If DICOM elements needed for routing decisions can't be queried for, we
        will retrieve an example data set for that study.

        Parameters
        ----------
        src
            The data source

        query
            A query that defines the data to route

        query_res
            A QueryResult that defines the data to route

        Returns
        -------
        result : dict
            Maps tuples of StaticRoute objects to QueryResults defining all of
            the data that should be sent to those routes.
        """
        route_level = self._route_level
        if route_level == QueryLevel.IMAGE:
            raise ValueError("Can't pre-route at IMAGE level")

        # Try to get required DICOM elements by doing a query if needed
        query, query_res = await self._fill_qr(src, query, query_res)

        # Nothing to do...
        if len(self._dynamic) == 0:
            return {tuple(self._static): query_res}
        log.info("Trying to resolve dynamic routes with queries")

        # Iteratively try to extract example data sets with all the elements
        # needed for routing from our QueryResult, while also performing higher
        # level-of-detail queries as needed. In the end the missing_qr will
        # specify a single image for each chunk of data we don't have an
        # example data set for
        example_data: Dict[str, Dataset] = {}
        missing_qr = query_res
        while True:
            new_missing_qr = QueryResult(level=missing_qr.level)
            for pth, sub_uids in missing_qr.walk():
                if pth.level < route_level:
                    continue
                if pth.level != missing_qr.level:
                    # We only want to visit one sub-element
                    # TODO: Allow user defined sorting here?
                    del sub_uids[1:]
                    continue
                lvl_uid = pth.uids[-1]
                ds = deepcopy(missing_qr[lvl_uid])
                for k in self.required_elems:
                    if k not in ds:
                        new_missing_qr.add(ds)
                        break
                else:
                    route_uid = pth.uids[route_level]
                    assert route_uid not in example_data
                    example_data[route_uid] = ds
            missing_qr = new_missing_qr
            if len(missing_qr) == 0 or missing_qr.level == QueryLevel.IMAGE:
                break
            missing_qr = await src.query(
                QueryLevel(missing_qr.level + 1), query, missing_qr
            )

        # For any studies where we don't have example data, fetch some
        if len(missing_qr) != 0:
            log.info("Fetching example data to resolve dynamic routes")
            async for ds in src.retrieve(missing_qr):
                route_uid = get_uid(route_level, ds)
                assert route_uid not in example_data
                example_data[route_uid] = ds
        assert len(example_data) == query_res.get_count(route_level)

        # Resolve all dynamic routes into data specific static routes
        res: Dict[Tuple[StaticRoute, ...], QueryResult] = {}
        for route_uid, ds in example_data.items():
            sub_routes = copy(self._static)
            for route in self._dynamic:
                static_routes = route.get_static_routes(ds)
                if static_routes:
                    sub_routes.extend(static_routes)
            if sub_routes:
                sub_routes_tup = tuple(sub_routes)
                if sub_routes_tup not in res:
                    res[sub_routes_tup] = QueryResult(query_res.level)
                sub_qr = query_res.sub_query(DataNode(route_level, route_uid))
                res[sub_routes_tup] |= sub_qr
            else:
                log.info("Skipping chunk at routing stage: %s", route_uid)
                # TODO: Track this in report
        log.info("All dynamic routes have been resolved")
        return res

    @asynccontextmanager
    async def route(
        self,
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
        report: Optional[DynamicTransferReport] = None,
    ) -> AsyncIterator["asyncio.Queue[Dataset]"]:
        """Produces queue where datasets can be put for dynamic routing

        Parameters
        ----------
        keep_errors
            Set to true to send all data, even if it is inconsistent/duplicate

        report
            Pass a DynamicTransferReport in to be filled out on the fly

            Provides insight into what data is being routed where
        """
        if not self.can_dyn_route:
            raise NoValidTransferMethodError()
        data_q: "asyncio.Queue[Optional[Dataset]]" = asyncio.Queue()
        route_task = asyncio.create_task(self._route(data_q, keep_errors, report))
        try:
            yield data_q  # type: ignore
        finally:
            if not route_task.done():
                await data_q.put(None)
            await route_task

    async def _route(
        self,
        data_q: "asyncio.Queue[Optional[Dataset]]",
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]],
        report: Optional[DynamicTransferReport],
    ) -> None:
        if report is None:
            extern_report = False
            report = DynamicTransferReport()
        else:
            extern_report = True
        report.keep_errors = keep_errors  # type: ignore
        assoc_cache = SendAssociationCache(self._assoc_cache_time)
        try:
            n_pushed = 0
            while True:
                try:
                    ds = await asyncio.wait_for(
                        data_q.get(), min(assoc_cache.next_timeout, 5.0)
                    )
                except asyncio.TimeoutError:
                    await assoc_cache.update_cache()
                    continue
                # TODO: Do we want this? Or should we just use task canceling?
                #       What happens if a user pushes None accidentally? Just
                #       use a different sentinel value?
                if ds is None:
                    break
                filter_dest_map = self.get_filter_dest_map(ds)
                n_filt = len([f for f in filter_dest_map if f is not None])
                # Only make copy of the data set if needed
                if n_filt > 1:
                    orig_ds = deepcopy(ds)
                else:
                    orig_ds = ds
                min_old_ds = minimal_copy(ds)
                for filt, dests in filter_dest_map.items():
                    static_route = StaticRoute(dests, filt=filt)
                    # Update report
                    if filt is not None:
                        filt_ds = filt(orig_ds)
                        if filt_ds is not None:
                            min_new_ds = minimal_copy(filt_ds)
                    else:
                        filt_ds = orig_ds
                        min_new_ds = min_old_ds
                    if filt_ds is None:
                        continue
                    if not report.add(static_route, min_old_ds, min_new_ds):
                        continue
                    # Initiate the transfers
                    coros = [assoc_cache.send(filt_ds, dest) for dest in dests]
                    log.debug("Router forwarding data set to %d dests", len(dests))
                    # TODO: Want to return exceptions here?
                    op_reports = await asyncio.gather(*coros)
                    for op_report, dest in zip(op_reports, dests):
                        if op_report is not None:
                            report.add_store_report(dest, op_report)

                n_pushed += 1
                # Periodically check to avoid association timeouts under high
                # traffic
                if n_pushed % 100 == 0:
                    await assoc_cache.update_cache()

        finally:
            log.debug("Emptying the association cache")
            await assoc_cache.empty_cache()
            report.done = True
            log.debug("Done with routing")
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def _fill_qr(
        self,
        src: DataRepo[Any, Any, Any, Any],
        query: Optional[Dataset],
        query_res: Optional[QueryResult],
    ) -> Tuple[Dataset, QueryResult]:
        """Perform a query against the src if needed"""
        if query is None:
            query = Dataset()
        req_elems = self.required_elems
        if query_res is None:
            level = self._route_level
        else:
            level = query_res.level
            if level < self._route_level:
                level = self._route_level
            elif not req_elems:
                # Nothing we need to query for
                return (query, query_res)
            elif req_elems.is_enumerable():
                if query_res.prov.queried_elems is not None and all(
                    e in query_res.prov.queried_elems for e in req_elems
                ):
                    # All required elems were already queried for
                    return (query, query_res)
                # Check if all required elems already exist
                # TODO: Iterating every data set seems wasteful...
                needs_query = False
                for ds in query_res:
                    for elem in req_elems:
                        if elem not in ds:
                            log.debug("Router needs to query due to missing elements")
                            needs_query = True
                            break
                if not needs_query:
                    return (query, query_res)
        if req_elems.is_enumerable():
            for e in req_elems:
                setattr(query, e, "")
        log.info("The Router is perfoming an intial query against the source: %s", src)
        return (query, await src.query(level, query, query_res))
