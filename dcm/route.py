'''Define static/dynamic routes for DICOM data'''
from __future__ import annotations
import asyncio, logging
from copy import copy, deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from typing import (Optional, Tuple, Callable, Dict, List, Union, Iterable,
                    AsyncIterator, AsyncContextManager, Any, cast)
from contextlib import asynccontextmanager

import janus
from pydicom import Dataset

from .lazyset import LazySet, FrozenLazySet
from .query import (QueryLevel, QueryResult, DataNode, InconsistentDataError,
                    get_uid, minimal_copy)
from .filt import Filter, DataTransform, get_transform
from .util import DuplicateDataError, IndividualReport, MultiListReport, MultiDictReport, MultiKeyedError
from .net import DicomOpReport, IncomingDataError, IncomingErrorType
from .store import DataBucket, DataRepo, TransferMethod, LocalWriteReport


log = logging.getLogger(__name__)


class NoValidTransferMethodError(Exception):
    '''Error raised when we are unable to select a valid transfer method'''
    def __init__(self, src_dest_pair: Optional[Tuple[DataBucket[Any, Any], DataBucket[Any, Any]]]=None):
        self.src_dest_pair = src_dest_pair

    def __str__(self) -> str:
        if self.src_dest_pair is None:
            return "No valid transfer method for one or more routes"
        else:
            return f"No valid transfer method between {self.src_dest_pair[0]} and {self.src_dest_pair[1]}"


# TODO: Have been working under the assumption the filter would be applied
#       before resolving dynamic routes, but it is more likely and common
#       that we would want to route on the original data, since we may have
#       a rather broad filter (i.e. anonymization) that screws up the elements
#       used for routing.
#
#       Any logic that would go into a pre-filter could just be placed in the
#       dynamic routing function. We might just need to duplicate that logic
#       into a filter if we also want to persist the changes which is an okay
#       trade-off compared to the complexity of allowing both pre/post filters
#
#       We do lose the ability to specify which elements might be
#       modified, how they might be modified, and what their dependencies are.
#       Do we implicitly disallow uninvertible shenanigans in the dynamic routing
#       function?
@dataclass(frozen=True)
class Route:
    '''Abstract base class for all Routes

    The main functionality of routes is to map datasets to destinations.

    Routes can have a filter associated with them, which take a dataset as
    input and return one as output. The dataset can be modified and None can be
    returned to reject the dataset.
    '''
    filt: Optional[Filter] = None
    '''Steaming data filter for editing and rejecting data sets'''
    ''''''

    def get_dests(self, data_set: Dataset) -> Optional[Tuple[DataBucket[Any, Any], ...]]:
        '''Return the destintations for the `data set`

        Must be implemented by all subclasses.'''
        raise NotImplementedError

    def get_filtered(self, data_set: Dataset) -> Optional[Dataset]:
        if not self.filt:
            return data_set
        return self.filt(data_set)


@dataclass(frozen=True)
class _StaticBase:
    dests: Tuple[DataBucket[Any, Any], ...]
    '''Static tuple of destinations'''

    methods: Tuple[TransferMethod, ...] = (TransferMethod.PROXY,)
    '''The transfer methods to use, in order of preference

    This will automatically be paired down to the methods supported by all the
    dests (or just allow PROXY if we have a filter). If no valid transfer
    methods are given a `NoValidTransferMethodError` will be raised.
    '''


@dataclass(frozen=True)
class StaticRoute(Route, _StaticBase):
    '''Static route that sends all (unfiltered) data to same dests'''
    def __post_init__(self) -> None:
        if self.filt is not None:
            if TransferMethod.PROXY not in self.methods:
                raise NoValidTransferMethodError()
            avail_methods = [TransferMethod.PROXY]
        else:
            avail_methods = []
            for meth in self.methods:
                if all(meth in d._supported_methods for d in self.dests):
                    avail_methods.append(meth)
            if len(avail_methods) == 0:
                raise NoValidTransferMethodError()
        object.__setattr__(self, 'dests', tuple(self.dests))
        object.__setattr__(self, 'methods', tuple(avail_methods))

    def get_dests(self, data_set: Dataset) -> Tuple[DataBucket[Any, Any], ...]:
        return self.dests

    def get_method(self, src: DataBucket[Any, Any]) -> TransferMethod:
        for method in self.methods:
            if method in src._supported_methods:
                return method
        raise NoValidTransferMethodError()

    def __str__(self) -> str:
        return 'Static: %s' % ','.join(str(d) for d in self.dests)


@dataclass(frozen=True)
class _DynamicBase:
    lookup: Callable[[Dataset], Optional[Tuple[DataBucket[Any, Any], ...]]]
    '''Callable takes a dataset and returns destinations'''

    route_level: QueryLevel = QueryLevel.STUDY
    '''The level in the DICOM hierarchy we are making routing decisions at'''

    required_elems: FrozenLazySet[str] = field(default_factory=FrozenLazySet)
    '''DICOM elements that we require to make a routing decision'''

    dest_methods: Optional[Dict[Optional[DataBucket[Any, Any]], Tuple[TransferMethod, ...]]] = None
    '''Specify transfer methods for (some) dests

    Use `None` as the key to specify the default transfer methods for all dests
    not explicitly listed.

    Only respected when pre-routing is used. Dynamic routing can only proxy.
    '''


@dataclass(frozen=True)
class DynamicRoute(Route, _DynamicBase):
    '''Dynamic route which determines destinations based on the data.

    Routing decisions are made before applying the filter to the data.
    '''
    def __post_init__(self) -> None:
        if self.dest_methods is not None:
            avail_meths: Dict[Optional[DataBucket[Any, Any]], Tuple[TransferMethod, ...]] = {}
            for dest, methods in self.dest_methods.items():
                if self.filt is not None:
                    if TransferMethod.PROXY not in methods:
                        raise NoValidTransferMethodError()
                    avail_meths[dest] = (TransferMethod.PROXY,)
                elif dest is None:
                    avail_meths[dest] = methods
                else:
                    meths = tuple(m for m in methods if m in dest._supported_methods)
                    if len(meths) == 0:
                        raise NoValidTransferMethodError()
                    avail_meths[dest] = meths
            object.__setattr__(self,
                               'dest_methods',
                               avail_meths)
        if self.route_level not in QueryLevel:
            raise ValueError("Invalid route_level: %s" % self.route_level)
        if not isinstance(self.required_elems, FrozenLazySet):
            object.__setattr__(self,
                               'required_elems',
                               FrozenLazySet(self.required_elems))

    def get_dests(self, data_set: Dataset) -> Optional[Tuple[DataBucket[Any, Any], ...]]:
        dests = self.lookup(data_set)
        if dests is None:
            return None
        return tuple(dests)

    def get_static_routes(self, data_set: Dataset) -> Optional[Tuple[StaticRoute, ...]]:
        '''Resolve this dynamic route into one or more static routes'''
        dests = self.lookup(data_set)
        if dests is None:
            return dests
        dests = tuple(dests)

        if self.dest_methods is not None:
            meths_dests_map: Dict[Tuple[TransferMethod, ...], List[DataBucket[Any, Any]]] = {}
            default_methods = self.dest_methods.get(None)
            if default_methods is None:
                default_methods = (TransferMethod.PROXY,)
            for dest in dests:
                d_methods = self.dest_methods.get(dest)
                if d_methods is None:
                    d_methods = default_methods
                if d_methods not in meths_dests_map:
                    meths_dests_map[d_methods] = []
                meths_dests_map[d_methods].append(dest)
            return tuple(StaticRoute(tuple(sub_dests),
                                     filt=deepcopy(self.filt),
                                     methods=meths)
                         for meths, sub_dests in meths_dests_map.items())
        else:
            return (StaticRoute(dests, filt=deepcopy(self.filt)),)

    def __str__(self) -> str:
        return 'Dynamic on: %s' % self.required_elems


class ProxyTransferError(Exception):
    def __init__(self,
                 store_errors: Optional[MultiKeyedError] = None,
                 inconsistent: Optional[Dict[StaticRoute, List[Tuple[Dataset, Dataset]]]] = None,
                 duplicate: Optional[Dict[StaticRoute, List[Tuple[Dataset, Dataset]]]] = None,
                 incoming_error: Optional[IncomingDataError] = None):
        self.store_errors = store_errors
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.incoming_error = incoming_error

    def __str__(self) -> str:
        res = ['ProxyTransferError:']
        if self.inconsistent is not None:
            res.append("%d inconsistent data sets" % len(self.inconsistent))
        if self.duplicate is not None:
            res.append("%d duplicate data sets" % len(self.duplicate))
        if self.store_errors is not None:
            for err in self.store_errors.errors:
                res.append(str(err))
        if self.incoming_error is not None:
            res.append(str(self.incoming_error))
        return '\n\t'.join(res)


# TODO: Some annoying overlap with IncomingDataReport here, but not clear we
#       can do much about it since we need a RetrieveReport when the src is
#       remote, and we need the `sent` dict here to track data transforms.
#
#       Can we make sure the same (minimized) data set is used in all report
#       structures? Does that alleviate all concerns about duplication?

# TODO: Update keep_errors handling here. I guess the `add` method should
#       return a bool like with the IncomingDataReports? Also, this means that
#       we might end up sending erroneous data, which can't be caputred in the
#       DataTransforms under `sent` here. I guess this is okay and mimics what
#       happens in a RetrieveReport
#
@dataclass
class ProxyReport(IndividualReport):
    '''Abstract base class for reports on proxy transfers'''

    sent: Dict[StaticRoute, DataTransform] = field(default_factory=dict)
    '''Tracks what data was sent where'''

    inconsistent: Dict[StaticRoute, List[Tuple[Dataset, Dataset]]] = field(default_factory=dict)
    '''Tracks inconsistent data'''

    duplicate: Dict[StaticRoute, List[Tuple[Dataset, Dataset]]] = field(default_factory=dict)
    '''Tracks duplicate data'''

    _keep_errors: Tuple[IncomingErrorType, ...] = field(default=tuple(), init=False)

    @property
    def keep_errors(self) -> Tuple[IncomingErrorType, ...]:
        '''Whether or not we are forwarding inconsistent/duplicate data'''
        return self._keep_errors

    @keep_errors.setter
    def keep_errors(self, val: Union[bool, Tuple[IncomingErrorType, ...]]) -> None:
        if val == True:
            self._keep_errors = tuple(IncomingErrorType)
        elif val == False:
            self._keep_errors = tuple()
        else:
            val = cast(Tuple[IncomingErrorType, ...], val)
            self._keep_errors = val

    @property
    def n_errors(self) -> int:
        n_errors = 0
        if not self.keep_errors:
            n_errors += self.n_inconsistent + self.n_duplicate
        return n_errors

    @property
    def n_warnings(self) -> int:
        n_warn = 0
        if self.keep_errors:
            n_warn += self.n_inconsistent + self.n_duplicate
        return n_warn

    @property
    def all_success(self) -> bool:
        return self.n_errors + self.n_warnings == 0

    @property
    def n_input(self) -> int:
        '''Number of input data sets'''
        # TODO: Using trans.new here won't count data sets that were filtered out
        res = sum(len(trans.new) for sr, trans in self.sent.items())
        res += self.n_inconsistent + self.n_duplicate
        return res

    @property
    def n_sent(self) -> int:
        '''Number of times datasets were sent out'''
        res = sum(len(trans.new) * len(sr.dests)
                  for sr, trans in self.sent.items())
        if self.keep_errors:
            res += sum(len(x) * len(sr.dests)
                       for sr, x in self.inconsistent.items())
            res += sum(len(x) * len(sr.dests)
                       for sr, x in self.duplicate.items())
        return res

    @property
    def n_inconsistent(self) -> int:
        return sum(len(x) for _, x in self.inconsistent.items())

    @property
    def n_duplicate(self) -> int:
        return sum(len(x) for _, x in self.duplicate.items())

    @property
    def n_reported(self) -> int:
        '''Number store results that have been reported so far'''
        raise NotImplementedError

    @property
    def all_reported(self) -> bool:
        '''True if all sent data sets have a reported result
        '''
        assert self.n_reported <= self.n_sent
        return self.n_sent == self.n_reported

    def add(self, route: StaticRoute, old_ds: Dataset, new_ds: Dataset) -> bool:
        '''Add the route with pre/post filtering dataset to the report'''
        if route not in self.sent:
            self.sent[route] = get_transform(QueryResult(QueryLevel.IMAGE),
                                             route.filt)
        try:
            self.sent[route].add(old_ds, new_ds)
        except InconsistentDataError:
            if route not in self.inconsistent:
                self.inconsistent[route] = []
            self.inconsistent[route].append((old_ds, new_ds))
            return IncomingErrorType.INCONSISTENT in self._keep_errors
        except DuplicateDataError:
            if route not in self.duplicate:
                self.duplicate[route] = []
            self.duplicate[route].append((old_ds, new_ds))
            return IncomingErrorType.DUPLICATE in self._keep_errors
        return True

    def log_issues(self) -> None:
        '''Produce log messages for any warning/error statuses'''
        n_inconsist = self.n_inconsistent
        if n_inconsist:
            if self.keep_errors:
                log.warning("Sent %d inconsistent data sets" % n_inconsist)
            else:
                log.error("Skipped %d inconsistent data sets" % n_inconsist)
        n_duplicate = self.n_duplicate
        if n_duplicate:
            if self.keep_errors:
                log.warning("Sent %d duplicate data sets" % n_duplicate)
            else:
                log.error("Skipped %d duplicate data sets" % n_duplicate)

    def check_errors(self) -> None:
        '''Raise an exception if any errors have occured so far'''
        if self.n_errors:
            inconsist = None
            if self.inconsistent:
                inconsist = self.inconsistent
            dupes = None
            if self.duplicate:
                dupes = self.duplicate
            raise ProxyTransferError(inconsistent=inconsist, duplicate=dupes)

    def clear(self) -> None:
        self.sent.clear()
        self.inconsistent.clear()
        self.duplicate.clear()


StoreReportType = Union[DicomOpReport, LocalWriteReport]


@dataclass
class DynamicTransferReport(ProxyReport):
    '''Track what data is being routed where and any store results'''
    store_reports: MultiDictReport[DataBucket[Any, Any], MultiListReport[StoreReportType]] = field(default_factory=MultiDictReport)

    @property
    def n_errors(self) -> int:
        return super().n_errors + self.store_reports.n_errors

    @property
    def n_warnings(self) -> int:
        return super().n_warnings + self.store_reports.n_warnings

    @property
    def n_reported(self) -> int:
        return self.store_reports.n_input

    def add_store_report(self, dest: DataBucket[Any, Any], store_report: StoreReportType) -> None:
        '''Add a DicomOpReport to keep track of'''
        if dest not in self.store_reports:
            self.store_reports[dest] = MultiListReport()
        self.store_reports[dest].append(store_report)

    def log_issues(self) -> None:
        '''Produce log messages for any warning/error statuses'''
        super().log_issues()
        self.store_reports.log_issues()

    def check_errors(self) -> None:
        '''Raise an exception if any errors have occured so far'''
        if self.n_errors:
            err = None
            try:
                super().check_errors()
            except ProxyTransferError as e:
                err = e
            else:
                err = ProxyTransferError()
            try:
                self.store_reports.check_errors()
            except MultiKeyedError as e:
                err.store_errors = e
            raise err

    def clear(self) -> None:
        '''Clear current info about data sets we have results for'''
        # TODO: If n_sent != n_reported here we will go out of sync. I guess
        #       this would need to be managed at a higher level if it is
        #       needed. Not clear if it makes sense to do anything about it
        #       here.
        super().clear()
        self.store_reports.clear()


@dataclass
class _CacheEntry:
    '''Entry in a SendAssociationCache'''
    ctx_mgr: AsyncContextManager['janus._AsyncQueueProxy[Dataset]']
    send_q: 'janus._AsyncQueueProxy[Dataset]'
    op_report: DicomOpReport
    last_use: datetime


# TODO: Make generic association caching in `net` module supporting
#       query/move/send. Could then use that everywhere, and use it to
#       manage max association limits on any node.
class SendAssociationCache:
    def __init__(self, timeout: float = 30.):
        '''Keeps cache of recent associations'''
        self._timeout = timeout
        self._cache: Dict[DataBucket[Any, Any], _CacheEntry] = {}

    @property
    def next_timeout(self) -> float:
        '''Number of seconds until the next cache entry will timeout'''
        next_timeout = self._timeout
        now = datetime.now()
        for cache_entry in self._cache.values():
            td = (now - cache_entry.last_use)
            timeout = max(self._timeout - td.total_seconds(), 0)
            if timeout < next_timeout:
                next_timeout = timeout
        return next_timeout

    async def send(self, ds: Dataset, dest: DataBucket[Any, Any]) -> Optional[DicomOpReport]:
        '''Send a data set to dests, utilizing the cache of active associations'''
        res = None
        cache_entry = self._cache.get(dest, None)
        if cache_entry is None:
            op_report = DicomOpReport()
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
        '''Close associations that haven't been used in a while

        Returns reports for all closed associations.
        '''
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
        '''Close all associations

        Returns dict of dest/op_report for all closed associations.
        '''
        reports = {}
        for dest, cache_entry in self._cache.items():
            await cache_entry.ctx_mgr.__aexit__(None, None, None)
            reports[dest] = cache_entry.op_report
        self._cache.clear()
        return reports


class InsufficientElemsError(Exception):
    '''We don't have the required DICOM elements for the operation'''


class Router:
    '''Work with multiple dynamic/static routes'''
    def __init__(self, routes : Iterable[Route], assoc_cache_time: int = 20):
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
        '''All required DICOM elements for making routing decisions'''
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

    def get_filter_dest_map(self, ds: Dataset) -> Dict[Optional[Filter], Tuple[DataBucket[Any, Any], ...]]:
        '''Get dict mapping filters to lists of destinations'''
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
        return {k : tuple(v) for k, v in selected.items()}

    async def pre_route(self,
                        src: DataRepo[Any, Any, Any],
                        query: Union[Dict[str, Any], Dataset] = None,
                        query_res: QueryResult = None
                       ) -> Dict[Tuple[StaticRoute,...], QueryResult]:
        '''Pre-calculate any dynamic routing for data on `src`

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
        '''
        route_level = self._route_level
        if route_level == QueryLevel.IMAGE:
            raise ValueError("Can't pre-route at IMAGE level")

        # Try to get required DICOM elements by doing a query if needed
        query, query_res = await self._fill_qr(src, query, query_res)

        # Nothing to do...
        if len(self._dynamic) == 0:
            return {tuple(self._static) : query_res}

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
                if pth.level >= route_level:
                    if pth.level != query_res.level:
                        # We only want to visit one sub-element
                        # TODO: Try to choose one with least instances?
                        del sub_uids[1:]
                        continue
                    lvl_uid = pth.uids[-1]
                    ds = deepcopy(missing_qr[lvl_uid])
                    for k in self.required_elems:
                        # TODO: Some PACS might return a blank field instead of missing the attr?
                        if k not in ds:
                            new_missing_qr.add(ds)
                            break
                    else:
                        route_uid = pth.uids[route_level]
                        assert route_uid not in example_data
                        example_data[route_uid] = ds
            if len(new_missing_qr) == 0:
                missing_qr = new_missing_qr
                break
            if missing_qr.level == QueryLevel.IMAGE:
                break
            missing_qr = await src.query(QueryLevel(missing_qr.level + 1),
                                         query,
                                         new_missing_qr)

        # For any studies where we don't have example data, fetch some
        if len(missing_qr) != 0:
            log.debug("Fetching example data to make routing decisions")
            async for ds in src.retrieve(missing_qr):
                route_uid = get_uid(route_level, ds)
                assert route_uid not in example_data
                example_data[route_uid] = ds
        assert len(example_data) == query_res.get_count(route_level)

        # Resolve all dynamic routes into data specific static routes
        res: Dict[Tuple[StaticRoute,...], QueryResult] = {}
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
        return res

    @asynccontextmanager
    async def route(self,
                    keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
                    report: Optional[DynamicTransferReport] = None) \
                        -> AsyncIterator['asyncio.Queue[Dataset]']:
        '''Produces queue where datasets can be put for dynamic routing

        Parameters
        ----------
        keep_errors
            Set to true to send all data, even if it is inconsistent/duplicate

        report
            Pass a DynamicTransferReport in to be filled out on the fly

            Provides insight into what data is being routed where
        '''
        if not self.can_dyn_route:
            raise NoValidTransferMethodError()
        data_q: 'asyncio.Queue[Dataset]' = asyncio.Queue()
        route_task = asyncio.create_task(self._route(data_q,
                                                     keep_errors,
                                                     report)
                                        )
        try:
            yield data_q
        finally:
            if not route_task.done():
                await data_q.put(None)
            await route_task

    async def _route(self,
                     data_q: 'asyncio.Queue[Dataset]',
                     keep_errors: Union[bool, Tuple[IncomingErrorType, ...]],
                     report: Optional[DynamicTransferReport]) -> None:
        if report is None:
            extern_report = False
            report = DynamicTransferReport()
        else:
            extern_report = True
        report.keep_errors = keep_errors # type: ignore
        assoc_cache = SendAssociationCache(self._assoc_cache_time)
        try:
            n_pushed = 0
            while True:
                try:
                    ds = await asyncio.wait_for(data_q.get(),
                                                assoc_cache.next_timeout)
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
                    log.debug("Router forwarding data set to %d dests" %
                              len(dests))
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
            await assoc_cache.empty_cache()
            report.done = True
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def _fill_qr(self,
                       src: DataRepo[Any, Any, Any],
                       query: Optional[Dataset],
                       query_res: Optional[QueryResult]
                      ) -> Tuple[Dataset, QueryResult]:
        '''Perform a query against the src if needed'''
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
                if (query_res.prov.queried_elems is not None and
                    all(e in query_res.prov.queried_elems for e in req_elems)
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
                setattr(query, e, '')
        log.debug("The Router._fill_qr method is performing a query")
        return (query, await src.query(level, query, query_res))
