'''Define static/dynamic routes for DICOM data'''
import asyncio, logging
from copy import copy, deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Dict, List
from contextlib import asynccontextmanager

from .query import (QueryLevel, QueryResult, DataNode, InconsistentDataError,
                    get_uid, minimal_copy)
from .net import DcmNode, DicomOpReport
from .filt import (LazySet, FrozenLazySet, Filter, DataTransform, DummyTransform,
                   FilterTransform, get_transform)
from .util import DuplicateDataError


log = logging.getLogger(__name__)


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

    def get_dests(self, data_set):
        '''Return the destintations for the `data set`

        Must be implemented by all subclasses.'''
        raise NotImplementedError

    def get_filtered(self, data_set):
        if not self.filt:
            return data_set
        return self.filt(data_set)


@dataclass(frozen=True)
class _StaticBase:
    dests: Tuple


@dataclass(frozen=True)
class StaticRoute(Route, _StaticBase):
    '''Static route that sends all data to same dests'''
    def __post_init__(self):
        object.__setattr__(self, 'dests', tuple(self.dests))

    def get_dests(self, data_set):
        return self.dests

    def __str__(self):
        return 'Static: %s' % ','.join(str(d) for d in self.dests)


@dataclass(frozen=True)
class _DynamicBase:
    lookup: Callable
    '''Function that takes a data set and returns a destination'''

    route_level: QueryLevel = QueryLevel.STUDY
    '''The level in the DICOM hierarchy we are making routing decisions at'''

    required_elems: FrozenLazySet = field(default_factory=FrozenLazySet)
    '''DICOM elements that we require to make a routing decision'''


@dataclass(frozen=True)
class DynamicRoute(Route, _DynamicBase):
    '''Dynamic route which determines destinations based on the data.

    Routing decisions are made before applying the filter to the data.
    '''
    def __post_init__(self):
        if self.route_level not in QueryLevel:
            raise ValueError("Invalid route_level: %s" % self.route_level)
        if not isinstance(self.required_elems, FrozenLazySet):
            object.__setattr__(self,
                               'required_elems',
                               FrozenLazySet(self.required_elems))

    def get_dests(self, data_set):
        return tuple(self.lookup(data_set))

    def __str__(self):
        return 'Dynamic on: %s' % self.required_elems


# TODO: Make generic association caching in `net` module supporting
#       query/move/send. Could then use that everywhere, and use it to
#       manage max association limits on any node.
class SendAssociationCache:
    def __init__(self, timeout=30):
        '''Keeps cache of recent associations'''
        self._timeout = timeout
        self._cache = {}
        self._op_reports = {}

    @property
    def next_timeout(self):
        '''Number of seconds until the next cache entry will timeout'''
        next_timeout = self._timeout
        now = datetime.now()
        for _, _, _, last_use in self._cache.values():
            timeout = max(self._timeout - (now - last_use).total_seconds(), 0)
            if timeout < next_timeout:
                next_timeout = timeout
        return next_timeout

    async def send(self, ds, dest):
        '''Send a data set to dests, utilizing the cache of active associations'''
        res = None
        op_report, ctx_mgr, send_q, last_use = \
            self._cache.get(dest, (None, None, None, None))
        if send_q is None:
            op_report = DicomOpReport()
            res = op_report
            ctx_mgr = dest.send(op_report)
            send_q = await ctx_mgr.__aenter__()
        last_use = datetime.now()
        self._cache[dest] = (op_report, ctx_mgr, send_q, last_use)
        await send_q.put(ds)
        return res

    async def update_cache(self):
        '''Close associations that haven't been used in a while

        Returns dict of dest/op_report for all closed associations.
        '''
        curr_time = datetime.now()
        reports = {}
        for dest, (op_report, ctx_mgr, send_q, last_use) in self._cache.items():
            age = curr_time - last_use
            if age.total_seconds() > self._timeout:
                await ctx_mgr.__aexit__(None, None, None)
                reports[dest] = op_report
        for dest in reports:
            del self._cache[dest]
        return reports

    async def empty_cache(self):
        '''Close all associations

        Returns dict of dest/op_report for all closed associations.
        '''
        reports = {}
        for dest, (op_report, ctx_mgr, _, _) in self._cache.items():
            await ctx_mgr.__aexit__(None, None, None)
            reports[dest] = op_report
        self._cache.clear()
        return reports


class TransferError(Exception):
    def __init__(self, store_errors=None, inconsistent=None, duplicate=None,
                 retrieve_error=None):
        self.store_errors = store_errors
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.retrieve_error = retrieve_error

    def __str__(self):
        res = ['TransferError:']
        if self.inconsistent is not None:
            res.append("%d inconsistent data sets" % len(self.inconsistent))
        if self.duplicate is not None:
            res.append("%d duplicate data sets" % len(self.duplicate))
        if self.store_errors is not None:
            for err in self.store_errors:
                res.append(str(err))
        if self.retrieve_error is not None:
            res.append(str(self.retrieve_error))
        return '\n\t'.join(res)


# TODO: Should we add an IncomingDataReport here? I guess it applies to all
#       possible transfers. However, we already track the incoming data in the
#       DataTransform objects in `sent`.
#
#       Would provide slightly more info in weird cases (data started erroneous
#       and filtering changed the data but it was still erroneous)
#
#       This is in contrast to the RetrieveReport which really does bring more
#       info to the table.
#
#       Is it likely this report will grow some other useful info? Performance
#       maybe? Or just tracking data movement through pipeline...
@dataclass
class LocalTransferReport:
    '''Abstract base class for reports on local transfers'''
    sent: Dict[StaticRoute, DataTransform] = field(default_factory=dict)
    inconsistent: Dict[StaticRoute, List] = field(default_factory=dict)
    duplicate: Dict[StaticRoute, List] = field(default_factory=dict)
    keep_errors: bool = False
    done: bool = False

    @property
    def n_errors(self):
        n_errors = 0
        if not self.keep_errors:
            n_errors += self.n_inconsistent + self.n_duplicate
        return n_errors

    @property
    def n_warnings(self):
        n_warn = 0
        if self.keep_errors:
            n_warn += self.n_inconsistent + self.n_duplicate
        return n_warn

    @property
    def all_success(self):
        return self.n_errors + self.n_warnings == 0

    @property
    def n_input(self):
        '''Number of input data sets'''
        # TODO: Using trans.new here won't count data sets that were filtered out
        res = sum(len(trans.new) for sr, trans in self.sent.items())
        res += self.n_inconsistent + self.n_duplicate
        return res

    @property
    def n_sent(self):
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
    def n_inconsistent(self):
        return sum(len(x) for _, x in self.inconsistent.items())

    @property
    def n_duplicate(self):
        return sum(len(x) for _, x in self.duplicate.items())

    @property
    def n_reported(self):
        '''Number store results that have been reported so far'''
        raise NotImplementedError

    @property
    def all_reported(self):
        '''True if all sent data sets have a reported result
        '''
        assert self.n_reported <= self.n_sent
        return self.n_sent == self.n_reported

    def add(self, route, old_ds, new_ds):
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
        except DuplicateDataError:
            if route not in self.duplicate:
                self.duplicate[route] = []
            self.duplicate[route].append((old_ds, new_ds))

    def log_issues(self):
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

    def check_errors(self):
        '''Raise an exception if any errors have occured so far'''
        if self.n_errors:
            inconsist = None
            if self.inconsistent:
                inconsist = self.inconsistent
            dupes = None
            if self.duplicate:
                dupes = self.duplicate
            raise TransferError(inconsistent=inconsist, duplicate=dupes)

    def clear(self):
        self.sent.clear()
        self.inconsistent.clear()
        self.duplicate.clear()


@dataclass
class DynamicTransferReport(LocalTransferReport):
    '''Track what data is being routed where and any store results'''
    store_reports: Dict[DcmNode, List[DicomOpReport]] = field(default_factory=dict)

    @property
    def n_errors(self):
        n_errors = super().n_errors
        for dest, op_reports in self.store_reports.items():
            for op_report in op_reports:
                n_errors += op_report.n_errors
        return n_errors

    @property
    def n_warnings(self):
        n_warn = super().n_warnings
        for dest, op_reports in self.store_reports.items():
            for op_report in op_reports:
                n_warn += op_report.n_warnings
        return n_warn

    @property
    def n_reported(self):
        return sum(len(rep)
                   for _, reps in self.store_reports.items()
                   for rep in reps)

    def add_op_report(self, dest, op_report):
        '''Add a DicomOpReport to keep track of'''
        if dest not in self.store_reports:
            self.store_reports[dest] = []
        self.store_reports[dest].append(op_report)

    def log_issues(self):
        '''Produce log messages for any warning/error statuses'''
        super().log_issues()
        for dest, op_reports in self.store_reports.items():
            for op_report in op_reports:
                op_report.log_issues()

    def check_errors(self):
        '''Raise an exception if any errors have occured so far'''
        if self.n_errors:
            err = None
            try:
                super().check_errors()
            except TransferError as e:
                err = e
            else:
                err = TransferError()
            store_errors = {}
            for dest, op_reports in self.store_reports:
                errors = []
                for op_report in op_reports:
                    errors += op_report.errors
                if errors:
                    store_errors[dest] = errors
            if store_errors:
                err.store_errors = store_errors
            raise err

    def clear(self):
        '''Clear current info about data sets we have results for'''
        # TODO: If n_sent != n_reported here we will go out of sync. I guess
        #       this would need to be managed at a higher level if it is
        #       needed. Not clear if it makes sense to do anything about it
        #       here.
        super().clear()
        trim_list = []
        for dest, op_reports in self.store_reports.items():
            if op_reports[-1].done:
                assert all(r.done for r in op_reports)
                op_reports.clear()
            else:
                assert all(r.done for r in op_reports[:-1])
                trim_list.append(dest)
        for dest in trim_list:
            self.store_reports[dest] = self.store_reports[dest][-1:]


class InsufficientElemsError(Exception):
    '''We don't have the required DICOM elements for the operation'''


class Router:
    '''Work with multiple dynamic/static routes'''
    def __init__(self, routes, assoc_cache_time=20):
        self._routes = routes
        self._assoc_cache_time = assoc_cache_time
        self._static = []
        self._dynamic = []
        self._required_elems = LazySet()
        self._route_level = QueryLevel.PATIENT
        for route in routes:
            if isinstance(route, DynamicRoute):
                self._dynamic.append(route)
                self._required_elems |= route.required_elems
                self._route_level = max(self._route_level, route.route_level)
            else:
                self._static.append(route)
        if len(self._dynamic) == 0:
            self._route_level = QueryLevel.STUDY

    @property
    def required_elems(self):
        '''All required DICOM elements for making routing decisions'''
        return self._required_elems

    @property
    def can_pre_route(self):
        return self._route_level != QueryLevel.IMAGE

    def get_filter_dest_map(self, ds):
        '''Get dict mapping filters to lists of destinations'''
        selected = {}
        for route in self._routes:
            dests = route.get_dests(ds)
            if not dests:
                continue
            filt = route.filt
            if filt not in selected:
                selected[filt] = dests
            else:
                selected[filt] += dests
        return selected

    async def pre_route(self, src, query=None, query_res=None,
                        req_queried=False):
        '''Pre-calculate any dynamic routing for data on `src`

        If DICOM elements needed for routing decisions can't be queried for, we
        will retrieve an example data set for that study.

        Parameters
        ----------
        src : DataRepo
            The data source

        query : Dataset or dict
            A query that defines the data to route

        query_res : QueryResult
            A QueryResult that defines the data to route

        req_queried : bool
            Pass True if the `query_res` tried to get all `required_elems`

            Any attempts to query for missing elements will be skipped

        Returns
        -------
        result : dict
            Maps tuples of StaticRoute objects to QueryResults defining all of
            the data that should be sent to those routes.
        '''
        route_level = self._route_level
        if route_level == QueryLevel.IMAGE:
            raise QueryLevelMismatchError("Can't pre-route at IMAGE level")

        # Try to get required DICOM elements by doing a query if needed
        query, query_res = await self._fill_qr(src, query, query_res, req_queried)

        # Nothing to do...
        if len(self._dynamic) == 0:
            return {tuple(self._routes) : query_res}

        # Iteratively try to extract example data sets with all the elements
        # needed for routing from our QueryResult, while also performing higher
        # level-of-detail queries as needed. In the end the missing_qr will
        # specify a single image for each chunk of data we don't have an
        # example data set for
        example_data = {}
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
            missing_qr = await src.query(missing_qr.level + 1,
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
        res = {}
        for route_uid, ds in example_data.items():
            sub_routes = copy(self._static)
            for route in self._dynamic:
                dests = route.get_dests(ds)
                if dests:
                    static_route = StaticRoute(dests, deepcopy(route.filt))
                    sub_routes.append(static_route)
            if sub_routes:
                sub_routes = tuple(sub_routes)
                if sub_routes not in res:
                    res[sub_routes] = QueryResult(query_res.level)
                sub_qr = query_res.sub_query(DataNode(route_level, route_uid))
                res[sub_routes] |= sub_qr
        return res

    async def _route(self, data_q, keep_errors, report):
        if report is None:
            extern_report = False
            report = DynamicTransferReport()
        else:
            extern_report = True
        report.keep_errors = keep_errors
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
                    static_route = StaticRoute(dests, filt)
                    # Update report
                    if filt is not None:
                        filt_ds = filt(orig_ds)
                        min_new_ds = minimal_copy(filt_ds)
                    else:
                        filt_ds = orig_ds
                        min_new_ds = min_old_ds
                    report.add(static_route, min_old_ds, min_new_ds)
                    # Initiate the transfers
                    coros = [assoc_cache.send(filt_ds, dest) for dest in dests]
                    log.debug("Router forwarding data set to %d dests" %
                              len(dests))
                    op_reports = await asyncio.gather(*coros)
                    for op_report, dest in zip(op_reports, dests):
                        if op_report is not None:
                            report.add_op_report(dest, op_report)

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

    @asynccontextmanager
    async def route(self, keep_errors=False, report=None):
        '''Produces queue where datasets can be put for dynamic routing

        Parameters
        ----------
        keep_errors : bool
            Set to true to send all data, even if it is inconsistent/duplicate

        report : DynamicTransferReport or None
            Pass a DynamicTransferReport in to be filled out on the fly

            Provides insight into what data is being routed where
        '''
        data_q = asyncio.Queue()
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

    async def _fill_qr(self, src, query, query_res, req_queried):
        if query is None:
            query = {}
        needs_query = False
        req_elems = self.required_elems
        if query_res is None:
            needs_query = True
            level = self._route_level
        else:
            level = query_res.level
            if level < self._route_level:
                needs_query = True
                level = self._route_level
            elif req_elems and req_elems.is_enumerable() and not req_queried:
                for ds in query_res:
                    for elem in req_elems:
                        if elem not in ds:
                            log.debug("Router needs to query due to missing elements")
                            needs_query = True
                            break
        if not needs_query:
            return (query, query_res)
        # TODO: The `query` could be a dataset, can't assume dict
        if req_elems.is_enumerable():
            for e in req_elems:
                query[e] = ''
        log.debug("The Router._fill_qr method is performing a query")
        return (query, await src.query(level, query, query_res))
