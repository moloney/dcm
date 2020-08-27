'''Synchronize DICOM data between local and/or remote locations'''
from __future__ import annotations
import logging, itertools, textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import (Optional, Tuple, Dict, List, Union, Iterable, Any, Set, Type,
                    Callable, TypeVar, Iterator, AsyncIterator, cast, Generic)
from typing_extensions import Protocol
from contextlib import AsyncExitStack, asynccontextmanager

from pydicom import Dataset

from .query import (QueryLevel, QueryResult, DataNode, get_all_uids,
                    minimal_copy, uid_elems)
from .store import (TransferMethod, DataChunk, DataBucket, OobCapable, DataRepo,
                    LocalIncomingReport)
from .filt import get_transform, Filter
from .route import (Route, StaticRoute, Router, ProxyTransferError,
                    ProxyReport, StoreReportType, DynamicTransferReport,
                    NoValidTransferMethodError)
from .diff import diff_data_sets, DataDiff
from .net import (IncomingDataReport, IncomingDataError, IncomingErrorType,
                  DicomOpReport, RetrieveReport)
from .report import (BaseReport, MultiAttrReport, MultiListReport, MultiDictReport, 
                     MultiKeyedError, ProgressHookBase)
from .util import dict_to_ds


log = logging.getLogger(__name__)

class StaticStoreReport(MultiDictReport[DataBucket[Any, Any], StoreReportType]):
    '''Transfer report that only captures storage'''


IncomingReportType = Union[IncomingDataReport, RetrieveReport, LocalIncomingReport]


class StaticProxyTransferReport(ProxyReport):
    '''Static proxy transfer report'''

    def __init__(self, 
                 description: Optional[str] = None, 
                 depth: int = 0,
                 n_expected: Optional[int] = None,
                 prog_hook: Optional[ProgressHookBase[Any]] = None,
                 keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
                 incoming_report: Optional[IncomingReportType] = None,
                 ):
        self.incoming_report: IncomingReportType
        if incoming_report is None:
            self.incoming_report = RetrieveReport(depth=depth+1, prog_hook=prog_hook)
        else:
            self.incoming_report = incoming_report
        if n_expected is None and self.incoming_report.n_expected is not None:
            n_expected = self.incoming_report.n_expected
        self.store_reports: StaticStoreReport = StaticStoreReport(prog_hook=prog_hook)
        super().__init__(description, depth, prog_hook, n_expected, keep_errors)

    @property
    def n_success(self) -> int:
        return (super().n_success
                + self.incoming_report.n_success
                + self.store_reports.n_success)

    @property
    def n_errors(self) -> int:
        return (super().n_errors
                + self.incoming_report.n_errors
                + self.store_reports.n_errors)

    @property
    def n_warnings(self) -> int:
        return (super().n_warnings
                + self.incoming_report.n_warnings
                + self.store_reports.n_warnings)

    @property
    def n_reported(self) -> int:
        return self.store_reports.n_input

    def add_store_report(self,
                         dest: DataBucket[Any, Any],
                         store_report: StoreReportType) -> None:
        '''Add a DicomOpReport or LocalWriteReport to keep track of'''
        assert dest not in self.store_reports
        if self.n_expected is not None and store_report.n_expected is None:
            store_report.n_expected = self.n_expected
        self.store_reports[dest] = store_report

    def log_issues(self) -> None:
        '''Produce log messages for any warning/error statuses'''
        self.incoming_report.log_issues()
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
            try:
                self.incoming_report.check_errors()
            except IncomingDataError as e:
                err.incoming_error = e
            raise err

    def clear(self) -> None:
        super().clear()
        self.incoming_report.clear()
        self.store_reports.clear()

    def _set_depth(self, val: int) -> None:
        if val != self._depth:
            self._depth = val
            self.incoming_report.depth = val + 1
            self.store_reports.depth = val + 1


class StaticOobTransferReport(MultiDictReport[TransferMethod, StaticStoreReport]):
    '''Transfer report for out-of-band transfers'''


class StaticTransferError(Exception):
    def __init__(self,
                 proxy_error: Optional[ProxyTransferError] = None,
                 oob_error: Optional[MultiKeyedError] = None):
        self.proxy_error = proxy_error
        self.oob_error = oob_error

    def __str__(self) -> str:
        res = ['StaticTransferError:']
        if self.proxy_error is not None:
            res.append("\tProxy Error: %s" % str(self.proxy_error))
        if self.oob_error is not None:
            res.append("\tOut-of-band Error: %s" % str(self.oob_error))
        return '\n'.join(res)


class StaticTransferReport(MultiAttrReport):
    '''Capture all possible info about a single StaticTranfer'''

    def __init__(self, 
                 description: Optional[str] = None, 
                 depth: int = 0,
                 prog_hook: Optional[ProgressHookBase[Any]] = None,
                 ):   
        self.proxy_report: Optional[StaticProxyTransferReport] = None
        self.oob_report: Optional[StaticOobTransferReport] = None
        self._report_attrs = ['proxy_report', 'oob_report']
        super().__init__(description, depth, prog_hook) 

    def check_errors(self) -> None:
        '''Raise an exception if any errors have occured so far'''
        if self.has_errors:
            err = StaticTransferError()
            if self.proxy_report is not None:
                try:
                    self.proxy_report.check_errors()
                except ProxyTransferError as e:
                    err.proxy_error = e
            if self.oob_report is not None:
                try:
                    self.oob_report.check_errors()
                except MultiKeyedError as e:
                    err.oob_error = e
            raise err


T_report = TypeVar('T_report', bound=Union[DynamicTransferReport, StaticTransferReport], covariant=True)


@dataclass
class Transfer(Generic[T_report]):
    chunk: DataChunk

    report: T_report


@dataclass
class DynamicTransfer(Transfer['DynamicTransferReport']):
    chunk: DataChunk

    report: DynamicTransferReport = field(default_factory=DynamicTransferReport)


@dataclass
class StaticTransfer(Transfer['StaticTransferReport']):
    chunk: DataChunk

    report: StaticTransferReport = field(default_factory=StaticTransferReport)

    method_routes_map: Dict[TransferMethod, Tuple[StaticRoute, ...]] = field(default_factory=dict)

    @property
    def proxy_filter_dest_map(self) -> Dict[Optional[Filter], Tuple[DataBucket[Any, Any], ...]]:
        '''Get dict mapping filters to destinations for proxy transfers'''
        filter_dest_map: Dict[Optional[Filter], Set[DataBucket[Any, Any]]] = {}
        routes = self.method_routes_map.get(TransferMethod.PROXY, tuple())
        for route in routes:
            filt = route.filt
            if filt not in filter_dest_map:
                filter_dest_map[filt] = set(d for d in route.dests)
            else:
                filter_dest_map[filt].update(route.dests)
        return {k : tuple(v) for k, v in filter_dest_map.items()}

    def get_dests(self, method: TransferMethod) -> Tuple[DataBucket[Any, Any], ...]:
        res = set()
        for route in self.method_routes_map.get(method, []):
            for dest in route.dests:
                res.add(dest)
        return tuple(res)


DiffFiltType = Callable[[DataDiff], Optional[DataDiff]]


def make_basic_validator(diff_filters: Optional[Iterable[DiffFiltType]] = None
                        ) -> Callable[[Dataset, Dataset], None]:
    '''Create validator that logs a warning on any differing elements

    List of filter functions can be supplied to modify/delete the diffs
    '''
    def basic_validator(src_ds: Dataset, dest_ds: Dataset) -> None:
        diffs = diff_data_sets(src_ds, dest_ds)
        if diff_filters is not None:
            warn_diffs = []
            d: Optional[DataDiff]
            for d in diffs:
                for filt in diff_filters:
                    assert d is not None
                    d = filt(d)
                    if d is None:
                        break
                else:
                    assert d is not None
                    warn_diffs.append(d)
        else:
            warn_diffs = diffs
        if len(warn_diffs) != 0:
            msg = ["Found differeing elements for ds %s:" %
                   '/'.join(get_all_uids(src_ds))
                  ]
            for d in warn_diffs:
                msg.append(textwrap.indent(str(d), '\t'))
            log.warn('\n'.join(msg))
    return basic_validator


T = TypeVar('T')


async def _sync_iter_to_async(sync_gen: Iterator[T]) -> AsyncIterator[T]:
    for result in sync_gen:
        yield result


TransferReportTypes = Union[DynamicTransferReport,
                            StaticTransferReport,
                           ]


DestType = Union[DataBucket, Route]


class RepoRequiredError(Exception):
    '''Operation requires a DataRepo but a DataBucket was provided'''


class SyncReport(MultiAttrReport):
    def __init__(self, 
                 description: Optional[str] = None, 
                 depth: int = 0,
                 prog_hook: Optional[ProgressHookBase[Any]] = None,
                 ):
        self._src_qr_report: Optional[MultiListReport[DicomOpReport]] = None 
        self.trans_reports: MultiListReport[TransferReportTypes] = MultiListReport('transfers', 
                                                                                   depth=depth + 1, 
                                                                                   prog_hook=prog_hook)
        self._report_attrs = ['_src_qr_report', 'trans_reports']
        super().__init__(description, depth, prog_hook)

    @property
    def src_qr_report(self) -> MultiListReport[DicomOpReport]:
        if self._src_qr_report is None:
            self._src_qr_report = MultiListReport('src_query', 
                                                  depth=self._depth + 1, 
                                                  prog_hook=self._prog_hook)
        return self._src_qr_report


# TODO: Does it make more sense to allow a query to be passed in here instead
#       having the base_query in the NetRepo? We only have once source here,
#       and the base_query functionality could be confusing, so I lean towards
#       yes.
#
#       How would the query/query_res interact though? Seems like the query_res
#       should override the query instead of the query refining the query_res?
#       This would be different than every other part of our API that takes
#       both though...
# TODO: If we end up needing to do the query ourselves, we should include any
#       req_elems from any DynamicRoutes so we don't have to do those queries
#       again in pre-route. One question is how does pre-route know that we
#       already tried to query for elements that the remote doesn't provide?
class SyncManager:
    '''Can generate and execute transfers needed to sync `src` and `dests`
    
    Data will only be retrieved locally from `src` at most once and then
    forwarded to all destinations that need it.

    Data that already exists on the destinations will be skipped, unless
    `force_all` is set to True.

    Parameters
    ----------
    src
        The source of data we are transferring

    dests
        One or more destinations for the data

    trust_level
        Assume data matches if sub-component counts match at this level

        Setting this to a level higher than IMAGE can speed up things
        significantly at the cost of accuracy. Has no effect if `force_all` is
        set to True.

    force_all
        Don't skip data that already exists on the destinations
    
    keep_errors
        Whether or not we try to sync erroneous data

    report
        Allows live introspection of the sync process, detailed results
    '''
    def __init__(self,
                 src: DataBucket[Any, Any],
                 dests: List[DestType],
                 trust_level: QueryLevel = QueryLevel.IMAGE,
                 force_all: bool = False,
                 keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
                 validators: Optional[Iterable[Callable[[Dataset, Dataset], None]]] = None,
                 report: Optional[SyncReport] = None,
                 ):
        self._src = src
        self._trust_level = trust_level
        self._force_all = force_all
        self._keep_errors: Tuple[IncomingErrorType, ...]
        if keep_errors == False:
            self._keep_errors = tuple()
        elif keep_errors == True:
            self._keep_errors = tuple(IncomingErrorType)
        else:
            keep_errors = cast(Tuple[IncomingErrorType, ...], keep_errors)
            self._keep_errors = keep_errors
        self._validators = validators
        if report is None:
            self._extern_report = False
            self.report = SyncReport()
        else:
            self._extern_report = True
            self.report = report

        # Make sure all dests are Route objects
        self._routes = []
        plain_dests: List[DataBucket[Any, Any]] = []
        for dest in dests:
            if isinstance(dest, Route):
                self._routes.append(dest)
        if plain_dests:
            self._routes.append(StaticRoute(tuple(plain_dests)))
        if len(self._routes) == 0:
            raise ValueError("At least one dest must be specified")
        self._has_filt = any(r.filt is not None for r in self._routes)

        # Precompute TransferMethod to routes map for static routes
        self._static_meth_routes = \
            self._get_meth_routes(r for r in self._routes
                                  if isinstance(r, StaticRoute))

        # Create an internal Router object
        self._router = Router(self._routes)

        # If we need to do dynamic routing due to a naive data source, make
        # sure it is possible
        if (not isinstance(self._src, DataRepo) and
            self._router.has_dynamic_routes and
            not self._router.can_dyn_route
           ):
            raise NoValidTransferMethodError()

    async def gen_transfers(self,
                            query_res: QueryResult = None,
                            ) -> AsyncIterator[Transfer[Any]]:
        '''Generate the needed transfers

        Parameters
        ----------
        query_res
            Only transfer data that matches this QueryResult
        '''
        if self.report.done:
            raise ValueError("The SyncManager has been closed")
        if not isinstance(self._src, DataRepo) and query_res is not None:
            raise RepoRequiredError("Can't pass in query_res with naive "
                                    "data source")

        # If any routes have filters, we need to tell the chunks to keep
        # inconsistent/duplicate data so the filters have a chance to fix it
        _chunk_keep_errors = set(self._keep_errors)
        if self._has_filt:
            _chunk_keep_errors.add(IncomingErrorType.INCONSISTENT)
            _chunk_keep_errors.add(IncomingErrorType.DUPLICATE)
        chunk_keep_errors = tuple(_chunk_keep_errors)

        n_trans = 0
        res: Transfer[Any]
        if not isinstance(self._src, DataRepo) or not self._router.can_pre_route:
            log.info(f"Processing all data from data source: {self._src}")
            if self._src.n_chunks is not None:
                self.report.trans_reports.n_expected = self._src.n_chunks
            async for chunk in self._src.gen_chunks():
                chunk.keep_errors = chunk_keep_errors
                chunk.report.prog_hook = self.report._prog_hook
                if self._router.has_dynamic_routes:
                    res = DynamicTransfer(chunk)
                else:
                    res = StaticTransfer(chunk, method_routes_map=self._static_meth_routes.copy())
                self.report.trans_reports.append(res.report)
                n_trans += 1
                yield res
        else:
            # We have a smart data repo and can precompute any dynamic routing
            # and try to avoid transferring data that already exists
            log.info(f"Processing select data from source: {self._src}")
            expected_sub_qrs = None
            qr_gen: AsyncIterator[QueryResult]
            if query_res is not None:
                gen_level = min(query_res.level, QueryLevel.STUDY)
                qr_gen = _sync_iter_to_async(query_res.level_sub_queries(gen_level))
                expected_sub_qrs = query_res.get_count(gen_level)
            else:
                q = dict_to_ds({elem : '*' for elem in self._router.required_elems})
                qr_report = self.report.src_qr_report
                qr_gen = self._src.queries(QueryLevel.STUDY, q, report=qr_report)

            n_sub_qr = 0
            avg_trans_per_qr = 1.0
            async for sub_qr in qr_gen:
                log.info(f"Processing {n_sub_qr} out of {expected_sub_qrs} sub-qr (avg trans per qr = {avg_trans_per_qr}")
                if expected_sub_qrs is None and qr_report.done:
                    expected_sub_qrs = qr_report.n_input
                if expected_sub_qrs is not None:
                    self.report.trans_reports.n_expected = int(expected_sub_qrs * avg_trans_per_qr)

                sr_qr_map = await self._router.pre_route(self._src,
                                                         query_res=sub_qr)
                pre_count = n_trans
                for static_routes, qr in sr_qr_map.items():
                    if self._force_all:
                        missing_info = {tuple(static_routes) : [qr]}
                    else:
                        missing_info = await self._get_missing(static_routes, qr)
                    for sub_routes, missing_qrs in missing_info.items():
                        meth_routes = self._get_meth_routes(sub_routes)
                        for missing_qr in missing_qrs:
                            async for chunk in self._src.gen_query_chunks(missing_qr):
                                chunk.keep_errors = chunk_keep_errors
                                chunk.report.prog_hook = self.report._prog_hook
                                if len(meth_routes) == 0:
                                    import pdb ; pdb.set_trace()
                                res = StaticTransfer(chunk, method_routes_map=meth_routes.copy())
                                self.report.trans_reports.append(res.report)
                                n_trans += 1
                                yield res
                avg_trans_per_qr = max(0.1, ((avg_trans_per_qr * n_sub_qr) + (n_trans - pre_count)) / (n_sub_qr + 1))
                n_sub_qr += 1
            log.info(f"Generated {n_trans} transfers")

    async def exec_transfer(self, transfer: Transfer[Any]) -> None:
        '''Execute the given transfer'''
        if self.report.done:
            raise ValueError("The SyncManager has been closed")
        if isinstance(transfer, DynamicTransfer):
            log.debug("Executing dynamic transfer")
            await self._do_dynamic_transfer(transfer)
        elif isinstance(transfer, StaticTransfer):
            log.debug("Executing static transfer")
            await self._do_static_transfer(transfer)
        else:
            raise TypeError("Not a valid Transfer sub-class: %s" % transfer)

    # TODO: If we decide to keep some associations open across transfers, we 
    #       can do the cleanup here but below three methods will need to become
    #       async
    def close(self) -> None:
        if not self._extern_report:
            self.report.log_issues()
            self.report.check_errors()
        self.report.done = True

    def __enter__(self) -> SyncManager:
        return self
    
    def __exit__(self, 
                 exctype: Optional[Type[BaseException]],
                 excinst: Optional[BaseException],
                 exctb: Optional[TracebackType]) -> None:
        self.close()

    def _get_meth_routes(self,
                         routes: Iterable[StaticRoute]
                        ) -> Dict[TransferMethod, Tuple[StaticRoute, ...]]:
        method_routes_map: Dict[TransferMethod, List[StaticRoute]] = {}
        for route in routes:
            method = route.get_method(self._src)
            if method not in method_routes_map:
                method_routes_map[method] = []
            method_routes_map[method].append(route)
        return {k: tuple(v) for k, v in method_routes_map.items()}

    async def _get_missing(self,
                           static_routes: Tuple[StaticRoute, ...],
                           query_res: QueryResult
                          ) -> Dict[Tuple[StaticRoute, ...], List[QueryResult]]:
        '''Determine what data is missing for the `static_routes`

        Each route might produce naive (i.e. `DataBucket`) or smart (i.e.
        DataRepo) destinations, but we can't do much in the naive case except
        assume that all of the data is missing.

        Parameters
        ----------
        static_routes : list of StaticRoute
            Determines one or more destinations for the data

        query_res : QueryResult
            Subset of data we want to check the existance of on destinations

        Returns
        -------
        missing_info : dict
            Maps tuples of routes to list of QueryResults that specify the
            data missing from that set of destinations.
        '''
        assert isinstance(self._src, DataRepo)
        log.debug("Finding missing data for src %s" % self._src)
        src_qr = query_res

        # Pair up dests with filters and split into two groups, those we can
        # check for missing data and those we can not
        dest_filt_tuples: List[Tuple[DataBucket[Any, Any], Optional[Filter]]] = []
        checkable: List[Tuple[DataRepo[Any, Any, Any, Any], Optional[Filter]]] = []
        non_checkable = []
        df_trans_map = {}
        for route in static_routes:
            filt = route.filt
            can_invert_uids = True
            if filt is not None:
                invertible_uids = filt.invertible_uids
                can_invert_uids = all(uid in invertible_uids
                                      for uid in uid_elems.values())
            for dest in route.dests:
                df_tuple = (dest, filt)
                dest_filt_tuples.append(df_tuple)
                df_trans_map[df_tuple] = get_transform(src_qr, filt)
                if isinstance(dest, DataRepo) and can_invert_uids:
                    df_tuple = cast(Tuple[DataRepo[Any, Any, Any, Any], Optional[Filter]], df_tuple)
                    checkable.append(df_tuple)
                else:
                    non_checkable.append(df_tuple)

        # Can't check any dests to see what is missing, so nothing to do
        if len(checkable) == 0:
            return {tuple(static_routes) : [query_res]}

        # We group data going to same sets of destinations
        res: Dict[Tuple[Tuple[DataRepo[Any, Any, Any, Any], Optional[Filter]], ...], List[QueryResult]] = {}
        for n_dest in reversed(range(1, len(checkable)+1)):
            for df_set in itertools.combinations(checkable, n_dest):
                if df_set not in res:
                    res[tuple(df_set)] = []

        # Check for missing data at each query level, starting from coarsest
        # (i.e. entire missing patients, then stuides, etc.)
        curr_matching = {df : df_trans_map[df].new for df in checkable}
        curr_src_qr = src_qr
        for curr_level in range(src_qr.level, QueryLevel.IMAGE + 1):
            curr_level = QueryLevel(curr_level)
            if len(curr_src_qr) == 0:
                break
            log.debug("Checking for missing data at level %s" % curr_level)
            if curr_level > curr_src_qr.level:
                # We need more details for the source QueryResult
                log.debug("Querying src in _get_missing more details")
                curr_src_qr = await self._src.query(level=curr_level,
                                                    query_res=curr_src_qr)
                df_trans_map = {df : get_transform(curr_src_qr & qr_trans.old, df[1])
                                for df, qr_trans in df_trans_map.items()}

            # Compute what is missing for each dest and matching for any dest
            # at this level
            missing = {}
            full_matching: Optional[QueryResult] = None
            for df in checkable:
                dest, filt = df
                log.debug("Checking for missing data on dest: %s", dest)
                assert isinstance(dest, DataRepo)
                curr_qr_trans = df_trans_map[df]
                dest_qr = await dest.query(level=curr_level,
                                           query_res=curr_matching[df])
                missing[df] = curr_qr_trans.old.sub(curr_qr_trans.reverse(dest_qr).qr,
                                                    ignore_subcounts=True)
                matching = curr_qr_trans.new & dest_qr
                if curr_level == self._trust_level:
                    done_uids = []
                    for uid in matching.uids():
                        node = DataNode(curr_level, uid)
                        if (curr_qr_trans.new.sub_query(node, curr_level) ==
                            dest_qr.sub_query(node)
                           ):
                            done_uids.append(uid)
                    for uid in done_uids:
                        del matching[uid]
                curr_matching[df] = matching
                old_matching = curr_qr_trans.reverse(matching).qr
                df_trans_map[df] = get_transform(old_matching, filt)
                if full_matching is None:
                    full_matching = old_matching
                else:
                    full_matching |= old_matching
                log.debug("missing = %s \n matching = %s" , missing[df], curr_matching[df])
            log.debug("full matching = %s", full_matching)

            # Reduce the source qr to only data that matches on at least one dest
            if full_matching is not None:
                log.debug("remaing pre-update = %s", curr_src_qr)
                curr_src_qr = curr_src_qr & full_matching
                log.debug("remaining post-update = %s", curr_src_qr)

            # Update the results with the missing data for this level
            for df_set, qr_list in res.items():
                # Build set of all missing data across destinations
                set_missing = None
                for df in df_set:
                    if set_missing is None:
                        set_missing = deepcopy(missing[df])
                    else:
                        set_missing = set_missing & missing[df]
                assert set_missing is not None
                if len(set_missing) > 0:
                    for df in df_set:
                        missing[df] -= set_missing
                    if (len(qr_list) > 0 and
                        qr_list[-1].level == set_missing.level
                       ):
                        qr_list[-1] |= set_missing
                    else:
                        qr_list.append(set_missing)

        # Convert back to routes and return result
        sr_res = {}
        for df_set, qr_list in res.items():
            if len(qr_list) == 0:
                continue
            filt_dest_map: Dict[Optional[Filter], List[DataBucket[Any, Any]]] = {}
            for dest, filt in df_set:
                if filt not in filt_dest_map:
                    filt_dest_map[filt] = []
                filt_dest_map[filt].append(dest)
            routes = []
            for filt, dests in filt_dest_map.items():
                routes.append(StaticRoute(tuple(dests), filt=filt))
            sr_res[tuple(routes)] = qr_list
        return sr_res

    async def _do_dynamic_transfer(self, transfer: DynamicTransfer) -> None:
        # TODO: We could keep this context manager open until the
        #       TransferExecutor.close method is called, and thus avoid some
        #       overhead from opening/closing associations, but this makes the
        #       reporting quite tricky, and each transfer should be relatively
        #       slow compared to the overhead of setting up and tearing down
        #       associations.
        async with self._router.route(report=transfer.report) as routing_q:
            async for ds in transfer.chunk.gen_data():
                await routing_q.put(ds)

    async def _do_static_proxy_transfer(self,
                                        transfer: StaticTransfer,
                                        report: StaticProxyTransferReport
                                       ) -> None:
        filter_dest_map = transfer.proxy_filter_dest_map
        n_filt = len(filter_dest_map)
        if None in filter_dest_map:
            n_filt -= 1
        dests = transfer.get_dests(TransferMethod.PROXY)
        async with AsyncExitStack() as stack:
            d_q_map = {}
            for dest in dests:
                #TODO: Might be a LocalWriteReport
                store_rep = dest.get_empty_send_report()
                store_rep.n_expected = transfer.chunk.n_expected
                report.add_store_report(dest, store_rep)
                d_q_map[dest] = await stack.enter_async_context(dest.send(report=store_rep))
            async for ds in transfer.chunk.gen_data():
                if n_filt:
                    orig_ds = deepcopy(ds)
                else:
                    orig_ds = ds
                min_orig_ds = minimal_copy(orig_ds)
                for filt, sub_dests in filter_dest_map.items():
                    static_route = StaticRoute(sub_dests, filt=filt)
                    sub_queues = [d_q_map[d] for d in sub_dests]
                    if filt is not None:
                        filt_ds = filt(orig_ds)
                        if filt_ds is not None:
                            min_filt_ds = minimal_copy(filt_ds)
                    else:
                        filt_ds = orig_ds
                        min_filt_ds = min_orig_ds
                    if filt_ds is None:
                        continue
                    if not report.add(static_route, min_orig_ds, min_filt_ds):
                        continue
                    if filt_ds is not None:
                        for q in sub_queues:
                            await q.put(filt_ds)
        report.done = True

    async def _do_static_transfer(self, transfer: StaticTransfer) -> None:
        # TODO: Can't automatically overlap the proxy and out-of-band transfers
        #       since they both may require associations with the same src.
        #       Would need to know the available resources, and those needed
        #       by each transfer, including a way for a transfer to reserve
        #       resources for future use
        #
        #       Our current API also doesn't allow the user to do this manually...
        for method, routes in transfer.method_routes_map.items():
            if method == TransferMethod.PROXY:
                proxy_report = StaticProxyTransferReport(incoming_report=transfer.chunk.report)
                proxy_report.keep_errors = self._keep_errors
                transfer.report.proxy_report = proxy_report
                await self._do_static_proxy_transfer(transfer, proxy_report)
            else:
                oob_report = StaticOobTransferReport(prog_hook=self.report._prog_hook)
                oob_report[method] = StaticStoreReport()
                transfer.report.oob_report = oob_report
                for dest in transfer.get_dests(method):
                    oob_dest = cast(OobCapable[Any, Any], dest)
                    log.debug(f"Doing out-of-band transfer to {dest}")
                    oob_report[method][dest] = oob_dest.get_empty_oob_report()
                    oob_report[method][dest].n_expected = transfer.chunk.n_expected
                    oob_report[method][dest].description = f'{method} to {dest}'
                    await oob_dest.oob_transfer(method,
                                                transfer.chunk,
                                                report=oob_report[method][dest])
        transfer.report.done = True


async def sync_data(src : DataBucket[Any, Any],
                    dests : List[DestType],
                    query_res: Optional[QueryResult] = None,
                    trust_level: QueryLevel = QueryLevel.IMAGE,
                    force_all: bool = False,
                    validators: Optional[Iterable[Callable[[Dataset, Dataset], None]]] = None,
                    keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
                    report: Optional[SyncReport] = None,
                    ) -> None:
    '''Convienance function to build TransferPlanner and execute all transfers
    '''
    with SyncManager(src, dests, trust_level, force_all, keep_errors, validators, report) as sm:
        async for transfer in sm.gen_transfers(query_res):
            await sm.exec_transfer(transfer)
