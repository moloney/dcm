'''Synchronize DICOM data between local and/or remote locations'''
import os, logging, shutil, itertools, textwrap, warnings
from collections import OrderedDict
from copy import deepcopy
from tempfile import mkdtemp
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from contextlib import AsyncExitStack, asynccontextmanager

from .query import (QueryLevel, QueryResult, DataNode, InconsistentDataError,
                    get_all_uids, minimal_copy)
from .net import DicomOpReport, DcmNode, RetrieveReport, RetrieveError
from .store import DataChunk, DataRepo
from .filt import get_transform
from .route import (Route, StaticRoute, DynamicRoute, Router,
                    LocalTransferReport, DynamicTransferReport, TransferError)
from .diff import diff_data_sets


log = logging.getLogger(__name__)


def make_basic_validator(diff_filters=None):
    '''Create validator that logs a warning on any differing elements

    List of filter functions can be supplied to modify/delete the diffs
    '''
    def basic_validator(src_ds, dest_ds):
        diffs = diff_data_sets(src_ds, dest_ds)
        if diff_filters is not None:
            warn_diffs = []
            for d in diffs:
                for filt in diff_filters:
                    d = filt(d)
                    if d is None:
                        break
                else:
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


@dataclass
class DynamicTransfer:
    chunk: DataChunk


@dataclass
class StaticTransfer:
    chunk: DataChunk
    routes: Tuple[StaticRoute]

    @property
    def dests(self):
        res = set()
        for route in self.routes:
            for dest in route.dests:
                res.add(dest)
        return tuple(res)

    @property
    def filter_dest_map(self):
        filter_dest_map = {}
        for route in self.routes:
            filt = route.filt
            if filt not in filter_dest_map:
                filter_dest_map[filt] = set(d for d in route.dests)
            else:
                filter_dest_map[filt].update(route.dests)
        return {k : tuple(v) for k, v in filter_dest_map.items()}


@dataclass
class StaticTransferReport(LocalTransferReport):
    '''Track a static data transfer and the corresponding storage reports'''
    store_reports: Dict[DcmNode, DicomOpReport] = field(default_factory=dict)

    @property
    def n_errors(self):
        n_errors = super().n_errors
        for dest, op_report in self.store_reports.items():
            n_errors += op_report.n_errors
        return n_errors

    @property
    def n_warnings(self):
        n_warn = super().n_warnings
        for dest, op_report in self.store_reports.items():
            n_warn += op_report.n_warnings
        return n_warn

    @property
    def n_reported(self):
        return sum(len(rep) for _, rep in self.store_reports.items())

    def add_op_report(self, dest, op_report):
        '''Add a DicomOpReport to keep track of'''
        assert dest not in self.store_reports
        self.store_reports[dest] = op_report

    def log_issues(self):
        '''Produce log messages for any warning/error statuses'''
        super().log_issues()
        for dest, op_report in self.store_reports.items():
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
            for dest, op_report in self.store_reports.items():
                errors = op_report.errors
                if errors:
                    store_errors[dest] = errors
            if store_errors:
                err.store_errors = store_errors
            raise err

    def clear(self):
        super().clear()
        for dest, op_report in self.store_reports.items():
            op_report.clear()


#TODO: Do we need a custom clear method here (depends if we add one to
#      RetrieveReport)
@dataclass
class ProxyTransferReport(StaticTransferReport):
    '''Track a static transfer where we retrieve the data and forward it
    '''
    retrieve_report: Optional[RetrieveReport] = field(default_factory=lambda: RetrieveReport())

    @property
    def n_errors(self):
        rrep = self.retrieve_report
        if rrep is None:
            rrep_n_err = 0
        else:
            rrep_n_err = rrep.n_errors
            if not rrep.keep_errors:
                rrep_n_err -= len(rrep.inconsistent) + len(rrep.duplicate)
        return super().n_errors + rrep_n_err

    @property
    def n_warnings(self):
        rrep = self.retrieve_report
        if rrep is None:
            rrep_n_warn = 0
        else:
            rrep_n_warn = rrep.n_warnings
            if rrep.keep_errors:
                rrep_n_warn -= len(rrep.inconsistent) + len(rrep.duplicate)
        return super().n_warnings + rrep_n_warn

    def log_issues(self):
        '''Produce log messages for any warning/error statuses'''
        # TODO: Clear dupe/inconsist in retrieve report here?
        super().log_issues()
        if self.retrieve_report is not None:
            self.retrieve_report.log_issues()

    def check_errors(self):
        '''Raise an exception if any errors have occured so far'''
        if self.n_errors:
            trans_err = None
            try:
                super().check_errors()
            except TransferError as e:
                trans_err = e
            else:
                trans_err = TransferError()

            # Ignore dupe/inconsitent errors on retrieve so we give filters a
            # chance to correct errors. If the error isn't fixed it will be
            # captured in trans_err already
            rrep = self.retrieve_report
            if rrep is not None:
                rrep_n_err = rrep.n_errors
                if not rrep.keep_errors:
                    rrep_n_err -= len(rrep.inconsistent) + len(rrep.duplicate)
                if rrep_n_err:
                    if not rrep.keep_errors:
                        rrep = deepcopy(rrep)
                        rrep.inconsistent.clear()
                        rrep.duplicate.clear()
                    try:
                        rrep.check_errors()
                    except RetrieveError as e:
                        trans_err.retrieve_error = e
                    else:
                        assert False

            raise trans_err


@dataclass
class SyncReport:
    transfer_reports: List = field(default_factory=list)

    @property
    def done(self):
        return all(r.done for r in self.transfer_reports)

    @property
    def n_warnings(self):
        return sum(r.n_warnings for r in self.transfer_reports)

    @property
    def n_errors(self):
        return sum(r.n_errors for r in self.transfer_reports)

    @property
    def all_success(self):
        return self.n_errors + self.n_warnings == 0

    def log_issues(self):
        '''Produce log messages for any warning/error statuses'''
        for trans_rep in self.transfer_reports:
            trans_rep.log_issues()

    def check_errors(self):
        '''Raise an exception if any errors have occured so far'''
        for trans_rep in self.transfer_reports:
            trans_rep.check_errors()

    def clear(self):
        incomplete = []
        for rep_idx, trans_rep in enumerate(self.transfer_reports):
            if not trans_rep.done():
                incomplete.append(trans_rep)
                trans_rep.clear()
        self.transfer_reports = incomplete


async def _sync_gen_to_async(sync_gen):
    for result in sync_gen:
        yield result


class TransferExecutor:
    '''Manage the execution of a series of transfers'''
    def __init__(self, router, validators=None, always_proxy=False,
                 keep_errors=False):
        self.report = SyncReport()
        self._router = router
        self._validators = validators
        self._always_proxy = always_proxy
        self._keep_errors = keep_errors


    async def _do_dynamic_transfer(self, transfer):
        # TODO: We could keep this context manager open until the
        #       TransferExecutor.close method is called, and thus avoid some
        #       overhead from opening/closing associations, but this makes the
        #       reporting quite tricky, and each transfer should be relatively
        #       slow compared to the overhead of setting up and tearing down
        #       associations.
        report = DynamicTransferReport()
        self.report.transfer_reports.append(report)
        async with self._router.route(report=report) as routing_q:
            async for ds in transfer.chunk.gen_data():
                await routing_q.put(ds)

    async def _do_static_transfer(self, transfer):
        moved_dests = []
        send_dests = transfer.dests
        filter_dest_map = transfer.filter_dest_map
        n_filt = len(filter_dest_map)
        if None in filter_dest_map and self._validators is None and not self._always_proxy:
            # TODO: Attempt to do direct move where src/dest support it
            #       Should still proxy everything by default if multiple dests
            #       to save source bandwidth. Probably need a way to force
            #       moves instead of proxying to acommodate restricted
            #       permissions
            if len(filter_dest_map) == 1:
                send_dests = [d for d in send_dests if not d in moved_dests]
        log.debug(f"Performing transfer to {send_dests}")
        if len(send_dests) == 0:
            return
        report = ProxyTransferReport()
        report.retrieve_report = transfer.chunk.report
        self.report.transfer_reports.append(report)
        async with AsyncExitStack() as stack:
            d_q_map = {}
            for dest in send_dests:
                store_rep = DicomOpReport()
                report.store_reports[dest] = store_rep
                d_q_map[dest] = await stack.enter_async_context(dest.send(report=store_rep))
            async for ds in transfer.chunk.gen_data():
                if n_filt > 1:
                    orig_ds = deepcopy(ds)
                else:
                    orig_ds = ds
                min_orig_ds = minimal_copy(orig_ds)
                for filt, dests in filter_dest_map.items():
                    dests = [d for d in dests if d in send_dests]
                    if len(dests) == 0:
                        continue
                    static_route = StaticRoute(dests, filt)
                    sub_queues = [d_q_map[d] for d in dests]
                    if filt is not None:
                        filt_ds = filt(orig_ds)
                        min_filt_ds = minimal_copy(filt_ds)
                    else:
                        filt_ds = orig_ds
                        min_filt_ds = min_orig_ds
                    report.add(static_route, min_orig_ds, min_filt_ds)
                    if filt_ds is not None:
                        for q in sub_queues:
                            await q.put(filt_ds)

    async def exec_transfer(self, transfer):
        '''Execute the given transfer'''
        if isinstance(transfer, DynamicTransfer):
            log.debug("Executing dynamic transfer")
            await self._do_dynamic_transfer(transfer)
        elif isinstance(transfer, StaticTransfer):
            log.debug("Executing static transfer")
            await self._do_static_transfer(transfer)
        else:
            raise TypeError("Not a valid Transfer sub-class: %s" % transfer)

    async def close(self):
        # TODO: Some clean up to do here?
        pass


class RepoRequiredError(Exception):
    '''Operation requires a DataRepo but a DataBucket was provided'''
    pass


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

# TODO: How to interleave queries/transfers? Three main places src queries can
#       happen, top-level in gen_transfers, in pre_route, and in get_missing.
#       There is the potential for the PACS to have a small limit on the number
#       of simultaneous associations for any given client, which will prevent
#       certain overlapping schemes from working.
class TransferPlanner:
    '''Plans efficient transfer scheme for data from `src` along `routes`

    Data will only be retrieved locally from `src` at most once and then
    forwarded to all destinations that need it.

    Data that already exists on the destinations will be skipped, unless
    `force_all` is set to True.

    Parameters
    ----------
    src: DataBucket
        The source of data we are transferring

    dests: list of DataBucket or Route
        One or more destinations for the data

    trust_level: QueryLevel
        Assume data matches if sub-component counts match at this level

        Setting this to a level higher than IMAGE can speed up things
        significantly at the cost of accuracy. Has no effect if `force_all` is
        set to True.

    force_all: bool
        Don't skip data that already exists on the destinations
    '''
    def __init__(self, src, dests, trust_level=QueryLevel.IMAGE,
                 force_all=False):
        self._src = src
        self._trust_level = trust_level
        self._force_all = force_all

        # Make sure all dests are Route objects
        self._routes = []
        plain_dests = []
        for dest in dests:
            if isinstance(dest, Route):
                self._routes.append(dest)
            else:
                plain_dests.append(dest)
        if plain_dests:
            self.routes.append(StaticRoute(plain_dests))
        self._router = Router(self._routes)

    async def _get_missing(self, static_routes, query_res):
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
        log.debug("Finding missing data for src %s" % self._src)
        src_qr = query_res

        # Pair up dests with filters and split into two groups, those we can
        # check for missing data and those we can not
        dest_filt_tuples = []
        checkable = []
        non_checkable = []
        df_trans_map = {}
        for route in static_routes:
            filt = route.filt
            can_invert_uids = True
            if filt is not None:
                can_invert_uids = filt.invertible_uids
            for dest in route.dests:
                df_tuple = (dest, filt)
                dest_filt_tuples.append(df_tuple)
                df_trans_map[df_tuple] = get_transform(src_qr, filt)
                if isinstance(dest, DataRepo) and can_invert_uids:
                    checkable.append(df_tuple)
                else:
                    non_checkable.append(df_tuple)

        # Can't check any dests to see what is missing, so nothing to do
        if len(checkable) == 0:
            return {tuple(static_routes) : [query_res]}

        # We group data going to same sets of destinations
        res = OrderedDict()
        for n_dest in reversed(range(1, len(checkable)+1)):
            for df_set in itertools.combinations(checkable, n_dest):
                if df_set not in res:
                    res[df_set] = []

        # Check for missing data at each query level, starting from coarsest
        # (i.e. entire missing patients, then stuides, etc.)
        curr_matching = {df : df_trans_map[df].new for df in checkable}
        curr_src_qr = src_qr
        #curr_qr_trans = src_qr_trans #TODO: How to update this?
        for curr_level in QueryLevel:
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
            full_matching = None
            for df in checkable:
                dest, filt = df
                curr_qr_trans = df_trans_map[df]
                dest_qr = await dest.query(level=curr_level,
                                           query_res=curr_matching[df])
                missing[df] = curr_qr_trans.old - curr_qr_trans.reverse(dest_qr).qr
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

            # Reduce the source qr to only data that matches on at least one dest
            curr_src_qr = curr_src_qr & full_matching

            # Update the results with the missing data for this level
            for df_set, qr_list in res.items():
                # Build set of all missing data across destinations
                set_missing = None
                for df in df_set:
                    if set_missing is None:
                        set_missing = missing[df]
                    else:
                        set_missing = set_missing & missing[df]
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
            filt_dest_map = {}
            for dest, filt in df_set:
                if filt not in filt_dest_map:
                    filt_dest_map[filt] = []
                filt_dest_map[filt].append(dest)
            routes = []
            for filt, dests in filt_dest_map.items():
                routes.append(StaticRoute(dests, filt))
            sr_res[tuple(routes)] = qr_list
        return sr_res

    async def gen_transfers(self, query_res=None, req_queried=False):
        '''Generate the needed transfers

        query_res: QueryResult
            Only transfer data that matches this QueryResult
        '''
        if not isinstance(self._src, DataRepo) and query_res is not None:
            raise RepoRequiredError("Can't pass in query_res with naive "
                                    "data source")

        n_trans = 0
        if not isinstance(self._src, DataRepo) or not self._router.can_pre_route:
            log.info("Processing all data from data source: %s" % self._src)
            has_dyn = any(isinstance(r, DynamicRoute) for r in self._routes)
            async for chunk in self._src.gen_chunks():
                if has_dyn:
                    yield DynamicTransfer(chunk)
                else:
                    yield StaticTransfer(chunk, self._routes)
                n_trans += 1
        else:
            # We have a smart data repo and can precompute any dynamic routing
            # and try to avoid transferring data that already exists
            log.info("Processing select data from source: %s" % self._src)
            if query_res is not None:
                gen_level = min(query_res.level, QueryLevel.STUDY)
                qr_gen = _sync_gen_to_async(query_res.level_sub_queries(gen_level))
            else:
                # TODO: Should track max assoc for remote and do this query
                #       ahead of time all at once if that limit is low
                q = {elem : '*' for elem in self._router.required_elems}
                qr_gen = self._src.queries(QueryLevel.STUDY, q)
                req_queried = True

            async for sub_qr in qr_gen:
                sr_qr_map = await self._router.pre_route(self._src,
                                                         query_res=sub_qr,
                                                         req_queried=req_queried)
                for static_routes, qr in sr_qr_map.items():
                    if self._force_all:
                        missing_info = {tuple(static_routes) : [qr]}
                    else:
                        missing_info = await self._get_missing(static_routes, qr)
                    for sub_routes, missing_qrs in missing_info.items():
                        for missing_qr in missing_qrs:
                            async for chunk in self._src.gen_query_chunks(missing_qr):
                                yield StaticTransfer(chunk, sub_routes)
                                n_trans += 1
            log.info("Generated transfers for %d chunks of data" % n_trans)

    @asynccontextmanager
    async def executor(self, validators=None, always_proxy=False,
                       keep_errors=False):
        '''Produces a TransferExecutor for executing a series of transfers'''
        # TODO: Just make the executor a contexmanager and return it here
        try:
            executor = TransferExecutor(self._router, validators, always_proxy, keep_errors)
            yield executor
        finally:
            await executor.close()


async def sync_data(src, dests, query_res=None, trust_level=QueryLevel.IMAGE,
                    force_all=False, validators=None, always_proxy=False,
                    keep_errors=False):
    '''Convienance function to build TransferPlanner and execute all transfers
    '''
    planner = TransferPlanner(src, dests, trust_level, force_all)
    async with planner.executor(validators, always_proxy, keep_errors) as ex:
        report = ex.report
        async for transfer in planner.gen_transfers(query_res):
            await ex.exec_transfer(transfer)
    report.log_issues()
    report.check_errors()
