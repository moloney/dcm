'''DICOM Networking'''
import asyncio, time, logging, warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import asynccontextmanager
from functools import partial
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, FrozenSet
from pathlib import Path

import janus
from fifolock import FifoLock
import pydicom
from pydicom.dataset import Dataset
from pydicom.datadict import keyword_for_tag
from pydicom.uid import (ExplicitVRLittleEndian,
                         ImplicitVRLittleEndian,
                         DeflatedExplicitVRLittleEndian,
                         ExplicitVRBigEndian,
                         UID)

from pynetdicom import (AE, evt, build_context, DEFAULT_TRANSFER_SYNTAXES,
                        QueryRetrievePresentationContexts,
                        StoragePresentationContexts,
                        VerificationPresentationContexts,
                        sop_class)
from pynetdicom.status import code_to_category
from pynetdicom.sop_class import StorageServiceClass, MRImageStorage
from pynetdicom.pdu_primitives import SOPClassCommonExtendedNegotiation


from .info import __version__
from .query import (QueryLevel, QueryResult, InconsistentDataError, uid_elems,
                    req_elems, opt_elems, choose_level, minimal_copy,
                    get_all_uids)


log = logging.getLogger(__name__)


UID_PREFIX = '2.25'

IMPLEMENTATION_UID = '%s.84718903' % UID_PREFIX

siemens_mr_sop_neg = SOPClassCommonExtendedNegotiation()
siemens_mr_sop_neg.sop_class_uid = '1.3.12.2.1107.5.9.1'
siemens_mr_sop_neg.service_class_uid = StorageServiceClass.uid
siemens_mr_sop_neg.related_general_sop_class_identification = [MRImageStorage]
c_store_ext_sop_negs = [siemens_mr_sop_neg]

# TODO: We still need the below presentation when we are the SCU? Or we can
# use the extended negotiation there too?

## Add in SOP Class for storing proprietary Siemens data
#private_sop_classes = {'SiemensProprietaryMRStorage' : '1.3.12.2.1107.5.9.1'}
#sop_class._STORAGE_CLASSES.update(private_sop_classes)
#sop_class._generate_sop_classes(private_sop_classes)
#for private_uid in private_sop_classes.values():
#    StoragePresentationContexts.append(build_context(private_uid))


# TODO: Everytime we establish/close an association, it should be done in a
#       separate thread, since this is a blocking operation (though usually
#       quite fast). Probably makes sense to wait until we refactor associations
#       into some sort of AssociationManager object that can also manage
#       caching of associations and limits on number of simultaneous
#       associations with any given remote.


@dataclass(frozen=True)
class DcmNode:
    '''DICOM network entity info'''
    host: str
    ae_title: str = 'ANYAE'
    port: int = 11112
    query_models: tuple = ('S', 'P')

    def __str__(self):
        return '%s:%s:%s' % (self.host, self.ae_title, self.port)


class FailedAssociationError(Exception):
    '''We were unable to associate with a remote network node'''


class BatchDicomOperationError(Exception):
    '''Base class for errors from DICOM batch network operations'''
    def __init__(self, op_errors):
        self.op_errors = op_errors


@dataclass
class DicomOpReport:
    '''Track status results from DICOM operations'''
    provider: Optional[DcmNode] = None
    user: Optional[DcmNode] = None
    op_type: Optional[str] = None
    op_data: dict = field(default_factory=dict)
    n_success: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    done: bool = False

    @property
    def n_errors(self):
        return len(self.errors)

    @property
    def n_warnings(self):
        return len(self.warnings)

    @property
    def all_success(self):
        return self.n_errors + self.n_warnings == 0

    def add(self, status, data_set):
        if (not hasattr(status, 'Status') or
            code_to_category(status.Status) == 'Failure'
           ):
            if self.op_type == 'c-store':
                data_set = minimal_copy(data_set)
            self.errors.append((status, data_set))
            log.debug("%s op got error status: %s" % (self.op_type, status))
        elif code_to_category(status.Status) == 'Warning':
            if self.op_type == 'c-store':
                data_set = minimal_copy(data_set)
            self.warnings.append((status, data_set))
            log.debug("%s op got warning status: %s" % (self.op_type, status))
        else:
            self.n_success += 1

    def __len__(self):
        return self.n_success + self.n_warnings + self.n_errors

    def log_issues(self):
        '''Log a summary of error/warning statuses'''
        if self.n_errors != 0:
            log.error("Got %d error and %d warning statuses out of %d %s ops" %
                      (self.n_errors, self.n_warnings, len(self), self.op_type))
        elif self.n_warnings != 0:
            log.warning("Got %d warning statuses out of %d %s ops" %
                        (self.n_warnings, len(self), self.op_type))

    def check_errors(self):
        '''Raise an exception if any errors occured'''
        if self.n_errors != 0:
            raise BatchDicomOperationError(self.errors)

    def clear(self):
        '''Clear out all current operation results'''
        self.n_success = 0
        self.warnings = []
        self.errors = []


class IncomingDataError(Exception):
    '''Captures errors detected in incoming data stream'''
    def __init__(self, inconsistent, duplicate):
        self.inconsistent = inconsistent
        self.duplicate = duplicate

    def __str__(self):
        res = ['IncomingDataError:']
        for err_type in ('inconsistent', 'duplicate'):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            n_errors = len(errors)
            if n_errors != 0:
                res.append('%d %s,' % (n_errors, err_type))
        return ' '.join(res)


class RetrieveError(Exception):
    '''Capture errors that happened during a retrieve operation'''
    def __init__(self, inconsistent, duplicate, unexpected, missing, move_errors):
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.unexpected = unexpected
        self.missing = missing
        self.move_errors = move_errors

    def __str__(self):
        res = ['RetrieveError:']
        for err_type in ('inconsistent', 'unexpected', 'duplicate', 'missing', 'move_errors'):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            n_errors = len(errors)
            if n_errors != 0:
                res.append('%d %s,' % (n_errors, err_type))
        return ' '.join(res)


# TODO: Do we need this split out from RetrieveReport? It doesn't capture much
#       of interest that doesn't already get captured in the TransferReport
#       since we use DataTransforms there (to catch incoming vs outgoing data)
#       which will also capture data that was inconsistent/duplicate and then
#       got fixed by the filters.
#
#
@dataclass
class IncomingDataReport:
    retrieved: QueryResult = field(default_factory=lambda: QueryResult(QueryLevel.IMAGE))
    inconsistent: list = field(default_factory=list)
    duplicate: list = field(default_factory=list)
    keep_errors: bool = False
    _done: field(init=False, repr=False) = False

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, val):
        if not val:
            raise ValueError("Setting `done` to False is not allowed")
        if self._done:
            raise ValueError("Report was already marked done")
        self._done = True

    @property
    def n_errors(self):
        res = 0
        if not self.keep_errors:
            res += len(self.inconsistent) + len(self.duplicate)
        return res

    @property
    def n_warnings(self):
        res = 0
        if self.keep_errors:
            res += len(self.inconsistent) + len(self.duplicate)
        return res

    @property
    def all_success(self):
        return self.n_errors + self.n_warnings == 0

    def add(self, data_set):
        assert not self.done
        try:
            dupe = data_set in self.retrieved
        except InconsistentDataError:
            self.inconsistent.append(get_all_uids(data_set))
            return False
        else:
            if dupe:
                self.duplicate.append(get_all_uids(data_set))
                return False
        self.retrieved.add(data_set)
        return True

    def log_issues(self):
        '''Log any warnings and errors'''
        error_msg = []
        for err_type in ('inconsistent', 'duplicate'):
            errors = getattr(self, err_type)
            n_errors = len(errors)
            if n_errors != 0:
                error_msg.append('%d %s' % (n_errors, err_type))
        if error_msg:
            if self.keep_errors:
                log.warn("Incoming data issues: %s" %
                         ' '.join(error_msg))
            else:
                log.error("Incoming data issues: %s" %
                          ' '.join(error_msg))

    def check_errors(self):
        if self.n_errors:
            raise IncomingDataError(self.inconsistent, self.duplicate)


# TODO: Do we need a clear method here?
@dataclass
class RetrieveReport(IncomingDataReport):
    '''Track details about a retrieve operation'''
    requested: Optional[QueryResult] = None
    missing: Optional[QueryResult] = None
    unexpected: list = field(default_factory=list)
    move_report: DicomOpReport = field(default_factory=DicomOpReport)

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, val):
        if not val:
            raise ValueError("Setting `done` to False is not allowed")
        if self._done:
            raise ValueError("RetrieveReport was already marked done")
        assert self.move_report.done
        self.missing = self.requested - self.retrieved
        self._done = True

    @property
    def n_errors(self):
        res = self.move_report.n_errors
        if self._done:
            res += len(self.missing)
        if not self.keep_errors:
            res += len(self.inconsistent) + len(self.unexpected) + len(self.duplicate)
        return res

    @property
    def n_warnings(self):
        res = self.move_report.n_warnings
        if self.keep_errors:
            res += len(self.inconsistent) + len(self.unexpected) + len(self.duplicate)
        return res

    def add(self, data_set):
        assert not self.done
        try:
            expected = data_set in self.requested
        except InconsistentDataError:
            self.inconsistent.append(get_all_uids(data_set))
            return False
        else:
            if not expected:
                self.unexpected.append(get_all_uids(data_set))
                return False
        # TODO: In theory this next line could also raise an
        #       InconsistentDataError, not sure how likely that is...
        if data_set in self.retrieved:
            self.duplicate.append(get_all_uids(data_set))
            return False
        # TODO: Can we avoid duplicate work making min copies here and in the store report?
        self.retrieved.add(minimal_copy(data_set))
        return True

    def log_issues(self):
        '''Log any warnings and errors'''
        self.move_report.log_issues()
        error_msg = []
        for err_type in ('inconsistent', 'unexpected', 'duplicate'):
            errors = getattr(self, err_type)
            n_errors = len(errors)
            if n_errors != 0:
                error_msg.append('%d %s' % (n_errors, err_type))
        if error_msg and self.keep_errors:
            log.warn("Data errors during retrieval: %s" % ' '.join(error_msg))
            error_msg = []
        if self._done:
            n_missing = len(self.missing)
            if n_missing != 0:
                error_msg.append('%d missing' % n_missing)
        if error_msg:
            log.error("Data errors during retrieval: %s" % ' '.join(error_msg))

    def check_errors(self):
        '''Raise an exception if any errors occured'''
        if self.n_errors != 0:
            raise RetrieveError(self.inconsistent if not self.keep_errors else [],
                                self.duplicate if not self.keep_errors else [],
                                self.unexpected if not self.keep_errors else [],
                                self.missing,
                                self.move_report.errors)


def _query_worker(res_q, rep_q, assoc, level, queries, query_model, split_level=None):
    '''Worker function for performing queries in a separate thread
    '''
    if split_level is None:
        if level == QueryLevel.PATIENT or level == QueryLevel.STUDY:
            split_level = level
        else:
            split_level= QueryLevel(level - 1)
    elif split_level > level:
        raise ValueError("The split_level can't be higher than the query level")
    split_attr = uid_elems[split_level]
    last_split_val = None

    for query in queries:
        res = QueryResult(level)
        missing_attrs = set()
        for status, rdat in assoc.send_c_find(query, query_model=query_model):
            rep_q.put((status, rdat))
            if rdat is None:
                break
            log.debug("Got query response:\n%s" % rdat)
            split_val = getattr(rdat, split_attr)
            if last_split_val != split_val:
                if len(res) != 0:
                    res_q.put((res, missing_attrs))
                    res = QueryResult(level)
                    missing_attrs = set()
            last_split_val = split_val
            rdat_keys = rdat.keys()
            # TODO: Can't we just examine the qr at a higher level to determine this?
            for tag in query.keys():
                if tag not in rdat_keys:
                    keyword = keyword_for_tag(tag)
                    missing_attrs.add(keyword)
            res.add(rdat)
        if len(res) != 0:
            res_q.put((res, missing_attrs))
    res_q.put(None)
    rep_q.put(None)


def _move_worker(rep_q, assoc, dest, query_res, query_model):
    '''Worker function for perfoming move operations in another thread'''
    for d in query_res:
        d.QueryRetrieveLevel = query_res.level.name
        log.debug("Sending move request:\n%s" % d)
        responses = assoc.send_c_move(d,
                                      dest.ae_title,
                                      query_model=query_model)
        time.sleep(0.01)
        for status, rdat in responses:
            rep_q.put((status, rdat))
    rep_q.put(None)


def _send_worker(send_q, rep_q, assoc):
    '''Worker function for performing move operations in another thread'''
    while True:
        ds = send_q.get()
        if ds is None:
            log.debug("Shutting down send worker thread")
            rep_q.put(None)
            break
        log.debug("Send worker got a data set")
        status = assoc.send_c_store(ds)
        rep_q.put((status, ds))


class UnsupportedQueryModelError(Exception):
    '''The requested query model isn't supported by the remote entity'''


@dataclass(frozen=True)
class EventFilter:
    '''Define a filter that matches a subset of pynetdicom events'''
    event_types: Optional[FrozenSet[evt.InterventionEvent]] = None
    ae_titles: Optional[FrozenSet[str]] = None

    def matches(self, event):
        '''Test if the `event` matches the filter'''
        # TODO: The _event attr is made public as 'evt' in future pynetdicom
        if self.event_types is not None and event._event not in self.event_types:
            return False
        if self.ae_titles is not None:
            norm_ae = event.assoc.requestor.ae_title.decode('ascii').strip()
            if norm_ae not in self.ae_titles:
                return False
        return True

    def collides(self, other):
        '''Test if this filter collides with the `other` filter'''
        evt_collision = remote_collision = False
        if self.event_types is None or other.event_types is None:
            evt_collision = True
        elif len(self.event_types & other.event_types) != 0:
            evt_collision = True
        if self.ae_titles is None or other.ae_titles is None:
            remote_collision = True
        elif len(self.ae_titles & other.ae_titles) != 0:
            remote_collision = True
        print("collides is returning: %s" % evt_collision and remote_collision)
        return evt_collision and remote_collision


class FilteredListenerLockBase(asyncio.Future):
    '''Base class for creating event filter aware lock types with fifolock'''
    @classmethod
    def is_compatible(cls, holds):
        for lock_type, n_holds in holds.items():
            if n_holds > 0 and cls.event_filter.collides(lock_type.event_filter):
                return False
        return True


default_evt_pc_map = {evt.EVT_C_ECHO : VerificationPresentationContexts,
                      evt.EVT_C_STORE : StoragePresentationContexts,
                      evt.EVT_C_FIND : QueryRetrievePresentationContexts,
                     }
'''Map event types to the default presentation contexts the AE needs to accept
'''


def make_sync_to_async_cb(async_cb, loop):
    '''Return sync callback that bridges to an async callback'''
    def sync_cb(event):
        # Calling get_event_loop from sync code can return the wrong loop
        # sometimes, in which case the below code hangs without error.
        # This assertion should catch that issue.
        assert loop.is_running()
        f = asyncio.run_coroutine_threadsafe(async_cb(event), loop)
        res = f.result()
        return res
    return sync_cb


# TODO: Do data validation here, and if requested return error status when
#       we detect issues with the incoming data
def _make_retrieve_cb(res_q):
    async def retrieve_cb(event):
        await res_q.put((event.dataset, event.file_meta))
        return 0x0 # Success
    return retrieve_cb


class LocalEntity:
    '''Low level interface to DICOM networking functionality

    Params
    ------
    local : DcmNode
        The local DICOM network node properties
    '''

    def __init__(self, local, transfer_syntaxes=None, max_threads=8):
        self._local = local
        if transfer_syntaxes is None:
            self._default_ts = DEFAULT_TRANSFER_SYNTAXES
        else:
            self._default_ts = transfer_syntaxes
        self._thread_pool = ThreadPoolExecutor(max_threads)
        self._listener_lock = FifoLock()
        self._lock_types = {}
        self._event_handlers = {}
        self._listen_mgr = None

    @property
    def local(self):
        return self._local

    async def echo(self, remote):
        '''Perfrom an "echo" against `remote` to test connectivity

        Returns True if successful, else False.
        '''
        loop = asyncio.get_event_loop()
        echo_ae = AE(ae_title=self._local.ae_title)
        assoc = echo_ae.associate(remote.host,
                                  remote.port,
                                  VerificationPresentationContexts,
                                  remote.ae_title)
        if not assoc.is_established:
            log.error("Failed to associate with remote: %s" % str(remote))
            return False
        try:
            status = await loop.run_in_executor(self._thread_pool,
                                                assoc.send_c_echo)
        finally:
            assoc.release()

        if status and status.Status == 0x0:
            return True
        return False

    def _prep_query(self, level, query, query_res):
        '''Resolve/check `level` and `query` args for query methods'''
        # Build up our base query dataset
        if query is None:
            query = Dataset()
        elif not isinstance(query, Dataset):
            qdict = query
            query = Dataset()
            for key, val in qdict.items():
                setattr(query, key, val)
        else:
            query = deepcopy(query)

        # Deterimine level if not specified, otherwise make sure it is valid
        if level is None:
            if query_res is None:
                default_level = QueryLevel.STUDY
            else:
                default_level = query_res.level
            level = choose_level(query, default_level)
        elif level not in QueryLevel:
            raise ValueError("Unknown 'level' for query: %s" % level)
        return level, query

    async def query(self, remote, level=None, query=None, query_res=None,
                    query_model=None, report=None):
        '''Query the `remote` entity all at once

        See documentation for the `queries` method for details
        '''
        level, query = self._prep_query(level, query, query_res)
        res = QueryResult(level)
        async for sub_res in self.queries(remote,
                                          level,
                                          query,
                                          query_res,
                                          query_model,
                                          report):
            res |= sub_res
        return res

    async def queries(self, remote, level=None, query=None, query_res=None,
                      query_model=None, report=None):
        '''Query the `remote` entity in an iterative manner

        Params
        ------
        remote : DcmNode
            The network node we are querying

        level : QueryLevel
            The level of detail we want to query

            If not provided, we try to automatically determine the most
            appropriate level based on the `query`, then the `query_res`,
            and finally fall back to using `QueryLevel.STUDY`.

        query : dicom.Dataset or dict
            Specifies the DICOM attributes we want to query with/for

        query_res : QueryResult
            A query result that we want to refine

        report : DicomOpReport
            If provided, will store status report from DICOM operations
        '''
        if report is None:
            extern_report = False
            report = DicomOpReport()
        else:
            extern_report = True
        report.provider = remote
        report.user = self._local
        report.op_type = 'c-find'
        level, query = self._prep_query(level, query, query_res)

        # If we are missing some required higher-level identifiers, we
        # perform a recursive pre-query to get those first
        if level == QueryLevel.SERIES and getattr(query, 'StudyInstanceUID', '') == '':
            if query_res is None or query_res.level < QueryLevel.STUDY:
                log.debug("Performing recursive STUDY level query against %s" %
                          (remote,))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    query_res = await self.query(remote,
                                                 QueryLevel.STUDY,
                                                 query,
                                                 query_res)
        elif level == QueryLevel.IMAGE and getattr(query, 'SeriesInstanceUID', '') == '':
            if query_res is None or query_res.level < QueryLevel.SERIES:
                log.debug("Performing recursive SERIES level query against %s" %
                          (remote,))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    query_res = await self.query(remote,
                                                 QueryLevel.SERIES,
                                                 query,
                                                 query_res)
        log.debug("Performing %s level query against %s" % (level.name, remote))

        # Determine the query model
        query_model = self._choose_query_model(remote, level, query_model)

        # Add in any required and default optional elements to query
        for lvl in QueryLevel:
            for req_attr in req_elems[lvl]:
                if getattr(query, req_attr, None) is None:
                    setattr(query, req_attr, '')
            if lvl == level:
                break
        auto_attrs = set()
        for opt_attr in opt_elems[level]:
            if getattr(query, opt_attr, None) is None:
                setattr(query, opt_attr, '')
                auto_attrs.add(opt_attr)

        # Set the QueryRetrieveLevel
        query.QueryRetrieveLevel = level.name

        # If QueryResult was given we potentially generate multiple
        # queries, one for each dataset referenced by the QueryResult
        if query_res is None:
            queries = [query]
        else:
            queries = []
            for path, sub_uids in query_res.walk():
                if path.level == min(level, query_res.level):
                    q = deepcopy(query)
                    for lvl in QueryLevel:
                        if lvl > path.level:
                            break
                        setattr(q, uid_elems[lvl], path.uids[lvl])
                    queries.append(q)
                    sub_uids.clear()
            log.debug("QueryResult expansion results in %d sub-queries" %
                      len(queries))

        # Build a queue for results from query thread
        loop = asyncio.get_running_loop()
        res_q = janus.Queue(loop=loop)
        rep_q = janus.Queue(loop=loop)

        # Build an AE to perform the query
        qr_ae = AE(ae_title=self._local.ae_title)

        # TODO: We should be making the association in a seperate thread too, right?
        # Create association with the remote node
        log.debug("Making association with %s" % (remote,))
        assoc = qr_ae.associate(remote.host,
                                remote.port,
                                QueryRetrievePresentationContexts,
                                remote.ae_title)
        if not assoc.is_established:
            raise FailedAssociationError("Can't associate with remote "
                                         "node: %s" % str(remote))

        # Fire up a thread to perform the query and produce QueryResult chunks
        try:
            rep_builder_task = \
                asyncio.create_task(self._report_builder(rep_q.async_q,
                                                         report)
                                   )
            query_fut = loop.run_in_executor(self._thread_pool,
                                             partial(_query_worker,
                                                     res_q.sync_q,
                                                     rep_q.sync_q,
                                                     assoc,
                                                     level,
                                                     queries,
                                                     query_model,
                                                    )
                                             )
            qr_fut_done = False
            qr_fut_exception = None
            assoc_released = False
            while True:
                try:
                    res = await asyncio.wait_for(res_q.async_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if qr_fut_exception:
                        raise qr_fut_exception
                    assert qr_fut_done == False
                    # Check if the worker thread is done
                    if query_fut.done():
                        qr_fut_done = True
                        qr_fut_exception = query_fut.exception()
                        assoc.release()
                        assoc_released = True
                else:
                    if res is None:
                        break
                    qr, missing_attrs = res
                    for missing_attr in missing_attrs:
                        if missing_attr not in auto_attrs:
                            warnings.warn(f"Remote node {remote} doesn't "
                                          "support querying on {missing_attr}")
                    yield qr
            await query_fut
            await rep_builder_task
        finally:
            report.done = True
            if not assoc_released:
                assoc.release()
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def _fwd_event(self, event):
        for filt, handler in self._event_handlers.items():
            if filt.matches(event):
                log.debug(f"Calling async handler for {event} event")
                return await handler(event)
        log.warn(f"Can't find handler for the {event} event")
        return 0x0122

    def _setup_listen_mgr(self, sync_cb, presentation_contexts):
        log.debug("Starting a threaded listener")
        # TODO: How to handle presentation contexts in generic way?
        ae = AE(ae_title=self._local.ae_title)
        for context in presentation_contexts:
            ae.add_supported_context(context.abstract_syntax,
                                     self._default_ts)
        self._listen_mgr = ae.start_server((self._local.host,
                                            self._local.port),
                                           block=False)
        for evt_type in default_evt_pc_map.keys():
            log.debug(f"Binding to event {evt_type}")
            self._listen_mgr.bind(evt_type, sync_cb)

    async def _cleanup_listen_mgr(self):
        log.debug("Cleaning up threaded listener")
        try:
            self._listen_mgr.shutdown()
        finally:
            self._listen_mgr = None

    def _get_lock_type(self, event_filter):
        lock_type = self._lock_types.get(event_filter)
        if lock_type is None:
            lock_type = type('FilteredListenerLock',
                             (FilteredListenerLockBase,),
                             {'event_filter' : event_filter})
            self._lock_types[event_filter] = lock_type
        return lock_type

    @asynccontextmanager
    async def listen(self, handler, event_filter=None, presentation_contexts=None):
        '''Listen for incoming DICOM network events

        The async callback `handler` will be called for each DICOM network
        event that matches the `event_filter`. The default filter matches all
        events.

        Locks are used internally to prevent two listeners with colliding event
        filters from running simultaneously. Avoiding deadlocks due to
        dependency loops must be handled at a higher level.
        '''
        if event_filter is None:
            event_filter = EventFilter()
        if presentation_contexts is None:
            presentation_contexts = []
            if event_filter.event_types is None:
                for p_contexts in default_evt_pc_map.values():
                    presentation_contexts += p_contexts
            else:
                for event_type in event_filter.event_types:
                    presentation_contexts += default_evt_pc_map[event_type]
        LockType = self._get_lock_type(event_filter)
        log.debug("About to wait on listener lock")
        async with self._listener_lock(LockType):
            log.debug("Listener lock acquired")
            # If we don't already have a threaded listener running, set it up
            if self._listen_mgr is None:
                sync_cb = make_sync_to_async_cb(self._fwd_event,
                                                asyncio.get_running_loop())
                self._setup_listen_mgr(sync_cb, presentation_contexts)
            # Some sanity checks that our locking scheme is working
            assert event_filter not in self._event_handlers
            assert all(not event_filter.collides(f)
                       for f in self._event_handlers)
            self._event_handlers[event_filter] = handler
            try:
                yield
            finally:
                del self._event_handlers[event_filter]
                # If no one else is listening, or waiting to listen, clean up
                # the threaded listener
                if (len(self._listener_lock._waiters) == 0 and
                    sum(self._listener_lock._holds.values()) == 1
                   ):
                    await self._cleanup_listen_mgr()
        log.debug("Listener lock released")

    async def move(self, src, dest, query_res, transfer_syntax=None,
                   query_model=None, report=None):
        '''Move DICOM files from one network entity to another'''
        if report is None:
            extern_report = False
            report = DicomOpReport()
        else:
            extern_report = True
        report.provider = src
        report.user = self._local
        report.op_type = 'c-move'
        report.op_data = {'dest' : dest}
        loop = asyncio.get_running_loop()
        rep_q = janus.Queue(loop=loop)
        # Setup the association
        query_model = self._choose_query_model(src, query_res.level, query_model)
        if transfer_syntax is None:
            transfer_syntax = self._default_ts
        #import pdb ; pdb.set_trace()
        move_ae = AE(ae_title=self._local.ae_title)
        #TODO: Need different contexts here...
        assoc = move_ae.associate(src.host,
                                  src.port,
                                  QueryRetrievePresentationContexts,
                                  src.ae_title)
        if not assoc.is_established:
            raise FailedAssociationError("Failed to associate with "
                                         "remote: %s" % str(src))
        try:
            rep_builder_task = \
                asyncio.create_task(self._report_builder(rep_q.async_q,
                                                         report)
                                   )
            await loop.run_in_executor(self._thread_pool,
                                       partial(_move_worker,
                                               rep_q.sync_q,
                                               assoc,
                                               dest,
                                               query_res,
                                               query_model)
                                              )
            await rep_builder_task
        finally:
            log.debug("Setting done flag on move op report")
            report.done = True
            assoc.release()
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def retrieve(self, remote, query_res, report=None, keep_errors=False):
        '''Generate data sets from `remote` based on `query_res`.

        Parameters
        ----------
        remote : DcmNode
            The remote DICOM network node we want to get data from

        query_res : QueryResult
            The data we want to retrieve

        report : RetrieveReport
            If provided this will be populated asynchronously

        keep_errors : bool
            Generate data sets even if we detect issues with them

            By default inconsistent, unexpected, and duplicate data are skipped
        '''
        if report is None:
            extern_report = False
            report = RetrieveReport(requested=query_res)
        else:
            report.requested = query_res
            extern_report = True
        event_filter = EventFilter(event_types=frozenset((evt.EVT_C_STORE,)),
                                   ae_titles=frozenset((remote.ae_title,))
                                  )
        res_q = asyncio.Queue()
        retrieve_cb = _make_retrieve_cb(res_q)
        async with self.listen(retrieve_cb, event_filter):
            move_task = asyncio.create_task(self.move(remote,
                                                      self._local,
                                                      query_res,
                                                      report=report.move_report)
                                           )
            while True:
                try:
                    ds, file_meta  = await asyncio.wait_for(res_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Check if the move task is done
                    if move_task.done():
                        break
                    else:
                        continue
                # Add the data set to our report and handle errors
                success = report.add(ds)
                if not success and not keep_errors:
                    continue
                # Add the file_meta to the data set and yield it
                file_meta.ImplementationVersionName = __version__
                ds.file_meta = file_meta
                ds.is_little_endian = file_meta.TransferSyntaxUID.is_little_endian
                ds.is_implicit_VR = file_meta.TransferSyntaxUID.is_implicit_VR
                yield ds
            log.debug("About to await move task")
            await move_task
        report.done = True
        if not extern_report:
            report.log_issues()
            log.debug("About to check errors")
            report.check_errors()

    async def download(self, remote, query_res, dest_dir):
        '''Uses `retrieve` method to download data and saves it to `dest_dir`
        '''
        # TODO: Handle collisions/overwrites
        loop = asyncio.get_running_loop()
        dest_dir = Path(dest_dir)
        out_files = []
        async for ds in self.retrieve(remote, query_res):
            out_path = str(dest_dir / ds.SOPInstanceUID) + '.dcm'
            await loop.run_in_executor(self._thread_pool,
                                       partial(pydicom.dcmwrite, out_path, ds))
            out_files.append(out_path)
        return out_files

    async def _report_builder(self, res_q, report):
        while True:
            res = await res_q.get()
            if res is None:
                break
            status, data_set = res
            # TODO: OpReport needs to minimize the stored data set as needed
            report.add(status, data_set)

    @asynccontextmanager
    async def send(self, remote, transfer_syntax=None, report=None):
        '''Produces a queue where you can put data sets to be sent to `remote`

        Parameters
        ----------
        remote : DcmNode
            The remote node we are sending the data to

        transfer_syntax:
            The DICOM transfer syntax to use for the transfer

        report: DicomOpReport
            If provided, the results of the send operations will be put here

            This report will be updated throughout the transfer and can be
            examined within the context manager for partial results. The report
            will not always be up-to-date with the most recent datasets put
            into the queue (until the context manager is closed).

            If this is provided, the normal warning/error behavior is skipped
            as it is assumed the caller will handle this themselves (e.g. by
            calling the `log_issues` and `check_errors` methods on the report).
        '''
        if transfer_syntax is None:
            transfer_syntax = self._default_ts
        if report is None:
            extern_report = False
            report = DicomOpReport()
        else:
            extern_report = True
        report.provider = remote
        report.user = self._local
        report.op_type = 'c-store'
        loop = asyncio.get_running_loop()
        send_q = janus.Queue(10, loop=loop)
        rep_q = janus.Queue(10, loop=loop)
        send_ae = AE(ae_title=self._local.ae_title)
        log.debug(f"About to associate with {remote} to send data")
        assoc = send_ae.associate(remote.host,
                                  remote.port,
                                  StoragePresentationContexts,
                                  remote.ae_title,
                                  ext_neg=c_store_ext_sop_negs)
        if not assoc.is_established:
            raise FailedAssociationError("Failed to associate with "
                                         "remote: %s" % str(remote))
        try:
            rep_builder_task = \
                asyncio.create_task(self._report_builder(rep_q.async_q,
                                                         report)
                                   )
            send_fut = loop.run_in_executor(self._thread_pool,
                                            partial(_send_worker,
                                                    send_q.sync_q,
                                                    rep_q.sync_q,
                                                    assoc)
                                           )
            yield send_q.async_q
        finally:
            try:
                # Signal send worker to shutdown, then wait for it
                log.debug("Shutting down send worker")
                await send_q.async_q.put(None)
                await send_fut
                await rep_builder_task
            finally:
                log.debug("Releasing send association")
                report.done = True
                assoc.release()

        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def upload(self, remote, src_paths, transfer_syntax=None):
        '''Uses `send` method to upload data from local `src_paths`'''
        loop = asyncio.get_running_loop()
        async with self.send(remote, transfer_syntax) as send_q:
            for src_path in src_paths:
                log.debug(f"Uploading file: {src_path}")
                ds = await loop.run_in_executor(self._thread_pool,
                                                partial(pydicom.dcmread,
                                                        str(src_path))
                                               )
                await send_q.put(ds)

    def _choose_query_model(self, remote, level, query_model=None):
        '''Pick an appropriate query model'''
        if level == QueryLevel.PATIENT:
            if 'P' not in remote.query_models:
                raise UnsupportedQueryModelError()
            if query_model is not None and query_model != 'P':
                raise ValueError("Doing a PATIENT level query requires the "
                                 "'P' query model")
            return 'P'
        if query_model is not None:
            if query_model not in remote.query_models:
                raise UnsupportedQueryModelError()
            return query_model
        else:
            if 'S' in remote.query_models:
                return 'S'
            elif 'P' in remote.query_models:
                return 'P'
            elif 'O' in remote.query_models:
                return 'O'
            else:
                raise UnsupportedQueryModelError()
