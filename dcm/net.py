'''DICOM Networking'''
from __future__ import annotations
import asyncio, time, logging, warnings, enum, json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import asynccontextmanager
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import (List, Set, Dict, Tuple, Optional, FrozenSet, Union, Any,
                    AsyncIterator, Iterator, Callable, Awaitable, Type,
                    TYPE_CHECKING, cast)
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

from pynetdicom import (AE, Association, evt, build_context,
                        DEFAULT_TRANSFER_SYNTAXES,
                        QueryRetrievePresentationContexts,
                        StoragePresentationContexts,
                        VerificationPresentationContexts,
                        sop_class)
from pynetdicom.status import code_to_category
from pynetdicom.sop_class import (StorageServiceClass, MRImageStorage)
from pynetdicom.pdu_primitives import SOPClassCommonExtendedNegotiation
from pynetdicom.transport import ThreadedAssociationServer


from .info import __version__
from .query import (QueryLevel, QueryResult, InconsistentDataError, uid_elems,
                    req_elems, opt_elems, choose_level, minimal_copy,
                    get_all_uids)
from .util import IndividualReport, serializer, Serializable


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


QR_MODELS = {'PatientRoot' :
                {'find' : sop_class.PatientRootQueryRetrieveInformationModelFind,
                 'move' : sop_class.PatientRootQueryRetrieveInformationModelMove,
                 'get' : sop_class.PatientRootQueryRetrieveInformationModelGet},
             'StudyRoot' :
                 {'find' : sop_class.StudyRootQueryRetrieveInformationModelFind,
                  'move' : sop_class.StudyRootQueryRetrieveInformationModelMove,
                  'get' : sop_class.StudyRootQueryRetrieveInformationModelGet},
             'PatientStudyOnly' :
                 {'find' : sop_class.PatientStudyOnlyQueryRetrieveInformationModelFind,
                  'move' : sop_class.PatientStudyOnlyQueryRetrieveInformationModelMove,
                  'get' : sop_class.PatientStudyOnlyQueryRetrieveInformationModelGet},
            }

@serializer
@dataclass(frozen=True)
class DcmNode:
    '''DICOM network entity info'''
    host: str
    '''Hostname of the node'''

    ae_title: str = 'ANYAE'
    '''DICOM AE Title of the node'''

    port: int = 11112
    '''DICOM port for the node'''

    qr_models: Tuple[str, ...] = ('StudyRoot', 'PatientRoot')
    '''Supported DICOM QR models for the node'''

    def __str__(self) -> str:
        return '%s:%s:%s' % (self.host, self.ae_title, self.port)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> DcmNode:
        return cls(**json_dict)



class BatchDicomOperationError(Exception):
    '''Base class for errors from DICOM batch network operations'''
    def __init__(self, op_errors: List[Dataset]):
        self.op_errors = op_errors


@dataclass
class DicomOpReport(IndividualReport):
    '''Track status results from DICOM operations'''

    provider: Optional[DcmNode] = None
    '''The service provider'''

    user: Optional[DcmNode] = None
    '''The service user'''

    op_type: Optional[str] = None
    '''The type of operation performed'''

    op_data: Dict[str, Any] = field(default_factory=dict)
    '''Additional data describing the operation specifics'''

    warnings: List[Dataset] = field(default_factory=list, init=False)
    '''List of data sets we got warning status for'''

    errors: List[Dataset] = field(default_factory=list, init=False)
    '''List of data sets we got error status for'''

    _n_success: int = field(default=0, init=False)

    _n_input: int = field(default=0, init=False)

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_success(self) -> int:
        return self._n_success

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    def add(self, status: Dataset, data_set: Dataset) -> None:
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
            self._n_success += 1
        self._n_input += 1

    def log_issues(self) -> None:
        '''Log a summary of error/warning statuses'''
        if self.n_errors != 0:
            log.error("Got %d error and %d warning statuses out of %d %s ops" %
                      (self.n_errors, self.n_warnings, len(self), self.op_type))
        elif self.n_warnings != 0:
            log.warning("Got %d warning statuses out of %d %s ops" %
                        (self.n_warnings, len(self), self.op_type))

    def check_errors(self) -> None:
        '''Raise an exception if any errors occured'''
        if self.n_errors != 0:
            raise BatchDicomOperationError(self.errors)

    def clear(self) -> None:
        '''Clear out all current operation results'''
        self._n_success = 0
        self.warnings = []
        self.errors = []


class IncomingErrorType(enum.Enum):
    INCONSISTENT = enum.auto()
    DUPLICATE = enum.auto()
    UNEXPECTED = enum.auto()


class IncomingDataError(Exception):
    '''Captures errors detected in incoming data stream'''
    def __init__(self,
                 inconsistent: Optional[List[Tuple[str, ...]]],
                 duplicate: Optional[List[Tuple[str, ...]]]):
        self.inconsistent = inconsistent
        self.duplicate = duplicate

    def __str__(self) -> str:
        res = ['IncomingDataError:']
        for err_type in ('inconsistent', 'duplicate'):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            n_errors = len(errors)
            if n_errors != 0:
                res.append('%d %s,' % (n_errors, err_type))
        return ' '.join(res)


@dataclass
class IncomingDataReport(IndividualReport):
    '''Generic incoming data report'''

    retrieved: QueryResult = field(default_factory=lambda: QueryResult(QueryLevel.IMAGE))
    '''Track the valid incoming data'''

    inconsistent: List[Tuple[str, ...]] = field(default_factory=list)
    '''Track the inconsistent incoming data'''

    duplicate: List[Tuple[str, ...]] = field(default_factory=list)
    '''Track the duplicate incoming data'''

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
    def n_input(self) -> int:
        return len(self.retrieved) + len(self.inconsistent) + len(self.duplicate)

    @property
    def n_success(self) -> int:
        res = len(self.retrieved)
        return res

    @property
    def n_errors(self) -> int:
        res = 0
        if IncomingErrorType.INCONSISTENT not in self.keep_errors:
            res += len(self.inconsistent)
        if IncomingErrorType.DUPLICATE not in self.keep_errors:
            res += len(self.duplicate)
        return res

    @property
    def n_warnings(self) -> int:
        res = 0
        if IncomingErrorType.INCONSISTENT in self.keep_errors:
            res += len(self.inconsistent)
        if IncomingErrorType.DUPLICATE in self.keep_errors:
            res += len(self.duplicate)
        return res

    def add(self, data_set: Dataset) -> bool:
        '''Add an incoming data set, returns True if it should should be used
        '''
        assert not self.done
        try:
            dupe = data_set in self.retrieved
        except InconsistentDataError:
            self.inconsistent.append(get_all_uids(data_set))
            return IncomingErrorType.INCONSISTENT in self.keep_errors
        else:
            if dupe:
                self.duplicate.append(get_all_uids(data_set))
                return IncomingErrorType.DUPLICATE in self.keep_errors
        self.retrieved.add(data_set)
        return True

    def log_issues(self) -> None:
        '''Log any warnings and errors'''
        error_msg = []
        warn_msg = []
        for err_type in IncomingErrorType:
            err_attr = err_type.name.lower()
            if not hasattr(self, err_attr):
                continue
            errors = getattr(self, err_attr)
            n_errors = len(errors)
            msg = f"{n_errors} {err_type}"
            if n_errors != 0:
                if err_type in self.keep_errors:
                    warn_msg.append(msg)
                else:
                    error_msg.append(msg)
        if warn_msg:
            log.warn("Incoming data issues: %s" % ' '.join(warn_msg))
        if error_msg:
            log.error("Incoming data issues: %s" % ' '.join(error_msg))

    def check_errors(self) -> None:
        if self.n_errors:
            kwargs = {}
            if IncomingErrorType.INCONSISTENT not in self.keep_errors:
                kwargs['inconsistent'] = self.inconsistent
            if IncomingErrorType.DUPLICATE not in self.keep_errors:
                kwargs['duplicate'] = self.duplicate
            raise IncomingDataError(**kwargs)

    def clear(self) -> None:
        self.inconsistent = []
        self.duplicate = []


class RetrieveError(IncomingDataError):
    '''Capture errors that happened during a retrieve operation'''
    def __init__(self,
                 inconsistent: Optional[List[Tuple[str,...]]],
                 duplicate: Optional[List[Tuple[str,...]]],
                 unexpected: Optional[List[Tuple[str,...]]],
                 missing: Optional[QueryResult],
                 move_errors: Optional[List[Dataset]]):
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.unexpected = unexpected
        self.missing = missing
        self.move_errors = move_errors

    def __str__(self) -> str:
        res = ['RetrieveError:']
        for err_type in ('inconsistent', 'unexpected', 'duplicate', 'missing', 'move_errors'):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            n_errors = len(errors)
            if n_errors != 0:
                res.append('%d %s,' % (n_errors, err_type))
        return ' '.join(res)


# TODO: Update this to use new keep_errors setup.
@dataclass
class RetrieveReport(IncomingDataReport):
    '''Track details about a retrieve operation'''
    requested: Optional[QueryResult] = None
    '''The data that was requested'''

    missing: Optional[QueryResult] = None
    '''Any requested data that was not recieved'''

    unexpected: List[Tuple[str, ...]] = field(default_factory=list)
    '''Any data sets that we received but did not expect'''

    move_report: DicomOpReport = field(default_factory=DicomOpReport)
    '''Report on move operations'''

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, val: bool) -> None:
        if not val:
            raise ValueError("Setting `done` to False is not allowed")
        if self._done:
            raise ValueError("RetrieveReport was already marked done")
        assert self.move_report.done
        assert self.requested is not None and self.retrieved is not None
        self.missing = self.requested - self.retrieved
        self._done = True

    @property
    def n_expected(self) -> Optional[int]:
        if self.requested is None:
            return None
        return self.requested.n_instances()

    @property
    def n_input(self) -> int:
        return super().n_input + len(self.unexpected)

    @property
    def n_errors(self) -> int:
        res = super().n_errors + self.move_report.n_errors
        if self._done:
            assert self.missing is not None
            res += len(self.missing)
        if IncomingErrorType.UNEXPECTED not in self._keep_errors:
            res += len(self.unexpected)
        return res

    @property
    def n_warnings(self) -> int:
        res = super().n_warnings + self.move_report.n_warnings
        if IncomingErrorType.UNEXPECTED in self._keep_errors:
            res += len(self.unexpected)
        return res

    def add(self, data_set: Dataset) -> bool:
        assert not self._done
        assert self.requested is not None
        try:
            expected = data_set in self.requested
        except InconsistentDataError:
            self.inconsistent.append(get_all_uids(data_set))
            return IncomingErrorType.INCONSISTENT in self._keep_errors
        else:
            if not expected:
                self.unexpected.append(get_all_uids(data_set))
                return IncomingErrorType.UNEXPECTED in self._keep_errors
        if data_set in self.retrieved:
            self.duplicate.append(get_all_uids(data_set))
            return IncomingErrorType.DUPLICATE in self._keep_errors
        # TODO: Can we avoid duplicate work making min copies here and in the store report?
        self.retrieved.add(minimal_copy(data_set))
        return True

    def log_issues(self) -> None:
        '''Log any warnings and errors'''
        super().log_issues()
        self.move_report.log_issues()
        if self._done:
            assert self.missing is not None
            n_missing = len(self.missing)
            if n_missing != 0:
                log.error('Incoming data issues: {n_missing} missing')

    def check_errors(self) -> None:
        '''Raise an exception if any errors occured'''
        if self.n_errors != 0:
            raise RetrieveError(self.inconsistent if IncomingErrorType.INCONSISTENT in self._keep_errors else [],
                                self.duplicate if IncomingErrorType.DUPLICATE in self._keep_errors else [],
                                self.unexpected if IncomingErrorType.UNEXPECTED in self._keep_errors else [],
                                self.missing,
                                self.move_report.errors)

    def clear(self) -> None:
        super().clear()
        self.move_report.clear()
        self.unexpected = []


class FailedAssociationError(Exception):
    '''We were unable to associate with a remote network node'''


def _query_worker(res_q: janus._SyncQueueProxy[Optional[Tuple[QueryResult, Set[str]]]],
                  rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
                  assoc: Association,
                  level: QueryLevel,
                  queries: Iterator[Dataset],
                  query_model: sop_class.SOPClass,
                  split_level: Optional[QueryLevel] = None) -> None:
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
        missing_attrs: Set[str] = set()
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


def _move_worker(rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
                 assoc: Association,
                 dest: DcmNode,
                 query_res: QueryResult,
                 query_model: sop_class.SOPClass) -> None:
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


def _send_worker(send_q: janus._SyncQueueProxy[Optional[Dataset]],
                 rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
                 assoc: Association) -> None:
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
    '''Set of event types we want to match'''

    ae_titles: Optional[FrozenSet[str]] = None
    '''Set of AE Titles we want to match'''

    def matches(self, event: evt.Event) -> bool:
        '''Test if the `event` matches the filter'''
        # TODO: The _event attr is made public as 'evt' in future pynetdicom
        if self.event_types is not None and event._event not in self.event_types:
            return False
        if self.ae_titles is not None:
            norm_ae = event.assoc.requestor.ae_title.decode('ascii').strip()
            if norm_ae not in self.ae_titles:
                return False
        return True

    def collides(self, other: EventFilter) -> bool:
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


if TYPE_CHECKING:
    _FutureType = asyncio.Future[None]
else:
    _FutureType = asyncio.Future


class FilteredListenerLockBase(_FutureType):
    '''Base class for creating event filter aware lock types with fifolock'''

    event_filter: EventFilter

    @classmethod
    def is_compatible(cls, holds: Dict[Type[FilteredListenerLockBase], int]) -> bool:
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


def make_sync_to_async_cb(async_cb: Callable[[evt.Event], Awaitable[int]],
                          loop: asyncio.AbstractEventLoop
                         ) -> Callable[[evt.Event], int]:
    '''Return sync callback that bridges to an async callback'''
    def sync_cb(event: evt.Event) -> int:
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
def _make_retrieve_cb(res_q: asyncio.Queue[Tuple[Dataset, Dataset]]
                     ) -> Callable[[evt.Event], Awaitable[int]]:
    async def retrieve_cb(event: evt.Event) -> int:
        await res_q.put((event.dataset, event.file_meta))
        return 0x0 # Success
    return retrieve_cb


def is_specified(query: Dataset, attr: str) -> bool:
    val = getattr(query, attr, '')
    if val != '' and '*' not in val:
        return True
    return False


SOPList = List[sop_class.SOPClass]


# TODO: Need to listen for association aborted events and handle them
class LocalEntity:
    '''Low level interface to DICOM networking functionality

    Params
    ------
    local
        The local DICOM network node properties

    transfer_syntaxes
        The transfer syntaxes to use for any data transfers

    max_threads
        Size of thread pool
    '''

    def __init__(self,
                 local: DcmNode,
                 transfer_syntaxes: Optional[SOPList] = None,
                 max_threads: int = 8):
        self._local = local
        if transfer_syntaxes is None:
            self._default_ts = DEFAULT_TRANSFER_SYNTAXES
        else:
            self._default_ts = transfer_syntaxes
        self._thread_pool = ThreadPoolExecutor(max_threads)
        self._listener_lock = FifoLock()
        self._lock_types: Dict[EventFilter, type] = {}
        self._event_handlers: Dict[EventFilter, Callable[[evt.Event], Awaitable[int]]] = {}
        self._listen_mgr: Optional[ThreadedAssociationServer] = None

    @property
    def local(self) -> DcmNode:
        '''DcmNode for our local entity'''
        return self._local

    async def echo(self, remote: DcmNode) -> bool:
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

    def _prep_query(self,
                    level: Optional[QueryLevel],
                    query: Optional[Dataset],
                    query_res : Optional[QueryResult]
                   ) -> Tuple[QueryLevel, Dataset]:
        '''Resolve/check `level` and `query` args for query methods'''
        # Build up our base query dataset
        if query is None:
            query = Dataset()
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

    async def query(self,
                    remote: DcmNode,
                    level: Optional[QueryLevel] = None,
                    query: Optional[Dataset] = None,
                    query_res: Optional[QueryResult] = None,
                    report: Optional[DicomOpReport] = None) -> QueryResult:
        '''Query the `remote` entity all at once

        See documentation for the `queries` method for details
        '''
        level, query = self._prep_query(level, query, query_res)
        res = QueryResult(level)
        async for sub_res in self.queries(remote,
                                          level,
                                          query,
                                          query_res,
                                          report):
            res |= sub_res
        return res

    async def queries(self,
                      remote: DcmNode,
                      level: Optional[QueryLevel] = None,
                      query: Optional[Dataset] = None,
                      query_res: Optional[QueryResult] = None,
                      report: Optional[DicomOpReport] = None
                     ) -> AsyncIterator[QueryResult]:
        '''Query the `remote` entity in an iterative manner

        Params
        ------
        remote
            The network node we are querying

        level
            The level of detail we want to query

            If not provided, we try to automatically determine the most
            appropriate level based on the `query`, then the `query_res`,
            and finally fall back to using `QueryLevel.STUDY`.

        query
            Specifies the DICOM attributes we want to query with/for

        query_res
            A query result that we want to refine

        report
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

        # Determine the query model
        query_model = self._choose_qr_model(remote, 'find', level)

        # If we are missing some required higher-level identifiers, we perform
        # a recursive pre-query to get those first
        if (level == QueryLevel.STUDY and query_model == 'PatientRoot' and
            not is_specified(query, 'PatientID')
           ):
            if query_res is None:
                log.debug("Performing recursive PATIENT level query against %s" %
                          (remote,))
                query_res = await self.query(remote,
                                             QueryLevel.PATIENT,
                                             query,
                                             query_res)
        if level == QueryLevel.SERIES and not is_specified(query, 'StudyInstanceUID'):
            if query_res is None or query_res.level < QueryLevel.STUDY:
                log.debug("Performing recursive STUDY level query against %s" %
                          (remote,))
                query_res = await self.query(remote,
                                             QueryLevel.STUDY,
                                             query,
                                             query_res)
        elif level == QueryLevel.IMAGE and not is_specified(query, 'SeriesInstanceUID'):
            if query_res is None or query_res.level < QueryLevel.SERIES:
                log.debug("Performing recursive SERIES level query against %s" %
                          (remote,))
                query_res = await self.query(remote,
                                             QueryLevel.SERIES,
                                             query,
                                             query_res)
        log.debug("Performing %s level query against %s" % (level.name, remote))

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

        # Pull out a list of the attributes we are querying on
        queried_attrs = set(e.keyword for e in query)

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
                        queried_attrs.add(uid_elems[lvl])
                    queries.append(q)
                    sub_uids.clear()
            log.debug("QueryResult expansion results in %d sub-queries" %
                      len(queries))

        # Build a queue for results from query thread
        loop = asyncio.get_running_loop()
        res_q: janus.Queue[Tuple[QueryResult, Set[str]]] = janus.Queue(loop=loop)
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue(loop=loop)

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
            query_fut = asyncio.ensure_future(loop.run_in_executor(self._thread_pool,
                                                                   partial(_query_worker,
                                                                           res_q.sync_q,
                                                                           rep_q.sync_q,
                                                                           assoc,
                                                                           level,
                                                                           queries,
                                                                           query_model,
                                                                          )
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
                    qr.prov.src = remote
                    qr.prov.queried_attrs = queried_attrs.copy()
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

    async def _fwd_event(self, event: evt.Event) -> int:
        for filt, handler in self._event_handlers.items():
            if filt.matches(event):
                log.debug(f"Calling async handler for {event} event")
                return await handler(event)
        log.warn(f"Can't find handler for the {event} event")
        return 0x0122

    def _setup_listen_mgr(self,
                          sync_cb: Callable[[evt.Event], Any],
                          presentation_contexts: SOPList,
                         ) -> None:
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

    async def _cleanup_listen_mgr(self) -> None:
        log.debug("Cleaning up threaded listener")
        assert self._listen_mgr is not None
        try:
            self._listen_mgr.shutdown()
        finally:
            self._listen_mgr = None

    def _get_lock_type(self, event_filter: EventFilter) -> type:
        lock_type = self._lock_types.get(event_filter)
        if lock_type is None:
            lock_type = type('FilteredListenerLock',
                             (FilteredListenerLockBase,),
                             {'event_filter' : event_filter})
            self._lock_types[event_filter] = lock_type
        return lock_type

    @asynccontextmanager
    async def listen(self,
                     handler: Callable[[evt.Event], Awaitable[int]],
                     event_filter: Optional[EventFilter] = None,
                     presentation_contexts: Optional[SOPList] = None
                    ) -> AsyncIterator[None]:
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

    async def move(self,
                   src: DcmNode,
                   dest: DcmNode,
                   query_res: QueryResult,
                   transfer_syntax: Optional[SOPList] = None,
                   report: DicomOpReport = None) -> None:
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
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue(loop=loop)
        # Setup the association
        query_model = self._choose_qr_model(src, 'move', query_res.level)
        if transfer_syntax is None:
            transfer_syntax = self._default_ts
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

    async def retrieve(self,
                       remote: DcmNode,
                       query_res: QueryResult,
                       report: RetrieveReport = None,
                       keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False
                      ) -> AsyncIterator[Dataset]:
        '''Generate data sets from `remote` based on `query_res`.

        Parameters
        ----------
        remote
            The remote DICOM network node we want to get data from

        query_res
            The data we want to retrieve

        report
            If provided this will be populated asynchronously

        keep_errors
            Generate data sets even if we detect issues with them

            By default inconsistent, unexpected, and duplicate data are skipped
        '''
        if report is None:
            extern_report = False
            report = RetrieveReport()
        else:
            extern_report = True
        report.requested = query_res
        # TODO: Stop ignoring type errors here once mypy fixes issue #3004
        report.keep_errors = keep_errors # type: ignore
        event_filter = EventFilter(event_types=frozenset((evt.EVT_C_STORE,)),
                                   ae_titles=frozenset((remote.ae_title,))
                                  )
        res_q: asyncio.Queue[Tuple[Dataset, Dataset]] = asyncio.Queue()
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
                if not success:
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

    async def download(self,
                       remote: DcmNode,
                       query_res: QueryResult,
                       dest_dir: Union[str, Path]) -> List[str]:
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

    # TODO: Update type annotation once we have report inheritance hierarchy
    async def _report_builder(self,
                              res_q: janus._AsyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
                              report: Any) -> None:
        while True:
            res = await res_q.get()
            if res is None:
                break
            status, data_set = res
            # TODO: OpReport needs to minimize the stored data set as needed
            report.add(status, data_set)

    @asynccontextmanager
    async def send(self,
                   remote: DcmNode,
                   transfer_syntax: Optional[SOPList] = None,
                   report: Optional[DicomOpReport] = None
                   ) -> AsyncIterator[janus._AsyncQueueProxy[Dataset]]:
        '''Produces a queue where you can put data sets to be sent to `remote`

        Parameters
        ----------
        remote
            The remote node we are sending the data to

        transfer_syntax
            The DICOM transfer syntax to use for the transfer

        report
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
        send_q: janus.Queue[Dataset] = janus.Queue(10, loop=loop)
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue(10, loop=loop)
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

    async def upload(self,
                     remote: DcmNode,
                     src_paths: List[str],
                     transfer_syntax: Optional[SOPList] = None) -> None:
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

    def _choose_qr_model(self,
                         remote: DcmNode,
                         op_type: str,
                         level: QueryLevel) -> sop_class.SOPClass:
        '''Pick an appropriate query model'''
        if level == QueryLevel.PATIENT:
            for query_model in ('PatientRoot', 'PatientStudyOnly'):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()
        elif level == QueryLevel.STUDY:
            for query_model in ('StudyRoot', 'PatientRoot', 'PatientStudyOnly'):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()
        else:
            for query_model in ('StudyRoot', 'PatientRoot'):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()
