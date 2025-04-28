"""High level async DICOM networking interface
"""
from __future__ import annotations
import asyncio, threading, time, logging, inspect
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import asynccontextmanager
from functools import partial
from dataclasses import dataclass
from typing import (
    Iterable,
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    FrozenSet,
    Union,
    Any,
    AsyncIterator,
    Iterator,
    Callable,
    Awaitable,
    Type,
    TYPE_CHECKING,
)
from pathlib import Path
from queue import Empty

import janus
from fifolock import FifoLock
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.datadict import keyword_for_tag

from pynetdicom import (
    AE,
    Association,
    evt,
    QueryRetrievePresentationContexts,
    VerificationPresentationContexts,
    StoragePresentationContexts,
    build_context,
)
from pynetdicom.sop_class import SOPClass
from pynetdicom.transport import ThreadedAssociationServer
from pynetdicom.presentation import PresentationContext

from .node import (
    DEFAULT_PRIVATE_SOP_CLASSES,
    DcmNode,
    DcmNodeBase,
    DicomRole,
    QueryModel,
    RemoteNode,
    DicomOpType,
)
from .query import (
    InvalidDicomError,
    QueryLevel,
    QueryResult,
    InconsistentDataError,
    uid_elems,
    req_elems,
    opt_elems,
    choose_level,
)
from .reports import BaseReport, MultiListReport
from .reports.net_report import (
    DicomOp,
    DicomOpReport,
    IncomingErrorType,
    RetrieveReport,
)
from .util import create_thread_task


log = logging.getLogger(__name__)


UID_PREFIX = "2.25"


IMPLEMENTATION_UID = "%s.84718903" % UID_PREFIX


class FailedAssociationError(Exception):
    """We were unable to associate with a remote network node"""


def _query_worker(
    res_q: janus._SyncQueueProxy[Optional[Tuple[QueryResult, Set[str]]]],
    rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Optional[Dataset]]]],
    assoc: Association,
    level: QueryLevel,
    queries: Iterator[Dataset],
    qr_class: SOPClass,
    split_level: Optional[QueryLevel] = None,
    shutdown: Optional[threading.Event] = None,
) -> None:
    """Worker function for performing queries in a separate thread"""
    if split_level is None:
        if level == QueryLevel.PATIENT or level == QueryLevel.STUDY:
            split_level = level
        else:
            split_level = QueryLevel(level - 1)
    elif split_level > level:
        raise ValueError("The split_level can't be higher than the query level")
    split_attr = uid_elems[split_level]
    last_split_val = None

    resp_count = 0
    try:
        for q_idx, query in enumerate(queries):
            # Before sending another c-find, check if we were asked to shutdown
            if q_idx > 0 and shutdown is not None and shutdown.is_set():
                log.debug("Query worker got shutdown event")
                return
            log.debug("Sending query:\n%s", query)
            res = QueryResult(level)
            missing_attrs: Set[str] = set()
            for status, rdat in assoc.send_c_find(query, query_model=qr_class):
                rep_q.put((status, rdat))
                if rdat is None:
                    break
                # Check periodically if we should shutdown prematurely
                if resp_count + 1 % 20 == 0:
                    if shutdown is not None and shutdown.is_set():
                        log.debug("Query worker got shutdown event")
                        return
                resp_count += 1
                log.debug("Got query response:\n%s", rdat)
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
                is_consistent = True
                try:
                    dupe = rdat in res
                except InconsistentDataError:
                    log.error(
                        "Got inconsistent data in query reponse from %s", assoc.ae
                    )
                    is_consistent = False
                if dupe:
                    log.warning(
                        "Got duplicate data in query response from %s", assoc.ae
                    )
                elif is_consistent:
                    try:
                        res.add(rdat)
                    except InvalidDicomError:
                        log.error("Got invalid data in query reponse from %s", assoc.ae)
            if len(res) != 0:
                res_q.put((res, missing_attrs))
    finally:
        # Signal consumers on other end of queues that we are done
        res_q.put(None)
        rep_q.put(None)


def _make_move_request(ds: Dataset) -> Dataset:
    res = Dataset()
    for uid_attr in uid_elems.values():
        uid_val = getattr(ds, uid_attr, None)
        if uid_val is not None:
            setattr(res, uid_attr, uid_val)
    return res


def _move_worker(
    rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Optional[Dataset]]]],
    assoc: Association,
    dest: DcmNode,
    query_res: QueryResult,
    qr_class: SOPClass,
    shutdown: Optional[threading.Event] = None,
) -> None:
    """Worker function for perfoming move operations in another thread"""
    in_shutdown = False
    for d_idx, d in enumerate(query_res):
        if d_idx != 0 and shutdown is not None and shutdown.is_set():
            log.debug("Move worker exiting from shutdown event")
            break
        move_req = _make_move_request(d)
        move_req.QueryRetrieveLevel = query_res.level.name
        log.debug("Worker thread is calling send_c_move")
        responses = assoc.send_c_move(move_req, dest.ae_title, query_model=qr_class)
        time.sleep(0.05)
        log.debug("Worker is about to iterate responses")
        for r_idx, (status, rdat) in enumerate(responses):
            log.debug("Got c-move response")
            rep_q.put((status, rdat))
            log.debug("Queued c-move reponse for report")
            if r_idx % 20 == 0:
                if shutdown is not None and shutdown.is_set():
                    log.debug("Move worker exiting from shutdown event")
                    in_shutdown = True
                    break
        log.debug("All responses from c-move have been recieved")
        if in_shutdown:
            break
    else:
        log.debug("Move worker exiting normally")
    rep_q.put(None)


def _send_worker(
    send_q: janus._SyncQueueProxy[Optional[Dataset]],
    rep_q: janus._SyncQueueProxy[Optional[Tuple[Union[Exception, Dataset], Dataset]]],
    assoc: Association,
    shutdown: Optional[threading.Event] = None,
) -> None:
    """Worker function for performing send operations in another thread"""
    n_sent = 0
    while True:
        no_input = False
        try:
            # TODO: Stop ignoring types when this is fixed: https://github.com/aio-libs/janus/issues/267
            ds = send_q.get(timeout=0.2)  # type: ignore
        except Empty:
            no_input = True
        else:
            if ds is None:
                log.debug("Send worker got None, shutting down...")
                rep_q.put(None)
                break
        if no_input or n_sent % 20 == 0:
            if shutdown is not None and shutdown.is_set():
                break
        if no_input:
            continue
        log.debug("Send worker got a data set")
        assert ds is not None
        try:
            status = assoc.send_c_store(ds)
        except Exception as e:
            rep_q.put((e, ds))
        else:
            rep_q.put((status, ds))


@dataclass(frozen=True)
class EventFilter:
    """Define a filter that matches a subset of pynetdicom events"""

    event_types: Optional[FrozenSet[evt.InterventionEvent]] = None
    """Set of event types we want to match"""

    ae_titles: Optional[FrozenSet[str]] = None
    """Set of AE Titles we want to match"""

    def matches(self, event: evt.Event) -> bool:
        """Test if the `event` matches the filter"""
        # TODO: The _event attr is made public as 'evt' in future pynetdicom
        if self.event_types is not None and event._event not in self.event_types:
            return False
        if self.ae_titles is not None:
            norm_ae = event.assoc.requestor.ae_title.strip()
            if norm_ae not in self.ae_titles:
                return False
        return True

    def collides(self, other: EventFilter) -> bool:
        """Test if this filter collides with the `other` filter"""
        evt_collision = remote_collision = False
        if self.event_types is None or other.event_types is None:
            evt_collision = True
        elif len(self.event_types & other.event_types) != 0:
            evt_collision = True
        if self.ae_titles is None or other.ae_titles is None:
            remote_collision = True
        elif len(self.ae_titles & other.ae_titles) != 0:
            remote_collision = True
        return evt_collision and remote_collision


if TYPE_CHECKING:
    _FutureType = asyncio.Future[None]
else:
    _FutureType = asyncio.Future


class FilteredListenerLockBase(_FutureType):
    """Base class for creating event filter aware lock types with fifolock"""

    event_filter: EventFilter

    @classmethod
    def is_compatible(cls, holds: Dict[Type[FilteredListenerLockBase], int]) -> bool:
        for lock_type, n_holds in holds.items():
            if n_holds > 0 and cls.event_filter.collides(lock_type.event_filter):
                return False
        return True


def _make_default_store_scp_pcs(
    transfer_syntaxes: Optional[List[str]] = None,
) -> List[PresentationContext]:
    pres_contexts = deepcopy(StoragePresentationContexts)
    for sop_name, sop_uid in DEFAULT_PRIVATE_SOP_CLASSES:
        pres_contexts.append(build_context(sop_uid, transfer_syntaxes))
    return pres_contexts


DEFAULT_EVT_PC_MAP = {
    evt.EVT_C_ECHO: VerificationPresentationContexts,
    evt.EVT_C_STORE: _make_default_store_scp_pcs(),
    evt.EVT_C_FIND: QueryRetrievePresentationContexts,
}
"""Map event types to the default presentation contexts the AE needs to accept
"""


def make_sync_to_async_cb(
    async_cb: Callable[[evt.Event], Awaitable[int]], loop: asyncio.AbstractEventLoop
) -> Callable[[evt.Event], int]:
    """Return sync callback that bridges to an async callback"""

    def sync_cb(event: evt.Event) -> int:
        # Calling get_event_loop from sync code can return the wrong loop
        # sometimes, in which case the below code hangs without error.
        # This assertion should catch that issue.
        assert loop.is_running()
        f = asyncio.run_coroutine_threadsafe(async_cb(event), loop)
        res = f.result()
        return res

    return sync_cb


def make_queue_data_cb(
    res_q: asyncio.Queue[Tuple[Dataset, FileMetaDataset]]
) -> Callable[[evt.Event], Awaitable[int]]:
    """Return callback that queues dataset/metadata from incoming events"""

    async def callback(event: evt.Event) -> int:
        await res_q.put((event.dataset, event.file_meta))
        return 0x0  # Success

    return callback


def is_specified(query: Dataset, attr: str) -> bool:
    val = getattr(query, attr, "")
    if val != "" and "*" not in val:
        return True
    return False


SOPList = Iterable[SOPClass]


class _SingletonEntity(type):
    """Make sure we have a single LocalEntity for each ae_title/port combo"""

    _instances = {}  # type: ignore
    _init = {}  # type: ignore

    def __init__(cls, name, bases, dct):  # type: ignore
        super(_SingletonEntity, cls).__init__(name, bases, dct)
        cls._init[cls] = dct.get("__init__", None)

    def __call__(cls, *args, **kwargs):  # type: ignore
        init = cls._init[cls]
        local_node = inspect.getcallargs(init, None, *args, **kwargs)["local"]
        if local_node is None:
            raise ValueError("The 'local' arg can't be None")
        key = (cls, local_node.ae_title, local_node.port)
        if key not in cls._instances:
            cls._instances[key] = super(_SingletonEntity, cls).__call__(*args, **kwargs)
        return cls._instances[key]


# TODO: Need to set max threads based on number of sources, or somehow be smarter
#       about it as it is a potential source of deadlocks if too low
# TODO: Need to listen for association aborted events and handle them
class LocalEntity(metaclass=_SingletonEntity):
    """Low level interface to DICOM networking functionality

    Parameters
    ----------

    local
        The local DICOM network node properties

    max_threads
        Size of thread pool
    """

    def __init__(
        self,
        local: DcmNode,
        max_threads: int = 32,
    ):
        self._local = local
        self._ae = AE(ae_title=self._local.ae_title)
        # Certain operations can be slow to start up
        self._ae.dimse_timeout = 180

        self._thread_pool = ThreadPoolExecutor(max_threads)
        self._listener_lock = FifoLock()
        self._lock_types: Dict[EventFilter, type] = {}
        self._event_handlers: Dict[
            EventFilter, Callable[[evt.Event], Awaitable[int]]
        ] = {}
        self._listen_mgr: Optional[ThreadedAssociationServer] = None

    @property
    def local(self) -> DcmNode:
        """DcmNode for our local entity"""
        return self._local

    async def echo(self, remote: RemoteNode) -> bool:
        """Perfrom an "echo" against `remote` to test connectivity

        Returns True if successful, else False. Throws FailedAssociationError if the
        initial association with the `remote` fails.
        """
        loop = asyncio.get_event_loop()
        async with self._associate(remote, DicomOpType.ECHO) as assoc:
            status = await loop.run_in_executor(self._thread_pool, assoc.send_c_echo)
        if status and status.Status == 0x0:
            return True
        return False

    async def query(
        self,
        remote: RemoteNode,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[MultiListReport[DicomOpReport]] = None,
    ) -> QueryResult:
        """Query the `remote` entity all at once

        See documentation for the `queries` method for details
        """
        level, query = self._prep_query(level, query, query_res)
        res = QueryResult(level)
        async for sub_res in self.queries(remote, level, query, query_res, report):
            res |= sub_res
        return res

    async def queries(
        self,
        remote: RemoteNode,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[MultiListReport[DicomOpReport]] = None,
    ) -> AsyncIterator[QueryResult]:
        """Query the `remote` entity in an iterative manner

        Parameters
        ----------
        remote
            The network node we are querying

        level
            The level of detail we want to query

            If not provided, we try to automatically determine the most appropriate
            level based on the `query`, then the `query_res`, and finally fall back to
            using `QueryLevel.STUDY`.

        query
            Specifies the DICOM attributes we want to query with/for

        query_res
            A query result that we want to refine

        report
            If provided, will store status report from DICOM operations
        """
        if report is None:
            extern_report = False
            report = MultiListReport(meta_data={"remote": remote, "level": level})
        else:
            extern_report = True
        if report._description is None:
            report.description = "queries"
        report._meta_data["remote"] = remote

        level, query = self._prep_query(level, query, query_res)

        report._meta_data["level"] = level
        if query is not None:
            report._meta_data["query"] = query
        if query_res is not None:
            self._add_qr_meta(report, query_res)

        # Determine the query model
        query_model = remote.get_query_model(level)
        qr_class = remote.get_abstract_syntaxes(
            DicomOpType.FIND, query_model=query_model
        )[0]

        # If we are missing some required higher-level identifiers, we perform
        # a recursive pre-query to get those first
        if (
            level == QueryLevel.STUDY
            and query_model == QueryModel.PATIENT_ROOT
            and not is_specified(query, "PatientID")
        ):
            if query_res is None:
                log.debug(
                    "Performing recursive PATIENT level query against %s" % (remote,)
                )
                query_res = await self.query(
                    remote, QueryLevel.PATIENT, query, query_res
                )
        if level == QueryLevel.SERIES and not is_specified(query, "StudyInstanceUID"):
            if query_res is None or query_res.level < QueryLevel.STUDY:
                log.debug(
                    "Performing recursive STUDY level query against %s" % (remote,)
                )
                query_res = await self.query(remote, QueryLevel.STUDY, query, query_res)
        elif level == QueryLevel.IMAGE and not is_specified(query, "SeriesInstanceUID"):
            if query_res is None or query_res.level < QueryLevel.SERIES:
                log.debug(
                    "Performing recursive SERIES level query against %s" % (remote,)
                )
                query_res = await self.query(
                    remote, QueryLevel.SERIES, query, query_res
                )
        log.debug("Performing %s level query against %s" % (level.name, remote))

        # Add in any required and default optional elements to query
        for lvl in QueryLevel:
            for req_attr in req_elems[lvl]:
                if getattr(query, req_attr, None) is None:
                    setattr(query, req_attr, "")
            if lvl == level:
                break
        auto_attrs = set()
        for opt_attr in opt_elems[level]:
            if getattr(query, opt_attr, None) is None:
                setattr(query, opt_attr, "")
                auto_attrs.add(opt_attr)

        # Set the QueryRetrieveLevel
        query.QueryRetrieveLevel = level.name

        # Pull out a list of the attributes we are querying on
        queried_elems = set(e.keyword for e in query)

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
                        queried_elems.add(uid_elems[lvl])
                    queries.append(q)
                    sub_uids.clear()
            log.debug("QueryResult expansion results in %d sub-queries" % len(queries))
        if len(queries) > 1:
            report.n_expected = len(queries)

        # Build a queue for results from query thread
        res_q: janus.Queue[Tuple[QueryResult, Set[str]]] = janus.Queue()
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue()

        # Make shutdown event for query worker
        query_shutdown = threading.Event()

        # Setup args for building reports
        dicom_op = DicomOp(provider=remote, user=self._local, op_type=DicomOpType.FIND)
        op_report_attrs = {
            "dicom_op": dicom_op,
            "prog_hook": report._prog_hook,
        }

        # Create association with the remote node
        log.debug("Making association with %s for query" % (remote,))
        # While we will signal the worker thread to shutdown before sending any C-CANCEL
        # there is no gaurantee the worker will see it, so we send the C-CANCEL and
        # close the association before waiting for the thread to finish
        query_fut = None
        rep_builder_task = None
        loop = asyncio.get_running_loop()
        try:
            async with self._associate(remote, DicomOpType.FIND, query_model) as assoc:
                # Fire up a thread to perform the query and produce QueryResult chunks
                rep_builder_task = asyncio.create_task(
                    self._multi_report_builder(rep_q.async_q, report, op_report_attrs)
                )
                query_fut = create_thread_task(
                    _query_worker,
                    (res_q.sync_q, rep_q.sync_q, assoc, level, queries, qr_class),
                    loop=loop,
                    thread_pool=self._thread_pool,
                    shutdown=query_shutdown,
                )
                qr_fut_done = False
                qr_fut_exception = None
                try:
                    while True:
                        try:
                            res = await asyncio.wait_for(
                                res_q.async_q.get(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            if qr_fut_exception:
                                raise qr_fut_exception
                            assert qr_fut_done == False
                            # Check if the worker thread is done
                            if query_fut.done():
                                qr_fut_done = True
                                qr_fut_exception = query_fut.exception()
                        else:
                            if res is None:
                                break
                            qr, missing_attrs = res
                            for missing_attr in missing_attrs:
                                if missing_attr not in auto_attrs:
                                    # TODO: Still getting some false positives with this
                                    #       warning, presumably from pre-queries. Disable
                                    #       it until it is more reliable.
                                    pass
                                    # warnings.warn(f"Remote node {remote} doesn't "
                                    #              f"support querying on {missing_attr}")
                            qr.prov.source = remote
                            qr.prov.queried_elems = queried_elems.copy()
                            yield qr
                finally:
                    log.debug("Setting shutdown event for query worker")
                    await loop.run_in_executor(self._thread_pool, query_shutdown.set)

        finally:
            if query_fut is not None:
                log.debug("Waiting for query worker thread to finish")
                await query_fut
                log.debug("The query worker thread has closed")
            log.debug("Waiting for query report builder task to finish")
            if rep_builder_task is not None:
                try:
                    await rep_builder_task
                except BaseException as e:
                    log.exception("Exception from report builder task")
        if not extern_report:
            report.log_issues()
            report.check_errors()

    @asynccontextmanager
    async def listen(
        self,
        handler: Callable[[evt.Event], Awaitable[int]],
        event_filter: Optional[EventFilter] = None,
        presentation_contexts: Optional[List[PresentationContext]] = None,
    ) -> AsyncIterator[None]:
        """Listen for incoming DICOM network events

        The async callback `handler` will be called for each DICOM network
        event that matches the `event_filter`. The default filter matches all
        events.

        Locks are used internally to prevent two listeners with colliding event
        filters from running simultaneously. Avoiding deadlocks due to
        dependency loops must be handled at a higher level.
        """
        if event_filter is None:
            event_filter = EventFilter()
        if presentation_contexts is None:
            log.debug("Building list of default presentation contexts")
            presentation_contexts = []
            if event_filter.event_types is None:
                for p_contexts in DEFAULT_EVT_PC_MAP.values():
                    presentation_contexts += p_contexts
            else:
                for event_type in event_filter.event_types:
                    presentation_contexts += DEFAULT_EVT_PC_MAP[event_type]
        LockType = self._get_lock_type(event_filter)
        log.debug("About to wait on listener lock")
        async with self._listener_lock(LockType):
            log.debug("Listener lock acquired")
            # If we don't already have a threaded listener running, set it up
            if self._listen_mgr is None:
                sync_cb = make_sync_to_async_cb(
                    self._fwd_event, asyncio.get_running_loop()
                )
                self._setup_listen_mgr(sync_cb, presentation_contexts)
            # Some sanity checks that our locking scheme is working
            assert event_filter not in self._event_handlers
            assert all(not event_filter.collides(f) for f in self._event_handlers)
            self._event_handlers[event_filter] = handler
            try:
                yield
            finally:
                del self._event_handlers[event_filter]
                # If no one else is listening, or waiting to listen, clean up
                # the threaded listener
                if (
                    len(self._listener_lock._waiters) == 0
                    and sum(self._listener_lock._holds.values()) == 1
                ):
                    await self._cleanup_listen_mgr()
        log.debug("Listener lock released")

    async def move(
        self,
        source: RemoteNode,
        dest: DcmNodeBase,
        query_res: QueryResult,
        transfer_syntaxes: Optional[SOPList] = None,
        report: Optional[MultiListReport[DicomOpReport]] = None,
    ) -> None:
        """Move DICOM files from one network entity to another"""
        if report is None:
            extern_report = False
            report = MultiListReport()
        else:
            extern_report = True
        if report._description is None:
            report.description = "move"
        report._meta_data["source"] = source
        report._meta_data["dest"] = dest
        self._add_qr_meta(report, query_res)
        query_model = source.get_query_model(query_res.level)
        qr_class = source.get_abstract_syntaxes(
            DicomOpType.MOVE, query_model=query_model
        )[0]

        # Setup queue args for building reports
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue()
        dicom_op = DicomOp(provider=source, user=self._local, op_type=DicomOpType.MOVE)
        op_report_attrs = {
            "dicom_op": dicom_op,
            "prog_hook": report._prog_hook,
        }

        # Make shutdown event for worker
        move_shutdown = threading.Event()

        # Setup the association
        log.debug(f"About to associate with {source} to move data")
        worker_task = None
        rep_builder_task = None
        loop = asyncio.get_running_loop()
        try:
            async with self._associate(
                source, DicomOpType.MOVE, query_model, transfer_syntaxes
            ) as assoc:
                rep_builder_task = asyncio.create_task(
                    self._multi_report_builder(rep_q.async_q, report, op_report_attrs)
                )
                try:
                    worker_task = create_thread_task(
                        _move_worker,
                        (rep_q.sync_q, assoc, dest, query_res, qr_class),
                        thread_pool=self._thread_pool,
                        loop=loop,
                        shutdown=move_shutdown,
                    )
                    while not worker_task.done():
                        await asyncio.sleep(0.1)
                finally:
                    log.debug("Setting shutdown event for move worker")
                    await loop.run_in_executor(self._thread_pool, move_shutdown.set)
        except asyncio.CancelledError:
            # No matter what we set the shutdown event for the worker thread above,
            # so nothing to do here but prevent any CancelledError from propogating
            pass
        finally:
            log.debug("Waiting for move worker task thread to finish")
            if worker_task is not None:
                await worker_task
            log.debug("Move worker task has finished")
            if rep_builder_task is not None:
                await rep_builder_task
            log.debug("Report builder task is done")
        if not extern_report:
            report.log_issues()
            report.check_errors()
        log.debug("Call to LocalEntity.move has completed")

    async def retrieve(
        self,
        remote: RemoteNode,
        query_res: QueryResult,
        report: Optional[RetrieveReport] = None,
        keep_errors: Optional[Union[bool, Tuple[IncomingErrorType, ...]]] = None,
    ) -> AsyncIterator[Dataset]:
        """Generate data sets from `remote` based on `query_res`.

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
        """
        if report is None:
            extern_report = False
            report = RetrieveReport()
        else:
            extern_report = True
        report.requested = query_res
        report._meta_data["remote"] = remote
        self._add_qr_meta(report, query_res)

        # TODO: Stop ignoring type errors here once mypy fixes issue #3004
        if keep_errors is not None:
            report.keep_errors = keep_errors  # type: ignore
        event_filter = EventFilter(
            event_types=frozenset((evt.EVT_C_STORE,)),
            ae_titles=frozenset((remote.ae_title,)),
        )
        res_q: asyncio.Queue[Tuple[Dataset, FileMetaDataset]] = asyncio.Queue(10)
        retrieve_cb = make_queue_data_cb(res_q)
        log.debug("The 'retrieve' method is about to start listening")
        async with self.listen(retrieve_cb, event_filter):
            log.debug("The 'retrieve' method is about to fire up a move task")
            move_task = asyncio.create_task(
                self.move(remote, self._local, query_res, report=report.move_report)
            )
            try:
                while True:
                    try:
                        ds, file_meta = await asyncio.wait_for(res_q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Check if the move task is done
                        if move_task.done():
                            break
                        else:
                            continue
                    # Remove any Group 0x0002 elements that may have been included and
                    # also parse the dataset and make sure it is valid
                    try:
                        ds = ds[0x00030000:]
                    except Exception as e:
                        log.warning("Error parsing incoming data set: %s", str(e))
                        success = report.add_invalid(ds)
                        continue
                    else:
                        # Add the data set to our report and handle other errors
                        success = report.add(ds)
                    if not success:
                        log.debug("Retrieved data set filtered due to error")
                        continue

                    # Add the file_meta to the data set and yield it
                    # TODO: Get implementation UID and set it here
                    # file_meta.ImplementationUID = ...
                    # file_meta.ImplementationVersionName = VERSION
                    ds.file_meta = file_meta
                    ds.is_little_endian = file_meta.TransferSyntaxUID.is_little_endian
                    ds.is_implicit_VR = file_meta.TransferSyntaxUID.is_implicit_VR
                    yield ds
            except GeneratorExit:
                log.info("Retrieve generator closed early, cancelling move")
                move_task.cancel()
            finally:
                log.debug("Waiting for move task to finish")
                await move_task
        report.done = True
        if not extern_report:
            report.log_issues()
            log.debug("About to check errors")
            report.check_errors()
        log.debug("The LocalEntity.retrieve method has completed")

    @asynccontextmanager
    async def send(
        self,
        remote: RemoteNode,
        transfer_syntaxes: Optional[SOPList] = None,
        report: Optional[DicomOpReport] = None,
    ) -> AsyncIterator[janus._AsyncQueueProxy[Dataset]]:
        """Produces a queue where you can put data sets to be sent to `remote`

        Parameters
        ----------
        remote
            The remote node we are sending the data to

        transfer_syntaxes
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
        """
        if report is None:
            extern_report = False
            report = DicomOpReport(
                dicom_op=DicomOp(remote, self._local, DicomOpType.STORE)
            )
        else:
            extern_report = True
        report.dicom_op.provider = remote
        report.dicom_op.user = self._local
        report.dicom_op.op_type = DicomOpType.STORE
        send_q: janus.Queue[Optional[Dataset]] = janus.Queue(10)
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue(10)

        log.debug(f"About to associate with {remote} to send data")
        async with self._associate(
            remote, DicomOpType.STORE, transfer_syntaxes=transfer_syntaxes
        ) as assoc:
            try:
                rep_builder_task = asyncio.create_task(
                    self._single_report_builder(rep_q.async_q, report)
                )
                send_fut = create_thread_task(
                    _send_worker,
                    (send_q.sync_q, rep_q.sync_q, assoc),
                    thread_pool=self._thread_pool,
                )
                # Want it to be an error if external user sends None, so we have type
                # mis-match here
                yield send_q.async_q  # type: ignore
            finally:
                try:
                    # Signal send worker to shutdown, then wait for it
                    log.debug("Shutting down send worker")
                    await send_q.async_q.put(None)
                    await send_fut
                    log.debug("Send worker has shutdown, waiting for report builder")
                    await rep_builder_task
                finally:
                    report.done = True
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def download(
        self,
        remote: RemoteNode,
        query_res: QueryResult,
        dest_dir: Union[str, Path],
        report: Optional[RetrieveReport] = None,
    ) -> List[str]:
        """Uses `retrieve` method to download data and saves it to `dest_dir`"""
        # TODO: Handle collisions/overwrites
        loop = asyncio.get_running_loop()
        dest_dir = Path(dest_dir)
        out_files = []
        async for ds in self.retrieve(remote, query_res, report=report):
            out_path = str(dest_dir / ds.SOPInstanceUID) + ".dcm"
            await loop.run_in_executor(
                self._thread_pool,
                partial(pydicom.dcmwrite, out_path, ds, write_like_original=False),
            )
            out_files.append(out_path)
        return out_files

    async def upload(
        self,
        remote: RemoteNode,
        src_paths: List[str],
        transfer_syntax: Optional[SOPList] = None,
    ) -> None:
        """Uses `send` method to upload data from local `src_paths`"""
        loop = asyncio.get_running_loop()
        async with self.send(remote, transfer_syntax) as send_q:
            for src_path in src_paths:
                log.debug(f"Uploading file: {src_path}")
                ds = await loop.run_in_executor(
                    self._thread_pool, partial(pydicom.dcmread, str(src_path))
                )
                await send_q.put(ds)

    @asynccontextmanager
    async def _associate(
        self,
        remote: RemoteNode,
        op_type: DicomOpType,
        query_model: Optional[QueryModel] = None,
        transfer_syntaxes: Optional[SOPList] = None,
    ) -> AsyncIterator[Association]:
        # Create presentation contexts
        abs_syntaxes = remote.get_abstract_syntaxes(
            op_type, DicomRole.USER, query_model
        )
        pres_contexts = remote.get_presentation_contexts(
            abs_syntaxes, transfer_syntaxes
        )

        loop = asyncio.get_event_loop()
        assoc = await loop.run_in_executor(
            self._thread_pool,
            self._ae.associate,
            remote.host,
            remote.port,
            pres_contexts,
            remote.ae_title,
        )
        if not assoc.is_established:
            raise FailedAssociationError(
                "Failed to associate with remote: %s", str(remote)
            )
        log.debug("Successfully associated with remote: %s", remote)
        try:
            yield assoc
        except (KeyboardInterrupt, GeneratorExit, asyncio.CancelledError):
            if query_model is not None and assoc.is_established:
                assert len(abs_syntaxes) == 1
                log.debug("Sending c-cancel to remote: %s", remote)
                try:
                    await loop.run_in_executor(
                        self._thread_pool, assoc.send_c_cancel, 1, None, abs_syntaxes[0]
                    )
                except Exception as e:
                    log.info("Exception occured when sending c-cancel: %s", e)
            raise
        finally:
            log.debug("Releasing association")
            await loop.run_in_executor(self._thread_pool, assoc.release)

    def _prep_query(
        self,
        level: Optional[QueryLevel],
        query: Optional[Dataset],
        query_res: Optional[QueryResult],
    ) -> Tuple[QueryLevel, Dataset]:
        """Resolve/check `level` and `query` args for query methods"""
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

    async def _fwd_event(self, event: evt.Event) -> int:
        for filt, handler in self._event_handlers.items():
            if filt.matches(event):
                log.debug(f"Calling async handler for {event} event")
                return await handler(event)
        log.warn(f"Can't find handler for the {event} event")
        return 0x0122

    def _setup_listen_mgr(
        self,
        sync_cb: Callable[[evt.Event], Any],
        presentation_contexts: List[PresentationContext],
    ) -> None:
        log.debug("Starting a threaded listener")
        # TODO: How to handle presentation contexts in generic way?
        ae = AE(ae_title=self._local.ae_title)
        ae.dimse_timeout = 180
        for context in presentation_contexts:
            assert context.abstract_syntax is not None
            ae.add_supported_context(context.abstract_syntax, context.transfer_syntax)
        self._listen_mgr = ae.start_server(
            (self._local.host, self._local.port), block=False
        )
        assert self._listen_mgr is not None
        for evt_type in DEFAULT_EVT_PC_MAP.keys():
            log.debug(f"Binding to event {evt_type}")
            self._listen_mgr.bind(evt_type, sync_cb)

    async def _cleanup_listen_mgr(self) -> None:
        log.debug("Cleaning up threaded listener")
        assert self._listen_mgr is not None
        try:
            self._listen_mgr.server_close()
            self._listen_mgr.shutdown()
            log.debug("Threaded listener shutdown successfully")
        finally:
            self._listen_mgr = None
            log.debug("Threaded listener has been cleaned up")

    def _get_lock_type(self, event_filter: EventFilter) -> type:
        lock_type = self._lock_types.get(event_filter)
        if lock_type is None:
            lock_type = type(
                "FilteredListenerLock",
                (FilteredListenerLockBase,),
                {"event_filter": event_filter},
            )
            self._lock_types[event_filter] = lock_type
        return lock_type

    async def _single_report_builder(
        self,
        res_q: janus._AsyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
        report: DicomOpReport,
    ) -> None:
        while True:
            res = await res_q.get()
            if res is None:
                # if not report.done:
                #    assert report.n_input == 0
                #    report.done = True
                break
            status, data_set = res
            report.add(status, data_set)

    async def _multi_report_builder(
        self,
        res_q: janus._AsyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
        report: MultiListReport[DicomOpReport],
        def_report_attrs: Dict[str, Any],
    ) -> None:
        curr_op_report = DicomOpReport(**def_report_attrs)
        report.append(curr_op_report)
        while True:
            res = await res_q.get()
            log.debug("_multi_report_builder got an input")
            if res is None:
                log.debug("_multi_report_builder got None and is shutting down")
                if not curr_op_report.done:
                    if len(curr_op_report) != 0:
                        pass
                        # import pdb ; pdb.set_trace()
                    curr_op_report.done = True
                report.done = True
                break
            if curr_op_report.done:
                curr_op_report = DicomOpReport(**def_report_attrs)
                report.append(curr_op_report)
            status, data_set = res
            curr_op_report.add(status, data_set)

    def _add_qr_meta(self, report: BaseReport, query_res: QueryResult) -> None:
        report._meta_data["qr_level"] = query_res.level
        report._meta_data["patient_ids"] = [x for x in query_res.patients()]
        if query_res.level > QueryLevel.PATIENT:
            report._meta_data["study_uids"] = [x for x in query_res.studies()]
        if query_res.level > QueryLevel.STUDY and query_res.n_series() == 1:
            report._meta_data["series_uids"] = [x for x in query_res.series()]
