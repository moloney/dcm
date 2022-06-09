"""High level async DICOM networking interface
"""
from __future__ import annotations
import asyncio, threading, time, logging, enum, inspect
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import asynccontextmanager
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import (
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
    cast,
)
from pathlib import Path
from queue import Empty
from textwrap import indent

import janus
from fifolock import FifoLock
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.datadict import keyword_for_tag

from pynetdicom import (
    AE,
    Association,
    evt,
    build_context,
    QueryRetrievePresentationContexts,
    StoragePresentationContexts,
    VerificationPresentationContexts,
    sop_class,
)
from pynetdicom._globals import ALL_TRANSFER_SYNTAXES
from pynetdicom.status import code_to_category
from pynetdicom.sop_class import StorageServiceClass, SOPClass

# from pynetdicom.pdu_primitives import SOPClassCommonExtendedNegotiation, SOPClassExtendedNegotiation
from pynetdicom.transport import ThreadedAssociationServer
from pynetdicom.presentation import PresentationContext


from . import __version__
from .query import (
    InvalidDicomError,
    QueryLevel,
    QueryResult,
    InconsistentDataError,
    uid_elems,
    req_elems,
    opt_elems,
    choose_level,
    minimal_copy,
    get_all_uids,
)
from .report import (
    BaseReport,
    CountableReport,
    MultiListReport,
    MultiError,
    ProgressHookBase,
)
from .util import (
    json_serializer,
    JsonSerializable,
    InlineConfigurable,
    create_thread_task,
)


log = logging.getLogger(__name__)


UID_PREFIX = "2.25"


IMPLEMENTATION_UID = "%s.84718903" % UID_PREFIX


# The DICOM standard only allows an association requestor to propose 128
# presentation contexts, which is already less than the number of unqiue
# Storage SOPClasses. We also need to add at least one custom SOPClass (for
# Siemens non-standard MR data). The "ideal" solution here would be to use
# extended negotiation to propose more than 128 presentation contexts, but
# many systems in the wild do not support this. So the "least bad" option for
# a default is to drop some of the less commonly use SOPClasses. Unfortunately
# I don't have objective data on which classes these are, so I have simply
# choosen ones that seen to be unlikely to come up in my domain.

dropped_storage_classes = set(
    [
        "LensometryMeasurementsStorage",
        "AutorefractionMeasurementsStorage",
        "KeratometryMeasurementsStorage",
        "SubjectiveRefractionMeasurementsStorage",
        "VisualAcuityMeasurementsStorage",
        "OphthalmicVisualFieldStaticPerimetryMeasurementsStorage",
        "SpectaclePrescriptionReportStorage",
        "OphthalmicAxialMeasurementsStorage",
        "IntraocularLensCalculationsStorage",
        "MacularGridThicknessAndVolumeReport",
        "OphthalmicVisualFieldStaticPerimetryMeasurementsStorage",
        "OphthalmicThicknessMapStorage",
        "CornealTopographyMapStorage",
    ]
)


private_sop_classes = {"SiemensProprietaryMRStorage": "1.3.12.2.1107.5.9.1"}

# This is needed so 'uid_to_service_class' will work when called for incoming
# data sets
sop_class._STORAGE_CLASSES.update(private_sop_classes)


def _make_default_store_scu_pcs(
    transfer_syntaxes: Optional[List[str]] = None,
) -> List[PresentationContext]:
    # TODO: Do we actally need the SOPClass objects for anything?
    include_sops: List[SOPClass] = []
    pres_contexts = []
    max_len = 128 - len(private_sop_classes)
    for sop_name, sop_uid in sop_class._STORAGE_CLASSES.items():
        if sop_name not in dropped_storage_classes:
            if len(include_sops) == max_len:
                log.warn(
                    "Too many storage SOPClasses, dropping more from end " "of the list"
                )
                break
            sop = SOPClass(sop_uid)
            sop._service_class = StorageServiceClass
            include_sops.append(sop)
            pres_contexts.append(build_context(sop_uid, transfer_syntaxes))
    for sop_name, sop_uid in private_sop_classes.items():
        sop = SOPClass(sop_uid)
        sop._service_class = StorageServiceClass
        include_sops.append(sop)
        pres_contexts.append(build_context(sop_uid, transfer_syntaxes))
    assert len(pres_contexts) <= 128
    return pres_contexts


def _make_default_store_scp_pcs(
    transfer_syntaxes: Optional[List[str]] = None,
) -> List[PresentationContext]:
    pres_contexts = deepcopy(StoragePresentationContexts)
    for sop_name, sop_uid in private_sop_classes.items():
        pres_contexts.append(build_context(sop_uid, transfer_syntaxes))
    return pres_contexts


default_store_scu_pcs = _make_default_store_scu_pcs()
default_store_scp_pcs = _make_default_store_scp_pcs()


# TODO: We should have an option to use extended negotiation using the below
#       code as a basis
# siemens_mr_sop_neg = SOPClassCommonExtendedNegotiation()
# siemens_mr_sop_neg.sop_class_uid = '1.3.12.2.1107.5.9.1'
# siemens_mr_sop_neg.service_class_uid = StorageServiceClass.uid
# siemens_mr_sop_neg.related_general_sop_class_identification = [MRImageStorage]
# c_store_ext_sop_negs = [siemens_mr_sop_neg]


# TODO: Everytime we establish/close an association, it should be done in a
#       separate thread, since this is a blocking operation (though usually
#       quite fast). Probably makes sense to wait until we refactor associations
#       into some sort of AssociationManager object that can also manage
#       caching of associations and limits on number of simultaneous
#       associations with any given remote.


QR_MODELS = {
    "PatientRoot": {
        "find": sop_class.PatientRootQueryRetrieveInformationModelFind,
        "move": sop_class.PatientRootQueryRetrieveInformationModelMove,
        "get": sop_class.PatientRootQueryRetrieveInformationModelGet,
    },
    "StudyRoot": {
        "find": sop_class.StudyRootQueryRetrieveInformationModelFind,
        "move": sop_class.StudyRootQueryRetrieveInformationModelMove,
        "get": sop_class.StudyRootQueryRetrieveInformationModelGet,
    },
    "PatientStudyOnly": {
        "find": sop_class.PatientStudyOnlyQueryRetrieveInformationModelFind,
        "move": sop_class.PatientStudyOnlyQueryRetrieveInformationModelMove,
        "get": sop_class.PatientStudyOnlyQueryRetrieveInformationModelGet,
    },
}


@json_serializer
@dataclass(frozen=True)
class DcmNode(JsonSerializable, InlineConfigurable["DcmNode"]):
    """DICOM network entity info"""

    host: str
    """Hostname of the node"""

    ae_title: str = "ANYAE"
    """DICOM AE Title of the node"""

    port: int = 11112
    """DICOM port for the node"""

    qr_models: Tuple[str, ...] = ("StudyRoot", "PatientRoot")
    """Supported DICOM QR models for the node"""

    def __str__(self) -> str:
        return "%s:%s:%s" % (self.host, self.ae_title, self.port)

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        """Parse inline string format 'host[:ae_title][:port]'

        Both the second components are optional
        """
        toks = in_str.split(":")
        if len(toks) > 3:
            raise ValueError("Too many tokens for node specification: %s" % in_str)
        res: Dict[str, Union[str, int]] = {"host": toks[0]}
        if len(toks) == 3:
            res["ae_title"] = toks[1]
            res["port"] = int(toks[2])
        elif len(toks) == 2:
            try:
                res["port"] = int(toks[1])
            except ValueError:
                res["ae_title"] = toks[1]
        return res


sub_op_attrs = {
    stat_type.lower(): f"NumberOf{stat_type}Suboperations"
    for stat_type in ("Remaining", "Completed", "Warning", "Failed")
}


class BatchDicomOperationError(Exception):
    """Base class for errors from DICOM batch network operations"""

    def __init__(
        self,
        op_errors: List[Tuple[Union[Dataset, Exception], Optional[Dataset]]],
        n_remote_errors: int,
    ):
        self.op_errors = op_errors
        self.n_remote_errors = n_remote_errors


@dataclass
class DicomOp:
    """Describes a DICOM network operatiom"""

    provider: Optional[DcmNode] = None
    """The service provider"""

    user: Optional[DcmNode] = None
    """The service user"""

    op_type: Optional[str] = None
    """The type of operation performed"""

    op_data: Dict[str, Any] = field(default_factory=dict)
    """Additional data describing the operation specifics"""


class DicomOpReport(CountableReport):
    """Track status results from DICOM operations"""

    def __init__(
        self,
        description: Optional[str] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
        dicom_op: Optional[DicomOp] = None,
    ):
        self.dicom_op = DicomOp() if dicom_op is None else dicom_op
        self.warnings: List[Tuple[Dataset, Optional[Dataset]]] = []
        self.errors: List[Tuple[Union[Dataset, Exception], Optional[Dataset]]] = []
        self._n_success = 0
        self._remote_warnings = 0
        self._remote_errors = 0
        self._has_sub_ops = False
        self._final_status: Optional[Dataset] = None
        super().__init__(
            description, {"dicom_op": self.dicom_op}, depth, prog_hook, n_expected
        )

    @property
    def n_success(self) -> int:
        return self._n_success

    @property
    def n_errors(self) -> int:
        return len(self.errors) + self._remote_errors

    @property
    def n_warnings(self) -> int:
        return len(self.warnings) + self._remote_warnings

    def add(
        self, status: Union[Dataset, Exception], data_set: Optional[Dataset]
    ) -> None:
        if isinstance(status, Exception) or not hasattr(status, "Status"):
            if self.dicom_op.op_type == "c-store":
                assert data_set is not None
                data_set = minimal_copy(data_set)
            self.errors.append((status, data_set))
            self.count_input()
            return
        status_category = code_to_category(status.Status)
        if status_category == "Pending":
            self._has_sub_ops = True
            remaining = getattr(status, sub_op_attrs["remaining"], None)
            if remaining is None:
                # We don't have sub-operation counts, so we just count
                # 'pending' results as success
                self._n_success += 1
            else:
                n_success = getattr(status, sub_op_attrs["completed"])
                n_warn = getattr(status, sub_op_attrs["warning"])
                n_error = getattr(status, sub_op_attrs["failed"])
                if self.n_expected is None:
                    self.n_expected = remaining + 1
                if n_success != self._n_success:
                    if not (self._n_success is None or self._n_success <= n_success):
                        log.warning("DicomOpReport success count mismatch")
                    self._n_success = n_success
                else:
                    if self.dicom_op.op_type == "c-store":
                        assert data_set is not None
                        data_set = minimal_copy(data_set)
                    if n_warn != self.n_warnings:
                        if not self.n_warnings < n_warn:
                            log.warning("DicomOpReport warning count mismatch")
                        self.warnings.append((status, data_set))
                    elif n_error != self.n_errors:
                        if not self.n_errors < n_error:
                            log.warning("DicomOpReport error count mismatch")
                        self.errors.append((status, data_set))
        else:
            if self._has_sub_ops or data_set is None:
                self._final_status = status
                n_success = getattr(status, sub_op_attrs["completed"], None)
                if n_success is not None:
                    n_warn = getattr(status, sub_op_attrs["warning"])
                    n_error = getattr(status, sub_op_attrs["failed"])
                    if self._n_success is None:
                        self._n_success = n_success
                    else:
                        if not self._n_success <= n_success:
                            log.warning("DicomOpReport success count mismatch")
                        self._n_success = n_success
                    # Some errors/warnings are only reported at the end which we need
                    # to catch here
                    if n_warn < len(self.warnings):
                        log.warning("DicomOpReport warning count mismatch")
                    elif n_warn > len(self.warnings):
                        self._remote_warnings = n_warn - len(self.warnings)
                    if n_error < len(self.errors):
                        log.warning("DicomOpReport error count mismatch")
                    elif n_error > len(self.errors):
                        self._remote_errors = n_error - len(self.errors)
                elif status_category != "Success":
                    # If we don't have operation sub-counts, need to make
                    # sure any final error/warning status doesn't get lost
                    if status_category == "Warning":
                        self.warnings.append((status, data_set))
                    elif status_category == "Failure":
                        self.errors.append((status, data_set))
                log.debug(
                    f"DicomOpReport (%s <%s> %s) got final status after sub-ops, marking self as done",
                    self.dicom_op.user,
                    self.dicom_op.op_type,
                    self.dicom_op.provider,
                )
                self.done = True
            else:
                if status_category == "Success":
                    self._n_success += 1
                else:
                    if self.dicom_op.op_type == "c-store":
                        data_set = minimal_copy(data_set)
                    if status_category == "Warning":
                        self.warnings.append((status, data_set))
                    elif status_category == "Failure":
                        self.errors.append((status, data_set))
        self.count_input()

    def log_issues(self) -> None:
        """Log a summary of error/warning statuses"""
        if self.n_errors != 0:
            if self.errors:
                log.error(
                    "Got %d error and %d warning statuses out of %d %s ops"
                    % (
                        len(self.errors),
                        len(self.warnings),
                        len(self),
                        self.dicom_op.op_type,
                    )
                )
            if self._remote_errors:
                log.error(
                    "Got %d remote errors, and %d remote warnings",
                    self._remote_errors,
                    self._remote_warnings,
                )
        elif self.n_warnings != 0:
            if self.warnings:
                log.warning(
                    "Got %d warning statuses out of %d %s ops"
                    % (len(self.warnings), len(self), self.dicom_op.op_type)
                )
            if self._remote_warnings:
                log.warning("Got %d remote warnings", self._remote_warnings)

    def check_errors(self) -> None:
        """Raise an exception if any errors occured"""
        if self.n_errors != 0:
            raise BatchDicomOperationError(self.errors, self._remote_errors)

    def clear(self) -> None:
        """Clear out all current operation results"""
        if self._n_expected is not None:
            self._n_expected -= self._n_input
        self._n_input = 0
        self._n_success = 0
        self._remote_warnings = 0
        self._remote_errors = 0
        self.warnings = []
        self.errors = []

    def _auto_descr(self) -> str:
        return f"dicom-{self.dicom_op.op_type}"


class IncomingErrorType(enum.Enum):
    INCONSISTENT = enum.auto()
    DUPLICATE = enum.auto()
    UNEXPECTED = enum.auto()
    INVALID = enum.auto()


class IncomingDataError(Exception):
    """Captures errors detected in incoming data stream"""

    def __init__(
        self,
        inconsistent: Optional[List[Tuple[str, ...]]],
        duplicate: Optional[List[Tuple[str, ...]]],
        invalid: Optional[List[Tuple[str, ...]]],
    ):
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.invalid = invalid

    def __str__(self) -> str:
        res = ["IncomingDataError:"]
        for err_type in ("inconsistent", "duplicate", "invalid"):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            n_errors = len(errors)
            if n_errors != 0:
                res.append("%d %s," % (n_errors, err_type))
        return " ".join(res)


class IncomingDataReport(CountableReport):
    """Generic incoming data report"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
    ):
        self.keep_errors = keep_errors  # type: ignore
        self.retrieved = QueryResult(level=QueryLevel.IMAGE)
        self.inconsistent: List[Tuple[str, ...]] = []
        self.duplicate: List[Tuple[str, ...]] = []
        self.invalid: List[Tuple[str, ...]] = []
        super().__init__(description, meta_data, depth, prog_hook, n_expected)

    @property
    def keep_errors(self) -> Tuple[IncomingErrorType, ...]:
        """Whether or not we are forwarding inconsistent/duplicate data"""
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
    def n_success(self) -> int:
        res = len(self.retrieved)
        return res

    @property
    def n_errors(self) -> int:
        res = 0
        for err_type in IncomingErrorType:
            err_attr = err_type.name.lower()
            if not hasattr(self, err_attr):
                continue
            if err_type not in self._keep_errors:
                res += len(getattr(self, err_attr))
        return res

    @property
    def n_warnings(self) -> int:
        res = 0
        for err_type in IncomingErrorType:
            err_attr = err_type.name.lower()
            if not hasattr(self, err_attr):
                continue
            if err_type in self._keep_errors:
                res += len(getattr(self, err_attr))
        return res

    def add(self, data_set: Dataset) -> bool:
        """Add an incoming data set, returns True if it should should be used"""
        assert not self.done
        self.count_input()
        try:
            dupe = data_set in self.retrieved
        except InconsistentDataError:
            self.inconsistent.append(get_all_uids(data_set))
            return IncomingErrorType.INCONSISTENT in self._keep_errors
        else:
            if dupe:
                self.duplicate.append(get_all_uids(data_set))
                return IncomingErrorType.DUPLICATE in self._keep_errors
        try:
            self.retrieved.add(minimal_copy(data_set))
        except InvalidDicomError:
            self.invalid.append(get_all_uids(data_set))
            return IncomingErrorType.INVALID in self._keep_errors
        return True

    def log_issues(self) -> None:
        """Log any warnings and errors"""
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
            log.warn("Incoming data issues: %s" % " ".join(warn_msg))
        if error_msg:
            log.error("Incoming data issues: %s" % " ".join(error_msg))

    def check_errors(self) -> None:
        if self.n_errors:
            kwargs = {}
            if IncomingErrorType.INCONSISTENT not in self.keep_errors:
                kwargs["inconsistent"] = self.inconsistent
            if IncomingErrorType.DUPLICATE not in self.keep_errors:
                kwargs["duplicate"] = self.duplicate
            if IncomingErrorType.INVALID not in self.keep_errors:
                kwargs["invalid"] = self.invalid
            raise IncomingDataError(**kwargs)

    def clear(self) -> None:
        self.inconsistent = []
        self.duplicate = []
        self.invalid = []

    def __str__(self) -> str:
        lines = [super().__str__()]
        res = 0
        for err_type in IncomingErrorType:
            err_attr = err_type.name.lower()
            if not hasattr(self, err_attr):
                continue
            errors = getattr(self, err_attr)
            if errors:
                lines.append(f"  * {err_type}:")
                for uid in errors:
                    lines.append(f"      {uid}")
        return "\n".join(lines)


class RetrieveError(IncomingDataError):
    """Capture errors that happened during a retrieve operation"""

    def __init__(
        self,
        inconsistent: Optional[List[Tuple[str, ...]]],
        duplicate: Optional[List[Tuple[str, ...]]],
        invalid: Optional[List[Tuple[str, ...]]],
        unexpected: Optional[List[Tuple[str, ...]]],
        missing: Optional[QueryResult],
        move_errors: Optional[MultiError],
    ):
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.invalid = invalid
        self.unexpected = unexpected
        self.missing = missing
        self.move_errors = move_errors

    def __str__(self) -> str:
        res = ["RetrieveError:"]
        for err_type in (
            "inconsistent",
            "unexpected",
            "invalid",
            "duplicate",
            "missing",
            "move_errors",
        ):
            errors = getattr(self, err_type)
            if errors is None:
                continue
            if isinstance(errors, MultiError):
                n_errors = len(errors.errors)
            else:
                n_errors = len(errors)
            if n_errors != 0:
                res.append("%d %s," % (n_errors, err_type))
        return " ".join(res)


class RetrieveReport(IncomingDataReport):
    """Track details about a retrieve operation"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
        requested: Optional[QueryResult] = None,
    ):
        self.requested = requested
        self.missing: Optional[QueryResult] = None
        self.unexpected: List[Tuple[str, ...]] = []
        self.move_report: MultiListReport[DicomOpReport] = MultiListReport()
        super().__init__(
            description, meta_data, depth, prog_hook, n_expected, keep_errors
        )

    @property
    def requested(self) -> Optional[QueryResult]:
        return self._requested

    @requested.setter
    def requested(self, val: Optional[QueryResult]) -> None:
        self._requested = val
        if self._requested is not None:
            n_expected = self._requested.n_instances()
            if n_expected is not None:
                self.n_expected = n_expected

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
        self.count_input()
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
        try:
            self.retrieved.add(minimal_copy(data_set))
        except InvalidDicomError:
            self.invalid.append(get_all_uids(data_set))
            return IncomingErrorType.INVALID in self._keep_errors
        return True

    def log_issues(self) -> None:
        """Log any warnings and errors"""
        super().log_issues()
        self.move_report.log_issues()
        if self._done:
            assert self.missing is not None
            n_missing = len(self.missing)
            if n_missing != 0:
                log.error(f"Incoming data issues: {n_missing} missing")

    def check_errors(self) -> None:
        """Raise an exception if any errors occured"""
        if self.n_errors != 0:
            move_err = None
            try:
                self.move_report.check_errors()
            except MultiError as e:
                move_err = e
            raise RetrieveError(
                self.inconsistent
                if IncomingErrorType.INCONSISTENT in self._keep_errors
                else [],
                self.duplicate
                if IncomingErrorType.DUPLICATE in self._keep_errors
                else [],
                self.invalid if IncomingErrorType.INVALID in self._keep_errors else [],
                self.unexpected
                if IncomingErrorType.UNEXPECTED in self._keep_errors
                else [],
                self.missing,
                move_err,
            )

    def clear(self) -> None:
        super().clear()
        self.move_report.clear()
        self.unexpected = []

    def __str__(self) -> str:
        lines = [super().__str__()]
        if self.unexpected:
            lines.append(f"  * unexpected:")
            for uid in self.unexpected:
                lines.append(f"      {uid}")
        if self.missing:
            lines.append(f"  * missing:")
            lines.append(indent(self.missing.to_tree(), "      "))
        return "\n".join(lines)

    def _set_done(self, val: bool) -> None:
        super()._set_done(val)
        if not self.move_report.done:
            assert len(self.move_report) == 1 and self.move_report[0].n_input == 0
            self.move_report[0].done = True
        assert self.move_report.done
        assert self.requested is not None and self.retrieved is not None
        self.missing = self.requested - self.retrieved


class FailedAssociationError(Exception):
    """We were unable to associate with a remote network node"""


def _query_worker(
    res_q: janus._SyncQueueProxy[Optional[Tuple[QueryResult, Set[str]]]],
    rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
    assoc: Association,
    level: QueryLevel,
    queries: Iterator[Dataset],
    query_model: sop_class.SOPClass,
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
    for query in queries:
        log.debug("Sending query:\n%s", query)
        res = QueryResult(level)
        missing_attrs: Set[str] = set()
        for status, rdat in assoc.send_c_find(query, query_model=query_model):
            rep_q.put((status, rdat))
            if rdat is None:
                break
            if resp_count + 1 % 20 == 0:
                if shutdown is not None and shutdown.is_set():
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
                log.error("Got inconsistent data in query reponse from %s", assoc.ae)
                is_consistent = False
            if dupe:
                log.warning("Got duplicate data in query response from %s", assoc.ae)
            elif is_consistent:
                try:
                    res.add(rdat)
                except InvalidDicomError:
                    log.error("Got invalid data in query reponse from %s", assoc.ae)
        if len(res) != 0:
            res_q.put((res, missing_attrs))
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
    rep_q: janus._SyncQueueProxy[Optional[Tuple[Dataset, Dataset]]],
    assoc: Association,
    dest: DcmNode,
    query_res: QueryResult,
    query_model: sop_class.SOPClass,
    shutdown: Optional[threading.Event] = None,
) -> None:
    """Worker function for perfoming move operations in another thread"""
    for d in query_res:
        move_req = _make_move_request(d)
        move_req.QueryRetrieveLevel = query_res.level.name
        responses = assoc.send_c_move(move_req, dest.ae_title, query_model=query_model)
        time.sleep(0.01)
        for r_idx, (status, rdat) in enumerate(responses):
            rep_q.put((status, rdat))
            if r_idx % 20 == 0:
                if shutdown is not None and shutdown.is_set():
                    log.debug("Move worker exiting from shutdown event")
                    return
    rep_q.put(None)
    log.debug("Move worker exiting normally")


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


class UnsupportedQueryModelError(Exception):
    """The requested query model isn't supported by the remote entity"""


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
            norm_ae = event.assoc.requestor.ae_title.decode("ascii").strip()
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


default_evt_pc_map = {
    evt.EVT_C_ECHO: VerificationPresentationContexts,
    evt.EVT_C_STORE: default_store_scp_pcs,
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


SOPList = List[sop_class.SOPClass]


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

    Params
    ------
    local
        The local DICOM network node properties

    transfer_syntaxes
        The transfer syntaxes to use for any data transfers

    max_threads
        Size of thread pool
    """

    def __init__(
        self,
        local: DcmNode,
        transfer_syntaxes: Optional[SOPList] = None,
        max_threads: int = 32,
    ):
        self._local = local
        if transfer_syntaxes is None:
            self._default_ts = ALL_TRANSFER_SYNTAXES[:]
        else:
            self._default_ts = transfer_syntaxes
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

    async def echo(self, remote: DcmNode) -> bool:
        """Perfrom an "echo" against `remote` to test connectivity

        Returns True if successful, else False. Throws FailedAssociationError if the
        initial association with the `remote` fails.
        """
        loop = asyncio.get_event_loop()
        async with self._associate(remote, VerificationPresentationContexts) as assoc:
            status = await loop.run_in_executor(self._thread_pool, assoc.send_c_echo)
        if status and status.Status == 0x0:
            return True
        return False

    async def query(
        self,
        remote: DcmNode,
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
        remote: DcmNode,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[MultiListReport[DicomOpReport]] = None,
    ) -> AsyncIterator[QueryResult]:
        """Query the `remote` entity in an iterative manner

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
        query_model = self._choose_qr_model(remote, "find", level)

        # If we are missing some required higher-level identifiers, we perform
        # a recursive pre-query to get those first
        if (
            level == QueryLevel.STUDY
            and query_model == "PatientRoot"
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

        # Setup args for building reports
        dicom_op = DicomOp(provider=remote, user=self._local, op_type="c-find")
        op_report_attrs = {
            "dicom_op": dicom_op,
            "prog_hook": report._prog_hook,
        }

        # Create association with the remote node
        log.debug("Making association with %s for query" % (remote,))
        async with self._associate(
            remote, QueryRetrievePresentationContexts, query_model
        ) as assoc:
            # Fire up a thread to perform the query and produce QueryResult chunks
            rep_builder_task = asyncio.create_task(
                self._multi_report_builder(rep_q.async_q, report, op_report_attrs)
            )
            query_fut = create_thread_task(
                _query_worker,
                (res_q.sync_q, rep_q.sync_q, assoc, level, queries, query_model),
                thread_pool=self._thread_pool,
            )
            qr_fut_done = False
            qr_fut_exception = None
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
            await query_fut
            await rep_builder_task
        if not extern_report:
            report.log_issues()
            report.check_errors()

    @asynccontextmanager
    async def listen(
        self,
        handler: Callable[[evt.Event], Awaitable[int]],
        event_filter: Optional[EventFilter] = None,
        presentation_contexts: Optional[SOPList] = None,
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
        source: DcmNode,
        dest: DcmNode,
        query_res: QueryResult,
        transfer_syntax: Optional[SOPList] = None,
        report: MultiListReport[DicomOpReport] = None,
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
        query_model = self._choose_qr_model(source, "move", query_res.level)
        if transfer_syntax is None:
            transfer_syntax = self._default_ts
        # Setup queue args for building reports
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue()
        dicom_op = DicomOp(provider=source, user=self._local, op_type="c-move")
        op_report_attrs = {
            "dicom_op": dicom_op,
            "prog_hook": report._prog_hook,
        }
        # Setup the association
        log.debug(f"About to associate with {source} to move data")
        async with self._associate(
            source, QueryRetrievePresentationContexts, query_model
        ) as assoc:
            rep_builder_task = asyncio.create_task(
                self._multi_report_builder(rep_q.async_q, report, op_report_attrs)
            )
            await create_thread_task(
                _move_worker,
                (rep_q.sync_q, assoc, dest, query_res, query_model),
                thread_pool=self._thread_pool,
            )
            await rep_builder_task
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def retrieve(
        self,
        remote: DcmNode,
        query_res: QueryResult,
        report: RetrieveReport = None,
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
                    # Add the data set to our report and handle errors
                    success = report.add(ds)
                    if not success:
                        log.debug("Retrieved data set filtered due to error")
                        continue
                    # Remove any Group 0x0002 elements that may have been included
                    ds = ds[0x00030000:]
                    # Add the file_meta to the data set and yield it
                    # TODO: Get implementation UID and set it here
                    # file_meta.ImplementationUID = ...
                    # file_meta.ImplementationVersionName = __version__
                    ds.file_meta = file_meta
                    ds.is_little_endian = file_meta.TransferSyntaxUID.is_little_endian
                    ds.is_implicit_VR = file_meta.TransferSyntaxUID.is_implicit_VR
                    yield ds
            except GeneratorExit:
                log.info("Retrieve generator closed early, cancelling move")
                await move_task.cancel()
            else:
                log.debug("Retrieve generator exhausted, about to await move task")
                await move_task
        report.done = True
        if not extern_report:
            report.log_issues()
            log.debug("About to check errors")
            report.check_errors()

    @asynccontextmanager
    async def send(
        self,
        remote: DcmNode,
        transfer_syntax: Optional[SOPList] = None,
        report: Optional[DicomOpReport] = None,
    ) -> AsyncIterator[janus._AsyncQueueProxy[Dataset]]:
        """Produces a queue where you can put data sets to be sent to `remote`

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
        """
        if transfer_syntax is None:
            transfer_syntax = self._default_ts
        if report is None:
            extern_report = False
            report = DicomOpReport()
        else:
            extern_report = True
        report.dicom_op.provider = remote
        report.dicom_op.user = self._local
        report.dicom_op.op_type = "c-store"
        send_q: janus.Queue[Optional[Dataset]] = janus.Queue(10)
        rep_q: janus.Queue[Optional[Tuple[Dataset, Dataset]]] = janus.Queue(10)

        log.debug(f"About to associate with {remote} to send data")
        async with self._associate(remote, default_store_scu_pcs) as assoc:
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
        remote: DcmNode,
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
        remote: DcmNode,
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
        remote: DcmNode,
        pres_contexts: List[PresentationContext],
        query_model: Optional[SOPClass] = None,
    ) -> AsyncIterator[Association]:
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
        except (GeneratorExit, asyncio.CancelledError):
            # If it appears we
            if query_model is not None and assoc.is_established:
                try:
                    await loop.run_in_executor(
                        self._thread_pool, assoc.send_c_cancel, 1, None, query_model
                    )
                except Exception as e:
                    log.info("Exception occured when seding c-cancel: %s", e)
            raise
        finally:
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
        presentation_contexts: SOPList,
    ) -> None:
        log.debug("Starting a threaded listener")
        # TODO: How to handle presentation contexts in generic way?
        ae = AE(ae_title=self._local.ae_title)
        ae.dimse_timeout = 180
        for context in presentation_contexts:
            ae.add_supported_context(context.abstract_syntax, self._default_ts)
        self._listen_mgr = ae.start_server(
            (self._local.host, self._local.port), block=False
        )
        for evt_type in default_evt_pc_map.keys():
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
            if res is None:
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

    def _choose_qr_model(
        self, remote: DcmNode, op_type: str, level: QueryLevel
    ) -> sop_class.SOPClass:
        """Pick an appropriate query model"""
        if level == QueryLevel.PATIENT:
            for query_model in ("PatientRoot", "PatientStudyOnly"):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()
        elif level == QueryLevel.STUDY:
            for query_model in ("StudyRoot", "PatientRoot", "PatientStudyOnly"):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()
        else:
            for query_model in ("StudyRoot", "PatientRoot"):
                if query_model in remote.qr_models:
                    return QR_MODELS[query_model][op_type]
            raise UnsupportedQueryModelError()

    def _add_qr_meta(self, report: BaseReport, query_res: QueryResult) -> None:
        report._meta_data["qr_level"] = query_res.level
        report._meta_data["patient_ids"] = [x for x in query_res.patients()]
        if query_res.level > QueryLevel.PATIENT:
            report._meta_data["study_uids"] = [x for x in query_res.studies()]
        if query_res.level > QueryLevel.STUDY and query_res.n_series() == 1:
            report._meta_data["series_uids"] = [x for x in query_res.series()]
