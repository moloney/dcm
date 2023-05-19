import enum, logging
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pydicom.dataset import Dataset
from pynetdicom.status import code_to_category

from ..node import DcmNodeBase, DicomOpType
from ..query import (
    InconsistentDataError,
    InvalidDicomError,
    QueryLevel,
    QueryResult,
    get_all_uids,
    minimal_copy,
)
from . import CountableReport, MultiError, MultiListReport, ProgressHookBase


log = logging.getLogger(__name__)


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
    """Describes a DICOM network operation"""

    provider: Optional[DcmNodeBase] = None
    """The service provider"""

    user: Optional[DcmNodeBase] = None
    """The service user"""

    op_type: Optional[DicomOpType] = None
    """The type of operation performed"""

    op_data: Dict[str, Any] = field(default_factory=dict)
    """Additional data describing the operation specifics"""


SUB_OP_ATTRS = {
    stat_type.lower(): f"NumberOf{stat_type}Suboperations"
    for stat_type in ("Remaining", "Completed", "Warning", "Failed")
}


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
            remaining = getattr(status, SUB_OP_ATTRS["remaining"], None)
            if remaining is None:
                # We don't have sub-operation counts, so we just count
                # 'pending' results as success
                self._n_success += 1
            else:
                n_success = getattr(status, SUB_OP_ATTRS["completed"])
                n_warn = getattr(status, SUB_OP_ATTRS["warning"])
                n_error = getattr(status, SUB_OP_ATTRS["failed"])
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
                n_success = getattr(status, SUB_OP_ATTRS["completed"], None)
                if n_success is not None:
                    n_warn = getattr(status, SUB_OP_ATTRS["warning"])
                    n_error = getattr(status, SUB_OP_ATTRS["failed"])
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
            return self._add_invalid(data_set)
        return True

    def add_invalid(self, data_set: Dataset) -> bool:
        """Add an incoming dataset that is known to be invalid by the caller"""
        assert not self.done
        self.count_input()
        return self._add_invalid(data_set)

    def _add_invalid(self, data_set: Dataset) -> bool:
        try:
            all_uids = get_all_uids(data_set)
        except Exception:
            all_uids = ("NA",)
        self.invalid.append(all_uids)
        return IncomingErrorType.INVALID in self._keep_errors

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
        self.move_report.done = True
        if self.requested:
            self.missing = self.requested - self.retrieved
        else:
            self.missing = QueryResult(QueryLevel.IMAGE)
