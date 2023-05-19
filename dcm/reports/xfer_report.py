"""Reports used for routes / routers"""
from __future__ import annotations
import logging
from typing import Optional, Dict, Tuple, Union, List, Any, cast

from pydicom import Dataset

from . import (
    CountableReport,
    MultiListReport,
    MultiDictReport,
    MultiKeyedError,
    MultiAttrReport,
    ProgressHookBase,
)
from .net_report import (
    IncomingErrorType,
    IncomingDataReport,
    RetrieveReport,
    DicomOpReport,
)
from ..util import DuplicateDataError
from ..query import (
    QueryLevel,
    QueryResult,
    InconsistentDataError,
)
from ..filt import DataTransform, get_transform
from ..route import ProxyTransferError, StaticRoute
from ..store.base import (
    DataBucket,
    LocalWriteReport,
    LocalIncomingReport,
    TransferMethod,
)


log = logging.getLogger(__name__)


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
class ProxyReport(CountableReport):
    """Abstract base class for reports on proxy transfers"""

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
        self.sent: Dict[StaticRoute, DataTransform] = {}
        self.inconsistent: Dict[StaticRoute, List[Tuple[Dataset, Dataset]]] = {}
        self.duplicate: Dict[StaticRoute, List[Tuple[Dataset, Dataset]]] = {}
        self._n_success = 0
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
        return self._n_success

    @property
    def n_errors(self) -> int:
        n_errors = 0
        if not self.keep_errors:
            n_errors += self.n_inconsistent + self.n_duplicate
        if self.done:
            n_errors += self.n_outstanding
        return n_errors

    @property
    def n_warnings(self) -> int:
        n_warn = 0
        if self.keep_errors:
            n_warn += self.n_inconsistent + self.n_duplicate
        return n_warn

    @property
    def n_sent(self) -> int:
        """Number of times datasets were sent out"""
        res = sum(len(trans.new) * len(sr.dests) for sr, trans in self.sent.items())
        if self.keep_errors:
            res += sum(len(x) * len(sr.dests) for sr, x in self.inconsistent.items())
            res += sum(len(x) * len(sr.dests) for sr, x in self.duplicate.items())
        return res

    @property
    def n_inconsistent(self) -> int:
        return sum(len(x) for _, x in self.inconsistent.items())

    @property
    def n_duplicate(self) -> int:
        return sum(len(x) for _, x in self.duplicate.items())

    @property
    def n_reported(self) -> int:
        """Number store results that have been reported so far"""
        raise NotImplementedError

    @property
    def n_outstanding(self) -> int:
        return self.n_sent - self.n_reported

    @property
    def all_reported(self) -> bool:
        """True if all sent data sets have a reported result"""
        n_out = self.n_outstanding
        assert n_out >= 0
        return n_out == 0

    def add(self, route: StaticRoute, old_ds: Dataset, new_ds: Dataset) -> bool:
        """Add the route with pre/post filtering dataset to the report"""
        self.count_input()
        if route not in self.sent:
            self.sent[route] = get_transform(QueryResult(QueryLevel.IMAGE), route.filt)
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
        else:
            self._n_success += 1
        return True

    def log_issues(self) -> None:
        """Produce log messages for any warning/error statuses"""
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
        """Raise an exception if any errors have occured so far"""
        if self.n_errors:
            inconsist = None
            if self.inconsistent:
                inconsist = self.inconsistent
            dupes = None
            if self.duplicate:
                dupes = self.duplicate
            raise ProxyTransferError(inconsistent=inconsist, duplicate=dupes)

    def clear(self) -> None:
        self.inconsistent.clear()
        self.duplicate.clear()
        if self.all_reported:
            self.sent.clear()


StoreReportType = Union[DicomOpReport, LocalWriteReport]


class DynamicTransferReport(ProxyReport):
    """Track what data is being routed where and any store results"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
    ):
        self.store_reports: MultiDictReport[
            DataBucket[Any, Any], MultiListReport[StoreReportType]
        ] = MultiDictReport(prog_hook=prog_hook)
        super().__init__(
            description, meta_data, depth, prog_hook, n_expected, keep_errors
        )

    @property
    def n_success(self) -> int:
        return super().n_success + self.store_reports.n_success

    @property
    def n_errors(self) -> int:
        return super().n_errors + self.store_reports.n_errors

    @property
    def n_warnings(self) -> int:
        return super().n_warnings + self.store_reports.n_warnings

    @property
    def n_reported(self) -> int:
        return self.store_reports.n_sub_input

    def add_store_report(
        self, dest: DataBucket[Any, Any], store_report: StoreReportType
    ) -> None:
        """Add a DicomOpReport to keep track of"""
        if dest not in self.store_reports:
            self.store_reports[dest] = MultiListReport(prog_hook=self._prog_hook)
        self.store_reports[dest].append(store_report)

    def log_issues(self) -> None:
        """Produce log messages for any warning/error statuses"""
        super().log_issues()
        self.store_reports.log_issues()

    def check_errors(self) -> None:
        """Raise an exception if any errors have occured so far"""
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
        """Clear current info about data sets we have results for"""
        # TODO: If n_sent != n_reported here we will go out of sync. I guess
        #       this would need to be managed at a higher level if it is
        #       needed. Not clear if it makes sense to do anything about it
        #       here.
        super().clear()
        if self.all_reported:
            self.store_reports.clear()


class StaticStoreReport(MultiDictReport[DataBucket[Any, Any], StoreReportType]):
    """Transfer report that only captures storage"""


class StaticProxyTransferReport(ProxyReport):
    """Static proxy transfer report"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        n_expected: Optional[int] = None,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        keep_errors: Union[bool, Tuple[IncomingErrorType, ...]] = False,
    ):
        self.store_reports: StaticStoreReport = StaticStoreReport(prog_hook=prog_hook)
        super().__init__(
            description, meta_data, depth, prog_hook, n_expected, keep_errors
        )

    @property
    def n_success(self) -> int:
        return super().n_success + self.store_reports.n_success

    @property
    def n_errors(self) -> int:
        return super().n_errors + self.store_reports.n_errors

    @property
    def n_warnings(self) -> int:
        return super().n_warnings + self.store_reports.n_warnings

    @property
    def n_reported(self) -> int:
        return self.store_reports.n_sub_input

    def add_store_report(
        self, dest: DataBucket[Any, Any], store_report: StoreReportType
    ) -> None:
        """Add a DicomOpReport or LocalWriteReport to keep track of"""
        assert dest not in self.store_reports
        if self.n_expected is not None and store_report.n_expected is None:
            store_report.n_expected = self.n_expected
        self.store_reports[dest] = store_report

    def log_issues(self) -> None:
        """Produce log messages for any warning/error statuses"""
        super().log_issues()
        self.store_reports.log_issues()

    def check_errors(self) -> None:
        """Raise an exception if any errors have occured so far"""
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
        super().clear()
        if self.all_reported:
            self.store_reports.clear()

    def _set_depth(self, val: int) -> None:
        if val != self._depth:
            self._depth = val
            self.store_reports.depth = val + 1


IncomingReportType = Union[IncomingDataReport, RetrieveReport, LocalIncomingReport]


class StaticOobTransferReport(MultiDictReport[TransferMethod, StaticStoreReport]):
    """Transfer report for out-of-band transfers"""


class StaticTransferError(Exception):
    def __init__(
        self,
        proxy_error: Optional[ProxyTransferError] = None,
        oob_error: Optional[MultiKeyedError] = None,
    ):
        self.proxy_error = proxy_error
        self.oob_error = oob_error

    def __str__(self) -> str:
        res = ["StaticTransferError:"]
        if self.proxy_error is not None:
            res.append("\tProxy Error: %s" % str(self.proxy_error))
        if self.oob_error is not None:
            res.append("\tOut-of-band Error: %s" % str(self.oob_error))
        return "\n".join(res)


class StaticTransferReport(MultiAttrReport):
    """Capture all possible info about a single StaticTranfer"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        incoming_report: Optional[IncomingReportType] = None,
    ):
        self._incoming_report = None
        self._proxy_report: Optional[StaticProxyTransferReport] = None
        self._oob_report: Optional[StaticOobTransferReport] = None
        self._report_attrs = ["incoming_report", "_proxy_report", "_oob_report"]
        super().__init__(description, meta_data, depth, prog_hook)
        if incoming_report is not None:
            self.incoming_report = incoming_report

    @property
    def incoming_report(self) -> Optional[IncomingReportType]:
        return self._incoming_report

    @incoming_report.setter
    def incoming_report(self, val: IncomingReportType) -> None:
        if self._incoming_report is not None:
            raise ValueError("The incoming report was already set")
        self._incoming_report = val
        self._incoming_report.depth = self._depth + 1
        self._incoming_report.prog_hook = self._prog_hook

    @property
    def proxy_report(self) -> StaticProxyTransferReport:
        if self._proxy_report is None:
            self._proxy_report = StaticProxyTransferReport(
                depth=self._depth + 1, prog_hook=self._prog_hook
            )
        if (
            self._proxy_report.n_expected is None
            and self._incoming_report is not None
            and self._incoming_report.n_expected is not None
        ):
            self._proxy_report.n_expected = self._incoming_report.n_expected
        return self._proxy_report

    @property
    def oob_report(self) -> StaticOobTransferReport:
        if self._oob_report is None:
            self._oob_report = StaticOobTransferReport(
                depth=self._depth + 1, prog_hook=self._prog_hook
            )
        if (
            self._oob_report.n_expected is None
            and self._incoming_report is not None
            and self._incoming_report.n_expected is not None
        ):
            self._oob_report.n_expected = self._incoming_report.n_expected
        return self._oob_report

    def check_errors(self) -> None:
        """Raise an exception if any errors have occured so far"""
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
