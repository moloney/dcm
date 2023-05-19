import logging
from typing import Any, Dict, Optional, Union

from . import (
    MultiAttrReport,
    MultiListReport,
    MultiDictReport,
    ProgressHookBase,
)
from .net_report import DicomOpReport
from ..route import Route
from .xfer_report import DynamicTransferReport, StaticTransferReport
from ..store.base import DataBucket, DataRepo


log = logging.getLogger(__name__)


TransferReportTypes = Union[
    DynamicTransferReport,
    StaticTransferReport,
]
DestType = Union[DataBucket, Route]
SourceMissingQueryReportType = MultiListReport[MultiListReport[DicomOpReport]]
DestMissingQueryReportType = MultiDictReport[
    DataRepo[Any, Any, Any, Any], SourceMissingQueryReportType
]


class SyncQueriesReport(MultiAttrReport):
    """Report for queries being performed during sync"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
    ):
        self._init_src_qr_report: Optional[MultiListReport[DicomOpReport]] = None
        self._missing_src_qr_reports: Optional[
            MultiListReport[SourceMissingQueryReportType]
        ] = None
        self._missing_dest_qr_reports: Optional[
            MultiListReport[DestMissingQueryReportType]
        ] = None
        self._report_attrs = [
            "_init_src_qr_report",
            "_missing_src_qr_reports",
            "_missing_dest_qr_reports",
        ]
        super().__init__(description, meta_data, depth, prog_hook)

    @property
    def init_src_qr_report(self) -> MultiListReport[DicomOpReport]:
        if self._init_src_qr_report is None:
            self._init_src_qr_report = MultiListReport(
                "init-src-qr", depth=self._depth + 1, prog_hook=self._prog_hook
            )
        return self._init_src_qr_report

    @property
    def missing_src_qr_reports(self) -> MultiListReport[SourceMissingQueryReportType]:
        if self._missing_src_qr_reports is None:
            self._missing_src_qr_reports = MultiListReport(
                "missing-src-qrs", depth=self._depth + 1, prog_hook=self._prog_hook
            )
        return self._missing_src_qr_reports

    @property
    def missing_dest_qr_reports(self) -> MultiListReport[DestMissingQueryReportType]:
        if self._missing_dest_qr_reports is None:
            self._missing_dest_qr_reports = MultiListReport(
                "missing-dest-qrs", depth=self._depth + 1, prog_hook=self._prog_hook
            )
        return self._missing_dest_qr_reports


class SyncReport(MultiAttrReport):
    """Top level report from a sync operation"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
    ):
        self._queries_report: Optional[SyncQueriesReport] = None
        self.trans_reports: MultiListReport[TransferReportTypes] = MultiListReport(
            "transfers", depth=depth + 1, prog_hook=prog_hook
        )
        self._report_attrs = ["_queries_report", "trans_reports"]
        super().__init__(description, meta_data, depth, prog_hook)

    @property
    def queries_report(self) -> SyncQueriesReport:
        if self._queries_report is None:
            self._queries_report = SyncQueriesReport(
                "sync-queries", depth=self._depth + 1, prog_hook=self._prog_hook
            )
        return self._queries_report
