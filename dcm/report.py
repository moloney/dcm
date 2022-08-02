"""Base classes for all reporting infrastructure

Most high-level operations are in fact batch operations which large numbers of
sub-operations. In is not uncommon for these sub-operations to produces errors or
warning statuses that shouldn't interrupt the whole batch operation. We use a
variety of "report" classes to capture this kind of information and provide
real-time insight into an ongoing async operation.
"""
from collections import deque
from contextlib import contextmanager
import logging, inspect
from dataclasses import dataclass, field
from datetime import datetime
import typing
from typing import (
    Iterable,
    Optional,
    Dict,
    List,
    Any,
    Union,
    TypeVar,
    Generic,
    Iterator,
    ItemsView,
    KeysView,
    ValuesView,
    Callable,
)
from typing_extensions import get_args

import rich.progress

from .util import Args_Type, decorate_sync_async


log = logging.getLogger(__name__)


@dataclass
class ProgressTaskBase:
    description: str

    total: Optional[int]

    start_time: datetime

    min_seconds: float

    show_indeterminate: bool

    _visible: bool = field(default=False, init=False, repr=False)


T = TypeVar("T", bound=ProgressTaskBase)


class ProgressHookBase(Generic[T]):
    """Base class for hooking in to progress updates from a report"""

    def_min_seconds: float = 3.0

    def_show_indeterminate: bool = False

    def create_task(
        self, description: str, total: Optional[int] = None, **kwargs: Any
    ) -> T:
        raise NotImplementedError

    def set_total(self, task: T, total: int) -> None:
        raise NotImplementedError

    def advance(self, task: T, amount: float = 1.0) -> None:
        raise NotImplementedError

    def end(self, task: T) -> None:
        raise NotImplementedError


@dataclass
class RichProgressTask(ProgressTaskBase):

    _task: Optional[rich.progress.TaskID] = field(default=None, init=False, repr=False)


class RichProgressHook(ProgressHookBase[RichProgressTask]):
    """Hook for console progress bar provided by `rich` package"""

    def __init__(self, progress: rich.progress.Progress):
        self._progress = progress

    def create_task(
        self, description: str, total: Optional[int] = None, **kwargs: Any
    ) -> RichProgressTask:
        if "min_seconds" not in kwargs:
            kwargs["min_seconds"] = self.def_min_seconds
        if "show_indeterminate" not in kwargs:
            kwargs["show_indeterminate"] = self.def_show_indeterminate
        res = RichProgressTask(description, total, datetime.now(), **kwargs)
        # self._update_task(res)
        return res

    def set_total(self, task: RichProgressTask, total: int) -> None:
        if total == task.total:
            return
        task.total = total
        self._update_task(task, total_dirty=True)

    def advance(self, task: RichProgressTask, amount: float = 1.0) -> None:
        self._update_task(task, advance=amount)

    def end(self, task: RichProgressTask) -> None:
        if task._task is not None:
            self._progress.update(task._task, visible=False)
            self._progress.stop_task(task._task)
            task._task = None

    def _update_task(
        self,
        task: RichProgressTask,
        advance: Optional[float] = None,
        total_dirty: bool = False,
    ) -> None:
        task_opts: Dict[str, Any] = {}
        if task.total is None:
            if not task.show_indeterminate:
                return
            task_opts["start"] = False
        elif total_dirty:
            task_opts["total"] = task.total
        if task._visible == False:
            if (
                task.min_seconds <= 0.0
                or (datetime.now() - task.start_time).total_seconds() > task.min_seconds
            ):
                task._visible = True
                task_opts["visible"] = True
        if not task._visible and advance is None:
            return
        if task._task is None:
            task_opts["total"] = task.total
            task_opts["visible"] = task._visible
            task._task = self._progress.add_task(task.description, **task_opts)
            if advance is not None:
                self._progress.advance(task._task, advance)
        else:
            self._progress.update(task._task, advance=advance, **task_opts)


class BaseReport:
    """Abstract base class for all reports"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
    ):
        self._description = description
        self._meta_data = {} if meta_data is None else meta_data
        self._depth = -1
        self._set_depth(depth)
        self._prog_hook: Optional[ProgressHookBase[Any]] = None
        self._set_prog_hook(prog_hook)
        self._start_time = datetime.now()
        self._end_time: Optional[datetime] = None
        self._done = False

    @property
    def description(self) -> str:
        if self._description is None:
            self._description = self._auto_descr()
        return self._description

    @description.setter
    def description(self, val: str) -> None:
        self._description = val

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, val: int) -> None:
        self._set_depth(val)

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, val: bool) -> None:
        self._set_done(val)

    @property
    def prog_hook(self) -> Optional[ProgressHookBase[Any]]:
        return self._prog_hook

    @prog_hook.setter
    def prog_hook(self, val: Optional[ProgressHookBase[Any]]) -> None:
        self._set_prog_hook(val)

    @property
    def has_errors(self) -> bool:
        """True if any errors were reported"""
        raise NotImplementedError

    @property
    def has_warnings(self) -> bool:
        """True if any warnings were reported"""
        raise NotImplementedError

    @property
    def all_success(self) -> bool:
        return not (self.has_errors or self.has_warnings)

    def log_issues(self) -> None:
        """Log a summary of error/warning statuses"""
        raise NotImplementedError

    def check_errors(self) -> None:
        """Raise an exception if any errors occured"""
        raise NotImplementedError

    # TODO: This is tricky to support for some subclasses, separate it out?
    def clear(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        lines = [f"{self.description}:"]
        for k, v in self._meta_data.items():
            if not isinstance(v, list):
                lines.append(f"  * {k}: {v}")
            else:
                lines.append(f"  * {k}:")
                for sub_v in v:
                    lines.append(f"      {sub_v}")
        done_stat = "COMPLETED" if self._done else "PENDING"
        lines.append(f"  * status: {done_stat}")
        lines.append(f"  * start time: {self._start_time}")
        if self._end_time is not None:
            lines.append(f"  * end time: {self._end_time}")
            lines.append(f"  * duration: {self._end_time - self._start_time}")
        has_err = self.has_errors
        has_warn = self.has_warnings
        fail_details = ""
        if has_err:
            if has_warn:
                fail_details = "errors and warnings detected"
            else:
                fail_details = "errors detected"
        elif has_warn:
            fail_details = "warnings detected"
        if fail_details:
            lines.append(f"  * success: False ({fail_details})")
        else:
            lines.append(f"  * success: True")
        return "\n".join(lines)

    def _auto_descr(self) -> str:
        res = self.__class__.__name__.lower()
        if res.endswith("report") and len(res) > len("report"):
            res = res[: -len("report")]
        return res

    def _set_done(self, val: bool) -> None:
        if not val:
            raise ValueError("Setting `done` to False is not allowed")
        if self._done:
            raise ValueError("Report was already marked done")
        self._done = True
        self._end_time = datetime.now()

    def _set_depth(self, val: int) -> None:
        if val < 0:
            raise ValueError("Negative depth not allowed")
        self._depth = val

    def _set_prog_hook(self, val: Optional[ProgressHookBase[Any]]) -> None:
        if self._prog_hook is not None:
            assert val == self._prog_hook
            return
        if val is None:
            return
        self._prog_hook = val


def optional_report(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for functions that optionally take a 'report' argument

    If the report is not supplied, one will be created automatically and after the
    function is called the report's `log_issues` and `check_errors` methods will be
    called. If the user supplies the report themselves, it is up to them to call
    these methods if they want to.
    """
    sig = inspect.signature(func)
    report_type = typing.get_type_hints(func)["report"]
    if not isinstance(report_type, type) or not issubclass(report_type, BaseReport):
        type_stack = deque([report_type])
        report_type = None
        while type_stack:
            curr_type = type_stack.popleft()
            for sub_type in get_args(curr_type):
                if isinstance(sub_type, type):
                    if issubclass(sub_type, BaseReport):
                        report_type = sub_type
                        break
                else:
                    type_stack.append(sub_type)
        if report_type is None:
            raise ValueError(f"The 'report' arg isn't the correct type: {report_type}")

    @contextmanager
    def check_report(
        args: Iterable[Any], kwargs: Dict[str, Any]
    ) -> Iterator[Args_Type]:
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        report = bound_args.arguments["report"]
        if report is None:
            extern_report = False
            report = report_type()
            bound_args.arguments["report"] = report
        else:
            extern_report = True
        yield (bound_args.args, bound_args.kwargs)
        if not extern_report:
            report.log_issues()
            report.check_errors()

    return decorate_sync_async(check_report, func)


class MultiError(Exception):
    def __init__(self, errors: List[Exception]):
        self.errors = errors

    def __str__(self) -> str:
        res = ["Multiple Errors:"] + [str(e) for e in self.errors]
        return "\n\t".join(res)


class MultiReport(BaseReport):
    """Abstract base class for reports that contain other sub-reports"""

    def gen_reports(self) -> Iterator[BaseReport]:
        """Base classes must provide this"""
        raise NotImplementedError

    @property
    def has_errors(self) -> bool:
        """True if any errors were reported"""
        return any(r.has_errors for r in self.gen_reports())

    @property
    def has_warnings(self) -> bool:
        """True if any warnings were reported"""
        return any(r.has_errors for r in self.gen_reports())

    def log_issues(self) -> None:
        for r in self.gen_reports():
            r.log_issues()

    def check_errors(self) -> None:
        errors = []
        for sub_report in self.gen_reports():
            try:
                sub_report.check_errors()
            except Exception as e:
                errors.append(e)
        if errors:
            raise MultiError(errors)

    def __str__(self) -> str:
        lines = [super().__str__()]
        lines.append("\n  Sub-Reports:")
        for rep in self.gen_reports():
            rep_str = str(rep).replace("\n", "\n    ")
            lines.append(f"    * {rep_str}")
        return "\n".join(lines)

    def _set_done(self, val: bool) -> None:
        super()._set_done(val)
        for sub_report in self.gen_reports():
            if not sub_report.done:
                log.warning(
                    "Sub-report '%s' not marked done before parent '%s'",
                    sub_report.description,
                    self.description,
                )
            # TODO: Raise here?

    def _set_depth(self, val: int) -> None:
        if val != self._depth:
            super()._set_depth(val)
            for r in self.gen_reports():
                r.depth = val + 1

    def _set_prog_hook(self, val: Optional[ProgressHookBase[Any]]) -> None:
        super()._set_prog_hook(val)
        for r in self.gen_reports():
            r.prog_hook = val


class MultiAttrReport(MultiReport):
    """Abstract base class combining a few sub-reports as attrs"""

    _report_attrs: List[str]
    """Subclasses need to provide this"""

    def gen_reports(self) -> Iterator[BaseReport]:
        for report_attr in self._report_attrs:
            val = getattr(self, report_attr)
            if val is not None:
                yield val


class CountableReport(BaseReport):
    '''Abstract base class for reports with single "count"'''

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
    ):
        self._n_expected = n_expected
        self._n_input = 0
        self._task = None
        super().__init__(description, meta_data, depth, prog_hook)

    @property
    def n_expected(self) -> Optional[int]:
        """Number of expected inputs, or None if unknown"""
        return self._n_expected

    @n_expected.setter
    def n_expected(self, val: int) -> None:
        self._n_expected = val
        if self._prog_hook is not None:
            if self._task is None:
                self._init_task()
            self._prog_hook.set_total(self._task, val)

    @property
    def n_input(self) -> int:
        """Number of inputs seen so far"""
        return self._n_input

    def count_input(self) -> None:
        self._n_input += 1
        if self._prog_hook is not None:
            if self._task is None:
                self._init_task()
            self._prog_hook.advance(self._task)

    @property
    def n_success(self) -> int:
        """Number of successfully handled inputs"""
        raise NotImplementedError

    @property
    def n_errors(self) -> int:
        """Number of errors, where a single input can cause multiple errors"""
        raise NotImplementedError

    @property
    def n_warnings(self) -> int:
        """Number of warnings, where a single input can cause multiple warnings"""
        raise NotImplementedError

    @property
    def has_errors(self) -> bool:
        """True if any errors were reported"""
        return self.n_errors != 0

    @property
    def has_warnings(self) -> bool:
        """True if any warnings were reported"""
        return self.n_warnings != 0

    def __len__(self) -> int:
        return self.n_success + self.n_warnings + self.n_errors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._description}, n_expected={self._n_expected}, n_input={self._n_input})"

    def __str__(self) -> str:
        lines = [super().__str__()]
        lines.append(f"  * n_success: {self.n_success}")
        if self.n_warnings > 0:
            lines.append(f"  * n_warnings: {self.n_warnings}")
        if self.n_errors > 0:
            lines.append(f"  * n_errors: {self.n_errors}")
        return "\n".join(lines)

    def _set_done(self, val: bool) -> None:
        super()._set_done(val)
        if self._prog_hook is not None and self._task is not None:
            self._prog_hook.end(self._task)

    def _init_task(self) -> None:
        assert self._prog_hook is not None
        self._task = self._prog_hook.create_task(
            self.description, total=self._n_expected
        )

    def _set_prog_hook(self, val: Optional[ProgressHookBase[Any]]) -> None:
        super()._set_prog_hook(val)
        if val is not None and self._description is not None:
            self._init_task()


R = TypeVar("R", bound=Union[BaseReport, "SummaryReport[Any]"])


class SummaryReport(MultiReport, CountableReport, Generic[R]):
    """Abstract base class for all SummaryReports"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
    ):
        CountableReport.__init__(
            self, description, meta_data, depth, prog_hook, n_expected
        )

    def gen_reports(self) -> Iterator[R]:
        raise NotImplementedError

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.gen_reports() if r.all_success)

    @property
    def n_warnings(self) -> int:
        return sum(1 for r in self.gen_reports() if r.has_warnings)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.gen_reports() if r.has_errors)

    @property
    def n_sub_success(self) -> int:
        total = 0
        for r in self.gen_reports():
            if hasattr(r, "n_sub_success"):
                total += r.n_sub_success
            elif hasattr(r, "n_success"):
                total += r.n_success
            elif r.all_success:
                total += 1
        return total

    @property
    def n_sub_warnings(self) -> int:
        total = 0
        for r in self.gen_reports():
            if hasattr(r, "n_sub_warnings"):
                total += r.n_sub_warnings
            elif hasattr(r, "n_warnings"):
                total += r.n_warnings
            elif r.has_warnings:
                total += 1
        return total

    @property
    def n_sub_errors(self) -> int:
        total = 0
        for r in self.gen_reports():
            if hasattr(r, "n_sub_errors"):
                total += r.n_sub_errors
            elif hasattr(r, "n_errors"):
                total += r.n_errors
            elif r.has_errors:
                total += 1
        return total

    def __str__(self) -> str:
        lines = [BaseReport.__str__(self)]
        lines.append(f"  * n_success: {self.n_success} ({self.n_sub_success} sub-ops)")
        if self.n_warnings > 0:
            lines.append(
                f"  * n_warnings: {self.n_warnings} ({self.n_sub_warnings} sub-ops)"
            )
        if self.n_errors > 0:
            lines.append(f"  * n_errors: {self.n_errors} ({self.n_sub_errors} sub-ops)")
        if not self.all_success:
            lines.append("\n  Sub-Reports:")
            for rep in self.gen_reports():
                if rep.all_success:
                    continue
                rep_str = str(rep).replace("\n", "\n    ")
                lines.append(f"    * {rep_str}")
        return "\n".join(lines)

    def _prep_sub_report(self, sub_report: R) -> None:
        sub_report._depth = self._depth + 1
        sub_report.prog_hook = self._prog_hook


class MultiListReport(SummaryReport[R]):
    """Sequence of reports of the same type"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
    ):
        self._sub_reports: List[R] = []
        super().__init__(description, meta_data, depth, prog_hook, n_expected)

    def __getitem__(self, idx: int) -> R:
        return self._sub_reports[idx]

    def __len__(self) -> int:
        return len(self._sub_reports)

    def __iter__(self) -> Iterator[R]:
        for report in self._sub_reports:
            yield report

    def append(self, sub_report: R) -> None:
        self._prep_sub_report(sub_report)
        self._sub_reports.append(sub_report)
        self.count_input()

    def gen_reports(self) -> Iterator[R]:
        for report in self._sub_reports:
            yield report

    def clear(self) -> None:
        incomplete = []
        for sub_report in self._sub_reports:
            if not sub_report.done:
                incomplete.append(sub_report)
                sub_report.clear()
        self._n_input = len(incomplete)
        if self._n_expected is not None:
            self._n_expected -= len(self._sub_reports) - self._n_input
        self._sub_reports = incomplete


class MultiKeyedError(Exception):
    def __init__(self, errors: Dict[Any, Exception]):
        self.errors = errors

    def __str__(self) -> str:
        res = ["Multiple Errors:"] + [f"{k}: {e}" for k, e in self.errors.items()]
        return "\n\t".join(res)


K = TypeVar("K")


# TODO: Now that reports have 'description' do we need this?
class MultiDictReport(SummaryReport[R], Generic[K, R]):
    """Collection of related reports, each identified with a unique key"""

    def __init__(
        self,
        description: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        depth: int = 0,
        prog_hook: Optional[ProgressHookBase[Any]] = None,
        n_expected: Optional[int] = None,
    ):
        self._sub_reports: Dict[K, R] = {}
        super().__init__(description, meta_data, depth, prog_hook, n_expected)

    def __getitem__(self, key: K) -> R:
        return self._sub_reports[key]

    def __setitem__(self, key: K, sub_report: R) -> None:
        if key in self._sub_reports:
            raise ValueError(f"Already have report with key: {key}")
        self._prep_sub_report(sub_report)
        self._sub_reports[key] = sub_report
        self.count_input()

    def __len__(self) -> int:
        return len(self._sub_reports)

    def __iter__(self) -> Iterator[K]:
        for key in self._sub_reports:
            yield key

    def __contains__(self, key: K) -> bool:
        return key in self._sub_reports

    def keys(self) -> KeysView[K]:
        return self._sub_reports.keys()

    def values(self) -> ValuesView[R]:
        return self._sub_reports.values()

    def items(self) -> ItemsView[K, R]:
        return self._sub_reports.items()

    def gen_reports(self) -> Iterator[R]:
        for report in self._sub_reports.values():
            yield report

    def check_errors(self) -> None:
        """Raise an exception if any errors have occured so far"""
        errors = {}
        for key, sub_report in self._sub_reports.items():
            try:
                sub_report.check_errors()
            except Exception as e:
                errors[key] = e
        if errors:
            raise MultiKeyedError(errors)

    def clear(self) -> None:
        incomplete = {}
        for key, sub_report in self._sub_reports.items():
            if not sub_report.done:
                incomplete[key] = sub_report
                sub_report.clear()
        self._n_input = len(incomplete)
        if self._n_expected is not None:
            self._n_expected -= len(self._sub_reports) - self._n_input
        self._sub_reports = incomplete

    def __str__(self) -> str:
        lines = [BaseReport.__str__(self)]
        lines.append(f"  * n_success: {self.n_success} ({self.n_sub_success} sub-ops)")
        if self.n_warnings > 0:
            lines.append(
                f"  * n_warnings: {self.n_warnings} ({self.n_sub_warnings} sub-ops)"
            )
        if self.n_errors > 0:
            lines.append(f"  * n_errors: {self.n_errors} ({self.n_sub_errors} sub-ops)")
        if not self.all_success:
            for key, rep in self._sub_reports.items():
                if rep.all_success:
                    continue
                rep_str = str(rep).replace("\n", "\n    ")
                lines.append(f"  {key}:")
                lines.append(f"    {rep_str}")
        return "\n".join(lines)
