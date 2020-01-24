'''Various utility functions'''
from __future__ import annotations
import os, json

from dataclasses import dataclass, field, fields, astuple
from contextlib import asynccontextmanager
from typing import (AsyncGenerator, Any, AsyncIterator, Dict, List, TypeVar,
                    Optional, Union, Generic, Iterator, Tuple, KeysView,
                    ValuesView, ItemsView, Type)
from typing_extensions import Protocol

from pydicom import Dataset


def dict_to_ds(data_dict: Dict[str, Any]) -> Dataset:
    '''Convert a dict to a pydicom.Dataset'''
    ds = Dataset()
    for k, v in data_dict.items():
        setattr(ds, k, v)
    return ds


class DicomDataError(Exception):
    '''Base class for exceptions from erroneous dicom data'''


class DuplicateDataError(DicomDataError):
    '''A duplicate dataset was found'''



class Serializable(Protocol):

    def to_json_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> Serializable:
        raise NotImplementedError


class _Serializer:
    '''Defines class decorator for registering JSON serializable objects

    Adapted from: https://stackoverflow.com/questions/51975664/serialize-and-deserialize-objects-from-user-defined-classes'''
    def __init__(self, classname_key: str = '__class__'):
        self._key = classname_key
        self._classes: Dict[str, Type[Serializable]] = {}

    def __call__(self, class_: Any) -> Any:
        assert hasattr(class_, 'to_json_dict') and hasattr(class_, 'from_json_dict')
        self._classes[class_.__name__] = class_
        return class_

    def decoder_hook(self, d: Dict[str, Any]) -> Union[Dict[str, Any], Serializable]:
        classname = d.pop(self._key, None)
        if classname:
            return self._classes[classname].from_json_dict(d)
        return d

    def encoder_default(self, obj: Serializable) -> Dict[str, Any]:
        d = obj.to_json_dict()
        d[self._key] = type(obj).__name__
        return d

    def dumps(self, obj: Serializable, **kwargs: Any) -> str:
        return json.dumps(obj, default=self.encoder_default, **kwargs)

    def loads(self, json_str: str) -> Serializable:
        return json.loads(json_str, object_hook=self.decoder_hook)


serializer = _Serializer()
'''Class decorator for registering JSON serializable objects'''

@dataclass
class Report:
    '''Abstract base class for all reports'''

    @property
    def n_expected(self) -> Optional[int]:
        '''Number of expected inputs, or None if unknown'''
        # TODO: Probably makes more sense to return None here if we are keeping this method
        raise NotImplementedError

    @property
    def n_input(self) -> int:
        '''Number of inputs seen so far'''
        raise NotImplementedError

    @property
    def n_success(self) -> int:
        '''Number of successfully handled inputs'''
        raise NotImplementedError

    @property
    def n_errors(self) -> int:
        '''Number of errors, where a single input can cause multiple errors'''
        raise NotImplementedError

    @property
    def n_warnings(self) -> int:
        '''Number of warnings, where a single input can cause multiple warnings
        '''
        raise NotImplementedError

    @property
    def all_success(self) -> bool:
        return self.n_errors + self.n_warnings == 0

    def __len__(self) -> int:
        return self.n_success + self.n_warnings + self.n_errors

    def log_issues(self) -> None:
        '''Log a summary of error/warning statuses'''
        raise NotImplementedError

    def check_errors(self) -> None:
        '''Raise an exception if any errors occured'''
        raise NotImplementedError

    # TODO: This is tricky to support for some subclasses, separate it out?
    def clear(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        lines = [f'{self.__class__.__name__}:']
        for f in fields(self):
            f_name = f.name
            val_str = str(getattr(self, f_name)).replace('\n', '\n\t')
            lines.append(f'\t{f_name}: {val_str}')
        return '\n'.join(lines)

@dataclass
class IndividualReport(Report):
    '''An individual report that needs to be marked done when complete'''
    _done: bool = False

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, val: bool) -> None:
        if not val:
            raise ValueError("Setting `done` to False is not allowed")
        if self._done:
            raise ValueError("Report was already marked done")
        self._done = True


class MultiError(Exception):
    def __init__(self, errors: List[Exception]):
        self.errors = errors

    def __str__(self) -> str:
        res = ['Multiple Errors:'] + [str(e) for e in self.errors]
        return '\n\t'.join(res)


R = TypeVar('R', bound=Union[IndividualReport, 'MultiReport[Any]'])


@dataclass
class MultiReport(Report, Generic[R]):
    '''Abstract base class for all MultiReports'''
    def gen_reports(self) -> Iterator[R]:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return all(r.done for r in self.gen_reports())

    @property
    def n_expected(self) -> Optional[int]:
        expected = [r.n_expected for r in self.gen_reports()]
        if any(e is None for e in expected):
            return None
        return sum(expected)

    @property
    def n_input(self) -> int:
        return sum(r.n_input for r in self.gen_reports())

    @property
    def n_warnings(self) -> int:
        return sum(r.n_warnings for r in self.gen_reports())

    @property
    def n_errors(self) -> int:
        return sum(r.n_errors for r in self.gen_reports())

    @property
    def all_success(self) -> bool:
        return self.n_errors + self.n_warnings == 0

    def log_issues(self) -> None:
        '''Produce log messages for any warning/error statuses'''
        for report in self.gen_reports():
            report.log_issues()

    def __str__(self) -> str:
        lines = [f'{self.__class__.__name__}:']
        for rep in self.gen_reports():
            rep_str = str(rep).replace('\n', '\n\t')
            lines.append(f'\t* {rep_str}')
        return '\n'.join(lines)


@dataclass
class MultiListReport(MultiReport[R]):
    '''Sequence of related reports'''

    sub_reports: List[R] = field(default_factory=list)

    def __getitem__(self, idx: int) -> R:
        return self.sub_reports[idx]

    def __len__(self) -> int:
        return len(self.sub_reports)

    def __iter__(self) -> Iterator[R]:
        for report in self.sub_reports:
            yield report

    def append(self, val: R) -> None:
        self.sub_reports.append(val)

    def gen_reports(self) -> Iterator[R]:
        for report in self.sub_reports:
            yield report

    def check_errors(self) -> None:
        '''Raise an exception if any errors have occured so far'''
        errors = []
        for sub_report in self.sub_reports:
            try:
                sub_report.check_errors()
            except Exception as e:
                errors.append(e)
        if errors:
            raise MultiError(errors)

    def clear(self) -> None:
        incomplete = []
        for sub_report in self.sub_reports:
            if not sub_report.done:
                incomplete.append(sub_report)
                sub_report.clear()
        self.sub_reports = incomplete


class MultiKeyedError(Exception):
    def __init__(self, errors: Dict[Any, Exception]):
        self.errors = errors

    def __str__(self) -> str:
        res = ['Multiple Errors:'] + [f'{k}: {e}' for k, e in self.errors.items()]
        return '\n\t'.join(res)


K = TypeVar('K')


@dataclass
class MultiDictReport(MultiReport[R], Generic[K, R]):
    '''Collection of related reports, each identified with a unique key'''
    sub_reports: Dict[K, R] = field(default_factory=dict)

    def __getitem__(self, key: K) -> R:
        return self.sub_reports[key]

    def __setitem__(self, key: K, val: R) -> None:
        if key in self.sub_reports:
            raise ValueError(f"Already have report with key: {key}")
        self.sub_reports[key] = val

    def __len__(self) -> int:
        return len(self.sub_reports)

    def __iter__(self) -> Iterator[K]:
        for key in self.sub_reports:
            yield key

    def __contains__(self, key: K) -> bool:
        return key in self.sub_reports

    def keys(self) -> KeysView[K]:
        return self.sub_reports.keys()

    def values(self) -> ValuesView[R]:
        return self.sub_reports.values()

    def items(self) -> ItemsView[K, R]:
        return self.sub_reports.items()

    def gen_reports(self) -> Iterator[R]:
        for report in self.sub_reports.values():
            yield report

    def check_errors(self) -> None:
        '''Raise an exception if any errors have occured so far'''
        errors = {}
        for key, sub_report in self.sub_reports.items():
            try:
                sub_report.check_errors()
            except Exception as e:
                errors[key] = e
        if errors:
            raise MultiKeyedError(errors)

    def clear(self) -> None:
        incomplete = {}
        for key, sub_report in self.sub_reports.items():
            if not sub_report.done:
                incomplete[key] = sub_report
                sub_report.clear()
        self.sub_reports = incomplete

    def __str__(self) -> str:
        lines = [f'{self.__class__.__name__}:']
        for key, rep in self.sub_reports.items():
            rep_str = str(rep).replace('\n', '\n\t')
            lines.append(f'\t{rep_str}')
        return '\n'.join(lines)


PathInputType = Union[bytes, str, 'os.PathLike']


# Generic element type
T = TypeVar('T')


@asynccontextmanager
async def aclosing(thing : AsyncGenerator[T, None]) -> AsyncIterator[AsyncGenerator[T, None]]:
    '''Context manager that ensures that an async iterator is closed

    See PEP 533 for an explanation on why this is (unfortunately) needded.
    '''
    try:
        yield thing
    finally:
        await thing.aclose()


def fstr_eval(f_str: str,
              context: Dict[str, Any],
              raw_string: bool = False,
              ) -> str:
    """Evaluate a string as an f-string literal.

    Adapted from: https://stackoverflow.com/questions/54700826/how-to-evaluate-a-variable-as-a-python-f-string

    Parameters
    ----------
    f_str
        The string to interpolate

    context
        The variables available when evaluating the string

    raw_string:
        Evaluate as a raw literal (don't escape \\). Defaults to False.
    """
    # Prefix all local variables with _ to reduce collisions in case
    # eval is called in the local namespace.
    ta = "'''" # triple-apostrophes constant, for readability
    if ta in f_str:
        raise ValueError("Triple-apostrophes ''' are forbidden. " + \
                         'Consider using """ instead.')

    # Strip apostrophes from the end of _s and store them in _ra.
    # There are at most two since triple-apostrophes are forbidden.
    if f_str.endswith("''"):
        ra = "''"
        f_str = f_str[:-2]
    elif f_str.endswith("'"):
        ra = "'"
        f_str = f_str[:-1]
    else:
        ra = ""

    prefix = 'rf' if raw_string else 'f'
    return eval(prefix + ta + f_str + ta, context) + ra
