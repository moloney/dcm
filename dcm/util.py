"""Various utility functions"""
from __future__ import annotations
import os, logging, string
import asyncio, threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum, IntEnum
from typing import (
    AsyncGenerator,
    Any,
    AsyncIterator,
    Dict,
    List,
    Tuple,
    TypeVar,
    Optional,
    Union,
    Generic,
    Iterable,
    Type,
    Callable,
    Literal,
    get_origin,
    get_args,
)
from typing_extensions import Protocol

import attrs
import cattrs
from cattrs.preconf.json import make_converter as make_json_converter
from pydicom import Dataset
from pydicom.tag import BaseTag, Tag, TagType
from pydicom.datadict import tag_for_keyword
from rich.progress import Progress, Task


log = logging.getLogger(__name__)


def dict_to_ds(data_dict: Dict[str, Any]) -> Dataset:
    """Convert a dict to a pydicom.Dataset"""
    ds = Dataset()
    for k, v in data_dict.items():
        setattr(ds, k, v)
    return ds


def str_to_tag(in_str: str) -> BaseTag:
    """Convert string representation to pydicom Tag

    The string can be a keyword, or two numbers separated by a comma
    """
    if in_str[0].isupper():
        res = tag_for_keyword(in_str)
        if res is None:
            raise ValueError("Invalid element ID: %s" % in_str)
        return Tag(res)
    try:
        group_num, elem_num = [int(x.strip(), 0) for x in in_str.split(",")]
    except Exception:
        raise ValueError("Invalid element ID: %s" % in_str)
    return Tag(group_num, elem_num)


class DicomDataError(Exception):
    """Base class for exceptions from erroneous dicom data"""


class DuplicateDataError(DicomDataError):
    """A duplicate dataset was found"""


json_serializer = make_json_converter()
"""JSON (de)serializer

Handles most classes automatically, otherwise classes should inherit 
`CustomJsonSerializable` and provide the two required methods, which can in turn use
this on sub objects.
"""


class CustomJsonSerializable(Protocol):
    """Base class for objects that need custom JSON (de)serialization"""

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "CustomJsonSerializable":
        raise NotImplementedError

    def to_json_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


json_serializer.register_structure_hook(
    CustomJsonSerializable, lambda v, t: t.from_json_dict(v)  # type: ignore
)
json_serializer.register_unstructure_hook(
    CustomJsonSerializable, lambda i: i.to_json_dict()
)


def _flexible_enum_struct(data: Any, cls: Type[Enum]) -> Enum:
    """More flexible Enum structuring hook allows names as well as values"""
    for e in cls:
        if data == e.value:
            return e
    if isinstance(data, str):
        for e in cls:
            if data.upper() == e.name:
                return e
    raise ValueError(f"Unable to convert '{data}' to {cls.__name__}")


# TODO: Hopefully this won't be needed with newer versions of cattrs (see issue 278)
def is_primativish(t: Any) -> bool:
    """Match types that are primatives, or simple collections of primatives"""
    if t in (str, bytes, int, float, bool, None, Ellipsis):
        return True
    origin = get_origin(t)
    if origin is Literal:
        return True
    if (basetype := cattrs._compat.get_newtype_base(t)) is not None:
        return is_primativish(basetype)
    if origin in [Union, Tuple, List]:
        return all(is_primativish(ty) for ty in get_args(t))
    return False


# Setup any common hooks for JSON / TOML
for conv in (json_serializer,):
    # Convert Enums to name and allow name or value as input
    conv.register_structure_hook(Enum, _flexible_enum_struct)
    conv.register_unstructure_hook(Enum, lambda v: v.name)
    conv.register_structure_hook(IntEnum, _flexible_enum_struct)
    conv.register_unstructure_hook(IntEnum, lambda v: v.name)
    # Handle pydicom Datasets <-> dict
    conv.register_structure_hook(Dataset, lambda v, _: dict_to_ds(v))
    # Workaround cattrs issue 278
    conv.register_structure_hook_func(is_primativish, lambda v, _: v)
    # TODO: Need similar unstructure hook here?


TC_Type = TypeVar("TC_Type", covariant=True)


class TomlConfigurable(Generic[TC_Type], Protocol):
    """Protocol for objects that are configurable through TOML"""

    @classmethod
    def from_toml_dict(cls, toml_dict: Dict[str, Any]) -> TC_Type:
        return cls(**toml_dict)  # type: ignore

    @classmethod
    def from_toml_val(cls, val: Dict[str, Any]) -> TC_Type:
        return cls.from_toml_dict(val)


class InlineConfigurable(Generic[TC_Type], TomlConfigurable[TC_Type], Protocol):
    """Protocol for objects that are TOML and inline configurable"""

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_toml_val(cls, val: Union[str, Dict[str, Any]]) -> TC_Type:
        if isinstance(val, str):
            val = cls.inline_to_dict(val)
        return cls.from_toml_dict(val)


PathInputType = Union[str, "os.PathLike"]


@asynccontextmanager
async def aclosing(
    thing: AsyncGenerator[Any, None]
) -> AsyncIterator[AsyncGenerator[Any, None]]:
    """Context manager that ensures that an async iterator is closed

    See PEP 533 for an explanation on why this is (unfortunately) needed.
    """
    try:
        yield thing
    finally:
        await thing.aclose()


def fstr_eval(
    f_str: str,
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
    ta = "'''"  # triple-apostrophes constant, for readability
    if ta in f_str:
        raise ValueError(
            "Triple-apostrophes ''' are forbidden. " + 'Consider using """ instead.'
        )

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

    prefix = "rf" if raw_string else "f"
    return eval(prefix + ta + f_str + ta, context) + ra


class FallbackFormatter(string.Formatter):
    """String formatter that uses plain str conversion when formatting fails"""

    def format_field(self, value: Any, spec: str) -> str:
        try:
            return super().format_field(value, spec)
        except ValueError:
            return str(value)


fallback_fmt = FallbackFormatter()


_default_thread_shutdown = threading.Event()


def make_done_callback(
    shutdown: threading.Event,
) -> Callable[[asyncio.Future[Any]], None]:
    def done_callback(task: asyncio.Future[Any]) -> None:
        try:
            ex = task.exception()
            if ex is not None:
                loop = task.get_loop()
                loop.call_exception_handler(
                    {
                        "message": "unhandled exception from task",
                        "exception": ex,
                        "task": task,
                    }
                )
                shutdown.set()

        except asyncio.CancelledError:
            pass

    return done_callback


def create_thread_task(
    func: Callable[..., Any],
    args: Optional[Iterable[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    thread_pool: Optional[ThreadPoolExecutor] = None,
    shutdown: Optional[threading.Event] = None,
) -> asyncio.Future[Any]:
    """Helper to turn threads into tasks with clean shutdown option

    Canceling a thread is not generally possible, so we pass an additional
    kwarg `shutdown` to `func` which will point to an event that can be
    monitored periodically for shutdown events.

    Worker threads are also prone to hiding exceptions, so we automatically
    add a callback to report exceptions in a timely manner and make sure the shutdown
    event is set, so any other threads sharing that event will also shutdown.
    """
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = {}
    if loop is None:
        loop = asyncio.get_running_loop()
    if shutdown is None:
        shutdown = _default_thread_shutdown
    kwargs["shutdown"] = shutdown
    pfunc = partial(func, *args, **kwargs)
    task = asyncio.ensure_future(loop.run_in_executor(thread_pool, pfunc))
    task.add_done_callback(make_done_callback(shutdown))
    return task
