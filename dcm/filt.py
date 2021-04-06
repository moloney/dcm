"""DICOM data filtering"""
from __future__ import annotations
import logging, operator, re
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Any, Dict, Union, MutableMapping
from typing_extensions import Protocol

from pydicom import Dataset, DataElement
from pydicom.uid import generate_uid

from .util import DuplicateDataError, TomlConfigurable, InlineConfigurable
from .lazyset import LazySet, FrozenLazySet, AllElems
from .query import (
    QueryLevel,
    QueryResult,
    DataNode,
    get_uid,
    uid_elems,
    QueryLevelMismatchError,
    InconsistentDataError,
)


log = logging.getLogger(__name__)


uid_elem_set = FrozenLazySet(uid_elems.values())


@dataclass(frozen=True)
class _BaseFilter:
    read_elems: FrozenLazySet[str] = field(
        default_factory=lambda: FrozenLazySet(AllElems)
    )
    """Elements the function needs to read to make any modifications"""

    write_elems: FrozenLazySet[str] = field(
        default_factory=lambda: FrozenLazySet(AllElems)
    )
    """Elements that might be modified by the function"""

    # TODO: Invertible is misleading, you could "invert" from a fan-out mapping
    #       (one input gets multiple outputs) but we don't want to include that
    #       here.
    invertible_elems: FrozenLazySet[str] = field(default_factory=FrozenLazySet)
    """Elements where values are only modified in isolated/deterministic way

    The same input value for these elements should always get the same output
    value.
    """

    def __call__(self, data_set: Dataset) -> Dataset:
        raise NotImplementedError

    @property
    def uninvertible_elems(self) -> FrozenLazySet[str]:
        return self.write_elems - self.invertible_elems

    @property
    def invertible_uids(self) -> FrozenLazySet[str]:
        return uid_elem_set - self.uninvertible_elems


@dataclass(frozen=True)
class _SingleFilter:
    func: Callable[[Dataset], Optional[Dataset]]
    """A function that takes a DICOM data set and can return a modified
    version, or None to signify the data should be skipped"""


@dataclass(frozen=True)
class Filter(_BaseFilter, _SingleFilter):
    """Callable for modifying data sets with info about its read/write needs

    By default it is assumed the filter needs access to the full data set and
    can modify any element in any way. More info can be provided by setting
    the various `*_elems` attributes which will allow for various optimizations.
    """

    def __post_init__(self) -> None:
        # Make sure invertible_elems is subset of write_elems
        if self.invertible_elems:
            object.__setattr__(
                self, "write_elems", self.write_elems | self.invertible_elems
            )
        # Make sure all LazySets are Frozen
        for attr in ("read_elems", "write_elems", "invertible_elems"):
            val = getattr(self, attr)
            if not isinstance(val, FrozenLazySet):
                object.__setattr__(self, attr, FrozenLazySet(val))

    def __call__(self, data_set: Dataset) -> Dataset:
        return self.func(data_set)

    def get_dependencies(
        self, elems: Any
    ) -> Tuple[List[Callable[[Dataset], Optional[Dataset]]], FrozenLazySet[str]]:
        return ([self.func], self.read_elems)


@dataclass(frozen=True)
class _MultiFilter:
    filters: Tuple[Filter, ...]
    """The collection of filters"""


@dataclass(frozen=True)
class MultiFilter(_BaseFilter, _MultiFilter):
    def __post_init__(self) -> None:
        read_elems: LazySet[str] = LazySet()
        write_elems: LazySet[str] = LazySet()
        invertible_elems: LazySet[str] = LazySet()
        uninvertible: LazySet[str] = LazySet()
        for filt in self.filters:
            read_elems |= filt.read_elems
            write_elems |= filt.write_elems
            invertible_elems |= filt.invertible_elems
            uninvertible |= filt.write_elems - filt.invertible_elems
        invertible_elems -= uninvertible
        object.__setattr__(self, "read_elems", FrozenLazySet(read_elems))
        object.__setattr__(self, "write_elems", FrozenLazySet(write_elems))
        object.__setattr__(self, "invertible_elems", FrozenLazySet(invertible_elems))

    def get_dependencies(
        self, elems: Any
    ) -> Tuple[List[Callable[[Dataset], Optional[Dataset]]], FrozenLazySet[str]]:
        funcs = []
        dep_elems: LazySet[str] = LazySet()
        for filt in self.filters:
            if (dep_elems & filt.write_elems) or (elems & filt.write_elems):
                funcs.append(filt.func)
                dep_elems |= filt.read_elems
        return (funcs, FrozenLazySet(dep_elems))

    def __call__(self, data_set: Dataset) -> Dataset:
        for filt in self.filters:
            data_set = filt(data_set)
            if data_set is None:
                break
        return data_set


@dataclass
class DataCollection:
    """Collection of dicom data that can include inconsistent/duplicate data"""

    qr: QueryResult
    """Captures all the consistent non-duplicate data"""

    inconsistent: List[Dataset] = field(default_factory=list)
    """List of data sets that can't be stored in `qr` due to inconsistancy"""

    duplicate: List[Dataset] = field(default_factory=list)
    """List of data sets that are duplicates of those in `qr`"""


class DataTransform:
    """Transforms data, and keeps track so it can be reversed

    Abstract base class. All subclasses should have two QueryResult attributes
    `old` and `new` (which should be treated as read-only) plus the methods
    below.
    """

    old: QueryResult
    """Pre-transform data"""

    new: QueryResult
    """Post-transform data"""

    def add(self, old_ds: Dataset, new_ds: Dataset) -> None:
        """Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        """
        raise NotImplementedError

    def reverse(self, new_qr: QueryResult) -> DataCollection:
        """Invert the transformation for some subset of the new QueryResult

        Returns a DataCollection."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"DataTransform: {self.new}"


class DummyTransform(DataTransform):
    """Dummy transform for efficiently handling no-op"""

    def __init__(self, qr: QueryResult):
        self.old = self.new = qr

    def add(self, old_ds: Dataset, new_ds: Dataset) -> None:
        """Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        """
        assert old_ds is new_ds
        if old_ds in self.old:
            raise DuplicateDataError
        self.old.add(old_ds)

    def reverse(self, new_qr: QueryResult) -> DataCollection:
        """Invert the transformation for some subset of the new QueryResult

        Returns a DataCollection"""
        return DataCollection(new_qr)


class FilterTransform(DataTransform):
    """Data transform described by Filter"""

    def __init__(self, qr: QueryResult, filt: _BaseFilter):
        self.old = qr  # TODO: Make a copy here? Or at least document that we don't...
        self.new = QueryResult(qr.level)
        self._new_to_old: Dict[QueryLevel, Dict[str, str]] = {
            lvl: {} for lvl in QueryLevel
        }
        self.fixed_inconsistent: Dict[str, Dataset] = {}
        self.fixed_duplicates: Dict[str, Dataset] = {}
        self.fully_invertible = filt.invertible_uids
        for old_ds in qr:
            new_ds = filt(old_ds)
            if new_ds is not None:
                try:
                    new_uid = self._add_new(old_ds, new_ds)
                except InconsistentDataError:
                    log.warning("Filter made data inconsistent")
                    raise
                if new_uid is None:
                    log.warning("Filter made data into a duplicate")
                    raise DuplicateDataError()

    def _add_new(self, old_ds: Dataset, new_ds: Dataset) -> Optional[str]:
        new_dupe = new_ds in self.new
        if new_dupe:
            return None
        self.new.add(new_ds)
        for lvl, uid_attr in uid_elems.items():
            new_uid = get_uid(lvl, new_ds)
            old_uid = get_uid(lvl, old_ds)
            self._new_to_old[lvl][new_uid] = old_uid
            if lvl == self.old.level:
                break
        return new_uid

    def add(self, old_ds: Dataset, new_ds: Dataset) -> None:
        """Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        """
        old_inconsistent = old_dupe = False
        try:
            old_dupe = old_ds in self.old
        except InconsistentDataError:
            old_inconsistent = True
        else:
            if not old_dupe:
                self.old.add(old_ds)
        if new_ds is not None:
            try:
                new_uid = self._add_new(old_ds, new_ds)
            except InconsistentDataError:
                if not old_inconsistent:
                    log.warning("Filter made data inconsistent")
                    self.old.remove(old_ds)
                raise
            if new_uid is None:
                if not old_dupe:
                    log.warning("Filter made data into a duplicate")
                    self.old.remove(old_ds)
                raise DuplicateDataError()
            if old_inconsistent:
                self.fixed_inconsistent[new_uid] = old_ds
            elif old_dupe:
                self.fixed_duplicates[new_uid] = old_ds

    def reverse(self, new_qr: QueryResult) -> DataCollection:
        """Invert the remapping for some subset of the new QueryResult

        Returns a DataCollection so we can also capture data that was duplicate
        or inconsistent prior to filtering (and thus can't be included in a
        QueryResult)
        """
        if new_qr.level != self.new.level:
            if new_qr.level > self.new.level or not self.fully_invertible:
                raise QueryLevelMismatchError()
        res = QueryResult(new_qr.level)
        inconsist = []
        dupes = []
        for ds in new_qr:
            assert ds in self.new
            new_uid = get_uid(new_qr.level, ds)
            old_uid = self._new_to_old[new_qr.level][new_uid]
            if new_qr.level == self.new.level:
                old_dupe = self.fixed_duplicates.get(new_uid)
                if old_dupe is not None:
                    dupes.append(old_dupe)
                    continue
                old_inconsist = self.fixed_inconsistent.get(new_uid)
                if old_inconsist is not None:
                    inconsist.append(old_inconsist)
                    continue
            old_node = DataNode(new_qr.level, old_uid)
            res |= self.old.sub_query(old_node, max_level=new_qr.level)
        return DataCollection(res, inconsist, dupes)


def get_transform(qr: QueryResult, filt: Optional[_BaseFilter]) -> DataTransform:
    if filt is None:
        return DummyTransform(qr)
    else:
        return FilterTransform(qr, filt)


def make_uid_update_cb(
    uid_prefix: str = "2.25", add_uid_entropy: Optional[List[Any]] = None
) -> Callable[[Dataset, DataElement], None]:
    if add_uid_entropy is None:
        add_uid_entropy = []
    if uid_prefix[-1] != ".":
        uid_prefix += "."

    def update_uids_cb(ds: Dataset, elem: DataElement) -> None:
        """Callback for updating UID values except `SOPClassUID`"""
        if elem.VR == "UI" and elem.keyword != "SOPClassUID":
            if elem.VM > 1:
                elem.value = [
                    generate_uid(uid_prefix, [x] + add_uid_entropy)  # type: ignore
                    for x in elem.value
                ]
            else:
                elem.value = generate_uid(
                    uid_prefix, [elem.value] + add_uid_entropy
                )  # type: ignore

    return update_uids_cb


def make_edit_filter(
    edit_dict: Dict[str, Any],
    update_uids: bool = True,
    uid_prefix: str = "2.25",
    add_uid_entropy: Optional[List[Any]] = None,
) -> _BaseFilter:
    """Make a Filter that edits some DICOM attributes

    Parameters
    ----------
    edit_dict : dict
        Maps keywords to new values (or None to delete the element)

    update_uids : bool
        Set to False to avoid automatically updating UID values

    add_uid_entropy : list or None
        One or more strings used as "entropy" when remapping UIDs
    """
    update_uids_cb = make_uid_update_cb(uid_prefix, add_uid_entropy)

    def edit_func(ds: Dataset) -> Dataset:
        # TODO: Handle nested attributes (VR of SQ)
        for attr_name, val in edit_dict.items():
            if val is None and hasattr(ds, attr_name):
                delattr(ds, attr_name)
            elif hasattr(ds, attr_name):
                setattr(ds, attr_name, val)
        if update_uids:
            ds.walk(update_uids_cb)
            if hasattr(ds, "file_meta"):
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        return ds

    if update_uids is False:
        write_elems = FrozenLazySet(edit_dict.keys())
    else:
        # TODO: Should be able to be more specific here...
        write_elems = FrozenLazySet(AllElems)
    edit_filter = Filter(edit_func, write_elems=write_elems)
    return edit_filter


def make_reject_filter(
    reject_dict: Dict[str, Tuple[Callable[[Dataset, Dataset], Any], Any]]
) -> _BaseFilter:
    """Make a filter that rejects certain datasets

    This is useful when you want to limit a transfer based on attributes that can't be
    queried for.

    Parameters
    ----------
    reject_dict
        Maps keywords to (operator, rvalue) tuples defining which datasets to reject
    """

    def reject_func(ds: Dataset) -> Optional[Dataset]:
        for attr_name, (op, val) in reject_dict.items():
            if not hasattr(ds, attr_name):
                log.warning(
                    "Dataset doesn't have '{attr_name}' element for reject filter"
                )
                continue
            if op(getattr(ds, attr_name), val):
                return None
        return ds

    read_elems = FrozenLazySet(reject_dict.keys())
    reject_filter = Filter(
        reject_func, read_elems=read_elems, write_elems=FrozenLazySet()
    )
    return reject_filter


CMP_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "~=": lambda l, r: re.match(r, l),
    "in": lambda l, r: l in r,
}


class Selector(Protocol):
    """Abstract base for objects that can select/reject data sets"""

    def get_read_elems(self) -> FrozenLazySet[str]:
        raise NotImplementedError

    def test_ds(self, ds: Dataset) -> bool:
        raise NotImplementedError

    def get_filter(self) -> Filter:
        return Filter(
            lambda ds: ds if self.test_ds(ds) else None,
            read_elems=self.get_read_elems(),
            write_elems=FrozenLazySet(),
        )


@dataclass(frozen=True)
class SingleSelector(Selector, InlineConfigurable["SingleSelector"]):
    """Define a selection based on a single attribute

    Can be defined in config file or as single string
    """

    attr: str

    op: str

    rvalue: Any

    invert: bool = False

    reject_missing: bool = True

    _op: Optional[Callable[[Any, Any], bool]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        cmp_op = CMP_OPS.get(self.op)
        if cmp_op is None:
            raise ValueError(f"Unknown operator: {self.op}")
        if self.invert:
            object.__setattr__(self, "_op", lambda l, r: not cmp_op(l, r))
        else:
            object.__setattr__(self, "_op", cmp_op)

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        """Support simple forms to be encoded in single string"""
        # TODO: Can we use safe_eval on rvalue to convert it?
        #       Would still need more that that to support dates, use TOML?
        match = re.match(
            r"^\s*(?P<invert>!?)\s*(?P<attr>\S+)\s+(?P<op>\S+)\s+(?P<rvalue>.+?)\s*$",
            in_str,
        )
        if not match:
            raise ValueError(f"Invalid in_str for Selector: {in_str}")
        res: Dict[str, Any] = match.groupdict()
        if res["invert"] == "!":
            res["invert"] = True
        return res

    def get_read_elems(self) -> FrozenLazySet[str]:
        return FrozenLazySet((self.attr,))

    def test_ds(self, ds: Dataset) -> bool:
        lvalue = getattr(ds, self.attr, None)
        if lvalue is None:
            return not self.reject_missing
        # TODO: Probably want some type coalescing for DS, maybe dates?
        assert self._op is not None
        return self._op(lvalue, self.rvalue)


@dataclass(frozen=True)
class MultiSelector(Selector, TomlConfigurable["MultiSelector"]):
    """Logically combine selectors for more complex selections"""

    all_of: Tuple[Selector, ...] = tuple()

    any_of: Tuple[Selector, ...] = tuple()

    none_of: Tuple[Selector, ...] = tuple()

    def __post_init__(self) -> None:
        if len(self.all_of) + len(self.any_of) + len(self.none_of) == 0:
            raise ValueError("Must give at least one type of selector")
        for attr in ("all_of", "any_of", "none_of"):
            val = getattr(self, attr)
            if not isinstance(val, tuple):
                object.__setattr__(self, attr, tuple(val))

    def get_read_elems(self) -> FrozenLazySet[str]:
        elems: LazySet[str] = LazySet()
        for selectors in (self.all_of, self.any_of, self.none_of):
            for s in selectors:
                elems |= s.get_read_elems()
        return FrozenLazySet(elems)

    def test_ds(self, ds: Dataset) -> bool:
        if self.none_of and any(s.test_ds(ds) for s in self.none_of):
            return False
        if not all(s.test_ds(ds) for s in self.all_of):
            return False
        if self.any_of and not any(s.test_ds(ds) for s in self.any_of):
            return False
        return True
