'''DICOM data filtering'''
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

from pydicom import Dataset
from pydicom.uid import generate_uid

from .util import DuplicateDataError
from .query import (QueryLevel, QueryResult, DataNode, get_uid, uid_elems,
                    QueryLevelMismatchError, InconsistentDataError)


log = logging.getLogger(__name__)


class _AllElems:
    '''Sentinal representing all elements'''
    __slots__ = tuple()
    def __repr__(self):
        return 'AllElems'

AllElems = _AllElems()


class LazyEnumerationError(Exception):
    '''Raised when attempting to iterate an LazySet that can't be enumerated
    '''

class _BaseLazySet:
    '''Common functionality for LazySet and FrozenLazySet'''
    def __init__(self, elems=None, exclude=None):
        if exclude is None:
            if isinstance(elems, _BaseLazySet):
                elems, exclude = elems._elems.copy(), elems._exclude.copy()
            else:
                exclude = self._set_type()
                if elems is None:
                    elems = self._set_type()
                else:
                    if elems is not AllElems:
                        if not isinstance(elems, self._set_type):
                            elems = self._set_type(elems)
                        else:
                            elems = elems.copy()
        else:
            if elems is not AllElems:
                raise ValueError("The 'elems' must be set to AllElems if "
                                 "'exclude' is not None")
            if exclude is not AllElems:
                if not isinstance(exclude, self._set_type):
                    exclude = self._set_type(exclude)
                else:
                    exclude = exclude.copy()
        self._elems = elems
        self._exclude = exclude

    def __contains__(self, elem):
        if self._elems is AllElems:
            return elem not in self._exclude
        return elem in self._elems

    def __repr__(self):
        return f'LazySet({self._elems}, exclude={self._exclude})'

    def __str__(self):
        if self._elems is AllElems:
            res = 'All Elements'
            if self._except:
                res += 'except %s' % set(self._except)
        else:
            res = str(set(self._elems))
        return res

    def __and__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                return LazySet(AllElems, self._exclude | other._exclude)
            return LazySet(self._elems - other._exclude)
        elif self._elems is AllElems:
            return LazySet(other._elems - self._exclude)
        return LazySet(self._elems & other._elems)


    def __or__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                return LazySet(AllElems, self._exclude & other._exclude)
            else:
                return LazySet(AllElems, other._exclude - self._elems)
        elif self._elems is AllElems:
            return LazySet(AllElems, self._exclude - other._elems)
        else:
            return LazySet(self._elems | other._elems)

    def __sub__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                return LazySet(AllElems, self._exclude | other._exclude)
            else:
                return LazySet(self._elems & other._exclude)
        elif self._elems is AllElems:
            return LazySet(AllElems, self._exclude | other._elems)
        else:
            return LazySet(self._elems - other._elems)

    def __bool__(self):
        return self._elems is AllElems or len(self._elems) != 0

    def __iter__(self):
        if self._elems is AllElems:
            raise LazyEnumerationError
        for elem in self._elems:
            yield elem

    def __len__(self):
        if self._elems is AllElems:
            raise LazyEnumerationError
        return len(self._elems)

    def excludes(self):
        for elem in self._excludes:
            yield elem

    def is_enumerable(self):
        return self._elems is not AllElems

    def collides(self, other):
        return not (self & other)


class LazySet(_BaseLazySet):
    '''Set like object that can contain all elements without enumerating them

    Can also represent exclusive sets (everything except...)
    '''

    _set_type = set

    def __iand__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._exclude |= other._exclude
            else:
                self._elems -= other._exclude
        elif self._elems is AllElems:
            self._elems = other._elems - self._exclude
            self._exclude = set()
        else:
            self._elems &= other._elems
        return self

    def __ior__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._exclude = self._exclude & other._exclude
            else:
                self._elems = AllElems
                self._exclude = other._exclude - self._elems
        elif self._elems is AllElems:
            self._exclude -= other._elems
        else:
            self._elems |= other._elems
        return self

    def __isub__(self, other):
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._exclude |= other._exclude
            else:
                self._elems &= other._exclude
        elif self._elems is AllElems:
            self._exclude |= other._elems
        else:
            self._elems -= other._elems
        return self


class FrozenLazySet(_BaseLazySet):
    '''Frozen LazySet'''

    _set_type = frozenset


uid_elem_set = FrozenLazySet(uid_elems.values())


@dataclass(frozen=True)
class Filter:
    '''Callable for modifying data sets with info about its read/write needs

    By default it is assumed the filter needs access to the full data set and
    can modify any element in any way. More info can be provided by setting
    the various `*_elems` attributes which will allow for various optimizations.
    '''
    func: Callable[[Dataset], Optional[Dataset]]
    '''A function that takes a DICOM data set and can return a modified
    version, or None to signify the data should be skipped'''

    read_elems: FrozenLazySet = field(default_factory=lambda: FrozenLazySet(AllElems))
    '''Elements the function needs to read to make any modifications'''

    write_elems: FrozenLazySet = field(default_factory=lambda: FrozenLazySet(AllElems))
    '''Elements that might be modified by the function'''

    # TODO: Invertible is misleading, you could "invert" from a fan-out
    invertible_elems: FrozenLazySet = field(default_factory=FrozenLazySet)
    '''Elements where values are only modified in isolated/deterministic way

    The same input value for these elements should always get the same output
    value.
    '''

    def __post_init__(self):
        # Make sure invertible_elems is subset of write_elems
        if self.invertible_elems:
            object.__setattr__(self, 'write_elems', self.write_elems | self.invertible_elems)
        # Make sure all LazySets are Frozen
        for attr in ('read_elems', 'write_elems', 'invertible_elems'):
            val = getattr(self, attr)
            if not isinstance(val, FrozenLazySet):
                object.__setattr__(self, attr, FrozenLazySet(val))

    @property
    def uninvertible_elems(self):
        return self.write_elems - self.invertible_elems

    @property
    def invertible_uids(self):
        return not (uid_elem_set & self.uninvertible_elems)

    def __call__(self, data_set):
        return self.func(data_set)

    def get_dependencies(self, elems):
        return ([self.func], self.read_elems)


# TODO: Relate this to Filter through inheritance?
@dataclass(frozen=True)
class MultiFilter:
    filters: Tuple[Filter]

    read_elems: Optional[FrozenLazySet] = None
    '''Elements the filter needs to read to make any modifications'''

    write_elems: Optional[FrozenLazySet] = None
    '''Elements that might be modified by the function'''

    invertible_elems: Optional[FrozenLazySet] = None
    '''Elements where the values are modified in isolated/deterministic way'''

    def __post_init__(self):
        read_elems = LazySet()
        write_elems = LazySet()
        invertible_elems = LazySet()
        uninvertible = LazySet()
        for filt in self.filters:
            read_elems |= filt.read_elems
            write_elems |= filt.write_elems
            invertible_elems |= filt.invertible_elems
            uninvertible |= filt.write_elems - filt.invertible_elems
        invertible_elems -= uninvertible
        object.__setattr__(self, 'read_elems', FrozenLazySet(read_elems))
        object.__setattr__(self, 'write_elems',  FrozenLazySet(write_elems))
        object.__setattr__(self, 'invertible_elems',  FrozenLazySet(invertible_elems))

    @property
    def uninvertible_elems(self):
        return self.write_elems - self.invertible_elems

    @property
    def invertible_uids(self):
        return not (uid_elem_set & self.uninvertible_elems)

    def get_dependencies(self, elems):
        funcs = []
        dep_elems = LazySet()
        for filt in self.filters:
            if (dep_elems & filt.write_elems) or (elems & filt.write_elems):
                funcs.append(filt.func)
                dep_elems |= filt.read_elems
        return (funcs, dep_elems)

    def __call__(self, data_set):
        for filt in self.filters:
            data_set = filt(data_set)
            if data_set is None:
                break
        return data_set


@dataclass
class DataCollection:
    '''Collection of dicom data that can include inconsistent/duplicate data
    '''
    qr: QueryResult
    inconsistent: List = field(default_factory=list)
    duplicate: List = field(default_factory=list)


class DataTransform:
    '''Transforms data, and keeps track so it can be reversed

    Abstract base class. All subclasses should have two QueryResult attributes
    `old` and `new` (which should be treated as read-only) plus the methods
    below.
    '''
    def add(self, old_ds, new_ds):
        '''Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        '''
        raise NotImplementedError

    def reverse(self, new_qr):
        '''Invert the transformation for some subset of the new QueryResult

        Returns a DataCollection.'''
        raise NotImplementedError


class DummyTransform(DataTransform):
    '''Dummy transform for efficiently handling no-op'''
    def __init__(self, qr):
        self.old = self.new = qr

    def add(self, old_ds, new_ds):
        '''Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        '''
        assert old_ds is new_ds
        if old_ds in self.old:
            raise DuplicateDataError
        self.old.add(old_ds)

    def reverse(self, new_qr):
        '''Invert the transformation for some subset of the new QueryResult

        Returns a DataCollection'''
        return DataCollection(new_qr)


class FilterTransform(DataTransform):
    '''Data transform described by Filter'''
    def __init__(self, qr, filt):
        self.old = qr #TODO: Make a copy here? Or at least document that we don't...
        self.new = QueryResult(qr.level)
        self._new_to_old = {lvl: {} for lvl in QueryLevel}
        self.fixed_inconsistent = {}
        self.fixed_duplicates = {}
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

    def _add_new(self, old_ds, new_ds):
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

    def add(self, old_ds, new_ds):
        '''Add pre/post transformed data set

        Will raise InconsistentDataError or DuplicateDataError based on the
        post-filtering data
        '''
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
                    self.old.delete(old_ds)
                raise
            if new_uid is None:
                if not old_dupe:
                    log.warning("Filter made data into a duplicate")
                    self.old.delete(old_ds)
                raise DuplicateDataError()
            if old_inconsistent:
                self.fixed_inconsistent[new_uid] = old_ds
            elif old_dupe:
                self.fixed_duplicates[new_uid] = old_ds

    def reverse(self, new_qr):
        '''Invert the remapping for some subset of the new QueryResult

        Returns a DataCollection so we can also capture data that was duplicate
        or inconsistent prior to filtering (and thus can't be included in a
        QueryResult)
        '''
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


def get_transform(qr, filt):
    if filt is None:
        return DummyTransform(qr)
    else:
        return FilterTransform(qr, filt)


def make_uid_update_cb(uid_prefix='2.25', add_uid_entropy=None):
    if add_uid_entropy is None:
        add_uid_entropy = []
    if uid_prefix[-1] != '.':
        uid_prefix += '.'
    def update_uids_cb(ds, elem):
        '''Callback for updating UID values except `SOPClassUID`'''
        if elem.VR == 'UI' and elem.keyword != 'SOPClassUID':
            if elem.VM > 1:
                elem.value = [generate_uid(uid_prefix,
                                           [x] + add_uid_entropy)
                              for x in elem.value]
            else:
                elem.value = generate_uid(uid_prefix,
                                          [elem.value] + add_uid_entropy)
    return update_uids_cb


def make_edit_filter(edit_dict, update_uids=True, uid_prefix='2.25',
                     add_uid_entropy=None):
    '''Make a Filter that edits some DICOM attributes

    Parameters
    ----------
    edit_dict : dict
        Maps keywords to new values (or None to delete the element)

    update_uids : bool
        Set to False to avoid automatically updating UID values

    add_uid_entropy : list or None
        One or more strings used as "entropy" when remapping UIDs
    '''
    update_uids_cb = make_uid_update_cb(add_uid_entropy)
    def edit_func(ds):
        # TODO: Handle nested attributes (VR of SQ)
        for attr_name, val in edit_dict.items():
            if val is None and hasattr(ds, attr_name):
                delattr(ds, attr_name)
            elif hasattr(ds, attr_name):
                setattr(ds, attr_name, val)
        if update_uids:
            ds.walk(update_uids_cb)
            if hasattr(ds, 'file_meta'):
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        return ds
    if update_uids is False:
        write_elems = LazySet(edit_dict.keys())
    else:
        write_elems = LazySet(AllElems)
    edit_filter = Filter(edit_func, write_elems=write_elems)
    return edit_filter