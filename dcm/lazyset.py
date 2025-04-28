"""Define set classes that can contain everything, or everything except specific elems
"""
from __future__ import annotations
from enum import Enum
from typing import (
    Optional,
    Set,
    Iterator,
    Iterable,
    Union,
    Any,
    ClassVar,
    Type,
    FrozenSet,
    TypeVar,
    Generic,
)
from typing_extensions import Final


class _AllElems(Enum):
    token = 0


AllElems: Final = _AllElems.token


class LazyEnumerationError(Exception):
    """Raised when attempting to iterate a LazySet that can't be enumerated"""


# Generic LazySet type
S = TypeVar("S", bound="_BaseLazySet[Any]")


# Generic element type
T = TypeVar("T")


class _BaseLazySet(Generic[T]):
    """Common functionality for LazySet and FrozenLazySet"""

    _set_type: ClassVar[Union[Type[Set[Any]], Type[FrozenSet[Any]]]] = set

    _elems: Union[Set[T], FrozenSet[T], _AllElems]

    _exclude: Union[Set[T], FrozenSet[T]]

    def __init__(
        self,
        elems: Optional[
            Union[_BaseLazySet[T], Set[T], FrozenSet[T], Iterable[T], _AllElems]
        ] = None,
        exclude: Optional[Union[Set[T], FrozenSet[T], Iterable[T], _AllElems]] = None,
    ):
        if exclude is None:
            if isinstance(elems, _BaseLazySet):
                if elems._elems is AllElems:
                    self._elems = elems._elems
                elif isinstance(elems._elems, self._set_type):
                    self._elems = elems._elems.copy()
                else:
                    self._elems = self._set_type(elems._elems)
                if isinstance(elems._exclude, self._set_type):
                    self._exclude = elems._exclude.copy()
                else:
                    self._exclude = self._set_type(elems._exclude)
            else:
                self._exclude = self._set_type()
                if elems is None:
                    self._elems = self._set_type()
                else:
                    if elems is AllElems:
                        self._elems = elems
                    elif isinstance(elems, self._set_type):
                        self._elems = elems.copy()  # type: ignore
                    else:
                        self._elems = self._set_type(elems)
        else:
            if elems is not AllElems:
                raise ValueError(
                    "The 'elems' must be set to AllElems if " "'exclude' is not None"
                )
            if exclude is AllElems:
                self._elems = self._set_type()
                self._exclude = self._set_type()
            else:
                self._elems = elems
                if isinstance(exclude, self._set_type):
                    self._exclude = exclude.copy()  # type: ignore
                else:
                    self._exclude = self._set_type(exclude)

    def __contains__(self, elem: T) -> bool:
        if self._elems is AllElems:
            return elem not in self._exclude
        return elem in self._elems

    def __repr__(self) -> str:
        return f"{type(self)}({self._elems}, exclude={self._exclude})"

    def __str__(self) -> str:
        if self._elems is AllElems:
            res = "All Elements"
            if self._exclude:
                res += " excluding %s" % set(self._exclude)
        else:
            res = str(set(self._elems))
        return res

    def __and__(self: S, other: _BaseLazySet[T]) -> S:
        if other._elems is AllElems:
            assert other._exclude is not AllElems
            if self._elems is AllElems:
                return type(self)(AllElems, self._exclude | other._exclude)
            return type(self)(self._elems - other._exclude)
        elif self._elems is AllElems:
            return type(self)(other._elems - self._exclude)
        return type(self)(self._elems & other._elems)

    def __or__(self: S, other: _BaseLazySet[T]) -> S:
        if other._elems is AllElems:
            if self._elems is AllElems:
                return type(self)(AllElems, self._exclude & other._exclude)
            return type(self)(AllElems, other._exclude - self._elems)
        elif self._elems is AllElems:
            return type(self)(AllElems, self._exclude - other._elems)
        return type(self)(self._elems | other._elems)

    def __sub__(self: S, other: _BaseLazySet[T]) -> S:
        if other._elems is AllElems:
            if self._elems is AllElems:
                return type(self)(other._exclude - self._exclude)
            return type(self)(self._elems & other._exclude)
        elif self._elems is AllElems:
            return type(self)(AllElems, self._exclude | other._elems)
        return type(self)(self._elems - other._elems)

    def __bool__(self) -> bool:
        return self._elems is AllElems or len(self._elems) != 0

    def __iter__(self) -> Iterator[T]:
        if self._elems is AllElems:
            raise LazyEnumerationError
        for elem in self._elems:
            yield elem

    def __len__(self) -> int:
        if self._elems is AllElems:
            raise LazyEnumerationError
        return len(self._elems)

    def __eq__(self, other: object) -> bool:
        o_elems = getattr(other, "_elems", None)
        o_exclude = getattr(other, "_exclude", None)
        return self._elems == o_elems and self._exclude == o_exclude

    def excludes(self) -> Iterator[T]:
        for elem in self._exclude:
            yield elem

    def is_enumerable(self) -> bool:
        return self._elems is not AllElems

    def collides(self, other: _BaseLazySet[T]) -> bool:
        return not (self & other)


class LazySet(_BaseLazySet[T]):
    """Set like object that can contain all elements without enumerating them

    Can also represent exclusive sets (everything except...)
    """

    _set_type = set

    _elems: Union[Set[T], _AllElems]

    _exclude: Set[T]

    def add(self, elem: T) -> None:
        if self._elems is AllElems:
            self._exclude.discard(elem)
        else:
            self._elems.add(elem)

    def remove(self, elem: T) -> None:
        if self._elems is AllElems:
            if elem in self._exclude:
                raise KeyError(elem)
            self._exclude.add(elem)
        else:
            self._elems.remove(elem)

    def discard(self, elem: T) -> None:
        if self._elems is AllElems:
            self._exclude.add(elem)
        else:
            self._elems.discard(elem)

    def __iand__(self, other: _BaseLazySet[T]) -> LazySet[T]:
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._exclude |= other._exclude
            else:
                assert self._elems is not AllElems
                self._elems -= other._exclude
        elif self._elems is AllElems:
            self._elems = self._set_type(other._elems) - self._exclude
            self._exclude = set()
        else:
            self._elems &= other._elems
        return self

    def __ior__(self, other: _BaseLazySet[T]) -> LazySet[T]:
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._exclude = self._exclude & other._exclude
            else:
                self._exclude = self._set_type(other._exclude) - self._elems
                self._elems = AllElems
        elif self._elems is AllElems:
            self._exclude -= other._elems
        else:
            self._elems |= other._elems
        return self

    def __isub__(self, other: _BaseLazySet[T]) -> LazySet[T]:
        if other._elems is AllElems:
            if self._elems is AllElems:
                self._elems = self._set_type(other._exclude) - self._exclude
                self._exclude = self._set_type()
            else:
                self._elems &= other._exclude
        elif self._elems is AllElems:
            self._exclude |= other._elems
        else:
            self._elems -= other._elems
        return self


class FrozenLazySet(_BaseLazySet[T]):
    """Frozen LazySet"""

    _set_type = frozenset

    def __hash__(self) -> int:
        return hash((self._elems, self._exclude))
