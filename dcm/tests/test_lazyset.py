import operator
from copy import deepcopy

import pytest

from ..lazyset import LazySet, FrozenLazySet, AllElems, LazyEnumerationError

_ops = {
    "&": (operator.and_, operator.iand),
    "|": (operator.or_, operator.ior),
    "-": (operator.sub, operator.isub),
}


def check_op(l, op, r, expected, check_inplace=True):
    op_func, iop_func = _ops[op]
    assert op_func(l, r) == expected
    if check_inplace:
        l = deepcopy(l)
        iop_func(l, r)
        assert l == expected


def test_lazyset_inclusive():
    s = LazySet()
    assert not bool(s)
    assert s.is_enumerable()
    elems = [1, None, "abc", (1, 2, 3)]
    for elem in elems:
        assert elem not in s
        s.add(elem)
        assert elem in s
    for val in s:
        assert val in elems
    assert len(list(s.excludes())) == 0
    assert bool(s)
    assert len(s) == len(elems)
    assert s == LazySet(set(elems))
    assert s == LazySet(frozenset(elems))
    assert s == LazySet(s)
    for elem in elems:
        s.add(elem)
    assert len(s) == len(elems)
    s2 = deepcopy(s)
    for elem in elems:
        s2.remove(elem)
        assert elem not in s2
        with pytest.raises(KeyError):
            s2.remove(elem)
    assert len(s2) == 0
    s2 = deepcopy(s)
    for elem in elems:
        s2.discard(elem)
        assert elem not in s2
        s2.discard(elem)
    assert len(s2) == 0
    s2 = LazySet(elems[:2])
    s3 = LazySet(elems[2:])
    check_op(s, "-", s2, s3)
    check_op(s, "-", s3, s2)
    check_op(s, "&", s2, s2)
    check_op(s, "&", s3, s3)
    check_op(s2, "&", s3, LazySet())
    check_op(s, "|", s2, s)
    check_op(s, "|", s3, s)
    check_op(s2, "|", s3, s)


def test_lazyset_exclusive():
    s = LazySet(AllElems)
    assert bool(s)
    assert not s.is_enumerable()
    elems = [1, None, "abc", (1, 2, 3)]
    for elem in elems:
        assert elem in s
        s.remove(elem)
        assert not elem in s
        with pytest.raises(KeyError):
            s.remove(elem)
    assert len(list(s.excludes())) == len(elems)
    assert s == LazySet(AllElems, set(elems))
    assert s == LazySet(AllElems, frozenset(elems))
    assert s == LazySet(s)
    s2 = LazySet(AllElems)
    for elem in elems:
        assert elem in s2
        s2.discard(elem)
        assert elem not in s2
        s2.discard(elem)
        assert elem not in s2
    assert s == s2
    for elem in elems:
        s2.add(elem)
        assert elem in s2
    assert len(list(s2.excludes())) == 0
    assert s != s2
    s2 = LazySet(AllElems, elems[:2])
    s3 = LazySet(AllElems, elems[2:])
    check_op(s, "-", s2, LazySet())
    check_op(s2, "-", s, LazySet(elems[2:]))
    check_op(s, "-", s3, LazySet())
    check_op(s3, "-", s, LazySet(elems[:2]))
    check_op(s, "&", s2, s)
    check_op(s, "&", s3, s)
    check_op(s2, "&", s3, s)
    check_op(s, "|", s2, s2)
    check_op(s, "|", s3, s3)
    check_op(s2, "|", s3, LazySet(AllElems))
