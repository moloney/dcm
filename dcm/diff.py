'''Logic for comparing DICOM data sets and data sources'''

from copy import deepcopy
from hashlib import sha256
from typing import Optional, List

from pydicom import Dataset, DataElement
from pydicom.tag import Tag


def _shorten_bytes(val: bytes) -> bytes:
    if len(val) > 16:
        return (b'*%d bytes, hash = %s*' %
                (len(val), sha256(val).hexdigest().encode())
               )
    return val


class DataDiff(object):

    default_elem_fmt = '{elem.tag} {elem.name: <35} {elem.VR}: {value}'

    def __init__(self,
                 tag: Tag,
                 l_elem: Optional[DataElement],
                 r_elem: Optional[DataElement],
                 elem_fmt: str = default_elem_fmt):
        self.tag = tag
        self.l_elem = deepcopy(l_elem)
        self.r_elem = deepcopy(r_elem)
        self.elem_fmt = elem_fmt

    def _format_elem(self, elem: DataElement) -> str:
        value = elem.value
        if isinstance(value, bytes):
            value = _shorten_bytes(value)
        return self.elem_fmt.format(elem=elem, value=value)

    def __str__(self) -> str:
        res = []
        if self.l_elem is not None:
            res.append('< %s' % self._format_elem(self.l_elem))
        if self.r_elem is not None:
            res.append('> %s' % self._format_elem(self.r_elem))
        return '\n'.join(res)


def diff_data_sets(left: Dataset, right: Dataset) -> List[DataDiff]:
    '''Get list of all differences between `left` and `right` data sets'''
    l_elems = iter(left)
    r_elems = iter(right)
    l_elem = r_elem = None
    l_done = r_done = False
    diffs = []
    while True:
        if l_elem is None and not l_done:
            try:
                l_elem = next(l_elems)
            except StopIteration:
                l_done = True
                l_elem = None
        if r_elem is None and not r_done:
            try:
                r_elem = next(r_elems)
            except StopIteration:
                r_done = True
                r_elem = None
        if l_elem is None and r_elem is None:
            break
        if l_elem is None:
            assert r_elem is not None
            diffs.append(DataDiff(r_elem.tag, l_elem, r_elem))
            r_elem = None
        elif r_elem is None:
            assert l_elem is not None
            diffs.append(DataDiff(l_elem.tag, l_elem, r_elem))
            l_elem = None
        elif l_elem.tag < r_elem.tag:
            diffs.append(DataDiff(l_elem.tag, l_elem, None))
            l_elem = None
        elif r_elem.tag < l_elem.tag:
            diffs.append(DataDiff(r_elem.tag, None, r_elem))
            r_elem = None
        else:
            if l_elem.value != r_elem.value or l_elem.VR != r_elem.VR:
                diffs.append(DataDiff(l_elem.tag, l_elem, r_elem))
            l_elem = r_elem = None
    return diffs

