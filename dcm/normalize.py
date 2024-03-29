"""Convert DICOM data sets to just dict/list/tuple/str/float/int

Results can be easily serialized and the goal is to be fully reversable, but
that is not implemented yet
"""
from collections import OrderedDict
from base64 import b64encode

from pydicom.datadict import keyword_for_tag


def bytes_or_text(val):
    if isinstance(val, str):
        return val
    if all(ord(b" ") <= c <= ord(b"~") for c in val):
        return str(val)
    return (b64encode(val).decode(),)


def convert_or_none(converter, val):
    try:
        res = converter(val)
    except Exception:
        res = None
    return res


def norm_elem_val(elem):
    val = elem.value
    converter = default_conversions.get(elem.VR)
    if converter:
        if elem.VM > 1 or elem.VR == "SQ":
            val = [convert_or_none(converter, v) for v in val]
        else:
            val = convert_or_none(converter, val)
    return val


def make_elem_filter(include_elems):
    def filt(tag, keyword):
        if keyword in include_elems:
            return True
        return False

    return filt


# TODO: While this "elem_filter" is quite generic, it is pretty inefficient when we
#       just want a handful of elements from a large data set. Should consider an
#       alternative approach.
def normalize(data_set, elem_filter=None):
    """Convert a DICOM data set into basic python types that can be serialized"""
    res = OrderedDict()
    for tag in data_set.keys():
        key = keyword_for_tag(tag)
        if elem_filter and not elem_filter(tag, key):
            continue
        if key == "":
            key = "%04x,%04x" % (tag.group, tag.element)
        res[key] = norm_elem_val(data_set[tag])
    return res


default_conversions = {
    "PN": str,
    "UI": str,
    "CS": str,
    "DS": str,
    "IS": int,
    "AT": tuple,
    "OW": bytes_or_text,
    "OB": bytes_or_text,
    "OW or OB": bytes_or_text,
    "OB or OW": bytes_or_text,
    "UN": bytes_or_text,
    "SQ": normalize,
}
