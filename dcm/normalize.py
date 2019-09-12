'''Convert DICOM data sets to just dict/list/tuple/str/float/int

Results can be easily serialized and the goal is to be fully reversable, but
that is not implemented yet
'''
from collections import OrderedDict
from base64 import b64encode


def bytes_or_text(val):
    if isinstance(val, str):
        return val
    if all(ord(b' ') <= c <= ord(b'~') for c in val):
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
        if elem.VM > 1 or elem.VR == 'SQ':
            val = [convert_or_none(converter, v) for v in val]
        else:
            val = convert_or_none(converter, val)
    return val


def normalize(data_set, elem_filter=None):
    '''Convert a DICOM data set into basic python types that can be serialized
    '''
    res = OrderedDict()
    for elem in data_set:
        if elem_filter:
            elem = elem_filter(elem)
            if elem is None:
                continue
        key = elem.keyword
        if key == '':
            key = '%04x,%04x' % (elem.tag.group, elem.tag.element)
        res[key] = norm_elem_val(elem)
    return res


default_conversions = {'PN' : str,
                       'UI' : str,
                       'CS' : str,
                       'DS' : str,
                       'IS' : int,
                       'AT' : tuple,
                       'OW' : bytes_or_text,
                       'OB' : bytes_or_text,
                       'OW or OB' : bytes_or_text,
                       'OB or OW' : bytes_or_text,
                       'UN' : bytes_or_text,
                       'SQ' : normalize,
                      }

