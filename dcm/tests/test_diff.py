import random
from copy import deepcopy

from pytest import fixture, mark
from pydicom.dataset import Dataset

from .test_query import make_dataset

from ..diff import diff_data_sets


def test_diff_data_sets():
    ds1 = make_dataset()
    ds2 = deepcopy(ds1)
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 0
    ds2.PatientID = "Johnny Doe"
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 1
    assert diffs[0].tag == "PatientID"
    assert diffs[0].l_elem.value == ds1.PatientID
    assert diffs[0].r_elem.value == ds2.PatientID
    str_diff_lines = str(diffs[0]).split("\n")
    assert len(str_diff_lines) == 2
    assert str_diff_lines[0][0] == "<"
    assert str_diff_lines[1][0] == ">"
    ds1.PatientSex = "M"
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 2
    ds2.PatientAge = 30
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 3
    ds1.Signature = bytes(random.getrandbits(8) for _ in range(512))
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 4
    ds2.Signature = ds1.Signature
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 3
    ds2.Signature = bytes(random.getrandbits(8) for _ in range(512))
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 4
    ds2.EncryptedContent = bytes(random.getrandbits(8) for _ in range(1024))
    diffs = diff_data_sets(ds1, ds2)
    assert len(diffs) == 5
