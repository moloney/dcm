'''Tests for the dcm.conf module'''
from pytest import fixture, mark

from ..conf import _default_conf, DcmConfig

def test_load_default():
    # TODO: Our new API requires path input...
