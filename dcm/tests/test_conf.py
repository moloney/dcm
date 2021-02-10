'''Tests for the dcm.conf module'''
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import fixture, mark

from ..conf import _default_conf, DcmConfig
from ..route import TransferMethod
from ..lazyset import FrozenLazySet

from .conftest import has_dcmtk

def test_load_default(make_dcm_config):
    config = make_dcm_config()
    

def test_uncommented_default(make_dcm_config):
    # Load uncommented version of default config str
    contents = []
    for line in _default_conf.split('\n'):
        if line != '' and line[0] == '#':
            contents.append(line[1:])
        else:
            contents.append(line)
    contents = '\n'.join(contents)
    config = make_dcm_config(contents)

    assert config.default_local.ae_title == 'YOURAE'
    assert config.get_local_node('other_local').ae_title == 'MYOTHERAE'
    assert config.get_remote_node('yourpacs').ae_title == 'PACSAETITLE'
    assert config.get_local_dir('dicom_dir').root_path == Path("~/dicom").expanduser()
    movescu_pacs = config.get_route('movescu_pacs')
    assert movescu_pacs.methods == (TransferMethod.REMOTE_COPY,)
    filt_mypacs = config.get_route('filt_mypacs')
    assert filt_mypacs.filt is not None
    dyn_route = config.get_route('dyn_route')
    assert dyn_route.required_elems == FrozenLazySet(['DeviceSerialNumber'])