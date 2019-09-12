import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory

from pytest import fixture, mark

from ..query import QueryLevel
from ..net import DcmNode, LocalEntity

from .conftest import (local_nodes, dicom_files, dicom_files_w_qr, has_dcmtk, dcmtk_test_nodes, DATA_DIR)


@has_dcmtk
def test_echo(dcmtk_test_nodes):
    remote, _, _ = dcmtk_test_nodes[0]
    local = LocalEntity(local_nodes[0])
    result = asyncio.run(local.echo(remote))
    assert result


@has_dcmtk
@mark.parametrize('dcmtk_test_nodes', [['all']], indirect=True)
def test_query(dcmtk_test_nodes):
    remote, init_qr, _ = dcmtk_test_nodes[0]
    local = LocalEntity(local_nodes[0])
    remote_qr = asyncio.run(local.query(remote, level=QueryLevel.IMAGE))
    assert len(remote_qr) != 0
    assert remote_qr == init_qr


@has_dcmtk
@mark.parametrize('dcmtk_test_nodes', [['all']], indirect=True)
def test_download(dcmtk_test_nodes):
    remote, init_qr, _ = dcmtk_test_nodes[0]
    local = LocalEntity(local_nodes[0])
    qr = asyncio.run(local.query(remote))
    with TemporaryDirectory() as dest_dir:
        dest_dir = Path(dest_dir)
        dl_files = asyncio.run(local.download(remote, qr, dest_dir))
        found_files = [x for x in dest_dir.glob('**/*.dcm')]
        assert len(dl_files) == len(found_files)
        assert len(dl_files) == len(init_qr)


@has_dcmtk
def test_upload(dcmtk_test_nodes, dicom_files):
    remote, _, store_dir = dcmtk_test_nodes[0]
    local = LocalEntity(local_nodes[0])
    asyncio.run(local.upload(remote, dicom_files))
    store_dir = Path(store_dir)
    stored_files = [x for x in store_dir.glob('**/*.dcm')]
    assert len(stored_files) == len(dicom_files)


@mark.asyncio
@mark.parametrize('dcmtk_test_nodes', [['all', 'all']], indirect=True)
async def test_interleaved_retrieve(dcmtk_test_nodes):
    local = LocalEntity(local_nodes[0])
    remote1, init_qr1, _ = dcmtk_test_nodes[0]
    remote2, init_qr2, _ = dcmtk_test_nodes[1]
    qr1 = await local.query(remote1)
    qr2 = await local.query(remote2)
    r_gen1 = local.retrieve(remote1, qr1)
    r_gen2 = local.retrieve(remote2, qr2)
    r1_files = []
    r2_files = []
    r1_done = r2_done = False
    while not (r1_done and r2_done):
        if not r1_done:
            try:
                ds = await r_gen1.__anext__()
            except StopAsyncIteration:
                r1_done = True
            else:
                r1_files.append(ds)
        if not r2_done:
            try:
                ds = await r_gen2.__anext__()
            except StopAsyncIteration:
                r2_done = True
            else:
                r2_files.append(ds)
    assert len(r1_files) == len(init_qr1)
    assert len(r2_files) == len(init_qr2)
    # TODO: Compare UIDs to make sure we got the right files
