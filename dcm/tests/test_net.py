import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory

from pytest import fixture, mark

from ..query import QueryLevel
from ..net import DcmNode, LocalEntity

from .conftest import has_dcmtk


@has_dcmtk
def test_echo(make_local_node, make_dcmtk_nodes):
    local_node = make_local_node()
    remote, _, _ = make_dcmtk_nodes([local_node], None)
    local = LocalEntity(local_node)
    result = asyncio.run(local.echo(remote))
    assert result


@has_dcmtk
def test_query(make_local_node, make_dcmtk_nodes):
    local_node = make_local_node()
    remote, init_qr, _ = make_dcmtk_nodes([local_node], 'all')
    local = LocalEntity(local_node)
    remote_qr = asyncio.run(local.query(remote, level=QueryLevel.IMAGE))
    assert len(remote_qr) != 0
    assert remote_qr == init_qr


@has_dcmtk
def test_download(make_local_node, make_dcmtk_nodes):
    local_node = make_local_node()
    remote, init_qr, _ = make_dcmtk_nodes([local_node], 'all')
    local = LocalEntity(local_node)
    qr = asyncio.run(local.query(remote))
    with TemporaryDirectory() as dest_dir:
        dest_dir = Path(dest_dir)
        dl_files = asyncio.run(local.download(remote, qr, dest_dir))
        found_files = [x for x in dest_dir.glob('**/*.dcm')]
        assert len(dl_files) == len(found_files)
        assert len(dl_files) == len(init_qr)


@has_dcmtk
def test_upload(make_local_node, make_dcmtk_nodes, dicom_files):
    local_node = make_local_node()
    remote, _, store_dir = make_dcmtk_nodes([local_node], None)
    local = LocalEntity(local_node)
    asyncio.run(local.upload(remote, dicom_files))
    store_dir = Path(store_dir)
    stored_files = [x for x in store_dir.glob('**/*.dcm')]
    assert len(stored_files) == len(dicom_files)


@mark.parametrize('subset_spec',
                  ['all',
                   None,
                   'PATIENT-0/STUDY-0',
                   'PATIENT-0/STUDY-0/SERIES-0',
                   'PATIENT-0/STUDY-0/SERIES-0/IMAGE-0'])
@has_dcmtk
@mark.asyncio
async def test_retrieve(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset_spec):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote, init_qr, _ = make_dcmtk_nodes([local_node], 'all')
    ret_qr, ret_data = get_dicom_subset(subset_spec)
    res = []
    async for ds in local.retrieve(remote, ret_qr):
        res.append(ds)
    assert len(res) == len(ret_data)


@has_dcmtk
@mark.asyncio
async def test_interleaved_retrieve(make_local_node, make_dcmtk_nodes):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote1, init_qr1, _ = make_dcmtk_nodes([local_node], 'all')
    remote2, init_qr2, _ = make_dcmtk_nodes([local_node], 'all')
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
