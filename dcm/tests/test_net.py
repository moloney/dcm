import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory

import pytest
from pytest import fixture, mark

from ..query import QueryLevel
from ..net import DcmNode, LocalEntity, RetrieveReport

from .conftest import (has_dcmtk, DCMTK_VERSION, dcmtk_priv_sop_retr_xfail,
                       dcmtk_priv_sop_send_xfail)


test_query_subsets = ['all',
                      'PATIENT-1',
                      'PATIENT-0/STUDY-0',
                      'PATIENT-0/STUDY-0/SERIES-0',
                      'PATIENT-0/STUDY-0/SERIES-0/IMAGE-0',
                      'PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1',
                      'PATIENT-0/STUDY-1/SERIES-2',
                      None]


test_retr_subsets = [pytest.param('all', marks=dcmtk_priv_sop_retr_xfail),
                     'PATIENT-1',
                     'PATIENT-0/STUDY-0',
                     'PATIENT-0/STUDY-0/SERIES-0',
                     'PATIENT-0/STUDY-0/SERIES-0/IMAGE-0',
                     'PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1',
                     pytest.param('PATIENT-0/STUDY-1/SERIES-2', marks=dcmtk_priv_sop_retr_xfail),
                     None]


test_send_subsets = [pytest.param('all', marks=dcmtk_priv_sop_send_xfail),
                     'PATIENT-1',
                     'PATIENT-0/STUDY-0',
                     'PATIENT-0/STUDY-0/SERIES-0',
                     'PATIENT-0/STUDY-0/SERIES-0/IMAGE-0',
                     'PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1',
                     pytest.param('PATIENT-0/STUDY-1/SERIES-2', marks=dcmtk_priv_sop_send_xfail),
                     None]


@has_dcmtk
def test_echo(make_local_node, make_dcmtk_nodes):
    local_node = make_local_node()
    remote, _, _ = make_dcmtk_nodes([local_node], None)
    local = LocalEntity(local_node)
    result = asyncio.run(local.echo(remote))
    assert result


@mark.parametrize("subset", test_query_subsets)
@has_dcmtk
def test_query_all(make_local_node, make_dcmtk_nodes, subset):
    local_node = make_local_node()
    remote, full_qr, _ = make_dcmtk_nodes([local_node], subset)
    local = LocalEntity(local_node)
    result_qr = asyncio.run(local.query(remote, level=QueryLevel.IMAGE))
    assert result_qr == full_qr


@mark.parametrize("subset", test_query_subsets)
@has_dcmtk
def test_query_subset(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote, full_qr, _ = make_dcmtk_nodes([local_node], 'all')
    req_qr, req_data = get_dicom_subset(subset)
    local = LocalEntity(local_node)
    result_qr = asyncio.run(local.query(remote, query_res=req_qr, level=QueryLevel.IMAGE))
    assert result_qr == req_qr

@mark.parametrize("subset", test_retr_subsets)
@has_dcmtk
def test_download(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote, init_qr, _ = make_dcmtk_nodes([local_node], 'all')
    local = LocalEntity(local_node)
    ret_qr, ret_data = get_dicom_subset(subset)
    with TemporaryDirectory() as dest_dir:
        dest_dir = Path(dest_dir)
        dl_files = asyncio.run(local.download(remote, ret_qr, dest_dir))
        found_files = [x for x in dest_dir.glob('**/*.dcm')]
        assert len(dl_files) == len(found_files)
        assert len(dl_files) == ret_qr.n_instances()


@mark.parametrize("subset", test_send_subsets)
@has_dcmtk
def test_upload(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote, _, store_dir = make_dcmtk_nodes([local_node], None)
    local = LocalEntity(local_node)
    send_qr, send_data = get_dicom_subset(subset)
    dcm_files = [d[0] for d in send_data]
    asyncio.run(local.upload(remote, dcm_files))
    store_dir = Path(store_dir)
    stored_files = [x for x in store_dir.glob('**/*.dcm')]
    assert len(stored_files) == len(send_data)


@mark.parametrize('subset', test_retr_subsets)
@has_dcmtk
@mark.asyncio
async def test_retrieve(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote, init_qr, _ = make_dcmtk_nodes([local_node], 'all')
    ret_qr, ret_data = get_dicom_subset(subset)
    res = []
    async for ds in local.retrieve(remote, ret_qr):
        res.append(ds)
    assert len(res) == len(ret_data)


@mark.parametrize('subset', test_retr_subsets)
@has_dcmtk
@mark.asyncio
async def test_interleaved_retrieve(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote1, init_qr1, _ = make_dcmtk_nodes([local_node], 'all')
    remote2, init_qr2, _ = make_dcmtk_nodes([local_node], 'all')
    ret_qr, ret_data = get_dicom_subset(subset)
    r_gen1 = local.retrieve(remote1, ret_qr)
    r_gen2 = local.retrieve(remote2, ret_qr)
    r1_files = []
    r2_files = []
    r1_done = r2_done = False
    r1_error = r2_error = False
    while not (r1_done and r2_done):
        if not r1_done:
            try:
                ds = await r_gen1.__anext__()
            except StopAsyncIteration:
                print("First retrieve is done")
                r1_done = True
            except Exception as e:
                print("Got error from first retrieve: %s" % e)
                r1_error = True
                r1_done = True
            else:
                r1_files.append(ds)
        if not r2_done:
            try:
                ds = await r_gen2.__anext__()
            except StopAsyncIteration:
                print("Second retrieve is done")
                r2_done = True
            except Exception as e:
                print("Got error from second retrieve: %s" % e)
                r1_error = True
                r1_done = True
            else:
                r2_files.append(ds)
    assert r1_error == False
    assert r2_error == False
    assert len(r1_files) == len(ret_qr)
    assert len(r2_files) == len(ret_qr)
    # TODO: Compare UIDs to make sure we got the right files


@mark.parametrize('subset', test_send_subsets)
@has_dcmtk
@mark.asyncio
async def test_send(make_local_node, make_dcmtk_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote, _, store_dir = make_dcmtk_nodes([local_node], None)
    send_qr, send_data = get_dicom_subset(subset)
    async with local.send(remote) as send_q:
        for send_path, send_ds in send_data:
            await send_q.put(send_ds)
    store_dir = Path(store_dir)
    stored_files = [x for x in store_dir.glob('**/*.dcm')]
    assert len(stored_files) == len(send_data)
