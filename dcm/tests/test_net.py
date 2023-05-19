import asyncio
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pytest
from pytest import mark
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
)
from ..reports.net_report import RetrieveReport
from ..util import aclosing
from ..query import QueryLevel
from ..net import LocalEntity, DcmNode

from .conftest import (
    has_dcmtk,
    dcmtk_priv_sop_retr_xfail,
    dcmtk_priv_sop_send_xfail,
    pnd_priv_sop_xfail,
    get_stored_files,
)


test_query_subsets = [
    "all",
    "PATIENT-1",
    "PATIENT-0/STUDY-0",
    "PATIENT-0/STUDY-0/SERIES-0",
    "PATIENT-0/STUDY-0/SERIES-0/IMAGE-0",
    "PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1",
    "PATIENT-0/STUDY-1/SERIES-2",
    None,
]


test_retr_subsets = [
    "all",
    "PATIENT-1",
    "PATIENT-0/STUDY-0",
    "PATIENT-0/STUDY-0/SERIES-0",
    "PATIENT-0/STUDY-0/SERIES-0/IMAGE-0",
    "PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1",
    "PATIENT-0/STUDY-1/SERIES-2",
    None,
]


def get_retr_subsets():
    res = []
    for node_type in ("dcmtk", "pnd"):
        if node_type == "dcmtk":
            for sub in test_retr_subsets:
                marks = [has_dcmtk]
                if sub in ("all", "PATIENT-0/STUDY-1/SERIES-2"):
                    marks.append(dcmtk_priv_sop_retr_xfail)
                res.append(pytest.param(node_type, sub, marks=marks))
        else:
            assert node_type == "pnd"
            for sub in test_retr_subsets:
                if sub in ("all", "PATIENT-0/STUDY-1/SERIES-2"):
                    res.append(
                        pytest.param(
                            node_type,
                            sub,
                            marks=pnd_priv_sop_xfail,
                        )
                    )
                else:
                    res.append((node_type, sub))
    return res


test_send_subsets = [
    "all",
    "PATIENT-1",
    "PATIENT-0/STUDY-0",
    "PATIENT-0/STUDY-0/SERIES-0",
    "PATIENT-0/STUDY-0/SERIES-0/IMAGE-0",
    "PATIENT-0/STUDY-0;PATIENT-0/STUDY-1/SERIES-0;PATIENT-0/STUDY-1/SERIES-1;PATIENT-1",
    "PATIENT-0/STUDY-1/SERIES-2",
    None,
]


def get_send_subsets():
    res = []
    for node_type in ("dcmtk", "pnd"):
        if node_type == "dcmtk":
            for sub in test_retr_subsets:
                marks = [has_dcmtk]
                if sub in ("all", "PATIENT-0/STUDY-1/SERIES-2"):
                    marks.append(dcmtk_priv_sop_send_xfail)
                res.append(pytest.param(node_type, sub, marks=marks))
        else:
            assert node_type == "pnd"
            for sub in test_retr_subsets:
                if sub in ("all", "PATIENT-0/STUDY-1/SERIES-2"):
                    res.append(
                        pytest.param(
                            node_type,
                            sub,
                            marks=pnd_priv_sop_xfail,
                        )
                    )
                else:
                    res.append((node_type, sub))
    return res


@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_echo(make_local_node, make_remote_nodes):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], None)
    local = LocalEntity(local_node)
    result = asyncio.run(local.echo(remote.dcm_node))
    assert result


@mark.parametrize("subset", test_query_subsets)
@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_query_all(make_local_node, make_remote_nodes, subset):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], subset)
    local = LocalEntity(local_node)
    result_qr = asyncio.run(local.query(remote.dcm_node, level=QueryLevel.IMAGE))
    assert result_qr == remote.init_qr


@mark.parametrize("subset", test_query_subsets)
@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_query_subset(make_local_node, make_remote_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], "all")
    req_qr, _ = get_dicom_subset(subset)
    req_qr = req_qr & remote.init_qr
    local = LocalEntity(local_node)
    result_qr = asyncio.run(
        local.query(remote.dcm_node, query_res=req_qr, level=QueryLevel.IMAGE)
    )
    assert result_qr == req_qr


@mark.parametrize("node_type, subset", get_retr_subsets())
def test_download(make_local_node, make_remote_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], "all")
    local = LocalEntity(local_node)
    ret_qr, ret_data = get_dicom_subset(subset)
    with TemporaryDirectory() as dest_dir:
        dest_dir = Path(dest_dir)
        dl_files = asyncio.run(
            local.download(remote.dcm_node, ret_qr, dest_dir), debug=True
        )
        found_files = [x for x in dest_dir.glob("**/*.dcm")]
        assert len(dl_files) == len(found_files)
        assert len(dl_files) == ret_qr.n_instances()


@mark.parametrize("node_type, subset", get_send_subsets())
def test_upload(make_local_node, make_remote_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], None)
    local = LocalEntity(local_node)
    send_qr, send_data = get_dicom_subset(subset)
    dcm_files = [d[0] for d in send_data]
    asyncio.run(local.upload(remote.dcm_node, dcm_files))
    stored_files = get_stored_files(remote.store_dir)
    assert len(stored_files) == len(send_data)


@mark.parametrize("node_type, subset", get_retr_subsets())
@mark.asyncio
async def test_retrieve(make_local_node, make_remote_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote = make_remote_nodes([local_node], "all")
    ret_qr, ret_data = get_dicom_subset(subset)
    res = []
    async for ds in local.retrieve(remote.dcm_node, ret_qr):
        res.append(ds)
    assert len(res) == len(ret_data)


@mark.slow
@mark.asyncio
async def test_cancel_retrieve(make_local_node, make_pnd_nodes):
    """Make sure we send C-CANCEL when closing 'retrieve' generator"""
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote = make_pnd_nodes([local_node], "all")
    report = RetrieveReport()
    # Prematurely closing async generator should result in C-CANCEL being sent
    async with aclosing(
        local.retrieve(remote.dcm_node, remote.init_qr, report)
    ) as ret_gen:
        async for ds in ret_gen:
            break
    print("About to finalize remote node")
    stdout, stderr = remote.finalize()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    print("****")
    print(stdout)
    print("****")
    print(stderr)
    assert re.search(
        "(C-CANCEL|SubOperationsTerminatedDueToCancelIndication)",
        stderr,
        re.MULTILINE,
    )


@mark.parametrize("node_type, subset", get_retr_subsets())
@mark.asyncio
async def test_interleaved_retrieve(
    make_local_node, make_remote_nodes, get_dicom_subset, subset
):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote1 = make_remote_nodes([local_node], "all")
    remote2 = make_remote_nodes([local_node], "all")
    ret_qr, ret_data = get_dicom_subset(subset)
    r_gen1 = local.retrieve(remote1.dcm_node, ret_qr)
    r_gen2 = local.retrieve(remote2.dcm_node, ret_qr)
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


@mark.parametrize("node_type, subset", get_send_subsets())
@mark.asyncio
async def test_send(make_local_node, make_remote_nodes, get_dicom_subset, subset):
    local_node = make_local_node()
    local = LocalEntity(local_node)
    remote = make_remote_nodes([local_node], None)
    send_qr, send_data = get_dicom_subset(subset)
    async with local.send(remote.dcm_node) as send_q:
        for send_path, send_ds in send_data:
            await send_q.put(send_ds)
    stored_files = get_stored_files(remote.store_dir)
    assert len(stored_files) == len(send_data)
