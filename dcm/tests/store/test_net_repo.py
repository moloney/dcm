import pytest
from pytest import mark

from ...query import QueryLevel
from ..conftest import has_dcmtk, get_stored_files
from ..test_net import get_retr_subsets, get_send_subsets, test_query_subsets


@mark.asyncio
@mark.parametrize("node_type, subset", get_retr_subsets())
async def test_gen_chunks(make_net_repo, subset):
    net_repo, init_qr, _ = make_net_repo(subset=subset)
    n_dcm_gen = 0
    async for chunk in net_repo.gen_chunks():
        async for dcm in chunk.gen_data():
            n_dcm_gen += 1
            assert dcm in init_qr
    assert n_dcm_gen == len(init_qr)


@mark.asyncio
@mark.parametrize("node_type, subset", get_send_subsets())
async def test_send(make_net_repo, get_dicom_subset, subset):
    net_repo, _, store_dir = make_net_repo(subset=None)
    send_qr, send_data = get_dicom_subset(subset)
    async with net_repo.send() as send_q:
        for send_path, send_ds in send_data:
            await send_q.put(send_ds)
    n_files = len(get_stored_files(store_dir))
    assert n_files == len(send_data)


@mark.asyncio
@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
@mark.parametrize("subset", test_query_subsets)
async def test_query(make_net_repo, get_dicom_subset, subset):
    net_repo, init_qr, _ = make_net_repo(subset="all")
    req_qr, _ = get_dicom_subset(subset)
    req_qr = req_qr & init_qr
    res_qr = await net_repo.query(query_res=req_qr, level=QueryLevel.IMAGE)
    assert req_qr == res_qr
