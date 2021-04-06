import pydicom

from pytest import mark

from ...query import QueryLevel
from ..conftest import has_dcmtk
from ..test_net import test_retr_subsets, test_send_subsets, test_query_subsets


@has_dcmtk
@mark.asyncio
@mark.parametrize("subset", test_retr_subsets)
async def test_gen_chunks(make_dcmtk_net_repo, subset):
    net_repo, init_qr, _ = make_dcmtk_net_repo(subset=subset)
    n_dcm_gen = 0
    async for chunk in net_repo.gen_chunks():
        async for dcm in chunk.gen_data():
            n_dcm_gen += 1
            assert dcm in init_qr
    assert n_dcm_gen == len(init_qr)


@has_dcmtk
@mark.asyncio
@mark.parametrize("subset", test_send_subsets)
async def test_send(make_dcmtk_net_repo, get_dicom_subset, subset):
    net_repo, _, store_dir = make_dcmtk_net_repo(subset=None)
    send_qr, send_data = get_dicom_subset(subset)
    async with net_repo.send() as send_q:
        for send_path, send_ds in send_data:
            await send_q.put(send_ds)
    n_files = len(list(store_dir.glob("**/*.dcm")))
    assert n_files == len(send_data)


@has_dcmtk
@mark.asyncio
@mark.parametrize("subset", test_query_subsets)
async def test_query(make_dcmtk_net_repo, get_dicom_subset, subset):
    net_repo, init_qr, _ = make_dcmtk_net_repo(subset="all")
    req_qr, req_data = get_dicom_subset(subset)
    res_qr = await net_repo.query(query_res=req_qr, level=QueryLevel.IMAGE)
    assert req_qr == res_qr
