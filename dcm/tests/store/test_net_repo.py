import pydicom

from pytest import mark

from ...query import QueryLevel
from ..conftest import has_dcmtk


@has_dcmtk
@mark.asyncio
@mark.parametrize('data_subset', [None, 'all'])
async def test_gen_chunks(make_dcmtk_net_repo):
    net_repo, init_qr, _ = make_dcmtk_net_repo(subset='all')
    n_dcm_gen = 0
    async for chunk in net_repo.gen_chunks():
        async for dcm in chunk.gen_data():
            n_dcm_gen += 1
    assert n_dcm_gen == len(init_qr)


@has_dcmtk
@mark.asyncio
async def test_send(dicom_files, make_dcmtk_net_repo):
    net_repo, _, store_dir = make_dcmtk_net_repo(subset=None)
    async with net_repo.send() as send_q:
        for dcm_path in dicom_files:
            dcm = pydicom.dcmread(str(dcm_path))
            await send_q.put(dcm)
    n_files = len(list(store_dir.glob('**/*.dcm')))
    assert n_files == len(dicom_files)


@has_dcmtk
@mark.asyncio
@mark.parametrize('data_subset', [None, 'all'])
async def test_query(make_dcmtk_net_repo):
    net_repo, init_qr, _ = make_dcmtk_net_repo(subset='all')
    qr = await net_repo.query(level=QueryLevel.IMAGE)
    assert init_qr == qr