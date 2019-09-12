import os
from pathlib import Path
from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory
from glob import glob

import pydicom

from pytest import mark

from ...store.net_repo import NetRepo

from ..conftest import has_dcmtk, local_nodes


@has_dcmtk
@mark.asyncio
@mark.parametrize('dcmtk_test_nodes', [['all']], indirect=True)
async def test_gen_chunks(dcmtk_test_nodes):
    src, init_qr, _ = dcmtk_test_nodes[0]
    local_dir = NetRepo(local_nodes[0], src)
    n_dcm_gen = 0
    async for chunk in local_dir.gen_chunks():
        print(chunk.qr)
        async for dcm in chunk.gen_data():
            print(dcm)
            n_dcm_gen += 1
    assert n_dcm_gen == len(init_qr)


#@mark.asyncio
#async def test_send(dicom_files):
#    with TemporaryDirectory() as tmp_dir:
#        local_dir = LocalDir(tmp_dir)
#        async with local_dir.send() as send_q:
#            for dcm_path in dicom_files:
#                dcm = pydicom.dcmread(str(dcm_path))
#                await send_q.put(dcm)
#        n_files = len(glob(tmp_dir + '/**/*.dcm', recursive=True))
#        assert n_files == len(dicom_files)
