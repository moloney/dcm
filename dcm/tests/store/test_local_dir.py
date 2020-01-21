import os
from pathlib import Path
from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory
from glob import glob

import pydicom

from pytest import mark

from ...store.local_dir import LocalDir

from ..conftest import dicom_dir, dicom_files


@mark.asyncio
async def test_gen_chunks(make_local_dir):
    local_dir, init_qr, _ = make_local_dir('all', max_chunk=2)
    n_dcm_gen = 0
    async for chunk in local_dir.gen_chunks():
        async for dcm in chunk.gen_data():
            print(dcm)
            n_dcm_gen += 1
    assert n_dcm_gen == len(init_qr)


@mark.asyncio
async def test_send(dicom_files):
    with TemporaryDirectory() as tmp_dir:
        local_dir = LocalDir(tmp_dir)
        async with local_dir.send() as send_q:
            for dcm_path in dicom_files:
                dcm = pydicom.dcmread(str(dcm_path))
                await send_q.put(dcm)
        n_files = len(glob(tmp_dir + '/**/*.dcm', recursive=True))
        assert n_files == len(dicom_files)
