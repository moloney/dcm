import os, logging, asyncio, re
from contextlib import asynccontextmanager
from functools import partial
from glob import iglob

import pydicom
from pydicom.dataset import Dataset, FileDataset
import janus

from . import DataChunk, DataBucket
from ..net import IMPLEMENTATION_UID


log = logging.getLogger(__name__)


read_f = partial(pydicom.dcmread, force=True)


class LocalDataChunk(DataChunk):
    '''Chunk of data from a local directory'''
    def __init__(self, files):
        self._files = files
        self._file_idx = {}

    def __repr__(self):
        return f'LocalDataChunk([{self._files[0]},...,{self._files[-1]}])'

    # TODO: Fill out report if requested
    async def gen_data(self):
        loop = asyncio.get_running_loop()
        for f in self._files:
            f = str(f)
            ds = await loop.run_in_executor(None, read_f, f)
            self._file_idx[ds.SOPInstanceUID] = f
            yield ds


    async def get_data_set(self, instance_uid, loop=None):
        if loop is None:
            loop = asyncio.get_running_loop()
        f = self._file_idx[instance_uid]
        return await loop.run_in_executor(None, read_f, f)


def _dir_crawl_worker(res_q, root_path, recurse=True, file_ext='dcm',
                      max_chunk=1000):
    curr_files = []
    glob_comps = [root_path]
    if recurse:
        glob_comps.append('**')
    if file_ext:
        glob_comps.append('*.%s' % file_ext)
    else:
        glob_comps.append('*')
    glob_exp = os.path.join(*glob_comps)
    for path in iglob(glob_exp, recursive=recurse):
        if not os.path.isfile(path):
            continue
        curr_files.append(path)
        if len(curr_files) == max_chunk:
            res_q.put(LocalDataChunk(curr_files))
            curr_files = []
    if len(curr_files) != 0:
        res_q.put(LocalDataChunk(curr_files))


class DefaultDicomWrapper:
    def __init__(self, ds, default='unknown'):
        self._ds = ds
        self._default = default

    def __getattr__(self, attr):
        return self._ds.get(attr, self._default)


def make_out_path(out_fmt, ds):
    out_toks = out_fmt.split(os.sep)
    dsw = DefaultDicomWrapper(ds)
    return os.sep.join([re.sub('[^A-Za-z0-9_.-]', '_', t.format(d=dsw))
                        for t in out_toks]
                      )


def _disk_write_worker(data_queue, root_path, out_fmt, force_overwrite=False):
    '''Take data sets from a queue and write to disk'''
    while True:
        log.debug("disk_writer is waiting on data")
        ds = data_queue.get()
        if ds is None:
            break
        log.debug("disk_writer thread got a data set")
        out_path = os.path.join(root_path,
                                make_out_path(out_fmt, ds))

        log.info('Storing DICOM file: %s' % out_path)
        if os.path.exists(out_path):
            if force_overwrite:
                log.info('File exists, overwriting: %s' %
                         out_path)
            else:
                log.warning('File exists, skipping: %s' %
                            out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Create meta-data section if needed
        if not isinstance(ds, FileDataset):
            meta = Dataset()
            meta.MediaStorageSOPClassUID = ds.SOPClassUID
            meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            meta.ImplementationClassUID = IMPLEMENTATION_UID

            # Save data set as a file
            file_ds = FileDataset(out_path,
                                  {},
                                  file_meta=meta,
                                  preamble=b"\0" * 128)
            file_ds.update(ds)
            ds = file_ds
            ds.is_little_endian = True
            ds.is_implicit_VR = True

        ds.save_as(out_path)


class LocalDir(DataBucket):
    '''Local directory of data without additional meta data'''

    is_local = True

    default_out_fmt = ('{d.PatientID}/'
                       '{d.StudyInstanceUID}/'
                       '{d.SeriesNumber:03d}-{d.SeriesDescription}/'
                       '{d.SOPInstanceUID}')
    '''Default format for output paths when saving data
    '''

    def __init__(self, path, recurse=True, file_ext='dcm',
                 max_chunk=1000, out_fmt=None, force_overwrite=False):
        self._root_path = path
        self._recurse = recurse
        self._max_chunk = max_chunk
        self._force_overwrite = force_overwrite
        if out_fmt is None:
            self._out_fmt = self.default_out_fmt
        else:
            self._out_fmt = out_fmt
        self._file_ext = file_ext
        if self._file_ext:
            self._out_fmt += '.%s' % file_ext

    async def gen_chunks(self):
        loop = asyncio.get_running_loop()
        res_q = janus.Queue(loop=loop)
        crawl_fut = loop.run_in_executor(None,
                                         partial(_dir_crawl_worker,
                                                 res_q.sync_q,
                                                 self._root_path,
                                                 self._recurse,
                                                 self._file_ext,
                                                 self._max_chunk,
                                                )
                                        )
        while True:
            try:
                chunk = await asyncio.wait_for(res_q.async_q.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # Check if the worker thread exited prematurely
                if crawl_fut.done():
                    break
            else:
                yield chunk
        await crawl_fut

    @asynccontextmanager
    async def send(self, report=None):
        loop = asyncio.get_running_loop()
        send_q = janus.Queue(10, loop=loop)
        send_fut = loop.run_in_executor(None,
                                        partial(_disk_write_worker,
                                                send_q.sync_q,
                                                self._root_path,
                                                self._out_fmt,
                                                self._force_overwrite)
                                       )
        try:
            yield send_q.async_q
        finally:
            if not send_fut.done():
                await send_q.async_q.put(None)
            await send_fut

    def __str__(self):
        return 'LocalDir(%s)' % self._root_path