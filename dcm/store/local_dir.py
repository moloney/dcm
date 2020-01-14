from __future__ import annotations
import os, logging, asyncio, re, shutil
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import partial
from glob import iglob
from typing import Optional, AsyncIterator, Any, Callable, List, Dict, Tuple

from pydicom.dataset import Dataset, FileDataset
import janus

from . import DataBucket, LocalBucket, TransferMethod, LocalChunk, LocalWriteReport
from ..util import fstr_eval, PathInputType, IndividualReport
from ..net import IMPLEMENTATION_UID, IncomingDataReport


log = logging.getLogger(__name__)


def _dir_crawl_worker(res_q: 'janus._SyncQueueProxy[LocalChunk]',
                      root_path : str,
                      recurse: bool = True,
                      file_ext: str = 'dcm',
                      max_chunk: int = 1000) -> None:
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
            res_q.put(LocalChunk(curr_files))
            curr_files = []
    if len(curr_files) != 0:
        res_q.put(LocalChunk(curr_files))


class DefaultDicomWrapper:
    def __init__(self, ds: Dataset, default: str = 'unknown'):
        self._ds = ds
        self._default = default

    def __getattr__(self, attr: str) -> Any:
        return self._ds.get(attr, self._default)


def make_out_path(out_fmt: str, ds: Dataset) -> str:
    out_toks = out_fmt.split(os.sep)
    context = {'d': DefaultDicomWrapper(ds)}
    return os.sep.join([re.sub('[^A-Za-z0-9_.-]', '_', fstr_eval(t, context))
                        for t in out_toks]
                      )


def _disk_write_worker(data_queue: 'janus._SyncQueueProxy[Dataset]',
                       root_path: str,
                       out_fmt: str,
                       force_overwrite: bool = False,
                       report: Optional[LocalWriteReport] = None) -> LocalWriteReport:
    '''Take data sets from a queue and write to disk'''
    if report is None:
        report = LocalWriteReport()
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

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ds.save_as(out_path)
        except Exception as e:
            report.add_error(out_path, e)
        else:
            report.add_success(out_path)
    return report


# TODO: Take force_overwrite kwarg and handle it correctly
def _oob_transfer_worker(paths_queue: 'janus._SyncQueueProxy[Optional[Tuple[PathInputType, PathInputType]]]',
                         transfer_op: Callable[..., Any],
                         report: Optional[LocalWriteReport] = None
                        ) -> LocalWriteReport:
    if report is None:
        report = LocalWriteReport()
    while True:
        log.debug("_oob_transfer_worker is waiting on data")
        in_paths = paths_queue.get()
        if in_paths is None:
            break
        src, dest = in_paths
        log.debug("_oob_transfer_worker got some paths")
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            transfer_op(src, dest)
        except Exception as e:
            report.add_error(dest, e)
        else:
            report.add_success(dest)
    return report


class LocalDir(LocalBucket):
    '''Local directory of data without additional meta data'''

    is_local = True

    default_out_fmt = ('{d.PatientID}/'
                       '{d.StudyInstanceUID}/'
                       '{d.SeriesNumber:03d}-{d.SeriesDescription}/'
                       '{d.SOPInstanceUID}')
    '''Default format for output paths when saving data
    '''

    def __init__(self,
                 path: str,
                 recurse: bool = True,
                 file_ext: str = 'dcm',
                 max_chunk: int = 1000,
                 out_fmt: Optional[str] = None,
                 force_overwrite: bool = False):
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

    def __str__(self) -> str:
        return 'LocalDir(%s)' % self._root_path

    async def gen_chunks(self) -> AsyncIterator[LocalChunk]:
        loop = asyncio.get_running_loop()
        res_q: janus.Queue[LocalChunk] = janus.Queue(loop=loop)
        crawl_fut = asyncio.ensure_future(loop.run_in_executor(None,
                                                               partial(_dir_crawl_worker,
                                                                       res_q.sync_q,
                                                                       self._root_path,
                                                                       self._recurse,
                                                                       self._file_ext,
                                                                       self._max_chunk,
                                                                      )
                                                               )
                                         )
        while True:
            try:
                chunk = await asyncio.wait_for(res_q.async_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if the worker thread exited prematurely
                if crawl_fut.done():
                    break
            else:
                yield chunk
        await crawl_fut

    @asynccontextmanager
    async def send(self,
                   report: Optional[LocalWriteReport] = None
                  ) -> AsyncIterator['janus._AsyncQueueProxy[Dataset]']:
        loop = asyncio.get_running_loop()
        send_q: janus.Queue[Dataset] = janus.Queue(10, loop=loop)
        send_fut = asyncio.ensure_future(loop.run_in_executor(None,
                                                              partial(_disk_write_worker,
                                                                      send_q.sync_q,
                                                                      self._root_path,
                                                                      self._out_fmt,
                                                                      self._force_overwrite)
                                                             )
                                        )
        try:
            yield send_q.async_q
        finally:
            if not send_fut.done():
                await send_q.async_q.put(None)
            await send_fut

    async def oob_transfer(self,
                           method: TransferMethod,
                           chunk: LocalChunk,
                           report: Optional[LocalWriteReport] = None
                          ) -> None:
        if method is TransferMethod.PROXY or method not in self._supported_methods:
            raise ValueError(f"Invalid transfer method: {method}")
        extern_report = False if report is None else True
        # At least for now, python seeems to lack the ability to define only
        # the required args to a callable while ignoring kwargs
        op: Callable[..., Any]
        if method == TransferMethod.LOCAL_COPY:
            op = shutil.copy
        elif method == TransferMethod.MOVE:
            op = shutil.move
        elif method == TransferMethod.LINK:
            op = os.link
        elif method == TransferMethod.SYMLINK:
            op = os.symlink
        loop = asyncio.get_running_loop()
        oob_q: janus.Queue[Optional[Tuple[PathInputType, PathInputType]]] = janus.Queue(10, loop=loop)
        oob_fut = asyncio.ensure_future(loop.run_in_executor(None,
                                                             partial(_oob_transfer_worker,
                                                                     oob_q.sync_q,
                                                                     op,
                                                                     report
                                                                    )
                                                            )
                                        )
        async for src_path, ds in chunk.gen_paths_and_data():
            dest_path = os.path.join(self._root_path,
                                     make_out_path(self._out_fmt, ds))
            if method == TransferMethod.SYMLINK:
                src_path = os.path.abspath(src_path)
            await oob_q.async_q.put((src_path, dest_path))
        await oob_q.async_q.put(None)
        report = await(oob_fut)
        if not extern_report:
            report.log_issues()
            report.check_errors()
