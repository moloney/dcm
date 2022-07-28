"""Data storage abstraction for local directories"""
from __future__ import annotations
import os, logging, asyncio, re, shutil, threading
from contextlib import asynccontextmanager
from glob import iglob
from pathlib import Path
from queue import Empty
from typing import Optional, AsyncIterator, Any, Callable, Tuple, cast, Union, Dict

from pydicom.dataset import Dataset
import janus

from .base import LocalBucket, TransferMethod, LocalChunk, LocalWriteReport
from ..util import fstr_eval, PathInputType, InlineConfigurable, create_thread_task


log = logging.getLogger(__name__)


def _dir_crawl_worker(
    res_q: "janus._SyncQueueProxy[LocalChunk]",
    root_path: Path,
    recurse: bool = True,
    file_ext: str = "dcm",
    max_chunk: int = 1000,
    shutdown: Optional[threading.Event] = None,
) -> None:
    curr_files = []
    # TODO: Update this to use pathlib.Path glob functionality
    glob_comps = [str(root_path)]
    if recurse:
        glob_comps.append("**")
    if file_ext:
        glob_comps.append("*.%s" % file_ext)
    else:
        glob_comps.append("*")
    glob_exp = os.path.join(*glob_comps)
    for pidx, path in enumerate(iglob(glob_exp, recursive=recurse)):
        if pidx + 1 % 50 == 0:
            if shutdown is not None and shutdown.is_set():
                return
        if not os.path.isfile(path):
            continue
        curr_files.append(path)
        if len(curr_files) == max_chunk:
            res_q.put(LocalChunk(curr_files))
            curr_files = []
    if len(curr_files) != 0:
        res_q.put(LocalChunk(curr_files))


class DefaultDicomWrapper:
    def __init__(self, ds: Dataset, default: str = "unknown"):
        self._ds = ds
        self._default = default

    def __getattr__(self, attr: str) -> Any:
        return self._ds.get(attr, self._default)


def make_out_path(out_fmt: str, ds: Dataset) -> str:
    out_toks = out_fmt.split("/")
    context = {"d": DefaultDicomWrapper(ds)}
    return os.sep.join(
        [re.sub("[^A-Za-z0-9_.-]", "_", fstr_eval(t, context)) for t in out_toks]
    )


def _disk_write_worker(
    data_queue: "janus._SyncQueueProxy[Dataset]",
    root_path: Path,
    out_fmt: str,
    force_overwrite: bool,
    report: LocalWriteReport,
    shutdown: Optional[threading.Event] = None,
) -> None:
    """Take data sets from a queue and write to disk"""
    while True:
        log.debug("disk_writer is waiting on data")
        no_input = False
        try:
            # TODO: Stop ignoring types when this is fixed: https://github.com/aio-libs/janus/issues/267
            ds = data_queue.get(timeout=0.2)  # type: ignore
        except Empty:
            log.debug("disk_writer timed out waiting on queue")
            no_input = True
        else:
            if ds is None:
                break
        if report.n_input + 1 % 20 == 0 or no_input:
            if shutdown is not None and shutdown.is_set():
                break
        if no_input:
            continue
        log.debug("disk_writer thread got a data set")
        out_path = root_path / make_out_path(out_fmt, ds)

        if os.path.exists(out_path):
            if force_overwrite:
                log.debug("File exists, overwriting: %s", out_path)
            else:
                log.debug("File exists, skipping: %s", out_path)
                report.add_skipped(out_path)
                continue
        else:
            log.debug("Storing DICOM file: %s" % out_path)

        # Build any needed dirs and write the file out
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ds.save_as(out_path, write_like_original=False)
        except Exception as e:
            report.add_error(out_path, e)
        else:
            report.add_success(out_path)


def _oob_transfer_worker(
    paths_queue: "janus._SyncQueueProxy[Optional[Tuple[PathInputType, PathInputType]]]",
    transfer_op: Callable[..., Any],
    force_overwrite: bool,
    report: LocalWriteReport,
    shutdown: Optional[threading.Event] = None,
) -> None:
    while True:
        log.debug("_oob_transfer_worker is waiting on data")
        no_input = False
        try:
            # TODO: Stop ignoring types when this is fixed: https://github.com/aio-libs/janus/issues/267
            in_paths = paths_queue.get(timeout=0.5)  # type: ignore
        except Empty:
            no_input = True
            log.debug("Out-of-band worker timed out waiting for data")
        else:
            if in_paths is None:
                break
        if report.n_input + 1 % 20 == 0 or no_input:
            if shutdown is not None and shutdown.is_set():
                break
        if no_input:
            continue
        in_paths = cast(Tuple[PathInputType, PathInputType], in_paths)
        src, dest = (os.fspath(x) for x in in_paths)
        log.debug("_oob_transfer_worker got some paths: (%s -> %s)", src, dest)
        existing_backup: Optional[Union[str, bytes]] = None
        if os.path.exists(dest):
            if force_overwrite:
                log.debug("File exists, overwriting: %s", dest)
                if isinstance(dest, bytes):
                    existing_backup = dest + b"~"
                else:
                    existing_backup = dest + "~"
                os.rename(dest, existing_backup)
            else:
                log.debug("File exists, skipping: %s", dest)
                report.add_skipped(dest)
                continue
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            transfer_op(src, dest)
        except Exception as e:
            report.add_error(dest, e)
            if existing_backup is not None:
                os.rename(existing_backup, dest)
        else:
            report.add_success(dest)
            if existing_backup is not None:
                os.remove(existing_backup)


class LocalDir(LocalBucket, InlineConfigurable["LocalDir"]):
    """Local directory of data without any additional meta data"""

    is_local = True

    default_out_fmt = (
        "{d.PatientID}/"
        "{d.StudyInstanceUID}/"
        "{d.Modality}/"
        "{d.SeriesNumber:03d}-{d.ProtocolName}/"
        "{d.SOPInstanceUID}"
    )
    """Default format for output paths when saving data
    """

    def __init__(
        self,
        path: PathInputType,
        recurse: bool = True,
        file_ext: str = "dcm",
        max_chunk: int = 1000,
        out_fmt: Optional[str] = None,
        force_overwrite: bool = False,
        make_missing: bool = True,
    ):
        self._root_path = Path(path).expanduser()
        if not self._root_path.exists():
            if make_missing:
                self._root_path.mkdir(parents=True)
            else:
                raise ValueError(f"Path doesn't exist: {self._root_path}")
        elif not self._root_path.is_dir():
            raise ValueError(f"Path is a file not a directory: {self._root_path}")
        self._recurse = recurse
        self._max_chunk = max_chunk
        self._force_overwrite = force_overwrite
        if out_fmt is None:
            self._out_fmt = self.default_out_fmt
        else:
            self._out_fmt = out_fmt
        self._file_ext = file_ext
        if self._file_ext:
            self._out_fmt += ".%s" % file_ext
        self.description = str(self._root_path)

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        """Parse inline string format 'path[:out_fmt][:file_ext]'

        Both the second components are optional
        """
        toks = in_str.split(":")
        # Handle the fact that the path might have a colon in it already in windows
        if (
            len(toks) > 1
            and os.name == "nt"
            and len(toks[0]) == 1
            and toks[1][0] == os.sep
        ):
            toks = [":".join(toks[:2])] + toks[2:]
        res = {"path": toks[0]}
        if len(toks) == 2:
            res["out_fmt"] = toks[1]
        elif len(toks) == 3:
            if toks[1]:
                res["out_fmt"] = toks[1]
            res["file_ext"] = toks[2]
        elif len(toks) >= 4:
            raise ValueError(f"Invalid short form for LocalDir: {in_str}")
        return res

    @property
    def root_path(self) -> Path:
        return self._root_path

    def __str__(self) -> str:
        return f"LocalDir({self._root_path})"

    async def gen_chunks(self) -> AsyncIterator[LocalChunk]:
        res_q: janus.Queue[LocalChunk] = janus.Queue()
        crawl_fut = create_thread_task(
            _dir_crawl_worker,
            (
                res_q.sync_q,
                self._root_path,
                self._recurse,
                self._file_ext,
                self._max_chunk,
            ),
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
    async def send(
        self, report: Optional[LocalWriteReport] = None
    ) -> AsyncIterator["janus._AsyncQueueProxy[Dataset]"]:
        if report is None:
            extern_report = False
            report = LocalWriteReport()
        else:
            extern_report = True
        report._meta_data["root_path"] = self._root_path
        send_q: janus.Queue[Optional[Dataset]] = janus.Queue(10)
        send_fut = create_thread_task(
            _disk_write_worker,
            (
                send_q.sync_q,
                self._root_path,
                self._out_fmt,
                self._force_overwrite,
                report,
            ),
        )
        try:
            yield send_q.async_q  # type: ignore
        finally:
            if not send_fut.done():
                log.debug("Signaling disk writer thread to shutdown")
                await send_q.async_q.put(None)
            log.debug("Waiting for disk writer thread to finish")
            await send_fut
            log.debug("The disk writer thread has finished")
            report.done = True
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def oob_transfer(
        self,
        method: TransferMethod,
        chunk: LocalChunk,
        report: Optional[LocalWriteReport] = None,
    ) -> None:
        if method is TransferMethod.PROXY or method not in self._supported_methods:
            raise ValueError(f"Invalid transfer method: {method}")
        if report is None:
            extern_report = False
            report = LocalWriteReport()
        else:
            extern_report = True
        report._meta_data["root_path"] = self._root_path
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
        oob_q: janus.Queue[Optional[Tuple[PathInputType, PathInputType]]] = janus.Queue(
            10
        )
        log.info("Starting out-of-band transfer worker")
        oob_fut = create_thread_task(
            _oob_transfer_worker, (oob_q.sync_q, op, self._force_overwrite, report)
        )
        async for src_path, ds in chunk.gen_paths_and_data():
            dest_path = self._root_path / make_out_path(self._out_fmt, ds)
            if method == TransferMethod.SYMLINK:
                src_path = os.path.abspath(src_path)
            await oob_q.async_q.put((src_path, dest_path))
        log.info("About to shutdown oob transfer worker")
        await oob_q.async_q.put(None)
        await oob_fut
        log.info("Oob transfer worker shutdown, marking report done")
        report.done = True
        if not extern_report:
            report.log_issues()
            report.check_errors()
