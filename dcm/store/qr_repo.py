"""Lightweight local DataRepo that just keeps a JSON serialized QueryResult around"""

from contextlib import asynccontextmanager
import enum
import json, logging
from pathlib import Path
from typing import AsyncIterator, Optional, Set, Type

import janus
from pydicom import Dataset

from ..util import PathInputType, TomlConfigurable, create_thread_task
from ..query import (
    DataNode,
    DataPath,
    QueryLevel,
    QueryResult,
    expand_queries,
    get_level_and_query,
)
from .local_dir import _disk_write_worker, LocalDir, get_root_dir
from . import DataRepo, LocalWriteReport, RepoChunk, _read_f


log = logging.getLogger(__name__)


class QrRepoWriteReport(LocalWriteReport):
    pass


class QrRepoChunk(RepoChunk):
    def __init__(self, repo: "QrRepo", qr: QueryResult):
        self.repo = repo
        self.qr = qr

    async def gen_data(self) -> AsyncIterator[Dataset]:
        async for _, ds in self.gen_paths_and_data():
            yield ds

    async def gen_paths_and_data(self) -> AsyncIterator[Tuple[PathInputType, Dataset]]:
        """Generate both the paths and the corresponding data sets"""
        loop = asyncio.get_running_loop()
        for min_ds in self.qr:
            ds_path = min_ds.StorageMediaFileSetID
            ds = await loop.run_in_executor(None, _read_f, ds_path)
            # if not self.report.add(ds):
            #    continue
            yield ds_path, ds
        # self.report.done = True


class InvalidQrRepoError(Exception):
    pass


DEFUALT_INCL_ELEMS = []


class IndexModes(enum.IntEnum):
    ASSUME_CLEAN = 0
    CHECK_INDEXED = 1
    CHECK_PATHS = 2
    REINDEX = 3


class QrRepo(DataRepo, TomlConfigurable["QrRepo"]):

    default_out_fmt = LocalDir.default_out_fmt

    chunk_type: Type[QrRepoChunk] = QrRepoChunk

    def __init__(
        self,
        path: PathInputType,
        qr_path: PathInputType = "dcm_qr.json",
        index_mode: IndexModes = IndexModes.ASSUME_CLEAN,
        file_ext: str = "dcm",
        max_chunk: int = 1000,
        out_fmt: Optional[str] = None,
        overwrite_existing: bool = False,
        make_missing: bool = True,
    ):
        self._root_dir = get_root_dir(path, make_missing)
        self._max_chunk = max_chunk
        if out_fmt is None:
            self._out_fmt = self.default_out_fmt
        else:
            self._out_fmt = out_fmt
        self._file_ext = file_ext
        if self._file_ext:
            self._out_fmt += ".%s" % file_ext
        self._overwrite = overwrite_existing
        self.description = str(self._root_path)
        if qr_path.is_relative():
            self._qr_path = self._repo_dir / qr_path
        else:
            self._qr_path = qr_path
        if self._qr_path.exists():
            with open(self._qr_path, "rt") as qr_f:
                self._qr = QueryResult.from_json_dict(json.load(qr_f))
                if self._qr.level != QueryLevel.IMAGE:
                    raise InvalidQrRepoError()
        else:
            if len(list(self._local_bucket.root_path.iterdir())) != 0:
                log.warning("No QueryResult JSON found in non-empty dir")
            self._qr = QueryResult(level=QueryLevel.IMAGE)
        if index_mode != IndexModes.ASSUME_CLEAN:
            raise NotImplementedError  # TODO: enable indexing / validation at start up

    async def gen_chunks(self) -> AsyncIterator[RepoChunk]:
        """Generate the data in this bucket, one chunk at a time"""
        chunk_qr = QueryResult(level=QueryLevel.IMAGE)
        for ds in self._qr:
            chunk_qr.add(ds)
            if len(chunk_qr) == self._max_chunk:
                yield QrRepoChunk(self, chunk_qr)
                chunk_qr = QueryResult(level=QueryLevel.IMAGE)
        if len(chunk_qr) == self._max_chunk:
            yield QrRepoChunk(self, chunk_qr)

    @asynccontextmanager
    async def send(
        self, report: Optional[QrRepoWriteReport] = None
    ) -> AsyncIterator["janus._AsyncQueueProxy[Dataset]"]:
        """Produces a Queue that you can put data sets into for storage"""
        if report is None:
            extern_report = False
            report = QrRepoWriteReport()
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
                self._overwrite,
                self._qr,
                report,
            ),
        )
        try:
            yield send_q.async_q  # type: ignore
        finally:
            if not send_fut.done():
                await send_q.async_q.put(None)
            log.debug("awaiting send_fut")
            await send_fut
            log.debug("awaited send_fut")
            report.done = True
        if not extern_report:
            report.log_issues()
            report.check_errors()

    async def queries(
        self,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[T_qreport] = None,
    ) -> AsyncIterator[QueryResult]:
        """Returns async generator that produces partial QueryResult objects"""
        # If QueryResult was given we potentially generate multiple
        # queries, one for each dataset referenced by the QueryResult
        level, query = get_level_and_query(level, query, query_res)
        queries, queried_elems = expand_queries(level, query, query_res)
        res = QueryResult(level)
        sub_paths = set()
        for sub_query in queries:
            last_incl: Optional[Set[DataNode]] = None
            incl_paths: Optional[DataPath] = None
            for curr_lvl, uid_elem in uid_elems.items():
                uid_qval = sub_query.get(uid_elem, "*")
                if uid_qval == "":
                    uid_qval = "*"
                # TODO: Need to recognize date ranges here too
                if "*" not in uid_qval:
                    # We have precise (one or none) match specification
                    incl_paths = []
                    try:
                        matched_path = self._qr.get_path(DataNode(curr_lvl, uid_qval))
                    except KeyError:
                        break  # TODO: is this right?
                    if last_incl is not None and matched_path.parent not in last_incl:
                        break
                    incl_paths = [matched_path]
                elif uid_qval == "*":
                    # We are matching everything at this level
                    if last_incl is not None:
                        incl_paths = []
                        for parent in last_incl:
                            incl_paths += self._qr.children(parent)
                else:
                    # We are matching zero or more at this level
                    uid_regex = re.compile(...)
                    incl_paths = []
                    if last_incl is not None:
                        for parent in last_incl:
                            for child in self._qr.children(parent):
                                if uid_regex.match(child.uid):
                                    incl_paths.append(self._qr.get_path(child))
                    else:
                        for sub_path in self._qr.get_level_paths(level):
                            if uid_regex.match(sub_path.end.uid):
                                incl_paths.append(sub_path)
                if curr_lvl == level:
                    # We are done refining results
                    if incl_paths is None:
                        # We matched everything
                        yield deepcopy(self._qr)
                    res = QueryResult(level)
                    for incl_path in incl_paths:
                        res |= self._qr.sub_query(incl_path.end, level)
                    yield res
                else:
                    # Prepare to refine results at next QueryLevel
                    if incl_paths is not None:
                        if len(incl_paths) == 0:
                            break
                        last_incl = set(p.end for p in incl_paths)
                        incl_paths = None

    async def query(
        self,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[T_qreport] = None,
    ) -> QueryResult:
        """Perform a query against the data repo"""
        level, query = get_level_and_query(level, query, query_res)
        res = QueryResult(level)
        async for sub_res in self.queries(level, query, query_res, report):
            res |= sub_res
        return res

    def retrieve(
        self, query_res: QueryResult, report: Optional[T_rreport] = None
    ) -> AsyncIterator[Dataset]:
        """Returns an async generator that will produce datasets"""
        raise NotImplementedError
        # for ds in query_res:
        #    try:
        #        if ds not in self._qr:
        #            continue
