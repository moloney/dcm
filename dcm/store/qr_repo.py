"""Lightweight local DataRepo that just keeps a JSON serialized QueryResult around"""

import asyncio, json, logging, fnmatch, re
from pathlib import Path
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import (
    AsyncIterator,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    Dict,
    Any,
)

import janus
from pydicom import Dataset

from ..util import PathInputType, InlineConfigurable, atomic_open, create_thread_task
from ..query import (
    MIN_ATTRS,
    UID_ELEMS,
    DataNode,
    DataPath,
    InconsistentDataError,
    QueryLevel,
    QueryResult,
    expand_queries,
    get_level_and_query,
)
from ..net import IncomingDataReport
from ..report import optional_report
from .local_dir import _dir_crawl_worker, _disk_write_worker, LocalDir, get_root_dir
from .base import (
    LocalChunk,
    LocalRepo,
    LocalRepoChunk,
    IndexInitMode,
    LocalWriteReport,
    LocalQueryReport,
    _read_f,
)


log = logging.getLogger(__name__)


class InvalidQrRepoError(Exception):
    pass


DEFUALT_INDEX_ELEMS = MIN_ATTRS + (
    "EchoTime",
    "InversionTime",
    "RepetitionTime",
    "FlipAngle",
    "BodyPartExamined",
)


class QrRepo(LocalRepo, InlineConfigurable["QrRepo"]):
    """Simple local data repository using QueryResult JSON serialization for persistence"""

    default_out_fmt = LocalDir.default_out_fmt

    chunk_type: Type[LocalRepoChunk] = LocalRepoChunk

    def __init__(
        self,
        path: PathInputType,
        index_elems: Iterable[str] = DEFUALT_INDEX_ELEMS,
        file_ext: str = "dcm",
        max_chunk: int = 1000,
        out_fmt: Optional[str] = None,
        overwrite_existing: bool = False,
        make_missing: bool = True,
    ):
        self._root_path = get_root_dir(path, make_missing)
        self._index_elems = index_elems
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
        self._qr_path = self._root_path / "dcm_meta.json"
        if self._qr_path.exists():
            with open(self._qr_path, "rt") as qr_f:
                self._qr = QueryResult.from_json_dict(json.load(qr_f))
                if self._qr.level != QueryLevel.IMAGE:
                    raise InvalidQrRepoError()
        else:
            if len(list(self._root_path.iterdir())) != 0:
                log.warning("No QueryResult JSON found in non-empty dir")
            self._qr = QueryResult(level=QueryLevel.IMAGE)
        self._path_set: Optional[FrozenSet[str]] = None

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        """Parse inline string format 'path[:out_fmt][:file_ext]'

        Both the second components are optional
        """
        return LocalDir.inline_to_dict(in_str)

    @classmethod
    async def is_repo(cls, path: PathInputType) -> bool:
        try:
            root_path = get_root_dir(path)
        except:
            return False
        qr_path = root_path / "dcm_meta.json"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, qr_path.exists)

    @classmethod
    async def build(
        cls: Type["QrRepo"],
        path: PathInputType,
        index_init: IndexInitMode = IndexInitMode.ASSUME_CLEAN,
        scan_fs: bool = False,
        **init_kwargs: Any,
    ) -> "QrRepo":
        """Build LocalRepo with control of index initialization and new file handling"""
        repo = QrRepo(path, **init_kwargs)
        if index_init != IndexInitMode.ASSUME_CLEAN:
            # TODO: Handle index initialization options
            # TODO: Probably also need another option ("update_on_mismatch"?) to control
            #       whether we update the QR or not when we notice a missing file or a
            #       file where meta data has changed.
            raise NotImplementedError
        if scan_fs:
            log.info("About to scan FS for new files")
            loop = asyncio.get_running_loop()
            res_q: janus.Queue[LocalChunk] = janus.Queue()
            crawl_fut = create_thread_task(
                _dir_crawl_worker,
                (
                    res_q.sync_q,
                    repo._root_path,
                    True,
                    repo._file_ext,
                    repo._max_chunk,
                    repo.path_set,
                ),
            )
            n_found = 0
            while True:
                try:
                    chunk = await asyncio.wait_for(res_q.async_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if the worker thread exited
                    if crawl_fut.done():
                        break
                else:
                    async for ds_path, ds in chunk.gen_paths_and_data():
                        try:
                            dupe = ds in repo._qr
                        except InconsistentDataError:
                            log.warn("Refusing to index inconsistent data: %s", ds_path)
                            continue
                        if not dupe:
                            log.debug("Found a new file to index: %s", ds_path)
                            ds.StorageMediaFileSetID = str(ds_path)
                            # TODO: Still need some minimization here and in send...
                            repo._qr.add(ds)
                            n_found += 1
            await crawl_fut
            log.info(f"Found {n_found} new files")
            if n_found:
                repo._path_set = None
                await loop.run_in_executor(None, repo._sync_qr)
        return repo

    @property
    def path_set(self) -> FrozenSet[str]:
        """Get a set of all paths currently indexed"""
        if self._path_set is None:
            self._path_set = frozenset(ds.StorageMediaFileSetID for ds in self._qr)
        return self._path_set

    async def gen_chunks(self) -> AsyncIterator[LocalRepoChunk]:
        """Generate the data in this bucket, one chunk at a time"""
        chunk_qr = QueryResult(level=QueryLevel.IMAGE)
        for ds in self._qr:
            chunk_qr.add(ds)
            if len(chunk_qr) == self._max_chunk:
                yield LocalRepoChunk(self, chunk_qr)
                chunk_qr = QueryResult(level=QueryLevel.IMAGE)
        if len(chunk_qr) != 0:
            yield LocalRepoChunk(self, chunk_qr)

    @asynccontextmanager
    @optional_report
    async def send(
        self, report: Optional[LocalWriteReport] = None
    ) -> AsyncIterator["janus._AsyncQueueProxy[Dataset]"]:
        """Produces a Queue that you can put data sets into for storage"""
        assert report is not None
        loop = asyncio._get_running_loop()
        report._meta_data["root_path"] = self._root_path
        send_q: janus.Queue[Optional[Dataset]] = janus.Queue(10)
        send_fut = create_thread_task(
            _disk_write_worker,
            (
                send_q.sync_q,
                self._root_path,
                self._out_fmt,
                self._overwrite,
                report,
                self._qr,
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
        self._path_set = None
        await loop.run_in_executor(None, self._sync_qr)

    @optional_report
    async def queries(
        self,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[LocalQueryReport] = None,
    ) -> AsyncIterator[QueryResult]:
        """Returns async generator that produces partial QueryResult objects"""
        # If QueryResult was given we potentially generate multiple
        # queries, one for each dataset referenced by the QueryResult
        assert report is not None
        level, query = get_level_and_query(level, query, query_res)
        queries, queried_elems = expand_queries(level, query, query_res)
        res = QueryResult(level)
        for sub_query in queries:
            # Iterate through QueryLevels, narrowing results when able
            log.debug("Processing query: %s", sub_query)
            last_incl: Optional[Set[DataNode]] = None
            incl_nodes: Optional[List[DataNode]] = None
            for curr_lvl, uid_elem in UID_ELEMS.items():
                uid_qval = sub_query.get(uid_elem, "*")
                if uid_qval == "":
                    uid_qval = "*"
                # TODO: Need to recognize date ranges here too
                if "*" not in uid_qval:
                    log.debug(
                        "Checking at level %s for exact_uid %s", curr_lvl, uid_qval
                    )
                    # We have precise (one or none) match specification
                    try:
                        matched_path = self._qr.get_path(DataNode(curr_lvl, uid_qval))
                    except KeyError:
                        break
                    if (
                        last_incl is not None
                        and matched_path.parent.end not in last_incl
                    ):
                        # We found a match but with the wrong parent
                        report.add_inconsistent(sub_query)
                        break
                    incl_nodes = [matched_path.end]
                elif uid_qval == "*":
                    # We are matching everything at this level
                    if last_incl is not None:
                        incl_nodes = []
                        for parent in last_incl:
                            incl_nodes += self._qr.children(parent)
                else:
                    # We are matching zero or more at this level
                    incl_nodes = []
                    uid_regex = re.compile(fnmatch.translate(uid_qval))
                    if last_incl is not None:
                        for parent in last_incl:
                            for child in self._qr.children(parent):
                                if uid_regex.match(child.uid):
                                    incl_nodes.append(child)
                    else:
                        for sub_path in self._qr.level_paths(level):
                            if uid_regex.match(sub_path.end.uid):
                                incl_nodes.append(sub_path.end)
                if curr_lvl == level:
                    # We are done refining results
                    report.add_success(sub_query)
                    if incl_nodes is None:
                        # We matched everything
                        yield deepcopy(self._qr)
                        return
                    res = QueryResult(level)
                    # TODO: Are underlying datasets getting deepcopied here?
                    for incl_node in incl_nodes:
                        res |= self._qr.sub_query(incl_node, level)
                    yield res
                else:
                    # Prepare to refine results at next QueryLevel
                    if incl_nodes is not None:
                        if len(incl_nodes) == 0:
                            report.add_success(sub_query)
                            break
                        last_incl = set(incl_nodes)
                        incl_nodes = None

    async def query(
        self,
        level: Optional[QueryLevel] = None,
        query: Optional[Dataset] = None,
        query_res: Optional[QueryResult] = None,
        report: Optional[LocalQueryReport] = None,
    ) -> QueryResult:
        """Perform a query against the data repo"""
        level, query = get_level_and_query(level, query, query_res)
        res = QueryResult(level)
        async for sub_res in self.queries(level, query, query_res, report):
            res |= sub_res
        return res

    @optional_report
    async def retrieve(
        self, query_res: QueryResult, report: Optional[IncomingDataReport] = None
    ) -> AsyncIterator[Dataset]:
        """Returns an async generator that will produce datasets"""
        assert report is not None
        loop = asyncio.get_running_loop()
        for dpath in query_res.level_paths(query_res.level):
            try:
                if not self._qr.check_path(dpath):
                    continue
            except InconsistentDataError:
                continue  # TODO: Track in report?
            for sub_dpath, _ in self._qr.walk(dpath.end):
                min_ds = self._qr.path_data_set(sub_dpath)
                ds_path = min_ds.StorageMediaFileSetID
                ds = await loop.run_in_executor(None, _read_f, ds_path)
                if report.add(ds):
                    yield ds
        report.done = True

    def _sync_qr(self) -> None:
        """Sync the in-memory and on disk QueryResults"""
        with atomic_open(self._qr_path, mode="wt") as out_f:
            out_f.write(json.dumps(self._qr.to_json_dict()))