from __future__ import annotations
import logging
from copy import deepcopy
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator, Dict, Any

from pydicom import Dataset
import janus

from . import TransferMethod, DcmNetChunk, DcmRepo
from ..report import MultiListReport
from ..query import QueryLevel, QueryResult
from ..net import DcmNode, LocalEntity, DicomOpReport, RetrieveReport


log = logging.getLogger(__name__)


class NetRepo(DcmRepo):
    '''Smart data store corresponding to a DICOM network entity'''

    is_local = False

    def __init__(self,
                 local: DcmNode,
                 remote: DcmNode,
                 level: Optional[QueryLevel] = None,
                 base_query: Optional[Dataset] = None,
                 chunk_size: int = 1000):
        self._local_ent = LocalEntity(local)
        self._remote = remote
        self._level = level
        self._base_query = deepcopy(base_query)
        self.chunk_size = chunk_size
        self.description = remote.ae_title

    def __getstate__(self) -> Dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k != '_local_ent'}
        state['local'] = self._local_ent._local
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._local_ent = LocalEntity(state['local'])
        del state['local']
        for attr, val in state.items():
            setattr(self, attr, val)

    def __repr__(self) -> str:
        return f'NetRepo({self._local_ent.local}, {self._remote})'

    def __str__(self) -> str:
        return str(self._remote)

    @property
    def remote(self) -> DcmNode:
        return self._remote

    @property
    def base_query(self) -> Dataset:
        return deepcopy(self._base_query)

    async def query(self,
                    level: Optional[QueryLevel] = None,
                    query: Optional[Dataset] = None,
                    query_res: Optional[QueryResult] = None,
                    report: Optional[MultiListReport[DicomOpReport]] = None) -> QueryResult:
        if level is None:
            if query_res is not None:
                level = query_res.level
            else:
                level = self._level
        if self._base_query is not None:
            if query is None:
                query = deepcopy(self._base_query)
            else:
                query = deepcopy(query)
                query.update(self._base_query)
        return await self._local_ent.query(self._remote,
                                           level,
                                           query,
                                           query_res,
                                           report=report)

    def queries(self,
                level: Optional[QueryLevel] = None,
                query: Optional[Dataset] = None,
                query_res: Optional[QueryResult] = None,
                report: Optional[MultiListReport[DicomOpReport]] = None
                ) -> AsyncIterator[QueryResult]:
        '''Returns async generator that produces partial QueryResult objects'''
        if level is None:
            if query_res is not None:
                level = query_res.level
            else:
                level = self._level
        q = deepcopy(query)
        if self._base_query is not None:
            if q is None:
                q = Dataset()
            q.update(self._base_query)
        return self._local_ent.queries(self._remote, level, q, query_res, report=report)

    async def gen_chunks(self) -> AsyncIterator[DcmNetChunk]:
        qr = await self.query()
        async for chunk in self.gen_query_chunks(qr):
            yield chunk

    async def gen_query_chunks(self,
                               query_res: QueryResult
                               ) -> AsyncIterator[DcmNetChunk]:
        '''Generate chunks of data corresponding to `query_res`
        '''
        curr_qr = QueryResult(query_res.level)
        for path, sub_uids in query_res.walk():
            n_inst: Optional[int]
            if path.level == QueryLevel.IMAGE:
                n_inst = 1
            else:
                n_inst = query_res.n_instances(path.end)
            if n_inst is not None:
                if n_inst + len(curr_qr) < self.chunk_size:
                    curr_qr |= query_res.sub_query(path.end)
                    sub_uids.clear()
                else:
                    if len(curr_qr) == 0 and path.level == query_res.level:
                        curr_qr |= query_res.sub_query(path.end)
                        sub_uids.clear()
                    if len(curr_qr) != 0:
                        yield DcmNetChunk(self, curr_qr)
                        curr_qr = QueryResult(query_res.level)
            elif path.level == query_res.level:
                if len(curr_qr) != 0:
                    yield DcmNetChunk(self, curr_qr)
                    curr_qr = QueryResult(query_res.level)
                yield DcmNetChunk(self, query_res.sub_query(path.end))
        if len(curr_qr) != 0:
            yield DcmNetChunk(self, curr_qr)

    @asynccontextmanager
    async def send(self,
                   report: Optional[DicomOpReport] = None
                   ) -> AsyncIterator['janus._AsyncQueueProxy[Dataset]']:
        async with self._local_ent.send(self._remote, report=report) as send_q:
            yield send_q

    def retrieve(self,
                 query_res: QueryResult,
                 report: Optional[RetrieveReport] = None
                 ) -> AsyncIterator[Dataset]:
        return self._local_ent.retrieve(self._remote, query_res, report)

    async def oob_transfer(self,
                           method: TransferMethod,
                           chunk: DcmNetChunk,
                           report: Optional[MultiListReport[DicomOpReport]] = None
                           ) -> None:
        '''Perform out-of-band transfer instead of proxying data'''
        if method != TransferMethod.REMOTE_COPY:
            raise ValueError("Unsupported transfer method: %s" % method)
        await self._local_ent.move(chunk.repo.remote,
                                   self._remote,
                                   chunk.qr,
                                   report=report)
