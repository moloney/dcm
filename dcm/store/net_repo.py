import logging
from copy import deepcopy
from contextlib import asynccontextmanager

from . import RepoChunk, DataRepo
from ..util import aclosing
from ..query import QueryLevel, QueryResult
from ..net import LocalEntity, RetrieveReport


log = logging.getLogger(__name__)


class DicomNetDataChunk(RepoChunk):
    '''Data chunk on a remote DICOM server
    '''

    def __init__(self, repo, qr):
        super().__init__(repo, qr)
        self.report = RetrieveReport()

    async def gen_data(self, report=None):
        log.debug("Generating chunk data")
        try:
            async with aclosing(self.repo.retrieve(self.qr, report=self.report)) as rgen:
                async for data_set in rgen:
                    yield data_set
        finally:
            log.debug("Done generating chunk data")

    async def move(self, dest):
        '''Perform direct move to dest without getting files locally'''
        raise NotImplementedError # TODO: Do direct move




# TODO: Might need a max_associations param. If a remote has a very small
#       limit on simultaneous associations we might need to avoid overlapping
#       queries and data transfers.
# TODO: Pass through reports wherever needed
class NetRepo(DataRepo):
    '''Smart data store corresponding to a DICOM network entity'''

    is_local = False

    def __init__(self, local, remote, level=None, base_query=None,
                 chunk_size=1000):
        self._local_ent = LocalEntity(local)
        self._remote = remote
        self._level = level
        self._base_query = base_query
        if self._base_query is None:
            self._base_query = {}
        self.chunk_size = chunk_size

    def __getstate__(self):
        state = {k : v for k, v in self.__dict__.items() if k != '_local_ent'}
        state['local'] = self._local_ent._local
        return state

    def __setstate__(self, state):
        self._local_ent = LocalEntity(state['local'])
        del state['local']
        for attr, val in state.items():
            setattr(self, attr, val)

    @property
    def base_query(self):
        return deepcopy(self._base_query)

    async def query(self, level=None, query=None, query_res=None):
        if level is None:
            if query_res is not None:
                level = query_res.level
            else:
                level = self._level
        q = deepcopy(query)
        if self._base_query is not None:
            if q is None:
                q = {}
            q.update(self._base_query)
        return await self._local_ent.query(self._remote, level, q, query_res)

    def queries(self, level=None, query=None, query_res=None):
        '''Returns async generator that produces partial QueryResult objects'''
        if level is None:
            if query_res is not None:
                level = query_res.level
            else:
                level = self._level
        q = deepcopy(query)
        if self._base_query is not None:
            if q is None:
                q = {}
            q.update(self._base_query)
        return self._local_ent.queries(self._remote, level, q, query_res)

    async def gen_chunks(self):
        qr = await self.query()
        async for chunk in self.gen_query_chunks(qr):
            yield chunk

    async def gen_query_chunks(self, query_res):
        '''Generate chunks of data corresponding to `query_res`
        '''
        curr_qr = QueryResult(query_res.level)
        for path, sub_uids in query_res.walk():
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
                        yield DicomNetDataChunk(self, curr_qr)
                        curr_qr = QueryResult(query_res.level)
            elif path.level == query_res.level:
                if len(curr_qr) != 0:
                    yield DicomNetDataChunk(self, curr_qr)
                    curr_qr = QueryResult(query_res.level)
                yield DicomNetDataChunk(self, query_res.sub_query(path.end))
        if len(curr_qr) != 0:
            yield DicomNetDataChunk(self, curr_qr)

    @asynccontextmanager
    async def send(self, report=None):
        async with self._local_ent.send(self._remote, report=report) as send_q:
            yield send_q

    def move_chunk(self, data_chunk):
        # TODO: Do direct transfer here
        raise NotImplementedError

    def retrieve(self, query_res, report=None):
       '''Returns an async generator that w'''
       return self._local_ent.retrieve(self._remote, query_res, report)

    def __str__(self):
        return 'NetRepo(%s)' % (self._remote,)