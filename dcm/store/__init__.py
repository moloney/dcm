from contextlib import asynccontextmanager

class DataChunk:
    '''Naive chunk of data that can just provide a sequence of data sets'''

    report = None

    async def gen_data(self):
        '''Generator produces the data sets in this chunk'''
        raise NotImplementedError


class RepoChunk(DataChunk):
    '''Smarter chunk of data referencing a DataRepo/QueryResult combo'''

    def __init__(self, repo, qr):
        self.repo = repo
        self.qr = qr

    def __str__(self):
        return "RepoChunk(%s, %s)" % (self.repo, self.qr)


class DataBucket:
    '''Base class for all naive data stores, can't query or get specific data

    Can simply produce one or more DataChunk instances with the
    `gen_data` method, or store files through the `send` method
    '''

    async def gen_chunks(self):
        '''Generate a data in this bucket, one chunk at a time'''
        raise NotImplementedError

    @asynccontextmanager
    async def send(self):
        '''Produces a Queue that you can put data sets into for storage'''
        raise NotImplementedError


class DataRepo(DataBucket):
    '''Base class for smarter data stores, with query/get functionality
    '''
    async def query(self, level=None, query=None, query_res=None):
        raise NotImplementedError

    async def gen_query_chunks(self, query_res):
        raise NotImplementedError