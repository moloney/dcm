'''Various utility functions'''
from contextlib import asynccontextmanager


class DicomDataError(Exception):
    '''Base class for exceptions from erroneous dicom data'''


class DuplicateDataError(DicomDataError):
    '''A duplicate dataset was found'''


@asynccontextmanager
async def aclosing(thing):
    try:
        yield thing
    finally:
        await thing.aclose()
