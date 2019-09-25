'''Various utility functions'''
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, AsyncIterator

class DicomDataError(Exception):
    '''Base class for exceptions from erroneous dicom data'''


class DuplicateDataError(DicomDataError):
    '''A duplicate dataset was found'''


@asynccontextmanager
async def aclosing(thing : AsyncGenerator[Any, None]) -> AsyncIterator[AsyncGenerator[Any, None]]:
    '''Context manager that ensures that an async iterator is closed

    See PEP 533 for an explanation on why this is (unfortunately) needded.
    '''
    try:
        yield thing
    finally:
        await thing.aclose()
