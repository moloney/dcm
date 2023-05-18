from enum import IntEnum


class QueryLevel(IntEnum):
    """Represents the depth for a query, with larger values meaning more detail"""

    PATIENT = 0
    STUDY = 1
    SERIES = 2
    IMAGE = 3
