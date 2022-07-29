"""This sub-package captures all of the data storage abstractions

We want to handle data stored in naive local directories or remote network
repositories in a seamless manner, while still allowing users to take advantage
of their specific capabilities.
"""

from . import local_dir, net_repo
from .base import TransferMethod

__all__ = ["local_dir", "net_repo", "TransferMethod"]
