"""Define static/dynamic routes for copying DICOM data between storage abstractions"""
from __future__ import annotations
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Optional,
    Tuple,
    Callable,
    Dict,
    List,
    Any,
)

from pydicom import Dataset

from .lazyset import LazySet, FrozenLazySet
from .query import QueryLevel
from .filt import Filter, Selector
from .reports import MultiKeyedError
from .util import TomlConfigurable
from .reports.net_report import IncomingDataError
from .store.base import DataBucket, TransferMethod


log = logging.getLogger(__name__)


class NoValidTransferMethodError(Exception):
    """Error raised when we are unable to select a valid transfer method"""

    def __init__(
        self,
        src_dest_pair: Optional[
            Tuple[DataBucket[Any, Any], DataBucket[Any, Any]]
        ] = None,
    ):
        self.src_dest_pair = src_dest_pair

    def __str__(self) -> str:
        if self.src_dest_pair is None:
            return "No valid transfer method for one or more routes"
        else:
            return f"No valid transfer method between {self.src_dest_pair[0]} and {self.src_dest_pair[1]}"


# TODO: Have been working under the assumption the filter would be applied
#       before resolving dynamic routes, but it is more likely and common
#       that we would want to route on the original data, since we may have
#       a rather broad filter (i.e. anonymization) that screws up the elements
#       used for routing.
#
#       Any logic that would go into a pre-filter could just be placed in the
#       dynamic routing function. We might just need to duplicate that logic
#       into a filter if we also want to persist the changes which is an okay
#       trade-off compared to the complexity of allowing both pre/post filters
#
#       We do lose the ability to specify which elements might be
#       modified, how they might be modified, and what their dependencies are.
#       Do we implicitly disallow uninvertible shenanigans in the dynamic routing
#       function?
@dataclass(frozen=True)
class Route:
    """Abstract base class for all Routes

    The main functionality of routes is to map datasets to destinations.

    Routes can have a filter associated with them, which take a dataset as
    input and return one as output. The dataset can be modified and None can be
    returned to reject the dataset.
    """

    filt: Optional[Filter] = None
    """Streaming data filter for editing and rejecting data sets"""

    def get_dests(
        self, data_set: Dataset
    ) -> Optional[Tuple[DataBucket[Any, Any], ...]]:
        """Return the destintations for the `data set`

        Must be implemented by all subclasses."""
        raise NotImplementedError

    def get_filtered(self, data_set: Dataset) -> Optional[Dataset]:
        if self.filt is None:
            return data_set
        return self.filt(data_set)


@dataclass(frozen=True)
class _StaticBase:
    dests: Tuple[DataBucket[Any, Any], ...]
    """Static tuple of destinations"""

    methods: Tuple[TransferMethod, ...] = (TransferMethod.PROXY,)
    """The transfer methods to use, in order of preference

    This will automatically be paired down to the methods supported by all the
    dests (or just allow PROXY if we have a filter). If no valid transfer
    methods are given a `NoValidTransferMethodError` will be raised.
    """


@dataclass(frozen=True)
class StaticRoute(Route, _StaticBase, TomlConfigurable["StaticRoute"]):
    """Static route that sends all (unfiltered) data to same dests"""

    def __post_init__(self) -> None:
        if self.filt is not None:
            if TransferMethod.PROXY not in self.methods:
                raise NoValidTransferMethodError()
            avail_methods = [TransferMethod.PROXY]
        else:
            avail_methods = []
            for meth in self.methods:
                if all(meth in d._supported_methods for d in self.dests):
                    avail_methods.append(meth)
            if len(avail_methods) == 0:
                raise NoValidTransferMethodError()
        object.__setattr__(self, "dests", tuple(self.dests))
        object.__setattr__(self, "methods", tuple(avail_methods))

    @classmethod
    def from_toml_dict(cls, toml_dict: Dict[str, Any]) -> StaticRoute:
        kwargs = deepcopy(toml_dict)
        methods = kwargs.get("methods")
        if methods is not None:
            kwargs["methods"] = tuple(TransferMethod[m.upper()] for m in methods)
        return cls(**kwargs)

    def get_dests(self, data_set: Dataset) -> Tuple[DataBucket[Any, Any], ...]:
        return self.dests

    def get_method(self, src: DataBucket[Any, Any]) -> TransferMethod:
        for method in self.methods:
            if method in src._supported_methods:
                return method
        raise NoValidTransferMethodError()

    def __str__(self) -> str:
        return "Static: %s" % ",".join(str(d) for d in self.dests)


@dataclass(frozen=True)
class _DynamicBase:
    lookup: Callable[[Dataset], Optional[Tuple[DataBucket[Any, Any], ...]]]
    """Callable takes a dataset and returns destinations"""

    route_level: QueryLevel = QueryLevel.STUDY
    """The level in the DICOM hierarchy we are making routing decisions at"""

    required_elems: FrozenLazySet[str] = field(default_factory=FrozenLazySet)
    """DICOM elements that we require to make a routing decision"""

    dest_methods: Optional[
        Dict[Optional[DataBucket[Any, Any]], Tuple[TransferMethod, ...]]
    ] = None
    """Specify transfer methods for (some) dests

    Use `None` as the key to specify the default transfer methods for all dests
    not explicitly listed.

    Only respected when pre-routing is used. Dynamic routing can only proxy.
    """


@dataclass(frozen=True)
class DynamicRoute(Route, _DynamicBase):
    """Dynamic route which determines destinations based on the data.

    Routing decisions are made before applying the filter to the data.
    """

    def __post_init__(self) -> None:
        if self.dest_methods is not None:
            avail_meths: Dict[
                Optional[DataBucket[Any, Any]], Tuple[TransferMethod, ...]
            ] = {}
            for dest, methods in self.dest_methods.items():
                if self.filt is not None:
                    if TransferMethod.PROXY not in methods:
                        raise NoValidTransferMethodError()
                    avail_meths[dest] = (TransferMethod.PROXY,)
                elif dest is None:
                    avail_meths[dest] = methods
                else:
                    meths = tuple(m for m in methods if m in dest._supported_methods)
                    if len(meths) == 0:
                        raise NoValidTransferMethodError()
                    avail_meths[dest] = meths
            object.__setattr__(self, "dest_methods", avail_meths)
        if self.route_level not in QueryLevel:
            raise ValueError("Invalid route_level: %s" % self.route_level)
        if not isinstance(self.required_elems, FrozenLazySet):
            object.__setattr__(
                self, "required_elems", FrozenLazySet(self.required_elems)
            )

    def get_dests(
        self, data_set: Dataset
    ) -> Optional[Tuple[DataBucket[Any, Any], ...]]:
        dests = self.lookup(data_set)
        if dests is None:
            return None
        return tuple(dests)

    def get_static_routes(self, data_set: Dataset) -> Optional[Tuple[StaticRoute, ...]]:
        """Resolve this dynamic route into one or more static routes"""
        dests = self.lookup(data_set)
        if dests is None:
            return dests
        dests = tuple(dests)

        if self.dest_methods is not None:
            meths_dests_map: Dict[
                Tuple[TransferMethod, ...], List[DataBucket[Any, Any]]
            ] = {}
            default_methods = self.dest_methods.get(None)
            if default_methods is None:
                default_methods = (TransferMethod.PROXY,)
            for dest in dests:
                d_methods = self.dest_methods.get(dest)
                if d_methods is None:
                    d_methods = default_methods
                if d_methods not in meths_dests_map:
                    meths_dests_map[d_methods] = []
                meths_dests_map[d_methods].append(dest)
            return tuple(
                StaticRoute(tuple(sub_dests), filt=deepcopy(self.filt), methods=meths)
                for meths, sub_dests in meths_dests_map.items()
            )
        else:
            return (StaticRoute(dests, filt=deepcopy(self.filt)),)

    def __str__(self) -> str:
        return "Dynamic on: %s" % self.required_elems


@dataclass(frozen=True)
class SelectorDestMap(TomlConfigurable["SelectorDestMap"]):
    """Allow construction of dynamic routes from static config"""

    routing_map: Tuple[Tuple[Selector, Tuple[DataBucket[Any, Any], ...]], ...]
    """One or more tuples of (selector, dests) pairs"""

    default_dests: Optional[Tuple[DataBucket[Any, Any], ...]] = None
    """The default destinations to use when no selectors match"""

    exclude: Optional[Tuple[Selector, ...]] = None
    """Exclude data at routing step (versus `filt` which is applied to each image)"""

    stop_on_first: bool = True
    """Just return dests associated with first selector that matches"""

    route_level: QueryLevel = QueryLevel.STUDY
    """The level in the DICOM hierarchy we are making routing decisions at"""

    dest_methods: Optional[
        Dict[Optional[DataBucket[Any, Any]], Tuple[TransferMethod, ...]]
    ] = None
    """Specify transfer methods for (some) dests

    Use `None` as the key to specify the default transfer methods for all dests
    not explicitly listed.

    Only respected when pre-routing is used. Dynamic routing can only proxy.
    """

    required_elems: FrozenLazySet[str] = field(
        default_factory=FrozenLazySet, init=False
    )
    """DICOM elements that we require to make a routing decision"""

    filt: Optional[Filter] = None
    """Steaming data filter for editing and rejecting data sets"""

    def __post_init__(self) -> None:
        req_elems: LazySet[str] = LazySet()
        for sel, _ in self.routing_map:
            req_elems |= sel.get_read_elems()
        if self.exclude:
            for sel in self.exclude:
                req_elems |= sel.get_read_elems()
        object.__setattr__(self, "required_elems", FrozenLazySet(req_elems))

    @classmethod
    def from_toml_dict(cls, toml_dict: Dict[str, Any]) -> SelectorDestMap:
        kwargs = deepcopy(toml_dict)
        route_level = kwargs.get("route_level")
        if route_level is not None:
            kwargs["route_level"] = QueryLevel[route_level.upper()]
        return cls(**kwargs)

    def get_dynamic_route(self) -> DynamicRoute:
        """Return equivalent DynamicRoute object"""

        def lookup_func(ds: Dataset) -> Optional[Tuple[DataBucket[Any, Any], ...]]:
            res: List[DataBucket[Any, Any]] = []
            if self.exclude:
                if any(sel.test_ds(ds) for sel in self.exclude):
                    return None
            for sel, dests in self.routing_map:
                if sel.test_ds(ds):
                    if self.stop_on_first:
                        return dests
                    else:
                        res += dests
            if not res:
                return self.default_dests
            return tuple(res)

        return DynamicRoute(
            lookup_func,
            route_level=self.route_level,
            required_elems=self.required_elems,
            dest_methods=self.dest_methods,
            filt=self.filt,
        )


class ProxyTransferError(Exception):
    def __init__(
        self,
        store_errors: Optional[MultiKeyedError] = None,
        inconsistent: Optional[Dict[StaticRoute, List[Tuple[Dataset, Dataset]]]] = None,
        duplicate: Optional[Dict[StaticRoute, List[Tuple[Dataset, Dataset]]]] = None,
        incoming_error: Optional[IncomingDataError] = None,
    ):
        self.store_errors = store_errors
        self.inconsistent = inconsistent
        self.duplicate = duplicate
        self.incoming_error = incoming_error

    def __str__(self) -> str:
        res = ["ProxyTransferError:"]
        if self.inconsistent is not None:
            res.append("%d inconsistent data sets" % len(self.inconsistent))
        if self.duplicate is not None:
            res.append("%d duplicate data sets" % len(self.duplicate))
        if self.store_errors is not None:
            for err in self.store_errors.errors:
                res.append(str(err))
        if self.incoming_error is not None:
            res.append(str(self.incoming_error))
        return "\n\t".join(res)
