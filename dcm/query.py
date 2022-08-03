"""Provides an abstraction around the DICOM query model and data hierarchy
"""
from __future__ import annotations
import logging
from copy import deepcopy
from collections import OrderedDict, defaultdict
from itertools import chain
from functools import partial
from enum import IntEnum
from dataclasses import dataclass, field
from typing import (
    Iterable,
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterator,
    Callable,
    Union,
    Set,
)
from xml.etree.ElementInclude import include

from pydicom.dataset import Dataset
from tree_format import format_tree

from .util import DicomDataError, JsonSerializable, json_serializer, fallback_fmt
from .normalize import normalize, make_elem_filter


log = logging.getLogger(__name__)


class QueryLevel(IntEnum):
    """Represents the depth for a query, with larger values meaning more detail"""

    PATIENT = 0
    STUDY = 1
    SERIES = 2
    IMAGE = 3


UID_ELEMS = {
    QueryLevel.PATIENT: "PatientID",
    QueryLevel.STUDY: "StudyInstanceUID",
    QueryLevel.SERIES: "SeriesInstanceUID",
    QueryLevel.IMAGE: "SOPInstanceUID",
}
"""Map QueryLevels to the DICOM keyword that gives the corresponding UID"""


def get_uid(level: QueryLevel, data_set: Dataset) -> str:
    """Get the UID from the `data_set` for the given `level`"""
    return getattr(data_set, UID_ELEMS[level])


def get_all_uids(data_set: Dataset) -> Tuple[str, ...]:
    """Get tuple of UIDs corresponding to levels, with trailing empty UIDs trimmed"""
    uids = []
    last_found = -1
    for lvl in QueryLevel:
        lvl_uid = getattr(data_set, UID_ELEMS[lvl], "")
        if lvl_uid != "":
            last_found = lvl
        uids.append(lvl_uid)
    return tuple(uids[: last_found + 1])


@dataclass(frozen=True)
class DataNode:
    """Identifies a node in the DICOM data hierarchy"""

    level: QueryLevel
    """The level of the node in the hierarchy"""

    uid: str
    """The unique id at the node's level in the hierarchy"""


@dataclass(frozen=True)
class DataPath:
    """Identifies the path to a node in the DICOM data hierarchy"""

    @classmethod
    def from_uids(cls, uids: Tuple[str, ...]) -> "DataPath":
        return cls(QueryLevel(len(uids) - 1), uids)

    level: QueryLevel
    """The level of the node in the hierarchy"""

    uids: Tuple[str, ...]

    end: DataNode = field(init=False)

    def __post_init__(self) -> None:
        if self.level not in QueryLevel:
            raise ValueError("Unknown level: %s" % self.level)
        if len(self.uids) != self.level + 1:
            raise ValueError(
                "Expected %d UIDs, got %d" % (self.level + 1, len(self.uids))
            )
        object.__setattr__(self, "end", DataNode(self.level, self.uids[-1]))

    def __add__(self, new_end: DataNode) -> DataPath:
        if new_end.level != self.level + 1:
            raise ValueError(
                "Trying to add node at level %s to path with level %s",
                new_end.level,
                self.level,
            )
        return DataPath(new_end.level, self.uids + (new_end.uid,))

    @property
    def parent_uid(self) -> str:
        return self.uids[-2]

    @property
    def parent(self) -> DataPath:
        return self.from_uids(self.uids[:-1])


REQ_ELEMS = {
    QueryLevel.PATIENT: [
        "PatientID",
        "PatientName",
    ],
    QueryLevel.STUDY: [
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
    ],
    QueryLevel.SERIES: [
        "SeriesInstanceUID",
        "SeriesNumber",
        "Modality",
    ],
    QueryLevel.IMAGE: [
        "SOPInstanceUID",
        "InstanceNumber",
    ],
}
"""Required attributes for each query level (accumulates at each level)"""


BLANKABLE_REQ_ELEMS = [
    "PatientID",
    "PatientName",
    "StudyDate",
    "StudyTime",
    "SeriesNumber",
    "InstanceNumber",
]
"""Some 'required' attributes can be blank, treat missing values as blank in this case
"""


OPT_ELEMS: Dict[QueryLevel, List[str]] = {
    QueryLevel.PATIENT: [
        "NumberOfPatientRelatedStudies",
        "NumberOfPatientRelatedSeries",
        "NumberOfPatientRelatedInstances",
    ],
    QueryLevel.STUDY: [
        "StudyDescription",
        "AccessionNumber",
        "ModalitiesInStudy",
        "NumberOfStudyRelatedSeries",
        "NumberOfStudyRelatedInstances",
    ],
    QueryLevel.SERIES: [
        "SeriesDescription",
        "ProtocolName",
        "SeriesTime",
        "NumberOfSeriesRelatedInstances",
    ],
    QueryLevel.IMAGE: [],
}
"""Optional attributes we always try to query (exclusive to each level)"""


LEVEL_FILTERS = {
    lvl: make_elem_filter(REQ_ELEMS[lvl] + OPT_ELEMS[lvl]) for lvl in QueryLevel
}
"""Element filters for each level"""


LEVEL_IDENTIFIERS = {
    QueryLevel.PATIENT: [
        "NumberOfPatientRelatedStudies",
        "NumberOfPatientRelatedSeries",
        "NumberOfPatientRelatedInstances",
    ],
    QueryLevel.STUDY: [
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "NumberOfStudyRelatedSeries",
        "NumberOfStudyRelatedInstances",
    ],
    QueryLevel.SERIES: [
        "SeriesInstanceUID",
        "SeriesNumber",
        "Modality",
        "SeriesDescription",
        "ProtocolName",
        "NumberOfSeriesRelatedInstances",
    ],
    QueryLevel.IMAGE: [
        "SOPInstanceUID",
        "InstanceNumber",
    ],
}
"""Maps query levels to elements that imply that level is needed"""


MIN_ATTRS = tuple(chain.from_iterable(chain(REQ_ELEMS.values(), OPT_ELEMS.values())))


def minimal_copy(ds: Dataset, include_elems: Iterable[str] = MIN_ATTRS) -> Dataset:
    """Make reduced copy with only the attributes needed for a QueryResult"""
    res = Dataset()
    for attr in include_elems:
        val = getattr(ds, attr, None)
        if val is not None:
            setattr(res, attr, val)
        elif val in BLANKABLE_REQ_ELEMS:
            setattr(res, attr, "")
    return res


def choose_level(qdat: Dataset, default: QueryLevel = QueryLevel.STUDY) -> QueryLevel:
    """Try to choose the correct level for a given query"""
    for lvl in reversed(QueryLevel):
        for attr in LEVEL_IDENTIFIERS[lvl]:
            if hasattr(qdat, attr):
                return lvl
    return default


def get_subcount_attr(data_level: QueryLevel, count_level: QueryLevel) -> str:
    """"""
    if data_level == QueryLevel.IMAGE:
        raise ValueError("The data_level can not be IMAGE")
    if count_level == QueryLevel.PATIENT:
        raise ValueError("The count_level can not be PATIENT")
    if count_level <= data_level:
        raise ValueError("The count_level must be higher than the data_level")
    if data_level == QueryLevel.PATIENT:
        attr_prefix = "NumberOfPatientRelated"
    elif data_level == QueryLevel.STUDY:
        attr_prefix = "NumberOfStudyRelated"
    else:
        assert data_level == QueryLevel.SERIES
        attr_prefix = "NumberOfSeriesRelated"
    if count_level == QueryLevel.STUDY:
        return attr_prefix + "Studies"
    elif count_level == QueryLevel.SERIES:
        return attr_prefix + "Series"
    else:
        return attr_prefix + "Instances"
    assert False


def info_to_dataset(level: QueryLevel, info: Dict[str, Any]) -> Dataset:
    """Turn normalized `info` dict back into DICOM data set"""
    res = Dataset()
    for key, val in info.items():
        if not key.startswith("n_"):
            setattr(res, key, val)
    # Make sure these take precedence over any "NumberOf..." elements
    for key in ("n_studies", "n_series", "n_instances"):
        val = info.get(key)
        if val is None:
            continue
        if key == "n_studies":
            attr = get_subcount_attr(level, QueryLevel.STUDY)
        elif key == "n_series":
            attr = get_subcount_attr(level, QueryLevel.SERIES)
        else:
            assert key == "n_instances"
            attr = get_subcount_attr(level, QueryLevel.IMAGE)
        setattr(res, attr, val)
    return res


class InsufficientQueryLevelError(Exception):
    """Can't perform operation due to the query level being too shallow"""


class QueryLevelMismatchError(Exception):
    """Query levels differ and they need to be the same for this operation"""


class InconsistentDataError(DicomDataError):
    """The data set violates the established patient/study/series heirarchy"""


class InvalidDicomError(DicomDataError):
    """A DICOM dataset is invalid"""


@json_serializer
@dataclass
class QueryProv:
    """Track how a QueryResult was created

    Allows consumers to avoid duplicate work, and users can avoid repeating
    themselves when chaining operations
    """

    source: Optional[JsonSerializable] = None
    """The source of this query result"""

    queried_elems: Optional[Set[str]] = None
    """The attributes that were queried for"""

    removed_existing_on: Optional[JsonSerializable] = None
    """A remote we queried and removed any data that already exists on it"""

    def __bool__(self) -> bool:
        return any(
            getattr(self, a) is not None
            for a in ("source", "queried_elems", "removed_existing_on")
        )

    def merged(self, other: QueryProv, keep_attrs: bool) -> QueryProv:
        result = QueryProv()
        if self.source == other.source:
            result.source = self.source
        if (
            keep_attrs
            and self.queried_elems is not None
            and other.queried_elems is not None
        ):
            result.queried_elems = self.queried_elems & other.queried_elems
        if self.removed_existing_on == other.removed_existing_on:
            result.removed_existing_on = self.removed_existing_on
        return result

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "queried_elems": list(self.queried_elems)
            if self.queried_elems is not None
            else None,
            "removed_existing_on": self.removed_existing_on,
        }

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> QueryProv:
        queried = json_dict["queried_elems"]
        if queried:
            queried = set(queried)
        return cls(
            json_dict["source"],
            queried,
            json_dict["removed_existing_on"],
        )


@json_serializer
class QueryResult:
    """High level representation of a collection of DICOM data sets

    Object is both set-like as it is a set of unique DICOM datasets, and also
    tree-like as these data sets describe a patient/study/series/image
    hierarchy.

    Parameters
    ----------
    level
        The max level in the hierarchy we have data for

    data_sets
        Initial DICOM datasets to add to the result

    prov
        Optional provenance information about how this QueryResult was built
    """

    def __init__(
        self,
        level: QueryLevel,
        data_sets: Optional[List[Dataset]] = None,
        prov: Optional[QueryProv] = None,
    ):
        if level not in QueryLevel:
            raise ValueError("Invalid query level")
        self._level = level
        self._data: Dict[str, Dataset] = {}
        self._levels: Dict[QueryLevel, Dict[str, Any]] = {}
        for q_lvl in QueryLevel:
            self._levels[q_lvl] = {}
        if data_sets is not None:
            for ds in data_sets:
                self.add(ds)
        if prov is None:
            self.prov = QueryProv()
        else:
            self.prov = prov

    @property
    def level(self) -> QueryLevel:
        """The maximum depth of detail provided"""
        return self._level

    def uids(self) -> Iterator[str]:
        """Generates the UIDs for each contained data set"""
        for uid in self._data.keys():
            yield uid

    def __len__(self) -> int:
        """Number of contained data sets"""
        return len(self._data)

    def __getitem__(self, uid: str) -> Dataset:
        """Lookup contained data sets by uid"""
        return self._data[uid]

    def __delitem__(self, uid: str) -> None:
        """Remove contained data sets by uid"""
        self.remove(self._data[uid])

    def __iter__(self) -> Iterator[Dataset]:
        for ds in self._data.values():
            yield ds

    def clear(self) -> None:
        """Clear all contained data sets"""
        for uid in self._data:
            del self[uid]

    def patients(self) -> Iterator[str]:
        """Genarates all PatientID values in this query result"""
        for pat_id in self._levels[QueryLevel.PATIENT]:
            yield pat_id

    def studies(self, patient_id: Optional[str] = None) -> Iterator[str]:
        """Genarates StudyUIDs in this query result, or for a specific patient"""
        if self._level < QueryLevel.STUDY:
            raise InsufficientQueryLevelError()
        if patient_id is not None:
            it = self._levels[QueryLevel.PATIENT][patient_id]["children"]
        else:
            it = self._levels[QueryLevel.STUDY]
        for study_uid in it:
            yield study_uid

    def series(self, study_uid: Optional[str] = None) -> Iterator[str]:
        """Genarates SeriesUIDs in this query result, or for a specific study"""
        if self._level < QueryLevel.SERIES:
            raise InsufficientQueryLevelError()
        if study_uid is not None:
            it = self._levels[QueryLevel.STUDY][study_uid]["children"]
        else:
            it = self._levels[QueryLevel.SERIES]
        for series_uid in it:
            yield series_uid

    def instances(self, series_uid: Optional[str] = None) -> Iterator[str]:
        """Generates InstanceUIDs for this query result, or a specific series"""
        if self._level < QueryLevel.IMAGE:
            raise InsufficientQueryLevelError()
        if series_uid is not None:
            it = self._levels[QueryLevel.SERIES][series_uid]["children"]
        else:
            it = self._levels[QueryLevel.IMAGE]
        for instance_uid in it:
            yield instance_uid

    def add(self, data_set: Dataset) -> None:
        """Add a data set to the query result

        Does nothing if an equivalent data set has already been added
        """
        last_info = None
        branch_found = False
        for lvl in QueryLevel:
            lvl_uid = getattr(data_set, UID_ELEMS[lvl])
            lvl_info = self._levels[lvl].get(lvl_uid)
            if lvl_info is None:
                branch_found = True
                parent_uid = None
                if last_info is not None:
                    parent_uid = last_info["level_uid"]
                lvl_info = OrderedDict()
                normed_data = normalize(data_set, LEVEL_FILTERS[lvl])
                for attr in REQ_ELEMS[lvl]:
                    val = normed_data.get(attr, None)
                    if val is None:
                        if attr not in BLANKABLE_REQ_ELEMS:
                            raise InvalidDicomError(
                                f"Dataset is missing required elem: {attr}"
                            )
                        log.warning(
                            "Dataset missing required (but blankable) elem: %s", attr
                        )
                        val = ""
                    lvl_info[attr] = val
                for attr in OPT_ELEMS[lvl]:
                    val = normed_data.get(attr)
                    if val is not None:
                        lvl_info[attr] = val
                lvl_info["level_uid"] = lvl_uid
                lvl_info["parent_uid"] = parent_uid
                lvl_info["children"] = {}
                self._levels[lvl][lvl_uid] = lvl_info
                if last_info is not None and lvl_uid not in last_info["children"]:
                    last_info["children"][lvl_uid] = lvl_info
            elif branch_found == True or (
                last_info is not None and lvl_uid not in last_info["children"]
            ):
                raise InconsistentDataError()
            if lvl == self._level:
                if branch_found:
                    self._data[lvl_uid] = data_set
                break
            last_info = lvl_info

    def remove(self, data_set: Dataset) -> None:
        """Remove a single data set"""
        if data_set not in self:
            raise KeyError("Data set not in QueryResult")

        last_uid = None
        last_empty = False
        for lvl in reversed(QueryLevel):
            if lvl > self._level:
                continue
            lvl_uid = getattr(data_set, UID_ELEMS[lvl])
            if lvl == self._level:
                del self._data[lvl_uid]
            lvl_info = self._levels[lvl][lvl_uid]
            if last_uid is not None and last_empty:
                del lvl_info["children"][last_uid]
            if len(lvl_info["children"]) == 0:
                del self._levels[lvl][lvl_uid]
                last_empty = True
            else:
                last_empty = False
            last_uid = lvl_uid

    def __contains__(self, data_set: Dataset) -> bool:
        """Test if an equivalent data set has already been added

        Raises an InconsistentDataError if the data violates the current data
        hierarchy, e.g. has a known SeriesInstanceUID but a different
        StudyInstanceUID.
        """
        res = True
        last_info = None
        for lvl in QueryLevel:
            try:
                lvl_uid = getattr(data_set, UID_ELEMS[lvl])
            except AttributeError:
                # The PatientID can be blank, though many systems won't support it...
                if lvl == QueryLevel.PATIENT:
                    lvl_uid = ""
                else:
                    break
            lvl_info = self._levels[lvl].get(lvl_uid)
            if lvl_info is None:
                res = False
            else:
                if res == False or (
                    last_info is not None and lvl_uid not in last_info["children"]
                ):
                    raise InconsistentDataError()
            if lvl == self._level:
                break
            last_info = lvl_info
        return res

    def __eq__(self, other: object) -> bool:
        """True if the level, UIDs, and any sub-counts match"""
        if not isinstance(other, QueryResult):
            raise NotImplementedError
        if self._level != other._level:
            return False
        if self._data.keys() != other._data.keys():
            return False
        # At this point the data sets should be equal, except for differences
        # in sub-counts or due to data heirarchy inconsistencies
        for path, sub_uids in self.walk():
            if path.level != self._level:
                continue
            try:
                oth_path = other.get_path(path.end)
            except KeyError:
                raise InconsistentDataError()
            if path != oth_path:
                raise InconsistentDataError()
            if path.level < QueryLevel.IMAGE:
                n_inst = self.n_instances(path.end)
                oth_n_inst = other.n_instances(path.end)
                if n_inst != oth_n_inst:
                    return False
            if path.level < QueryLevel.SERIES:
                n_series = self.n_series(path.end)
                oth_n_series = other.n_series(path.end)
                if n_series != oth_n_series:
                    return False
            if path.level < QueryLevel.STUDY:
                n_studies = self.n_studies(path.end)
                oth_n_studies = other.n_studies(path.end)
                if n_studies != oth_n_studies:
                    return False
        return True

    def equivalent(self, other: QueryResult) -> bool:
        """Slightly looser equality testing allows different levels

        Provided the lower level QueryResult has sub-counts that match the
        higher level one, they are considered equivalent
        """
        if self._level == other._level:
            return self == other
        if self._level < other._level:
            low, high = self, other
        else:
            high, low = self, other
        for pth, low_sub_uids in low.walk():
            try:
                high_info = high.path_info(pth)
            except KeyError:
                return False
            low_info = low.path_info(pth)
            if low_info != high_info:
                return False
        return True

    def children(self, node: Optional[DataNode] = None) -> Iterator[DataNode]:
        """Generate child nodes of the given one"""
        if node is None:
            child_lvl = QueryLevel.PATIENT
            itr = self._levels[QueryLevel.PATIENT].keys()
        else:
            if node.level == QueryLevel.IMAGE:
                return
            child_lvl = QueryLevel(node.level + 1)
            itr = self._levels[node.level][node.uid]["children"].keys()
        for child_uid in itr:
            yield DataNode(child_lvl, child_uid)

    def walk(
        self, start_node: Optional[DataNode] = None
    ) -> Iterator[Tuple[DataPath, List[str]]]:
        """Generator that traverses the hierarchy in depth-first order

        Parameters
        ----------
        start_node
            Limit the traversal to the subtree rooted here

        Returns
        -------
        curr_path
            Identifies the current traversal path

        sub_uids
            List of uids for child nodes, can be trimmed to avoid traversing
        """
        uid_stack: List[str] = []
        if start_node is not None:
            curr_level = start_node.level
            next_nodes = [start_node]
            if curr_level != QueryLevel.PATIENT:
                uid_stack = [x for x in self.get_path(start_node).uids[:-1]]
        else:
            curr_level = QueryLevel.PATIENT
            next_nodes = [
                DataNode(curr_level, uid) for uid in self._levels[curr_level].keys()
            ]

        while next_nodes:
            new_node = next_nodes.pop()
            if new_node.level != curr_level:
                if new_node.level < curr_level:
                    for i in range(curr_level - new_node.level):
                        uid_stack.pop()
                curr_level = new_node.level
            uid_stack.append(new_node.uid)
            sub_uids = list(self._levels[curr_level][new_node.uid]["children"].keys())
            yield (DataPath(curr_level, tuple(uid_stack)), sub_uids)
            if sub_uids:
                next_lvl = QueryLevel(curr_level + 1)
                next_nodes.extend([DataNode(next_lvl, uid) for uid in sub_uids])
            else:
                uid_stack.pop()

    def level_paths(self, level: QueryLevel) -> Iterator[DataPath]:
        """Generate all paths at the given level"""
        if level > self._level:
            raise InsufficientQueryLevelError()
        for curr_path, sub_uids in self.walk():
            if curr_path.level < level:
                continue
            # We hit the target level, so stop recursion
            sub_uids.clear()
            yield curr_path

    def get_path(self, node: DataNode) -> DataPath:
        """Get path to the node in hierarchy"""
        if node.level > self._level:
            raise InsufficientQueryLevelError()
        path = [node.uid]
        parent_uid = self._levels[node.level][node.uid]["parent_uid"]
        parent_lvl = node.level
        while parent_uid is not None:
            parent_lvl -= 1  # type: ignore
            parent = self._levels[parent_lvl][parent_uid]
            path.append(parent["level_uid"])
            parent_uid = parent["parent_uid"]
        return DataPath(node.level, tuple(reversed(path)))

    def check_path(self, path: DataPath) -> bool:
        """Return true if this `path` exists in our data hierarchy

        Raises InconsistentDataError if the path is not consistent with the hierarchy
        """
        if path.level > self._level:
            raise InsufficientQueryLevelError()
        n_match = sum(1 if path.uids[l] in self._levels[l] else 0 for l in QueryLevel)
        if n_match == 0:
            return False
        elif n_match < path.level + 1:
            raise InconsistentDataError()
        return True

    def _get_info(self, lvl_info: Dict[str, Any]) -> Dict[str, Any]:
        res = {}
        for key, val in lvl_info.items():
            if key[0].isupper():
                res[key] = val
        return res

    def _inject_counts(self, node: Optional[DataNode], info: Dict[str, Any]) -> None:

        if node is None:
            info["n_patients"] = self.n_patients()
            level = -1
        else:
            level = node.level
        if level <= QueryLevel.PATIENT:
            n_studies = self.n_studies(node)
            if n_studies is not None:
                info["n_studies"] = n_studies
        if level <= QueryLevel.STUDY:
            n_series = self.n_series(node)
            if n_series is not None:
                info["n_series"] = n_series
        if level <= QueryLevel.SERIES:
            n_instances = self.n_instances(node)
            if n_instances is not None:
                info["n_instances"] = n_instances

    def node_info(self, node: Optional[DataNode]) -> Dict[str, Any]:
        """Get meta data specific to this node in the hierarchy"""
        level: Union[QueryLevel, int]
        if node is not None:
            res = self._get_info(self._levels[node.level][node.uid])
        else:
            res = {}
        self._inject_counts(node, res)
        return res

    def path_info(self, data_path: DataPath) -> Dict[str, Any]:
        """Get all meta data along a path through the hierarchy

        Parameters
        ----------
        data_path
            The path we want to collect meta data for
        """
        if data_path.level > self._level:
            raise InsufficientQueryLevelError()
        info: Dict[str, Any] = {}
        curr_dict = self._levels[QueryLevel.PATIENT]
        for lvl, uid in enumerate(data_path.uids):
            lvl_info = curr_dict[uid]
            for key, val in lvl_info.items():
                if key[0].isupper():
                    info[key] = val
            curr_dict = lvl_info["children"]
        self._inject_counts(data_path.end, info)
        return info

    def path_data_set(self, data_path: DataPath) -> Dataset:
        """Get a DICOM data set containing the meta data for the `data_path`"""
        if data_path.level == self.level:
            return self._data[data_path.end.uid]
        info = self.path_info(data_path)
        return info_to_dataset(data_path.level, info)

    def n_patients(self) -> int:
        """Number of patients in the data"""
        return len(self._levels[QueryLevel.PATIENT])

    def n_studies(self, node: Optional[DataNode] = None) -> Optional[int]:
        """Number of studies in the data

        Parameters
        ----------
        node
            Get the number of studies under this data node, or globally if None
        """
        if node is not None:
            if node.level > QueryLevel.PATIENT:
                raise ValueError("Can't lookup n_studies for level %s" % node.level)
            info = self._levels[node.level][node.uid]
            if node.level == self._level:
                return info.get("NumberOfPatientRelatedStudies")
            else:
                return len(info["children"])
        res = 0
        for child in self.children(node):
            sub_res = self.n_studies(child)
            if sub_res is None:
                return None
            res += sub_res
        return res

    def n_series(self, node: Optional[DataNode] = None) -> Optional[int]:
        """Number of series in the data

        Parameters
        ----------
        node :
            Get the number of series under this data node, or globally if None
        """
        if node is not None:
            if node.level > QueryLevel.STUDY:
                raise ValueError("Can't lookup n_series for level %s" % node.level)
            info = self._levels[node.level][node.uid]
            if node.level == self._level:
                if node.level == QueryLevel.PATIENT:
                    return info.get("NumberOfPatientRelatedSeries")
                else:
                    return info.get("NumberOfStudyRelatedSeries")
            else:
                if node.level == QueryLevel.STUDY:
                    return len(info["children"])
        res = 0
        for child in self.children(node):
            sub_res = self.n_series(child)
            if sub_res is None:
                return None
            res += sub_res
        return res

    def n_instances(self, node: Optional[DataNode] = None) -> Optional[int]:
        """Number of instances in the data

        Parameters
        ----------
        node :
            Get the number of instances under this data node, globally if None
        """
        if node is not None:
            if node.level > QueryLevel.SERIES:
                raise ValueError("Can't lookup n_instances for level %s" % node.level)
            info = self._levels[node.level][node.uid]
            if node.level == self._level:
                if node.level == QueryLevel.PATIENT:
                    return info.get("NumberOfPatientRelatedInstances")
                elif node.level == QueryLevel.STUDY:
                    return info.get("NumberOfStudyRelatedInstances")
                else:
                    return info.get("NumberOfSeriesRelatedInstances")
            else:
                if node.level == QueryLevel.SERIES:
                    return len(info["children"])
        res = 0
        for child in self.children(node):
            sub_res = self.n_instances(child)
            if sub_res is None:
                return None
            res += sub_res
        return res

    def get_count(
        self, level: QueryLevel, node: Optional[DataNode] = None
    ) -> Optional[int]:
        """Get number of nodes at the `level` in the subtree under `node`"""
        if level == QueryLevel.PATIENT:
            assert node is None
            return self.n_patients()
        elif level == QueryLevel.STUDY:
            return self.n_studies(node)
        elif level == QueryLevel.SERIES:
            return self.n_series(node)
        elif level == QueryLevel.IMAGE:
            return self.n_instances(node)
        assert False

    def sub_query(
        self, node: DataNode, max_level: Optional[QueryLevel] = None
    ) -> QueryResult:
        """Get a subset as its own QueryResult

        Parameters
        ----------
        node
            The node in the hierarchy we want as a subset

        max_level
            The level of detail to include in the result
        """
        prov = QueryProv(
            self.prov.source, removed_existing_on=self.prov.removed_existing_on
        )
        if node.level > self._level:
            raise InsufficientQueryLevelError()
        if max_level is None:
            max_level = self._level
            if self.prov.queried_elems is not None:
                prov.queried_elems = self.prov.queried_elems.copy()
        else:
            if max_level < node.level:
                raise ValueError("The max_level is lower than the node level")
        res = QueryResult(max_level, prov=prov)
        last_branch = None
        for dpath, sub_uids in self.walk(node):
            if dpath.level == self._level:
                # Optimization for when max_level < self._level, since we only need
                # to add one data set from each "branch" of the tree
                if dpath.uids[max_level] == last_branch:
                    continue
                res.add(self._data[dpath.uids[-1]])
                last_branch = dpath.uids[max_level]
        return res

    def level_sub_queries(self, level: QueryLevel) -> Iterator[QueryResult]:
        """Generate sub queries at the given `level`"""
        for dpath in self.level_paths(level):
            yield self.sub_query(dpath.end)

    def chunk(self, max_instances: int = 1000) -> Iterator[QueryResult]:
        """Generate sub queries constrained by size

        If n_instances is unknown, just yield series level (or highest available) sub
        queries.
        """
        n_inst = self.n_instances()
        if n_inst is None:
            # We don't have info to constrain by size
            for sub_qr in self.level_sub_queries(min(self._level, QueryLevel.SERIES)):
                yield sub_qr
        elif n_inst < max_instances:
            yield deepcopy(self)
        else:
            chunk_qr = QueryResult(self._level)
            for curr_path, sub_uids in self.walk():
                n_inst = self.n_instances(curr_path.end)
                assert n_inst is not None
                if len(chunk_qr) + n_inst <= max_instances:
                    chunk_qr |= self.sub_query(curr_path.end)
                    sub_uids.clear()
                elif curr_path.level == self._level:
                    if len(chunk_qr) != 0:
                        yield chunk_qr
                        chunk_qr = QueryResult(self._level)
                    if n_inst >= max_instances:
                        # Result is too big but we can't get any smaller
                        yield self.sub_query(curr_path.end)
                    else:
                        chunk_qr |= self.sub_query(curr_path.end)
            if chunk_qr:
                yield chunk_qr

    def reduced(self, level: QueryLevel) -> QueryResult:
        """Create lower level of detail copy"""
        if level >= self._level:
            raise ValueError("The level provided to 'reduced' must be lower")
        res = QueryResult(level)
        for dpath in self.level_paths(level):
            ds = info_to_dataset(level, self.path_info(dpath))
            res.add(ds)
        return res

    def sub(self, other: QueryResult, ignore_subcounts: bool = False) -> QueryResult:
        """Take the difference between two QueryResutls, with options

        Using __sub__ is equivalent to calling this with default options
        """
        res = QueryResult(self._level, prov=self.prov)
        for dpath in self.level_paths(min(self._level, other._level)):
            dnode = dpath.end
            try:
                other.get_path(dnode)
            except KeyError:
                # This sub-tree of data doesn't exist in other
                res |= self.sub_query(dnode)
                continue
            if dpath.level == QueryLevel.IMAGE or ignore_subcounts:
                continue
            # This node exists in other, check for sub-count differences
            diff_info = {}
            self_info = self.path_info(dpath)
            other_info = other.path_info(dpath)
            sub_counts = ("n_studies", "n_series", "n_instances")
            for key in sub_counts:
                val = self_info.get(key)
                other_val = other_info.get(key)
                if val is None or other_val is None or val <= other_val:
                    continue
                diff_info[key] = val - other_val
            if diff_info:
                if res._level > other._level:
                    # We have a difference in sub-counts and we don't know
                    # which specific data sets are missing, so we need to
                    # produce a lower detail result that just has a reduced
                    # sub-count
                    res = res.reduced(other._level)
                for key, val in self_info.items():
                    if key not in sub_counts:
                        diff_info[key] = val
                res.add(info_to_dataset(res.level, diff_info))
        return res

    def __and__(self, other: QueryResult) -> QueryResult:
        """Take intersection of two QueryResult objects"""
        if self._level != other._level:
            if self._level > other._level:
                pref, non_pref = self, other
            else:
                pref, non_pref = other, self
        else:
            pref, non_pref = self, other
        res = QueryResult(max(self._level, other._level), prov=deepcopy(pref.prov))
        for ds in pref._data.values():
            if ds in non_pref:
                res.add(ds)
        return res

    def __or__(self, other: QueryResult) -> QueryResult:
        """Take union of two QueryResult objects"""
        if not self.prov and len(self._data) == 0:
            self.prov = other.prov
        else:
            prov = self.prov.merged(other.prov, self._level == other._level)
        res = QueryResult(min(self._level, other._level), prov=prov)
        for ds in self._data.values():
            res.add(ds)
        for ds in other._data.values():
            res.add(ds)
        return res

    def __ior__(self, other: QueryResult) -> QueryResult:
        """In place union between two QueryResult"""
        if other._level < self._level:
            raise ValueError(
                "Can't do in-place intersection with lower level QueryResult"
            )
        if not self.prov and len(self._data) == 0:
            self.prov = other.prov
        else:
            self.prov = self.prov.merged(other.prov, True)
        for ds in other._data.values():
            self.add(ds)
        return self

    def __sub__(self, other: QueryResult) -> QueryResult:
        """Take difference between one QueryResult and another"""
        return self.sub(other)

    def __isub__(self, other: QueryResult) -> QueryResult:
        if other._level < self._level:
            raise ValueError("Can't do in-place subtraction of lower level QueryResult")
        for dpath in self.level_paths(self._level):
            dnode = dpath.end
            try:
                other.get_path(dnode)
            except KeyError:
                continue
            # This node exists in other, check for sub-count differences
            diff_info = {}
            self_info = self.path_info(dpath)
            other_info = other.path_info(dpath)
            sub_counts = ("n_studies", "n_series", "n_instances")
            for key in sub_counts:
                val = self_info.get(key)
                other_val = other_info.get(key)
                if val is None or other_val is None or val <= other_val:
                    continue
                diff_info[key] = val - other_val
            if not diff_info:
                self.remove(self.path_data_set(dpath))
            else:
                ds = self._data[dnode.uid]
                for counter_name, counter_diff in diff_info.items():
                    if counter_name == "n_studies":
                        counter_attr = get_subcount_attr(self._level, QueryLevel.STUDY)
                    elif counter_name == "n_series":
                        counter_attr = get_subcount_attr(self._level, QueryLevel.SERIES)
                    else:
                        counter_attr = get_subcount_attr(self._level, QueryLevel.IMAGE)
                    setattr(ds, counter_attr, counter_diff)
        return self

    def __xor__(self, other: QueryResult) -> QueryResult:
        return (self - other) | (other - self)

    def to_json_dict(self) -> Dict[str, Any]:
        """Dump a JSON representation of the heirarchy"""
        return {
            "level": self._level.name,
            "patients": self._levels[QueryLevel.PATIENT],
            "prov": self.prov.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> QueryResult:
        """Create a QueryResult from a previous `to_json` call"""
        level = getattr(QueryLevel, json_dict["level"])
        res = cls(level, prov=QueryProv.from_json_dict(json_dict["prov"]))
        patients = json_dict["patients"]
        visit_q = list(patients.values())
        visited_stack: List[Dict[str, Any]] = []
        while len(visit_q) != 0:
            info = visit_q.pop()
            if info is None:
                visited_stack.pop()
                continue
            visited_stack.append(info)
            if len(info["children"]) != 0:
                visit_q.append(None)
                visit_q.extend(info["children"].values())
            else:
                ds = Dataset()
                for lvl_info in visited_stack:
                    for key, val in lvl_info.items():
                        if key[0].isupper():
                            setattr(ds, key, val)
                res.add(ds)
                visited_stack.pop()
        return res

    _def_cntr_fmts = (
        "{n_patients} patients",
        "{n_studies} studies",
        "{n_series} series",
        "{n_instances} instances",
    )

    # Not sure why, buy mypy currently fails to infer the values type here
    _def_level_fmts: Dict[Optional[QueryLevel], Tuple[str, ...]] = {
        None: _def_cntr_fmts,
        QueryLevel.PATIENT: (
            ("ID: {PatientID}", "Name: {PatientName}") + _def_cntr_fmts[1:]
        ),
        QueryLevel.STUDY: (
            ("Date: {StudyDate}", "Time: {StudyTime}") + _def_cntr_fmts[2:]
        ),
        QueryLevel.SERIES: (
            ("{SeriesNumber:03d}", "{SeriesDescription}", "{Modality}")
            + _def_cntr_fmts[3:]
        ),
        QueryLevel.IMAGE: ("{InstanceNumber:05d}", "{SOPInstanceUID}"),
    }
    """Default format for each line item in output from `to_tree`"""

    _def_sort_elems = {
        QueryLevel.PATIENT: "PatientID",
        QueryLevel.STUDY: "StudyDate",
        QueryLevel.SERIES: "SeriesNumber",
        QueryLevel.IMAGE: "InstanceNumber",
    }

    def to_line(
        self,
        node: Optional[DataNode] = None,
        level_fmts: Optional[Dict[Optional[QueryLevel], Tuple[str, ...]]] = None,
        sep: str = " | ",
        missing: str = "NA",
    ) -> str:
        """Get line of text describing a single node"""
        level = None
        if node is not None:
            level = node.level
        if level_fmts is not None:
            fmt_toks = level_fmts[level]
        else:
            fmt_toks = self._def_level_fmts[level]
        line_fmt = sep.join(fmt_toks)
        node_info = defaultdict(lambda: missing, self.node_info(node))
        return fallback_fmt.vformat(line_fmt, args=[], kwargs=node_info)

    def _make_sorted_child_getter(
        self, sort_elems: Dict[QueryLevel, str], max_level: QueryLevel
    ) -> Callable[[DataNode], Iterator[DataNode]]:
        def sorted_child_getter(node: DataNode) -> Iterator[DataNode]:
            if node is not None and max_level == node.level:
                return
            children = [x for x in self.children(node)]
            if len(children) == 0:
                return
            sort_elem = sort_elems[children[0].level]
            children = sorted(
                children, key=lambda n: self.node_info(n).get(sort_elem, 0)
            )
            for child in children:
                yield child

        return sorted_child_getter

    def to_tree(
        self,
        max_level: QueryLevel = QueryLevel.IMAGE,
        level_fmts: Optional[Dict[Optional[QueryLevel], str]] = None,
        sep: str = " | ",
        missing: str = "NA",
    ) -> str:
        """Produce a formatted text tree representation"""
        formatter = partial(
            self.to_line, level_fmts=level_fmts, sep=sep, missing=missing
        )
        child_getter = self._make_sorted_child_getter(self._def_sort_elems, max_level)
        return format_tree(None, formatter, child_getter)

    def __str__(self) -> str:
        n_pat = self.n_patients()
        if n_pat == 0:
            descr = "Empty"
        elif self.n_patients() == 1:
            descr_comps = []
            sep = " | "
            missing = "NA"
            for pth, sub_uids in self.walk():
                if len(sub_uids) != 1:
                    descr_comps.append(self.to_line(pth.end))
                    break
                else:
                    fmt_toks = [
                        x
                        for x in self._def_level_fmts[pth.level]
                        if x not in self._def_cntr_fmts
                    ]
                    line_fmt = sep.join(fmt_toks)
                    node_info = defaultdict(lambda: missing, self.node_info(pth.end))
                    descr_comps.append(
                        fallback_fmt.vformat(line_fmt, args=[], kwargs=node_info)
                    )
            descr = sep.join(descr_comps)
        else:
            descr = self.to_line(None)
        return "%s Level QR: %s" % (self.level.name, descr)


def get_level_and_query(
    level: Optional[QueryLevel],
    query: Optional[Dataset],
    query_res: Optional[QueryResult],
) -> Tuple[QueryLevel, Dataset]:
    """Resolve/check `level` and `query` args for query methods"""
    # Build up our base query dataset
    if query is None:
        query = Dataset()
    else:
        query = deepcopy(query)

    # Deterimine level if not specified, otherwise make sure it is valid
    if level is None:
        if query_res is None:
            default_level = QueryLevel.STUDY
        else:
            default_level = query_res.level
        level = choose_level(query, default_level)
    elif level not in QueryLevel:
        raise ValueError("Unknown 'level' for query: %s" % level)
    return level, query


def expand_queries(
    level: QueryLevel, query: Dataset, query_res: Optional[QueryResult] = None
) -> Tuple[List[Dataset], Set[str]]:
    queried_elems = set(e.keyword for e in query)
    if query_res is None:
        queries = [query]
    else:
        queries = []
        for path, sub_uids in query_res.walk():
            if path.level == min(level, query_res.level):
                q = deepcopy(query)
                for lvl in QueryLevel:
                    if lvl > path.level:
                        break
                    setattr(q, UID_ELEMS[lvl], path.uids[lvl])
                    queried_elems.add(UID_ELEMS[lvl])
                queries.append(q)
                sub_uids.clear()
    return queries, queried_elems
