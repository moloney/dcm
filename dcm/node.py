"""Capture info abount DICOM network nodes including per-node configuration."""
import sys, logging, enum
from attrs import define, frozen, field
import re
import string
from typing import FrozenSet, Tuple, List, Dict, Iterable, Union, Optional, Any

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

from pynetdicom import (
    sop_class,
    build_context,
    VerificationPresentationContexts,
)
from pynetdicom.sop_class import SOPClass, uid_to_sop_class
from pynetdicom.presentation import PresentationContext
from pynetdicom._globals import ALL_TRANSFER_SYNTAXES, DEFAULT_TRANSFER_SYNTAXES

from ._globals import QueryLevel
from .util import (
    CustomJsonSerializable,
    TomlConfigurable,
    InlineConfigurable,
    json_serializer,
)


log = logging.getLogger(__name__)


class DicomOpType(enum.Enum):
    """Enumerate the available types of DICOM operations"""

    ECHO = enum.auto()
    FIND = enum.auto()
    MOVE = enum.auto()
    GET = enum.auto()
    STORE = enum.auto()


class DicomRole(enum.Enum):
    """Enumerate the roles available in DICOM operations"""

    USER = enum.auto()
    PROVIDER = enum.auto()


class QueryModel(enum.Enum):
    """Enumerate the DICOM query models available"""

    PATIENT_ROOT = enum.auto()
    STUDY_ROOT = enum.auto()
    PATIENT_STUDY_ONLY = enum.auto()


_verification = VerificationPresentationContexts[0].abstract_syntax
assert _verification is not None
VERIFICATION_AS = SOPClass(_verification)


QR_MODELS = {
    QueryModel.PATIENT_ROOT: {
        DicomOpType.FIND: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientRootQueryRetrieveInformationModelFind"]
        ),
        DicomOpType.MOVE: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientRootQueryRetrieveInformationModelMove"]
        ),
        DicomOpType.GET: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientRootQueryRetrieveInformationModelGet"]
        ),
    },
    QueryModel.STUDY_ROOT: {
        DicomOpType.FIND: uid_to_sop_class(
            sop_class._QR_CLASSES["StudyRootQueryRetrieveInformationModelFind"]
        ),
        DicomOpType.MOVE: uid_to_sop_class(
            sop_class._QR_CLASSES["StudyRootQueryRetrieveInformationModelMove"]
        ),
        DicomOpType.GET: uid_to_sop_class(
            sop_class._QR_CLASSES["StudyRootQueryRetrieveInformationModelGet"]
        ),
    },
    QueryModel.PATIENT_STUDY_ONLY: {
        DicomOpType.FIND: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientStudyOnlyQueryRetrieveInformationModelFind"]
        ),
        DicomOpType.MOVE: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientStudyOnlyQueryRetrieveInformationModelMove"]
        ),
        DicomOpType.GET: uid_to_sop_class(
            sop_class._QR_CLASSES["PatientStudyOnlyQueryRetrieveInformationModelGet"]
        ),
    },
}


@frozen
class SOPClassExpression(InlineConfigurable["SOPClassExpression"]):
    """Flexible matching of SOPClass by UID or name (with regex support for names)"""

    expr: str = field(converter=str)
    """Interpreted as UID if starts with a digit, else a regex to match against names"""

    def __str__(self) -> str:
        return self.expr

    @staticmethod
    def inline_to_dict(in_str: str) -> Dict[str, Any]:
        return {"expr": in_str}

    def matches(self, sop_class: SOPClass) -> bool:
        """Return true if the expression matches the SOPClass"""
        if self.expr[0] in string.digits:
            return sop_class == self.expr
        return bool(re.search(self.expr, sop_class.keyword))


def _make_all_expr(
    input: Iterable[Union[str, SOPClassExpression]]
) -> Tuple[SOPClassExpression, ...]:
    return tuple(SOPClassExpression(x) for x in input)


@frozen
class SOPClassFilter(TomlConfigurable["SOPClassFilter"]):
    """Provide configurable filtering on SOPClasses

    The strings in the `include` and `exclude` attributes can be the UIDs or names of
    SOPClasses, or a regular expression that is matched against SOPCLass names.
    """

    include: Tuple[SOPClassExpression, ...] = field(
        default=tuple(),
        converter=_make_all_expr,
    )
    """Match SOPClasses that should be included, even if matched by `exclude` expression
    """

    exclude: Tuple[SOPClassExpression, ...] = field(
        default=tuple(),
        converter=_make_all_expr,
    )
    """Match SOPClasses that should be excluded"""

    def get_filtered(self, sop_classes: List[SOPClass]) -> List[SOPClass]:
        """Apply the filter to a list of SOP Classes"""
        res: List[SOPClass] = []
        for sop_class in sop_classes:
            if self.include and any(e.matches(sop_class) for e in self.include):
                res.append(sop_class)
                continue
            if any(e.matches(sop_class) for e in self.exclude):
                log.debug("Dropping storage class %s", sop_class.name)
                continue
            res.append(sop_class)
        return res


DEFAULT_DROP_CLASS_REGEXES = (
    "Lensometry",
    "[Rr]efractionMeasurement",
    "Keratometry",
    "Ophthalmic",
    "VisualAcuity",
    "Spectacle",
    "Intraocular",
    "Macular",
    "Corneal",
    "Encapsulated.+Storage",
)


DEFAULT_PRIVATE_SOP_CLASSES = (("SiemensProprietaryMRStorage", "1.3.12.2.1107.5.9.1"),)
# This is needed so 'uid_to_service_class' will work when called for incoming
# data sets
for cls_name, cls_uid in DEFAULT_PRIVATE_SOP_CLASSES:
    sop_class._STORAGE_CLASSES[cls_name] = cls_uid


DEFAULT_STORE_SCU_SOP_FILTER = SOPClassFilter(exclude=DEFAULT_DROP_CLASS_REGEXES)


@frozen
class DcmNodeBase(Protocol):
    """Unique address for DICOM network node"""

    host: str
    """Hostname of the node"""

    ae_title: str = "ANYAE"
    """DICOM AE Title of the node"""

    port: int = 11112
    """DICOM port for the node"""

    def __str__(self) -> str:
        return "%s:%s:%s" % (self.host, self.ae_title, self.port)

    @classmethod
    def inline_to_dict(cls, in_str: str) -> Dict[str, Any]:
        """Parse inline string format 'host[:ae_title][:port]'

        Both the second components are optional
        """
        toks = in_str.split(":")
        if len(toks) > 3:
            raise ValueError("Too many tokens for node specification: %s" % in_str)
        res: Dict[str, Union[str, int]] = {"host": toks[0]}
        if len(toks) == 3:
            res["ae_title"] = toks[1]
            res["port"] = int(toks[2])
        elif len(toks) == 2:
            try:
                res["port"] = int(toks[1])
            except ValueError:
                res["ae_title"] = toks[1]
        return res


class UnsupportedQueryModelError(Exception):
    """The requested query model isn't supported by the remote entity"""


class UnsupportedOperationError(Exception):
    """The requested DICOM operation isn't supported by the remote entity"""


@frozen(slots=False)
class DcmNode(DcmNodeBase, InlineConfigurable["DcmNode"]):
    pass


# TODO: Make these singletons on base host/ae/port combo?
@frozen
class RemoteNode(DcmNodeBase, CustomJsonSerializable, InlineConfigurable["RemoteNode"]):
    """Track remote DcmNode with additional config info"""

    supported_ops: FrozenSet[DicomOpType] = frozenset(
        (DicomOpType.ECHO, DicomOpType.FIND, DicomOpType.MOVE, DicomOpType.STORE)
    )
    """DICOM network operations supported by this node"""

    query_models: Tuple[QueryModel, ...] = (
        QueryModel.STUDY_ROOT,
        QueryModel.PATIENT_ROOT,
        QueryModel.PATIENT_STUDY_ONLY,
    )
    """Supported DICOM query models for the node"""

    transfer_syntax_filter: SOPClassFilter = SOPClassFilter()
    """Filter the transfer syntaxes allowed with this node"""

    default_store_scu_sop_filter: SOPClassFilter = SOPClassFilter(
        exclude=DEFAULT_DROP_CLASS_REGEXES
    )
    """Default filter for abstract syntaxes to propose as storage user"""

    private_store_classes: Optional[
        Tuple[Tuple[str, str], ...]
    ] = DEFAULT_PRIVATE_SOP_CLASSES
    """Any private storage classes supported by this node"""

    max_assoc: int = 5
    """Maximum simultaneous associations to make with this node"""

    transfer_syntaxes: Tuple[SOPClass, ...] = field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "transfer_syntaxes",
            self.transfer_syntax_filter.get_filtered(
                [SOPClass(x) for x in ALL_TRANSFER_SYNTAXES]
            ),
        )

    def get_query_model(self, level: QueryLevel) -> QueryModel:
        """Choose a query model for the given QueryLevel

        Raises UnsupportedQueryModelError there is no appropriate option for this remote
        """
        if level == QueryLevel.PATIENT:
            for query_model in (QueryModel.PATIENT_ROOT, QueryModel.PATIENT_STUDY_ONLY):
                if query_model in self.query_models:
                    return query_model
        elif level == QueryLevel.STUDY:
            for query_model in (QueryModel.STUDY_ROOT, QueryModel.PATIENT_STUDY_ONLY):
                if query_model in self.query_models:
                    return query_model
        else:
            for query_model in (QueryModel.STUDY_ROOT, QueryModel.PATIENT_ROOT):
                if query_model in self.query_models:
                    return query_model
        raise UnsupportedQueryModelError()

    def get_abstract_syntaxes(
        self,
        op: DicomOpType,
        role: Optional[DicomRole] = None,
        query_model: Optional[QueryModel] = None,
    ) -> List[SOPClass]:
        if op not in self.supported_ops:
            raise UnsupportedOperationError(
                f"Remote {self} doesn't support {op} operation"
            )
        if op == DicomOpType.ECHO:
            return [VERIFICATION_AS]
        elif op in (DicomOpType.FIND, DicomOpType.MOVE, DicomOpType.GET):
            if query_model is None:
                raise ValueError("The 'query_model' must be provided for this op type")
            elif query_model not in self.query_models:
                raise ValueError(
                    f"The {query_model} model not supported on node: {self}"
                )
            return [QR_MODELS[query_model][op]]
        else:
            assert op == DicomOpType.STORE
            if role is None:
                raise ValueError("The 'role' must be provided for this op type")
            if self.private_store_classes:
                priv_uids = set(x[1] for x in self.private_store_classes)
            else:
                priv_uids = set()
            store_classes = [
                SOPClass(uid)
                for uid in sop_class._STORAGE_CLASSES.values()
                if uid not in priv_uids
            ]
            if role == DicomRole.USER:
                store_classes = self.default_store_scu_sop_filter.get_filtered(
                    store_classes
                )
                max_len = 128 - len(priv_uids)
                if len(store_classes) > max_len:
                    log.warning("Dropping additional storage classes")
                    store_classes = store_classes[:max_len]
            for priv_uid in priv_uids:
                store_classes.append(SOPClass(priv_uid))
            return store_classes

    def get_presentation_contexts(
        self,
        abstract_syntaxes: Iterable[SOPClass],
        transfer_syntaxes: Optional[Iterable[SOPClass]] = None,
    ) -> List[PresentationContext]:
        """Get the presentation contexts to use for given `abstract_syntaxes`"""
        if transfer_syntaxes is None:
            transfer_syntaxes = self.transfer_syntaxes
        ts_uids = [str(ts) for ts in transfer_syntaxes]
        return [build_context(abs_syn, ts_uids) for abs_syn in abstract_syntaxes]

    def to_json_dict(self) -> Dict[str, Any]:
        res = json_serializer.unstructure_attrs_asdict(self)
        del res["transfer_syntaxes"]
        return res

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "RemoteNode":
        return json_serializer.structure_attrs_fromdict(json_dict, cls)
