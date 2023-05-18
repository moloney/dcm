from pynetdicom import sop_class
from pynetdicom.sop_class import SOPClass, uid_to_sop_class
from pynetdicom._globals import ALL_TRANSFER_SYNTAXES, DEFAULT_TRANSFER_SYNTAXES

from pytest import raises

from .._globals import QueryLevel
from ..node import (
    SOPClassExpression,
    SOPClassFilter,
    DcmNode,
    RemoteNode,
    QueryModel,
    DicomOpType,
    DicomRole,
    UnsupportedOperationError,
)


def test_sop_class_expr():
    # Non-UID inputs treated as regex and compared to keywords
    expr = SOPClassExpression("MRImage")
    assert expr.matches(sop_class.MRImageStorage)
    assert expr.matches(sop_class.EnhancedMRImageStorage)
    # UID inputs must be exact match
    expr = SOPClassExpression(sop_class.MRImageStorage)
    assert expr.matches(sop_class.MRImageStorage)
    assert not expr.matches(sop_class.EnhancedMRImageStorage)


def test_sop_class_filter():
    orig_classes = [SOPClass(x) for x in DEFAULT_TRANSFER_SYNTAXES]
    # Empty filter keeps everything
    filt = SOPClassFilter()
    assert filt.get_filtered(orig_classes) == orig_classes
    # Can exclude using expressions
    filt = SOPClassFilter(exclude=["Explicit", "BigEndian"])
    assert filt.get_filtered(orig_classes) == [SOPClass("1.2.840.10008.1.2")]
    # Includes override excludes
    filt = SOPClassFilter(include=["Deflated"], exclude=["Explicit", "BigEndian"])
    assert filt.get_filtered(orig_classes) == [
        SOPClass("1.2.840.10008.1.2"),
        SOPClass("1.2.840.10008.1.2.1.99"),
    ]


def test_make_node_inline():
    node = DcmNode.from_toml_val("mypacs.org")
    assert node.host == "mypacs.org" and node.ae_title == "ANYAE" and node.port == 11112
    node = DcmNode.from_toml_val("mypacs.org:104")
    assert node.host == "mypacs.org" and node.ae_title == "ANYAE" and node.port == 104
    node = DcmNode.from_toml_val("mypacs.org:MYAE")
    assert node.host == "mypacs.org" and node.ae_title == "MYAE" and node.port == 11112
    node = DcmNode.from_toml_val("mypacs.org:MYAE:104")
    assert node.host == "mypacs.org" and node.ae_title == "MYAE" and node.port == 104
    with raises(ValueError):
        node = DcmNode.from_toml_val("mypacs.org:MYAE:104:blah")


def test_remote_node():
    node = RemoteNode("mypacs.org")
    assert node.transfer_syntaxes == [SOPClass(x) for x in ALL_TRANSFER_SYNTAXES]
    assert node.get_query_model(QueryLevel.PATIENT) == QueryModel.PATIENT_ROOT
    assert node.get_query_model(QueryLevel.STUDY) == QueryModel.STUDY_ROOT
    assert node.get_query_model(QueryLevel.SERIES) == QueryModel.STUDY_ROOT
    assert node.get_query_model(QueryLevel.IMAGE) == QueryModel.STUDY_ROOT
    assert node.get_abstract_syntaxes(DicomOpType.ECHO) == [sop_class.Verification]
    assert node.get_abstract_syntaxes(
        DicomOpType.FIND, query_model=QueryModel.PATIENT_ROOT
    ) == [sop_class.PatientRootQueryRetrieveInformationModelFind]
    assert node.get_abstract_syntaxes(
        DicomOpType.MOVE, query_model=QueryModel.PATIENT_ROOT
    ) == [sop_class.PatientRootQueryRetrieveInformationModelMove]
    with raises(UnsupportedOperationError):
        node.get_abstract_syntaxes(DicomOpType.GET, query_model=QueryModel.PATIENT_ROOT)
    assert node.get_abstract_syntaxes(
        DicomOpType.FIND, query_model=QueryModel.STUDY_ROOT
    ) == [sop_class.StudyRootQueryRetrieveInformationModelFind]
    assert node.get_abstract_syntaxes(
        DicomOpType.MOVE, query_model=QueryModel.STUDY_ROOT
    ) == [sop_class.StudyRootQueryRetrieveInformationModelMove]
    with raises(UnsupportedOperationError):
        node.get_abstract_syntaxes(DicomOpType.GET, query_model=QueryModel.STUDY_ROOT)
    storescp_syntaxes = node.get_abstract_syntaxes(
        DicomOpType.STORE, role=DicomRole.PROVIDER
    )
    assert len(storescp_syntaxes) == len(sop_class._STORAGE_CLASSES)
    storescu_syntaxes = node.get_abstract_syntaxes(
        DicomOpType.STORE, role=DicomRole.USER
    )
    assert len(storescu_syntaxes) == 128
    assert storescu_syntaxes[-1] == SOPClass(node.private_store_classes[-1][1])
    storescu_pcs = node.get_presentation_contexts(storescu_syntaxes)
    assert len(storescu_pcs) == len(storescu_syntaxes)
    assert len(storescu_pcs[0].transfer_syntax) == len(node.transfer_syntaxes)
    json_dict = node.to_json_dict()
    node2 = RemoteNode.from_json_dict(json_dict)
    assert node == node2
