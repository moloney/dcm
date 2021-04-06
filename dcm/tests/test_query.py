import random, itertools
from pytest import fixture, mark
from pydicom.dataset import Dataset

from ..query import QueryLevel, QueryResult, uid_elems, req_elems


def make_dataset(attrs=None, level=QueryLevel.IMAGE):
    if attrs is None:
        attrs = {}
    ds = Dataset()
    for lvl in QueryLevel:
        for attr in req_elems[lvl]:
            val = attrs.get(attr)
            if val is None:
                if attr == "PatientID":
                    val = "test_id"
                elif attr == "PatientName":
                    val = "test_name"
                elif attr == "StudyInstanceUID":
                    val = "1.2.3." + str(random.randint(1, 1000000000))
                elif attr == "StudyDate":
                    val = "20150101"
                elif attr == "StudyTime":
                    val = "120101.1"
                elif attr == "SeriesInstanceUID":
                    val = "1.2.3." + str(random.randint(1, 1000000000))
                elif attr == "SeriesNumber":
                    val = 1
                elif attr == "Modality":
                    val = "MR"
                elif attr == "SOPInstanceUID":
                    val = "1.2.3." + str(random.randint(1, 1000000000))
                elif attr == "InstanceNumber":
                    val = 1
            setattr(ds, attr, val)
        if lvl == level:
            break
    return ds


@fixture()
def hierarchy_data(request):
    level, num_patients, num_studies, num_series, num_instances, n_dupes = request.param
    data_sets = []
    pat_ids = []
    study_uids = []
    series_uids = []
    inst_uids = []

    curr_attrs = {}
    for pat_idx in range(num_patients):
        pat_id = "testid_%d" % pat_idx
        pat_ids.append(pat_id)
        curr_attrs["PatientID"] = pat_id
        if level == QueryLevel.PATIENT:
            data_sets.append(make_dataset(curr_attrs))
            continue
        for study_idx in range(num_studies):
            study_uid = "1.2.3.%d.%d" % (pat_idx, study_idx)
            study_uids.append(study_uid)
            curr_attrs["StudyInstanceUID"] = study_uid
            if level == QueryLevel.STUDY:
                data_sets.append(make_dataset(curr_attrs))
                continue
            for series_idx in range(num_series):
                series_uid = study_uid + ".%d" % series_idx
                series_uids.append(series_uid)
                curr_attrs["SeriesInstanceUID"] = series_uid
                if level == QueryLevel.SERIES:
                    data_sets.append(make_dataset(curr_attrs))
                    continue
                for inst_idx in range(num_instances):
                    inst_uid = series_uid + ".%d" % inst_idx
                    inst_uids.append(inst_uid)
                    curr_attrs["SOPInstanceUID"] = inst_uid
                    assert level == QueryLevel.IMAGE
                    data_sets.append(make_dataset(curr_attrs))
    for dupe_idx in range(n_dupes):
        data_sets.append(data_sets[dupe_idx % len(data_sets)])
    return level, data_sets, pat_ids, study_uids, series_uids, inst_uids


add_rm_data_params = list(
    itertools.product(QueryLevel, [1, 2], [1, 3], [3, 4], [5, 6], [0, 3])
)


@mark.parametrize("hierarchy_data", add_rm_data_params, indirect=True)
def test_add_remove(hierarchy_data):
    """Test basic add/remove operations on a QueryResult"""
    level, data_sets, pat_ids, study_uids, series_uids, inst_uids = hierarchy_data
    for lvl in QueryLevel:
        if lvl > level:
            break
        qr = QueryResult(lvl)
        for ds in data_sets:
            qr.add(ds)
        assert sorted(x for x in qr.patients()) == sorted(pat_ids)
        if lvl > QueryLevel.PATIENT:
            assert sorted(x for x in qr.studies()) == sorted(study_uids)
            for pat_id in qr.patients():
                pat_idx = int(pat_id.split("_")[-1])
                uid_prefix = "1.2.3.%d" % pat_idx
                for study_uid in qr.studies(pat_id):
                    assert study_uid in study_uids
                    assert study_uid.startswith(uid_prefix)
        if lvl > QueryLevel.STUDY:
            assert sorted(x for x in qr.series()) == sorted(series_uids)
            for study_uid in qr.studies():
                for series_uid in qr.series(study_uid):
                    assert series_uid.startswith(study_uid)
        if lvl > QueryLevel.SERIES:
            assert sorted(x for x in qr.instances()) == sorted(inst_uids)
            for series_uid in qr.series():
                for instance_uid in qr.instances(series_uid):
                    assert instance_uid.startswith(series_uid)
        for ds in data_sets:
            if ds in qr:
                qr.remove(ds)
        for ds in data_sets:
            assert ds not in qr
        assert len(qr) == 0
        assert len([x for x in qr.patients()]) == 0
        if lvl > QueryLevel.PATIENT:
            assert len([x for x in qr.studies()]) == 0
        if lvl > QueryLevel.STUDY:
            assert len([x for x in qr.series()]) == 0
        if lvl > QueryLevel.SERIES:
            assert len([x for x in qr.instances()]) == 0


walk_data_params = list(
    itertools.product(QueryLevel, [1, 3], [1, 5], [1, 7], [1, 11], [0])
)


@mark.parametrize("hierarchy_data", walk_data_params, indirect=["hierarchy_data"])
def test_walk(hierarchy_data):
    """Test the QueryResult.walk generator"""
    level, data_sets, pat_ids, study_uids, series_uids, inst_uids = hierarchy_data
    qr = QueryResult(level)
    for ds in data_sets:
        qr.add(ds)
    seen_sets = {lvl: set() for lvl in QueryLevel}
    for path, sub_uids in qr.walk():
        curr_uid = path.uids[-1]
        assert curr_uid not in seen_sets[path.level]
        seen_sets[path.level].add(curr_uid)
        for lvl, uid in enumerate(path.uids):
            assert uid in seen_sets[lvl]
        assert qr.get_path(path.end) == path
    assert sorted(seen_sets[QueryLevel.PATIENT]) == sorted(pat_ids)
    assert sorted(seen_sets[QueryLevel.STUDY]) == sorted(study_uids)
    assert sorted(seen_sets[QueryLevel.SERIES]) == sorted(series_uids)
    assert sorted(seen_sets[QueryLevel.IMAGE]) == sorted(inst_uids)


@mark.parametrize("hierarchy_data", walk_data_params, indirect=["hierarchy_data"])
def test_equality(hierarchy_data):
    level, data_sets, pat_ids, study_uids, series_uids, inst_uids = hierarchy_data
    qr1 = QueryResult(level)
    qr2 = QueryResult(level)
    for ds in data_sets:
        qr1.add(ds)
        qr2.add(ds)
    assert qr1 == qr2
    qr2.remove(data_sets[-1])
    assert qr1 != qr2
    # TODO: Include tests with sub-counts


@mark.parametrize("hierarchy_data", walk_data_params, indirect=["hierarchy_data"])
def test_json(hierarchy_data):
    level, data_sets, pat_ids, study_uids, series_uids, inst_uids = hierarchy_data
    qr1 = QueryResult(level)
    for ds in data_sets:
        qr1.add(ds)
    json_dict = qr1.to_json_dict()
    qr2 = QueryResult.from_json_dict(json_dict)
    assert qr1 == qr2


def test_intersect():
    data = [make_dataset() for _ in range(5)]
    qr1 = QueryResult(QueryLevel.IMAGE)
    for ds in data[:3]:
        qr1.add(ds)
    qr2 = QueryResult(QueryLevel.IMAGE)
    assert len(qr1 & qr2) == 0
    for ds in data[3:]:
        qr2.add(ds)
    assert len(qr1 & qr2) == 0
    qr2.add(data[2])
    res = qr1 & qr2
    assert len(res) == 1
    assert data[2] in res


def test_intersect_multi_level():
    series_ds = make_dataset(
        {
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.0",
        },
        level=QueryLevel.SERIES,
    )
    inst_ds = make_dataset(
        {
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.0",
            "SOPInstanceUID": "1.2.3.4.0.0",
        },
        level=QueryLevel.IMAGE,
    )
    excl_ds = make_dataset(
        {
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.1",
            "SOPInstanceUID": "1.2.3.4.1.0",
        },
        level=QueryLevel.IMAGE,
    )
    series_qr = QueryResult(QueryLevel.SERIES)
    series_qr.add(series_ds)
    inst_qr = QueryResult(QueryLevel.IMAGE)
    inst_qr.add(inst_ds)
    inst_qr.add(excl_ds)
    assert len(inst_qr) == 2
    res1 = series_qr & inst_qr
    res2 = inst_qr & series_qr
    assert res1 == res2
    assert res1.level == res2.level == QueryLevel.IMAGE
    assert len(res1) == 1
    assert inst_ds in res1
    assert excl_ds not in res1


def test_union():
    data = [make_dataset() for _ in range(5)]
    qr1 = QueryResult(QueryLevel.IMAGE)
    for ds in data[:3]:
        qr1.add(ds)
    qr2 = QueryResult(QueryLevel.IMAGE)
    assert qr1 | qr2 == qr1
    for ds in data[3:]:
        qr2.add(ds)
    res1 = qr1 | qr2
    assert len(res1) == len(qr1) + len(qr2)
    qr2.add(data[2])
    res2 = qr1 | qr2
    assert res1 == res2


def test_subtract():
    data = [make_dataset() for _ in range(5)]
    qr1 = QueryResult(QueryLevel.IMAGE)
    for ds in data[:3]:
        qr1.add(ds)
    qr2 = QueryResult(QueryLevel.IMAGE)
    assert qr1 - qr2 == qr1
    for ds in data[3:]:
        qr2.add(ds)
    assert qr1 - qr2 == qr1
    qr2.add(data[2])
    res = qr1 - qr2
    assert len(res) == len(qr1) - 1
    assert data[2] not in res
