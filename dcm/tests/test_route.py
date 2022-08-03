import asyncio

import pydicom
import pytest
from pytest import mark

from ..query import QueryLevel
from ..route import StaticRoute, DynamicRoute, Router

from .conftest import has_dcmtk


def make_id_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == "TestPatient1":
            return [dest1]
        else:
            return [dest2]

    return lookup_func


@mark.parametrize(
    "node_type, node_subsets",
    (
        pytest.param("dcmtk", ["all", None, None, None], marks=has_dcmtk),
        ("pnd", ["all", None, None, None]),
        ("qr", ["all", None, None, None]),
    ),
)
@mark.asyncio
async def test_pre_route(make_local_node, make_repo, node_subsets):
    local_node = make_local_node()
    src_repo, _ = await make_repo(local_node, subset=node_subsets[0])
    dest1_repo, _ = await make_repo(local_node, subset=node_subsets[1])
    dest2_repo, _ = await make_repo(local_node, subset=node_subsets[2])
    dest3_repo, _ = await make_repo(local_node, subset=node_subsets[3])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(
        make_id_lookup(dest2_repo, dest3_repo), required_elems=["PatientID"]
    )
    router = Router([static_route, dyn_route])
    res = await router.pre_route(src_repo)
    for routes, qr in res.items():
        print("%s -> %s" % ([str(r) for r in routes], qr))
        assert all(isinstance(r, StaticRoute) for r in routes)
        assert static_route in routes
        resolved = [r for r in routes if r != static_route]
        pat_ids = list(qr.patients())
        if "TestPatient1" in pat_ids:
            assert len(pat_ids) == 1
            assert len(resolved) == 1
            dests = resolved[0].get_dests(None)
            assert len(dests) == 1
            assert dests[0] == dest2_repo
        else:
            assert len(resolved) == 1
            dests = resolved[0].get_dests(None)
            assert len(dests) == 1
            assert dests[0] == dest3_repo


def make_echo_lookup(dest1, dest2):
    def lookup_func(ds):
        echo_time = float(getattr(ds, "EchoTime", "100.0"))
        if echo_time > 10.0:
            return [dest1]
        else:
            return [dest2]

    return lookup_func


@mark.parametrize(
    "node_type, node_subsets",
    (
        pytest.param("dcmtk", ["all", None, None, None], marks=has_dcmtk),
        ("pnd", ["all", None, None, None]),
        ("qr", ["all", None, None, None]),
    ),
)
@mark.asyncio
async def test_pre_route_with_dl(make_local_node, make_repo, node_subsets):
    local_node = make_local_node()
    src_repo, src_node = await make_repo(local_node, subset=node_subsets[0])
    dest1_repo, _ = await make_repo(local_node, subset=node_subsets[1])
    dest2_repo, _ = await make_repo(local_node, subset=node_subsets[2])
    dest3_repo, _ = await make_repo(local_node, subset=node_subsets[3])
    static_route = StaticRoute([dest1_repo])
    # Setup a dynamic route where we route on an element that can't be queried for
    # thus forcing the router to download example data sets
    dyn_route = DynamicRoute(
        make_echo_lookup(dest2_repo, dest3_repo),
        route_level=QueryLevel.SERIES,
        required_elems=["EchoTime"],
    )
    router = Router([static_route, dyn_route])
    res = await router.pre_route(src_repo)
    for routes, qr in res.items():
        print("%s -> %s" % ([str(r) for r in routes], qr))
        assert all(isinstance(r, StaticRoute) for r in routes)
        assert static_route in routes
        resolved = [r for r in routes if r != static_route]
        assert len(resolved) == 1
        assert len(resolved[0].dests) == 1
        dyn_dest = resolved[0].dests[0]
        pat_ids = list(qr.patients())
        for pat_id in pat_ids:
            studies = list(qr.studies(pat_id))
            for study_uid in studies:
                series = list(qr.series(study_uid))
                for series_uid in series:
                    instances = list(src_node.init_qr.instances(series_uid))
                    example_path = list(src_node.store_dir.glob(f"{instances[0]}*"))[0]
                    assert example_path.exists()
                    with open(example_path, "rb") as f:
                        ds = pydicom.dcmread(f)
                    echo_time = float(getattr(ds, "EchoTime", "100.0"))
                    if echo_time > 10.0:
                        assert dyn_dest == dest2_repo
                    else:
                        assert dyn_dest == dest3_repo
