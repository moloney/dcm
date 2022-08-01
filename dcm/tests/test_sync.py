from pathlib import Path
from copy import deepcopy

import pytest
from pytest import mark

from ..query import QueryResult, QueryLevel
from ..route import StaticRoute, DynamicRoute
from ..sync import SyncManager
from ..store.base import TransferMethod
from ..util import json_serializer

from .conftest import (
    has_dcmtk,
    dcmtk_priv_sop_retr_xfail,
    dcmtk_priv_sop_send_xfail,
    pnd_priv_sop_xfail,
    get_stored_files,
)


priv_sop_marks = [dcmtk_priv_sop_retr_xfail, dcmtk_priv_sop_send_xfail]


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == "TestPat1":
            return [dest1]
        else:
            return [dest2]

    return lookup_func


gen_transfer_sets = [
    [None] * 3,
    ["all"] * 3,
    ["PATIENT-0"] * 3,
    ["PATIENT-0/STUDY-0"] * 3,
    ["PATIENT-0/STUDY-0/SERIES-0"] * 3,
    ["PATIENT-0/STUDY-0/SERIES-0/IMAGE-0"] * 3,
    ["PATIENT-1"] * 3,
]


def get_gen_transfer_sets():
    res = []
    for node_type in ("dcmtk", "pnd"):
        if node_type == "dcmtk":
            for sub in gen_transfer_sets:
                res.append(pytest.param(node_type, sub, marks=has_dcmtk))
        else:
            assert node_type == "pnd"
            for sub in gen_transfer_sets:
                res.append((node_type, sub))
    return res


sync_subsets = [
    [None] * 3,
    ["all"] * 3,
    ["PATIENT-0"] * 3,
    ["PATIENT-0/STUDY-1"] * 3,
    ["PATIENT-0/STUDY-0"] * 3,
    ["PATIENT-0/STUDY-0/SERIES-0"] * 3,
    ["PATIENT-0/STUDY-0/SERIES-0/IMAGE-0"] * 3,
    ["PATIENT-1"] * 3,
]


def get_repo_to_repo_subsets():
    res = []
    for node_type in ("dcmtk", "pnd"):
        if node_type == "dcmtk":
            for sub in sync_subsets:
                marks = [has_dcmtk]
                if sub[0] not in ("all", "PATIENT-0", "PATIENT-0/STUDY-1"):
                    marks += priv_sop_marks
                res.append(pytest.param(node_type, sub, marks=marks))
        else:
            assert node_type == "pnd"
            for sub in sync_subsets:
                res.append((node_type, sub))
    return res


def get_bucket_to_repo_subsets():
    res = []
    for node_type in ("dcmtk", "pnd"):
        if node_type == "dcmtk":
            for sub in sync_subsets:
                marks = [has_dcmtk] + priv_sop_marks
                res.append(pytest.param(node_type, sub, marks=marks))
        else:
            assert node_type == "pnd"
            for sub in sync_subsets:
                res.append(pytest.param(node_type, sub, marks=pnd_priv_sop_xfail))
    return res


@mark.parametrize("node_type, subset_specs", get_gen_transfer_sets())
@mark.asyncio
async def test_gen_transfers(make_local_node, make_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, src_node = await make_net_repo(local_node, subset="all")
    dest1_repo, dest1_node = await make_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, dest2_node = await make_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, dest3_node = await make_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_lookup = make_lookup(dest2_repo, dest3_repo)
    dyn_route = DynamicRoute(dyn_lookup, required_elems=["PatientID"])
    dests = [static_route, dyn_route]

    # Build QRs of what we expect to be transfered to each dest
    expect_qrs = {
        dest1_repo: src_node.init_qr - dest1_node.init_qr,
        dest2_repo: QueryResult(QueryLevel.IMAGE),
        dest3_repo: QueryResult(QueryLevel.IMAGE),
    }
    for ds in src_node.init_qr:
        dyn_dests = dyn_lookup(ds)
        for dest in dyn_dests:
            expect_qrs[dest].add(ds)
    trans_qrs = {}
    async with SyncManager(src_repo, dests) as sm:
        async for transfer in sm.gen_transfers():
            trans_level = transfer.chunk.qr.level
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(f"\n{dest} :\n{transfer.chunk.qr.to_tree()}")
                    if dest not in trans_qrs:
                        trans_qrs[dest] = {}
                    if trans_level not in trans_qrs[dest]:
                        trans_qrs[dest][trans_level] = deepcopy(transfer.chunk.qr)
                    else:
                        for ds in transfer.chunk.qr:
                            # Check this data is expected
                            assert ds in expect_qrs[dest]
                            # Check for duplicate transfers
                            for lvl_qr in trans_qrs[dest].values():
                                assert ds not in lvl_qr
                            trans_qrs[dest][trans_level].add(ds)


@mark.parametrize("node_type, subset_specs", get_repo_to_repo_subsets())
@mark.asyncio
async def test_repo_sync_single_static(make_local_node, make_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, src_node = await make_net_repo(local_node, subset="all")
    dest1_repo, dest1_node = await make_net_repo(local_node, subset=subset_specs[0])
    static_route = StaticRoute([dest1_repo])
    dests = [static_route]
    async with SyncManager(src_repo, dests) as sm:
        async for transfer in sm.gen_transfers():
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(f"{dest} : {json_serializer.dumps(transfer.chunk.qr)}")
            await sm.exec_transfer(transfer)
        print(sm.report)
    found_files = get_stored_files(dest1_node.store_dir)
    print(found_files)
    assert len(found_files) == len(src_node.init_qr)


@mark.parametrize("node_type, subset_specs", get_repo_to_repo_subsets())
@mark.asyncio
async def test_repo_sync_multi(make_local_node, make_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, src_node = await make_net_repo(local_node, subset="all")
    dest1_repo, dest1_node = await make_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, _ = await make_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, _ = await make_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(
        make_lookup(dest2_repo, dest3_repo), required_elems=["PatientID"]
    )
    dests = [static_route, dyn_route]
    async with SyncManager(src_repo, dests) as sm:
        async for transfer in sm.gen_transfers():
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(f"{dest} : {transfer.chunk.qr}")
            await sm.exec_transfer(transfer)
        print(sm.report)
    found_files = get_stored_files(dest1_node.store_dir)
    print(found_files)
    assert len(found_files) == len(src_node.init_qr)
    # TODO: Check that dynamic routing worked correctly


@mark.parametrize("node_type, subset_specs", get_bucket_to_repo_subsets())
@mark.asyncio
async def test_bucket_sync(
    make_local_dir, make_local_node, make_net_repo, subset_specs
):
    src_bucket, init_qr, _ = make_local_dir("all", max_chunk=2)
    local_node = make_local_node()
    dest1_repo, dest1_node = await make_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, _ = await make_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, _ = await make_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(
        make_lookup(dest2_repo, dest3_repo), required_elems=["PatientID"]
    )
    dests = [static_route, dyn_route]
    async with SyncManager(src_bucket, dests) as sm:
        async for transfer in sm.gen_transfers():
            await sm.exec_transfer(transfer)
    found_files = get_stored_files(dest1_node.store_dir)
    print(found_files)
    assert len(found_files) == len(init_qr)
