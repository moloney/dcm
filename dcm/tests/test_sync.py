import asyncio
from pathlib import Path
from copy import deepcopy

import pytest
from pytest import fixture, mark

from ..query import QueryResult, QueryLevel
from ..net import LocalEntity
from ..route import StaticRoute, DynamicRoute, Router
from ..sync import TransferPlanner
from ..store import TransferMethod
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir
from ..util import serializer

from .conftest import (has_dcmtk, DCMTK_VERSION, dcmtk_priv_sop_retr_xfail,
                       dcmtk_priv_sop_send_xfail)


priv_sop_marks = [dcmtk_priv_sop_retr_xfail, dcmtk_priv_sop_send_xfail]


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == 'TestPat1':
            return [dest1]
        else:
            return [dest2]
    return lookup_func


repo_to_repo_subsets = [pytest.param([None] * 3, marks=priv_sop_marks),
                        ['all'] * 3,
                        ['PATIENT-0'] * 3,
                        ['PATIENT-0/STUDY-1'] * 3,
                        pytest.param(['PATIENT-0/STUDY-0'] * 3, marks=priv_sop_marks),
                        pytest.param(['PATIENT-0/STUDY-0/SERIES-0'] * 3, marks=priv_sop_marks),
                        pytest.param(['PATIENT-0/STUDY-0/SERIES-0/IMAGE-0'] * 3, marks=priv_sop_marks),
                        pytest.param(['PATIENT-1'] * 3, marks=priv_sop_marks),
                        ]

bucket_to_repo_subsets = [pytest.param([None] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['all'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-0'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-0/STUDY-1'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-0/STUDY-0'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-0/STUDY-0/SERIES-0'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-0/STUDY-0/SERIES-0/IMAGE-0'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          pytest.param(['PATIENT-1'] * 3, marks=dcmtk_priv_sop_send_xfail),
                          ]


@mark.parametrize('subset_specs',
                  [[None] * 3,
                   ['all'] * 3,
                   ['PATIENT-0'] * 3,
                   ['PATIENT-0/STUDY-0'] * 3,
                   ['PATIENT-0/STUDY-0/SERIES-0'] * 3,
                   ['PATIENT-0/STUDY-0/SERIES-0/IMAGE-0'] * 3,
                   ['PATIENT-1'] * 3,
                   ])
@mark.asyncio
@has_dcmtk
async def test_gen_transfers(make_local_node, make_dcmtk_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, full_qr, _ = make_dcmtk_net_repo(local_node, subset='all')
    dest1_repo, dest1_init_qr, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, dest2_init_qr, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, dest3_init_qr, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_lookup = make_lookup(dest2_repo, dest3_repo)
    dyn_route = DynamicRoute(dyn_lookup, required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)

    # Build QRs of what we expect to be transfered to each dest
    expect_qrs = {dest1_repo: full_qr - dest1_init_qr,
                  dest2_repo: QueryResult(QueryLevel.IMAGE),
                  dest3_repo: QueryResult(QueryLevel.IMAGE)
                  }
    for ds in full_qr:
        dests = dyn_lookup(ds)
        for dest in dests:
            expect_qrs[dest].add(ds)
    trans_qrs = {}
    async with tp.executor() as executor:
        async for transfer in tp.gen_transfers():
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


@mark.parametrize('subset_specs', repo_to_repo_subsets)
@mark.asyncio
@has_dcmtk
async def test_repo_sync_single_static(make_local_node, make_dcmtk_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, full_qr, _ = make_dcmtk_net_repo(local_node, subset='all')
    dest1_repo, _, dest1_dir = make_dcmtk_net_repo(local_node, subset=subset_specs[0])
    static_route = StaticRoute([dest1_repo])
    dests = [static_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as e:
        async for transfer in tp.gen_transfers():
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(f"{dest} : {serializer.dumps(transfer.chunk.qr)}")
            await e.exec_transfer(transfer)
        print(e.report)
    dest1_dir = Path(dest1_dir)
    found_files = [x for x in dest1_dir.glob('**/*.dcm')]
    print(found_files)
    assert len(found_files) == len(full_qr)


@mark.parametrize('subset_specs', repo_to_repo_subsets)
@mark.asyncio
@has_dcmtk
async def test_repo_sync_multi(make_local_node, make_dcmtk_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, full_qr, _ = make_dcmtk_net_repo(local_node, subset='all')
    dest1_repo, _, dest1_dir = make_dcmtk_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo), required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as e:
        async for transfer in tp.gen_transfers():
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(f"{dest} : {transfer.chunk.qr}")
            await e.exec_transfer(transfer)
        print(e.report)
    dest1_dir = Path(dest1_dir)
    found_files = [x for x in dest1_dir.glob('**/*.dcm')]
    print(found_files)
    assert len(found_files) == len(full_qr)
    #TODO: Check that dynamic routing worked correctly


@mark.parametrize('subset_specs', bucket_to_repo_subsets)
@mark.asyncio
@has_dcmtk
async def test_bucket_sync(make_local_dir, make_local_node, make_dcmtk_net_repo, subset_specs):
    src_bucket, init_qr, _ = make_local_dir('all', max_chunk=2)
    local_node = make_local_node()
    dest1_repo, _, dest1_dir = make_dcmtk_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo),
                             required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_bucket, dests)
    async with tp.executor() as e:
        async for transfer in tp.gen_transfers():
            await e.exec_transfer(transfer)
    dest1_dir = Path(dest1_dir)
    found_files = [x for x in dest1_dir.glob('**/*.dcm')]
    print(found_files)
    assert len(found_files) == len(init_qr)
