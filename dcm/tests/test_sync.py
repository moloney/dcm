import asyncio
from pathlib import Path

from pytest import fixture, mark

from ..query import QueryResult, QueryLevel
from ..net import LocalEntity
from ..route import StaticRoute, DynamicRoute, Router
from ..sync import TransferPlanner
from ..store import TransferMethod
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir

from .conftest import has_dcmtk


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == 'TestPat1':
            return [dest1]
        else:
            return [dest2]
    return lookup_func


dest_subsets = [[None] * 3,
                ['all'] * 3,
                ['PATIENT-0'] * 3,
                ['PATIENT-0/STUDY-0'] * 3,
                ['PATIENT-0/STUDY-0/SERIES-0'] * 3,
                ['PATIENT-0/STUDY-0/SERIES-0/IMAGE-0'] * 3,
                ['PATIENT-1'] * 3,
                ]


@mark.parametrize('subset_specs', dest_subsets)
@mark.asyncio
@has_dcmtk
async def test_gen_transfers(make_local_node, make_dcmtk_net_repo, subset_specs):
    local_node = make_local_node()
    src_repo, _, _ = make_dcmtk_net_repo(local_node, subset='all')
    dest1_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[0])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[1])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=subset_specs[2])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo), required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as executor:
        async for transfer in tp.gen_transfers():
            for route in transfer.method_routes_map[TransferMethod.PROXY]:
                for dest in route.dests:
                    print(transfer.chunk.qr)


@mark.parametrize('subset_specs', dest_subsets)
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
                    print(f"{dest} : {transfer.chunk.qr}")
            await e.exec_transfer(transfer)
        print(e.report)
    dest1_dir = Path(dest1_dir)
    found_files = [x for x in dest1_dir.glob('**/*.dcm')]
    print(found_files)
    assert len(found_files) == len(full_qr)


@mark.parametrize('subset_specs', dest_subsets)
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


@mark.parametrize('subset_specs', dest_subsets)
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
