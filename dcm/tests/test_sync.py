import asyncio
from pathlib import Path

from pytest import fixture, mark

from ..net import LocalEntity
from ..route import StaticRoute, DynamicRoute, Router
from ..sync import TransferPlanner
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir

from .conftest import (dicom_files, has_dcmtk, dcmtk_test_nodes, make_local_factory, DATA_DIR)


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == 'TestPatient1':
            return [dest1]
        else:
            return [dest2]
    return lookup_func



@mark.parametrize('node_subsets', [['all', None, None, None]])
@mark.asyncio
@has_dcmtk
async def test_gen_transfers(make_local_node, make_dcmtk_net_repo, node_subsets):
    local_node = make_local_node()
    src_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[0])
    dest1_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[1])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[2])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[3])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo), required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as executor:
        async for transfer in tp.gen_transfers():
            pass


@mark.parametrize('node_subsets', [['all', None, None, None]])
@mark.asyncio
@has_dcmtk
async def test_repo_sync(make_local_node, make_dcmtk_net_repo, node_subsets):
    local_node = make_local_node()
    src_repo, full_qr, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[0])
    dest1_repo, _, dest1_dir = make_dcmtk_net_repo(local_node, subset=node_subsets[1])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[2])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[3])
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo), required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as e:
        async for transfer in tp.gen_transfers():
            await e.exec_transfer(transfer)
    dest1_dir = Path(dest1_dir)
    found_files = [x for x in dest1_dir.glob('**/*.dcm')]
    print(found_files)
    assert len(found_files) == len(full_qr)
    #TODO: Check that dynamic routing worked correctly


@mark.parametrize('node_subsets', [['all', None, None, None]])
@mark.asyncio
@has_dcmtk
async def test_bucket_sync(make_local_dir, make_local_node, make_dcmtk_net_repo, node_subsets):
    src_bucket, init_qr, _ = make_local_dir('all', max_chunk=2)
    local_node = make_local_node()
    dest1_repo, _, dest1_dir = make_dcmtk_net_repo(local_node, subset=node_subsets[1])
    dest2_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[2])
    dest3_repo, _, _ = make_dcmtk_net_repo(local_node, subset=node_subsets[3])
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
