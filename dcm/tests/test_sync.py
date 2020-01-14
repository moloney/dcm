import asyncio
from pathlib import Path

from pytest import fixture, mark

from ..net import LocalEntity
from ..route import StaticRoute, DynamicRoute, Router
from ..sync import TransferPlanner
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir

from .conftest import (local_nodes, dicom_files, has_dcmtk, dcmtk_test_nodes, make_local_factory, DATA_DIR)


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == 'TestPatient1':
            return [dest1]
        else:
            return [dest2]
    return lookup_func



@mark.parametrize('dcmtk_test_nodes', [['all', None, None, None]], indirect=True)
@mark.asyncio
@has_dcmtk
async def test_gen_transfers(dcmtk_test_nodes):
    src, file_set, _ = dcmtk_test_nodes[0]
    dest1, _, _ = dcmtk_test_nodes[1]
    dest2, _, _ = dcmtk_test_nodes[2]
    dest3, _, _ = dcmtk_test_nodes[3]
    src_repo = NetRepo(local_nodes[0], src)
    dest1_repo = NetRepo(local_nodes[0], dest1)
    dest2_repo = NetRepo(local_nodes[0], dest2)
    dest3_repo = NetRepo(local_nodes[0], dest3)
    static_route = StaticRoute([dest1_repo])
    dyn_route = DynamicRoute(make_lookup(dest2_repo, dest3_repo), required_elems=['PatientID'])
    dests = [static_route, dyn_route]
    tp = TransferPlanner(src_repo, dests)
    async with tp.executor() as executor:
        async for transfer in tp.gen_transfers():
            pass


@mark.parametrize('dcmtk_test_nodes', [['all', None, None, None]], indirect=True)
@mark.asyncio
@has_dcmtk
async def test_repo_sync(dcmtk_test_nodes):
    src, file_set, _ = dcmtk_test_nodes[0]
    dest1, _, dest1_dir = dcmtk_test_nodes[1]
    dest2, _, dest2_dir = dcmtk_test_nodes[2]
    dest3, _, dest3_dir = dcmtk_test_nodes[3]
    src_repo = NetRepo(local_nodes[0], src)
    dest1_repo = NetRepo(local_nodes[0], dest1)
    dest2_repo = NetRepo(local_nodes[0], dest2)
    dest3_repo = NetRepo(local_nodes[0], dest3)
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
    assert len(found_files) == len(file_set)
    #TODO: Check that dynamic routing worked correctly


@mark.parametrize('dcmtk_test_nodes', [[None, None, None]], indirect=True)
@mark.asyncio
@has_dcmtk
async def test_bucket_sync(dcmtk_test_nodes):
    src_bucket = LocalDir(DATA_DIR)
    dest1, _, dest1_dir = dcmtk_test_nodes[0]
    dest2, _, dest2_dir = dcmtk_test_nodes[1]
    dest3, _, dest3_dir = dcmtk_test_nodes[2]
    dest1_repo = NetRepo(local_nodes[0], dest1)
    dest2_repo = NetRepo(local_nodes[0], dest2)
    dest3_repo = NetRepo(local_nodes[0], dest3)
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
    #assert len(found_files) == len(file_set)
