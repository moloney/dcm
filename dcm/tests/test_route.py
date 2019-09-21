import asyncio

from pytest import fixture, mark

from ..net import LocalEntity
from ..route import StaticRoute, DynamicRoute, Router
from ..store.net_repo import NetRepo

from .conftest import (local_nodes, dicom_files, has_dcmtk, dcmtk_test_nodes, make_local_factory, DATA_DIR)


def make_lookup(dest1, dest2):
    def lookup_func(ds):
        if ds.PatientID == 'TestPatient1':
            return [dest1]
        else:
            return [dest2]
    return lookup_func


@has_dcmtk
@mark.parametrize('dcmtk_test_nodes', [['all', None, None, None]], indirect=True)
def test_pre_route(dcmtk_test_nodes):
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
    router = Router([static_route, dyn_route])
    res = asyncio.run(router.pre_route(src_repo))
    for routes, qr in res.items():
        print("%s -> %s" % ([str(r) for r in routes], qr))
        assert all(isinstance(r, StaticRoute) for r in routes)
        assert static_route in routes
        resolved = [r for r in routes if r != static_route]
        pat_ids = list(qr.patients())
        if 'TestPatient1' in pat_ids:
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

# TODO: Add a test where we need to fetch example data to resolve the dynamic route