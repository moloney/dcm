import os, time, shutil, tarfile, logging, re, asyncio
from dataclasses import dataclass
from copy import deepcopy
import subprocess as sp
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path
from typing import BinaryIO, Optional

import pydicom
from pynetdicom import AE
import psutil
import pytest
from pytest import fixture, mark

from ..conf import _default_conf, DcmConfig
from ..query import QueryLevel, QueryResult
from ..net import DcmNode, _make_default_store_scu_pcs
from ..store.base import IndexInitMode
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir
from ..store.qr_repo import QrRepo


logging_opts = {}


optional_markers = {
    "slow": {
        "help": "Run slow tests",
        "marker-descr": "Mark a test as being slow",
        "skip-reason": "Test only runs with the --{} option.",
    },
}


def pytest_addoption(parser):
    for marker, info in optional_markers.items():
        parser.addoption(
            "--{}".format(marker), action="store_true", default=False, help=info["help"]
        )
    parser.addoption(
        "--show-pynetdicom-logs",
        default=False,
        action="store_true",
        help="show pynetdicom logs",
    )
    parser.addoption(
        "--disable-backend",
        help="Disable testing against the specified backend ('dcmtk' or 'pnd')",
    )


def pytest_configure(config):
    for marker, info in optional_markers.items():
        config.addinivalue_line(
            "markers", "{}: {}".format(marker, info["marker-descr"])
        )
    if not config.getoption("--show-pynetdicom-logs"):
        for pnd_mod in ("events", "assoc", "dsutils", "acse"):
            lgr = logging.getLogger("pynetdicom.%s" % pnd_mod)
            lgr.setLevel(logging.WARN)


def pytest_collection_modifyitems(config, items):
    for marker, info in optional_markers.items():
        if not config.getoption("--{}".format(marker)):
            skip_test = mark.skip(reason=info["skip-reason"].format(marker))
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip_test)


DATA_DIR = Path(__file__).parent / "data"


PC_CONF_PATH = DATA_DIR / "pres_contexts.cfg"


DICOM_TAR = DATA_DIR / "dicom.tar.bz2"


@fixture(scope="session")
def dicom_dir():
    """Decompress DICOM files into a temp directory"""
    with TemporaryDirectory(prefix="dcm-test") as temp_dir:
        # Unpack tar to temp dir and yield list of paths
        with tarfile.open(DICOM_TAR, "r:bz2") as tf:
            tf.extractall(temp_dir)
        temp_dir = Path(temp_dir)
        yield temp_dir


@fixture(scope="session")
def dicom_files(dicom_dir):
    """Decompress DICOM files into a temp directory and return their paths"""
    return [x for x in dicom_dir.glob("**/*.dcm")]


@fixture(scope="session")
def dicom_data(dicom_files):
    """QueryResult and list of (path, dataset) tuples"""
    res_data = []
    res_qr = QueryResult(QueryLevel.IMAGE)
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(str(dcm_path))
        res_data.append((dcm_path, ds))
        res_qr.add(ds)
    return (res_qr, res_data)


# TODO: get rid of this in favor of above func
@fixture
def dicom_files_w_qr(dicom_files):
    qr = QueryResult(QueryLevel.IMAGE)
    for path in dicom_files:
        ds = pydicom.dcmread(str(path))
        qr.add(ds)
    return (dicom_files, qr)


@fixture
def get_dicom_subset(dicom_data):
    """Factory fixture for getting subset of dicom_data"""
    full_qr, full_data = dicom_data

    def _make_sub_qr(spec):
        if spec is None:
            return (QueryResult(QueryLevel.IMAGE), [])
        elif spec == "all":
            return (deepcopy(full_qr), full_data)
        else:
            curr_qr = QueryResult(QueryLevel.IMAGE)
            for subtree_spec in spec.split(";"):
                curr_node = None
                for lvl_comp in subtree_spec.split("/"):
                    lvl_name, lvl_idx = lvl_comp.split("-")
                    lvl_idx = int(lvl_idx)
                    curr_node = sorted(
                        full_qr.children(curr_node), key=lambda x: x.uid
                    )[lvl_idx]
                    assert curr_node.level == getattr(QueryLevel, lvl_name)
                curr_qr |= full_qr.sub_query(curr_node)
            curr_data = [x for x in full_data if x[1] in curr_qr]
            return (curr_qr, curr_data)

    return _make_sub_qr


@fixture
def make_local_dir(get_dicom_subset):
    """Factory fixture to build LocalDir stores"""
    curr_dirs = []
    with TemporaryDirectory(prefix="dcm-test") as temp_dir:
        temp_dir = Path(temp_dir)

        def _make_local_dir(subset="all", **kwargs):
            store_dir = temp_dir / f"store_dir{len(curr_dirs)}"
            os.makedirs(store_dir)
            curr_dirs.append(store_dir)
            init_qr, init_data = get_dicom_subset(subset)
            for dcm_path, _ in init_data:
                shutil.copy(dcm_path, store_dir)
            return (LocalDir(store_dir, **kwargs), init_qr, store_dir)

        yield _make_local_dir


@fixture
def make_local_node(base_port=63987, base_name="DCMTESTAE"):
    """Factory fixture to make a local DcmNode with a unqiue port/ae_title"""
    local_idx = [0]

    def _make_local_node():
        res = DcmNode(
            "localhost", f"{base_name}{local_idx[0]}", base_port + local_idx[0]
        )
        local_idx[0] += 1
        return res

    return _make_local_node


@dataclass
class TestNode:
    """Capture all relevant components of a test DICOM node"""

    node_type: str

    init_qr: QueryResult

    store_dir: Path

    dcm_node: Optional[DcmNode] = None

    proc: Optional[sp.Popen] = None

    stdout: Optional[BinaryIO] = None

    stderr: Optional[BinaryIO] = None

    _is_finalized: bool = False

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    def finalize(self):
        if not self._is_finalized and self.proc is not None:
            self.proc.terminate()
            self._is_finalized = True
            self.stdout.flush()
            self.stderr.flush()
            self.stdout.seek(0)
            self.stderr.seek(0)
            return self.stdout.read(), self.stderr.read()


@fixture
def make_qr_repo(get_dicom_subset):
    """Factory fixutre to build QrRepo stores"""
    curr_dirs = []
    with TemporaryDirectory(prefix="dcm-test") as temp_dir:
        temp_dir = Path(temp_dir)

        async def _make_qr_repo(local_node=None, subset="all", **kwargs):
            store_dir = temp_dir / f"store_dir{len(curr_dirs)}"
            os.makedirs(store_dir)
            curr_dirs.append(store_dir)
            init_qr, init_data = get_dicom_subset(subset)
            for dcm_path, _ in init_data:
                shutil.copy(dcm_path, store_dir)
            repo = await QrRepo.build(store_dir, scan_fs=True, **kwargs)
            return (repo, TestNode("qr", init_qr, store_dir))

        yield _make_qr_repo


DCMQRSCP_PATH = shutil.which("dcmqrscp")


DCMQRIDX_PATH = shutil.which("dcmqridx")


DCMTK_VER_RE = r".+\s+v([0-9]+).([0-9])+.([0-9])\+?\s+.*"


# Version 3.6.2 allows us to test retrieving private SOP classes
DCMTK_PRIV_RETR_VERS = (3, 6, 2)


# Unfortunately, we still can't test sending private SOP classes (see
# https://forum.dcmtk.org/viewtopic.php?f=1&t=4935&p=20112#p20112). If this
# gets fixed in a future version this should be updated
DCMTK_PRIV_SEND_VERS = (1000, 0, 0)


def get_dcmtk_version():
    if DCMQRSCP_PATH is None:
        return None
    try:
        sp_out = sp.check_output([DCMQRSCP_PATH, "--version"]).decode("latin-1")
    except (FileNotFoundError, sp.CalledProcessError):
        return None
    first = sp_out.split("\n")[0]
    return tuple(int(x) for x in re.match(DCMTK_VER_RE, first).groups())


DCMTK_VERSION = get_dcmtk_version()


DCMTK_BASE_PORT = 62760


DCMTK_BASE_NAME = "DCMTKAE"


has_dcmtk = mark.skipif(
    DCMTK_VERSION is None, reason="can't find DCMTK command 'dcmqrscp'"
)


dcmtk_priv_sop_retr_xfail = mark.xfail(
    DCMTK_VERSION is None or DCMTK_VERSION < DCMTK_PRIV_RETR_VERS,
    reason="dcmqrscp version doesn't support retrieving private " "SOPClasses",
)


dcmtk_priv_sop_send_xfail = mark.xfail(
    DCMTK_VERSION is None or DCMTK_VERSION < DCMTK_PRIV_SEND_VERS,
    reason="dcmqrscp version doesn't support sending private " "SOPClasses",
)


DCMTK_CONFIG_TMPL = """\
NetworkTCPPort  = {dcmtk_node.port}
MaxPDUSize      = 16384
MaxAssociations = 16

HostTable BEGIN
{client_lines}
HostTable END

AETable BEGIN
{dcmtk_node.ae_title} {store_dir} RW (500, 1024mb)  {client_names}
AETable END
"""


CLIENT_LINE_TMPL = "{name} = ({node.ae_title}, {node.host}, {node.port})"


def mk_dcmtk_config(dcmtk_node, store_dir, clients):
    """Make config file for DCMTK test node"""
    client_names = []
    client_lines = []
    for node_idx, node in enumerate(clients):
        name = "test_client%d" % node_idx
        line = CLIENT_LINE_TMPL.format(name=name, node=node)
        client_names.append(name)
        client_lines.append(line)
    client_names = " ".join(client_names)
    client_lines = "\n".join(client_lines)

    return DCMTK_CONFIG_TMPL.format(
        dcmtk_node=dcmtk_node,
        store_dir=store_dir,
        client_lines=client_lines,
        client_names=client_names,
    )


def _get_used_ports():
    return set(conn.laddr.port for conn in psutil.net_connections())


@fixture
def make_dcmtk_nodes(get_dicom_subset, pytestconfig):
    """Factory fixture for building dcmtk nodes"""
    used_ports = _get_used_ports()
    nodes = []
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        def _make_dcmtk_node(clients, subset="all"):
            if pytestconfig.getoption("--disable-backend") == "dcmtk":
                pytest.skip("The DCMTK test backend is disabled")
            node_idx = len(nodes)
            # TODO: We still have a race condition with port selection here,
            #       ideally we would put this whole thing in a loop and detect
            #       the server process failing with log message about the port
            #       being in use
            port = DCMTK_BASE_PORT + node_idx
            while port in used_ports:
                port += 1
            used_ports.add(port)
            ae_title = DCMTK_BASE_NAME + str(node_idx)
            dcmtk_node = DcmNode("localhost", ae_title, port)
            print(f"Building dcmtk node: {ae_title}:{port}")
            node_dir = tmp_dir / ae_title
            db_dir = node_dir / "db"
            stdout_path = node_dir / "stdout.log"
            stderr_path = node_dir / "stderr.log"
            conf_file = db_dir / "dcmqrscp.cfg"
            test_store_dir = db_dir / "TEST_STORE"
            os.makedirs(test_store_dir)
            full_conf = mk_dcmtk_config(dcmtk_node, test_store_dir, clients)
            print(full_conf)
            with open(conf_file, "wt") as conf_fp:
                conf_fp.write(full_conf)
            init_qr, init_data = get_dicom_subset(subset)
            init_files = []
            for in_path, ds in init_data:
                out_path = test_store_dir / in_path.name
                shutil.copy(in_path, out_path)
                init_files.append(out_path)
            # Index any initial files into the dcmtk db
            if init_files:
                print("Indexing initial files into dcmtk node...")
                sp.run([DCMQRIDX_PATH, str(test_store_dir)] + init_files)
                print("Done")
                time.sleep(1)
            # Fire up a dcmqrscp process
            dcmqrscp_args = [DCMQRSCP_PATH, "-c", str(conf_file)]
            dcmqrscp_args += ["-ll", "debug"]
            if DCMTK_VERSION >= (3, 6, 2):
                dcmqrscp_args += ["-xf", str(PC_CONF_PATH), "Default", "Default"]
            print("Starting dcmtk qrscp process...")
            sout_f = stdout_path.open("w+b")
            serr_f = stderr_path.open("w+b")
            proc = sp.Popen(dcmqrscp_args, stdout=sout_f, stderr=serr_f)
            print("Done")
            res = TestNode(
                "dcmtk", init_qr, test_store_dir, dcmtk_node, proc, sout_f, serr_f
            )
            nodes.append(res)
            time.sleep(2)
            return res

        try:
            yield _make_dcmtk_node
        finally:
            for node in nodes:
                if not node.is_finalized:
                    node.finalize()


@fixture
def make_dcmtk_net_repo(make_local_node, make_dcmtk_nodes):
    async def _make_net_repo(local_node=None, clients=[], subset="all"):
        if local_node is None:
            local_node = make_local_node()
        dcmtk_node = make_dcmtk_nodes([local_node] + clients, subset)
        return (NetRepo(local_node, dcmtk_node.dcm_node), dcmtk_node)

    return _make_net_repo


pnd_priv_sop_xfail = mark.xfail(reason="Private SOPClassUID not supported by PND nodes")


PND_BASE_NAME = "PNDAE"


PND_BASE_PORT = 62560


PND_CONF_TEMPLATE = """\
[DEFAULT]
# Our AE Title
ae_title: {pnd_node.ae_title}
# Our listen port
port: {pnd_node.port}
# Our maximum PDU size; 0 for unlimited
max_pdu: 16382
# The ACSE, DIMSE and network timeouts (in seconds)
acse_timeout: 30
dimse_timeout: 30
network_timeout: 30
# The address of the network interface to listen on
# If unset, listen on all interfaces
bind_address: 127.0.0.1
# Directory where SOP Instances received from Storage SCUs will be stored
#   This directory contains the QR service's managed SOP Instances
instance_location: {data_dir}
# Location of sqlite3 database for the QR service's managed SOP Instances
database_location: {db_path}

"""


PND_PYTHON_PATH = os.getenv("DCM_PND_FIXTURE_PYTHON", shutil.which("python"))


@fixture
def make_pnd_nodes(get_dicom_subset, pytestconfig):
    """Factory fixture for making pynetdicom qrscp nodes"""
    nodes = []
    procs = []
    used_ports = _get_used_ports()
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        def _make_pnd_node(clients, subset="all"):
            if pytestconfig.getoption("--disable-backend") == "pnd":
                pytest.skip("The PND test backend is disabled")
            node_idx = len(nodes)
            # TODO: We still have a race condition with port selection here,
            #       ideally we would put this whole thing in a loop and detect
            #       the server process failing with log message about the port
            #       being in use
            port = PND_BASE_PORT + node_idx
            while port in used_ports:
                port += 1
            used_ports.add(port)
            ae_title = PND_BASE_NAME + str(node_idx)
            pnd_node = DcmNode("localhost", ae_title, port)
            print(f"Building pynetdicom node: {ae_title}:{port}")
            node_dir = tmp_dir / ae_title
            stdout_path = node_dir / "stdout.log"
            stderr_path = node_dir / "stderr.log"
            db_path = node_dir / "pnd.db"
            data_dir = node_dir / "data"
            os.makedirs(data_dir)
            # Make the config file
            conf_path = node_dir / "qrscp.conf"
            conf_parts = [
                PND_CONF_TEMPLATE.format(
                    pnd_node=pnd_node, data_dir=data_dir, db_path=db_path
                )
            ]
            for client in clients:
                conf_parts.append(f"[{client.ae_title}]")
                conf_parts.append(f"    address: {client.host}")
                conf_parts.append(f"    port: {client.port}")
                conf_parts.append("")
            conf = "\n".join(conf_parts)
            print(conf)
            conf_path.write_text(conf)
            # Fire up a qrscp process
            sout_f = stdout_path.open("w+b")
            serr_f = stderr_path.open("w+b")
            proc = sp.Popen(
                [
                    PND_PYTHON_PATH,
                    "-m",
                    "pynetdicom",
                    "qrscp",
                    "-c",
                    str(conf_path),
                    "-ll",
                    "debug",
                ],
                stdout=sout_f,
                stderr=serr_f,
            )
            time.sleep(2)
            # We have to send any initial data as there is no index functionality
            init_qr, init_data = get_dicom_subset(subset)
            init_files = []
            init_ae = AE()
            init_assoc = init_ae.associate(
                "127.0.0.1",
                pnd_node.port,
                _make_default_store_scu_pcs(),
                ae_title=pnd_node.ae_title,
            )
            if not init_assoc.is_established:
                raise ValueError(
                    "Failed to associate with PND test node to populate it"
                )
            for in_path, ds in init_data:
                # Currently, can't store these files in the pynetdicom 'qrscp' app
                if ds.SOPClassUID == "1.3.12.2.1107.5.9.1":
                    init_qr.remove(ds)
                    continue
                init_assoc.send_c_store(ds)
            init_assoc.release()
            res = TestNode("pnd", init_qr, data_dir, pnd_node, proc, sout_f, serr_f)
            nodes.append(res)
            return res

        try:
            yield _make_pnd_node
        finally:
            for node in nodes:
                if not node.is_finalized:
                    node.finalize()


@fixture
def make_pnd_net_repo(make_local_node, make_pnd_nodes):
    async def _make_net_repo(local_node=None, clients=[], subset="all"):
        if local_node is None:
            local_node = make_local_node()
        pnd_node = make_pnd_nodes([local_node] + clients, subset)
        return (NetRepo(local_node, pnd_node.dcm_node), pnd_node)

    return _make_net_repo


@fixture
def make_remote_nodes(node_type, make_dcmtk_nodes, make_pnd_nodes):
    if node_type == "dcmtk":
        return make_dcmtk_nodes
    else:
        assert node_type == "pnd"
        return make_pnd_nodes


@fixture
def make_net_repo(node_type, make_dcmtk_net_repo, make_pnd_net_repo):
    if node_type == "dcmtk":
        return make_dcmtk_net_repo
    else:
        assert node_type == "pnd"
        return make_pnd_net_repo


@fixture
def make_repo(node_type, make_dcmtk_net_repo, make_pnd_net_repo, make_qr_repo):
    if node_type == "dcmtk":
        return make_dcmtk_net_repo
    elif node_type == "pnd":
        return make_pnd_net_repo
    else:
        assert node_type == "qr"
        return make_qr_repo


def get_stored_files(store_dir):
    return [
        x
        for x in Path(store_dir).glob("**/*")
        if not x.is_dir() and x.name not in ("index.dat", "dcm_meta.json")
    ]


# @fixture
# def make_store(make_dcmtk_nodes, dicom_data):
#    '''Factory fixture for building data stores'''
#    def _make_store(subset, cls, kwargs):
#        if cls == NetRepo:
#            dcmtk_node, _, _ = make_dcmtk_nodes(subset)
#            res = NetRepo()
#        elif cls == LocalDir:
#            pass
#
#
#    yield _make_store


@fixture
def make_dcm_config_file():
    with NamedTemporaryFile("wt", delete=False) as conf_f:

        def _make_dcm_config_file(config_str=_default_conf):
            conf_f.write(config_str)
            conf_f.file.flush()
            return conf_f.name

        yield _make_dcm_config_file
    try:
        conf_f.close()
        os.unlink(conf_f.name)
    except:
        pass


@fixture
def make_dcm_config(make_dcm_config_file):
    def _make_dcm_config(config_str=_default_conf):
        config_path = make_dcm_config_file(config_str)
        return DcmConfig(config_path)

    return _make_dcm_config
