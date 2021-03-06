import os, time, shutil, random, tarfile, logging, re
from copy import deepcopy
import subprocess as sp
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path

import pydicom
import psutil
from pytest import fixture, mark

from ..conf import _default_conf, DcmConfig
from ..query import QueryLevel, QueryResult
from ..net import DcmNode
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir


logging_opts = {}


def pytest_addoption(parser):
    parser.addoption(
        "--show-pynetdicom-logs",
        default=False,
        action="store_true",
        help="show pynetdicom logs",
    )
    parser.addoption("--dcmtk-log-level", default=None, help="adjust dcmtk log level")


def pytest_configure(config):
    dcmtk_log_level = config.getoption("--dcmtk-log-level")
    if dcmtk_log_level:
        logging_opts["dcmtk_level"] = dcmtk_log_level
    if not config.getoption("--show-pynetdicom-logs"):
        for pnd_mod in ("events", "assoc", "dsutils", "acse"):
            lgr = logging.getLogger("pynetdicom.%s" % pnd_mod)
            lgr.setLevel(logging.WARN)


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


dcmqrscp_path = shutil.which("dcmqrscp")


dcmqridx_path = shutil.which("dcmqridx")


DCMTK_VER_RE = r".+\s+v([0-9]+).([0-9])+.([0-9])\+?\s+.*"


# Version 3.6.2 allows us to test retrieving private SOP classes
DCMTK_PRIV_RETR_VERS = (3, 6, 2)


# Unfortunately, we still can't test sending private SOP classes (see
# https://forum.dcmtk.org/viewtopic.php?f=1&t=4935&p=20112#p20112). If this
# gets fixed in a future version this should be updated
DCMTK_PRIV_SEND_VERS = (1000, 0, 0)


def get_dcmtk_version():
    if dcmqrscp_path is None:
        return None
    try:
        sp_out = sp.check_output([dcmqrscp_path, "--version"]).decode("latin-1")
    except (FileNotFoundError, sp.CalledProcessError):
        return None
    first = sp_out.split("\n")[0]
    return tuple(int(x) for x in re.match(DCMTK_VER_RE, first).groups())


DCMTK_VERSION = get_dcmtk_version()


dcmtk_base_port = 62760


dcmtk_base_name = "DCMTKAE"


has_dcmtk = mark.skipif(
    DCMTK_VERSION is None, reason="can't find DCMTK command 'dcmqrscp'"
)


dcmtk_priv_sop_retr_xfail = mark.xfail(
    DCMTK_VERSION < DCMTK_PRIV_RETR_VERS,
    reason="dcmqrscp version doesn't support retrieving private " "SOPClasses",
)


dcmtk_priv_sop_send_xfail = mark.xfail(
    DCMTK_VERSION < DCMTK_PRIV_SEND_VERS,
    reason="dcmqrscp version doesn't support sending private " "SOPClasses",
)


dcmtk_config_tmpl = """\
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


client_line_tmpl = "{name} = ({node.ae_title}, {node.host}, {node.port})"


def mk_dcmtk_config(dcmtk_node, store_dir, clients):
    """Make config file for DCMTK test node"""
    client_names = []
    client_lines = []
    for node_idx, node in enumerate(clients):
        name = "test_client%d" % node_idx
        line = client_line_tmpl.format(name=name, node=node)
        client_names.append(name)
        client_lines.append(line)
    client_names = " ".join(client_names)
    client_lines = "\n".join(client_lines)

    return dcmtk_config_tmpl.format(
        dcmtk_node=dcmtk_node,
        store_dir=store_dir,
        client_lines=client_lines,
        client_names=client_names,
    )


def _get_used_ports():
    return set(conn.laddr.port for conn in psutil.net_connections())


@fixture
def make_dcmtk_nodes(get_dicom_subset):
    """Factory fixture for building dcmtk nodes"""
    procs = []
    used_ports = _get_used_ports()
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        def _make_dcmtk_node(clients, subset="all"):
            node_idx = len(procs)
            # TODO: We still have a race condition with port selection here,
            #       ideally we would put this whole thing in a loop and detect
            #       the server process failing with log message about the port
            #       being in use
            port = dcmtk_base_port + node_idx
            while port in used_ports:
                port += 1
            used_ports.add(port)
            ae_title = dcmtk_base_name + str(node_idx)
            dcmtk_node = DcmNode("localhost", ae_title, port)
            print(f"Building dcmtk node: {ae_title}:{port}")
            node_dir = tmp_dir / ae_title
            db_dir = node_dir / "db"
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
                sp.run([dcmqridx_path, str(test_store_dir)] + init_files)
            # Fire up a dcmqrscp process
            dcmqrscp_args = [dcmqrscp_path, "-c", str(conf_file)]
            if "dcmtk_level" in logging_opts:
                dcmqrscp_args += ["-ll", logging_opts["dcmtk_level"]]
            if DCMTK_VERSION >= (3, 6, 2):
                dcmqrscp_args += ["-xf", str(PC_CONF_PATH), "Default", "Default"]
            procs.append(sp.Popen(dcmqrscp_args))
            time.sleep(1)
            return (dcmtk_node, init_qr, test_store_dir)

        try:
            yield _make_dcmtk_node
        finally:
            for proc in procs:
                proc.terminate()


@fixture
def make_dcmtk_net_repo(make_local_node, make_dcmtk_nodes):
    def _make_net_repo(local_node=None, clients=[], subset="all"):
        if local_node is None:
            local_node = make_local_node()
        dcmtk_node, init_qr, store_dir = make_dcmtk_nodes(
            [local_node] + clients, subset
        )
        return (NetRepo(local_node, dcmtk_node), init_qr, store_dir)

    return _make_net_repo


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
    with NamedTemporaryFile("wt") as conf_f:

        def _make_dcm_config_file(config_str=_default_conf):
            conf_f.write(config_str)
            conf_f.file.flush()
            return conf_f.name

        yield _make_dcm_config_file


@fixture
def make_dcm_config(make_dcm_config_file):
    def _make_dcm_config(config_str=_default_conf):
        config_path = make_dcm_config_file(config_str)
        return DcmConfig(config_path)

    return _make_dcm_config
