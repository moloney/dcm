import os, time, shutil, random, tarfile, logging
from copy import deepcopy
import subprocess as sp
from tempfile import TemporaryDirectory
from pathlib import Path

import pydicom
from pytest import fixture, mark

from ..query import QueryLevel, QueryResult
from ..net import DcmNode
from ..store.net_repo import NetRepo
from ..store.local_dir import LocalDir


logging_opts = {}

def pytest_addoption(parser):
    parser.addoption(
        "--show-pynetdicom-logs", default=False, action='store_true',
        help="show pynetdicom logs"
    )
    parser.addoption(
        "--dcmtk-log-level", default=None, help="adjust dcmtk log level"
    )


def pytest_configure(config):
    dcmtk_log_level = config.getoption('--dcmtk-log-level')
    if dcmtk_log_level:
        logging_opts['dcmtk_level'] = dcmtk_log_level
    if not config.getoption('--show-pynetdicom-logs'):
        for pnd_mod in ('events', 'assoc', 'dsutils', 'acse'):
            lgr = logging.getLogger('pynetdicom.%s' % pnd_mod)
            lgr.setLevel(logging.WARN)



DATA_DIR = Path(__file__).parent / 'data'


DICOM_TAR = DATA_DIR / 'dicom.tar.bz2'


@fixture(scope='session')
def dicom_dir():
    '''Decompress DICOM files into a temp directory'''
    with TemporaryDirectory(prefix='dcm-test') as temp_dir:
        # Unpack tar to temp dir and yield list of paths
        with tarfile.open(DICOM_TAR, "r:bz2") as tf:
            tf.extractall(temp_dir)
        temp_dir = Path(temp_dir)
        yield temp_dir


@fixture(scope='session')
def dicom_files(dicom_dir):
    '''Decompress DICOM files into a temp directory and return their paths'''
    return [x for x in dicom_dir.glob('**/*.dcm')]


@fixture
def data_subset():
    return 'all'


@fixture()
def dicom_data(dicom_files, data_subset):
    '''QueryResult and list of (path, dataset) tuples matching `data_subset`'''
    res_data = []
    res_qr = QueryResult(QueryLevel.IMAGE)
    if data_subset is None:
        return (res_qr, res_data)
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(str(dcm_path))
        if data_subset == 'all' or ds in data_subset:
            res_data.append((dcm_path, ds))
            res_qr.add(ds)
    return (res_qr, res_data)

# TODO: get rid of this in favor of above func
@fixture()
def dicom_files_w_qr(dicom_files):
    qr = QueryResult(QueryLevel.IMAGE)
    for path in dicom_files:
        ds = pydicom.dcmread(str(path))
        qr.add(ds)
    return (dicom_files, qr)


@fixture
def make_local_dir(dicom_data):
    '''Factory fixture to build LocalDir stores'''
    curr_dirs = []
    with TemporaryDirectory(prefix='dcm-test') as temp_dir:
        temp_dir = Path(temp_dir)
        def _make_local_dir(subset='all', **kwargs):
            store_dir = temp_dir / f'store_dir{len(curr_dirs)}'
            os.makedirs(store_dir)
            curr_dirs.append(store_dir)
            init_qr, init_data = dicom_data
            for dcm_path, _ in init_data:
                print(f"Copying a file: {dcm_path} -> {store_dir}")
                shutil.copy(dcm_path, store_dir)
            return (LocalDir(store_dir, **kwargs), init_qr, store_dir)
        yield _make_local_dir


@fixture
def make_local_node(base_port=63987, base_name='DCMTESTAE'):
    '''Factory fixture to make a local DcmNode with a unqiue port/ae_title'''
    local_idx = [0]
    def _make_local_node():
        res = DcmNode('localhost',
                      f'{base_name}{local_idx[0]}',
                      base_port + local_idx[0])
        local_idx[0] += 1
        return res
    return _make_local_node


dcmtk_base_port = 62765


dcmtk_base_name = 'DCMTKAE'


has_dcmtk = mark.skipif(shutil.which('dcmqrscp') == None,
                        reason="can't find DCMTK command 'dcmqrscp'")


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
    '''Make config file for DCMTK test node'''
    client_names = []
    client_lines = []
    for node_idx, node in enumerate(clients):
        name = 'test_client%d' % node_idx
        line = client_line_tmpl.format(name=name, node=node)
        client_names.append(name)
        client_lines.append(line)
    client_names = ' '.join(client_names)
    client_lines = '\n'.join(client_lines)

    return dcmtk_config_tmpl.format(dcmtk_node=dcmtk_node,
                                    store_dir=store_dir,
                                    client_lines=client_lines,
                                    client_names=client_names,
                                   )


@fixture
def make_dcmtk_nodes(dicom_data):
    '''Factory fixture for building dcmtk nodes'''
    procs = []
    full_qr, full_data = dicom_data
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        def _make_dcmtk_node(clients, subset='all'):
            node_idx = len(procs)
            port = dcmtk_base_port + node_idx
            ae_title = dcmtk_base_name + str(node_idx)
            dcmtk_node = DcmNode('localhost', ae_title, port)
            node_dir = tmp_dir / ae_title
            db_dir = node_dir / 'db'
            conf_file = db_dir / 'dcmqrscp.cfg'
            test_store_dir = db_dir / 'TEST_STORE'
            os.makedirs(test_store_dir)
            full_conf = mk_dcmtk_config(dcmtk_node, test_store_dir, clients)
            with open(conf_file, 'wt') as conf_fp:
                conf_fp.write(full_conf)
            if subset == 'all':
                init_files = [p for p, _ in full_data]
                init_qr = deepcopy(full_qr)
            else:
                init_files = []
                init_qr = QueryResult(QueryLevel.IMAGE)
                if subset is not None:
                    for in_path, ds in full_data:
                        if ds in subset:
                            out_path = test_store_dir / in_path.name
                            shutil.copy(in_path, out_path)
                            init_files.append(out_path)
                            init_qr.add(ds)
            # Index any initial files into the dcmtk db
            if init_files:
                sp.run(['dcmqridx', str(test_store_dir)] + init_files)
            # Fire up a dcmqrscp process
            dcmqrscp_args = ['dcmqrscp', '-c', str(conf_file)]
            if 'dcmtk_level' in logging_opts:
                dcmqrscp_args += ['-ll', logging_opts['dcmtk_level']]
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
    def _make_net_repo(local_node=None, clients=[], subset='all'):
        if local_node is None:
            local_node = make_local_node()
        dcmtk_node, init_qr, store_dir = make_dcmtk_nodes([local_node]+clients,
                                                          subset)
        return (NetRepo(local_node, dcmtk_node), init_qr, store_dir)
    return _make_net_repo

#@fixture
#def make_store(make_dcmtk_nodes, dicom_data):
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


# TODO: Allow the included files to be defined as a QR. Provide higher level
#       fixture that provides LocalDir or NetRepo

# TODO: We should return a query_result for the initial files instead of a list
@fixture
def dcmtk_test_nodes(request, dicom_files_w_qr):
    '''Build one or more DICOM test nodes using DCMTK 'dcmqrscp' command'''
    file_requests = getattr(request, 'param', [None])
    dicom_files, full_qr = dicom_files_w_qr
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        procs = []
        results = []
        for node_idx, file_req in enumerate(file_requests):
            port = dcmtk_base_port + node_idx
            ae_title = dcmtk_base_name + str(node_idx)
            dcmtk_node = DcmNode('localhost', ae_title, port)
            node_dir = tmp_dir / ae_title
            db_dir = node_dir / 'db'
            conf_file = db_dir / 'dcmqrscp.cfg'
            test_store_dir = db_dir / 'TEST_STORE'
            os.makedirs(test_store_dir)
            full_conf = mk_dcmtk_config(dcmtk_node, test_store_dir)
            with open(conf_file, 'wt') as conf_fp:
                conf_fp.write(full_conf)
            # Allow initial files to be populated
            if file_req is not None:
                in_files, init_qr = select_files(dicom_files, file_req, full_qr)
                init_files = []
                for in_path in in_files:
                    out_path = test_store_dir / in_path.name
                    shutil.copy(in_path, out_path)
                    init_files.append(out_path)
                #init_files = file_factory(test_store_dir)
                # Index files into the dcmtk db
                sp.run(['dcmqridx', str(test_store_dir)] + init_files)
            else:
                init_files = []
                init_qr = QueryResult(QueryLevel.IMAGE)
            dcmqrscp_args = ['dcmqrscp', '-c', str(conf_file)]
            if 'dcmtk_level' in logging_opts:
                dcmqrscp_args += ['-ll', logging_opts['dcmtk_level']]
            procs.append(sp.Popen(dcmqrscp_args))
            results.append((dcmtk_node, init_qr, test_store_dir))

        time.sleep(1)
        try:
            yield results
        finally:
            for proc in procs:
                proc.terminate()





def select_files(dicom_files, file_req, full_qr):
    '''Select a subset of dicom files'''
    if file_req == 'all':
        res = (dicom_files.copy(), full_qr)
    else:
        res_files = []
        for path in dicom_files:
            img_uid = path.stem
            try:
                _ = file_req[img_uid]
            except KeyError:
                pass
            else:
                res_files.append(path)
        res = (res_files, file_req)
    return res


def make_local_factory(local_dir, random_drop_thresh=0.0):
    '''Populates dicom test node with data from local directory'''
    if random_drop_thresh > 0.0:
        seed = int(time.time())
        print("Using seed: %d" % seed)
        random.seed(seed)
    local_dir = Path(local_dir)
    in_files = [x for x in local_dir.glob('**/*.dcm')]
    def local_factory(dest_dir):
        dest_dir = Path(dest_dir)
        out_files = []
        for in_file in in_files:
            if random_drop_thresh > 0.0 and random.random() < random_drop_thresh:
                continue
            out_file = dest_dir / in_file.relative_to(local_dir)
            out_dir = out_file.parent
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            shutil.copy(in_file, out_dir)
            out_files.append(out_file)
        return out_files
    return local_factory

