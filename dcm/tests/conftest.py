import os, time, shutil, random, tarfile, logging
import subprocess as sp
from tempfile import TemporaryDirectory
from pathlib import Path

import pydicom
from pytest import fixture, mark

from ..query import QueryLevel, QueryResult
from ..net import DcmNode


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
    '''Decompress DICOM files into a temp directory and return their paths'''
    with TemporaryDirectory(prefix='dcm') as temp_dir:
        # Unpack tar to temp dir and yield list of paths
        with tarfile.open(DICOM_TAR, "r:bz2") as tf:
            tf.extractall(temp_dir)
        temp_dir = Path(temp_dir)
        yield temp_dir


@fixture(scope='session')
def dicom_files(dicom_dir):
    '''Decompress DICOM files into a temp directory and return their paths'''
    return [x for x in dicom_dir.glob('**/*.dcm')]


@fixture()
def dicom_files_w_qr(dicom_files):
    qr = QueryResult(QueryLevel.IMAGE)
    for path in dicom_files:
        ds = pydicom.dcmread(str(path))
        qr.add(ds)
    return (dicom_files, qr)


local_nodes = [DcmNode('localhost', 'DCMTESTAE1', 43293),
               DcmNode('localhost', 'DCMTESTAE2', 43294),
               DcmNode('localhost', 'DCMTESTAE3', 43295)]


dcmtk_base_port = 37592


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


def mk_dcmtk_config(dcmtk_node, store_dir, clients=None):
    '''Make config file for DCMTK test node'''
    if clients is None:
        clients = local_nodes
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
            if 'dcmrk_level' in logging_opts:
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


