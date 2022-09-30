import csv, time, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pytest
from pytest import mark
from click.testing import CliRunner
import pydicom

from dcm.query import QueryLevel

from ..cli import cli
from ..util import json_serializer

from .conftest import has_dcmtk


@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_echo(make_local_node, make_remote_nodes, make_dcm_config_file):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], None)
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "echo",
        "--local",
        str(local_node),
        str(remote.dcm_node),
    ]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert result.output == "Success\n"


@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_query(make_local_node, make_remote_nodes, make_dcm_config_file):
    local_node = make_local_node()
    remote = make_remote_nodes([local_node], "all")
    time.sleep(2)
    runner = CliRunner(mix_stderr=False)
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "query",
        "--assume-yes",
        "--level",
        "image",
        "--local",
        str(local_node),
        str(remote.dcm_node),
    ]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    res_qr = json_serializer.loads(result.output)
    assert remote.init_qr.equivalent(res_qr)
    # Test batch query
    batch_dicts = []
    for p in remote.init_qr.level_paths(QueryLevel.STUDY):
        sinfo = remote.init_qr.path_info(p)
        batch_dicts.append(
            {"PatientID": sinfo["PatientID"], "StudyDate": sinfo["StudyDate"]}
        )
    print(batch_dicts)
    with NamedTemporaryFile("w+t") as csv_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=batch_dicts[0].keys())
        csv_writer.writerows(batch_dicts)
        csv_f.flush()
        args += ["--batch-csv", csv_f.name]
        print(args)
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        res_qr = json_serializer.loads(result.output)
        assert remote.init_qr.equivalent(res_qr)


@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_sync(make_local_node, make_remote_nodes, make_dcm_config_file):
    local_node = make_local_node()
    src_remote = make_remote_nodes([local_node], "all")
    dest_remote = make_remote_nodes([local_node], None)
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "sync",
        "--local",
        str(local_node),
        "--source",
        str(src_remote.dcm_node),
        str(dest_remote.dcm_node),
    ]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    # TODO: Test data made it to dest


def test_sync_missing_source(make_dcm_config_file):
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "sync",
    ]
    with TemporaryDirectory(prefix="dcm-test") as temp_dir:
        temp_dir = Path(temp_dir)
        na_src = temp_dir / "non-existant"
        dest = temp_dir / "dest"
        dest.mkdir()
        result = runner.invoke(cli, args + ["--source", str(na_src), str(dest)])
        print(result.stdout)
        assert not na_src.exists()
        assert result.exit_code != 0


def _run_forward(config_path, local_node, dest_dir):
    runner = CliRunner()
    fwd_args = [
        "--config",
        config_path,
        "forward",
        "--inactive-timeout",
        "10",
        "--local",
        local_node,
        dest_dir,
    ]
    return runner.invoke(cli, fwd_args)


@mark.parametrize("node_type", (pytest.param("dcmtk", marks=has_dcmtk), "pnd"))
def test_forward(
    make_local_node, make_remote_nodes, make_dcm_config_file, make_local_dir
):
    local_node = make_local_node()
    src_remote = make_remote_nodes([local_node], "PATIENT-1")
    dest_bucket, _, dest_dir = make_local_dir(None)
    runner = CliRunner()
    config_path = make_dcm_config_file()
    with ThreadPoolExecutor(max_workers=1) as executor:
        fwd_future = executor.submit(
            _run_forward, config_path, str(local_node), str(dest_dir)
        )
        time.sleep(1)
        if fwd_future.done():
            # Must have been an exception
            print("Thread exited prematurely...")
            fwd_res = fwd_future.result(timeout=2.0)
            assert False
        assert fwd_future.running()
        runner = CliRunner()
        sync_args = [
            "--config",
            config_path,
            "sync",
            "--local",
            str(local_node),
            "--force-all",
            "--method",
            "REMOTE_COPY",
            "--source",
            str(src_remote.dcm_node),
            str(local_node),
        ]
        print(sync_args)
        sync_result = runner.invoke(cli, sync_args)
        while not fwd_future.done():
            time.sleep(1)
        fwd_res = fwd_future.result(timeout=2.0)
    assert sync_result.exit_code == 0
    assert fwd_res.exit_code == 0
    # TODO: Make sure the data made it


def test_dump(dicom_files, make_dcm_config_file):
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "dump",
        "-i",
        "SOPInstanceUID",
        "--plain-fmt",
        "{elem.value}",
        "--group",
        "8",
    ]
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path)
        dump_res = runner.invoke(cli, args + [str(dcm_path)])
        # print(dump_res.stdout)
        assert dump_res.exit_code == 0
        assert dump_res.stdout.strip() == ds.SOPInstanceUID
    args = [
        "--config",
        config_path,
        "dump",
        "--kw-regex",
        "InstanceUID",
        "--out-format",
        "json",
    ]
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path)
        dump_res = runner.invoke(cli, args + [str(dcm_path)])
        print(dump_res.stdout)
        assert dump_res.exit_code == 0
        res_dict = json.loads(dump_res.stdout)
        for kw in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
            assert res_dict[kw] == getattr(ds, kw)
