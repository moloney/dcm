import os, time, signal
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import pytest
from pytest import fixture, mark
from click.testing import CliRunner
import pydicom

from ..cli import cli
from ..util import json_serializer

from .conftest import has_dcmtk, DCMTK_VERSION


@has_dcmtk
def test_echo(make_local_node, make_dcmtk_nodes, make_dcm_config_file):
    local_node = make_local_node()
    remote, _, _ = make_dcmtk_nodes([local_node], None)
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = ["--config", config_path, "echo", "--local", str(local_node), str(remote)]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert result.output == "Success\n"


@has_dcmtk
def test_query(make_local_node, make_dcmtk_nodes, make_dcm_config_file):
    local_node = make_local_node()
    remote, init_qr, store_dir = make_dcmtk_nodes([local_node], "all")
    time.sleep(2)
    runner = CliRunner()
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
        str(remote),
    ]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    res_qr = json_serializer.loads(result.output)
    assert init_qr.equivalent(res_qr)


@has_dcmtk
def test_sync(make_local_node, make_dcmtk_nodes, make_dcm_config_file):
    local_node = make_local_node()
    src_remote, init_qr, src_dir = make_dcmtk_nodes([local_node], "all")
    dest_remote, _, dest_dir = make_dcmtk_nodes([local_node], None)
    runner = CliRunner()
    config_path = make_dcm_config_file()
    args = [
        "--config",
        config_path,
        "sync",
        "--local",
        str(local_node),
        "--source",
        str(src_remote),
        str(dest_remote),
    ]
    print(args)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    # TODO: Test data made it to dest


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


@has_dcmtk
def test_forward(
    make_local_node, make_dcmtk_nodes, make_dcm_config_file, make_local_dir
):
    local_node = make_local_node()
    src_remote, init_qr, src_dir = make_dcmtk_nodes([local_node], "PATIENT-1")
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
            str(src_remote),
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
        "--include-group",
        "8",
    ]
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path)
        dump_res = runner.invoke(cli, args + [str(dcm_path)])
        print(dump_res.stdout)
        assert dump_res.exit_code == 0
        assert dump_res.stdout.strip() == ds.SOPInstanceUID
