"""The readiness utility stays offline and never overwrites a frozen run."""
import socket

import pytest

from scripts.probe_poi_ocr_readiness import main, offline_network


def test_network_guard_refuses_connections_and_restores_on_failure():
    previous_connect, previous_create = socket.socket.connect, socket.create_connection
    with pytest.raises(RuntimeError, match="Network is disabled"):
        with offline_network():
            socket.create_connection(("example.invalid", 80))
    assert socket.socket.connect is previous_connect
    assert socket.create_connection is previous_create


def test_existing_output_refused_before_runtime_imports(tmp_path):
    video, output = tmp_path / "video.mp4", tmp_path / "frozen"
    video.write_bytes(b"video")
    output.mkdir()
    marker = output / "prediction.json"
    marker.write_text("frozen")
    with pytest.raises(FileExistsError):
        main(["--video", str(video), "--city", "Example City", "--end", "46", "--output", str(output)])
    assert marker.read_text() == "frozen"
