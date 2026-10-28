"""Runtime proofs use fake sessions/weights; no model or download is executed."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from monocular_osm.stages import poi_runtime as runtime


def _spec(tmp_path):
    for name in runtime._ASSETS.values():
        (tmp_path / name).write_bytes(b"synthetic weight file; never executed")
    return runtime.reader_spec(model_dir=tmp_path)


class FakeDownload:
    @classmethod
    def run(cls, *args, **kwargs):
        return "original behavior"


def _reader(*, gpu=False):
    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        def get_session_options(self):
            return SimpleNamespace(intra_op_num_threads=2, inter_op_num_threads=1)
    def component(**kwargs):
        return SimpleNamespace(session=SimpleNamespace(session=Session()), **kwargs)
    return SimpleNamespace(readtext=lambda image: [],
        _engine=SimpleNamespace(text_det=component(), text_rec=component(rec_batch_num=4), text_cls=component(cls_batch_num=4)))


def test_pinned_assets_cpu_parameters_actual_sessions_and_scoped_download_guard(tmp_path):
    spec = _spec(tmp_path)
    original = FakeDownload.__dict__["run"]
    seen = []
    def factory(**kwargs):
        seen.append(kwargs)
        return _reader()
    with runtime.pinned_reader(spec, reader_factory=factory, download_class=FakeDownload) as (reader, record):
        assert reader.readtext("synthetic image") == []
        with pytest.raises(RuntimeError, match="downloads are disabled"):
            FakeDownload.run("anything")
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(FakeDownload.run).result() == "original behavior"
    assert FakeDownload.__dict__["run"] is original
    assert seen == [{"params": spec["parameters"]}]
    assert spec["parameters"]["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert all(spec["parameters"][f"EngineConfig.onnxruntime.use_{name}"] is False
               for name in ("cuda", "dml", "cann", "coreml"))
    assert record["actual_sessions"]["text_rec"]["providers"] == ["CPUExecutionProvider"]
    assert record["actual_batches"] == {"rec_batch_num": 4, "cls_batch_num": 4}
    assert record["read_calls"] == 1
    assert all(len(asset["sha256"]) == 64 for asset in record["model_assets"].values())


def test_accelerator_provider_rejected_and_global_defaults_restored(tmp_path):
    spec = _spec(tmp_path)
    original = FakeDownload.__dict__["run"]
    with pytest.raises(ValueError, match="providers or thread bounds"):
        with runtime.pinned_reader(spec, reader_factory=lambda **kw: _reader(gpu=True), download_class=FakeDownload):
            pass
    assert FakeDownload.__dict__["run"] is original
    with pytest.raises(RuntimeError, match="deliberate"):
        with runtime.pinned_reader(spec, reader_factory=lambda **kw: _reader(), download_class=FakeDownload):
            raise RuntimeError("deliberate")
    assert FakeDownload.__dict__["run"] is original


def test_missing_cached_model_fails_without_constructing_a_reader(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.reader_spec(model_dir=tmp_path)
    with pytest.raises(ValueError, match="RapidOCR"):
        runtime.reader_spec("easyocr", model_dir=tmp_path)


def test_torch_threads_are_bounded_and_restored_on_failure():
    state = {"threads": 17}
    fake = SimpleNamespace(__version__="synthetic", get_num_threads=lambda: state["threads"],
        set_num_threads=lambda count: state.update(threads=count), get_num_interop_threads=lambda: 3)
    with pytest.raises(RuntimeError, match="deliberate"):
        with runtime.torch_cpu_threads(4, torch_module=fake) as record:
            assert state["threads"] == record["intra_op_threads"] == 4
            assert record["inter_op_threads"] == 3 and record["device"] == "cpu"
            raise RuntimeError("deliberate")
    assert state["threads"] == 17
