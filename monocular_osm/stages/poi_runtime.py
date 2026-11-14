"""Explicit, scoped CPU runtimes for the experimental POI component."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

from ..provenance import file_sha256

_DOWNLOAD_LOCK = threading.RLock()
_TORCH_LOCK = threading.RLock()
_ASSETS = {"Det": "PP-OCRv6_det_small.onnx", "Rec": "PP-OCRv6_rec_small.onnx",
           "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx"}


def reader_spec(engine="rapidocr", *, model_dir=None):
    if engine != "rapidocr":
        raise ValueError("POI OCR supports the explicit cached RapidOCR runtime only")
    if model_dir is None:
        module = importlib.util.find_spec("rapidocr")
        if module is None or module.origin is None:
            raise ImportError("POI OCR requires RapidOCR and cached PP-OCRv6 small assets")
        model_dir = Path(module.origin).parent / "models"
    assets = {}
    for task, name in _ASSETS.items():
        path = (Path(model_dir) / name).resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required cached OCR asset missing: {path}")
        assets[task] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    params = {f"{task}.model_path": asset["path"] for task, asset in assets.items()}
    params.update({"Global.log_level": "error", "Global.text_score": .3,
        "Global.use_det": True, "Global.use_rec": True, "Global.use_cls": True,
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False, "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False, "EngineConfig.onnxruntime.use_coreml": False,
        "Rec.rec_batch_num": 4, "Cls.cls_batch_num": 4})
    spec = {"backend": "rapidocr_onnxruntime", "parameters": params,
            "model_assets": assets, "downloads_disabled": True}
    spec["spec_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=False).encode()).hexdigest()
    return spec


def validate_reader_record(record):
    if record.get("backend") != "rapidocr_onnxruntime" or not record.get("actual_sessions"):
        raise ValueError("OCR runtime has no verified ONNX Runtime CPU session")
    for session in record["actual_sessions"].values():
        if (session.get("providers") != ["CPUExecutionProvider"]
                or session.get("intra_op_num_threads") != 2 or session.get("inter_op_num_threads") != 1):
            raise ValueError("OCR runtime providers or thread bounds differ from the strict CPU configuration")
    for batch in record.get("actual_batches", {}).values():
        if batch is not None and batch != 4:
            raise ValueError("OCR actual recognition/classification batch is not bounded to four")


def _record_reader(reader, spec):
    sessions, batches = {}, {}
    for name in ("text_det", "text_rec", "text_cls"):
        component = getattr(reader._engine, name, None)
        if component is None:  # RapidOCR initializes components lazily.
            continue
        session = component.session.session
        options = session.get_session_options()
        sessions[name] = {"providers": list(session.get_providers()),
                          "intra_op_num_threads": int(options.intra_op_num_threads),
                          "inter_op_num_threads": int(options.inter_op_num_threads)}
    for name, field in (("text_rec", "rec_batch_num"), ("text_cls", "cls_batch_num")):
        component = getattr(reader._engine, name, None)
        batches[field] = int(getattr(component, field)) if component is not None else None
    record = {**deepcopy(spec), "actual_sessions": sessions, "actual_batches": batches, "cache_hit": False}
    if sessions:
        validate_reader_record(record)
    return record


@contextmanager
def pinned_reader(spec, *, reader_factory=None, download_class=None):
    """Inject explicit assets/providers; block downloads only in this OCR thread."""
    if reader_factory is None:
        from ..scene_text import RapidOcrReader
        reader_factory = RapidOcrReader
    if download_class is None:
        from rapidocr.utils.download_file import DownloadFile
        download_class = DownloadFile
    with _DOWNLOAD_LOCK:
        previous_descriptor = download_class.__dict__["run"]
        previous = download_class.run
        owner = threading.get_ident()
        def guard(cls, *args, **kwargs):
            if threading.get_ident() == owner:
                raise RuntimeError("POI OCR model/font downloads are disabled; cached assets are required")
            return previous(*args, **kwargs)
        download_class.run = classmethod(guard)
        record = deepcopy(spec)
        try:
            reader = reader_factory(params=deepcopy(spec["parameters"]))
            class CountedReader:
                calls = 0
                def readtext(self, image):
                    self.calls += 1
                    return reader.readtext(image)
            counted = CountedReader()
            yield counted, record
            record.update(_record_reader(reader, spec))
            record["read_calls"] = counted.calls
        finally:
            download_class.run = previous_descriptor


@contextmanager
def torch_cpu_threads(threads=4, *, torch_module=None):
    """Bound CPU operator threads for retrieval and restore the caller's state."""
    if torch_module is None:
        import torch as torch_module
    with _TORCH_LOCK:
        previous = int(torch_module.get_num_threads())
        torch_module.set_num_threads(threads)
        try:
            actual = int(torch_module.get_num_threads())
            if actual != threads:
                raise RuntimeError("Torch did not honor the POI CPU thread bound")
            yield {"device": "cpu", "intra_op_threads": actual,
                   "inter_op_threads": int(torch_module.get_num_interop_threads()),
                   "torch_version": str(torch_module.__version__)}
        finally:
            torch_module.set_num_threads(previous)
