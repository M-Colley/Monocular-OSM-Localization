"""Dense pass 1 (--vpr-pass1-cap): streamed reference descriptors and the torch decode."""
import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from monocular_osm import kartaview_vpr as kv
from monocular_osm.pipeline import PipelineConfig


def _problem(n_q=24, n_r=300, seed=0):
    rng = np.random.default_rng(seed)
    latlon = np.column_stack([50.0 + rng.uniform(-0.02, 0.02, n_r), 8.0 + rng.uniform(-0.03, 0.03, n_r)])
    sims = rng.uniform(0.0, 0.3, (n_q, n_r)).astype(np.float32)
    # a drive: references near a line score higher, frame by frame
    for q in range(n_q):
        sims[q, (q * 7) % n_r] += 0.4
    return sims, latlon


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_decode_matches_the_numpy_decode(seed):
    sims, latlon = _problem(seed=seed)
    expected = kv._viterbi_decode(sims, latlon, 5.0)
    got = kv._viterbi_decode_torch(sims, latlon, 5.0, device="cpu")
    assert got.dtype == expected.dtype and np.array_equal(got, expected)


def test_large_sets_decode_on_the_gpu_or_fall_back(monkeypatch):
    sims, latlon = _problem(n_r=40)
    monkeypatch.setattr(kv, "_STREAM_MIN_REFS", 10)
    calls = []
    monkeypatch.setattr(kv, "_viterbi_decode_torch",
                        lambda s, l, dt, device: calls.append(device) or np.zeros(len(s), np.int32))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    kv._viterbi_decode(sims, latlon, 5.0)
    assert calls == ["cuda"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="memory bound"):
        kv._viterbi_decode(sims, latlon, 5.0)


class _Resp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


def _jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((64, 128, 3), value, np.uint8))
    return buf.tobytes()


def _fake_embed(device, imgs, *, batch_size=24, prepare=None):
    # one 2-d descriptor per image: its mean pixel and a constant
    return torch.tensor([[float(np.asarray(i).mean()), 1.0] for i in imgs])


def _refs(n):
    return [{"id": str(1000 + k), "lat": 50.0 + k * 1e-4, "lon": 8.0,
             "url": None if k == 3 else f"https://img/{k}"} for k in range(n)]


def test_streaming_keeps_descriptors_only_and_caches_them(tmp_path, monkeypatch):
    class Session:
        def get(self, url, timeout):
            k = int(url.rsplit("/", 1)[1])
            return _Resp(404) if k == 5 else _Resp(200, _jpeg(10 * k))
    monkeypatch.setattr("requests.Session", Session)
    monkeypatch.setattr(kv, "_embed", _fake_embed)
    refs = _refs(12)
    emb, xy = kv._embed_refs_streaming(refs, "cpu", str(tmp_path), "m", chunk=4)
    served = [r for r in refs if r["url"] and r["id"] != "1005"]   # no URL / HTTP 404 dropped
    assert emb.shape == (len(served), 2) and xy.shape == (len(served), 2)
    assert np.allclose(xy[:, 0], [r["lat"] for r in served])
    assert np.allclose(emb[:, 0].numpy(), [10 * (int(r["id"]) - 1000) for r in served], atol=2.0)
    assert not (tmp_path / "ref_img_store.npz").exists()            # no pixel store

    class Offline:
        def get(self, url, timeout):
            raise AssertionError("a warm cache must not touch the network")
    monkeypatch.setattr("requests.Session", Offline)
    emb2, xy2 = kv._embed_refs_streaming(refs, "cpu", str(tmp_path), "m", chunk=4)
    assert torch.equal(emb2, emb) and np.array_equal(xy2, xy)


def test_only_large_sets_stream(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(kv, "_embed_refs_streaming", lambda refs, *a, **k: seen.append(len(refs)) or (None, None))
    monkeypatch.setattr(kv, "_STREAM_MIN_REFS", 5)
    kv._embed_refs(_refs(6), "cpu", str(tmp_path), "m")
    assert seen == [6]


def test_pass1_cap_config_is_validated(tmp_path):
    def cfg(**kw):
        return PipelineConfig(url="u", city="Bern", data_dir=tmp_path / "d", output_dir=tmp_path / "o", **kw)
    assert cfg().vpr_pass1_cap is None
    assert cfg(vpr_pass1_cap=24000).vpr_pass1_cap == 24000
    for bad in (0, 29, 1.5, True):
        with pytest.raises(ValueError, match="vpr_pass1_cap"):
            cfg(vpr_pass1_cap=bad)


def test_cli_passes_the_pass1_cap_through():
    from monocular_osm.cli import build_arg_parser
    base = ["--url", "https://www.youtube.com/watch?v=x", "--city", "Bern"]
    assert build_arg_parser().parse_args(base + ["--vpr-pass1-cap", "24000"]).vpr_pass1_cap == 24000
    assert build_arg_parser().parse_args(base).vpr_pass1_cap is None
