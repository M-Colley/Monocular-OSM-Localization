"""Query experiments preserve the control and reference identity/order."""

import numpy as np
import pytest

from monocular_osm import kartaview_vpr as kv


def test_square_views_span_landscape_and_portrait_images():
    wide = np.broadcast_to(np.arange(6)[None, :, None], (2, 6, 3))
    assert [kv._square_query_crop(wide, i)[0, :, 0].tolist() for i in range(3)] == [
        [0, 1], [2, 3], [4, 5]]
    tall = wide.transpose(1, 0, 2)
    assert [kv._square_query_crop(tall, i)[:, 0, 0].tolist() for i in range(3)] == [
        [0, 1], [2, 3], [4, 5]]


def test_aspect_view_uses_vit_grid_without_tripling_the_pixel_budget():
    w, h = kv._aspect_query_size(np.zeros((375, 1242, 3), np.uint8))
    assert w % 14 == h % 14 == 0
    assert w / h == pytest.approx(1242 / 375, rel=0.04)
    assert w * h == pytest.approx(512 ** 2, rel=0.08)


def test_crops_merge_reference_scores_within_each_query_in_order(monkeypatch):
    torch = pytest.importorskip("torch")
    frames = [np.zeros((2, 6, 3), np.uint8) for _ in range(2)]

    def embed(device, views, **kwargs):
        assert len(views) == 6
        assert views[0].shape == (2, 2, 3)
        assert kwargs["batch_size"] == 4 and callable(kwargs["prepare"])
        return torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.4, 0.7],
                             [0.1, 0.9], [0.8, 0.2], [0.3, 0.4]])

    monkeypatch.setattr(kv, "_embed", embed)
    sims = kv._query_similarities(frames, [0, 1], torch.eye(2), "cpu", mode="crops")
    np.testing.assert_allclose(sims, [[0.9, 0.8], [0.8, 0.9]])


def test_warp_control_keeps_original_preprocessing_and_embedding_call(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(kv, "_prep", lambda im: f"prepped:{im}")

    def embed(device, images):
        calls.append((device, images))
        return torch.tensor([[0.1, 0.9], [0.7, 0.3]])

    monkeypatch.setattr(kv, "_embed", embed)
    sims = kv._query_similarities(["a", "b", "c"], [2, 0], torch.eye(2), "cpu")
    assert calls == [("cpu", ["prepped:c", "prepped:a"])]
    np.testing.assert_allclose(sims, [[0.1, 0.9], [0.7, 0.3]])


def test_unknown_query_transform_is_rejected():
    with pytest.raises(ValueError, match="unknown VPR query mode"):
        kv._query_similarities([], [], None, "cpu", mode="typo")


def test_explicit_warp_batch_is_honored_without_materializing_query_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    reads = []

    class Frames:
        def __getitem__(self, index):
            reads.append(index)
            return np.zeros((4, 8, 3), np.uint8)

    def embed(device, images, **kwargs):
        assert not reads                 # raw frames have not been loaded up front
        assert len(images) == 2 and kwargs["batch_size"] == 2
        assert images[0].shape == (4, 8, 3) and reads == [3]
        assert callable(kwargs["prepare"])
        return torch.eye(2)

    monkeypatch.setattr(kv, "_embed", embed)
    sims = kv._query_similarities(Frames(), [3, 1], torch.eye(2), "cpu", batch_size=2)
    np.testing.assert_array_equal(sims, np.eye(2))


def test_query_batch_must_be_positive():
    with pytest.raises(ValueError, match="batch size must be positive"):
        kv._query_similarities([], [], None, "cpu", batch_size=0)


@pytest.mark.parametrize("mode", ["warp", "crops", "aspect"])
def test_query_descriptors_are_reused_across_reference_regions_without_changing_scores(
        monkeypatch, mode):
    torch = pytest.importorskip("torch")
    frames = [np.zeros((4, 8, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr(kv, "_MODEL", object())
    calls = []

    def embed(device, images, **kwargs):
        calls.append(len(images))
        values = torch.arange(1, 2 * len(images) + 1, dtype=torch.float32).reshape(-1, 2)
        return torch.nn.functional.normalize(values, dim=1)

    monkeypatch.setattr(kv, "_embed", embed)
    cache = {}
    first_refs = torch.eye(2)
    other_refs = torch.tensor([[0.6, 0.8], [0.8, 0.6], [-1., 0.]])
    kv._query_similarities(frames, [0, 1], first_refs, "cpu", mode=mode,
                           batch_size=2, query_cache=cache)
    reused = kv._query_similarities(frames, [0, 1], other_refs, "cpu", mode=mode,
                                    batch_size=2, query_cache=cache)
    assert calls == [6 if mode == "crops" else 2]
    assert next(iter(cache.values())).shape == (calls[0], 2)
    independent = kv._query_similarities(frames, [0, 1], other_refs, "cpu", mode=mode,
                                         batch_size=2)
    np.testing.assert_array_equal(reused, independent)
    assert len(calls) == 2


def test_query_cache_invalidates_changed_frame_objects_models_devices_modes_and_indices(monkeypatch):
    torch = pytest.importorskip("torch")
    frames = [np.zeros((4, 8, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr(kv, "_MODEL", object())
    calls = []

    def embed(device, images, **kwargs):
        calls.append(1)
        return torch.ones((len(images), 2))

    monkeypatch.setattr(kv, "_embed", embed)
    cache = {}

    def query(source=frames, indices=(0, 1), mode="warp", device="cpu"):
        kv._query_similarities(source, indices, torch.eye(2), device, mode=mode,
                               batch_size=2, query_cache=cache)

    query()
    frames[0] = np.ones((4, 8, 3), np.uint8)
    query()
    assert len(cache) == 1
    query(source=list(frames))
    query(indices=(1, 0))
    monkeypatch.setattr(kv, "_MODEL", object())
    query()
    monkeypatch.setattr(kv, "_MODEL_NAME", "different actual backbone")
    query()
    query(mode="crops")
    query(device="different-device-token")
    assert len(calls) == 8


def test_cache_hits_do_not_decode_disk_backed_query_frames_again(monkeypatch):
    torch = pytest.importorskip("torch")
    reads = []

    class Frames:
        def __getitem__(self, index):
            reads.append(index)
            return np.zeros((4, 8, 3), np.uint8)

    def embed(device, images, **kwargs):
        for i in range(len(images)):
            _ = images[i]
        return torch.ones((len(images), 2))

    monkeypatch.setattr(kv, "_embed", embed)
    frames, cache = Frames(), {}
    for _ in range(2):
        kv._query_similarities(frames, [3, 1], torch.eye(2), "cpu", batch_size=2,
                               query_cache=cache)
    assert reads == [3, 1]


def test_public_track_calls_share_query_forwards_only_when_cache_is_explicit(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(kv, "_MODEL", object())
    monkeypatch.setattr(kv, "_resolve_backbone", lambda *args: "megaloc")
    frames = [np.zeros((4, 8, 3), np.uint8) for _ in range(3)]
    references = [{"id": i, "lat": 48.4, "lon": 10.0 + i * 0.0001} for i in range(40)]
    monkeypatch.setattr(kv, "_fetch_refs_for", lambda *args: references)
    monkeypatch.setattr(kv, "_embed_refs", lambda refs, *args, **kwargs: (
        torch.tensor([[0.6, 0.8]] * len(refs)),
        np.array([[ref["lat"], ref["lon"]] for ref in refs])))
    calls = []

    def embed(device, images, **kwargs):
        calls.append(len(images))
        return torch.nn.functional.normalize(torch.tensor([[1., 0.], [0.5, 0.5], [0., 1.]]), dim=1)

    monkeypatch.setattr(kv, "_embed", embed)
    cache = {}
    first = kv.kartaview_vpr_track(frames, (48.4, 10.0), n_query=3, device="cpu",
                                   sequence_decode=False, query_batch_size=2, query_cache=cache)
    regional = kv.kartaview_vpr_track(frames, (48.5, 10.1), n_query=3, device="cpu",
                                      sequence_decode=False, query_batch_size=2, query_cache=cache)
    assert calls == [3]
    fresh_video_cache = {}
    independent = kv.kartaview_vpr_track(frames, (48.5, 10.1), n_query=3, device="cpu",
                                         sequence_decode=False, query_batch_size=2,
                                         query_cache=fresh_video_cache)
    assert calls == [3, 3]
    for reused, fresh in zip(regional, independent):
        np.testing.assert_array_equal(reused, fresh)
    assert first is not None
