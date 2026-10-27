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
