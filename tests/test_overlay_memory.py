"""Overlay detection stays bounded with many high-resolution samples."""
import tracemalloc

import numpy as np

from monocular_osm.plane_speed import static_overlay_mask


def test_streamed_overlay_variance_matches_population_standard_deviation():
    rng = np.random.default_rng(12)
    frames = [rng.integers(0, 256, (60, 80), dtype=np.uint8) for _ in range(15)]
    for frame in frames:
        frame[4:12, 8:40] = 120
    expected = (np.std(np.stack(frames).astype(np.float64), axis=0) < 3).astype(np.uint8) * 255
    actual = static_overlay_mask(frames, n_sample=15, dilate_px=0)
    np.testing.assert_array_equal(actual, expected)


def test_overlay_estimation_does_not_materialize_the_sample_batch():
    class GeneratedFrames:
        def __len__(self):
            return 10000
        def __getitem__(self, index):
            image = np.full((512, 512), index % 255, dtype=np.uint8)
            image[:20, :100] = 100
            return image

    tracemalloc.start()
    try:
        mask = static_overlay_mask(GeneratedFrames(), n_sample=60, dilate_px=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 ** 2
    assert np.all(mask[:20, :100] == 255)
    assert not np.any(mask[30:, 110:])
