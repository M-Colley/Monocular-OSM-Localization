"""Private synthetic posterior grids; no maps, cases or labels."""
from copy import deepcopy
import math
import numpy as np
import pytest
from monocular_osm.graph_filter import GraphParticleFilter, FilterConfig


def posterior(xy, *, grid=None, cell=1.):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    pf = GraphParticleFilter.__new__(GraphParticleFilter)
    pf.cfg = FilterConfig(n_particles=len(xy), mode_grid_m=cell)
    pf.x, pf.y = xy[:, 0].copy(), xy[:, 1].copy()
    pf.logk = np.zeros(len(xy)); pf.eps = np.zeros(len(xy))
    pf._dense_grid = grid
    pf.rng = np.random.default_rng(17)
    return pf


def oracle(xy, weights, cell, K):
    """Independent dictionary enumeration and scalar summation."""
    occupied = {}
    for (x, y), w in zip(xy, weights):
        if w == 0: continue
        key = (math.floor(x / cell), math.floor(y / cell))
        occupied.setdefault(key, []).append((float(w), float(w*x), float(w*y)))
    cells = {key: tuple(math.fsum(v[j] for v in rows) for j in range(3)) for key, rows in occupied.items()}
    offsets = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    centres = sorted({(x-dx, y-dy) for x, y in cells for dx, dy in offsets})
    rows = []
    for x, y in centres:
        values = [cells.get((x+dx, y+dy), (0., 0., 0.)) for dx, dy in offsets]
        rows.append((math.fsum(v[0] for v in values), x, y, values))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return tuple(np.asarray([[v[j] for v in row[3]] for row in rows[:K]]) for j in range(3))


def test_broad_heaviest_mode_survives64_larger_single_cells():
    cluster = np.array([(5.+20*i, 5.+20*j) for i in range(3) for j in range(3)])
    islands = np.column_stack([1000.+100*np.arange(64), np.full(64, 1000.)])
    xy = np.vstack([cluster, islands]); w = np.r_[np.full(9, .009), np.full(64, .01)]; w /= w.sum()
    for grid in (None, (400, 100)):
        pf = posterior(xy, grid=grid, cell=20.)
        assert pf._block_stats(w, K=1)[0].sum() == pytest.approx(w[:9].sum())
        result = pf.summary(w)
        np.testing.assert_allclose(result["mode"], [25., 25.], atol=1e-12)
        assert result["n_modes"] == 1 and result["confidence"] > 4*w[-1]


def test_empty_centre_can_be_unique_heaviest_block():
    xy = np.array([[.25, .25], [2.25, 2.25]]); w = np.array([.5, .5])
    got = posterior(xy)._block_stats(w, K=1)
    assert got[0].shape == (1, 9) and got[0].sum() == 1.
    assert got[0][0, 0] == got[0][0, 8] == .5
    for a, b in zip(got, oracle(xy, w, 1., 1)): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("grid", [None, (1, 1), (3, 5), (4000, 5000)])
def test_padding_has_no_row_wrap_or_boundary_mass_duplication(grid):
    xy = np.array([[.25, 3.25], [1.25, .25]]); w = np.array([.375, .625])
    got = posterior(xy, grid=grid)._block_stats(w)
    assert got[0].sum(axis=1).max() == .625
    for a, b in zip(got, oracle(xy, w, 1., 64)): np.testing.assert_array_equal(a, b)
    one = posterior([[.25, .25]], grid=grid)._block_stats(np.array([1.]))
    np.testing.assert_array_equal(one[0].sum(axis=1), np.ones(9))


@pytest.mark.parametrize("K", [1, 3, 64, 100])
def test_complete_blocks_match_scalar_oracle_with_negative_cells(K):
    rng = np.random.default_rng(813)
    xy = rng.integers(-9, 10, (40, 2)).astype(np.float64)+.25
    w = rng.integers(0, 5, 40).astype(np.float64)/128.
    for grid in (None, (4000, 5000)):
        for a, b in zip(posterior(xy, grid=grid)._block_stats(w, K), oracle(xy, w, 1., K)):
            np.testing.assert_array_equal(a, b)


def test_negative_subcell_floor_and_dense_sparse_equivalence():
    xy = np.array([[-.01, -.01], [.01, .01], [-1.01, 2.01], [4., -3.]])
    w = np.array([.125, .25, .375, .25]); sparse = posterior(xy)._block_stats(w)
    for a, b in zip(posterior(xy, grid=(4000, 5000))._block_stats(w), sparse): np.testing.assert_array_equal(a, b)
    for a, b in zip(sparse, oracle(xy, w, 1., 64)): np.testing.assert_array_equal(a, b)


def test_lexical_ties_repeat_without_input_or_rng_mutation():
    xy = np.array([[-4.75, -4.75], [5.25, 5.25]]); w = np.array([.5, .5])
    pf = posterior(xy); rng_before = deepcopy(pf.rng.bit_generator.state)
    first = pf._block_stats(w, K=2)
    for a, b in zip(first, oracle(xy, w, 1., 2)): np.testing.assert_array_equal(a, b)
    for _ in range(3):
        for a, b in zip(pf._block_stats(w, K=2), first): np.testing.assert_array_equal(a, b)
    pf.summary(w)
    assert pf.rng.bit_generator.state == rng_before
    np.testing.assert_array_equal(np.column_stack([pf.x, pf.y]), xy)
    np.testing.assert_array_equal(w, [.5, .5])


@pytest.mark.parametrize("fault", ["empty", "zero", "negative", "nan", "position", "shape", "grid", "Kzero", "Kbool"])
def test_empty_zero_or_invalid_summary_refused(fault):
    pf = posterior([[0., 0.], [1., 1.]]); w = np.array([.5, .5]); K = 64
    if fault == "empty": pf = posterior([]); w = np.array([])
    if fault == "zero": w[:] = 0
    if fault == "negative": w[0] = -.1
    if fault == "nan": w[0] = np.nan
    if fault == "position": pf.x[0] = np.inf
    if fault == "shape": w = np.ones(3)
    if fault == "grid": pf.cfg.mode_grid_m = 0.
    if fault == "Kzero": K = 0
    if fault == "Kbool": K = True
    with pytest.raises(ValueError): pf._block_stats(w, K)


def test_zero_weight_particle_creates_no_candidate_or_mode_shift():
    xy = np.array([[-.25, .25], [1e8, -1e8]]); w = np.array([1., 0.])
    for a, b in zip(posterior(xy)._block_stats(w), posterior(xy[:1])._block_stats(w[:1])):
        np.testing.assert_array_equal(a, b)
