"""Validation for the fMRI dataset generator (`scripts/gen_fmri.py`).

The generator reimplements the authors' `construct_graph` in numpy because their networkx calls
are broken under this env (networkx 2.8.8 -> `sp.errstate`, removed in scipy 1.15). These tests
pin that reimplementation, the two opposite array-layout conventions, and the guards that stop a
degenerate operator being emitted silently.

Every check here should fail on at least one plausible mutant: |corr| instead of max(corr,0);
diagonal left in; the normalized and combinatorial Laplacians swapped; the (regions, time) and
(samples, regions) layouts mixed up; the subsample floor left at the authors' 1.

Run: conda run -n torch_env python -m pytest tests/test_gen_fmri.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from gen_fmri import (  # noqa: E402
    MIN_SUBSAMPLE,
    ccd_operator,
    construct_graph,
    load_signals,
    split_signals,
    subsample_indices,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "external", "fmri_stability", "data", "RS_TCs.mat")
has_data = pytest.mark.skipif(not os.path.exists(DATA), reason="RS_TCs.mat not fetched")


@pytest.fixture(scope="module")
def toy():
    """A small signal matrix (regions x time) with both signs of correlation present."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((6, 200))
    base[1] = base[0] + 0.1 * rng.standard_normal(200)      # strongly positive with region 0
    base[2] = -base[0] + 0.1 * rng.standard_normal(200)     # strongly NEGATIVE with region 0
    return base


# --------------------------------------------------------------------- graph construction
def test_rectification_is_positive_part_not_absolute_value(toy):
    """A = max(corr, 0). |corr| would turn the anticorrelated pair into a strong EDGE."""
    A, _, _ = construct_graph(toy)
    corr = np.corrcoef(toy)
    assert corr[0, 2] < -0.9, "fixture no longer has a strongly anticorrelated pair"
    assert A[0, 2] == 0.0, "anticorrelated pair must become a NON-edge"
    assert A[0, 1] == pytest.approx(corr[0, 1]), "positive weights pass through unchanged"
    assert (A >= 0).all()


def test_no_self_loops_and_symmetric(toy):
    A, S_lap, L = construct_graph(toy)
    assert np.allclose(np.diag(A), 0.0)
    for M in (A, S_lap, L):
        assert np.abs(M - M.T).max() < 1e-12


def test_unnormalized_laplacian_matches_networkx(toy):
    """Pin our numpy L = D - A against networkx (which still runs for the unnormalized case)."""
    nx = pytest.importorskip("networkx")
    A, _, L = construct_graph(toy)
    L_nx = nx.laplacian_matrix(nx.from_numpy_array(A)).toarray()
    assert np.abs(L - L_nx).max() < 1e-12


def test_laplacians_have_the_right_identities(toy):
    """L = D - A with row sums 0; S_lap = I - D^-1/2 A D^-1/2 with eigenvalues in [0, 2]."""
    A, S_lap, L = construct_graph(toy)
    deg = A.sum(1)
    assert np.allclose(np.diag(L), deg)
    assert np.abs(L.sum(1)).max() < 1e-10, "combinatorial Laplacian rows must sum to 0"
    with np.errstate(divide="ignore"):
        d_is = 1.0 / np.sqrt(deg)
    d_is[~np.isfinite(d_is)] = 0.0   # isolated nodes; the convention itself is tested separately
    assert np.allclose(S_lap, np.eye(len(deg)) - d_is[:, None] * A * d_is[None, :])
    ev = np.linalg.eigvalsh(S_lap)
    assert ev.min() > -1e-10 and ev.max() < 2.0 + 1e-10
    assert ev.min() < 1e-10, "a graph Laplacian must have a zero eigenvalue"
    # the two must not be interchangeable
    assert np.abs(S_lap - L).max() > 1e-3


def test_isolated_node_uses_the_networkx_convention():
    """A region anticorrelated with everything has degree 0; d^-1/2 -> 0, not inf/NaN."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3, 400))
    sig = np.vstack([base, -base.sum(0)])          # last region opposes all others
    A, S_lap, L = construct_graph(sig)
    deg = A.sum(1)
    assert (deg == 0).any(), "fixture failed to produce an isolated node"
    iso = int(np.argmin(deg))
    assert np.isfinite(S_lap).all() and np.isfinite(L).all()
    assert S_lap[iso, iso] == pytest.approx(1.0)
    assert np.allclose(L[iso], 0.0)


# --------------------------------------------------------------------- guards
def test_constant_region_raises_rather_than_emitting_nan():
    sig = np.random.default_rng(0).standard_normal((4, 50))
    sig[2] = 3.0                                    # zero variance -> corrcoef NaN
    with pytest.raises(ValueError, match="constant"):
        construct_graph(sig)


def test_too_few_timepoints_raises():
    sig = np.random.default_rng(0).standard_normal((4, 1))
    with pytest.raises(ValueError, match=r">= 2 timepoints"):
        construct_graph(sig)


def test_subsample_floor_is_two_not_the_authors_one():
    """Their max(1, int(p/100*T)) can yield a single sample, for which correlation is undefined."""
    assert MIN_SUBSAMPLE == 2
    with pytest.raises(ValueError, match="need >="):
        subsample_indices(100, 1.0)                 # 1% of 100 = 1 sample
    assert len(subsample_indices(100, 2.0)) == 2


def test_subsample_is_without_replacement_and_deterministic():
    a = subsample_indices(500, 10.0, seed=3)
    b = subsample_indices(500, 10.0, seed=3)
    assert len(a) == 50 and len(set(a.tolist())) == 50, "must be without replacement"
    assert np.array_equal(a, b), "same seed must give the same draw"
    assert not np.array_equal(a, subsample_indices(500, 10.0, seed=4))


# --------------------------------------------------------------------- CCD operator
def test_ccd_operator_is_ridged_covariance(toy):
    x = toy.T                                        # (samples, regions)
    tau = 0.01
    C, S = ccd_operator(x, tau=tau)
    assert np.allclose(S, C + tau * np.eye(C.shape[0]))
    assert np.abs(S - S.T).max() < 1e-14
    assert np.allclose(C, np.cov(x, rowvar=False))
    assert np.linalg.eigvalsh(S).min() > tau - 1e-10, "ridge must make S positive definite"


def test_the_two_layout_conventions_are_opposite(toy):
    """construct_graph takes (regions, time); ccd_operator takes (samples, regions). Mixing them
    up is silent when the matrix is square, so pin it on a rectangular case."""
    n_regions, n_time = toy.shape
    assert n_regions != n_time
    A, _, _ = construct_graph(toy)
    C, _ = ccd_operator(toy.T)
    assert A.shape == (n_regions, n_regions)
    assert C.shape == (n_regions, n_regions)
    # feeding the wrong layout produces the wrong-sized operator, not a silent wrong answer
    C_wrong, _ = ccd_operator(toy)
    assert C_wrong.shape == (n_time, n_time)


# --------------------------------------------------------------------- splitting
def test_split_is_a_partition_with_the_papers_proportions():
    n = 1190
    tr, va, te = split_signals(n)
    idx = np.concatenate([tr, va, te])
    assert len(idx) == n and len(set(idx.tolist())) == n, "splits must partition the timeline"
    assert len(te) == round(0.20 * n)                       # paper: 20% test
    assert len(va) == round(0.20 * (n - len(te)))           # paper: 20% of the remainder
    assert list(tr) == sorted(tr), "contiguous split must preserve temporal order"
    assert tr.max() < va.min() < te.min(), "blocks must not interleave in time"


def test_random_split_differs_and_is_seeded():
    a = np.concatenate(split_signals(500, rule="random", seed=1))
    b = np.concatenate(split_signals(500, rule="random", seed=1))
    c = np.concatenate(split_signals(500, rule="random", seed=2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError, match="unknown split rule"):
        split_signals(100, rule="stratified")


# --------------------------------------------------------------------- real data
@has_data
def test_real_data_shape_and_standardisation():
    sig = load_signals()
    assert sig.shape == (360, 1190)
    assert np.abs(sig.mean(1)).max() < 1e-12, "already per-region centred"
    assert np.allclose(sig.std(1), sig.std(1)[0], atol=1e-6), "already per-region standardised"


@has_data
def test_the_two_gsos_are_near_inverses():
    """Regression guard on the documented SIGN TRAP: C_n's top mode is the graph's SMOOTHEST.

    If this ever flips, every statement about the colouring exponent's sign across the two
    parameterisations becomes wrong.
    """
    from scipy.stats import spearmanr

    sig = load_signals()
    _, S_lap, _ = construct_graph(sig)
    C, _ = ccd_operator(sig.T)
    evC, VC = np.linalg.eigh(C)
    _, VL = np.linalg.eigh(S_lap)
    assert abs(float(VC[:, -1] @ VL[:, 0])) > 0.9, "top covariance mode ~ smoothest graph mode"
    assert abs(float(VC[:, -1] @ VL[:, -1])) < 0.1, "and NOT the roughest"
    smooth = np.einsum("ij,jk,ki->i", VC.T, S_lap, VC)
    assert spearmanr(evC, smooth).statistic < -0.5, "spectra must run in opposite directions"
