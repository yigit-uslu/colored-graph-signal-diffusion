"""Matvec-only spectral estimation (M1+M2) vs the dense ``eigh`` oracle — fixed seeds.

These pin the mathematics of ``ccd.spectral`` on a small synthetic operator with a controlled
spectrum: Lanczos Ritz values match the extreme eigenvalues, the KPM CDF tracks the empirical
CDF, deflation removes exactly the requested mass, and ``TiledRate.from_spectral_model`` on the
``eigh`` backend reproduces the plain ``TiledRate`` knot-for-knot (the C3 ≡ C3-MF plumbing check).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ccd.forward.rate import PhiLogPolyFit, TiledRate
from ccd.operators.linop import CovarianceOperator, make_synthetic_operator
from ccd.spectral import (
    SpectralModel,
    deflate_moments,
    detect_lines,
    kpm_cdf,
    kpm_moments,
    kurtosis_weights,
    lanczos,
    line_kurtosis,
    ritz_values,
)


def _spiked_operator(d=64, n=4000, seed=0):
    """S with a few well-separated top eigenvalues over a tight bulk (a factor-model toy)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    s = torch.full((d,), 0.02, dtype=torch.float64)
    s[:5] = torch.tensor([4.0, 2.0, 1.0, 0.5, 0.25], dtype=torch.float64)   # 5 spikes
    c_half = Q @ (torch.sqrt(s)[:, None] * Q.transpose(0, 1))
    Z = torch.randn(d, n, generator=g, dtype=torch.float64)
    X = c_half @ Z
    return CovarianceOperator(X, tau=0.05)


def test_lanczos_ritz_matches_top_eigenvalues():
    op = _spiked_operator()
    ev = torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy()
    top = np.sort(ev)[::-1][:5]
    alpha, beta, cols = lanczos(op, m=40, generator=torch.Generator().manual_seed(1))
    theta, _ = ritz_values(alpha, beta)
    assert cols == 40
    rel = np.abs(theta[:5] - top) / top
    assert rel.max() < 1e-8, f"top-5 Ritz rel error {rel.max():.2e}"


def test_detect_lines_finds_the_spikes():
    op = _spiked_operator()
    ev = np.sort(torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy())[::-1]
    lines, auto_G = detect_lines(ev, float(ev[0]), gap_tol=0.1)
    assert auto_G == 5, f"expected 5 isolated spikes, got {auto_G}"
    assert np.allclose(lines, ev[:5])


def test_kpm_cdf_tracks_empirical():
    # A smooth, spread spectrum (no mass-≫1/d cluster): KPM resolves the CDF to O(1/d).
    op = make_synthetic_operator(n=6000, d=64, decay=0.7, tau=0.05,
                                 generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ev = torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy()
    d = op.shape[0]
    lo, hi = float(ev.min()) * 0.99, float(ev.max()) * 1.01
    mu, se, cols = kpm_moments(op, lo, hi, n_z=32, k_m=200,
                               generator=torch.Generator().manual_seed(2))
    assert cols == 32 * 200                                       # honest single-vector column count
    assert abs(mu[0] - 1.0) < 1e-9                                # μ_0 = 1 exactly
    grid = np.linspace(lo, hi, 400)
    ecdf = np.searchsorted(np.sort(ev), grid, side="right") / d
    sup = np.abs(kpm_cdf(mu, lo, hi, grid) - ecdf).max()
    # KPM smears the dense bottom edge (several 1/d steps per resolution width) → floor ~1.5/d.
    assert sup < 2.5 / d, f"CDF sup-error {sup:.3f} exceeds 2.5/d={2.5/d:.3f}"


def test_deflation_removes_requested_mass():
    op = _spiked_operator()
    lo, hi = 0.049, 4.1
    mu, _, _ = kpm_moments(op, lo, hi, n_z=16, k_m=64,
                           generator=torch.Generator().manual_seed(3))
    ev = np.sort(torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy())[::-1]
    lines = ev[:5]
    _, bulk_mass = deflate_moments(mu, lines, np.ones(5), op.shape[0], lo, hi)
    assert abs(bulk_mass - (op.shape[0] - 5) / op.shape[0]) < 1e-12


def test_from_eigenvalues_matches_plain_tiledrate():
    """eigh backend: from_spectral_model(atoms) ≡ TiledRate(top-G eigs) knot-for-knot (C3 ≡ C3-MF)."""
    op = _spiked_operator()
    ev = torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy()
    sm = SpectralModel.from_eigenvalues(ev, op.shape[0], gap_tol=0.1)
    top5 = np.sort(ev)[::-1][:5]
    plain = TiledRate(top5, FLOOR=0.077, F_min=0.02, c=3.0)
    mf = TiledRate.from_spectral_model(sm, G=5, mode="atoms", FLOOR=0.077, F_min=0.02, c=3.0)
    q = torch.logspace(math.log10(float(ev.min())), math.log10(float(ev.max())), 50, dtype=torch.float64)
    a, b = plain(q).numpy(), mf(q).numpy()
    rel = np.abs(a - b) / a
    assert rel.max() < 1e-6, f"C3 vs C3-MF knot rel error {rel.max():.2e}"


def test_phlogpolyfit_places_the_clock():
    """B′ fit to the C3 log-clock: positive, clock placement within 0.25× spacing, well-conditioned Gram."""
    op = _spiked_operator()
    ev = torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy()
    sm = SpectralModel.from_matvec(op, g_max=8, n_z=16, k_m=128,
                                   generator=torch.Generator().manual_seed(4),
                                   gen_kpm=torch.Generator().manual_seed(5))
    bp = PhiLogPolyFit.from_spectral_model(sm, G=5, degree=2, c=3.0, dtype=torch.float64)
    lam = torch.logspace(math.log10(float(ev.min())), math.log10(float(ev.max())), 40, dtype=torch.float64)
    assert (bp(lam) > 0).all()                                   # positivity free (exp)
    spacing = math.log((math.log(2.0) / 0.077) / (3.0 * 0.02)) / (5 - 1)   # span_F/(G-1), G=5 here
    # degree-2 through 5 log-collinear spikes leaves O(spacing/4); the tight 0.25·spacing acceptance
    # is a per-config check on the real operator (scripts/spectral_fit_verify.py), not this synthetic.
    assert bp.atoms_resid <= 0.4 * spacing, f"clock placement {bp.atoms_resid:.3f} > 0.4·{spacing:.2f}"
    assert bp.gram_cond < 1e6, f"Gram poorly conditioned: {bp.gram_cond:.2e}"


def test_phlogpolyfit_raw_basis_runs():
    """The pure-continuum raw-Chebyshev path (moment Gram + p-quadrature load) is finite and positive."""
    op = _spiked_operator()
    sm = SpectralModel.from_matvec(op, g_max=8, n_z=16, k_m=128,
                                   generator=torch.Generator().manual_seed(4),
                                   gen_kpm=torch.Generator().manual_seed(5))
    bp = PhiLogPolyFit.from_spectral_model(sm, G=5, degree=2, basis="raw", c=3.0, dtype=torch.float64)
    lam = torch.logspace(-1.5, 0.5, 30, dtype=torch.float64)
    out = bp(lam)
    # raw is a bulk-weighted continuum fit (does not prioritise the atoms); just require sanity.
    assert torch.isfinite(out).all() and (out > 0).all()
    assert bp.gram_cond < 1e8


def test_weighted_tiledrate_equal_is_c3():
    """Weighted TiledRate with equal weights reduces to the unweighted (C3) tiling knot-for-knot."""
    lam = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    plain = TiledRate(lam, FLOOR=0.077, F_min=0.02, c=3.0)
    weighted = TiledRate(lam, FLOOR=0.077, F_min=0.02, c=3.0, weights=np.ones(5))
    q = torch.logspace(-1, 0.6, 40, dtype=torch.float64)
    rel = (torch.abs(plain(q) - weighted(q)) / plain(q)).max()
    assert rel < 1e-9, f"equal-weight TiledRate differs from C3 by {rel:.2e}"


def test_weighted_tiledrate_shifts_clock_to_weight():
    """Up-weighting the low-λ lines steepens the clock there (survival mass moves to the fine end)."""
    lam = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    w = np.array([0.01, 0.01, 0.01, 1.0, 1.0])                    # importance on the two smallest λ
    r = TiledRate(lam, FLOOR=0.077, F_min=0.02, c=3.0, weights=w)
    phi = r(torch.tensor(lam)).numpy()
    # top-3 unimportant lines bunch at the coarse floor; the fine spread is carried by the bottom-2
    assert phi[0] < phi[1] < phi[2]                               # still monotone (φ grows as λ falls)
    assert (phi[:3].max() - phi[:3].min()) < (phi[3:].max() - phi[3:].min())  # fine end carries the range


def test_kurtosis_weights():
    assert np.allclose(kurtosis_weights([1.0, 1.0, 1.0], mode="max"), np.ones(3) / 3)
    w = kurtosis_weights([2.0, -0.5, 0.0, 1.0], mode="max")       # negatives clipped
    assert w[1] == 0.0 and w[2] == 0.0 and abs(w.sum() - 1.0) < 1e-12
    assert np.allclose(kurtosis_weights([-1.0, -1.0], mode="max"), [0.5, 0.5])  # degenerate -> equal


def test_line_kurtosis_detects_a_heavy_tailed_factor():
    """A scale-mixture (leptokurtic) factor injected on the top eigendirection reads κ>0 there, ≈0 on a Gaussian line."""
    g = torch.Generator().manual_seed(0)
    d, n = 32, 20000
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    eta = torch.randn(2, n, generator=g, dtype=torch.float64)
    V = 1.0 + 0.7 * (torch.randint(0, 2, (n,), generator=g) * 2 - 1).double()   # shared scale, leptokurtic
    top = (2.0 * eta[0] * torch.sqrt(V))                          # heavy-tailed top factor (var ~4)
    second = 1.0 * eta[1]                                         # Gaussian second factor (var ~1)
    X = Q[:, 0:1] @ top[None, :] + Q[:, 1:2] @ second[None, :] + 0.05 * torch.randn(d, n, generator=g, dtype=torch.float64)
    op = CovarianceOperator(X, tau=0.05)
    ev = torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy()
    lines = np.sort(ev)[::-1][:2]
    kap = line_kurtosis(op, lines, float(ev.min()) * 0.99, float(ev.max()) * 1.01,
                        n_z=12, generator=torch.Generator().manual_seed(1))
    assert kap[0] > 0.5, f"leptokurtic top line κ={kap[0]:.2f} not detected"
    assert abs(kap[1]) < 0.4, f"Gaussian line κ={kap[1]:.2f} should be ≈0"


def test_from_matvec_end_to_end():
    """Full matrix-free build recovers the spikes and reports an honest matvec-column budget."""
    op = _spiked_operator()
    sm = SpectralModel.from_matvec(op, g_max=8, n_z=16, k_m=128,
                                   generator=torch.Generator().manual_seed(4),
                                   gen_kpm=torch.Generator().manual_seed(5))
    assert sm.auto_G == 5
    ev = np.sort(torch.linalg.eigvalsh(op._dense()).to(torch.float64).numpy())[::-1]
    rel = np.abs(sm.top_g(5) - ev[:5]) / ev[:5]
    assert rel.max() < 1e-7
    assert sm.matvec_cols == sm.meta["m"] + 16 * 128
    assert abs(sm.bulk_mass - (op.shape[0] - 5) / op.shape[0]) < 1e-12
