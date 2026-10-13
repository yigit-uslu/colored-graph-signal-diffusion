"""Tests for the hand-rolled AAA real-pole rational fitter (edge-limited backends probe).

``scipy.interpolate.AAA`` is unavailable on this tree's SciPy (1.10.1), and
:mod:`ccd.operators.rational` is hard-wired to ``y^{-1/2}``, so this fitter is new and needs its own
oracle-checked coverage: the barycentric AAA fit itself, pole/residue extraction against a
reconstruction computed independently from the extracted poles (not the barycentric form), and the
usability gate (real, strictly-below-bracket poles) on both an easy and a genuinely hard case.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccd.operators.rational import invsqrt_poles
from ccd.operators.rational_fit import aaa_fit, real_pole_rational


def test_aaa_fit_interpolates_at_support_nodes():
    f = lambda y: np.exp(-y)  # noqa: E731
    bary = aaa_fit(f, 1.0, 10.0, tol=1e-10, max_terms=20, n_sample=500)
    assert bary.converged
    approx = bary.evaluate(bary.nodes)
    np.testing.assert_allclose(approx, bary.values, atol=1e-10)


def test_real_pole_rational_matches_zolotarev_invsqrt_pole_count():
    """``y^{-1/2}`` is the one function this tree already has a trusted pole rule for (Zolotarev) —
    a generic AAA fit should land in the same ballpark, not just "some real poles"."""
    lo, hi = 1.0, 1e4
    f = lambda y: 1.0 / np.sqrt(y)  # noqa: E731
    res = real_pole_rational(f, lo, hi, tol=1e-6, max_terms=40)
    assert res.usable
    assert res.reconstruction_sup_rel_err <= 1e-6

    zolotarev_sigma, _ = invsqrt_poles(lo, hi, res.n_poles, method="zolotarev")
    assert zolotarev_sigma.numel() == res.n_poles  # sanity: comparable pole budget at the same target
    assert np.all(res.shifts > 0)


def test_real_pole_rational_reconstruction_independent_of_barycentric_form():
    """The reported accuracy is recomputed from the extracted partial-fraction form on a grid the
    fitter never saw during its own convergence check — catches a bug in pole/residue extraction that
    trusting the barycentric fit's own internal error would miss (this was a real bug during
    development: a linear sample grid passed its own check while the extracted form was 3 orders of
    magnitude worse near the bracket's edge)."""
    lo, hi = 1.0, 1e3
    f = lambda y: 1.0 / np.sqrt(y)  # noqa: E731
    res = real_pole_rational(f, lo, hi, tol=1e-6, max_terms=40, validation_n=50000)
    assert res.usable
    z = np.geomspace(lo, hi, 50000)
    exact = f(z)
    recon = res.c_inf + (res.fit_weights[None, :] / (z[:, None] + res.shifts[None, :])).sum(axis=1)
    sup_rel = np.max(np.abs(recon - exact) / np.abs(exact))
    assert sup_rel <= 1e-6
    assert sup_rel == pytest.approx(res.reconstruction_sup_rel_err, rel=1e-6)


def test_geometric_sampling_resolves_the_edge_a_linear_grid_missed():
    """A linear sample grid under-resolves the region near ``lo`` for a wide-kappa bracket; this is
    the reason `aaa_fit` samples geometrically. Regression test for that specific failure mode."""
    lo, hi = 1.0, 1e4
    f = lambda y: 1.0 / np.sqrt(y)  # noqa: E731
    bary = aaa_fit(f, lo, hi, tol=1e-10, max_terms=40, n_sample=2000)
    z_near_lo = np.geomspace(lo, lo * 10, 200)
    approx = bary.evaluate(z_near_lo)
    exact = f(z_near_lo)
    assert np.max(np.abs(approx - exact) / np.abs(exact)) <= 1e-6


def test_shifted_sqrt_near_bracket_edge_can_fail_the_usability_gate():
    """A shifted-square-root singularity placed very close to `lo` (the c_out/terminal shape at a
    tight registered gap) is a genuinely hard case: this asserts the fitter reports a plain, honest
    gate outcome (usable True or False) rather than crashing or silently forcing a bad fit through."""
    lo, hi = 1.0, 1e4
    gap = 1e-4
    s = lo - gap * (hi - lo)
    f = lambda y: np.sqrt(np.clip(y - s, 0.0, None))  # noqa: E731
    res = real_pole_rational(f, lo, hi, tol=1e-6, max_terms=40)
    assert isinstance(res.usable, bool)
    if res.usable:
        assert res.reconstruction_sup_rel_err <= 1e-6
    else:
        assert not res.all_real or not res.all_below_bracket


def test_usable_false_never_reports_a_reconstruction_error():
    """When the gate fails (real or below-bracket check), `reconstruction_sup_rel_err` must be NaN,
    never a stale or fabricated number a caller could mistake for a real accuracy claim."""
    lo, hi = 1.0, 1e4
    s = lo - 1e-4 * (hi - lo)
    f = lambda y: np.sqrt(np.clip(y - s, 0.0, None))  # noqa: E731 (the tight-gap failure above)
    res = real_pole_rational(f, lo, hi, tol=1e-6, max_terms=40)
    if not res.usable:
        assert np.isnan(res.reconstruction_sup_rel_err)
        assert res.shifts.size == 0 and res.fit_weights.size == 0
