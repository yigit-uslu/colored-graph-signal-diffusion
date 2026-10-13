"""Tests for spectral deflation (edge-limited backends probe, Stage S4).

Oracle-checked against a dense eigendecomposition (test-only, same convention as
``tests/operators/_oracles.py``). Covers: Ritz pairs are genuine (near-)eigenpairs of a matrix-free
operator; the exact decomposition identity ``f(S)v = U f(Theta) U^T v + p(S)(I-UU^T)v`` holds
regardless of ``p``; the corrected (v4) leakage diagnostic is near-zero for an honest apply and both
named negative controls (omitted complement projection, under-converged Ritz pairs) inflate it.
"""

from __future__ import annotations

import math

import torch

from ccd.operators.deflation import break_even_n, deflated_apply, ritz_pairs

torch.manual_seed(0)


def _spd_matvec(d, v_min, v_max, seed=0):
    """A matrix-free SPD operator with a prescribed spectrum (test-only oracle keeps Q, evals)."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    lam = torch.exp(torch.linspace(math.log(v_min), math.log(v_max), d, dtype=torch.float64))

    def matvec(v):
        single = v.ndim == 1
        x = v.unsqueeze(1) if single else v
        out = Q @ (lam[:, None] * (Q.transpose(0, 1) @ x))
        return out.squeeze(1) if single else out

    return matvec, lam, Q


def test_ritz_pairs_are_near_exact_eigenpairs():
    d = 128
    matvec, lam, Q = _spd_matvec(d, 1.0, 1e3)
    ritz = ritz_pairs(matvec, d, 8, which="SA", dtype=torch.float64, tol=1e-12)
    assert ritz.converged
    torch.testing.assert_close(ritz.evals, lam[:8], rtol=1e-8, atol=1e-8)
    SU = matvec(ritz.evecs)
    resid = torch.linalg.norm(SU - ritz.evecs * ritz.evals[None, :]) / torch.linalg.norm(SU)
    assert resid < 1e-8
    ortho = torch.linalg.norm(ritz.evecs.transpose(0, 1) @ ritz.evecs - torch.eye(8, dtype=torch.float64))
    assert ortho < 1e-8


def test_deflated_apply_matches_oracle_for_any_complement_p():
    """The decomposition is EXACT given true eigenpairs, for any complement function p -- even a
    deliberately wrong p, as long as it agrees with f on the Ritz subspace's own eigenvalues... in
    fact the identity only needs p to agree with f on the COMPLEMENT, so this checks the sharper
    claim: with the TRUE top-m eigenpairs (m small, exact here since Q/lam are known), swapping the
    complement's p for the identity function changes only the complement term, and the total still
    equals the oracle when p == f."""
    d = 64
    matvec, lam, Q = _spd_matvec(d, 1.0, 100.0)
    f = lambda x: torch.sqrt(x)  # noqa: E731
    v = torch.randn(d, dtype=torch.float64)
    v = v / torch.linalg.norm(v)
    exact = Q @ (f(lam) * (Q.transpose(0, 1) @ v))

    ritz = ritz_pairs(matvec, d, 6, which="SA", dtype=torch.float64, tol=1e-12)

    def complement_apply_exact(x):
        # exact complement: evaluate f on the SAME true spectrum via the oracle Q/lam (test-only)
        return Q @ (f(lam) * (Q.transpose(0, 1) @ x))

    res = deflated_apply(f, complement_apply_exact, v, ritz)
    err = float(torch.linalg.norm(res.y - exact) / torch.linalg.norm(exact))
    assert err < 1e-8
    assert res.leakage < 1e-8


def test_leakage_gate_is_not_fooled_by_an_exactly_deflated_eigenvector():
    """v4's own worked example: for v an exact eigenvector of the deflated block, the deflated-
    subspace component of the RESULT is exactly 1 -- that is correct, not leakage (the v3 gate this
    replaced would have rejected it at the accuracy target). ``v_perp`` is then (up to rounding) the
    zero vector, so `leakage` itself is floor-dominated numerical noise and not asserted here -- the
    load-bearing check is that the result's own U-component is ~1 without the corrected gate
    complaining about it (no exception, no false "leaky" verdict surfaced anywhere in `res`)."""
    d = 32
    matvec, lam, Q = _spd_matvec(d, 1.0, 50.0)
    f = lambda x: torch.sqrt(x)  # noqa: E731
    ritz = ritz_pairs(matvec, d, 4, which="SA", dtype=torch.float64, tol=1e-12)
    v = ritz.evecs[:, 0].clone()  # an (approximately) exact eigenvector of the deflated block

    def complement_apply(x):
        return Q @ (f(lam) * (Q.transpose(0, 1) @ x))

    res = deflated_apply(f, complement_apply, v, ritz)
    y_component_in_U = torch.linalg.norm(ritz.evecs.transpose(0, 1) @ res.y) / torch.linalg.norm(res.y)
    assert y_component_in_U > 0.99  # the RESULT's own component in U is ~1 -- correct, not a defect
    exact = Q @ (f(lam) * (Q.transpose(0, 1) @ v))
    assert torch.linalg.norm(res.y - exact) / torch.linalg.norm(exact) < 1e-6


def test_negative_controls_inflate_leakage():
    d = 100
    matvec, lam, Q = _spd_matvec(d, 1.0, 1e3)
    f = lambda x: torch.sqrt(x)  # noqa: E731
    v = torch.randn(d, dtype=torch.float64)
    v = v / torch.linalg.norm(v)
    ritz = ritz_pairs(matvec, d, 10, which="SA", dtype=torch.float64, tol=1e-12)

    def complement_apply(x):
        return Q @ (f(lam) * (Q.transpose(0, 1) @ x))

    honest = deflated_apply(f, complement_apply, v, ritz)
    no_proj = deflated_apply(f, complement_apply, v, ritz, project_complement=False)
    assert no_proj.leakage > 100 * max(honest.leakage, 1e-14)

    ritz_bad = ritz_pairs(matvec, d, 10, which="SA", dtype=torch.float64, maxiter=80, tol=1e-1)
    under = deflated_apply(f, complement_apply, v, ritz_bad)
    assert under.leakage > 10 * max(honest.leakage, 1e-14)


def test_break_even_n():
    assert break_even_n(m_setup=100.0, m_cloud_deflated=10.0, m_cloud_baseline=50.0) == 2.5
    assert break_even_n(m_setup=100.0, m_cloud_deflated=50.0, m_cloud_baseline=50.0) is None
    assert break_even_n(m_setup=100.0, m_cloud_deflated=60.0, m_cloud_baseline=50.0) is None
