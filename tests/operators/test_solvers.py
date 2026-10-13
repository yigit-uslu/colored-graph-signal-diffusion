"""Stage-1 oracle tests for the matrix-free steep-inverse primitives.

The solvers (:func:`cg`, :func:`shifted_cg`) and the rational pole rule
(:func:`invsqrt_poles`) are ``S``-agnostic — they take a plain matvec / a scalar interval — so
they are gated here in isolation against dense SPD baselines (float64), with no ``v_F`` wiring.
"""

from __future__ import annotations

import math
import pathlib
import re

import pytest
import torch

from ccd.operators import cg, invsqrt_poles, n_poles_for, shifted_cg
from _oracles import dense_solve, rel_err

torch.manual_seed(0)


def _spd(d, v_min, v_max, seed=0):
    """Dense SPD matrix with a log-spaced spectrum in ``[v_min, v_max]`` (float64)."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    lam = torch.exp(torch.linspace(math.log(v_min), math.log(v_max), d, dtype=torch.float64))
    return (Q * lam) @ Q.transpose(-2, -1)


class _Counter:
    """Wrap a dense matvec and count its calls (the matrix-free cost budget)."""

    def __init__(self, A):
        self.A, self.n = A, 0

    def __call__(self, x):
        self.n += 1
        return self.A @ x


def _vec(d, seed=1, batch=None):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(d, generator=g, dtype=torch.float64) if batch is None
            else torch.randn(d, batch, generator=g, dtype=torch.float64))


# --------------------------------------------------------------- rational pole rule (scalar)

@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e8])
def test_zolotarev_scalar_accuracy(kappa):
    v_min, v_max = 1.0, kappa
    N = n_poles_for(v_min, v_max, 1e-7)
    sigma, w = invsqrt_poles(v_min, v_max, N, method="zolotarev")
    assert bool((sigma > 0).all()) and bool((w > 0).all())     # every shifted system stays SPD
    y = torch.exp(torch.linspace(math.log(v_min), math.log(v_max), 2000, dtype=torch.float64))
    approx = (w[:, None] / (y[None, :] + sigma[:, None])).sum(0)
    assert float(((approx - y**-0.5).abs() / y**-0.5).max()) < 1e-6


@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e6, 1e8])
def test_gauss_fallback_scalar_accuracy(kappa):
    # The elementary (no-SciPy) rule. Strict positivity of sigma,w (what keeps every shifted system
    # SPD) must hold across the tail; accuracy is checked into the steep regime it warns about.
    v_min, v_max = 1.0, kappa
    N = n_poles_for(v_min, v_max, 1e-7, method="gauss")
    sigma, w = invsqrt_poles(v_min, v_max, N, method="gauss")
    assert bool((sigma > 0).all()) and bool((w > 0).all())
    y = torch.exp(torch.linspace(math.log(v_min), math.log(v_max), 2000, dtype=torch.float64))
    approx = (w[:, None] / (y[None, :] + sigma[:, None])).sum(0)
    assert float(((approx - y**-0.5).abs() / y**-0.5).max()) < 1e-6


def test_n_poles_grows_logarithmically():
    # log(kappa) growth: a 6-decade jump in kappa (log-kappa 4x) grows the pole count ~4x, NOT like
    # a Chebyshev degree ~sqrt(kappa) which would be ~1e4.
    n_lo = n_poles_for(1.0, 1e2, 1e-6)
    n_hi = n_poles_for(1.0, 1e8, 1e-6)
    assert n_hi > n_lo
    assert n_hi < 5 * n_lo and n_hi < 40                        # stays double-digit, far below sqrt(kappa)


def test_invsqrt_poles_rejects_bad_interval():
    with pytest.raises(ValueError):
        invsqrt_poles(0.0, 1.0, 8)
    with pytest.raises(ValueError):
        invsqrt_poles(2.0, 1.0, 8)


# --------------------------------------------------------------- CG for A^{-1} b

def test_cg_matches_dense_inverse():
    A = _spd(48, 0.01, 100.0)                                   # kappa = 1e4
    b = _vec(A.shape[0])
    x, info = cg(_Counter(A), b, tol=1e-10, maxiter=2000)
    assert info["converged"]
    assert rel_err(x, dense_solve(A, b, power=-1.0)) < 1e-7


def test_cg_with_shift():
    A = _spd(48, 0.01, 100.0)
    b = _vec(A.shape[0])
    x, info = cg(_Counter(A), b, shift=2.0, tol=1e-10, maxiter=2000)
    exact = dense_solve(A + 2.0 * torch.eye(A.shape[0], dtype=torch.float64), b, power=-1.0)
    assert info["converged"] and rel_err(x, exact) < 1e-7


def test_cg_batched_matches_percolumn():
    A = _spd(40, 0.02, 50.0)
    B = _vec(A.shape[0], batch=4)
    X, info = cg(_Counter(A), B, tol=1e-10, maxiter=2000)
    assert bool(info["converged"].all())
    for j in range(B.shape[1]):
        xj, _ = cg(_Counter(A), B[:, j], tol=1e-10, maxiter=2000)
        assert rel_err(X[:, j], xj) < 1e-6
    assert rel_err(X, dense_solve(A, B, power=-1.0)) < 1e-7


def test_cg_reports_nonconvergence():
    A = _spd(48, 1e-6, 100.0)                                   # kappa = 1e8, stiff
    b = _vec(A.shape[0])
    x, info = cg(_Counter(A), b, tol=1e-12, maxiter=3)          # far too few iters
    assert info["converged"] is False
    assert torch.isfinite(x).all()                             # no silent NaN/Inf on the truncated iterate


# --------------------------------------------------------------- multishift CG

def test_shifted_cg_per_shift_matches_dense():
    A = _spd(40, 0.02, 50.0)
    d = A.shape[0]
    b = _vec(d)
    shifts = torch.tensor([0.5, 3.0, 20.0], dtype=torch.float64)
    X, info = shifted_cg(_Counter(A), b, shifts, tol=1e-11, maxiter=2000, seed_shift=0.0)
    assert info["converged"]
    assert X.shape == (3, d)
    for j, s in enumerate(shifts):
        exact = dense_solve(A + float(s) * torch.eye(d, dtype=torch.float64), b, power=-1.0)
        assert rel_err(X[j], exact) < 1e-6


@pytest.mark.parametrize("kappa", [1e4, 1e6])
def test_shifted_cg_weighted_is_inverse_sqrt(kappa):
    A = _spd(48, 1.0, kappa)
    d = A.shape[0]
    b = _vec(d)
    N = n_poles_for(1.0, kappa, 1e-8)
    sigma, w = invsqrt_poles(1.0, kappa, N)
    y, info = shifted_cg(_Counter(A), b, sigma, w, tol=1e-11, maxiter=8000)
    assert info["converged"]
    assert rel_err(y, dense_solve(A, b, power=-0.5)) < 1e-5


def test_shifted_cg_square_reproduces_inverse():
    # (A^{-1/2})^2 b == A^{-1} b : apply the weighted invsqrt sum twice.
    A = _spd(40, 0.05, 50.0)
    d = A.shape[0]
    b = _vec(d)
    N = n_poles_for(0.05, 50.0, 1e-9)
    sigma, w = invsqrt_poles(0.05, 50.0, N)
    mid, _ = shifted_cg(_Counter(A), b, sigma, w, tol=1e-11, maxiter=4000)
    y, _ = shifted_cg(_Counter(A), mid, sigma, w, tol=1e-11, maxiter=4000)
    assert rel_err(y, dense_solve(A, b, power=-1.0)) < 1e-5


def test_shifted_cg_matvec_count_independent_of_n_poles():
    # One shared Krylov space: adding poles adds only scalar recurrences, not matvecs.
    A = _spd(48, 1.0, 1e4)
    b = _vec(A.shape[0])
    # (a) pinned seed_shift=0 -> the seed operator is exactly A for every N: identical matvec counts.
    pinned = []
    for N in (8, 20, 40):
        sigma, w = invsqrt_poles(1.0, 1e4, N)
        counter = _Counter(A)
        shifted_cg(counter, b, sigma, w, tol=1e-10, maxiter=4000, seed_shift=0.0)
        pinned.append(counter.n)
    assert pinned[0] == pinned[1] == pinned[2]
    # (b) the SHIPPED default seed_shift=min(shifts): min(sigma) drifts with N, so the seed
    #     conditioning moves the count by only a small additive constant — O(1), not O(N) in the poles.
    default = []
    for N in (8, 20, 40):
        sigma, w = invsqrt_poles(1.0, 1e4, N)
        counter = _Counter(A)
        shifted_cg(counter, b, sigma, w, tol=1e-10, maxiter=4000)
        default.append(counter.n)
    assert max(default) - min(default) <= 12
    assert max(default) < 4 * min(default)


def test_shifted_cg_batched():
    A = _spd(40, 1.0, 1e4)
    d = A.shape[0]
    B = _vec(d, batch=3)
    N = n_poles_for(1.0, 1e4, 1e-8)
    sigma, w = invsqrt_poles(1.0, 1e4, N)
    Y, info = shifted_cg(_Counter(A), B, sigma, w, tol=1e-11, maxiter=6000)
    assert bool(info["converged"].all())
    for j in range(B.shape[1]):
        yj, _ = shifted_cg(_Counter(A), B[:, j], sigma, w, tol=1e-11, maxiter=6000)
        assert rel_err(Y[:, j], yj) < 1e-6


# --------------------------------------------------------------- preconditioner / warm start (cg)

def test_cg_with_preconditioner():
    A = _spd(48, 0.01, 100.0)
    b = _vec(A.shape[0])
    # A PERFECT preconditioner M⁻¹ = A⁻¹ makes PCG converge in one step — proves precond is actually
    # applied (an ignored or inverted preconditioner could not do this).
    perfect = lambda r: dense_solve(A, r, power=-1.0)
    x, info = cg(_Counter(A), b, precond=perfect, tol=1e-8, maxiter=50)
    assert info["converged"] and info["iters"] == 1
    assert rel_err(x, dense_solve(A, b, power=-1.0)) < 1e-6
    _, cold = cg(_Counter(A), b, tol=1e-8, maxiter=2000)
    assert cold["iters"] > 5                                    # unpreconditioned genuinely needs many steps


def test_cg_warm_start():
    A = _spd(48, 0.02, 50.0)
    b = _vec(A.shape[0])
    exact = dense_solve(A, b, power=-1.0)
    x0 = (exact * 0.99).clone()
    x0_ref = x0.clone()
    x, warm = cg(_Counter(A), b, x0=x0, tol=1e-10, maxiter=2000)
    _, cold = cg(_Counter(A), b, tol=1e-10, maxiter=2000)
    assert rel_err(x, exact) < 1e-7
    assert warm["iters"] < cold["iters"]                       # warming near the solution saves work
    assert torch.equal(x0, x0_ref)                             # the passed x0 is not mutated (internal clone)


# --------------------------------------------------------------- honest reporting / per-column info

def test_shifted_cg_reports_nonconvergence():
    A = _spd(48, 1e-6, 100.0)                                  # kappa = 1e8, stiff
    b = _vec(A.shape[0])
    sigma, w = invsqrt_poles(1e-6, 100.0, 12)
    y, info = shifted_cg(_Counter(A), b, sigma, w, tol=1e-12, maxiter=3)
    assert info["converged"] is False                         # single-vector path returns a python bool
    assert torch.isfinite(y).all()                            # truncated iterate is finite, not NaN/Inf


def test_batched_info_is_per_column():
    A = _spd(40, 0.02, 50.0)
    B = _vec(A.shape[0], batch=3)
    _, info = cg(_Counter(A), B, tol=1e-10, maxiter=2000)
    conv, relres = info["converged"], info["relres"]
    assert torch.is_tensor(conv) and conv.dtype == torch.bool and conv.shape == (3,)
    assert torch.is_tensor(relres) and relres.shape == (3,)
    assert bool(conv.all()) and bool((relres <= 1e-10).all())  # per-column, not a single reduced value


def test_batched_nonconvergence_is_per_column():
    A = _spd(48, 1e-6, 100.0)
    B = _vec(A.shape[0], batch=3)
    _, info = cg(_Counter(A), B, tol=1e-12, maxiter=3)
    conv = info["converged"]
    assert torch.is_tensor(conv) and conv.shape == (3,)
    assert not bool(conv.all())                                # NOT `is False` — always False on a tensor


def test_shifted_cg_rejects_bad_args():
    A = _spd(20, 1.0, 10.0)
    b = _vec(20)
    shifts = torch.tensor([0.5, 3.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        shifted_cg(_Counter(A), b, shifts, seed_shift=1.0)     # seed_shift > min(shifts) -> a negative shift
    with pytest.raises(ValueError):
        shifted_cg(_Counter(A), b, shifts, weights=torch.tensor([1.0, 2.0, 3.0]))  # length mismatch


# --------------------------------------------------------------- matrix-free guard (new modules)

def test_no_eig_in_steep_inverse_modules():
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "ccd" / "operators"
    forbidden = re.compile(
        r"\beigh\b|eigvalsh|eigvals|matrix_exp|\bexpm\b|\bfunm\b|linalg\.eig\b|linalg\.solve"
        r"|linalg\.inv\b|torch\.inverse|\.svd\(|svdvals|cholesky|lu_factor|lu_solve"
    )
    for name in ("solvers.py", "rational.py"):
        assert not forbidden.search((src / name).read_text()), f"dense linear algebra leaked into {name}"
