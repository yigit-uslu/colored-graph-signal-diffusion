"""Oracle tests for the steep marginal inverses ``V_t^{-1}``, ``V_t^{-1/2}`` (Phase I4, Stage 2).

``V_t = v_F(S)`` with ``v_F(λ) = λ − τ e^{−Fφ(λ)}``, so ``V_t^{-1} = (1/v_F)(S)`` and
``V_t^{-1/2} = (v_F^{-1/2})(S)`` — the dense baseline is just :func:`dense_apply` of those scalar
functions. Gated on a float64 operator (so the matrix-free solve agrees with the eigendecomposition
baseline tightly), with a float32 realistic-precision path checked separately. The ``F → 0`` /
rank-deficient singular regime is deferred and must raise.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import ColoredForward, phi_power
from ccd.operators import apply_fun, chebyshev_degree, make_synthetic_operator, n_poles_for, spectrum_bracket
from _oracles import dense_apply, rel_err


def _setup(d=48, n=300, tau=0.05, seed=0, dtype=torch.float64):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed + 2), dtype=dtype)
    return op, lo, hi, v


def _vfn(op, a, F):
    return lambda lam: lam - op.tau * torch.exp(-F * lam ** a)


# --------------------------------------------------------------- V_t^{-1/2} and V_t^{-1} vs dense

@pytest.mark.parametrize("a", [1.0, 0.5, 0.0, -1.0])
@pytest.mark.parametrize("F", [2.0, 0.5])
def test_v_inverse_half_matches_dense(a, F):
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a))
    y, info = fwd.v_inverse_half(v, F, tol=1e-8, maxiter=6000)
    assert info["converged"]
    exact = dense_apply(lambda lam: _vfn(op, a, F)(lam) ** -0.5, op, v)
    assert rel_err(y, exact) < 1e-5


@pytest.mark.parametrize("a", [1.0, 0.0, -1.0])
def test_v_inverse_matches_dense(a):
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a))
    F = 1.0
    x, info = fwd.v_inverse(v, F, tol=1e-8, maxiter=6000)
    assert info["converged"]
    exact = dense_apply(lambda lam: 1.0 / _vfn(op, a, F)(lam), op, v)
    assert rel_err(x, exact) < 1e-5


def test_v_inverse_square_route_matches_cg():
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 1.5
    x_cg, _ = fwd.v_inverse(v, F, route="cg", tol=1e-8, maxiter=6000)
    x_sq, info = fwd.v_inverse(v, F, route="square", tol=1e-8, maxiter=6000)
    assert info["converged"]
    assert rel_err(x_sq, x_cg) < 1e-5


# --------------------------------------------------------------- composition identities

def test_composition_identities():
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 1.0
    # V^{-1/2} ∘ V^{-1/2} == V^{-1}
    h1, _ = fwd.v_inverse_half(v, F, tol=1e-9, maxiter=8000)
    h2, _ = fwd.v_inverse_half(h1, F, tol=1e-9, maxiter=8000)
    xinv, _ = fwd.v_inverse(v, F, tol=1e-9, maxiter=8000)
    assert rel_err(h2, xinv) < 1e-4
    # V^{1/2} ∘ V^{-1/2} == I  (V^{1/2} = apply_fun of sqrt(v_F))
    vsqrt = apply_fun(lambda lam: torch.sqrt(fwd._vmarg_fn(F)(lam)), op, h1, lo, hi)
    assert rel_err(vsqrt, v) < 1e-3          # V^{1/2} uses adaptive-degree apply_fun (float floor)
    # V ∘ V^{-1} == I
    assert rel_err(fwd.v_marginal(xinv, F), v) < 1e-3


# --------------------------------------------------------------- ties to the trusted Chebyshev path

def test_mild_regime_matches_chebyshev_invsqrt():
    # Where 1/sqrt(v_F) is still Chebyshev-resolvable (moderate F), the rational+CG path agrees with it.
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 4.0
    y, _ = fwd.v_inverse_half(v, F, tol=1e-9, maxiter=6000)
    cheb = apply_fun(lambda lam: fwd._vmarg_fn(F)(lam) ** -0.5, op, v, lo, hi, K=256)
    assert rel_err(y, cheb) < 1e-3


def test_steep_regime_error_flat_while_chebyshev_degree_blows_up():
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    errs, npoles, cheb_deg = [], [], []
    for F in (2.0, 0.5, 0.1):
        v_min, v_max = fwd._v_bracket(F)
        y, _ = fwd.v_inverse_half(v, F, tol=1e-8, maxiter=12000)
        f = lambda lam, F=F: _vfn(op, 1.0, F)(lam) ** -0.5
        errs.append(rel_err(y, dense_apply(f, op, v)))
        npoles.append(n_poles_for(v_min, v_max, 1e-6))
        cheb_deg.append(chebyshev_degree(f, lo, hi, tol=1e-6, k_max=6000, dtype=torch.float64))
    assert max(errs) < 1e-4                    # rational error stays FLAT as F shrinks — the whole point
    assert max(npoles) < 40                    # pole count grows only ~log(kappa)
    assert cheb_deg[-1] > 5 * npoles[-1]       # while a direct-Chebyshev 1/sqrt(v_F) degree blows up


# --------------------------------------------------------------- batched RHS / float32 / guards

def test_v_inverse_half_batched():
    op, lo, hi, _ = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(0.5))
    V = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    F = 1.0
    Y, info = fwd.v_inverse_half(V, F, tol=1e-8, maxiter=6000)
    assert bool(info["converged"].all())
    exact = dense_apply(lambda lam: _vfn(op, 0.5, F)(lam) ** -0.5, op, V)
    assert rel_err(Y, exact) < 1e-5


def test_v_inverse_float32_realistic_path():
    op, lo, hi, v = _setup(dtype=torch.float32)
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 2.0
    y, info = fwd.v_inverse_half(v, F, tol=1e-4, maxiter=3000)
    assert info["converged"]
    exact = dense_apply(lambda lam: _vfn(op, 1.0, F)(lam) ** -0.5, op, v)
    assert rel_err(y, exact) < 1e-3            # float32 apply floor, matching the marginals suite


def test_rejects_nonpositive_F():
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    with pytest.raises(ValueError):
        fwd.v_inverse_half(v, 0.0)
    with pytest.raises(ValueError):
        fwd.v_inverse(v, -1.0)
    with pytest.raises(ValueError):
        fwd.v_inverse(v, 1.0, route="bogus")


def test_rejects_non_finite_colouring():
    # A colouring whose v_F is non-finite on part of [lo,hi] must RAISE, not silently seed a NaN /
    # non-SPD solve (nan <= 0 is False, so the positivity check alone would let it through).
    op, lo, hi, v = _setup()
    bad = ColoredForward(op, lo, hi, phi=lambda lam: torch.log(lam - 0.5))   # NaN where lam < 0.5
    with pytest.raises(ValueError):
        bad.v_inverse_half(v, 1.0)
    with pytest.raises(ValueError):
        bad.v_inverse(v, 1.0)
