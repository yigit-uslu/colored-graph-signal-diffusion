"""Phase I3c — forward∘reverse round-trip (the integration test).

I0–I2 (forward) and I3a/I3b-1 (reverse) were each verified against closed forms *separately*; this
composes the **actual** forward operators with the **actual** reverse sampler to confirm the two
subsystems are mutual inverses. Two complementary round-trips:

1. **Deterministic pathwise identity** — the PF-ODE is an invertible flow, so encoding ``x`` forward
   ``F_lo→F_hi`` (variance-growing) then decoding back ``F_hi→F_lo`` recovers ``x`` exactly. Kept at
   ``F_lo>0``: reaching the data end ``F=0`` on the *encode* side needs ``1/√v_0`` (the deferred
   steep-inverse regime); the decode side is smooth.
2. **Stochastic distributional round-trip** — draw data ``x_0 ~ N(0, Ĉ_n)``, corrupt it with the real
   :func:`ccd.forward.forward_draw`, run the reverse SDE back to ``F=0``, and confirm the data
   covariance ``Ĉ_n`` is restored (distribution, not pathwise — the noise is fresh).

float64; ``rel_err`` gates.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import ColoredForward, forward_draw, phi_power, whiten
from ccd.operators import apply_fun, make_synthetic_operator, spectrum_bracket
from ccd.reverse import GaussianReversePFODE, GaussianReverseSDE, reverse_f_grid
from _oracles import rel_err

K_FINE = 160
COV_TOL = 0.06
FAMILY = [-1.0, 0.0, 0.5, 1.0]


def _setup(d=48, n=400, tau=0.05, seed=0):
    # n > d ⇒ Ĉ_n full rank ⇒ v_0 = λ−τ > 0 on the spectrum (smooth regime away from F=0's edge).
    g = torch.Generator().manual_seed(seed)
    op = make_synthetic_operator(n, d, tau=tau, decay=1.0, generator=g, dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    return op, lo, hi


# ------------------------------------------- 1. deterministic PF-ODE encode/decode identity
# Each member's active F*-window (F*=ln2·λ^{−a} over the bracket) differs sharply, so the F-range
# is chosen per a to genuinely exercise the encode — while keeping F_lo off the data end for a>0,
# where the encode factor √(v_{F_hi}/v_{F_lo}) develops a steep 1/√v_0 boundary layer near λ=τ.
@pytest.mark.parametrize("a, F_lo, F_hi", [
    (-1.0, 0.05, 5.0),   # coarse→fine: clocks sit low (F*=ln2·λ ≲ 0.8), exercise the small-F window
    (0.0, 0.5, 5.0),
    (0.5, 0.5, 5.0),
    (1.0, 0.5, 5.0),     # fine→coarse: F_lo kept high (steep encode boundary layer near the data end)
])
def test_pfode_encode_decode_is_pathwise_identity(a, F_lo, F_hi):
    op, lo, hi = _setup()
    rev = GaussianReversePFODE(op, lo, hi, phi=phi_power(a), K=K_FINE)
    x = torch.randn(op.d, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    # encode = forward PF-ODE flow F_lo→F_hi. transport_fn(F_a,F_b) is the exact flow factor
    # √(v_{F_b}/v_{F_a}) for any ordering; only step() enforces the reverse (F-decreasing) direction.
    x_hi = apply_fun(rev.transport_fn(F_lo, F_hi), op, x, lo, hi, K_FINE)
    assert rel_err(x_hi, x) > 0.1                       # non-vacuity: the encode actually moved x
    x_rt = rev.step(x_hi, F_hi, F_lo, scheme="exact")   # decode (reverse)
    assert rel_err(x_rt, x) < 1e-4                      # forward∘reverse = identity (pathwise)


# ----------------------------- 1b. steep-inverse encode reaches the data end (Phase I4, Stage 3)
# For a>0 the pathwise test above keeps F_lo off the data end (F_lo=0.5) because the COMBINED Chebyshev
# encode factor √(v_{F_hi}/v_{F_lo}) develops a steep 1/√v_0 boundary layer near λ=τ. Splitting it —
# V_{F_lo}^{-1/2} via the rational solver (whiten), then V_{F_hi}^{1/2} via smooth Chebyshev — is the
# SAME operator but stays accurate as F_lo→0, pushing the reachable data end ~10× closer.
def test_pfode_encode_via_steep_inverse_reaches_data_end():
    a, F_lo, F_hi = 1.0, 0.05, 5.0
    op, lo, hi = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a))
    rev = GaussianReversePFODE(op, lo, hi, phi=phi_power(a), K=K_FINE)
    x = torch.randn(op.d, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    v_hi_sqrt = lambda lam: torch.sqrt(torch.clamp(lam - op.tau * torch.exp(-F_hi * lam ** a), min=0.0))

    # split steep-inverse encode: V_{F_lo}^{-1/2} (rational solver) then V_{F_hi}^{1/2} (smooth Chebyshev)
    whitened = whiten(fwd, x, F_lo, tol=1e-10, maxiter=20000)
    x_hi = apply_fun(v_hi_sqrt, op, whitened, lo, hi, K_FINE)
    assert rel_err(x_hi, x) > 0.1                        # non-vacuity: the encode actually moved x
    err_split = rel_err(rev.step(x_hi, F_hi, F_lo, scheme="exact"), x)
    assert err_split < 1e-4                              # forward∘reverse = identity, near the data end

    # contrast: the combined-Chebyshev encode at the SAME F_lo fails the identity (the boundary layer)
    x_hi_cheb = apply_fun(rev.transport_fn(F_lo, F_hi), op, x, lo, hi, K_FINE)
    err_cheb = rel_err(rev.step(x_hi_cheb, F_hi, F_lo, scheme="exact"), x)
    assert err_cheb > 10 * err_split                     # the steep inverse is materially more accurate


# --------------------------------------- 2. stochastic pipeline round-trip (data→corrupt→restore)
@pytest.mark.slow
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_stochastic_pipeline_round_trip(a):
    op, lo, hi = _setup(d=16, n=200, seed=7)
    phi = phi_power(a)
    fwd = ColoredForward(op, lo, hi, phi=phi)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F, N = 5.0, 40_000
    gen = torch.Generator().manual_seed(0)

    # data draw x_0 ~ N(0, Ĉ_n) via the data factor (1/√n) X z  (Cov = (1/n) X Xᵀ = Ĉ_n exactly)
    z_n = torch.randn(op.n, N, generator=gen, dtype=torch.float64)
    eps = torch.randn(op.d, N, generator=gen, dtype=torch.float64)
    x0 = (op.X @ z_n) / (op.n ** 0.5)

    # forward corruption with the REAL forward_draw, then the reverse SDE back to the data end
    xF = forward_draw(fwd, x0, eps, F)
    x_hat0 = rev.integrate(xF, reverse_f_grid(F, 0.0, 8), scheme="exact",
                           generator=torch.Generator().manual_seed(1))

    c_hat = op._dense() - op.tau * torch.eye(op.d, dtype=op.dtype)      # Ĉ_n
    cov_F = (xF @ xF.transpose(-2, -1)) / N
    cov_rt = (x_hat0 @ x_hat0.transpose(-2, -1)) / N
    assert rel_err(cov_F, c_hat) > COV_TOL         # the forward genuinely corrupted the data (→ V_F≈S)
    assert rel_err(cov_rt, c_hat) < COV_TOL        # the reverse restored the data covariance Ĉ_n
