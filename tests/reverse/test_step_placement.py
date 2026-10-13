"""Phase I3b-2 — reverse step-placement grids.

Placement matters only for the **approximate** schemes; the exact schemes telescope and are
grid-independent. The I3b-2 study found the generation-order (``F*``-quantile) grid to be a poor,
``a=0``-degenerate choice, so ``src`` offers only the matrix-free ``uniform_logsnr`` / ``stiffness``
/ ``equidistribute`` placements. Gated here:

1. each new spacing is a valid strictly-decreasing grid with distinct nodes for every family member
   (in particular **not** degenerate at ``a=0``, unlike ``F*``-quantile);
2. the exact PF-ODE scheme is placement-invariant (placement is a no-op for it);
3. ``equidistribute`` beats ``uniform`` on the stiff ``a=−1`` member (the study's headline win);
4. validation: unknown/spectrum-based spacings and ``F_min≤0`` are rejected.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import phi_power
from ccd.operators import make_synthetic_operator, spectrum_bracket
from ccd.reverse import GaussianReversePFODE, reverse_f_grid
from _oracles import dense_apply, rel_err

K_FINE = 160
NEW_SPACINGS = ["uniform_logsnr", "stiffness", "equidistribute"]
FAMILY = [-1.0, 0.0, 0.5, 1.0]


def _setup(d=48, n=400, tau=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    op = make_synthetic_operator(n, d, tau=tau, decay=1.0, generator=g, dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    return op, lo, hi


def _grid(spacing, a, op, lo, hi, n=16):
    return reverse_f_grid(6.0, 0.05, n, spacing=spacing, phi=phi_power(a), tau=op.tau, lo=lo, hi=hi)


# ------------------------------------------------ 1. valid, non-degenerate grids for all a
@pytest.mark.parametrize("spacing", NEW_SPACINGS)
@pytest.mark.parametrize("a", FAMILY)
def test_new_spacing_is_valid_and_nondegenerate(spacing, a):
    op, lo, hi = _setup()
    n = 16
    g = _grid(spacing, a, op, lo, hi, n)
    assert g.shape == (n + 1,)
    assert bool((g[1:] < g[:-1]).all())          # strictly decreasing
    assert len(torch.unique(g)) == n + 1         # distinct nodes — no a=0 collapse (cf. F*-quantile)
    assert float(g[0]) == pytest.approx(6.0)
    assert float(g[-1]) == pytest.approx(0.05)


# ------------------------------------------- 2. placement is a no-op for the exact scheme
def test_exact_scheme_is_placement_invariant():
    op, lo, hi = _setup()
    rev = GaussianReversePFODE(op, lo, hi, phi=phi_power(1.0), K=K_FINE)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    ref = rev.integrate(v, reverse_f_grid(6.0, 0.05, 16, spacing="uniform"), scheme="exact")
    for spacing in NEW_SPACINGS:
        out = rev.integrate(v, _grid(spacing, 1.0, op, lo, hi), scheme="exact")
        # The residual (~1e-5) is per-step Chebyshev truncation over the steep [6, 0.05] range,
        # not a placement effect — it is ~1000× below the Euler scheme's placement sensitivity.
        assert rel_err(out, ref) < 3e-4


# ---------------------------------------- 3. equidistribute beats uniform on the stiff a=-1
def test_equidistribute_beats_uniform_for_stiff_family():
    op, lo, hi = _setup()
    a = -1.0                                     # coarse→fine: small-λ bands are stiff (φ=1/λ large)
    phi = phi_power(a)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi, K=K_FINE)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    v_a, v_b = (lambda lam: lam - op.tau * torch.exp(-6.0 * phi(lam))), \
               (lambda lam: lam - op.tau * torch.exp(-0.05 * phi(lam)))
    exact = dense_apply(lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0)), op, v)

    def euler_err(spacing):
        g = _grid(spacing, a, op, lo, hi)
        return rel_err(rev.integrate(v, g, scheme="euler"), exact)

    e_uniform = euler_err("uniform")
    e_equi = euler_err("equidistribute")
    assert e_equi < 0.5 * e_uniform              # study measured ≈9× at n=16


# ------------------------------------------------------------------ 4. validation
def test_reverse_f_grid_rejects_bad_inputs():
    op, lo, hi = _setup()
    # F*-quantile is intentionally NOT a src spacing (bench-only diagnostic).
    with pytest.raises(ValueError):
        reverse_f_grid(6.0, 0.05, 8, spacing="fstar_quantile", phi=phi_power(1.0), tau=op.tau, lo=lo, hi=hi)
    # the placement spacings need F_min > 0 (log-SNR / log-variance diverge at F=0).
    for spacing in NEW_SPACINGS:
        with pytest.raises(ValueError):
            reverse_f_grid(6.0, 0.0, 8, spacing=spacing, phi=phi_power(1.0), tau=op.tau, lo=lo, hi=hi)
    # and the required parameters must be supplied.
    with pytest.raises(ValueError):
        reverse_f_grid(6.0, 0.05, 8, spacing="stiffness")          # missing phi/tau/lo
