"""`scripts/metr_regime.py` checked against `src/ccd`, with no experiment driver in the import graph.

`metr_regime` defines the loss weight and the FiLM conditioning feature for a colored arm whose data
covariance `Σ_0` need not commute with the operator `S` (METR-LA's graph arm). Its own test file
reaches them through the paper's drivers. This one needs only `metr_marginals`, `metr_regime` and
`ccd`, and checks the two properties the definitions are justified by:

1. **The commuting reduction.** With `Σ_0 = S − τI` they are the band means of `ccd`'s own per-band
   scalars, `ccd.forward.bands.c_out_sq_fn` and `ccd.forward.rate.snr`; only the points averaged
   over changed (the spectrum's bands, not a uniform λ-grid).
2. **The fast feature is the dense one.** `BandSNR` evaluates `ν(F)` in `O(d)` from a diagonal it
   precomputes; `cond_feature` is the dense reference built from the forward's matrices. They are
   compared on a `Σ_0` that does not commute with `S`, where only the definition, not a shortcut,
   can make them agree.

Each check is paired with the wrong quantity it must reject: the uniform-grid weight, and the ratio
of traces `tr(m Σ_0 m)/tr(Σ_F)`, which was once shipped in place of the mean of the ratios.

Run: conda run -n torch_env python -m pytest tests/test_metr_regime_reductions.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import metr_regime as reg  # noqa: E402
from metr_marginals import MisspecifiedColoredForward  # noqa: E402

from ccd.denoiser import edm_weight as grid_edm_weight  # noqa: E402
from ccd.forward import ShiftedPowerRate, snr  # noqa: E402
from ccd.forward.bands import c_out_sq_fn  # noqa: E402
from ccd.operators import CovarianceOperator  # noqa: E402
from ccd.operators.backends import SpectralBackend  # noqa: E402

D = 8
TAU = 0.01
F_WINDOW = (0.02, 60.0)
LEVELS = (0.02, 0.3, 2.0, 15.0, 60.0)


def _orthogonal(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64, generator=g))
    return q


def _forward(sigma0: str) -> MisspecifiedColoredForward:
    """`S = U diag(λ) Uᵀ` with a spread spectrum (κ = 400), and `Σ_0` either `S − τI` (commuting)
    or a covariance with its own eigenbasis (not commuting)."""
    U = _orthogonal(0)
    lam = torch.logspace(math.log10(20.0), math.log10(0.05), D, dtype=torch.float64)
    op = CovarianceOperator(U * ((lam - TAU) * D).sqrt(), tau=TAU)       # X Xᵀ/D + τI = S
    if sigma0 == "commuting":
        s0 = (U * (lam - TAU)) @ U.T
    else:
        Q = _orthogonal(1)
        s0 = (Q * torch.logspace(1.0, -1.5, D, dtype=torch.float64)) @ Q.T
    phi = ShiftedPowerRate(-1.0, float(lam.min()), float(lam.max()), R=20.0)
    backend = SpectralBackend(lam, U, validate_against=op, rtol=1e-12)
    return MisspecifiedColoredForward(op, float(lam.min()), float(lam.max()), phi=phi, sigma0=s0,
                                      evals=lam, evecs=U, backend=backend)


def _ratio_of_traces(fwd, F: float) -> float:
    b = fwd._blocks(F)
    m, sig = b["m"].double(), b["sigma"].double()
    return float(torch.trace(m @ fwd.sigma0 @ m) / torch.trace(sig))


@pytest.mark.parametrize("F", LEVELS)
def test_with_a_commuting_sigma0_both_definitions_are_band_means_of_ccds_own_scalars(F):
    fwd = _forward("commuting")
    lam = fwd.evals
    blocks = fwd._blocks(F)
    want_w = 1.0 / float(c_out_sq_fn(fwd.phi, TAU, F)(lam).mean())
    want_snr = float(snr(lam, F, fwd.phi, TAU).mean())
    assert reg.edm_weight(blocks) == pytest.approx(want_w, rel=1e-12)
    assert reg.snr_bands(blocks, fwd.sigma0, fwd.evecs) == pytest.approx(want_snr, rel=1e-12)
    # Teeth: the uniform-grid weight and the ratio of traces are other quantities on this spectrum.
    assert grid_edm_weight(fwd, F) != pytest.approx(want_w, rel=1e-3)
    assert _ratio_of_traces(fwd, F) != pytest.approx(want_snr, rel=1e-3)


@pytest.mark.parametrize("F", LEVELS)
def test_the_fast_feature_is_the_dense_reference_on_a_non_commuting_sigma0(F):
    fwd = _forward("non-commuting")
    assert torch.linalg.norm(fwd.sigma0 @ fwd.op.matvec(torch.eye(D, dtype=torch.float64))
                             - fwd.op.matvec(fwd.sigma0)) > 1.0      # really does not commute
    fast = reg.BandSNR.from_forward(fwd, f_window=F_WINDOW)
    dense = reg.cond_feature(fwd._blocks(F), fwd.sigma0, fwd.evecs)
    assert float(fast.feature(F)) == pytest.approx(float(dense), rel=1e-12, abs=1e-12)
    # Teeth: the same reduction over the wrong directions (Σ_0's own eigenbasis rather than S's),
    # and the ratio of traces, both miss the dense value.
    _, Q = torch.linalg.eigh(fwd.sigma0)
    wrong_basis = reg.BandSNR(fwd.evals, Q, fwd.sigma0, fwd.phi, f_window=F_WINDOW)
    assert float(wrong_basis.feature(F)) != pytest.approx(float(dense), rel=1e-3)
    assert math.log(_ratio_of_traces(fwd, F)) != pytest.approx(float(dense), rel=1e-3)


def test_a_non_positive_mean_posterior_variance_is_refused_not_floored():
    fwd = _forward("non-commuting")
    blocks = dict(fwd._blocks(1.0))
    assert reg.edm_weight(blocks) > 0.0                                  # control: a real posterior
    blocks["c_out_sq"] = -torch.eye(D, dtype=torch.float64)
    with pytest.raises(ValueError, match="not positive"):
        reg.edm_weight(blocks)


def test_the_fast_feature_refuses_a_level_outside_the_window_it_was_proved_on():
    fast = reg.BandSNR.from_forward(_forward("non-commuting"), f_window=F_WINDOW)
    fast.feature(F_WINDOW[1])                                            # control: the edge is in
    with pytest.raises(ValueError, match="outside the window"):
        fast.feature(F_WINDOW[1] * 1.01)
