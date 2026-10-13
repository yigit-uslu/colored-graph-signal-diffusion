"""`NormPhi`'s promotion into `src` (implementation plan step 4b).

Two properties have to hold, and they pull in opposite directions:

1. **Bit-identity with the two script-side copies** it replaces — otherwise the promotion silently
   changes every PBMC figure that used them.
2. **A ×1 no-op on an already-pinned arm** — `ShiftedPowerRate` satisfies `min_λ φ = FLOOR` by
   construction, and that equality is what makes `F_max` common across the ladder. A `NormPhi` that
   rescaled a pinned arm would hand it a different sampler budget, turning every clock contrast into
   a budget contrast.

Both are checked on the real fMRI spectrum, because that is where the `a=0` arm — one half of the
G4 headline contrast — actually consumes it.

Run: conda run -n torch_env python -m pytest tests/forward/test_norm_phi.py -v
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from torch import nn

from ccd.forward import NormPhi as NormPhiExported
from ccd.forward.rate import NormPhi, PhiLogPoly, ShiftedPowerRate, phi_power

FMRI = pathlib.Path(__file__).resolve().parents[2] / "data" / "fmri"
FLOOR, F_MIN, TAU, R = 0.077, 0.02, 0.01, 120.0


class _HistoricalNormPhi(nn.Module):
    """Verbatim body of the two script-side copies, kept ONLY as the promotion's reference.

    `viz_gen_real.py:30` and `viz_minvar_phi.py:59` differed in docstring and in nothing else; this
    is their shared implementation, frozen here so the promotion is checked against what it
    replaced rather than against itself.
    """

    def __init__(self, base, spec, floor):
        super().__init__()
        self.base = base
        self.register_buffer("spec", spec)
        self.floor = float(floor)

    def forward(self, lam):
        b = self.base(lam)
        scale = (self.floor / self.base(self.spec).min()).to(device=b.device, dtype=b.dtype)
        return b * scale

    def effective_a(self):
        return self.base.effective_a()

    @property
    def theta(self):
        return self.base.theta


@pytest.fixture(scope="module")
def spec():
    if (FMRI / "evals.npy").is_file():
        import numpy as np
        return torch.as_tensor(np.load(FMRI / "evals.npy"), dtype=torch.float64)
    pytest.skip("fMRI bundle not built")


def test_it_is_importable_from_src_at_both_paths():
    """The promotion's whole point: five docstrings in `rate.py` referred to a class not in `src`."""
    assert NormPhiExported is NormPhi
    assert NormPhi.__module__ == "ccd.forward.rate"


def test_it_reproduces_the_script_copies_bit_identically(spec):
    """Not `allclose` — bit-identical. A promotion that changes a value is a silent re-run of every
    figure that used the old one."""
    for base in (phi_power(0.0), phi_power(-1.0), phi_power(0.5)):
        new = NormPhi(base, spec, FLOOR)(spec)
        old = _HistoricalNormPhi(base, spec, FLOOR)(spec)
        assert torch.equal(new, old), f"promotion changed values for {base}"

    # the learnable member too -- `viz_minvar_phi.py`'s copy wrapped exactly this
    logspec = torch.log(spec)
    poly = PhiLogPoly(float(logspec.mean()), float(logspec.std()), degree=2, dtype=torch.float64)
    with torch.no_grad():
        poly.theta.copy_(torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64))
    assert torch.equal(NormPhi(poly, spec, FLOOR)(spec), _HistoricalNormPhi(poly, spec, FLOOR)(spec))
    assert torch.equal(NormPhi(poly, spec, FLOOR).theta, poly.theta), "the theta passthrough too"


def test_it_pins_the_minimum_to_FLOOR(spec):
    for base in (phi_power(0.0), phi_power(-1.0), phi_power(1.0)):
        out = NormPhi(base, spec, FLOOR)(spec)
        assert float(out.min()) == pytest.approx(FLOOR, rel=1e-12)


def test_it_is_a_no_op_on_every_pinned_arm(spec):
    """`ShiftedPowerRate` is already pinned; wrapping it must not move it by even a ULP-scale factor.

    This is the property that keeps `F_max = −2 ln(ε_prior)/FLOOR` common across the ladder. A
    rescale here would be invisible in any single arm and would break only the *comparison*.
    """
    lo, hi = float(spec.min()), float(spec.max())
    for nominal in (-0.5, -1.0, -2.0, -4.0):
        phi = ShiftedPowerRate(nominal, lo, hi, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)
        raw = phi(spec)
        assert float(raw.min()) == pytest.approx(FLOOR, rel=1e-12), "precondition: already pinned"
        wrapped = NormPhi(phi, spec, FLOOR)(spec)
        assert torch.allclose(wrapped, raw, rtol=1e-12, atol=0.0), f"NormPhi rescaled ā({nominal})"
        ratio = float((wrapped / raw).max())
        assert abs(ratio - 1.0) < 1e-12, f"scale {ratio!r} on nominal a={nominal}"


def test_the_a0_arm_gets_the_same_F_max_as_the_pinned_arms(spec):
    """The budget identity the no-op buys, stated in the units that matter."""
    import math

    lo, hi = float(spec.min()), float(spec.max())
    eps_prior = 0.02
    # `math.log`, NOT `torch.log(torch.tensor(...))` -- the latter defaults to float32 and reads
    # 101.61098876556792 against the protocol's 101.61098715397782, a 1.6e-9 disagreement that has
    # nothing to do with NormPhi. The first draft of this test asserted the float32 value.
    F_max = lambda phi: -2.0 * math.log(eps_prior) / float(phi(spec).min())  # noqa: E731

    flat = NormPhi(phi_power(0.0), spec, FLOOR)
    pinned = ShiftedPowerRate(-1.0, lo, hi, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)
    assert F_max(flat) == pytest.approx(F_max(pinned), rel=1e-12)
    assert F_max(flat) == pytest.approx(101.61098715397782, rel=1e-12)


def test_it_is_exactly_scale_invariant_in_the_base(spec):
    """Multiplying the base by any positive constant leaves the output unchanged.

    This is what makes a min-variance objective over φ well-posed instead of collapsing to φ→0, and
    it is *why* the no-op above holds rather than a coincidence of the pinned construction.
    """
    base = phi_power(-1.0)
    ref = NormPhi(base, spec, FLOOR)(spec)
    for c in (1e-6, 3.0, 1e6):
        scaled = NormPhi(lambda lam, _c=c: _c * base(lam), spec, FLOOR)(spec)
        assert torch.allclose(scaled, ref, rtol=1e-12, atol=0.0), f"not invariant at c={c}"
