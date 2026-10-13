"""The ``a > 0`` two-point pin, moved out of ``rate.py`` into ``ccd.forward.oriented_rate``.

These are `77ff994`'s mirrored-pin tests, moved with the code they test (hand-off 2026-10-12): the
same names and assertions, built through :func:`pinned_power_rate` and read through
:func:`orientation` / :func:`pinned_at` instead of methods ``main``'s class does not have. The
``a < 0`` tests that stay in `test_colouring_family.py` exercise ``rate.py`` itself.

Hermetic: the same wide log-spaced spectrum as `test_colouring_family.py`'s pinned-law block.
"""

from __future__ import annotations

import math

import pytest
import torch

from ccd.forward.oriented_rate import (MirroredShiftedPowerRate, generation_order, orientation,
                                       pinned_at, pinned_power_rate)
from ccd.forward.rate import ShiftedPowerRate, phi_power
from ccd.forward.rate import generation_order as rate_generation_order

PIN_A = [-1.0, -1.5, -2.0]
PIN_FLOOR, PIN_FMIN, PIN_C = 0.077, 0.02, 3.0
PIN_LO, PIN_HI = 0.05, 8.86                                    # matches the real bracket's 177× ratio


def _pin_spectrum(n=256):
    """Strictly distinct, log-spaced spectrum spanning the same ratio as the real d=256 bracket."""
    return torch.logspace(math.log10(PIN_LO), math.log10(PIN_HI), n, dtype=torch.float64)


def _pin_R():
    return math.log(2.0) / (PIN_C * PIN_FMIN * PIN_FLOOR)      # = 150.0319…


def _pin(a, **kw):
    kw = dict(FLOOR=PIN_FLOOR, F_min=PIN_FMIN, c=PIN_C, **kw)
    return pinned_power_rate(a, PIN_LO, PIN_HI, **kw)


def test_shifted_power_rejects_a_zero():
    """a=0 has R=1 by definition and cannot be two-point pinned; non-finite a is refused with it."""
    for bad in (0.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="needs a finite a != 0"):
            _pin(bad)


def test_the_mirrored_class_refuses_a_non_positive_exponent():
    """a<0 belongs to rate.py's class; the mirrored one must not quietly build a second a<0 path."""
    for bad in (-1.5, 0.0, float("nan")):
        with pytest.raises(ValueError, match="needs a finite a > 0"):
            MirroredShiftedPowerRate(bad, PIN_LO, PIN_HI)


# ------------------------------------------------ the a>0 pin (hand-off #14 §1.1): mirrored, guarded

def _closed_form_decreasing(a, lam):
    """The ORIGINAL a<0 expression, written out independently of the class, for a bit-identity check."""
    R = _pin_R()
    rho = R ** (1.0 / abs(a))
    s = (PIN_HI - rho * PIN_LO) / (rho - 1.0)
    scale = PIN_FLOOR * (PIN_HI + s) ** (-a)
    return scale * (lam + s) ** a, s, scale


def _closed_form_increasing(a, lam, *, s_ulps=0):
    """`77ff994`'s a>0 expression, written out independently of the class; ``s_ulps`` makes a mutant."""
    R = _pin_R()
    rho = R ** (1.0 / abs(a))
    s = (PIN_HI - rho * PIN_LO) / (rho - 1.0)
    for _ in range(s_ulps):
        s = math.nextafter(s, math.inf)
    scale = PIN_FLOOR * (PIN_LO + s) ** (-a)
    return scale * (lam + s) ** a, s, scale


@pytest.mark.parametrize("a", PIN_A)
def test_the_registered_decreasing_path_is_bit_identical_to_the_closed_form(a):
    """🛑 a<0 through the new constructor IS rate.py's class: same type, same bits."""
    ev = _pin_spectrum(1000)
    r = _pin(a)
    assert type(r) is ShiftedPowerRate                         # constructed, not re-implemented
    want, s, scale = _closed_form_decreasing(a, ev)
    assert torch.equal(r(ev), want)
    assert r.s == s and r.scale == scale and pinned_at(r) == PIN_HI and orientation(r) == "decreasing"


@pytest.mark.parametrize("a", [-x for x in PIN_A])
def test_the_increasing_path_is_bit_identical_to_its_closed_form(a):
    """The moved a>0 arithmetic equals its closed form exactly, and the check sees a one-ulp shift."""
    ev = _pin_spectrum(1000)
    r = _pin(a)
    assert type(r) is MirroredShiftedPowerRate
    want, s, scale = _closed_form_increasing(a, ev)
    assert torch.equal(r(ev), want)
    assert r.s == s and r.scale == scale
    assert r.range() == float(((PIN_HI + s) / (PIN_LO + s)) ** a)
    mutant, s_mut, _ = _closed_form_increasing(a, ev, s_ulps=1)
    assert s_mut != s and not torch.equal(r(ev), mutant)       # the gate can fail


@pytest.mark.parametrize("a", [-x for x in PIN_A])
def test_the_increasing_pin_holds_at_both_ends(a):
    """φ(λ_min)=FLOOR (so NormPhi is still ×1 and F_max is common) and φ(λ_max)=R·FLOOR."""
    r, R = _pin(a), _pin_R()
    assert a > 0 and orientation(r) == "increasing" and pinned_at(r) == PIN_LO
    assert float(r(torch.tensor(PIN_LO, dtype=torch.float64))) == pytest.approx(PIN_FLOOR, rel=1e-12)
    assert float(r(torch.tensor(PIN_HI, dtype=torch.float64))) == pytest.approx(R * PIN_FLOOR, rel=1e-10)
    assert r.range() == pytest.approx(R, rel=1e-10)
    ev = _pin_spectrum()
    assert PIN_FLOOR / float(r(ev).min()) == pytest.approx(1.0, rel=1e-12)   # NormPhi no-op
    coarse, fine = r.clocks()
    assert coarse == pytest.approx(math.log(2.0) / PIN_FLOOR) and fine == pytest.approx(PIN_C * PIN_FMIN)


@pytest.mark.parametrize("a", [-x for x in PIN_A])
def test_the_increasing_pin_reverses_the_generation_order(a):
    """Strictly increasing φ ⇒ generation runs ASCENDING in the operator's eigenvalue."""
    ev = _pin_spectrum(1000)
    phi = _pin(a)(ev)
    assert bool((phi[1:] > phi[:-1]).all())
    assert torch.equal(generation_order(_pin(a), ev), torch.argsort(ev, descending=False))
    assert torch.equal(generation_order(_pin(-a), ev), torch.argsort(ev, descending=True))
    for r in (_pin(a), _pin(-a)):                              # called through, not re-implemented
        assert torch.equal(generation_order(r, ev), rate_generation_order(r, ev))


@pytest.mark.parametrize("a", PIN_A)
def test_the_mirrored_pin_shares_the_shift_and_reflects_the_shape(a):
    """The pin equation depends on |a| only, so ±a share one s; φ₊ is φ₋ reflected in log(λ+s).

    A reflection of the SHAPE, not a relabelling: the two laws put their steep end at opposite
    ends of the spectrum, which is why the artifact records it (hand-off #14 §1.1).
    """
    dec, inc = _pin(a), _pin(-a)
    assert dec.s == inc.s and dec.R == inc.R
    ev = _pin_spectrum(257)
    s = dec.s
    reflected = (PIN_LO + s) * (PIN_HI + s) / (ev + s) - s          # log(λ'+s) mirrored about the midpoint
    assert torch.allclose(torch.log(inc(ev)), torch.log(dec(reflected)), atol=1e-12)
    # The slopes are NOT mirror images in λ: the decreasing law is steep at the small end.
    local_dec, local_inc = dec.local_a(ev).abs(), inc.local_a(ev).abs()
    assert torch.equal(local_dec, local_inc)                           # |a|λ/(λ+s) at either sign...
    assert float(local_dec[0]) < float(local_dec[-1])                  # ...small at λ_min, so the
    # decreasing law's |dφ/dλ| = φ·|local_a|/λ concentrates where φ is large: at λ_min for a<0,
    # at λ_max for a>0.
    dphi_dec = (dec(ev) * local_dec / ev)
    dphi_inc = (inc(ev) * local_inc / ev)
    assert float(dphi_dec[0]) > float(dphi_dec[-1])
    assert float(dphi_inc[-1]) >= float(dphi_inc[0])                  # equality only at a=+1 (linear)
    if abs(a) != 1.0:
        assert float(dphi_inc[-1]) > float(dphi_inc[0])


@pytest.mark.parametrize("a", [-x for x in PIN_A])
def test_the_increasing_law_needs_no_v_guard(a):
    """φ'>0 ⇒ dv/dλ = 1 + τFφ'e^{−Fφ} > 0 for every F, so a tau that would fail at a<0 passes at a>0."""
    tau = 0.05
    r = _pin(a, tau=1e9)                                               # accepted: nothing to guard
    ev = torch.logspace(math.log10(PIN_LO), math.log10(PIN_HI), 20_001, dtype=torch.float64)
    for F in (PIN_FMIN, 0.5, 9.0):
        v = ev - tau * torch.exp(-F * r(ev))
        assert bool((v[1:] >= v[:-1]).all())
    with pytest.raises(ValueError, match="non-monotone"):
        _pin(-a, tau=1e9)                                              # ...and the a<0 guard still fires


def test_from_spectrum_builds_the_mirrored_class():
    """The inherited classmethod constructs the subclass, pinned on the spectrum's own extremes."""
    ev = _pin_spectrum()
    r = MirroredShiftedPowerRate.from_spectrum(1.5, ev, FLOOR=PIN_FLOOR, F_min=PIN_FMIN, c=PIN_C)
    assert type(r) is MirroredShiftedPowerRate
    # The extremes of the logspace grid, not PIN_LO/PIN_HI: they differ in the last bit.
    want = pinned_power_rate(1.5, float(ev.min()), float(ev.max()), FLOOR=PIN_FLOOR, F_min=PIN_FMIN,
                             c=PIN_C)
    assert (r.s, r.scale, r.range()) == (want.s, want.scale, want.range())


def test_orientation_and_pin_refuse_a_rate_that_is_not_pinned():
    """A plain power law has no pinned end; reading one off its sign would be the old label error."""
    for fn in (orientation, pinned_at):
        with pytest.raises(TypeError, match="two-point-pinned"):
            fn(phi_power(1.0))
