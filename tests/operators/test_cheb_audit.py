"""Validation for the blind Chebyshev-degree audit (`src/ccd/operators/cheb_audit.py`).

The audit decides a number every timing and every production apply depends on, so the load-bearing
properties are that it **can fail** (an under-resolved rung must be rejected), that its search is
monotone, that it never silently extends the ladder, and that the triple it reports actually attains
the error it reports.

Run: conda run -n torch_env python -m pytest tests/operators/test_cheb_audit.py -v
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch

from ccd.forward import bands
from ccd.forward.rate import ShiftedPowerRate
from ccd.operators.chebyshev import cheb_coeffs
from ccd.operators.cheb_audit import (
    NoPassingDegreeError,
    audit_degree,
    cheb_eval,
    default_arg_grid,
)

FMRI = pathlib.Path(__file__).resolve().parents[2] / "data" / "fmri"
LO, HI, TAU, FLOOR, F_MIN, R = 0.01, 197.0013, 0.01, 0.077, 0.02, 120.0


def _flat(lam):
    return torch.full_like(lam, FLOOR)


@pytest.fixture(scope="module")
def spectrum():
    return np.geomspace(LO, HI, 120)


# ------------------------------------------------------------------ the scalar evaluator


def test_cheb_eval_matches_a_direct_series_sum():
    """Clenshaw must agree with the naive sum; it is the audit's whole claim of equivalence."""
    lam = torch.linspace(LO, HI, 37, dtype=torch.float64)
    f = bands.m_fn(_flat, 1.0)
    K = 20
    c = cheb_coeffs(f, LO, HI, K, dtype=torch.float64)
    y = (2 * lam - (HI + LO)) / (HI - LO)
    naive = sum(c[k] * torch.cos(k * torch.arccos(y.clamp(-1, 1))) for k in range(K + 1))
    torch.testing.assert_close(cheb_eval(c, lam, LO, HI), naive, rtol=1e-11, atol=1e-11)


def test_cheb_eval_reproduces_an_exactly_representable_polynomial():
    lam = torch.linspace(LO, HI, 25, dtype=torch.float64)
    y = (2 * lam - (HI + LO)) / (HI - LO)
    quad = lambda l: ((2 * l - (HI + LO)) / (HI - LO)) ** 2          # noqa: E731
    c = cheb_coeffs(quad, LO, HI, 8, dtype=torch.float64)
    torch.testing.assert_close(cheb_eval(c, lam, LO, HI), y ** 2, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ the search


def _mild():
    """nominal a=-0.75 -- a genuine convergence curve: 1.6e-1 at K=8 down to 3.1e-12 at K=256.

    A FLAT clock is useless as a fixture here: phi constant makes `m_F` constant in lambda, exact at
    K=0, so every "error" is roundoff and the search returns the first rung trivially. Any test of a
    degree search needs a function that actually costs degree.
    """
    return ShiftedPowerRate(-0.75, LO, HI, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)


def test_returns_the_smallest_passing_rung_and_the_search_is_monotone(spectrum):
    rec = audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI,
                       clocks={"mild": _mild()}, single_args=[1.0], pair_args=[],
                       ladder=(8, 16, 32, 64, 128, 256), tol=1e-6)
    errs = [r.rel_err for r in rec.table]
    assert errs == sorted(errs, reverse=True), f"error must fall with degree; got {errs}"
    passing = [r.K for r in rec.table if r.rel_err <= 1e-6]
    assert rec.K == min(passing) == 128
    assert rec.error_at(rec.K) == rec.rel_err
    assert rec.error_at(64) > 1e-6 > rec.error_at(128), "the ladder must straddle the bar"


def test_an_under_resolved_rung_is_rejected(spectrum):
    """The bar must reject something, and the coarse rung must not sneak through."""
    rec = audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI,
                       clocks={"mild": _mild()}, single_args=[1.0], pair_args=[],
                       ladder=(8, 128), tol=1e-6)
    assert rec.K == 128
    assert rec.error_at(8) > 1e-1, "K=8 is badly under-resolved and must be reported as such"


def test_exhausting_the_ladder_raises_rather_than_extending_it(spectrum):
    clock = ShiftedPowerRate(-0.5, LO, HI, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)
    with pytest.raises(ValueError, match="NOT extended"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI,
                     clocks={"mild": clock}, single_args=[1.0], pair_args=[],
                     ladder=(8, 16, 24), tol=1e-12)


def test_the_reported_worst_triple_attains_the_reported_error(spectrum):
    """A search that reported a number without the cell it came from could not be audited itself."""
    clocks = {"flat": _flat,
              "steep": ShiftedPowerRate(-4.0, LO, HI, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)}
    singles = [0.05, 1.0, 20.0]
    rec = audit_degree({"m_F": bands.m_fn, "h_sqrt": bands.h_sqrt_fn}, {}, spectrum,
                       lo=LO, hi=HI, clocks=clocks, single_args=singles, pair_args=[],
                       ladder=(8, 16, 32, 64), tol=1e-9)

    factory = {"m_F": bands.m_fn, "h_sqrt": bands.h_sqrt_fn}[rec.worst_fn]
    F = float(rec.worst_arg.split("=")[1])
    fn = factory(clocks[rec.worst_clock], F)
    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    exact = fn(lam)
    c = cheb_coeffs(fn, LO, HI, rec.K, dtype=torch.float64)
    got = float((cheb_eval(c, lam, LO, HI) - exact).abs().max()) / float(exact.abs().max())
    assert got == pytest.approx(rec.rel_err, rel=1e-9)


def test_a_clamped_clock_needs_a_strictly_larger_degree_than_the_shifted_one(spectrum):
    """Pins the geometric-vs-algebraic claim the two-point pin is built on.

    `max(lam, lam_lo)^a` achieves the same pin but is only C0; the kink converges algebraically
    while the shifted form is analytic off the bracket. Both are audited on the same spectrum, same
    arguments, same ladder.
    """
    shifted = ShiftedPowerRate(-2.0, LO, HI, FLOOR=FLOOR, F_min=F_MIN, R=R, tau=TAU)
    lam_lo = HI * R ** (-1.0 / 2.0)              # clamp point that realises the SAME range R

    def clamped(lam):
        return FLOOR * (torch.clamp(lam, min=lam_lo) / HI) ** (-2.0)

    lam = torch.as_tensor(spectrum, dtype=torch.float64)
    for clock in (shifted, clamped):             # the comparison is only fair at matched range
        got = float(clock(lam).max() / clock(lam).min())
        assert got == pytest.approx(R, rel=1e-9), f"range {got} != R"

    ladder = (8, 16, 32, 64, 128, 256)
    kw = dict(lo=LO, hi=HI, single_args=[1.0], pair_args=[], ladder=ladder, tol=1e-6)
    rec = audit_degree({"m_F": bands.m_fn}, {}, spectrum, clocks={"s": shifted}, **kw)
    assert rec.K == 32, "the analytic form resolves geometrically"

    with pytest.raises(ValueError, match="NOT extended"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, clocks={"c": clamped}, **kw)

    # and the reason is the RATE, not the constant: the kink decays algebraically.
    from ccd.operators.chebyshev import cheb_coeffs as _cc
    fn = bands.m_fn(clamped, 1.0)
    exact = fn(lam)
    errs = [float((cheb_eval(_cc(fn, LO, HI, K, dtype=torch.float64), lam, LO, HI)
                   - exact).abs().max()) / float(exact.abs().max()) for K in (32, 256)]
    assert errs[0] / errs[1] < 100, f"algebraic decay expected, got ratio {errs[0] / errs[1]:.1f}"


# ------------------------------------------------------------------ guards


def test_empty_and_malformed_inputs_are_rejected(spectrum):
    kw = dict(lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[])
    with pytest.raises(ValueError, match="nothing to audit"):
        audit_degree({}, {}, spectrum, **kw)
    with pytest.raises(ValueError, match="no clocks"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI, clocks={},
                     single_args=[1.0], pair_args=[])
    with pytest.raises(ValueError, match="tol"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, tol=float("nan"), **kw)
    with pytest.raises(ValueError, match="bracket"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=HI, hi=LO,
                     clocks={"flat": _flat}, single_args=[1.0], pair_args=[])
    with pytest.raises(ValueError, match="ladder"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, ladder=(32, 16), **kw)
    with pytest.raises(ValueError, match="non-finite"):
        audit_degree({"m_F": bands.m_fn}, {}, np.array([1.0, np.nan]), **kw)


def test_ladder_exhaustion_carries_the_completed_table(spectrum):
    """A failed audit is a MEASUREMENT — §2.1b-bis's finding *is* the error-vs-degree curve.

    Rev 12 raised a bare `ValueError`, so `fmri_kstar_audit` could store only the message and the
    `all_arms` artifact had no table at all. `NoPassingDegreeError` carries it, and the distinct
    type is what lets a caller record an unresolvable function set while still letting a MALFORMED
    one abort — two different events that a shared `except ValueError` conflates.
    """
    ladder = (8, 16, 32)
    kw = dict(lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[],
              ladder=ladder, tol=1e-30)                    # unreachable by construction

    with pytest.raises(NoPassingDegreeError) as ei:
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, **kw)

    exc = ei.value
    assert isinstance(exc, ValueError), "existing handlers must keep working"
    assert [r.K for r in exc.table] == list(ladder), "every rung attempted, in order"
    assert exc.ladder == ladder and exc.tol == 1e-30
    assert exc.best is exc.table[min(range(len(exc.table)),
                                     key=lambda i: exc.table[i].rel_err)]
    assert all(r.rel_err > exc.tol for r in exc.table), "none may have passed -- it failed"
    assert all(r.worst_fn == "m_F" and r.worst_clock == "flat" for r in exc.table)

    # the type is load-bearing: a MALFORMED function must NOT arrive as NoPassingDegreeError
    with pytest.raises(ValueError) as bad:
        audit_degree({"broken": lambda clk, F: (lambda lam: lam[:1])}, {}, spectrum, **kw)
    assert not isinstance(bad.value, NoPassingDegreeError)
    assert "returned shape" in str(bad.value)


def test_the_argument_grid_covers_both_budgets_and_between_nodes():
    singles, pairs = default_arg_grid(0.02, 101.61, budgets=(200, 400), n_dense=512)
    assert min(singles) == pytest.approx(0.02) and max(singles) == pytest.approx(101.61)
    assert len(pairs) == 600, "one pair per transition of each grid"
    assert all(a > b for a, b in pairs), "pairs run F_a -> F_b with F_a > F_b"
    g400 = set(np.geomspace(0.02, 101.61, 401))
    assert not g400 >= set(singles), "grid nodes alone would not cover the training distribution"


# ------------------------------------------------------------------ the real operator


@pytest.mark.skipif(not (FMRI / "evals.npy").exists(), reason="fMRI bundle not built")
def test_the_pin_shift_collapses_at_mild_exponents_which_is_why_the_mild_arm_goes_dense():
    """The measured step-3 finding, pinned as arithmetic.

    `s = (lam_max - rho*lam_min)/(rho - 1)` with `rho = R^(1/|a|)`, so a mild exponent makes `rho`
    enormous and drives the branch point at `lam = -s` toward the bracket edge. The two-point pin
    buys range control by spending singularity distance. At R=120 the mildest ladder arm ends up
    with a Bernstein parameter of 1.0168, which cannot reach 1e-4 at any admissible degree -- so it
    is dense-dispositioned, per the ruled matrix-free/dense principle.
    """
    lam_max = float(np.load(FMRI / "evals.npy").max())
    lo, hi = TAU, lam_max
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)

    def bernstein(nominal_a):
        rho = R ** (1.0 / abs(nominal_a))
        s = (lam_max - rho * TAU) / (rho - 1.0)
        y = abs((-s - mid) / half)
        return s, y + math.sqrt(y * y - 1.0)

    s_mild, e_mild = bernstein(-0.5)
    s_steep, e_steep = bernstein(-4.0)
    assert s_mild < 0.005 and s_steep > 80.0
    assert e_mild < 1.02 and e_steep > 3.0
    assert e_mild ** -256 > 1e-3, "the mild arm cannot reach the bar at the top of the ladder"
    assert e_steep ** -64 < 1e-30, "the steep arm is spectral well inside it"

    # monotone: difficulty rises as the exponent softens, the opposite of the amplitude-branch story
    es = [bernstein(a)[1] for a in (-0.5, -1.0, -2.0, -4.0)]
    assert es == sorted(es), f"Bernstein parameter must increase with |a|; got {es}"


# ------------------------------------------------------------------ review round 6: false passes


def test_an_audit_with_zero_cases_raises_instead_of_passing(spectrum):
    """A band-function map with only pair arguments builds no cases, reports rel_err 0 and
    "passes" at the first rung -- an audit that checked nothing."""
    with pytest.raises(ValueError, match="ZERO cases"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI, clocks={"flat": _flat},
                     single_args=[], pair_args=[(1.0, 0.5)])
    with pytest.raises(ValueError, match="ZERO cases"):
        audit_degree({"m_F": bands.m_fn}, {}, spectrum, lo=LO, hi=HI, clocks={"flat": _flat},
                     single_args=[], pair_args=[])


def test_a_non_finite_function_is_not_mistaken_for_an_identically_zero_one(spectrum):
    """`denom == 0.0 or not isfinite(denom)` put NaN in the "no relative error" branch, so an
    all-NaN function was skipped and the audit passed."""
    with pytest.raises(ValueError, match="non-finite"):
        audit_degree({"nan": lambda c, F: (lambda lam: lam * float("nan"))}, {}, spectrum,
                     lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[])
    with pytest.raises(ValueError, match="non-finite"):
        audit_degree({"inf": lambda c, F: (lambda lam: lam / 0.0)}, {}, spectrum,
                     lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[])


def test_a_band_function_of_the_wrong_shape_is_reported_clearly(spectrum):
    """Previously this died inside `cheb_coeffs` with a size-mismatch RuntimeError naming neither
    the function nor the argument."""
    with pytest.raises(ValueError, match="returned shape"):
        audit_degree({"bad": lambda c, F: (lambda lam: lam[:3])}, {}, spectrum,
                     lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[])


def test_an_all_zero_case_set_is_reported_as_unmeasured_not_exact(spectrum):
    with pytest.raises(ValueError, match="identically zero"):
        audit_degree({"z": lambda c, F: (lambda lam: lam * 0.0)}, {}, spectrum,
                     lo=LO, hi=HI, clocks={"flat": _flat}, single_args=[1.0], pair_args=[])
