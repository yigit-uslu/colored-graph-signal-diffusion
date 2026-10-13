"""Tests for the G0 clock-coverage audit (`ccd.forward.clock_audit`).

Mutation targets these are written to catch, per `docs/fmri_a0_build_plan.md` §5:
  * `lambda` swapped for `lambda - tau` in the variance weight
  * the ridge term dropped from `F_star` (i.e. `ln2/phi`)
  * an off-by-one bucket boundary at `F_min`
  * a one-sided instead of symmetric near-ridge snap
  * a gate that never rejects
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from ccd.forward.clock_audit import (
    TOL_NULL_REL,
    band_weights,
    coverage,
    f_star,
    format_table,
    g0_gate,
)

TAU = 0.01
F_MIN, F_MAX = 0.02, 101.61
FLOOR = 0.077


@pytest.fixture(scope="module")
def toy():
    """A small spectrum with a genuine ridge null space, an exact-zero and a tiny-positive residual."""
    rng = np.random.default_rng(0)
    d = 12
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = np.array([9.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.05, TAU, TAU, TAU, TAU, TAU])
    # perturb three ridge bands to straddle tau by amounts far below the snap tolerance
    lam[-3:] += np.array([-1.7e-14, +2.6e-17, 0.0])
    lap = q @ np.diag(np.linspace(0.05, 1.9, d)) @ q.T
    lap = 0.5 * (lap + lap.T)
    return lam, q, lap


def phi_power(a, lam_max):
    return lambda lam: FLOOR * (np.asarray(lam) / lam_max) ** a


# ----------------------------------------------------------------- weighting convention


def test_variance_weight_uses_data_variance_not_eigenvalue(toy):
    """Mutation guard: weighting by `lam` instead of `lam - tau` gives the ridge bands mass."""
    lam, q, lap = toy
    var_w, _, live, _ = band_weights(lam, q, lap, tau=TAU)
    # every band at the ridge must carry exactly zero variance weight
    assert var_w[~live].sum() == 0.0
    # and the correct weights are proportional to lam - tau on the live bands
    expect = np.clip(lam - TAU, 0, None)
    expect = expect / expect.sum()
    np.testing.assert_allclose(var_w, expect, atol=1e-15)
    # the eigenvalue-weighted alternative is materially different
    wrong = lam / lam.sum()
    assert np.abs(var_w - wrong).max() > 1e-3


def test_qv_weight_is_a_quadratic_form_not_a_copy_of_variance(toy):
    lam, q, lap = toy
    var_w, qv_w, _, _ = band_weights(lam, q, lap, tau=TAU)
    assert math.isclose(qv_w.sum(), 1.0, rel_tol=1e-12)
    assert np.abs(qv_w - var_w).max() > 1e-3, "QV weight must not collapse onto the variance weight"


def test_qv_weights_reproduce_trace_L_C(toy):
    """Sum of unnormalised QV contributions must equal tr(L C) built independently.

    The reference uses the module's own SYMMETRIC snap, not `clip(lam-tau, 0, None)`: clipping
    leaves the `+2.6e-17` residual alive and the two conventions then disagree on that band.
    """
    lam, q, lap = toy
    d = lam - TAU
    v = np.where(np.abs(d) <= TOL_NULL_REL * lam.max(), 0.0, d)
    C = q @ np.diag(v) @ q.T
    _, qv_w, _, _ = band_weights(lam, q, lap, tau=TAU)
    contrib = v * np.einsum("ik,ij,jk->k", q, lap, q)
    np.testing.assert_allclose(contrib.sum(), np.trace(lap @ C), rtol=1e-10)
    np.testing.assert_allclose(qv_w, contrib / contrib.sum(), rtol=1e-10, atol=1e-15)


# ----------------------------------------------------------------- the symmetric snap


def test_snap_is_symmetric(toy):
    """A tiny POSITIVE residual must be snapped too, else the live mask is float noise."""
    lam, q, lap = toy
    _, _, live, n_snapped = band_weights(lam, q, lap, tau=TAU)
    assert n_snapped == 5, "all five ridge bands (below, above and exactly at tau) must snap"
    assert live.sum() == 7


def test_genuine_indefiniteness_raises_rather_than_clamping(toy):
    lam, q, lap = toy
    bad = lam.copy()
    bad[-1] = TAU - 1e-3                      # far below tol; a real operator failure
    with pytest.raises(ValueError, match="below the ridge"):
        band_weights(bad, q, lap, tau=TAU)


def test_snap_tolerance_is_relative_to_lam_max(toy):
    lam, q, lap = toy
    tol = TOL_NULL_REL * lam.max()
    assert tol == pytest.approx(1e-12 * 9.0)
    assert tol > 1.7e-14, "the measured worst residual must sit inside the tolerance"


# ----------------------------------------------------------------- F* and bucketing


def test_f_star_is_ridge_aware(toy):
    """Mutation guard: `ln2/phi` overstates every near-ridge clock."""
    lam, _, _ = toy
    phi = np.full_like(lam, FLOOR)
    exact = f_star(phi, lam, tau=TAU)
    naive = math.log(2.0) / phi
    assert exact[0] < naive[0]
    assert exact[-1] == pytest.approx(0.0, abs=1e-12), "a pure-ridge band has F* = 0 exactly"
    assert np.all(exact[:7] > 0)


def test_f_star_inverts_snr(toy):
    """F* must be the exact inverse of ccd.forward.rate.snr: SNR(F*) == 1."""
    import torch

    from ccd.forward.rate import snr

    lam, _, _ = toy
    live = lam > TAU * (1 + 1e-9)
    phi_c = lambda t: torch.full_like(torch.as_tensor(t), FLOOR)
    fs = f_star(np.full_like(lam, FLOOR), lam, tau=TAU)
    for k in np.flatnonzero(live):
        got = float(snr(torch.as_tensor(lam[k : k + 1]), float(fs[k]), phi_c, TAU))
        assert got == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("nudge,expect_in_window", [(1.0 + 1e-9, True), (1.0 - 1e-9, False)])
def test_bucket_boundary_at_F_min(toy, nudge, expect_in_window):
    """The `F* < F_min` boundary is exclusive: a band just above F_min is in-window, just below is not.

    A constant phi makes F*(lam) increasing in lam, so only the TOP band sits near the boundary and
    every smaller band is annihilated regardless -- the assertion is about that top band alone.
    """
    lam, q, lap = toy
    target = math.log(2.0 - TAU / lam[0]) / (F_MIN * nudge)     # puts the top band at F_min*nudge
    rec = coverage(lambda l: np.full_like(np.asarray(l), target), lam, q, lap,
                   tau=TAU, F_min=F_MIN, F_max=F_MAX, label="edge")
    assert (rec.n_window == 1) is expect_in_window
    assert rec.n_below == (6 if expect_in_window else 7)


def test_buckets_partition_the_live_bands(toy):
    lam, q, lap = toy
    rec = coverage(phi_power(-1.0, lam.max()), lam, q, lap,
                   tau=TAU, F_min=F_MIN, F_max=F_MAX, label="a=-1")
    assert rec.n_below + rec.n_window + rec.n_above == rec.n_live
    assert rec.var_below + rec.var_window + rec.var_above == pytest.approx(1.0, abs=1e-12)
    assert rec.qv_below + rec.qv_window + rec.qv_above == pytest.approx(1.0, abs=1e-12)


def test_steeper_negative_clock_annihilates_more(toy):
    lam, q, lap = toy
    recs = [coverage(phi_power(a, lam.max()), lam, q, lap, tau=TAU,
                     F_min=F_MIN, F_max=F_MAX, label=f"a={a}")
            for a in (0.0, -1.0, -2.0, -4.0)]
    shares = [r.qv_below for r in recs]
    assert shares == sorted(shares), f"annihilation must be monotone in |a|; got {shares}"
    assert shares[0] == 0.0


def test_phi_must_be_positive(toy):
    lam, q, lap = toy
    with pytest.raises(ValueError, match="strictly positive"):
        coverage(lambda l: np.asarray(l) - lam.max(), lam, q, lap,
                 tau=TAU, F_min=F_MIN, F_max=F_MAX)


# ----------------------------------------------------------------- the gate itself


def _rec(label, var_below, qv_below, *, clock_range=1.0, arm_class="flat"):
    from ccd.forward.clock_audit import CoverageRecord

    return CoverageRecord(label=label, arm_class=arm_class,
                          n_live=1, n_snapped=0, n_below=0, n_window=1, n_above=0,
                          var_below=var_below, var_window=1 - var_below, var_above=0.0,
                          qv_below=qv_below, qv_window=1 - qv_below, qv_above=0.0,
                          phi_min=1.0, phi_max=float(clock_range), clock_range=clock_range)


def _ladder(*ranges, var=0.0, qv=0.0):
    """A frozen-shape ladder: one flat `a=0` arm at R=1 plus pinned arms at the given ranges."""
    return [_rec("a=0", var, qv, clock_range=1.0, arm_class="flat")] + [
        _rec(f"arm{i}", var, qv, clock_range=r, arm_class="pinned") for i, r in enumerate(ranges)
    ]


def test_gate_passes_clean_ladder():
    passed, reasons = g0_gate([_rec("a=0", 0.0, 0.0), _rec("a=-1", 2e-6, 6e-6)])
    assert passed and reasons == []


def test_gate_rejects_absolute_excess():
    passed, reasons = g0_gate([_rec("a=0", 0.0, 0.0), _rec("a=-2", 0.26, 0.58)])
    assert not passed
    assert any("variance share" in r for r in reasons)
    assert any("E[QV] share" in r for r in reasons)


def test_gate_rejects_differential_annihilation_within_the_absolute_bar():
    """Both clocks sit under max_share, but they differ -- that is the confound the gate exists for."""
    passed, reasons = g0_gate([_rec("a=0", 0.0, 0.0), _rec("a=-2", 4e-3, 4e-3)],
                              max_share=5e-3, max_spread=1e-3)
    assert not passed
    assert any("spread" in r for r in reasons)


def test_gate_needs_at_least_one_record():
    with pytest.raises(ValueError):
        g0_gate([])


# ----------------------------------------------------------------- R3: the range clause


def test_gate_rejects_a_ladder_arm_built_at_the_wrong_range():
    """The R3 defect, pinned: annihilation is clean, but one arm is at R=150 among R=120."""
    recs = _ladder(120.0, 120.0, 150.0, 120.0)
    passed, _ = g0_gate(recs)                                  # no expect_R: silently passes
    assert passed, "without expect_R the gate cannot see the range at all — that IS the defect"

    passed, reasons = g0_gate(recs, expect_R=120.0)
    assert not passed
    assert len(reasons) == 1 and "arm2" in reasons[0] and "150" in reasons[0]


def test_gate_passes_the_frozen_ladder():
    passed, reasons = g0_gate(_ladder(*[120.0] * 4, var=2e-6, qv=6e-6), expect_R=120.0)
    assert passed and reasons == []


def test_a0_record_passes_the_range_clause_rather_than_failing_it():
    """`a=0` has R=1 by definition and does not share the pinned range (plan §1.7)."""
    passed, reasons = g0_gate(_ladder(120.0), expect_R=120.0)
    assert passed and reasons == []
    # and it is the arm_class that carves it out, not the label:
    relabelled = [_rec("whatever", 0.0, 0.0, clock_range=1.0, arm_class="flat"),
                  _rec("a=0", 0.0, 0.0, clock_range=120.0, arm_class="pinned")]
    passed, reasons = g0_gate(relabelled, expect_R=120.0)
    assert passed and reasons == [], "the carve-out must not be recovered by parsing labels"


def test_a_flat_arm_whose_phi_is_not_constant_is_rejected():
    recs = [_rec("a=0", 0.0, 0.0, clock_range=1.0000001, arm_class="flat")]
    passed, reasons = g0_gate(recs, expect_R=120.0)
    assert not passed and "not exactly 1" in reasons[0]


def test_range_clause_tolerance_is_relative_and_tight():
    assert g0_gate(_ladder(120.0 * (1 + 5e-7)), expect_R=120.0, rtol_R=1e-6)[0]
    assert not g0_gate(_ladder(120.0 * (1 + 5e-6)), expect_R=120.0, rtol_R=1e-6)[0]


def test_arm_class_must_be_declared_and_valid():
    with pytest.raises(ValueError, match="arm_class"):
        _rec("x", 0.0, 0.0, arm_class="negative")
    with pytest.raises(ValueError, match="expect_R"):
        g0_gate(_ladder(120.0), expect_R=1.0)


FMRI = pathlib.Path(__file__).resolve().parents[2] / "data" / "fmri"


@pytest.mark.skipif(not (FMRI / "S_ccd.npy").exists(), reason="fMRI bundle not built")
def test_live_band_count_on_the_real_bundle_is_247():
    """Pin the A0 snap convention against the two neighbouring thresholds.

    Three counts are in play across the plan documents: `gen_fmri` reports 246, an absolute 1e-10
    threshold gives 248, and the A0 relative rule gives 247. The band in dispute has
    `lam - tau = 1.657e-10`, i.e. 4.6e-13 of the trace. This test exists so the convention cannot
    drift silently between them -- see `docs/fmri_a0_build_plan.md` section 7.
    """
    w = np.linalg.eigvalsh(np.load(FMRI / "S_ccd.npy"))
    d = w - TAU
    assert int((np.abs(d) > TOL_NULL_REL * w.max()).sum()) == 247, "A0 relative convention"
    assert int((np.abs(d) > 1e-10).sum()) == 248, "parent absolute convention"
    straddler = np.abs(d)[(np.abs(d) > 1e-10) & (np.abs(d) <= TOL_NULL_REL * w.max())]
    assert straddler.size == 1 and straddler[0] == pytest.approx(1.657e-10, rel=1e-3)


@pytest.mark.skipif(not (FMRI / "S_ccd.npy").exists(), reason="fMRI bundle not built")
def test_the_partition_is_stable_but_the_within_snap_sign_split_is_not():
    """`(247 live, 113 snapped, 0 genuinely below)` is what may be sealed -- and nothing finer.

    The sign split *inside* the snapped cluster is eigensolver roundoff. Measured on identical
    bytes: the stored `evals.npy` gives 58/1/54 while a recomputation gives 55/0/58 -- and the
    recomputed split moves with the BLAS thread count alone (55/0/58 at 1-2 threads, 56/0/57 at 4,
    54/0/59 at 8). This test pins the invariant partition and demonstrates the instability, so that
    nobody later "reconciles" the two numbers by picking one.
    """
    S = np.load(FMRI / "S_ccd.npy")
    w = np.linalg.eigvalsh(S)
    tol = TOL_NULL_REL * float(w.max())
    d = w - TAU
    live, snapped, below = int((np.abs(d) > tol).sum()), int((np.abs(d) <= tol).sum()), int((d < -tol).sum())
    assert (live, snapped, below) == (247, 113, 0)
    assert live + snapped + below == w.size == 360, "the sealed counts must partition d"

    stored = np.load(FMRI / "evals.npy") - TAU
    assert int((np.abs(stored) > tol).sum()) == 247, "the partition survives the decomposition"
    within = np.abs(stored) <= tol
    assert int(within.sum()) == 113
    # the split differs between the two decompositions -- that IS the finding, not a defect:
    assert int((stored[within] < 0).sum()) != int((d[np.abs(d) <= tol] < 0).sum())


@pytest.mark.skipif(not (FMRI / "S_ccd.npy").exists(), reason="fMRI bundle not built")
def test_frozen_ladder_passes_g0_and_plain_clocks_fail():
    """End-to-end: the shipped ladder clears G0 on the real operator and the withdrawn clocks do not."""
    import torch

    from ccd.forward.rate import ShiftedPowerRate

    S = np.load(FMRI / "S_ccd.npy")
    w, U = np.linalg.eigh(S)
    w, U = w[::-1].copy(), U[:, ::-1].copy()
    lap = np.load(FMRI / "L.npy")
    lmax, lmin = float(w[0]), float(w[-1])
    R = 120.0
    c = math.log(2.0) / (R * F_MIN * FLOOR)
    kw = dict(tau=TAU, F_min=F_MIN, F_max=F_MAX)

    frozen = [coverage(lambda l: np.full_like(np.asarray(l, dtype=float), FLOOR),
                       w, U, lap, label="a=0", **kw)]
    for a in (-0.5, -1.0, -2.0, -4.0):
        rate = ShiftedPowerRate(a, lmin, lmax, FLOOR=FLOOR, F_min=F_MIN, c=c, tau=TAU)
        frozen.append(coverage(lambda l, r=rate: r(torch.as_tensor(np.asarray(l, dtype=float))).numpy(),
                               w, U, lap, label=f"a={a}", **kw))

    assert all(r.n_live == 247 for r in frozen)
    assert all(math.isclose(r.clock_range, R, rel_tol=1e-6) for r in frozen[1:])
    passed, reasons = g0_gate(frozen)
    assert passed, reasons

    plain = [coverage(lambda l, a=a: FLOOR * (np.asarray(l, dtype=float) / lmax) ** a,
                      w, U, lap, label=f"plain{a}", **kw) for a in (-1.0, -1.5, -2.0)]
    assert [round(r.qv_below, 4) for r in plain] == [0.1353, 0.3957, 0.5775]
    assert not g0_gate(frozen + plain)[0]


def test_format_table_is_stable(toy):
    lam, q, lap = toy
    recs = [coverage(phi_power(a, lam.max()), lam, q, lap, tau=TAU, F_min=F_MIN, F_max=F_MAX,
                     label=f"a={a}") for a in (0.0, -1.0)]
    out = format_table(recs)
    assert "clock" in out and "a=0.0" in out and "QV%" in out
    assert len(out.splitlines()) == 4
