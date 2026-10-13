"""Tests for `scripts/metr_marginals.py` — the exact marginals when `Σ_0 ≠ S − τI`.

📌 The module was called `ccd.forward.misspecified` in an earlier revision, under a claimed
matrix-free exemption the owner REFUSED — the premise was false, since `V_F` does have a composite
matvec. It is a dense, package-local module under `scripts/` and **no `src/ccd` module was added**;
this docstring carried the withdrawn name until the phase-6 re-seal (F9.1's family).

The load-bearing test is the REDUCTION: set `Σ_0 = S − τI` and every dense quantity here must
reproduce the sealed scalar path to machine precision. Without it the generalisation is just a
different formula that happens to run. The second tier checks the general case against an
independent modality — an empirical Gaussian conditional — because two implementations of the same
algebra agreeing would test the algebra, not the design.
"""

from __future__ import annotations

import math

import pytest
import torch

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from ccd.denoiser.edm import ColoredEDM
from ccd.forward import bands
from ccd.forward.marginals import ColoredForward
from metr_marginals import (MisspecifiedColoredEDM, MisspecifiedColoredForward,
                                      psd_inverse_sqrt, psd_sqrt)
from ccd.forward.rate import phi_power
from ccd.operators.backends import SpectralBackend
from ccd.operators.linop import CovarianceOperator

D, TAU = 24, 0.05
FS = (0.05, 0.3, 1.0, 4.0)


def _operator(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(D, 400, generator=g, dtype=torch.float64)
    op = CovarianceOperator(X, TAU)
    S = op._dense()
    w, U = torch.linalg.eigh(S)
    return op, S, w, U


def _pair(sigma0, *, a=-0.5, seed=0):
    op, S, w, U = _operator(seed)
    phi = phi_power(a)
    lo, hi = float(w.min()), float(w.max())
    mis = MisspecifiedColoredForward(op, lo, hi, phi=phi, sigma0=sigma0, evals=w, evecs=U)
    ref = ColoredForward(op, lo, hi, phi=phi, backend=SpectralBackend(w, U, validate_against=op))
    return op, S, w, U, phi, mis, ref


# ------------------------------------------------------------------ REDUCTION: Σ_0 = S − τI
def test_v_marginal_reduces_to_the_sealed_scalar_form():
    op, S, w, U, phi, mis, ref = _pair(None) if False else (None,) * 7
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = S - TAU * torch.eye(D, dtype=torch.float64)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    for F in FS:
        want = (U * bands.v_fn(phi, TAU, F)(w)) @ U.T
        got = mis.v_marginal_dense(F)
        assert torch.allclose(got, want, atol=1e-12, rtol=0), f"F={F}"


@pytest.mark.parametrize("name", ["c_skip", "c_out_sq", "c_out"])
def test_preconditioners_reduce_to_the_sealed_scalar_forms(name):
    """⭐ `c_out_sq` is written subtractively here and multiplicatively in `bands`; they must agree
    exactly when the factors commute, or the generalisation is wrong."""
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = S - TAU * torch.eye(D, dtype=torch.float64)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    fn = {"c_skip": bands.c_skip_fn, "c_out_sq": bands.c_out_sq_fn, "c_out": bands.c_out_fn}[name]
    for F in FS:
        want = (U * fn(phi, TAU, F)(w)) @ U.T
        got = edm._precond(F)[name]
        assert torch.allclose(got, want, atol=1e-11, rtol=1e-10), f"{name} F={F}"


def test_c_in_reduces_to_the_sealed_dense_whitener():
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = S - TAU * torch.eye(D, dtype=torch.float64)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    for F in FS:
        want = (U * bands.v_fn(phi, TAU, F)(w) ** -0.5) @ U.T
        assert torch.allclose(edm._precond(F)["c_in"], want, atol=1e-11, rtol=1e-10)


def test_the_reduction_also_holds_at_the_PRODUCTION_clock():
    """The reduction above runs `phi_power(-0.5)`; the registered arms run `ShiftedPowerRate` at
    nominal `a ~ -5`, a far steeper clock. The reduction is algebraic and should not care — but
    "should not care" is not a measurement, so this runs it at the shape the tranche will use."""
    from ccd.forward.rate import ShiftedPowerRate
    op, S, w, U = _operator()
    lo, hi = float(w.min()), float(w.max())
    # R is DERIVED from this spectrum, not copied from the panel: pinning can only shrink the range,
    # so R must sit below (lam_max/lam_min)^|a|, which is 63.8 here against the panel's 150. Taking
    # half of the reachable maximum keeps the clock steep and the test independent of the operator.
    a = -5.2
    R = 0.5 * (hi / lo) ** abs(a)
    phi = ShiftedPowerRate(a, lo, hi, R=R, FLOOR=0.077, F_min=0.02, c=3.0)
    sigma0 = S - TAU * torch.eye(D, dtype=torch.float64)
    mis = MisspecifiedColoredForward(op, lo, hi, phi=phi, sigma0=sigma0, evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    for F in FS:
        assert torch.allclose(mis.v_marginal_dense(F), (U * bands.v_fn(phi, TAU, F)(w)) @ U.T,
                              atol=1e-12), f"V_F at F={F}"
        for name, fn in (("c_skip", bands.c_skip_fn), ("c_out_sq", bands.c_out_sq_fn)):
            want = (U * fn(phi, TAU, F)(w)) @ U.T
            assert torch.allclose(edm._precond(F)[name], want, atol=1e-11, rtol=1e-10), f"{name} F={F}"


def test_psd_sqrt_rejects_material_indefiniteness_but_absorbs_roundoff():
    """CALIBRATED PAIR. An earlier version clamped every negative eigenvalue to zero — and a test of
    mine asserted that clamping as the contract, which is how a guard gets enshrined backwards.
    Production `c_out` calls this, so a genuinely indefinite conditional covariance must raise; only
    roundoff-scale negativity, sized from the dtype and the matrix norm, may be absorbed."""
    from metr_marginals import negativity_tolerance
    A = torch.eye(4, dtype=torch.float64)
    A[0, 0] = -0.5
    with pytest.raises(ValueError, match="materially indefinite"):
        psd_sqrt(A)
    with pytest.raises(ValueError, match="not positive"):
        psd_inverse_sqrt(A)
    B = torch.eye(4, dtype=torch.float64)
    B[0, 0] = -0.5 * negativity_tolerance(B)                    # inside roundoff: absorbed
    out = psd_sqrt(B)
    assert float(out[0, 0]) == 0.0
    C = torch.eye(4, dtype=torch.float64)
    C[0, 0] = -3.0 * negativity_tolerance(C)                    # just outside: refused
    with pytest.raises(ValueError, match="materially indefinite"):
        psd_sqrt(C)


def test_the_eigensystem_is_validated_against_the_operator():
    """Reviewer finding 5: the constructor advertised a validated backend and accepted any tensors,
    and the tests then built it without one, so nothing checked this anywhere."""
    op, S, w, U = _operator()
    lo, hi = float(w.min()), float(w.max())
    s0 = S - TAU * torch.eye(D, dtype=torch.float64)
    kw = dict(phi=phi_power(-0.5), sigma0=s0)
    MisspecifiedColoredForward(op, lo, hi, evals=w, evecs=U, **kw)         # the good one
    # a COLUMN-RESCALED eigenvector still satisfies S u = lam u exactly, and is the case that
    # `U f(lam) U^T` gets wrong by the scale squared
    U_bad = U.clone(); U_bad[:, 0] *= 1.5
    with pytest.raises(ValueError, match="orthonormal"):
        MisspecifiedColoredForward(op, lo, hi, evals=w, evecs=U_bad, **kw)
    with pytest.raises(ValueError, match="does not describe the operator"):
        MisspecifiedColoredForward(op, lo, hi, evals=w * 1.01, evecs=U, **kw)
    with pytest.raises(ValueError, match="positive definite"):
        MisspecifiedColoredForward(op, lo, hi, evals=w - float(w.min()) - 1e-9, evecs=U, **kw)
    with pytest.raises(ValueError, match="non-finite"):
        MisspecifiedColoredForward(op, lo, hi, evals=w * float("nan"), evecs=U, **kw)


def test_the_cache_is_bounded_and_a_forward_costs_one_pass():
    """Reviewer finding 3. `sample_F_loguniform` draws one F per minibatch from a CONTINUOUS range,
    so an unbounded per-F cache would hold ~80,000 entries of five dense matrices (~75 GB at
    d = 153), and the preconditioners were rebuilt on every call."""
    import metr_marginals as mm
    op, S, w, U = _operator()
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                     sigma0=S - TAU * torch.eye(D, dtype=torch.float64),
                                     evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    for F in torch.linspace(0.05, 5.0, 200).tolist():           # 200 distinct F, as training would
        edm.c_skip(torch.ones(D, dtype=torch.float64), F)
    assert len(mis._cache) <= mm.CACHE_SIZE, f"cache grew to {len(mis._cache)}"
    # the preconditioners live in the same cached entry, so nothing is recomputed per call
    ent = mis._blocks(1.0)
    for k in ("c_skip", "c_out_sq", "c_out", "V_inv_half"):
        assert k in ent
    assert edm._precond(1.0)["c_skip"] is ent["c_skip"]


# ------------------------------------------------------------------ the general, non-commuting case
def _offbasis_sigma0(S, seed=3):
    """A Σ_0 that is PSD, of the same scale as `S − τI`, and NOT a function of S."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    w = torch.linalg.eigvalsh(S) - TAU
    return (Q * w.flip(0)) @ Q.T


def test_the_generalisation_differs_off_eigenbasis_and_the_gap_decays_with_F():
    """NEGATIVE CONTROL — and it asserts the MECHANISM, not a threshold.

    The misspecification is carried entirely by `m_F Σ_0 m_F`, and `m_F = e^{−½Fφ(S)} → 0`, so the
    gap must be large at the data end and shrink monotonically as F grows. A flat "rel > 0.05 at
    every F" bar fails at large F for a correct implementation — which is how this test was first
    written, and what it caught was itself. The same decay is visible on the real panel
    (0.787 at F = 0.06 down to 0.295 at F = 9.0, `scratchpad/metr_review2_probe.py`)."""
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = _offbasis_sigma0(S)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    rels = []
    for F in FS:
        sealed = (U * bands.v_fn(phi, TAU, F)(w)) @ U.T
        rels.append(float((mis.v_marginal_dense(F) - sealed).norm() / sealed.norm()))
    assert rels[0] > 0.05, f"gap at the data end is only {rels[0]:.2e}; Σ_0 is not off-basis enough"
    assert all(a > b for a, b in zip(rels, rels[1:])), f"gap must decay with F; got {rels}"


def test_the_whitener_actually_whitens_the_true_marginal():
    """The defect this module exists to remove: `c_in V_true c_in` must be I, not [0.08, 16.6]."""
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = _offbasis_sigma0(S)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    for F in FS:
        W = edm._precond(F)["c_in"]
        Z = W @ mis.v_marginal_dense(F) @ W
        assert torch.allclose(Z, torch.eye(D, dtype=torch.float64), atol=1e-9)
        # and the sealed scalar whitener does NOT
        W_bad = (U * bands.v_fn(phi, TAU, F)(w) ** -0.5) @ U.T
        Z_bad = W_bad @ mis.v_marginal_dense(F) @ W_bad
        assert not torch.allclose(Z_bad, torch.eye(D, dtype=torch.float64), atol=1e-3)


def test_c_out_sq_is_symmetric_psd_off_eigenbasis():
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=_offbasis_sigma0(S), evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    for F in FS:
        A = edm._precond(F)["c_out_sq"]
        assert torch.allclose(A, A.T, atol=1e-12)
        assert float(torch.linalg.eigvalsh(A).min()) > -1e-10
        assert torch.allclose(edm._precond(F)["c_out"] @ edm._precond(F)["c_out"], A, atol=1e-9)


def test_the_multiplicative_form_would_not_even_be_symmetric():
    """Pins the trap the module warns about: `Σ_0 Σ_F V^{-1}` is not a covariance off-eigenbasis."""
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=_offbasis_sigma0(S), evals=w, evecs=U)
    b = mis._blocks(1.0)
    naive = mis.sigma0 @ b["sigma"] @ b["V_inv"]
    assert float((naive - naive.T).abs().max()) > 1e-3


# ------------------------------------------------------------------ independent modality
def test_c_skip_and_c_out_sq_match_an_empirical_gaussian_conditional():
    """CROSS-CHECK IN A DIFFERENT MODALITY, gated by the SAMPLING LAW rather than a picked tolerance.

    Draw the joint `(x_0, x_F)` and recover `E[x_0|x_F]` by least squares and `Cov(x_0|x_F)` by
    residual covariance — no operator algebra involved. The bounds are computed, not chosen
    (principle 4): for OLS, `vec(B̂ − B)` has covariance `Cov(x_0|x_F) ⊗ (n Cov(x_F))^{-1}`, so
    `E‖B̂ − B‖_F² = tr(c_out²)·tr(V_F^{-1})/n`; for the residual covariance the Wishart law gives
    `E‖Ĉ − C‖_F² = (tr(C)² + ‖C‖_F²)/n`. An earlier version asserted `< 2 %`, which passes or fails
    on `n` and the seed rather than on correctness.
    """
    op, S, w, U = _operator()
    phi = phi_power(-0.5)
    sigma0 = _offbasis_sigma0(S)
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi,
                                     sigma0=sigma0, evals=w, evecs=U)
    edm = MisspecifiedColoredEDM(mis)
    F, n = 1.0, 400_000
    b = mis._blocks(F)
    g = torch.Generator().manual_seed(11)
    A0, Af = psd_sqrt(sigma0), psd_sqrt(b["sigma"])
    x0 = torch.randn(n, D, generator=g, dtype=torch.float64) @ A0
    xF = x0 @ b["m"] + torch.randn(n, D, generator=g, dtype=torch.float64) @ Af

    c_skip, c_out_sq = edm._precond(F)["c_skip"], edm._precond(F)["c_out_sq"]
    B = torch.linalg.lstsq(xF, x0).solution.T
    predicted_B = math.sqrt(float(torch.diagonal(c_out_sq).sum())
                            * float(torch.diagonal(b["V_inv"]).sum()) / n)
    err_B = float((B - c_skip).norm())
    assert err_B < 4.0 * predicted_B, f"‖B̂−c_skip‖={err_B:.3e} vs law {predicted_B:.3e}"

    resid = x0 - xF @ c_skip.T
    emp = resid.T @ resid / n
    tr, fro = float(torch.diagonal(c_out_sq).sum()), float(c_out_sq.norm())
    predicted_C = math.sqrt((tr ** 2 + fro ** 2) / n)
    err_C = float((emp - c_out_sq).norm())
    assert err_C < 4.0 * predicted_C, f"‖Ĉ−c_out²‖={err_C:.3e} vs law {predicted_C:.3e}"

    # NEGATIVE CONTROL, with the defect SIZED FROM THE BOUND so it is provably outside it rather
    # than picked and hoped for. A first attempt used x1.02, which lands at 0.5x the 4-sigma bound —
    # i.e. the control could not have failed the way it was meant to.
    # Each gate gets its OWN control, sized from its OWN law. A single scaled `c_skip` does not
    # serve both: it moves the coefficient statistic at first order and the residual covariance only
    # at second, so one defect large enough for the first sat at 0.3x the second's bound — the
    # control could not have failed the way it was meant to.
    bad_skip = c_skip * (1.0 + 10.0 * predicted_B / float(c_skip.norm()))
    assert float((B - bad_skip).norm()) > 4.0 * predicted_B, "c_skip control is inside its bound"
    bad_cov = c_out_sq * (1.0 + 10.0 * predicted_C / float(c_out_sq.norm()))
    assert float((emp - bad_cov).norm()) > 4.0 * predicted_C, "c_out² control is inside its bound"


# ------------------------------------------------------------------ guards
def test_a_plain_colored_forward_is_refused():
    op, S, w, U = _operator()
    ref = ColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5))
    with pytest.raises(TypeError, match="silently reinstate"):
        MisspecifiedColoredEDM(ref)


@pytest.mark.parametrize("bad", ["asym", "not_psd", "wrong_shape"])
def test_sigma0_is_validated(bad):
    op, S, w, U = _operator()
    s0 = S - TAU * torch.eye(D, dtype=torch.float64)
    if bad == "asym":
        s0 = s0.clone(); s0[0, 1] += 1.0
    elif bad == "not_psd":
        s0 = s0 - 2.0 * float(torch.linalg.eigvalsh(s0).max()) * torch.eye(D, dtype=torch.float64)
    else:
        s0 = s0[:-1, :-1]
    with pytest.raises(ValueError):
        MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                   sigma0=s0, evals=w, evecs=U)


def test_non_positive_F_is_refused():
    op, S, w, U = _operator()
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                     sigma0=S - TAU * torch.eye(D, dtype=torch.float64),
                                     evals=w, evecs=U)
    for F in (0.0, -1.0):
        with pytest.raises(ValueError):
            mis.v_marginal_dense(F)


def test_c_in_refuses_solver_kwargs_it_would_not_use():
    op, S, w, U = _operator()
    mis = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                     sigma0=S - TAU * torch.eye(D, dtype=torch.float64),
                                     evals=w, evecs=U)
    with pytest.raises(ValueError, match="runs no solver"):
        MisspecifiedColoredEDM(mis).c_in(torch.ones(D, dtype=torch.float64), 1.0, tol=1e-8)


def test_psd_inverse_sqrt_refuses_a_singular_matrix():
    A = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="not positive definite"):
        psd_inverse_sqrt(A)


# ===========================================================================================
# R2 — invalid-input guards. Each one gets a mutation it must reject, and (where a tolerance
# exists) a just-inside-roundoff control it must ACCEPT, so the pair calibrates the threshold.
# ===========================================================================================
def _ok_pieces(seed=0, d=10, n=200, tau=0.05):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(d, n, generator=g, dtype=torch.float64)
    op = CovarianceOperator(X, tau)
    S = op._dense()
    w, U = torch.linalg.eigh(S)
    return op, S, w, U, S - tau * torch.eye(d, dtype=torch.float64)


@pytest.mark.parametrize("kind", ["nan_everywhere", "one_nan", "one_inf"])
def test_a_non_finite_sigma0_is_refused_by_THIS_module_not_by_lapack(kind):
    """The symmetry clause passes a NaN: `asym` is NaN and `NaN > tol` is False.

    Before the guard, the only thing stopping it was `torch.linalg.eigvalsh` happening to raise on
    this backend — incidental protection whose message ("failed to converge") names conditioning
    when the fault is a NaN in the input covariance.
    """
    op, S, w, U, sigma0 = _ok_pieces()
    bad = sigma0.clone()
    if kind == "nan_everywhere":
        bad = bad * float("nan")
    elif kind == "one_nan":
        bad[3, 3] = float("nan")
    else:
        bad[2, 5] = bad[5, 2] = float("inf")
    with pytest.raises(ValueError, match="non-finite entries"):
        MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                      sigma0=bad, evals=w, evecs=U)
    # the clause the guard protects really is blind to it
    asym = float((bad - bad.transpose(-2, -1)).abs().max())
    if kind != "one_inf":
        assert not (asym > 1e-9 * float(bad.abs().max())), "symmetry clause would have passed it"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_psd_sqrt_refuses_a_non_finite_matrix_before_decomposing_it(bad):
    d = 8
    A = torch.eye(d, dtype=torch.float64)
    A[2, 2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        psd_sqrt(A)


# 📌 psd_sqrt's negativity threshold already has a calibrated pair, written when that guard was
# added: `test_psd_sqrt_rejects_material_indefiniteness_but_absorbs_roundoff` above. R2 adds only the
# non-finite case it was missing; duplicating the pair here would have been two tests of one property.


@pytest.mark.parametrize("method,args", [("_vmarg_fn", (1.0,)), ("_vmarg_matvec", (1.0,)),
                                         ("_v_bracket", (1.0,))])
def test_the_commuting_only_helpers_raise_instead_of_returning_the_wrong_formula(method, args):
    """Silent access to `v_F(λ) = λ − τe^{−Fφ}` on a misspecified arm is worse than a loud error.

    Every sanctioned route into these is overridden, so today nothing reaches them — but that is a
    proof about sealed call sites which a future upstream edit can add to without a signal here.
    """
    op, S, w, U, sigma0 = _ok_pieces()
    fwd = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=phi_power(-0.5),
                                        sigma0=sigma0, evals=w, evecs=U)
    with pytest.raises(NotImplementedError, match="Sigma_0 = S - tau I"):
        getattr(fwd, method)(*args)
    # and the supported replacements do work
    assert fwd.v_marginal_dense(1.0).shape == (op.d, op.d)
    assert torch.isfinite(fwd._blocks(1.0)["V_eigvals"]).all()


def test_blocks_checks_finiteness_BEFORE_positivity_and_before_caching():
    """`float(w.min()) <= 0.0` is False when `w.min()` is NaN, so a NaN `V_F` reported as PD.

    The same false-comparison defect repaired in the constructor and in `psd_sqrt`, left unfinished
    in `_blocks`. Driven here through `phi`, because `sigma0`/`evals` are validated at construction:
    a rate returning NaN makes `m` and `Sigma_F` non-finite and hence `V_F` too.
    """
    op, S, w, U, sigma0 = _ok_pieces()

    class NaNRate:
        def __call__(self, lam):
            return torch.full_like(torch.as_tensor(lam, dtype=torch.float64), float("nan"))

    fwd = MisspecifiedColoredForward(op, float(w.min()), float(w.max()), phi=NaNRate(),
                                     sigma0=sigma0, evals=w, evecs=U)
    with pytest.raises(ValueError, match="non-finite entries|not finite"):
        fwd._blocks(1.0)
    assert 1.0 not in fwd._cache, "a non-finite block was cached and would be served on every hit"
    # the comparison the guard protects really is blind to it
    bad = torch.tensor([float("nan"), 1.0, 2.0])
    assert not (float(bad.min()) <= 0.0), "NaN spectrum would have reported as positive definite"
