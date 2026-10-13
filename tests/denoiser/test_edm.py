"""Oracle tests for the colored-EDM preconditioners and the analytic head (Phase V0).

The preconditioners ``c_skip, c_out, c_in`` are matrix-free functions of ``S`` checked against a dense
eigendecomposition oracle for the whole power-law colouring family **and** a non-power-law positive
polynomial ``φ`` — the generality gate: the EDM preconditioners are functions of the forward moments, so
they must reuse the same formulas for any (positive, same-``S``) graph-filter colouring. The two EDM
identities (``c_skip Σ_F m_F^{-1} = c_out²`` and the Gaussian posterior covariance ``= c_out²``) are
checked per band, and a non-positive colouring — an invalid rate — is rejected.

float64 with a resolved Chebyshev degree ``K`` so the smooth applies gate near machine precision.
"""

from __future__ import annotations

import pytest
import torch

from ccd.denoiser import ColoredEDM
from ccd.forward import ColoredForward, phi_power
from ccd.operators import make_synthetic_operator, spectrum_bracket
from _oracles import dense_apply, rel_err

K = 400                                        # resolves the smooth applies to ~1e-13 over F ≤ 4 (float64)
FS = [0.2, 1.0, 4.0]
PHI_POLY = lambda lam: 0.5 + lam + 0.3 * lam ** 2      # non-power-law, positive & monotone on the bracket
COLOURINGS = [
    ("a=-1", phi_power(-1.0)), ("a=0", phi_power(0.0)), ("a=0.5", phi_power(0.5)),
    ("a=1", phi_power(1.0)), ("poly", PHI_POLY),       # <- the general graph-filter-style colouring
]


def _setup(d=24, n=400, tau=0.05, seed=0):
    # n > d ⇒ Ĉ_n full rank ⇒ v_0 = λ−τ > 0 (smooth regime away from the F→0 edge).
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed),
                                 dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    return op, lo, hi


def _edm(phi):
    op, lo, hi = _setup()
    return ColoredEDM(ColoredForward(op, lo, hi, phi=phi, K=K)), op


def _cskip_scalar(lam, F, phi, tau):
    m = torch.exp(-0.5 * F * phi(lam))
    return (lam - tau) * m / (lam - tau * m * m)


def _cout_scalar(lam, F, phi, tau):
    e = torch.exp(-F * phi(lam))
    return torch.sqrt(torch.clamp((lam - tau) * lam * (1.0 - e) / (lam - tau * e), min=0.0))


# ----------------------------------------------------------- operator applies vs dense (the core gate)
@pytest.mark.parametrize("label,phi", COLOURINGS)
@pytest.mark.parametrize("F", FS)
def test_c_skip_matches_dense(label, phi, F):
    edm, op = _edm(phi)
    v = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    want = dense_apply(lambda l: _cskip_scalar(l, F, phi, op.tau), op, v)
    assert rel_err(edm.c_skip(v, F), want) < 1e-8


@pytest.mark.parametrize("label,phi", COLOURINGS)
@pytest.mark.parametrize("F", FS)
def test_c_out_sq_matches_dense(label, phi, F):
    # the smooth squared form c_out² = Σ_0 Σ_F / V_F — geometric convergence ⇒ machine precision.
    edm, op = _edm(phi)
    v = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    want = dense_apply(lambda l: _cout_scalar(l, F, phi, op.tau) ** 2, op, v)
    assert rel_err(edm.c_out_sq(v, F), want) < 1e-8


@pytest.mark.parametrize("label,phi", COLOURINGS)
@pytest.mark.parametrize("F", FS)
def test_c_out_matches_dense(label, phi, F):
    # c_out carries a √(λ−τ) branch point at λ=τ=lo (the Σ_0^{1/2} factor), so the sqrt apply converges
    # only algebraically: ~5e-7 at K=400 (negligible for float32 training), vs machine precision for the
    # smooth c_out_sq above. The gate is deliberately loose to reflect that intrinsic branch point.
    edm, op = _edm(phi)
    v = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    want = dense_apply(lambda l: _cout_scalar(l, F, phi, op.tau), op, v)
    assert rel_err(edm.c_out(v, F), want) < 1e-5


@pytest.mark.parametrize("label,phi", COLOURINGS)
@pytest.mark.parametrize("F", FS)
def test_c_in_matches_dense(label, phi, F):
    edm, op = _edm(phi)
    v = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    want = dense_apply(lambda l: (l - op.tau * torch.exp(-F * phi(l))) ** (-0.5), op, v)   # V_F^{-1/2}
    got = edm.c_in(v, F, tol=1e-11, maxiter=20000)
    assert rel_err(got, want) < 1e-7


# --------------------------------------------------------- the two EDM identities (per band, exact)
@pytest.mark.parametrize("label,phi", COLOURINGS)
def test_edm_identities_hold(label, phi):
    # verify the two EDM identities using the MODULE's own scalar functions (not a re-derivation), so
    # this also covers _cskip_fn / _cout_sq_fn.
    edm, op = _edm(phi)
    tau = op.tau
    lam = torch.linspace(edm.fwd.lo, edm.fwd.hi, 2049, dtype=torch.float64)
    for F in FS:
        m = torch.exp(-0.5 * F * phi(lam))
        SigmaF = lam * (1.0 - m * m)
        cskip = edm._cskip_fn(F)(lam)
        cout2 = edm._cout_sq_fn(F)(lam)
        # c_skip Σ_F m_F^{-1} = c_out²  (the paper's corollary c_skip Σ_t m_t^{-1} = c_out²)
        assert (cskip * SigmaF / m - cout2).abs().max() < 1e-12
        # Gaussian posterior covariance Σ_0 − c_skip m_F Σ_0 = c_out²  (Cov_Gauss(x0|xF) = c_out²)
        assert ((lam - tau) - cskip * m * (lam - tau) - cout2).abs().max() < 1e-12


# ----------------------------------------------------------------------- head semantics & behaviour
def test_head_is_cskip():
    edm, op = _edm(phi_power(1.0))
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
    assert torch.equal(edm.head(v, 1.5), edm.c_skip(v, 1.5))


def test_head_fades_with_noise():
    # E[x0|xF] = c_skip(S) xF: the Wiener head recovers the signal at low noise and forgets it at high
    # noise, so ‖head(v, F)‖ decreases in F.
    edm, op = _edm(phi_power(1.0))
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    assert edm.head(v, 0.05).norm() > edm.head(v, 20.0).norm()


def test_c_in_refuses_unconverged():
    # c_in delegates to whiten, which refuses a non-converged solve (require_converged=True default).
    edm, op = _edm(phi_power(1.0))
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    with pytest.raises(RuntimeError):
        edm.c_in(v, 0.01, tol=1e-12, maxiter=2)          # steep + starved solve → not converged → raises


def test_batched_matches_columnwise():
    edm, op = _edm(PHI_POLY)
    V = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
    F = 1.0
    batched = edm.c_skip(V, F)
    cols = torch.stack([edm.c_skip(V[:, j], F) for j in range(V.shape[1])], dim=1)
    assert rel_err(batched, cols) < 1e-10


# ------------------------------------------------------------------------------------- guards
def test_rejects_nonpositive_F():
    edm, _ = _edm(phi_power(1.0))
    v = torch.randn(edm.fwd.op.d, dtype=torch.float64)
    for F in (0.0, -1.0):
        for method in (edm.c_skip, edm.c_out, edm.c_out_sq, edm.head):
            with pytest.raises(ValueError):
                method(v, F)
        with pytest.raises(ValueError):
            edm.c_in(v, F)                               # rejects via the whiten → v_inverse_half delegate


def test_rejects_nonpositive_colouring():
    # A non-positive rate φ(λ)=−λ makes v_F = λ − τ e^{Fλ} < 0 near λ=τ: an invalid forward process,
    # which the shared positive-bracket guard must reject (the positivity condition from the review).
    op, lo, hi = _setup()
    edm = ColoredEDM(ColoredForward(op, lo, hi, phi=lambda lam: -lam, K=K))
    v = torch.randn(op.d, dtype=torch.float64)
    with pytest.raises(ValueError):
        edm.c_skip(v, 1.0)
    with pytest.raises(ValueError):
        edm.c_in(v, 1.0)


def test_rejects_negative_lobe_colouring():
    # The subtle case: a colouring with a NEGATIVE lobe away from the endpoints (φ<0 on a sub-band) is an
    # invalid rate (Σ_F<0 there) even though V_F stays SPD, so a v_min>0 check alone does NOT catch it.
    # The φ≥0 guard must reject it — otherwise c_out silently clamps a negative posterior variance to 0.
    op, lo, hi = _setup()
    phi_lobe = lambda lam: (lam - 0.3) ** 2 - 0.02       # <0 on λ∈(≈0.16,0.44); φ(τ),φ(hi)>0 so V_F SPD
    edm = ColoredEDM(ColoredForward(op, lo, hi, phi=phi_lobe, K=K))
    v = torch.randn(op.d, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-negative rate"):
        edm.c_skip(v, 1.0)
