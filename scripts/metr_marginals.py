r"""Exact marginals and EDM preconditioners when ``Sigma_0 != S - tau I`` — **Package B, dense.**

🛑 **THIS LIVES IN `scripts/`, NOT `src/ccd/`, AND THE REASON IS A REFUTATION OF MY OWN ARGUMENT.**
An earlier revision put it under `src/ccd/forward/` and claimed a third exemption from the
matrix-free invariant, on the grounds that the quantity "provably has no matrix-free form". **That
premise was false.** ``V_F`` is not a scalar function of ``S``, but it plainly has a *composite*
matvec

    V_F v = m(S) @ [ Sigma_0 @ ( m(S) v ) ] + Sigma_F(S) v

— two Chebyshev applies around one ``Sigma_0`` matvec — and ``V_F^{-1}`` / ``V_F^{-1/2}`` then follow
from the CG and rational multishift machinery already in :mod:`ccd.operators.solvers` and
:mod:`ccd.operators.rational`. I had conflated "not a scalar function of ``S``" with "no matrix-free
form"; they are different statements and only the first is true.

So there are two honest implementations, and this is the cheap one:

* **matrix-free composite operator in ``src/ccd``** — the general answer, and the only one that could
  serve a large-``d`` arm. Not built here.
* **dense, script-local, small-``d``** — this file. Plan §5.B says in terms that Package B "is NOT
  the matvec showcase ... run the eigh route", ``d = 153`` makes a dense ``eigh`` microseconds, and
  living in ``scripts/`` means the ``src/ccd`` invariant is neither violated nor stretched.

⚠️ **Consequence, declared: this file cannot serve Package A.** A dense ``eigh`` per ``F`` is
impossible at ``d = 8k–20k``. If a graph-native arm with a misspecified prior is ever wanted there,
it needs the composite matrix-free operator, which does not exist yet.

Why any of it is needed. :mod:`ccd.forward.marginals` derives ``V_F = S - tau e^{-F phi(S)}`` from
``Sigma_0 = S - tau I``, true **by construction** for ``S := C_n + tau I``. A graph-native arm runs
``S = sigma^2 (L+gamma I)^{-1}`` while the data still has covariance ``C_train``, and the two do not
commute. Measured on METR-LA fold 0 (`scratchpad/metr_review2_probe.py`): the sealed scalar ``V_F``
is wrong by 30–79 % across the registered ``F*`` window, and the whitener it induces leaves the data
with a spectrum of ``[0.084, 16.6]`` instead of ``[1, 1]``.

**The generalisation, and the trap in it.** From joint Gaussianity of ``(x_0, x_F = m_F x_0 + n)``:

    Cov(x_0, x_F) = Sigma_0 m_F        Cov(x_F) = V_F
    c_skip = Sigma_0 m_F V_F^{-1}                          (the exact MMSE denoiser E[x_0|x_F])
    c_out^2 = Sigma_0 - Sigma_0 m_F V_F^{-1} m_F Sigma_0   (the posterior covariance)
    c_in   = V_F^{-1/2}

⚠️ ``c_out^2`` must be written in that subtractive form. The sealed docstring's
``Sigma_0 Sigma_F V_F^{-1}`` is the same quantity **only when the three commute**; written literally
for non-commuting factors it is not even symmetric, so its square root is undefined. Same class of
error as the dropped ``M S M^T`` cross terms.

**What is NOT affected.** The ancestral-step factors ``c_x``, ``c_d``, ``q``
(:mod:`ccd.forward.bands`) are properties of the forward transition kernel alone and carry no
``Sigma_0``, so the sampler is untouched.

**What IS affected and is NOT in this file** — see :mod:`metr_regime` for the loss weight and the
noise-conditioning feature, both of which the sealed code computes from the commuting model.

**Caching.** ``ccd.denoiser.denoiser.sample_F_loguniform`` draws **one ``F`` per minibatch** from a
continuous range, so an 80,000-update run touches ~80,000 distinct ``F`` values. An unbounded
per-``F`` cache of five dense matrices would reach ~75 GB at ``d = 153``. The cache here is a bounded
LRU whose working set is 1 during training and 1 per sampler step, and every dense quantity for one
``F`` — blocks *and* preconditioners — is built in a single pass: **two** eigendecompositions per
distinct ``F`` (one for ``V_F``, serving both inverses *and* the positivity bracket; one for
``c_out``'s square root) and **zero** for any repeat visit to a cached ``F``.

⚠️ That sentence used to say "one eigendecomposition, not four" and it was false: measured, a
denoiser evaluation ran **five** — two in :meth:`MisspecifiedColoredForward._blocks` plus one per
preconditioner call from :meth:`_require_positive_bracket`, which recomputed the eigenvalues of a
matrix ``_blocks`` had just decomposed. Caching ``V``'s spectrum took a cold evaluation from 5 to 2
and a warm one from 3 to 0.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor

from ccd.denoiser.edm import ColoredEDM
from ccd.forward.marginals import ColoredForward
from ccd.forward.rate import ColouringRate
from ccd.operators.linop import CovarianceOperator

__all__ = ["CACHE_SIZE", "MisspecifiedColoredEDM", "MisspecifiedColoredForward",
           "psd_inverse_sqrt", "psd_sqrt"]


def _sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-2, -1))


def negativity_tolerance(A: Tensor) -> float:
    """Roundoff scale for "is this matrix indefinite, or merely rounded?" — derived, not picked.

    A symmetric eigensolve perturbs eigenvalues by ``O(eps * ||A||_2)``; the constant absorbs the
    backward-error factor and the ``d`` growth of the QR sweeps. Scale- and dtype-derived, so it
    tracks the matrix instead of a hand-chosen tolerance (principle 4).
    """
    eps = float(torch.finfo(A.dtype).eps)
    return 32.0 * eps * max(1.0, float(A.abs().max())) * max(1, A.shape[-1])


def psd_sqrt(A: Tensor) -> Tensor:
    """Symmetric PSD square root that **REFUSES a materially indefinite input**.

    🛑 An earlier version clamped every negative eigenvalue to zero, and production ``c_out`` used
    it — so a genuinely indefinite conditional covariance would have been silently converted into a
    plausible-looking operator, and a test of mine had *enshrined* the clamping as the contract.
    Only roundoff-scale negativity is absorbed now; anything larger raises.
    """
    # 🛑 Finiteness FIRST. `lo < -tol` is `False` when `lo` is NaN, so a non-finite matrix walked
    # straight through the negativity clause. Nothing here caught it — `torch.linalg.eigh` happened
    # to raise on this LAPACK build, on CPU and CUDA alike, which is incidental protection with a
    # message ("failed to converge") that names conditioning when the fault is a NaN in the input.
    if not torch.isfinite(A).all():
        raise ValueError("psd_sqrt received a non-finite matrix; refusing to eigendecompose it. "
                         "Every downstream comparison against a NaN is False, so a NaN would be "
                         "reported as 'not materially indefinite'.")
    w, U = torch.linalg.eigh(_sym(A))
    if not torch.isfinite(w).all():
        raise ValueError(f"psd_sqrt: eigenvalues are not finite (min={float(w.min())}); the "
                         "decomposition did not describe the input")
    tol = negativity_tolerance(A)
    lo = float(w.min())
    if lo < -tol:
        raise ValueError(f"matrix is materially indefinite (lam_min={lo:.3e}, roundoff tolerance "
                         f"{tol:.3e}); refusing to clamp what is not roundoff")
    return (U * torch.clamp(w, min=0.0).sqrt()) @ U.transpose(-2, -1)


def psd_inverse_sqrt(A: Tensor, *, eps: float | None = None) -> Tensor:
    """Symmetric inverse square root; raises rather than returning a silently wrong operator."""
    w, U = torch.linalg.eigh(_sym(A))
    lo = float(w.min())
    if not torch.isfinite(w).all() or lo <= 0.0:
        raise ValueError(f"matrix is not positive definite (lam_min={lo:.3e}); cannot whiten")
    if eps is not None and lo < eps:
        raise ValueError(f"lam_min={lo:.3e} is below the requested floor {eps:.3e}")
    return (U * w.rsqrt()) @ U.transpose(-2, -1)


#: Bounded LRU depth. `sample_F_loguniform` draws ONE F per minibatch, so the working set is 1
#: during training and 1 per sampler step; 4 covers a denoiser evaluation that revisits an F within
#: a step without ever letting the cache track the ~80,000 distinct F values a run touches.
CACHE_SIZE: int = 4


class MisspecifiedColoredForward(ColoredForward):
    """``ColoredForward`` whose ``Σ_0`` is supplied explicitly instead of assumed to be ``S − τI``.

    Args:
        op: the operator ``S`` driving the process (a graph-native prior here). Its dtype is the
            **apply** dtype: what the cached matrices are cast to before they meet a signal.
        lo, hi: spectrum bracket of ``S``.
        phi: the colouring rate.
        sigma0: ``(d, d)`` the data's actual covariance ``Σ_0`` — for a graph arm, ``Ĉ_train``.
            **float64, always** (see below).
        evals, evecs: the eigensystem of ``S``, **float64, always**. Supplied rather than computed
            so the caller's validated backend and this class cannot disagree about what ``S`` is.

    ``m_F`` and ``Σ_F`` are unchanged (both are scalar functions of ``S``); ``V_F`` and its inverses
    are not, and are the three methods overridden.

    🛑 **PRECISION IS SPLIT, AND IT IS NOT A STYLE CHOICE.** Every dense quantity is built in
    **float64** and cast to ``op.dtype`` only on the way out. Two independent reasons, both measured:

    * The registered training dtype is float32 (`REGIME`), and at float32 this class **could not be
      constructed at all**: ``|UᵀU − I|`` on the real fold-0 eigensystem is ``4.77e-7``, which is
      what float32 orthonormality *costs*, against a hard-coded ``1e-8`` threshold. The eigensystem
      was fine; the arithmetic was not. :class:`~ccd.operators.backends.SpectralBackend` had already
      solved this — it checks orthonormality in float64 and correspondence at ``op.dtype`` under a
      dtype-scoped ``rtol`` — and this class advertised that behaviour without having it.
    * ``c_out² = Σ_0 − c_skip m Σ_0`` is a **difference of nearly equal matrices** as ``F → 0``, and
      :func:`psd_sqrt` refuses negativity beyond a dtype- and scale-derived tolerance. Doing that
      subtraction in float32 would inflate the tolerance ~5e-4-fold and hand the guard a number too
      loose to reject anything — the exact way review finding F5 said a guard conceals a defect.

    Running the *network* in float64 is not the alternative: on this body it is ~30× slower on the
    card, which alone would exceed the tranche's 40 GPU-h ceiling.
    """

    #: Orthonormality is checked at this bound in float64 — reachable because the check is float64.
    ORTHO_TOL: float = 1e-8
    #: `S U ≈ U diag(λ)` is checked at the OPERATOR's dtype (that is the arithmetic the rest of the
    #: pipeline actually runs), so the bound is dtype-scoped exactly as `SpectralBackend`'s is.
    CORRESPONDENCE_TOL: dict = {torch.float64: 1e-8, torch.float32: 1e-6}

    def __init__(self, op: CovarianceOperator, lo: float, hi: float, *, phi: ColouringRate,
                 sigma0: Tensor, evals: Tensor, evecs: Tensor, backend=None, K: int | None = None):
        super().__init__(op, lo, hi, phi=phi, K=K, backend=backend)
        d = int(evals.shape[0])
        if sigma0.shape != (d, d):
            raise ValueError(f"sigma0 must be ({d}, {d}); got {tuple(sigma0.shape)}")
        if evecs.shape != (d, d):
            raise ValueError(f"evecs must be ({d}, {d}); got {tuple(evecs.shape)}")
        # ⭐ REFUSE a low-precision eigensystem rather than silently computing in it. A float32
        # `evecs` would make every block below float32 -- including the `c_out²` subtraction and the
        # `psd_sqrt` guard -- while the class still advertised exact dense marginals.
        wrong = {n: t.dtype for n, t in (("sigma0", sigma0), ("evals", evals), ("evecs", evecs))
                 if t.dtype != torch.float64}
        if wrong:
            raise ValueError(
                f"{wrong} must be float64. The dense blocks are computed in float64 and cast to "
                f"op.dtype ({op.dtype}) on the way out; supplying them at the apply dtype would "
                "move the whole computation there, which is what this class exists to avoid.")
        self.dtype_apply = op.dtype
        # ⭐ Validate the eigensystem rather than trusting the caller. `SpectralBackend` does exactly
        # this (`UᵀU ≈ I` AND `S U ≈ U diag(λ)`) and an earlier revision advertised that backend while
        # accepting any pair of tensors — and the tests then constructed the class without one, so
        # nothing checked it anywhere. The second condition is the load-bearing one: `S u = λ u` alone
        # accepts a column-rescaled eigenvector, which leaves `U f(λ) Uᵀ` wrong by that scale squared.
        if not (torch.isfinite(evals).all() and torch.isfinite(evecs).all()):
            raise ValueError("evals/evecs contain non-finite entries")
        if float(evals.min()) <= 0.0:
            raise ValueError(f"S must be positive definite; got lam_min={float(evals.min()):.3e}")
        orth = float((evecs.transpose(-2, -1) @ evecs
                      - torch.eye(d, dtype=evecs.dtype, device=evecs.device)).abs().max())
        if orth > self.ORTHO_TOL:
            raise ValueError(f"evecs are not orthonormal (max |UᵀU - I| = {orth:.3e} > "
                             f"{self.ORTHO_TOL:.1e}, checked in float64)")
        # Correspondence is checked through `op.matvec` at the OPERATOR's dtype: that ties this
        # class to the same arithmetic the rest of the pipeline runs, which is the whole point of
        # checking against the operator rather than against a dense matrix.
        try:
            ctol = self.CORRESPONDENCE_TOL[op.dtype]
        except KeyError:
            raise ValueError(f"no correspondence tolerance registered for op.dtype={op.dtype}; "
                             f"known: {sorted(str(k) for k in self.CORRESPONDENCE_TOL)}") from None
        resid = op.matvec(evecs.to(op.dtype)) - (evecs * evals).to(op.dtype)
        rel = float(resid.abs().max() / max(1.0, float(evals.abs().max())))
        if rel > ctol:
            raise ValueError(f"eigensystem does not describe the operator (max |S U - U diag(l)| / "
                             f"|l|_max = {rel:.3e} > {ctol:.1e} at dtype {op.dtype}); a rescaled or "
                             "stale eigenbasis was supplied")
        # 🛑 BEFORE the symmetry and PSD clauses, not after. With a NaN in `sigma0`, `asym` is NaN
        # and `NaN > tol` is `False`, so the symmetry clause passes it; the PSD clause then only
        # raised because `eigvalsh` happened to error on this backend. Neither is a guard.
        if not torch.isfinite(sigma0).all():
            n_bad = int((~torch.isfinite(sigma0)).sum())
            raise ValueError(f"sigma0 has {n_bad} non-finite entries. Every comparison below "
                             "against a NaN evaluates False, so it would pass the symmetry and PSD "
                             "checks rather than fail them.")
        asym = float((sigma0 - sigma0.transpose(-2, -1)).abs().max())
        if asym > 1e-9 * float(sigma0.abs().max()):
            raise ValueError(f"sigma0 is not symmetric (max asymmetry {asym:.3e})")
        if float(torch.linalg.eigvalsh(_sym(sigma0)).min()) < -1e-9 * float(sigma0.abs().max()):
            raise ValueError("sigma0 is not positive semi-definite")
        self.sigma0 = _sym(sigma0)
        self.evals = evals
        self.evecs = evecs
        self._cache: "OrderedDict[float, dict[str, Tensor]]" = OrderedDict()

    # ------------------------------------------------------------------ dense spectral helpers
    def _f_of_S(self, vals: Tensor) -> Tensor:
        """``U diag(vals) Uᵀ`` — a dense function of ``S`` on the supplied eigensystem."""
        return (self.evecs * vals) @ self.evecs.transpose(-2, -1)

    def _blocks(self, F: float) -> dict[str, Tensor]:
        """Every dense quantity at one ``F`` — blocks **and** preconditioners — in ONE pass.

        🛑 Two defects this replaces, both real. (a) The cache was an unbounded dict keyed on the
        exact float ``F``; since ``sample_F_loguniform`` draws one ``F`` per minibatch from a
        continuous range, an 80,000-update run would have created ~80,000 entries of five dense
        ``d×d`` matrices — **~75 GB at d = 153**, not the ~150 MB I estimated from the sampler's
        400-step grid. (b) The preconditioners were computed *outside* the cache and rebuilt on every
        call, so one denoiser evaluation cost three redundant eigendecompositions.

        Now: bounded LRU of depth ``CACHE_SIZE``, and **two** eigendecompositions per ``F`` total —
        one for ``V_F`` (from which both inverses, *and* the positivity bracket, come) and one for
        ``c_out``'s square root.

        Computed in float64 throughout; the cached matrices are cast to ``dtype_apply`` on the way
        out, and only ``V_eigvals`` stays float64. See the class docstring for why the split is
        forced rather than tidy.
        """
        key = float(F)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not (F > 0.0):
            raise ValueError(f"requires F > 0 (the steep F->0 / singular V_0 case is deferred); got F={F}")
        e = torch.exp(-F * self.phi(self.evals))
        m = self._f_of_S(e.sqrt())
        sig = self._f_of_S(self.evals * (1.0 - e))
        V = _sym(m @ self.sigma0 @ m + sig)
        # 🛑 Finiteness BEFORE positivity, and before anything derived is cached. `float(w.min()) <=
        # 0.0` is `False` when `w.min()` is NaN, so a non-finite `V_F` was reported as positive
        # definite and its inverses cached — the same false-comparison defect repaired in the
        # constructor and in `psd_sqrt`, left unfinished here.
        if not torch.isfinite(V).all():
            raise ValueError(f"V_F has {int((~torch.isfinite(V)).sum())} non-finite entries at "
                             f"F={F}; the positivity check below would report NaN as positive "
                             "definite, because every comparison against NaN is False")
        w, U = torch.linalg.eigh(V)                       # ONE eigh serves V_inv and V_inv_half
        if not torch.isfinite(w).all():
            raise ValueError(f"V_F's spectrum is not finite at F={F}; the decomposition does not "
                             "describe the matrix")
        if float(w.min()) <= 0.0:
            raise ValueError(f"V_F is not positive definite at F={F} (lam_min={float(w.min()):.3e})")
        V_inv = (U * w.reciprocal()) @ U.transpose(-2, -1)
        s0 = self.sigma0
        c_skip = s0 @ m @ V_inv
        # ⚠️ SUBTRACTIVE. `Sigma_0 Sigma_F V^-1` is the same quantity ONLY when the factors commute;
        # off-eigenbasis it is not symmetric and its square root is not defined.
        c_out_sq = _sym(s0 - c_skip @ m @ s0)
        # ⭐ Everything above is float64. `psd_sqrt`'s negativity guard therefore runs on the
        # float64 `c_out_sq`, where its dtype-derived tolerance is ~5e-4x tighter than it would be
        # on the float32 subtraction -- which is the difference between a guard that can reject a
        # defective covariance and one that cannot (review finding F5).
        out = {"m": m, "sigma": sig, "V": V, "V_inv": V_inv,
               "V_inv_half": (U * w.rsqrt()) @ U.transpose(-2, -1),
               "c_skip": c_skip, "c_out_sq": c_out_sq, "c_out": psd_sqrt(c_out_sq)}
        if self.dtype_apply != torch.float64:
            out = {k: v.to(self.dtype_apply) for k, v in out.items()}
        # `V_eigvals` is deliberately NOT cast: it exists for `_require_positive_bracket`, which is
        # a positivity statement about the float64 matrix, not about its rounded image.
        out["V_eigvals"] = w
        # Nothing non-finite enters the cache: a cached NaN block would be returned unchecked on
        # every subsequent hit for this F, including to `_precond`.
        bad = sorted(k for k, v in out.items() if not torch.isfinite(v).all())
        if bad:
            raise ValueError(f"derived blocks {bad} are non-finite at F={F}; refusing to cache them")
        self._cache[key] = out
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return out

    # ------------------------------------------------------------------ overrides
    def v_marginal_dense(self, F: float) -> Tensor:
        """``V_F = m_F Σ_0 m_F + Σ_F``, dense and exact."""
        return self._blocks(F)["V"]

    def v_marginal(self, v: Tensor, F: float) -> Tensor:
        return self._blocks(F)["V"] @ v

    def v_inverse(self, v: Tensor, F: float, **kw) -> tuple[Tensor, dict]:
        """Dense exact solve. The ``info`` dict mirrors the Krylov route's contract.

        ``converged`` is ``True`` because there is nothing to converge — a distinction a seal must
        record, exactly as `ColoredEDM.c_in`'s dense route notes.
        """
        return self._blocks(F)["V_inv"] @ v, {"iters": 0, "converged": True, "relres": 0.0,
                                              "route": "dense-exact"}

    def v_inverse_half(self, v: Tensor, F: float, **kw) -> tuple[Tensor, dict]:
        return self._blocks(F)["V_inv_half"] @ v, {"iters": 0, "converged": True, "relres": 0.0,
                                                   "route": "dense-exact"}

    # ------------------------------------------------------------------ refused inheritances
    #: The base class's ``v_F(λ) = λ − τe^{−Fφ}`` is the COMMUTING marginal. Every sanctioned route
    #: into it is overridden above (``v_marginal``, ``v_inverse``, ``v_inverse_half``,
    #: ``_require_positive_bracket``; ``ColoredEDM.c_in`` by the paired EDM), so today nothing here
    #: reaches it — but that is a proof about four sealed call sites that a future upstream edit can
    #: add to without any signal on this side. Raising converts a proof that must be re-done after
    #: every upstream change into a structural one: a loud unsupported-operation error is strictly
    #: better than silent access to the wrong formula.
    _COMMUTING_ONLY = ("`v_F(lam) = lam - tau*exp(-F*phi)` assumes Sigma_0 = S - tau I, which is "
                       "exactly what this class exists to drop. Use `v_marginal_dense(F)` (the "
                       "actual V_F) or `_blocks(F)['V_eigvals']` (its spectrum).")

    def _vmarg_fn(self, F: float):
        raise NotImplementedError(f"MisspecifiedColoredForward has no scalar marginal band. "
                                  f"{self._COMMUTING_ONLY}")

    def _vmarg_matvec(self, F: float, K: int | None = None):
        raise NotImplementedError(f"MisspecifiedColoredForward has no matrix-free V_F apply. "
                                  f"{self._COMMUTING_ONLY}")

    def _v_bracket(self, F: float, ngrid: int = 513) -> tuple[float, float]:
        raise NotImplementedError(f"MisspecifiedColoredForward brackets the ACTUAL V_F. "
                                  f"{self._COMMUTING_ONLY}")

    def _require_positive_bracket(self, F: float, who: str) -> tuple[float, float]:
        """Guard on the ACTUAL ``V_F``, not on the scalar ``v_F(λ)`` the base class brackets.

        The base class's grid bracket describes ``S − τe^{−Fφ}``, which is not this arm's marginal;
        using it here would guard the wrong matrix.
        """
        if F <= 0.0:
            raise ValueError(f"{who} requires F > 0; got F={F}")
        # ⭐ Read `V_F`'s spectrum from the cache; NEVER recompute it. `_blocks` already ran the
        # eigendecomposition this bracket needs, and this guard runs inside every one of `c_skip`,
        # `c_out`, `c_out_sq`, `c_in` -- so a fresh `eigvalsh` here cost **three extra
        # decompositions per denoiser evaluation**, measured, on top of the two `_blocks` does. The
        # docstring above already claimed one; it was wrong until this line.
        w = self._blocks(F)["V_eigvals"]
        lam = torch.linspace(self.lo, self.hi, 513, dtype=torch.float64)
        phi_min = float(self.phi(lam).min())
        if not (phi_min == phi_min) or phi_min < -1e-12:
            raise ValueError(f"{who}: colouring phi must be a non-negative rate; min phi={phi_min:.3e}")
        return float(w.min()), float(w.max())


class MisspecifiedColoredEDM(ColoredEDM):
    """``ColoredEDM`` whose preconditioners are the dense, exact ones for a misspecified ``Σ_0``.

    Every override is a dense matrix apply on the paired
    :class:`MisspecifiedColoredForward`; the scalar ``bands.*_fn`` route is bypassed entirely,
    because none of these three is a function of ``λ`` once ``Σ_0`` stops commuting with ``S``.
    """

    def __init__(self, fwd: MisspecifiedColoredForward):
        if not isinstance(fwd, MisspecifiedColoredForward):
            raise TypeError("MisspecifiedColoredEDM requires a MisspecifiedColoredForward; a plain "
                            "ColoredForward would silently reinstate the Sigma_0 = S - tau I "
                            "assumption this class exists to remove")
        super().__init__(fwd, precond_backend=None, solve_backend=None)

    def _precond(self, F: float) -> dict[str, Tensor]:
        """The four preconditioners, read from the forward's single-pass cache.

        🛑 Never recompute here. An earlier version rebuilt `c_skip`, `c_out_sq` and a fresh
        `psd_sqrt` (an eigendecomposition) on EVERY call, and `_precond` is hit up to three
        times per denoiser evaluation. The test that pins this asserts object IDENTITY with
        the cache entry, not equality — which is what caught the first attempt at this fix
        silently not applying.
        """
        b = self.fwd._blocks(F)
        return {"c_skip": b["c_skip"], "c_out_sq": b["c_out_sq"], "c_out": b["c_out"],
                "c_in": b["V_inv_half"]}

    def c_skip(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_skip")
        return self._precond(F)["c_skip"] @ v

    def c_out(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_out")
        return self._precond(F)["c_out"] @ v

    def c_out_sq(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_out_sq")
        return self._precond(F)["c_out_sq"] @ v

    def c_in(self, v: Tensor, F: float, **solver_kw) -> Tensor:
        if solver_kw:
            raise ValueError(f"solver kwargs {sorted(solver_kw)} were passed, but c_in is a dense "
                             "exact inverse square root here and runs no solver")
        self.fwd._require_positive_bracket(F, "c_in")
        return self._precond(F)["c_in"] @ v
