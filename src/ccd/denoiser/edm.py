"""Colored-EDM preconditioners and the analytic Gaussian head (Phase V0).

The EDM (Karras et al.) preconditioners for the colored forward process, as matrix-free functions
of the single operator ``S``. Per spectral band ``λ`` (every operator commutes with ``S``, so this
per-band form is exact):

    c_skip(λ) = Σ_0 m_F / V_F  = (λ−τ) e^{−Fφ(λ)/2} / v_F(λ)      (the Wiener / MMSE head)
    c_out(λ)  = (Σ_0 Σ_F / V_F)^{1/2}                             (residual normaliser)
    c_in(λ)   = V_F^{-1/2}                                        (input whitener)

with ``v_F(λ) = λ − τ e^{−Fφ(λ)}``, ``m_F = e^{−Fφ/2}``, ``Σ_F = λ(1−e^{−Fφ})`` and ``Σ_0 = λ−τ``.

These are pure functions of the forward *moments* ``(m_F, Σ_F, V_F, Σ_0)``; the colouring ``φ`` enters
**only** through those moments. They are therefore invariant, in functional form, to the choice of
``φ`` — any colouring reuses these formulas unchanged, including a general graph-filter colouring
``φ(S)=Σ_k θ_k S^k`` *provided it is a positive polynomial in the same ``S``* (positivity keeps ``v_F``
and ``Σ_F`` valid; a polynomial in ``S`` keeps everything commuting). Nothing here is specialised to
the power law ``φ(λ)=λ^a``: the module reads ``φ`` off the :class:`ColoredForward` it wraps. The
non-power-law case is gated in ``tests/denoiser/test_edm.py``.

``c_skip`` and ``c_out`` are one :func:`apply_fun` each; ``c_in`` is the steep inverse ``V_F^{-1/2}``
(delegated to :func:`ccd.forward.sampling.whiten`), which is why the Phase-I4 rational/CG machinery is
a prerequisite. No eigendecomposition and no matrix exponential is ever formed.

.. warning::
   This paragraph used to call ``c_skip`` and ``c_out`` *smooth* applies. **Measured false on a
   rank-deficient operator** (fMRI ``d=360``, ``rank(Ĉ_n)=248``, ``τ=0.01``): at ``K=256`` the
   relative errors are ``c_skip`` 3.9e-1 and ``c_out`` 2.1e-2, decaying only algebraically, with the
   argmax at ``λ=τ`` for every clock including ``a=0``. ``c_out`` carries ``√(λ−τ)``, a branch point
   sitting exactly at the bracket's lower endpoint; ``c_skip`` runs ``0 → 1`` across a boundary layer
   of relative width ~8e-8 (see the class docstring below, which has documented that layer since
   before this was measured). Neither is a regression -- ``tests/denoiser/test_edm.py`` exercises the
   full-rank regime at ``K=400`` and ``F ≤ 4`` -- but on a ridge-dominated spectrum both must be
   applied densely rather than through :func:`apply_fun`.
   See ``docs/fmri_stage_ab_implementation_plan.md`` §2.1b.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.forward import bands
from ccd.forward.marginals import ColoredForward
from ccd.forward.sampling import whiten
from ccd.operators.chebyshev import apply_fun


class ColoredEDM:
    """Colored-EDM preconditioners ``c_skip, c_out, c_in`` and the analytic head, for a fixed
    :class:`ColoredForward` (hence a fixed colouring ``φ``).

    Generic in ``φ``: the preconditioners are functions of the forward marginals, so this works for
    any colouring the underlying ``ColoredForward`` carries. Approximation accuracy follows that
    ``ColoredForward``'s Chebyshev degree ``K``: the adaptive default (``K=None``) targets single
    precision — adequate for float32 EDM training — while an explicit ``K`` gives float64-tight
    preconditioners (``c_skip``/``c_out`` reach machine precision at a resolved ``K``). ``c_skip``
    develops a boundary layer near ``λ=τ`` (through ``1/v_F``), so it may want a larger ``K`` than the
    bare marginals ``m_t, Σ_t`` -- and on a ridge-dominated spectrum **no admissible ``K`` suffices**:
    the layer's relative width is ~8e-8 there and ``c_skip`` still fails at 3.9e-1 at ``K=256``. This
    sentence predates that measurement and was correct; it is the module docstring's "smooth applies"
    that was wrong. Route ``c_skip``/``c_out`` through a dense spectral apply in that regime
    (``docs/fmri_stage_ab_implementation_plan.md`` §2.1b, Remedy C).
    """

    def __init__(self, fwd: ColoredForward, *, precond_backend=None, solve_backend=None):
        """``precond_backend`` routes ``c_skip``, ``c_out`` **and** ``c_out_sq`` (Remedy C, §2.1b).

        ``None`` keeps the historical matrix-free path exactly, so nothing outside a benchmark that
        opts in changes. Pass a :class:`~ccd.operators.backends.SpectralBackend` on a ridge-dominated
        spectrum, where these are edge-limited and no admissible degree resolves them.

        ⚠️ **All three, and any sealed disposition table must say so.** ``c_out_sq`` is the smooth
        square that ``c_out`` takes the root of, so it is not itself edge-limited and an earlier
        table listed only ``c_skip``/``c_out``/terminal ``S^{1/2}``. But it *is* routed here, and a
        seal recording two dense callables while the code applies three is a provenance error
        whichever side is "right". Leaving ``c_out_sq`` matrix-free while ``c_out`` goes dense would
        also make ``c_out² ≠ (c_out)²`` by the Chebyshev error, which the EDM loss weight and the UQ
        baseline both read.

        ``c_in`` is **not** routed here: it is a Krylov solve, not a polynomial apply. A dense
        ``c_in`` is a separate decision about the *primary* path (see the benchmark factory), not a
        preconditioner-backend concern.
        """
        self.fwd = fwd
        self.precond_backend = precond_backend
        self.solve_backend = solve_backend

    def _apply_precond(self, fn, v: Tensor) -> Tensor:
        if self.precond_backend is not None:
            return self.precond_backend.apply(fn, v)
        return apply_fun(fn, self.fwd.op, v, self.fwd.lo, self.fwd.hi, self.fwd.K)

    # ---- scalar functions f(λ) fed to ApplyFun (smooth; no inverse, no matrix exponential) ----

    # Delegated to `ccd.forward.bands` (implementation plan §2.1): one definition, shared by the
    # matrix-free path here, the dense backend, and the audit. Arithmetic unchanged.

    def _cskip_fn(self, F: float):
        return bands.c_skip_fn(self.fwd.phi, self.fwd.op.tau, F)

    def _cout_sq_fn(self, F: float):
        return bands.c_out_sq_fn(self.fwd.phi, self.fwd.op.tau, F)

    def _cout_fn(self, F: float):
        return bands.c_out_fn(self.fwd.phi, self.fwd.op.tau, F)

    # --------------------------------------------------------------- operator applications

    def c_skip(self, v: Tensor, F: float) -> Tensor:
        """Apply the Wiener-head preconditioner ``c_skip(S) = Σ_0 m_F V_F^{-1}`` (smooth apply).

        Requires ``F > 0`` with ``V_F`` finite and positive-definite on ``[lo, hi]`` (the shared
        forward guard); the steep ``F→0`` / rank-deficient edge is out of scope for V0.
        """
        self.fwd._require_positive_bracket(F, "c_skip")
        return self._apply_precond(self._cskip_fn(F), v)

    def c_out(self, v: Tensor, F: float) -> Tensor:
        """Apply the residual normaliser ``c_out(S) = (Σ_0 Σ_F V_F^{-1})^{1/2}``.

        Note the ``Σ_0^{1/2}=(S-\\tau I)^{1/2}`` factor puts a ``√(λ−τ)`` branch point at ``λ=τ`` — the
        left end of the spectral bracket — so this sqrt apply converges only *algebraically* in the
        Chebyshev degree (~1e-6 relative at moderate ``K``, ample for float32 EDM training) rather than
        geometrically. This is the same ``λ=τ`` boundary that motivates the steep machinery; the
        *squared* form :meth:`c_out_sq` is smooth and reaches machine precision, and is what the EDM loss
        weight and the UQ baseline actually use.
        """
        self.fwd._require_positive_bracket(F, "c_out")
        return self._apply_precond(self._cout_fn(F), v)

    def c_out_sq(self, v: Tensor, F: float) -> Tensor:
        """Apply ``c_out²(S) = Σ_0 Σ_F V_F^{-1}`` (smooth; the Gaussian posterior covariance, exact).

        Geometric Chebyshev convergence — the accurate quantity for EDM loss weighting and the
        analytic UQ baseline (``Cov_Gauss(x_0 | x_t) = c_out²``), unlike the sqrt :meth:`c_out`.
        """
        self.fwd._require_positive_bracket(F, "c_out_sq")
        return self._apply_precond(self._cout_sq_fn(F), v)

    def c_in(self, v: Tensor, F: float, **solver_kw) -> Tensor:
        """Apply the input whitener ``c_in(S) = V_F^{-1/2}``.

        Two routes, and they are different *algorithms*, not different accuracies:

        * **Krylov (default)** — :func:`whiten`, a rational + multishift-CG solve. ``solver_kw``
          (``tol``, ``maxiter``, ``n_poles``, ``method``, ``require_converged``) pass through, and it
          refuses a non-converged solve by default.
        * **Dense** — ``U diag(v_F^{-1/2}) Uᵀ``, when ``solve_backend`` was supplied. Exact, and it
          cannot fail to converge because there is nothing to converge. This is the fMRI benchmark's
          primary configuration.

        The dense route makes ``require_converged`` vacuous rather than satisfied — a distinction the
        seal must record, since "no non-converged solve occurred" then says nothing about the solver.
        """
        if self.solve_backend is not None:
            v_min, _ = self.fwd._require_positive_bracket(F, "c_in")
            if solver_kw:
                raise ValueError(
                    f"solver kwargs {sorted(solver_kw)} were passed, but c_in is routed through the "
                    "dense backend, which runs no solver. Drop them or drop `solve_backend` -- "
                    "silently ignoring them would make a tolerance look honoured when it was unused."
                )
            vfn = self.fwd._vmarg_fn(F)
            return self.solve_backend.apply(lambda lam: vfn(lam).clamp_min(v_min) ** -0.5, v)
        return whiten(self.fwd, v, F, **solver_kw)

    def head(self, v: Tensor, F: float) -> Tensor:
        """Analytic Gaussian denoiser ``D_head(v, F) = c_skip(S) v = E[x_0 | x_F]``.

        The exact MMSE denoiser for Gaussian data — the EDM denoiser with the learned body ``F_θ`` set
        to zero — and hence the zero-parameter baseline a VNN body must beat on non-Gaussian data.
        """
        return self.c_skip(v, F)
