"""The colored-EDM denoiser ``D_θ`` and its training loss (Phase V2).

Wraps the V0 preconditioners (:class:`ColoredEDM`) and the V1 body (:class:`VNNBody`) into the EDM denoiser

    D_θ(x, F) = c_skip(S) x + c_out(S) · F_θ(c_in(S) x; F),

trained by the (colored) EDM denoising objective. The head-only special case ``F_θ ≡ 0`` recovers the exact
Gaussian MMSE denoiser ``c_skip(S) x`` — the zero-parameter Wiener baseline the learned body must beat.

**Loss weighting.** The faithful colored-EDM weight is the per-band operator ``c_out(S,F)^{-2}``, but it has a
pole at ``λ=τ`` (``Σ_0=λ−τ→0``) and would need a second steep inverse (``Ĉ_n^{-1/2}`` target-whitening). For
the V2 toy gate we use instead the **scalar** weight ``w(F)=1/⟨c_out²(·,F)⟩`` (the reciprocal mean posterior
variance over the spectrum) — bounded, cheap, and unit-variance-on-average per noise level. The full per-band
weighting is a documented later refinement. A single ``F`` is used per minibatch so the function-of-``S``
operators stay single-``F`` (per-sample ``F`` would need a batched-``F`` apply).

Gradients flow only through the body and the (linear) ``c_out`` apply; ``c_in``/``c_skip`` are
``θ``-independent and computed under ``no_grad`` (this also keeps autograd out of the ``c_in`` CG solve).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ccd.denoiser.edm import ColoredEDM
from ccd.denoiser.vnn import VNNBody
from ccd.forward.marginals import ColoredForward
from ccd.forward.sampling import forward_draw


class ColoredDenoiser(nn.Module):
    """The EDM denoiser ``D_θ = c_skip(S)x + c_out(S)F_θ(c_in(S)x;F)`` for the colored process.

    Args:
        edm: the V0 :class:`ColoredEDM` (preconditioners + analytic head).
        body: the V1 :class:`VNNBody` (learned residual ``F_θ``); must wrap the same
            :class:`ColoredForward` as ``edm``.
    """

    def __init__(self, edm: ColoredEDM, body: VNNBody):
        super().__init__()
        if body.fwd is not edm.fwd:
            raise ValueError("edm and body must wrap the same ColoredForward (same operator/colouring)")
        self.edm = edm
        self.body = body

    @property
    def fwd(self) -> ColoredForward:
        return self.edm.fwd

    def forward(self, x: Tensor, F: float, **c_in_kw) -> Tensor:
        """Full denoiser ``D_θ(x, F)``. ``c_in_kw`` pass through to the ``c_in`` (whitening) solve."""
        with torch.no_grad():                                    # θ-independent; keeps autograd out of CG
            u = self.edm.c_in(x, F, **c_in_kw)
            skip = self.edm.c_skip(x, F)
        return skip + self.edm.c_out(self.body(u, F), F)          # grad flows via body → c_out only

    def head(self, x: Tensor, F: float) -> Tensor:
        """Analytic Wiener head ``c_skip(S)x`` — the zero-parameter baseline (``F_θ ≡ 0``)."""
        return self.edm.head(x, F)


def edm_weight(fwd: ColoredForward, F: float, *, ngrid: int = 257, floor: float = 1e-8) -> float:
    """Scalar EDM loss weight ``w(F)=1/⟨c_out²(·,F)⟩`` (reciprocal mean posterior variance over the spectrum).

    Bounded and cheap (a scalar grid of the smooth ``c_out²``); grows as ``F→0`` (small residual, upweighted),
    so keep ``F_min`` bounded away from 0.
    """
    tau = fwd.op.tau
    lam = torch.linspace(fwd.lo, fwd.hi, ngrid, dtype=torch.float64)
    e = torch.exp(-F * fwd.phi(lam))
    cout_sq = (lam - tau) * lam * (1.0 - e) / (lam - tau * e)     # Σ_0 Σ_F / V_F
    return float(1.0 / (cout_sq.mean().clamp_min(floor)))


def sample_F_loguniform(F_min: float, F_max: float, *, generator: torch.Generator | None = None) -> float:
    """Draw a single ``F`` log-uniformly on ``[F_min, F_max]`` (one ``F`` per minibatch)."""
    if not (0.0 < F_min < F_max):
        raise ValueError(f"need 0 < F_min < F_max; got F_min={F_min}, F_max={F_max}")
    u = float(torch.rand((), generator=generator))
    return math.exp(math.log(F_min) + u * (math.log(F_max) - math.log(F_min)))


def cout_sq_bands(fwd: ColoredForward, F: float, evals: Tensor) -> Tensor:
    """Per-band posterior variance ``c_out²(λ,F) = Σ_0 Σ_F / V_F`` at the given eigenvalues.

    The same closed form :meth:`ccd.denoiser.edm.ColoredEDM._cout_sq_fn` applies as an operator, but
    evaluated on a supplied spectrum instead of Chebyshev nodes. Clamped at 0: ``edm.py`` does not
    clamp, and ``Σ_0 = λ−τ`` goes *negative* (≈−9e-13) on the exactly-null directions of a
    rank-deficient ``Ĉ_n``, which would otherwise make the whitened weight negative — i.e. the loss
    not positive semi-definite.
    """
    tau = fwd.op.tau
    e = torch.exp(-F * fwd.phi(evals))
    return ((evals - tau) * (evals * (1.0 - e)) / (evals - tau * e)).clamp_min(0.0)


def edm_loss_perband(
    denoiser: ColoredDenoiser,
    x0: Tensor,
    F: float,
    *,
    evals: Tensor,
    evecs: Tensor,
    mode: str = "whiten",
    rank: int | None = 32,
    scale: float = 1.0,
    eps: Tensor | None = None,
    generator: torch.Generator | None = None,
    c_in_kw: dict | None = None,
) -> Tensor:
    r"""**Opt-in** per-band colored-EDM loss ``⟨r, W(S) r⟩`` on the top-``rank`` spectral subspace.

    The faithful EDM weight is the per-band operator ``c_out^{-2}``, which :func:`edm_loss`
    deliberately replaces with the scalar ``1/⟨c_out²⟩``. This function supplies the faithful weight —
    but **support-restricted**, which on a rank-deficient covariance is not a refinement, it is a
    precondition:

    * ``Ĉ_n`` on the PBMC ``d=256`` panel has rank 235/256, so ``c_out² = Σ_0·Σ_F/V_F`` is **exactly
      zero** on 21 directions and below 1e-4 on 26. The unrestricted weight is therefore unbounded,
      and (before clamping) negative from round-off.
    * Measured, the unrestricted whitened weight puts **99.8 % of its mass on the bottom-67 near-null
      bands and 2e-5 on the top-11** — it would train the model almost entirely on the numerical null
      space. At ``rank=32`` the top-11 share is 0.13–0.25 (arm-dependent) while **99.898 % of the
      between-class mass is retained**, because that mass is concentrated in the top few modes.

    So ``rank`` is not a nuisance knob: it decides whether the objective charges for the modes that
    carry the signal. It is registered, not tuned.

    Modes (a 3-level factor for the loss axis; ``scalar`` is here only so a factorial can run through
    one code path — it is exactly :func:`edm_loss`):

    * ``"scalar"`` — the existing scalar weight on the **full** ambient residual.
    * ``"uniform"`` — uniform weight on the retained bands, times the same scalar ``edm_weight(F)``.
      Isolates *support restriction* from *band reallocation*: the F-profile is preserved exactly.
    * ``"whiten"`` — the faithful ``1/c_out²`` on the retained bands.

    ``scale`` is a single **global, arm- and F-independent** constant to bring the loss into the
    optimiser's usual range. A *per-F* rescaling would reintroduce exactly the scalarisation defect
    this function exists to remove, so it is deliberately not offered.

    Args:
        evals: eigenvalues of ``S`` (descending), shape ``(d,)``.
        evecs: matching eigenvectors, shape ``(d, d)`` (column ``k`` ↔ ``evals[k]``).
        rank: keep the top-``rank`` bands; ``None`` keeps all ``d`` (only sane on a full-rank ``Ĉ_n``).

    Gradients flow to the body through the residual only: ``W`` is a function of the spectrum and
    ``φ``, never of ``θ``, and is detached — an undetached weight is minimised by ``φ→∞`` and would
    silently corrupt any learnable-``φ`` run.
    """
    fwd = denoiser.fwd
    if eps is None:
        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
    x_F = forward_draw(fwd, x0, eps, F)
    D = denoiser(x_F, F, **(c_in_kw or {}))
    r = D - x0

    if mode == "scalar":                                         # ≡ edm_loss, for factorial parity
        return float(scale) * edm_weight(fwd, F) * r.pow(2).mean()

    if evals.ndim != 1 or evecs.shape[0] != evals.shape[0] or evecs.shape[1] < evals.shape[0]:
        raise ValueError(f"need evals (d,) and evecs (d,d); got {tuple(evals.shape)}, {tuple(evecs.shape)}")
    k = evals.shape[0] if rank is None else int(rank)
    if not (0 < k <= evals.shape[0]):
        raise ValueError(f"rank must be in (0, {evals.shape[0]}]; got {rank}")

    with torch.no_grad():                                        # W is θ-independent by construction
        lam = evals[:k].to(dtype=x0.dtype, device=x0.device)
        if mode == "whiten":
            c2 = cout_sq_bands(fwd, F, lam)
            if not bool((c2 > 0).all()):
                raise ValueError(
                    f"c_out² vanishes on the retained support at F={F}: min={float(c2.min()):.3e}. "
                    "Lower `rank` — the support must exclude the null space of Ĉ_n.")
            w = 1.0 / c2
        elif mode == "uniform":
            w = torch.full((k,), edm_weight(fwd, F), dtype=x0.dtype, device=x0.device)
        else:
            raise ValueError(f"mode must be 'scalar', 'uniform' or 'whiten'; got {mode!r}")
        U = evecs[:, :k].to(dtype=x0.dtype, device=x0.device)

    r_eig = U.transpose(0, 1) @ r                                # (k, batch) — grad flows through r
    return float(scale) * (w[:, None] * r_eig.pow(2)).mean()


def edm_loss(
    denoiser: ColoredDenoiser,
    x0: Tensor,
    F: float,
    *,
    eps: Tensor | None = None,
    generator: torch.Generator | None = None,
    weight: float | None = None,
    c_in_kw: dict | None = None,
) -> Tensor:
    """Colored-EDM denoising loss ``w(F)·E‖D_θ(x_F,F) − x_0‖²`` at a single ``F``.

    Draws ``x_F = m_F x_0 + Σ_F^{1/2} ε`` (fresh ``ε`` if not given), runs the denoiser, and returns the
    weighted mean-squared denoising error (a scalar tensor with grad to the body params).
    """
    fwd = denoiser.fwd
    if eps is None:
        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
    x_F = forward_draw(fwd, x0, eps, F)                          # θ-independent forward draw
    D = denoiser(x_F, F, **(c_in_kw or {}))
    w = edm_weight(fwd, F) if weight is None else float(weight)
    return w * (D - x0).pow(2).mean()
