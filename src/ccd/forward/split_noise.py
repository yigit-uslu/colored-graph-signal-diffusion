r"""Split noise draws: ``N(0, S·b(S))`` without ever forming ``√λ`` (Remedy A, plan §2.1b).

The forward corruption needs ``N(0, Σ_F)`` with ``Σ_F = λ(1−e^{−Fφ})``, and the ancestral reverse
step needs ``N(0, q)`` with ``q = λ·g``. Both carry a bare ``λ``, so the direct route applies
``√(λ·b)`` — and that ``√λ`` is a branch point a distance ``τ`` below the bracket, which no
admissible Chebyshev degree resolves on a ridge-dominated spectrum (1.5e-4 at K=256, decaying
algebraically).

Functions of ``S`` commute, so the factor splits **exactly**:

    b^{1/2}(S) · N(0, S)  ~  N(0, b^{1/2} S b^{1/2})  =  N(0, S·b(S))

and ``N(0, S)`` needs no spectral function at all — it is drawn in closed form from the data factor,

    y  =  Xc z / √(n−1)  +  √τ w,      z ~ N(0, I_n),  w ~ N(0, I_d)
    Cov(y)  =  Xc Xcᵀ/(n−1) + τI  =  Ĉ + τI  =  S

leaving only ``b^{1/2}`` — analytic, and spectral by K≈64 — for the operator path.

**Two traps this class exists to close.**

1. **``√(n−1)``, never ``√n``.** The data factor must reproduce ``Ĉ`` at ``ddof=1``, the same
   convention as the frozen ``C_all.npy`` and as the B1 operator normalisation. ``√n`` would draw
   from ``((n−1)/n)Ĉ + τI``.
2. **The RAW centred frames, never ``op.X``.** ``CovarianceOperator`` is built from
   ``Xop = Xc·√(n/(n−1))`` so that its unbiased-looking ``X Xᵀ/n`` matvec equals ``Ĉ`` at ddof=1.
   Passing ``op.X`` here *and* keeping ``√(n−1)`` would inflate the drawn covariance by ``n/(n−1)``
   — the same B1 error re-entering through the noise path. The constructor refuses it.

**Two generators, never one.** The ``(n, batch)`` factor block and the ``(d, batch)`` ridge block are
drawn from *different* streams (``corruption_factor``/``innovation_factor`` vs
``corruption_noise``/``reverse_innovation``). The shared ``(d, batch)`` block is consumed identically
by every arm including VP — which reads only that one — so cross-arm bit-pairing survives the split.

⚠️ What that pairing does and does not buy: the shared block enters scaled by ``√τ``, so only the
ridge fraction ``τ·tr b(S)/tr[S b(S)]`` of the innovation variance is common with VP — measured
0.990 % on ``a=0`` but **11.24 %** on ``ā=−0.50`` at ``F_min``. It prevents *stream drift*; it is not
a variance-reduction pairing, and no Δ_color error bar may be claimed as paired on its basis.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.forward import bands
from ccd.forward.rate import ColouringRate

__all__ = ["SplitNoise"]


class SplitNoise:
    """Draws ``N(0, S·b(S))`` by splitting off an exact ``N(0, S)`` data-factor draw."""

    def __init__(self, Xc: Tensor, tau: float, backend, *, phi: ColouringRate,
                 operator_factor: Tensor | None = None, op=None, rtol: float = 1e-6):
        """``Xc`` is ``(d, n)`` **raw centred frames**; ``backend`` applies ``b^{1/2}(S)``.

        ⚠️ **The B1 check is only as strong as what the caller supplies.** An earlier version claimed
        the operator factor was refused "by construction"; it was not — ``operator_factor`` is
        optional, so a caller who omits it gets no check at all. Two levels are available and the
        benchmark factory must use the stronger:

        * ``operator_factor`` — refuses ``Xc`` *equal to* the operator's array. Cheap, but only
          catches the exact mistake.
        * ``op`` — **verifies the implied covariance**: ``Xc Xcᵀ/(n−1) + τI`` must equal what the
          operator applies, to ``rtol``. This binds the draw's law to the operator itself, so any
          rescaling, a different centring, or a stale copy of the frames is caught, not just the one
          known error.
        """
        if Xc.ndim != 2:
            raise ValueError(f"Xc must be (d, n); got {tuple(Xc.shape)}")
        d, n = Xc.shape
        if n < 2:
            raise ValueError(f"need n >= 2 frames for a ddof=1 factor; got n={n}")
        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0; got {tau}")

        if operator_factor is not None and operator_factor.shape == Xc.shape:
            if torch.allclose(Xc.to(torch.float64), operator_factor.to(torch.float64)):
                raise ValueError(
                    "SplitNoise was handed the OPERATOR's factor, not the raw centred frames. "
                    "`op.X` is already scaled by sqrt(n/(n-1)) (B1); combining it with the "
                    "sqrt(n-1) denominator here inflates the drawn covariance by n/(n-1). Pass the "
                    "raw centred frames."
                )

        self.Xc, self.tau, self.backend, self.phi = Xc, float(tau), backend, phi
        self.d, self.n = int(d), int(n)

        if op is not None:
            self._verify_implied_covariance(op, rtol)

    def _verify_implied_covariance(self, op, rtol: float) -> None:
        """``Xc Xcᵀ/(n−1) + τI  ==  S`` as the operator applies it — by matvec, on a probe basis.

        The draw's covariance is exactly this matrix, so checking it ties the noise path to the
        operator rather than to a convention someone remembered. `d` matvecs, once per run.
        """
        d = self.d
        I = torch.eye(d, dtype=op.dtype, device=op.device)
        implied = (self.Xc.to(op.dtype).to(op.device) @ self.Xc.to(op.dtype).to(op.device).T) \
            / (self.n - 1) + self.tau * I
        applied = op.matvec(I)
        num = float(torch.linalg.norm(implied - applied))
        den = float(torch.linalg.norm(applied))
        rel = num / den if den > 0 else num
        if not (rel <= rtol):
            raise ValueError(
                f"the frames handed to SplitNoise do not imply the operator's covariance: "
                f"||Xc Xc^T/(n-1) + tau I  -  S||/||S|| = {rel:.3e} > rtol={rtol:.1e}. "
                f"A ratio near {self.n / (self.n - 1):.6f} means the OPERATOR's factor was passed "
                "instead of the raw centred frames (B1); other values mean a different centring, a "
                "stale copy, or a rescaling."
            )

    # ---------------------------------------------------------------- the exact N(0, S) half

    def draw_prior(self, batch: int, *, g_factor: torch.Generator,
                   g_shared: torch.Generator) -> Tensor:
        """``N(0, S)`` in closed form: ``Xc z/√(n−1) + √τ w``. No spectral function involved."""
        z, w = self.draw_blocks(batch, g_factor=g_factor, g_shared=g_shared)
        return (self.Xc @ z) / (self.n - 1) ** 0.5 + (self.tau ** 0.5) * w

    # ---------------------------------------------------------------- the b^{1/2}(S) half

    def _draw(self, band_fn, batch: int, *, g_factor: torch.Generator,
              g_shared: torch.Generator) -> Tensor:
        return self.backend.apply(band_fn, self.draw_prior(batch, g_factor=g_factor,
                                                           g_shared=g_shared))

    def draw_forward(self, F: float, batch: int, *, g_factor: torch.Generator,
                     g_shared: torch.Generator) -> Tensor:
        """``N(0, Σ_F)`` with ``Σ_F = λ·h(F)``, ``h = 1−e^{−Fφ}``."""
        return self._draw(bands.h_sqrt_fn(self.phi, F), batch,
                          g_factor=g_factor, g_shared=g_shared)

    def draw_ancestral(self, F_a: float, F_b: float, batch: int, *, g_factor: torch.Generator,
                       g_shared: torch.Generator) -> Tensor:
        """``N(0, q)`` with ``q = λ·g(F_a, F_b)``.

        Pair-indexed by construction — a single ``draw(F, …)`` cannot express this, which is why the
        ancestral step needs its own entry point rather than a reused forward one.
        """
        if not (F_a > F_b >= 0.0):
            raise ValueError(f"need F_a > F_b >= 0 for a reverse step; got {F_a} -> {F_b}")
        return self._draw(bands.g_sqrt_fn(self.phi, F_a, F_b), batch,
                          g_factor=g_factor, g_shared=g_shared)

    def draw_blocks(self, batch: int, *, g_factor: torch.Generator,
                    g_shared: torch.Generator) -> tuple[Tensor, Tensor]:
        """The two blocks, separately: ``(z of shape (n,batch), w of shape (d,batch))``.

        Exposed because the teeth-control needs the **shared** block itself, not merely a generator
        that has been advanced the same number of times. Cross-arm pairing is a statement about the
        `w` that each arm actually uses; "the next draw is aligned" is a weaker property that an
        implementation could satisfy while injecting a different `w`.
        """
        z = torch.randn(self.n, batch, generator=g_factor,
                        dtype=self.Xc.dtype, device=self.Xc.device)
        w = torch.randn(self.d, batch, generator=g_shared,
                        dtype=self.Xc.dtype, device=self.Xc.device)
        return z, w

    def consume_like(self, batch: int, *, g_factor: torch.Generator,
                     g_shared: torch.Generator) -> Tensor:
        """Advance both generators exactly as a draw would; **return the shared block**.

        For the §3.3 teeth-control, which runs a deliberately degraded route but must consume
        identical generator words so terminals and innovations still pair bit-for-bit. Returning
        `w` lets the caller *use* the same shared innovation rather than merely having burned the
        same number of words — and lets a test assert equality of the block, not just of what comes
        after it.
        """
        return self.draw_blocks(batch, g_factor=g_factor, g_shared=g_shared)[1]

    def __repr__(self) -> str:
        return f"SplitNoise(d={self.d}, n={self.n}, tau={self.tau:g}, backend={self.backend!r})"
