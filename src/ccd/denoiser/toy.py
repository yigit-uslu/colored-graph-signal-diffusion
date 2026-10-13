"""Covariance-structured validation toys for the learned denoiser (Phase V2).

Two controlled non-Gaussian datasets in ℝ^d. The one that matters for the V2 gate is :class:`BlockGMM`; the
:class:`GMMinDCT` toy is kept as a documented **out-of-model control**.

**Why the distinction matters.** The VNN body applies its nonlinearity in the *node* domain, and a graph
filter can only reach directions that are functions of ``S`` (its eigenvectors). If the non-Gaussian direction
is a *delocalised* eigenvector (a low-frequency DCT mode, spread over all ``d`` nodes), a node-domain ``σ``
applied to ``U_j(U_jᵀx)`` sees the eigen-coordinate diluted by ``~1/√d`` — below where ``σ`` bends — so the
correction is linearised away. Empirically the body stays inert on such data even though a real (small) MMSE
gain exists. The in-model regime is therefore **node-local** non-Gaussianity along *localised* eigenvectors —
the CNN analogy done correctly (pixel-domain non-Gaussianity with spatial covariance structure), and the only
regime the matrix-free VNN can exploit (no eigendecomposition ⇒ no spectral-domain ``σ``).

:class:`BlockGMM` delivers that: ``x = B f + σ_local η`` with a per-block latent factor ``f_g`` that is
**bimodal**. Each block node's marginal is genuinely bimodal (node-domain non-Gaussianity, value ``O(f_g)`` not
diluted), while the block-shared factor makes ``S`` block-correlated — so the body must *block-average*
(a graph-filter operation, the factor eigenvectors are localised to ``|g|`` nodes) to estimate ``f_g``, then
apply the tanh, then redistribute. Both in-model **and** exercising the cross-node graph mixing.

Everything here is data generation (fixed indicator/cosine tables + matmuls + Gaussian/Rademacher draws) — no
eigendecomposition, no FFT, no solve — so it is safe under the matrix-free ``src/ccd`` guard.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ccd.operators.linop import CovarianceOperator


def dct_matrix(d: int, *, dtype: torch.dtype = torch.float64, device="cpu") -> Tensor:
    """Orthonormal DCT-II matrix ``W`` (``d×d``); row ``k`` is frequency mode ``k`` and ``W Wᵀ = I``."""
    j = torch.arange(d, dtype=dtype, device=device)
    k = torch.arange(d, dtype=dtype, device=device)
    W = torch.cos(math.pi * (2.0 * j[None, :] + 1.0) * k[:, None] / (2.0 * d))
    alpha = torch.full((d,), math.sqrt(2.0 / d), dtype=dtype, device=device)
    alpha[0] = math.sqrt(1.0 / d)
    return alpha[:, None] * W


class GMMinDCT:
    """Covariance-structured Gaussian-mixture toy in a DCT basis.

    Args:
        d: ambient dimension.
        rho: spectral-decay exponent of the per-axis variances ``s_j=(j+1)^{-ρ}`` (normalised ``s_0=1``).
        bimodal_axes: the DCT frequency indices ``J`` made bimodal (the S-aligned non-Gaussian directions).
        sep: bimodality strength ``r=μ_j/√s_j ∈ (0,1)``; ``μ_j=r√s_j``, ``σ_j²=s_j(1−r²)`` keep ``Var=s_j``.
        tau: ridge ``τ`` of the covariance operator ``S=Ĉ_n+τI``.
        dtype, device: tensor placement (pass ``device='cuda'`` for GPU pipelines).

    The mixture is mean-zero and symmetric (``±μ`` equally likely) ⇒ ``score(0)=0``, matching the body's
    origin symmetry.
    """

    def __init__(
        self,
        d: int = 48,
        *,
        rho: float = 1.0,
        bimodal_axes: tuple[int, ...] = (1, 2, 3),
        sep: float = 0.9,
        tau: float = 0.05,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        if not (0.0 < sep < 1.0):
            raise ValueError(f"sep must be in (0, 1); got {sep}")
        if any(a < 0 or a >= d for a in bimodal_axes):
            raise ValueError(f"bimodal_axes must be in [0, {d}); got {bimodal_axes}")
        self.d = int(d)
        self.tau = float(tau)
        self.dtype = dtype
        self.device = torch.device(device)
        self.bimodal_axes = tuple(int(a) for a in bimodal_axes)
        self.W = dct_matrix(d, dtype=dtype, device=device)                 # (d, d) rows = frequency modes

        j = torch.arange(d, dtype=dtype, device=device)
        s = (j + 1.0) ** (-float(rho))
        self.s = s / s[0]                                                  # per-axis variances (s_0 = 1)
        # per-axis bimodal split (variance-matched): μ_j = r√s_j, σ_j² = s_j(1−r²) on J; μ=0, σ²=s off J.
        self.mu = torch.zeros(d, dtype=dtype, device=device)
        self.sigma = torch.sqrt(self.s).clone()
        mask = torch.zeros(d, dtype=torch.bool, device=device)
        mask[list(self.bimodal_axes)] = True
        self.mu[mask] = float(sep) * torch.sqrt(self.s[mask])
        self.sigma[mask] = torch.sqrt(self.s[mask] * (1.0 - float(sep) ** 2))
        self._mask = mask

    @property
    def non_gaussian_axes(self) -> tuple[int, ...]:
        return self.bimodal_axes

    def _draw_dct(self, batch: int, generator: torch.Generator | None, gaussian: bool) -> Tensor:
        """Draw ``z`` (DCT coords), shape ``(d, batch)``.

        ``gaussian=True`` uses the **full** per-axis variance ``s_j`` on every axis (the variance-matched
        Gaussian reference); ``gaussian=False`` uses the within-mode std ``σ_j`` plus the ``±μ_j`` shift, so
        ``Var = σ_j²+μ_j² = s_j`` matches on every axis and the two share the same covariance ``S``.
        """
        eta = torch.randn(self.d, batch, generator=generator, dtype=self.dtype, device=self.device)
        if gaussian:
            return torch.sqrt(self.s)[:, None] * eta
        z = self.sigma[:, None] * eta
        signs = torch.randint(0, 2, (self.d, batch), generator=generator, device=self.device) * 2 - 1
        return z + signs.to(self.dtype) * self.mu[:, None]               # ±μ_j on bimodal axes (μ=0 else)

    def sample(self, batch: int, *, generator: torch.Generator | None = None, gaussian: bool = False) -> Tensor:
        """Draw ambient samples ``x = Wᵀ z`` of shape ``(d, batch)``.

        ``gaussian=True`` draws the variance-matched Gaussian reference (same ``S``, no bimodality).
        """
        return self.W.transpose(0, 1) @ self._draw_dct(batch, generator, gaussian)

    def make_operator(
        self, *, n: int = 4000, generator: torch.Generator | None = None, gaussian: bool = False
    ) -> CovarianceOperator:
        """Build ``S = Ĉ_n + τI`` from ``n`` samples (large ``n`` ⇒ ``U ≈ W``, tight S-alignment).

        The GMM and its Gaussian reference share the same population covariance, so either ``gaussian`` value
        yields (up to sampling noise) the same ``S`` — use the matching data for each gate.
        """
        X = self.sample(n, generator=generator, gaussian=gaussian)
        return CovarianceOperator(X, self.tau)


class BlockGMM:
    """Block/cluster factor GMM — the in-model V2 toy (node-local non-Gaussianity, block-correlated ``S``).

    ``x = B f + σ_local η`` where ``B`` (``d×G``) assigns each node to one of ``G`` contiguous blocks, ``f_g``
    is a per-block latent factor (**bimodal** on the chosen blocks), and ``η`` is per-node idiosyncratic
    noise. A block node's marginal ``x_i = f_{g(i)} + σ_local η_i`` is genuinely bimodal (node-domain), and the
    shared factor makes ``S`` block-correlated. The factor variances ``a_g²=(g+1)^{-ρ}`` are anisotropic.

    Args:
        d: ambient dimension (must be divisible by ``n_blocks``).
        n_blocks: number of blocks ``G``.
        rho: factor-variance decay ``a_g²=(g+1)^{-ρ}`` (normalised ``a_0²=1``).
        sep: bimodality strength ``r∈(0,1]`` on bimodal blocks (``μ_g=r a_g``, within-mode ``ς_g²=a_g²(1−r²)``
            ⇒ ``Var f_g = a_g²`` matches the Gaussian reference).
        idio_frac: idiosyncratic variance as a fraction of the mean factor variance
            (``σ_local² = idio_frac·⟨a_g²⟩``); smaller ⇒ stronger block structure.
        bimodal_blocks: which blocks are bimodal (default all ``G``).
        tau: ridge ``τ`` of ``S=Ĉ_n+τI``.

    Mean-zero and symmetric (``±μ_g``) ⇒ ``score(0)=0``. The GMM and its Gaussian reference share the same
    covariance ``S`` (variance-matched), so they differ only in higher cumulants along the block-factor modes.
    """

    def __init__(
        self,
        d: int = 48,
        n_blocks: int = 8,
        *,
        rho: float = 0.7,
        sep: float = 0.95,
        idio_frac: float = 0.2,
        bimodal_blocks: tuple[int, ...] | None = None,
        tau: float = 0.05,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        if d % n_blocks != 0:
            raise ValueError(f"d={d} must be divisible by n_blocks={n_blocks}")
        if not (0.0 < sep <= 1.0):
            raise ValueError(f"sep must be in (0, 1]; got {sep}")
        self.d = int(d)
        self.G = int(n_blocks)
        self.gsize = d // n_blocks
        self.tau = float(tau)
        self.dtype = dtype
        self.device = torch.device(device)
        self.bimodal_blocks = tuple(range(self.G)) if bimodal_blocks is None else tuple(int(b) for b in bimodal_blocks)
        if any(b < 0 or b >= self.G for b in self.bimodal_blocks):
            raise ValueError(f"bimodal_blocks must be in [0, {self.G}); got {self.bimodal_blocks}")

        g = torch.arange(self.G, dtype=dtype, device=device)
        a2 = (g + 1.0) ** (-float(rho))
        self.a2 = a2 / a2[0]                                              # per-block factor variances
        self.sigma_local = float(math.sqrt(idio_frac * float(self.a2.mean())))
        # per-block bimodal split (variance-matched): μ_g=r a_g, ς_g²=a_g²(1−r²) on J; μ=0, ς²=a_g² off J.
        self.mu = torch.zeros(self.G, dtype=dtype, device=device)
        self.varsig = self.a2.clone()
        bmask = torch.zeros(self.G, dtype=torch.bool, device=device)
        bmask[list(self.bimodal_blocks)] = True
        self.mu[bmask] = float(sep) * torch.sqrt(self.a2[bmask])
        self.varsig[bmask] = self.a2[bmask] * (1.0 - float(sep) ** 2)
        # B: (d, G) block-indicator (localised columns), and each node's block index.
        self.block_of = torch.arange(self.d, device=device) // self.gsize
        self.B = torch.zeros(self.d, self.G, dtype=dtype, device=device)
        self.B[torch.arange(self.d, device=device), self.block_of] = 1.0

    @property
    def non_gaussian_blocks(self) -> tuple[int, ...]:
        return self.bimodal_blocks

    def _draw_factors(self, batch: int, generator: torch.Generator | None, gaussian: bool) -> Tensor:
        """Draw latent factors ``f`` of shape ``(G, batch)``."""
        eta = torch.randn(self.G, batch, generator=generator, dtype=self.dtype, device=self.device)
        if gaussian:
            return torch.sqrt(self.a2)[:, None] * eta                    # Var f_g = a_g² on every block
        f = torch.sqrt(self.varsig)[:, None] * eta
        signs = torch.randint(0, 2, (self.G, batch), generator=generator, device=self.device) * 2 - 1
        return f + signs.to(self.dtype) * self.mu[:, None]              # ±μ_g on bimodal blocks

    def sample(self, batch: int, *, generator: torch.Generator | None = None, gaussian: bool = False) -> Tensor:
        """Draw ambient samples ``x = B f + σ_local η`` of shape ``(d, batch)``."""
        f = self._draw_factors(batch, generator, gaussian)
        idio = self.sigma_local * torch.randn(self.d, batch, generator=generator, dtype=self.dtype, device=self.device)
        return self.B @ f + idio

    def make_operator(
        self, *, n: int = 6000, generator: torch.Generator | None = None, gaussian: bool = False
    ) -> CovarianceOperator:
        """Build ``S = Ĉ_n + τI`` from ``n`` samples (the GMM and Gaussian reference share this ``S``)."""
        X = self.sample(n, generator=generator, gaussian=gaussian)
        return CovarianceOperator(X, self.tau)
