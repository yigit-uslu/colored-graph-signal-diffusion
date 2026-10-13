"""Scalar VP (variance-preserving) DDPM noise schedule.

A standard discrete-time VP-DDPM schedule: the forward kernel is the homoscedastic
``q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)`` with ``ᾱ_t = ∏_{s≤t}(1-β_s)``. In ambient ``ccd``
this serves as the **scalar ``a=0`` (whitened-VP) reference**: at colouring power ``a=0`` the
colored forward mean is the isotropic ``e^{-F/2} I`` and the per-band SNR is ``λ``-independent,
so the colored construction must reduce to exactly this schedule under ``F = -log ᾱ_t``. That
reduction is gated by ``tests/forward/test_marginals.py::test_scalar_a0_reduces_to_vpschedule``;
see also :mod:`ccd.forward`. The ``predict_start_from_noise`` / ``q_posterior`` helpers are the
standard DDPM reverse-step utilities.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def make_beta_schedule(
    kind: str,
    num_steps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> Tensor:
    """β_t schedule. ``kind`` ∈ {``linear``, ``quadratic``, ``cosine``}.

    ``cosine`` is the improved-DDPM (Nichol & Dhariwal) ``ᾱ_t = cos²(((t/T)+s)/(1+s)·π/2)``
    parametrisation, returned as the implied per-step β (clipped to ``[0, 0.999]``).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}")
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    if kind == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float64) ** 2
    if kind == "cosine":
        t = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
        abar = torch.cos(((t / num_steps) + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        abar = abar / abar[0]
        betas = 1 - abar[1:] / abar[:-1]
        return betas.clamp(0.0, 0.999)
    raise ValueError(f"kind must be one of linear/quadratic/cosine; got {kind!r}")


def _extract(a: Tensor, t: Tensor, ndim: int) -> Tensor:
    """Gather ``a`` at integer timesteps ``t`` (B,) and right-broadcast to ``ndim`` dims."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (ndim - 1)))


class VPSchedule(nn.Module):
    """Precomputed VP-DDPM coefficients; buffers move with ``.to(device)``.

    All coefficients are stored as float32. ``t`` is a ``long`` tensor of timesteps in
    ``[0, num_steps)``.
    """

    def __init__(self, betas: Tensor):
        super().__init__()
        if betas.dim() != 1:
            raise ValueError(f"betas must be 1-D; got shape {tuple(betas.shape)}")
        # precompute on CPU (it is tiny) so construction is device-agnostic — e.g. when `betas`
        # arrives on GPU from a map_location'd checkpoint; the module's `.to(device)` moves buffers.
        betas = betas.detach().to(torch.float64).cpu()
        if float(betas.min()) < 0 or float(betas.max()) >= 1:
            raise ValueError("betas must lie in [0, 1)")
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        posterior_var = betas * (1.0 - abar_prev) / (1.0 - abar)

        def reg(name: str, x: Tensor) -> None:
            self.register_buffer(name, x.to(torch.float32))

        self.num_steps = int(betas.shape[0])
        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_cumprod", abar)
        reg("alphas_cumprod_prev", abar_prev)
        reg("sqrt_alphas_cumprod", abar.sqrt())
        reg("sqrt_one_minus_alphas_cumprod", (1.0 - abar).sqrt())
        reg("sqrt_recip_alphas_cumprod", (1.0 / abar).sqrt())
        reg("sqrt_recipm1_alphas_cumprod", (1.0 / abar - 1.0).sqrt())
        reg("posterior_variance", posterior_var)
        reg("posterior_log_variance_clipped", posterior_var.clamp_min(1e-20).log())
        reg("posterior_mean_coef1", betas * abar_prev.sqrt() / (1.0 - abar))
        reg("posterior_mean_coef2", (1.0 - abar_prev) * alphas.sqrt() / (1.0 - abar))

    # --------------------------------------------------------------------- forward / inverse

    def q_sample(self, w0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Standard forward kernel ``w_t = √ᾱ_t w_0 + √(1-ᾱ_t) ε``."""
        nd = w0.dim()
        return (
            _extract(self.sqrt_alphas_cumprod, t, nd) * w0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, nd) * noise
        )

    def predict_start_from_noise(self, w_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Invert the kernel: ``ŵ_0 = √(1/ᾱ_t) w_t - √(1/ᾱ_t - 1) ε``."""
        nd = w_t.dim()
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, nd) * w_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, nd) * eps
        )

    def q_posterior(self, w0: Tensor, w_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Posterior ``q(w_{t-1} | w_t, w_0)`` → (mean, variance, log_variance_clipped)."""
        nd = w_t.dim()
        mean = (
            _extract(self.posterior_mean_coef1, t, nd) * w0
            + _extract(self.posterior_mean_coef2, t, nd) * w_t
        )
        var = _extract(self.posterior_variance, t, nd)
        logvar = _extract(self.posterior_log_variance_clipped, t, nd)
        return mean, var, logvar
