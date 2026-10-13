"""Stationary 1-D synthetic fields for the Mehler--Hermite spectrogram.

The spectrogram needs a target whose *frequency* axis and whose *eigen-band* axis are the same
axis. Stationarity delivers exactly that: the covariance is circulant, its eigenbasis is the
Fourier basis and its eigenvalues are the Wiener--Khinchin power spectral density (the
Cramér--Khinchin correspondence of the review). A band of frequencies **is** a set of
eigendirections of ``S``, so the note's ``(f, k)`` plane and the colouring's ``(λ, k)`` plane
coincide up to the fixed map ``λ_f = Ŝ(f) + τ``.

Two targets share one covariance path:

* **T0 — Gaussian substrate** ``G``: power-law PSD ``Ŝ(f) = Z (f² + f₀²)^{−η}``. Exactly Gaussian,
  so Mehler's ``Corr(He_k(g), He_k(y)) = r^k`` is exact and the diffused score is linear
  (``−V_F^{-1}x``) for every colouring — the analytic score oracle.
* **T1 — pointwise Hermite warp** ``X = Σ_k a_k He_k(G)/√k!`` normalised to unit variance.
  Genuinely skewed and excess-kurtotic; **node-local** non-Gaussianity (the stationary analogue of
  :class:`~ccd.denoiser.toy.BlockGMM`, i.e. the *in-model* regime for a node-domain body), still
  stationary, and with an autocovariance available in closed form
  (:func:`warped_psd`) because the pointwise map acts on the substrate correlation power-wise.

Conventions (following the spectrogram note):

* Fields are real, periodic, length ``d``; frequency lives on the ``rfft`` grid
  ``torch.fft.rfftfreq(d)``; the DC bin is excluded everywhere (``psd[0] = 0``).
* PSD entries are the **eigenvalues of the circulant covariance** in the DFT basis. Interior
  frequencies are doubly degenerate; :func:`mode_weights` carries those multiplicities, so
  ``var(x) = Σ_f w_f · psd_f`` exactly. Fields are normalised to population variance 1.
* Probabilists' Hermite ``He_k`` throughout (``E[He_j(g)He_k(g)] = δ_jk k!``) — never the
  physicists' ``H_k``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "mode_weights",
    "power_law_psd",
    "warped_psd",
    "gaussian_field",
    "hermite_e",
    "hermite_warp",
    "octave_bands",
    "band_variance",
    "band_mean_psd",
    "bandpass",
    "standardize",
]


# --------------------------------------------------------------------------- spectra


def mode_weights(d: int, *, dtype: torch.dtype = torch.float64, device="cpu") -> Tensor:
    """Multiplicity weights ``w_f`` on the rfft grid, ``(d//2+1,)``, with ``Σ_f w_f = 1``.

    Each interior frequency corresponds to **two** eigendirections of the circulant covariance
    (the sine/cosine pair), DC and — for even ``d`` — Nyquist to one. With these weights a
    band statistic is an exact population average over eigendirections:
    ``var(x) = Σ_f w_f psd_f`` and ``var(BP_b x) = Σ_{f∈b} w_f psd_f``.
    """
    w = torch.full((d // 2 + 1,), 2.0 / d, dtype=dtype, device=device)
    w[0] = 1.0 / d
    if d % 2 == 0:
        w[-1] = 1.0 / d
    return w


def power_law_psd(
    d: int,
    *,
    eta: float = 1.0,
    f0: float = 0.002,
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> Tensor:
    """Power-law PSD ``Ŝ(f) = Z (f² + f₀²)^{−η}`` on the rfft grid, DC zeroed, unit field variance.

    ``Z`` is fixed by ``Σ_f w_f Ŝ(f) = 1`` (:func:`mode_weights`), so a field drawn from this PSD
    has population variance exactly 1 — the condition Mehler's formulas assume. ``η`` sets how much
    spectral range the spectrogram spans (the eigenvalue span across the analysis bands grows like
    ``(f_max/f_min)^{2η}``); ``f₀`` is the low-frequency knee that keeps ``Ŝ`` bounded at DC.
    """
    f = torch.fft.rfftfreq(d, dtype=dtype, device=device)
    psd = torch.zeros_like(f)
    psd[1:] = (f[1:] ** 2 + float(f0) ** 2) ** (-float(eta))
    w = mode_weights(d, dtype=dtype, device=device)
    return psd / float((w * psd).sum())


def warped_psd(psd: Tensor, d: int, coeffs) -> Tensor:
    """PSD of the pointwise warp ``X = Σ_k a_k He_k(G)/√k!`` given the substrate PSD (exact).

    Mehler's covariance identity ``Cov[He_j(G_u), He_k(G_v)] = δ_jk k! C(u−v)^k`` makes the warped
    autocovariance a *power series in the substrate autocovariance*,
    ``C_X(u) = Σ_k (a_k²/‖a‖²) C_G(u)^k``, so the PSD follows by one inverse/forward transform pair.
    With ``C_G(0) = 1`` this gives ``C_X(0) = 1``: the warp is variance-preserving. Order ``k``
    contributes ``C_G^k``, whose spectrum is the ``k``-fold self-convolution of ``Ŝ_G`` — this is
    the spectral broadening of §2.6 of the note, i.e. high chaos order occupies broader frequency
    support.
    """
    a = torch.as_tensor(coeffs, dtype=psd.dtype, device=psd.device).reshape(-1)
    if a.numel() < 1:
        raise ValueError("coeffs must be non-empty (a_1, a_2, … for orders 1, 2, …)")
    wts = (a**2) / float((a**2).sum())
    cg = torch.fft.irfft(psd.to(torch.complex128 if psd.dtype == torch.float64 else torch.complex64), n=d)
    cx = torch.zeros_like(cg)
    for k, wk in enumerate(wts, start=1):
        cx = cx + wk * cg**k
    out = torch.fft.rfft(cx, n=d).real
    return torch.clamp(out, min=0.0)


# --------------------------------------------------------------------------- field generation


def gaussian_field(
    psd: Tensor,
    d: int,
    n: int = 1,
    *,
    generator: torch.Generator | None = None,
    standardise: bool = True,
) -> Tensor:
    """Draw ``n`` independent periodic stationary Gaussian fields with the given PSD, shape ``(d, n)``.

    The DFT coefficients of a stationary process satisfy ``E|X_f|² = d·psd_f`` and are independent
    across ``f``, with ``X_f`` real at DC and Nyquist; drawing them that way and inverting gives a
    field whose population covariance is exactly the circulant with spectrum ``psd``.

    ``standardise=True`` (default, and what the note mandates) rescales each realisation to exact
    zero mean and unit sample variance: ``He_k`` formulas assume unit variance, and a 1% variance
    error becomes a ``k``% error at order ``k``.
    """
    dtype, device = psd.dtype, psd.device
    m = d // 2 + 1
    scale = torch.sqrt(torch.as_tensor(float(d), dtype=dtype, device=device) * psd)
    re = torch.randn(m, n, generator=generator, dtype=dtype, device=device)
    im = torch.randn(m, n, generator=generator, dtype=dtype, device=device)
    # interior modes: complex with E|z|²=2 split evenly; DC/Nyquist: real with E z²=1
    z = torch.complex(re / math.sqrt(2.0), im / math.sqrt(2.0))
    z[0] = torch.complex(re[0], torch.zeros_like(re[0]))
    if d % 2 == 0:
        z[-1] = torch.complex(re[-1], torch.zeros_like(re[-1]))
    x = torch.fft.irfft(scale[:, None] * z, n=d, dim=0)
    return standardize(x) if standardise else x


def hermite_e(x: Tensor, k_max: int) -> Tensor:
    """Probabilists' Hermite ``He_0 … He_{k_max}`` of ``x``, stacked on a new leading axis.

    Uses the stable recurrence ``He_{k+1}(x) = x He_k(x) − k He_{k−1}(x)``. Never
    ``numpy.polynomial.hermite`` (those are the physicists' ``H_k``).
    """
    out = [torch.ones_like(x), x]
    for k in range(1, int(k_max)):
        out.append(x * out[-1] - k * out[-2])
    return torch.stack(out[: int(k_max) + 1], dim=0)


def hermite_warp(g: Tensor, coeffs=(1.0, 0.6, 0.4)) -> Tensor:
    """Pointwise Hermite warp ``X = Σ_k a_k He_k(G)/√k!`` normalised to unit variance.

    ``coeffs`` are ``(a_1, a_2, …)`` for orders ``1, 2, …``; the normaliser is ``‖a‖₂`` because the
    ``He_k/√k!`` are orthonormal under the standard Gaussian. The default ``(1.0, 0.6, 0.4)``
    gives genuine skewness and excess kurtosis while remaining a strictly increasing cubic
    (discriminant of ``f'`` is negative), hence a diffeomorphism — so the *undiffused* score of
    ``X`` is available in closed form by change of variables.
    """
    a = torch.as_tensor(coeffs, dtype=g.dtype, device=g.device).reshape(-1)
    he = hermite_e(g, a.numel())
    fact = torch.tensor(
        [math.sqrt(math.factorial(k)) for k in range(1, a.numel() + 1)], dtype=g.dtype, device=g.device
    )
    scaled = (a / fact).reshape(-1, *([1] * g.dim()))
    return (scaled * he[1:]).sum(dim=0) / float(torch.linalg.vector_norm(a))


# --------------------------------------------------------------------------- bands


def octave_bands(
    d: int, *, f_min: float = 0.004, n_bands: int = 6, width_octaves: float = 1.0,
    dtype: torch.dtype = torch.float64, device="cpu",
) -> tuple[Tensor, Tensor]:
    """Octave-spaced boxcar band masks in rfft space, ``(n_bands, d//2+1)`` bool, plus their centres.

    Band ``b`` is centred at ``f_b = f_min · 2^{b−1}`` and covers ``[f_b·2^{−W/2}, f_b·2^{W/2})``
    for width ``W`` octaves; centres are always one octave apart. DC is never included. Raises if
    the lowest band holds fewer than 8 modes or the top edge exceeds 0.45 (Nyquist effects), the two
    band-edge guards the spectrogram note specifies.

    ``width_octaves < 1`` narrows the bands without moving them. That matters for any statistic that
    is nonlinear across a band: on a ``Ŝ ~ f^{−2η}`` spectrum an octave-wide band spans a factor
    ``4^η`` in eigenvalue, so a band-aggregate correlation is an average over a range of ``λ`` and
    systematically *compresses* a measured spectral slope. Narrowing the bands is the control that
    separates that estimator effect from a real one.

    ``width_octaves > 1`` is needed to see **odd** cumulants at all. A third cumulant is a sum over
    frequency triples with ``f₁ + f₂ = f₃``; if all three must lie in ``[lo, hi)`` then ``f₃ ≥ 2·lo``
    forces ``hi > 2·lo``, i.e. **more than one octave**. At exactly one octave (``hi/lo = 2``) a
    boxcar band therefore has *no* bispectrum support and the band-passed skewness is identically
    zero however skewed the field is — measured here as ``−0.0000`` at ``hi/lo=2`` versus ``+0.53``
    (22.7σ) at ``hi/lo=4``. Even cumulants are unaffected (a quadruple can pair up), so an
    octave-band ``m=4`` test is fine. Bands wider than an octave overlap, which correlates estimates
    *across* bands but biases none of them.
    """
    if not (0.0 < width_octaves <= 4.0):
        raise ValueError(f"width_octaves must be in (0, 4]; got {width_octaves}")
    f = torch.fft.rfftfreq(d, dtype=dtype, device=device)
    centres = float(f_min) * 2.0 ** torch.arange(n_bands, dtype=dtype, device=device)
    half = 2.0 ** (float(width_octaves) / 2.0)
    lo = centres / half
    hi = centres * half
    masks = (f[None, :] >= lo[:, None]) & (f[None, :] < hi[:, None])
    masks[:, 0] = False
    n_low = int(masks[0].sum())
    if n_low < 8:
        raise ValueError(f"lowest band holds {n_low} modes (< 8); raise f_min or d")
    if float(hi[-1]) > 0.45:
        raise ValueError(f"top band edge {float(hi[-1]):.3f} > 0.45; lower f_min or n_bands")
    return masks, centres


def band_variance(psd: Tensor, masks: Tensor, d: int) -> Tensor:
    """Exact population band variances ``Var(BP_b x) = Σ_{f∈b} w_f psd_f``, shape ``(n_bands,)``.

    Pass ``psd = ones`` to get the white-noise band variances ``V_b^ε``, which under these weights
    equal the fraction of DFT modes in the band (the note asks that this be verified, not assumed —
    :func:`~tests.test_fields` does).
    """
    w = mode_weights(d, dtype=psd.dtype, device=psd.device)
    return (masks.to(psd.dtype) * (w * psd)[None, :]).sum(dim=1)


def band_mean_psd(psd: Tensor, masks: Tensor, d: int) -> Tensor:
    """Band-mean spectrum ``Ŝ_b`` — the multiplicity-weighted mean of ``psd`` over the band."""
    ones = torch.ones_like(psd)
    return band_variance(psd, masks, d) / band_variance(ones, masks, d)


def bandpass(x: Tensor, mask: Tensor, d: int) -> Tensor:
    """Boxcar band-pass of ``x`` (leading axis of length ``d``) by an rfft-space mask."""
    xh = torch.fft.rfft(x, dim=0)
    m = mask.to(xh.real.dtype)
    if xh.dim() > 1:
        m = m.reshape(-1, *([1] * (xh.dim() - 1)))
    return torch.fft.irfft(xh * m, n=d, dim=0)


def standardize(x: Tensor, dim: int = 0) -> Tensor:
    """Subtract the mean and divide by the (population) standard deviation along ``dim``."""
    xc = x - x.mean(dim=dim, keepdim=True)
    sd = xc.std(dim=dim, unbiased=False, keepdim=True)
    return xc / torch.clamp(sd, min=torch.finfo(x.dtype).tiny)
