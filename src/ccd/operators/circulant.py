"""Population circulant covariance ``S = C + τI`` as an FFT matvec oracle (Mehler-spectrogram support).

:class:`~ccd.operators.linop.CovarianceOperator` supplies the *empirical* ``S = Ĉ_n + τI``, whose
rank is capped at the sample count ``n``. That cap is the wrong instrument for the Mehler--Hermite
spectrogram: the analysis wants ``d = 2^16`` spectral bands, so an empirical operator would be
rank-deficient by construction and the entire fine half of every panel would be a ``τ``-plateau
artifact rather than data (cf. the PBMC rank/ridge facts).

For **stationary** data the covariance is circulant and the eigenbasis is literally the Fourier
basis (the Cramér--Khinchin correspondence of the review), so the *population* covariance is
available in closed form and matrix-free at ``O(d log d)``:

    S v = irfft( (Ŝ(f) + τ) · rfft(v) ),      spec(S) = { Ŝ(f) + τ }.

This class is deliberately **duck-typed** to :class:`~ccd.operators.linop.CovarianceOperator`
(``matvec``/``tau``/``shape``/``dtype``/``device``/``matvec_count``), so it drops into
:func:`~ccd.operators.chebyshev.apply_fun`, :func:`~ccd.operators.bracket.spectrum_bracket` and
:class:`~ccd.forward.marginals.ColoredForward` with no changes anywhere. Unlike the empirical
operator it additionally exposes the **exact** spectrum (:attr:`evals`), which the spectrogram uses
for its theory overlay and as the exact per-mode reference channel — the Chebyshev path is then a
*cross-check* against that reference rather than the thing under test.
"""

from __future__ import annotations

import torch
from torch import Tensor


class CirculantOperator:
    """SPD circulant operator ``S = C + τI`` accessed through :meth:`matvec` (FFT, no matrix formed).

    Args:
        psd: ``(d//2 + 1,)`` non-negative power spectral density on the **rfft grid**
            (``torch.fft.rfftfreq(d)``); entry ``k`` is the eigenvalue of ``C`` at frequency
            ``k/d``. Must be real and non-negative (it *is* the spectrum of ``C``).
        d: field length. ``psd`` must have length ``d//2 + 1``.
        tau: ridge ``τ > 0`` (also the exact smallest eigenvalue of ``S`` when ``min psd = 0``).

    Like :class:`~ccd.operators.linop.CovarianceOperator` the instance counts its matvecs
    (:attr:`matvec_count`, :meth:`reset_count`) so cost budgets stay assertable.
    """

    def __init__(self, psd: Tensor, d: int, tau: float):
        psd = torch.as_tensor(psd)
        d = int(d)
        if psd.dim() != 1:
            raise ValueError(f"psd must be 1-D on the rfft grid; got shape {tuple(psd.shape)}")
        if psd.numel() != d // 2 + 1:
            raise ValueError(f"psd must have length d//2+1 = {d // 2 + 1} for d={d}; got {psd.numel()}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0; got {tau}")
        if bool((psd < 0).any()):
            raise ValueError("psd must be non-negative (it is the spectrum of the circulant C)")
        if psd.is_complex():
            raise ValueError("psd must be real (a symmetric circulant has a real spectrum)")
        self.psd = psd
        self.d = d
        self.tau = float(tau)
        self._matvec_count = 0

    # ------------------------------------------------------------------ CovarianceOperator surface

    @property
    def shape(self) -> tuple[int, int]:
        return (self.d, self.d)

    @property
    def dtype(self) -> torch.dtype:
        return self.psd.dtype

    @property
    def device(self) -> torch.device:
        return self.psd.device

    @property
    def matvec_count(self) -> int:
        return self._matvec_count

    def reset_count(self) -> None:
        self._matvec_count = 0

    def matvec(self, v: Tensor) -> Tensor:
        """Apply ``S`` to ``v`` of shape ``(d,)`` or ``(d, B)`` — one rfft/irfft pair, ``O(d log d)``."""
        if v.shape[0] != self.d:
            raise ValueError(f"v must have leading dim d={self.d}; got {tuple(v.shape)}")
        self._matvec_count += 1
        lam = (self.psd + self.tau).to(dtype=v.dtype, device=v.device)
        if v.dim() > 1:
            lam = lam.reshape(-1, *([1] * (v.dim() - 1)))
        return torch.fft.irfft(lam * torch.fft.rfft(v, dim=0), n=self.d, dim=0)

    # ---------------------------------------------------------------------------- exact spectrum

    @property
    def evals_rfft(self) -> Tensor:
        """Eigenvalues ``λ_f = Ŝ(f) + τ`` on the rfft grid — one entry per *frequency*, ``(d//2+1,)``.

        This is the natural axis for the spectrogram: band ``b`` is a set of rfft bins, and its
        eigenvalues are these entries. Note the interior frequencies are doubly degenerate as
        eigenvalues of ``S`` (sine/cosine pair) — use :attr:`evals` when a length-``d`` spectrum
        with correct multiplicities is required.
        """
        return self.psd + self.tau

    @property
    def evals(self) -> Tensor:
        """Full length-``d`` spectrum of ``S`` with correct multiplicities (ascending, for ``eigh`` parity).

        The DC bin and (for even ``d``) the Nyquist bin are simple; every interior frequency is
        doubly degenerate.
        """
        lam = self.evals_rfft
        interior = lam[1:-1] if self.d % 2 == 0 else lam[1:]
        parts = [lam[:1], interior, interior]
        if self.d % 2 == 0:
            parts.insert(1, lam[-1:])
        return torch.sort(torch.cat(parts)).values

    def bracket(self, *, inflate: float = 0.0) -> tuple[float, float]:
        """Exact spectrum bracket ``[λ_min, λ_max]`` (optionally inflated), no power iteration needed."""
        lam = self.evals_rfft
        lo, hi = float(lam.min()), float(lam.max())
        return lo * (1.0 - inflate), hi * (1.0 + inflate)

    def _dense(self, *, max_d: int = 1024) -> Tensor:
        """Materialise the dense ``d×d`` ``S`` — TEST/ORACLE ONLY (guarded to small d)."""
        if self.d > max_d:
            raise RuntimeError(f"_dense is test-only; d={self.d} exceeds max_d={max_d}")
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        return self.matvec(eye)
