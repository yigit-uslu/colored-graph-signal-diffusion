"""Matvec-only spectral estimation for the colouring rule (matvec-colouring M1+M2).

The rule-based colouring schedule (:class:`ccd.forward.TiledRate`) needs only the
**sorted top-G eigenVALUES** of ``S = Ĉ_n + τI`` as PCHIP knot positions — not the
eigenvectors, and not the bulk. This module supplies that spectral input **matrix-free**,
so the whole recipe is "matvecs only, end to end":

* :func:`lanczos` — single-vector Lanczos with full reorthogonalisation → the tridiagonal
  ``(α, β)``; its Ritz values are the isolated factor lines (M1).
* :func:`kpm_moments` — Hutchinson–Chebyshev (KPM) moments of the *scalar* spectral density
  via Rademacher probes and the three-term recurrence (M2); Jackson-damped density/CDF and
  analytic deflation of the detected lines.
* :class:`SpectralModel` — the hybrid { bracket; isolated lines ``(λ_g, m_g/d)``; deflated
  Chebyshev-density bulk } consumed by ``TiledRate.from_spectral_model``. It carries two
  constructors: :meth:`from_matvec` (this module's estimators) and :meth:`from_eigenvalues`
  (the ``eigh`` **oracle baseline**, which merely *receives* already-computed eigenvalues so
  no eigendecomposition ever enters ``src`` — the guard stays intact).

**No-eig discipline.** Nothing here calls ``torch.linalg.eigh`` / ``eigvalsh`` / ``matrix_exp``
on ``S``. The only eigensolve is :func:`scipy.linalg.eigh_tridiagonal` on the tiny
``m×m`` (``m ≈ 40``) Lanczos tridiagonal — an ``O(m²)`` scalar solve, not an ``O(d³)`` dense
decomposition of ``S`` (scipy is already used in ``src``; cf. ``forward/rate.py``,
``operators/rational.py``). Every routine reports its **single-vector matvec column count**
(a block matvec on ``(d, B)`` costs ``B`` columns) so cost budgets are honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ccd.operators.linop import CovarianceOperator

# --------------------------------------------------------------------------- shift helpers


def _shift_ab(lo: float, hi: float) -> tuple[float, float]:
    """Affine map ``x = a λ + b`` sending ``[lo, hi] → [-1, 1]`` (Chebyshev domain)."""
    a = 2.0 / (hi - lo)
    b = -(hi + lo) / (hi - lo)
    return a, b


# --------------------------------------------------------------------------- M1: Lanczos


def lanczos(
    op: CovarianceOperator,
    *,
    m: int = 40,
    reorth: str = "full",
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Single-vector Lanczos on ``S`` — return ``(alpha, beta, matvec_cols)``.

    ``alpha`` (length ``k``) and ``beta`` (length ``k-1``) are the diagonal / off-diagonal
    of the symmetric tridiagonal ``T_k`` whose eigenvalues (Ritz values) approximate the
    extreme spectrum of ``S``. With ``reorth="full"`` every new Lanczos vector is
    re-orthogonalised (twice) against all previous ones, suppressing the spurious "ghost"
    Ritz copies that plague plain Lanczos — the price is storing the ``k`` basis vectors,
    fine at these sizes and flagged for scale via ``reorth``.

    ``matvec_cols`` counts single-vector applies of ``S`` (one per iteration).
    """
    d = op.shape[0]
    m = min(int(m), d)
    q = torch.randn(d, generator=generator, dtype=op.dtype, device=op.device)
    q = q / q.norm()

    Q = torch.empty(m, d, dtype=op.dtype, device=op.device)
    alpha = torch.empty(m, dtype=op.dtype, device=op.device)
    beta = torch.empty(m, dtype=op.dtype, device=op.device)

    op.reset_count()
    q_prev = torch.zeros_like(q)
    b_prev = 0.0
    k = 0
    for j in range(m):
        Q[j] = q
        w = op.matvec(q) - b_prev * q_prev
        a_j = torch.dot(q, w)
        w = w - a_j * q
        if reorth == "full":                                    # twice for numerical stability
            for _ in range(2):
                w = w - Q[: j + 1].transpose(0, 1) @ (Q[: j + 1] @ w)
        b_j = w.norm()
        alpha[j] = a_j
        k = j + 1
        if float(b_j) <= 1e-12 * float(op.tau) or j == m - 1:   # invariant subspace / done
            break
        beta[j] = b_j
        q_prev, q = q, w / b_j
        b_prev = float(b_j)

    a = alpha[:k].detach().to(torch.float64).cpu().numpy()
    b = beta[: k - 1].detach().to(torch.float64).cpu().numpy() if k > 1 else np.zeros(0)
    return a, b, op.matvec_count


def ritz_values(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ritz values (desc) and their tridiagonal-eigenvector last components ``|s_{m,i}|``.

    The Ritz residual ``‖S y_i − θ_i y_i‖ = β_last · |s_{m,i}|`` (Paige) needs no extra
    matvec — it is read off the last row of the tridiagonal eigenvectors.
    """
    from scipy.linalg import eigh_tridiagonal    # matrix-free-ok: m×m Lanczos tridiagonal, m≪d; scipy allowed in src

    if alpha.size == 1:
        return alpha.copy(), np.zeros(1)
    theta, S = eigh_tridiagonal(alpha, beta)     # matrix-free-ok: m×m Lanczos tridiagonal, m≪d (ascending)
    order = np.argsort(theta)[::-1]                              # descending
    last = np.abs(S[-1, :])[order]
    return theta[order].copy(), last.copy()


# --------------------------------------------------------------------------- line detection (shared)


def detect_lines(
    vals_desc: np.ndarray,
    lam_max: float,
    *,
    residual: np.ndarray | None = None,
    resid_tol: float = 1e-8,
    gap_tol: float = 0.1,
    g_max: int = 12,
) -> tuple[np.ndarray, int]:
    """Isolated top lines of a descending spectrum, and ``auto_G``.

    Walking from the largest value down, a value is accepted as an isolated factor line iff
    (i) its Ritz residual is below ``resid_tol·λ_max`` (converged; ``0`` for exact
    eigenvalues) **and** (ii) its **relative gap to the next value below**,
    ``(λ_i − λ_{i+1})/λ_i``, is at least ``gap_tol``. The first failure marks the top of the
    structureless bulk — everything from there down is one cluster (the rule does not tile it).

    Returns ``(lines_desc, auto_G)`` where ``lines_desc`` are the accepted line values.
    """
    v = np.asarray(vals_desc, dtype=float).reshape(-1)
    r = np.zeros_like(v) if residual is None else np.asarray(residual, dtype=float).reshape(-1)
    lines = []
    for i in range(min(g_max, v.size - 1)):
        converged = r[i] <= resid_tol * lam_max
        rel_gap = (v[i] - v[i + 1]) / max(v[i], 1e-300)
        if converged and rel_gap >= gap_tol:
            lines.append(v[i])
        else:
            break
    return np.asarray(lines, dtype=float), len(lines)


# --------------------------------------------------------------------------- M2: KPM


def jackson_weights(K: int) -> np.ndarray:
    """Jackson damping coefficients ``g_0..g_K`` (kills Gibbs ringing in the KPM density)."""
    k = np.arange(K + 1, dtype=float)
    Kp = K + 1.0
    return ((Kp - k) * np.cos(np.pi * k / Kp) + np.sin(np.pi * k / Kp) / np.tan(np.pi / Kp)) / Kp


def kpm_moments(
    op: CovarianceOperator,
    lo: float,
    hi: float,
    *,
    n_z: int = 16,
    k_m: int = 128,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Hutchinson–Chebyshev moments ``μ_k = (1/d) tr T_k(S̃)`` of the normalised spectral measure.

    ``S̃ = a S + b I`` maps ``spec(S) ⊆ [lo, hi]`` into ``[-1, 1]``. Rademacher probes
    ``z ∈ {±1}^d`` (a ``(d, n_z)`` block, one block matvec per Chebyshev step) drive the
    three-term recurrence ``T_{k+1} = 2 S̃ T_k − T_{k-1}``; ``μ_k`` is the probe-average of
    ``(1/d) zᵀ T_k(S̃) z``. Returns ``(mu, se, matvec_cols)`` with ``mu`` length ``k_m+1``,
    per-moment standard error ``se`` across probes, and ``matvec_cols = n_z · k_m``.
    """
    d = op.shape[0]
    a, b = _shift_ab(lo, hi)
    g = torch.Generator(device=op.device)
    if generator is not None:                                   # derive a seed we can re-key deterministically
        g = generator
    Z = (torch.randint(0, 2, (d, n_z), generator=g, device=op.device, dtype=op.dtype) * 2 - 1)

    def sh(W):
        return a * op.matvec(W) + b * W

    op.reset_count()
    per = np.empty((k_m + 1, n_z))                              # per-probe moment contributions
    W0 = Z
    per[0] = ((Z * W0).sum(0) / d).detach().to(torch.float64).cpu().numpy()
    if k_m >= 1:
        W1 = sh(Z)
        per[1] = ((Z * W1).sum(0) / d).detach().to(torch.float64).cpu().numpy()
        for kk in range(2, k_m + 1):
            W2 = 2.0 * sh(W1) - W0
            per[kk] = ((Z * W2).sum(0) / d).detach().to(torch.float64).cpu().numpy()
            W0, W1 = W1, W2

    mu = per.mean(1)
    se = per.std(1, ddof=1) / math.sqrt(n_z) if n_z > 1 else np.zeros(k_m + 1)
    return mu, se, op.matvec_count * n_z            # block call → n_z single-vector columns


def _damped(mu: np.ndarray) -> np.ndarray:
    """Apply Jackson damping to a raw moment vector."""
    return mu * jackson_weights(mu.size - 1)


def kpm_survival(mu: np.ndarray, lo: float, hi: float, lam: np.ndarray) -> np.ndarray:
    r"""Survival ``Ḡ(λ) = μ((λ, λ_max])`` from Jackson-damped KPM moments (analytic integral).

    With ``x = cosθ``, ``T_k(x)=cos kθ`` and density ``p_θ(θ) = (1/π)(ĝ_0 + 2 Σ_{k≥1} ĝ_k cos kθ)``
    (``ĝ_k = g_k μ_k``), the mass above ``λ`` integrates in closed form:
    ``Ḡ(λ) = (1/π)(ĝ_0 θ_0 + 2 Σ_{k≥1} ĝ_k sin(k θ_0)/k)``, ``θ_0 = arccos(a λ + b)``.
    """
    a, b = _shift_ab(lo, hi)
    gm = _damped(mu)
    x = np.clip(a * np.asarray(lam, dtype=float) + b, -1.0, 1.0)
    theta0 = np.arccos(x)                                       # θ_0 ∈ [0, π]; λ→λ_max ⇒ θ_0→0
    k = np.arange(1, gm.size)
    out = gm[0] * theta0
    out = out + 2.0 * (gm[k][None, :] * np.sin(theta0[:, None] * k[None, :]) / k[None, :]).sum(1)
    return np.clip(out / math.pi, 0.0, 1.0)


def kpm_cdf(mu: np.ndarray, lo: float, hi: float, lam: np.ndarray) -> np.ndarray:
    """KPM CDF ``F(λ) = μ((-∞, λ]) = μ_0 − Ḡ(λ)`` (``μ_0 ≈ 1`` for the full measure)."""
    return float(mu[0]) - kpm_survival(mu, lo, hi, lam)


def deflate_moments(
    mu: np.ndarray, lines: np.ndarray, mult: np.ndarray, d: int, lo: float, hi: float
) -> tuple[np.ndarray, float]:
    """Subtract the isolated lines analytically from the moments — no deflated matvecs.

    ``μ_k ← μ_k − (1/d) Σ_g m_g T_k(λ̃_g)``; returns ``(mu_bulk, bulk_mass)`` with
    ``bulk_mass = 1 − Σ_g m_g / d``.
    """
    a, b = _shift_ab(lo, hi)
    xg = np.clip(a * np.asarray(lines, dtype=float) + b, -1.0, 1.0)
    k = np.arange(mu.size)
    Tk = np.cos(k[:, None] * np.arccos(xg)[None, :])           # T_k(λ̃_g), shape (K+1, G)
    corr = (Tk * (np.asarray(mult, dtype=float) / d)[None, :]).sum(1)
    return mu - corr, 1.0 - float(np.asarray(mult, dtype=float).sum()) / d


# --------------------------------------------------------------------------- the hybrid model


def line_kurtosis(
    op: CovarianceOperator, lines, lo: float, hi: float, *, data=None,
    n_z: int = 8, width_frac: float = 0.4, k_max: int = 200,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Matvec-only per-line excess kurtosis of the data (M5, the band-importance probe specialised to detected lines).

    For each factor line ``λ_g`` a smooth spectral bump ``b_g(S̃)`` (a Gaussian in ``λ`` of width ``width_frac`` × the
    gap to the nearest line, applied by Chebyshev--Clenshaw) approximates the projector onto ``λ_g``'s eigenspace, so
    ``v_g=b_g(S̃)z`` points (up to a scalar) along the factor eigenvector; the excess kurtosis of the data projected on
    ``v_g``, ``\\langle(v_g^\\top x)^4\\rangle/\\langle(v_g^\\top x)^2\\rangle^2-3``, measures band ``g``'s non-Gaussian
    content. Averaged over ``n_z`` Gaussian probes. ``data`` defaults to ``op.X`` (the samples defining ``S``).

    Returns per-line excess kurtosis in descending-λ order (leptokurtic ⇒ ``>0``, platykurtic/bimodal ⇒ ``<0``).
    """
    from ccd.operators.chebyshev import apply_fun

    X = op.X if data is None else data
    lam = np.sort(np.asarray(lines, dtype=float).reshape(-1))[::-1]
    out = np.zeros(lam.size)
    for i in range(lam.size):
        c = float(lam[i])
        others = np.delete(lam, i)
        s = max(width_frac * float(np.min(np.abs(others - c))) if others.size else 0.1 * c, 1e-9)
        bump = lambda z, cc=c, ss=s: torch.exp(-0.5 * ((z - cc) / ss) ** 2)     # noqa: E731 (Gaussian band-pass)
        ks = []
        for _ in range(n_z):
            zz = torch.randn(op.d, generator=generator, dtype=op.dtype, device=op.device)
            v = apply_fun(bump, op, zz, lo, hi, k_max=k_max)                    # ≈ projector onto λ_g's eigenspace
            proj = X.transpose(-2, -1) @ v                                      # (n,) data coords along v ∝ factor f_g
            pc = proj - proj.mean()
            m2 = float((pc ** 2).mean())
            ks.append(float((pc ** 4).mean() / (m2 ** 2 + 1e-300) - 3.0))
        out[i] = float(np.mean(ks))
    return out


def kurtosis_weights(kurt, *, mode: str = "max") -> np.ndarray:
    """Turn per-line excess kurtosis into a normalised importance profile. ``max``→ ``max(κ,0)`` (leptokurtic target,
    the registered probe); ``abs``→ ``|κ|`` (target-agnostic). Degenerate (all ≤0) falls back to equal weights (≡C3)."""
    k = np.asarray(kurt, dtype=float)
    w = np.maximum(k, 0.0) if mode == "max" else np.abs(k)
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / max(w.size, 1)


@dataclass
class SpectralModel:
    """Hybrid spectral summary consumed by ``TiledRate.from_spectral_model``.

    Fields:
        lo, hi: enclosing bracket ``[lo, hi] ⊇ spec(S)``.
        lines: isolated factor eigenvalues (descending) — the rule's knot positions.
        mult: their multiplicities (``1`` for simple factor lines).
        d: ambient dimension.
        auto_G: number of lines the gap criterion resolved (may be < config G at the ceiling).
        mu, mu_se: KPM moments of the full measure and their per-moment SE (``None`` in atoms-only).
        bulk_mass: deflated bulk mass ``1 − Σ m_g/d`` (``None`` until deflation).
        backend: ``"matvec"`` or ``"eigh"``.
        matvec_cols: single-vector applies of ``S`` spent building the model (``0`` for eigh).
    """

    lo: float
    hi: float
    lines: np.ndarray
    mult: np.ndarray
    d: int
    auto_G: int
    mu: np.ndarray | None = None
    mu_se: np.ndarray | None = None
    bulk_mass: float | None = None
    backend: str = "matvec"
    matvec_cols: int = 0
    meta: dict = field(default_factory=dict)

    # ---- constructors -----------------------------------------------------

    @classmethod
    def from_eigenvalues(
        cls, evals, d: int, *, lo: float | None = None, hi: float | None = None,
        gap_tol: float = 0.1, g_max: int = 12,
    ) -> "SpectralModel":
        """Oracle baseline: package **already-computed** eigenvalues (no eigendecomposition here).

        Runs the *same* :func:`detect_lines` gap logic as the matvec path, so ``auto_G`` and the
        ceiling behaviour are backend-identical. The eigenvalues come from the caller's
        script-side ``eigh`` (the guard lives in ``src``; this only receives values).
        """
        ev = np.sort(np.asarray(evals, dtype=float).reshape(-1))[::-1]
        lam_max = float(ev[0])
        lo = float(ev[-1]) if lo is None else float(lo)
        hi = lam_max if hi is None else float(hi)
        lines, auto_G = detect_lines(ev, lam_max, residual=None, gap_tol=gap_tol, g_max=g_max)
        return cls(lo=lo, hi=hi, lines=lines, mult=np.ones(lines.size),
                   d=int(d), auto_G=auto_G, backend="eigh", matvec_cols=0)  # matrix-free-ok: label string, not a call

    @classmethod
    def from_matvec(
        cls, op: CovarianceOperator, *, g_max: int = 12, m: int | None = None,
        reorth: str = "full", n_z: int = 16, k_m: int = 128,
        resid_tol: float = 1e-8, gap_tol: float = 0.1,
        with_moments: bool = True, generator: torch.Generator | None = None,
        gen_kpm: torch.Generator | None = None,
    ) -> "SpectralModel":
        """Full matrix-free build: Lanczos lines + (optional) deflated KPM bulk."""
        d = op.shape[0]
        m = max(4 * g_max, 40) if m is None else int(m)
        alpha, beta, cols_l = lanczos(op, m=m, reorth=reorth, generator=generator)
        theta, last = ritz_values(alpha, beta)
        resid = (float(beta[-1]) if beta.size else 0.0) * last  # ‖S y − θ y‖ = β_last |s_last|
        lam_max = float(theta[0])
        lines, auto_G = detect_lines(theta, lam_max, residual=resid,
                                     resid_tol=resid_tol, gap_tol=gap_tol, g_max=g_max)
        # bracket: extreme Ritz values with a 1% margin each side (keeps KPM inside [-1,1])
        lo = float(theta[-1]) * (1.0 - 1e-2)
        hi = lam_max * (1.0 + 1e-2)
        sm = cls(lo=lo, hi=hi, lines=lines, mult=np.ones(lines.size), d=d,
                 auto_G=auto_G, backend="matvec", matvec_cols=cols_l)
        sm.meta = dict(ritz=theta, ritz_resid=resid, m=m)
        if with_moments:
            mu, se, cols_k = kpm_moments(op, lo, hi, n_z=n_z, k_m=k_m, generator=gen_kpm)
            sm.mu, sm.mu_se = mu, se
            _, sm.bulk_mass = deflate_moments(mu, lines, sm.mult, d, lo, hi)
            sm.matvec_cols += cols_k
        return sm

    # ---- rule inputs ------------------------------------------------------

    def top_g(self, G: int | None = None) -> np.ndarray:
        """Top-``G`` factor lines (descending) for :class:`~ccd.forward.TiledRate` (atoms mode).

        With ``G`` given (config override) returns exactly the top-``G`` detected lines; with
        ``G=None`` returns all ``auto_G`` resolved lines.
        """
        n = self.lines.size if G is None else int(G)
        if self.lines.size < n:
            raise ValueError(
                f"requested top-{n} lines but only {self.lines.size} resolved "
                f"(auto_G={self.auto_G}); spectrum may be at the recoverability ceiling"
            )
        return self.lines[:n].copy()

    def survival(self, lam) -> np.ndarray:
        """Importance survival ``Ḡ(λ)`` in **density** mode (Jackson KPM). Atoms mode is the rule itself."""
        if self.mu is None:
            raise ValueError("density-mode survival needs KPM moments (build with with_moments=True)")
        return kpm_survival(self.mu, self.lo, self.hi, np.asarray(lam, dtype=float))
