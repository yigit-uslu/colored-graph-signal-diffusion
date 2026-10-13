r"""Generic real-pole rational fitting of an arbitrary scalar function (edge-limited backends probe).

:mod:`ccd.operators.rational` supplies poles/weights for exactly one scalar function, ``y^{-1/2}``,
via the Zolotarev elliptic construction. Approximating a *different* function (``c_out``, ``c_skip``,
``q_sqrt``, ``terminal``) needs a general-purpose fitter, and none exists in this tree.
``scipy.interpolate.AAA`` is unavailable on SciPy 1.10.1 (added later), so this module hand-rolls the
AAA algorithm (Nakatsukasa, Sete, Trefethen 2018) from ``numpy.linalg.svd`` and
``scipy.linalg.eig`` — both present here.

AAA produces a barycentric rational interpolant; poles come from the ``(m+1)x(m+1)`` generalized
eigenvalue pencil of the barycentric form, and residues from ``N(pole)/D'(pole)`` evaluated on that
same barycentric data. The result is converted to a **partial-fraction form**
``f(y) approx c_inf + sum_i r_i/(y - p_i)``, matching what
:func:`ccd.operators.solvers.shifted_cg` consumes when poles are real and lie strictly below the
approximation bracket (``p_i < 0`` for a shift ``sigma_i = -p_i > 0``, mirroring
:func:`ccd.operators.rational.invsqrt_poles`'s convention exactly).

**This module does not claim every function fits.** AAA may return complex-conjugate pole pairs
(a real quadratic term simple-pole ``shifted_cg`` cannot consume) or real poles that are not safely
below the bracket. :func:`real_pole_rational` reports these plainly via a ``RationalFitResult`` rather
than silently discarding or forcing them — a fit that fails the real/negative-pole test is a genuine,
reportable stage-gate failure, not a bug to work around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

ScalarFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BarycentricFit:
    """AAA barycentric interpolant: ``r(z) = N(z)/D(z)``, ``N,D`` built from the support set."""

    nodes: np.ndarray       # (m,) support points z_j
    values: np.ndarray      # (m,) f(z_j)
    weights: np.ndarray     # (m,) barycentric weights w_j
    n_sample: int
    max_err: float          # sup|f - r| over the sampling grid at convergence
    converged: bool

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        diff = z[..., None] - self.nodes[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = self.weights[None, :] / diff
            num = terms @ self.values
            den = terms.sum(axis=-1)
            out = num / den
        exact = np.any(diff == 0.0, axis=-1)
        if np.any(exact):
            idx = np.argmin(np.abs(diff), axis=-1)
            out = np.where(exact, self.values[idx], out)
        return out


def aaa_fit(
    f: ScalarFn,
    lo: float,
    hi: float,
    *,
    tol: float = 1e-13,
    max_terms: int = 40,
    n_sample: int = 2000,
) -> BarycentricFit:
    """Hand-rolled AAA: greedily grow the support set until the sup error on a sample grid meets ``tol``.

    ``n_sample`` points span ``[lo, hi]`` **geometrically** (dense grid, not Chebyshev nodes — AAA is
    a *discrete* least squares method and needs an explicit sample set). A linear grid was tried first
    and rejected: for a wide-``kappa`` bracket it under-resolves the region near ``lo``, exactly where
    every target function in this comparison is edge-limited — AAA then converges on the (badly
    under-sampled) discrete set while the true continuous sup error near ``lo`` stays orders of
    magnitude worse (measured: linear sampling on ``y^{-1/2}``, ``kappa=1e4``, reported ``1e-11`` on its
    own sample grid but ``4.6e-4`` on an independent geometric grid; switching to geometric sampling
    throughout removed the gap). Each step picks the point of largest current residual as the next
    support node (greedy, the AAA default) and re-solves the barycentric weights by the SVD of the
    Loewner matrix restricted to the remaining (non-support) rows.
    """
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi; got lo={lo}, hi={hi}")
    z = np.geomspace(float(lo), float(hi), int(n_sample))
    fz = np.asarray(f(z), dtype=np.float64)
    if fz.shape != z.shape:
        raise ValueError(f"f must be evaluated elementwise; got shape {fz.shape} for input {z.shape}")

    R = np.full_like(fz, fz.mean())
    support_idx: list[int] = []
    weights = np.array([1.0])
    max_err = float(np.max(np.abs(fz - R)))

    for _ in range(max_terms):
        if max_err <= tol:
            break
        remaining = np.setdiff1d(np.arange(z.size), support_idx, assume_unique=False)
        if remaining.size == 0:
            break
        j_new = remaining[np.argmax(np.abs(fz[remaining] - R[remaining]))]
        support_idx.append(int(j_new))

        z_s, f_s = z[support_idx], fz[support_idx]
        rem = np.setdiff1d(np.arange(z.size), support_idx, assume_unique=False)
        if rem.size == 0:
            weights = np.ones(len(support_idx)) / len(support_idx)
            break
        loewner = (fz[rem, None] - f_s[None, :]) / (z[rem, None] - z_s[None, :])
        _, _, Vt = np.linalg.svd(loewner)  # matrix-free-ok: n_sample x m Loewner, m <= max_terms, never d
        weights = Vt[-1, :].conj()

        bary = BarycentricFit(z_s, f_s, weights, n_sample, float("inf"), False)
        R = bary.evaluate(z)
        R[support_idx] = f_s
        max_err = float(np.max(np.abs(fz - R)))

    converged = max_err <= tol
    return BarycentricFit(z[support_idx], fz[support_idx], weights, n_sample, max_err, converged)


@dataclass(frozen=True)
class RationalFitResult:
    """A partial-fraction rational fit: ``f(y) approx c_inf + sum_i residues[i]/(y - poles[i])``.

    ``all_real``: every extracted pole has |Im| below ``real_atol``.
    ``all_below_bracket``: every (real) pole satisfies ``pole <= -pole_margin`` (so the induced
    ``shifted_cg`` shift ``sigma = -pole >= pole_margin > 0``, strictly positive as that solver
    requires).
    ``usable``: ``all_real and all_below_bracket`` — the stage-gate condition itself.
    ``shifts``/``fit_weights``: the ``shifted_cg``-ready arrays when ``usable`` (empty otherwise).
    ``reconstruction_sup_rel_err``: the fit error **recomputed independently from the extracted
    poles/residues** (partial-fraction form), not read off the barycentric fit — catches a bug in
    pole/residue extraction that a barycentric-only check would miss.
    """

    barycentric: BarycentricFit
    poles: np.ndarray
    residues: np.ndarray
    c_inf: float
    all_real: bool
    all_below_bracket: bool
    usable: bool
    shifts: np.ndarray
    fit_weights: np.ndarray
    reconstruction_sup_rel_err: float
    n_poles: int
    real_atol: float
    pole_margin: float


def _extract_poles_residues(bary: BarycentricFit) -> tuple[np.ndarray, np.ndarray, float]:
    """Poles/residues of a barycentric AAA fit via the (m+1)x(m+1) generalized eigenvalue pencil.

    Standard AAA construction (Trefethen et al.): poles are the finite generalized eigenvalues of
    ``E v = lam B v`` with ``B = diag(0, 1, ..., 1)`` and ``E`` bordering ``diag(z_j)`` with the
    weights in its first row/column. Residues follow from ``N(pole)/D'(pole)``, both evaluated
    directly from the barycentric node/weight data (no re-fit).
    """
    m = bary.nodes.size
    B = np.eye(m + 1)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1))
    E[0, 1:] = bary.weights
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(bary.nodes)
    with np.errstate(all="ignore"):
        eigvals = scipy.linalg.eig(E, B, right=False)  # matrix-free-ok: (m+1)x(m+1) AAA pole pencil, never d
    poles = np.asarray(eigvals)  # matrix-free-ok: same pencil's output, not a new eigendecomposition
    poles = poles[np.isfinite(poles.real) & np.isfinite(poles.imag) & (np.abs(poles) < 1e14)]

    def D(z: np.ndarray) -> np.ndarray:
        diff = z[..., None] - bary.nodes[None, :]
        return (bary.weights[None, :] / diff).sum(axis=-1)

    def Dprime(z: np.ndarray) -> np.ndarray:
        diff = z[..., None] - bary.nodes[None, :]
        return -(bary.weights[None, :] / diff**2).sum(axis=-1)

    def N(z: np.ndarray) -> np.ndarray:
        diff = z[..., None] - bary.nodes[None, :]
        return ((bary.weights * bary.values)[None, :] / diff).sum(axis=-1)

    dprime = Dprime(poles)
    residues = np.where(np.abs(dprime) > 0, N(poles) / np.where(dprime == 0, 1.0, dprime), np.nan)
    c_inf = float((bary.weights * bary.values).sum() / bary.weights.sum())
    return poles, residues, c_inf


def real_pole_rational(
    f: ScalarFn,
    lo: float,
    hi: float,
    *,
    tol: float = 1e-6,
    max_terms: int = 40,
    n_sample: int = 2000,
    real_atol: float = 1e-6,
    pole_margin: float = 1e-10,
    validation_n: int = 20000,
) -> RationalFitResult:
    """Fit ``f`` on ``[lo, hi]`` by AAA, extract poles/residues, and test the ``shifted_cg`` gate.

    ``tol`` is the AAA sup-error stopping tolerance (barycentric fit, on the sample grid); the
    *reported* accuracy (``reconstruction_sup_rel_err``) is independently recomputed from the
    extracted partial-fraction form on a separate, denser validation grid (``validation_n`` points),
    per this line's standing practice of never trusting a number without re-deriving it from a more
    primitive representation.

    ``real_atol`` and ``pole_margin`` are the registered numerical floors for the usability gate:
    a pole counts as real if ``|Im(pole)| <= real_atol`` (absolute, since poles are O(bracket
    width) in magnitude and the bracket is registered in absolute units), and counts as "below the
    bracket" only if the induced shift clears ``pole_margin > 0`` — not just ``> 0`` — because
    ``shifted_cg`` treats a vanishingly small shift as numerically singular.
    """
    bary = aaa_fit(f, lo, hi, tol=tol * 1e-2, max_terms=max_terms, n_sample=n_sample)
    poles, residues, c_inf = _extract_poles_residues(bary)

    all_real = bool(np.all(np.abs(poles.imag) <= real_atol)) if poles.size else True
    shifts_signed = -poles.real
    all_below_bracket = bool(np.all(shifts_signed >= pole_margin)) if poles.size else True
    usable = all_real and all_below_bracket

    if usable and poles.size:
        shifts = shifts_signed
        fit_weights = residues.real
        z = np.geomspace(float(lo), float(hi), int(validation_n))
        exact = np.asarray(f(z), dtype=np.float64)
        recon = c_inf + (fit_weights[None, :] / (z[:, None] + shifts[None, :])).sum(axis=1)
        denom = np.abs(exact)
        denom = np.where(denom > 0, denom, 1.0)
        reconstruction_sup_rel_err = float(np.max(np.abs(recon - exact) / denom))
    else:
        shifts = np.empty(0)
        fit_weights = np.empty(0)
        reconstruction_sup_rel_err = float("nan")

    return RationalFitResult(
        barycentric=bary, poles=poles, residues=residues, c_inf=c_inf,
        all_real=all_real, all_below_bracket=all_below_bracket, usable=usable,
        shifts=shifts, fit_weights=fit_weights,
        reconstruction_sup_rel_err=reconstruction_sup_rel_err,
        n_poles=int(poles.size), real_atol=real_atol, pole_margin=pole_margin,
    )
