"""Clock-coverage audit (the G0 gate) for the colouring family.

A band whose ridge-aware unit-SNR clock ``F*(λ)`` falls below the sampler floor ``F_min`` is
**annihilated**: it is noise-dominated at every ``F`` inside the sampler window, so it is never seen
partially noised during training and its content at read-out is a prior draw rather than a learned
one. It carries the correct second moment and no non-Gaussian structure. Symmetrically, a band with
``F* > F_max`` never reaches the terminal prior.

Neither failure is visible to an analytic-head gate — the head is exactly Gaussian, so it matches a
covariance envelope however many bands were annihilated — which is why this audit is a *design* gate
run before training rather than a diagnostic run after it. See
``docs/fmri_fixed_operator_benchmark_plan.md`` §4 and ``docs/fmri_a0_build_plan.md`` §7.

This module is a pure function of a **supplied** spectrum: it never touches an operator and never
performs a spectral decomposition, so it satisfies the matrix-free ``src/ccd`` guard without a
pragma. The caller supplies ``(evals, evecs)``, exactly as :class:`~ccd.forward.rate.TiledRate`
requires its factor eigenvalues to be supplied.

Two conventions are load-bearing and are enforced here rather than left to the caller:

* **Weights are data variance ``λ − τ``, not the operator eigenvalue ``λ``.** The two differ
  materially near the ridge, and "share of ``tr(Ĉ)``" means the former.
* **The near-ridge snap is symmetric.** A residual of ``+2.6e-17`` is exactly as spurious as one of
  ``−1.7e-14``; clamping only the negatives would retain the former as a live band and let the live
  mask depend on floating-point noise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["ARM_CLASSES", "CoverageRecord", "band_weights", "coverage", "g0_gate"]

TOL_NULL_REL = 1e-12
"""Default relative near-ridge snap: ``tol = TOL_NULL_REL * λ_max``."""

ARM_CLASSES = frozenset({"pinned", "flat"})
"""How a clock's range is expected to be constrained (R3/S1).

``"pinned"`` -- a two-point-pinned ``ShiftedPowerRate`` arm, which must realise the ladder's common
range ``R``. ``"flat"`` -- the ``a=0`` arm, whose ``φ`` is constant, so ``R = 1`` **by definition**
and nothing is pinned. The distinction has to be carried explicitly because ``CoverageRecord`` holds
no exponent, and it must not be recovered by parsing ``label``: a label is a display string, and a
gate that reads one silently changes behaviour when someone renames an arm.
"""


@dataclass(frozen=True)
class CoverageRecord:
    """Per-clock coverage of the sampler window ``[F_min, F_max]``.

    Shares are fractions of ``tr(Ĉ)`` and of ``E[QV] = tr(L Ĉ)`` respectively, and each triple
    ``(below, window, above)`` sums to 1 over the live bands up to floating-point error.
    """

    label: str
    arm_class: str
    n_live: int
    n_snapped: int
    n_below: int
    n_window: int
    n_above: int
    var_below: float
    var_window: float
    var_above: float
    qv_below: float
    qv_window: float
    qv_above: float
    phi_min: float
    phi_max: float
    clock_range: float

    def __post_init__(self) -> None:
        if self.arm_class not in ARM_CLASSES:
            raise ValueError(
                f"arm_class must be one of {sorted(ARM_CLASSES)}; got {self.arm_class!r}"
            )

    def as_dict(self) -> dict:
        return asdict(self)


def _snap(evals: np.ndarray, tau: float, tol: float) -> np.ndarray:
    """Data variance ``λ − τ`` with a **symmetric** snap to zero on ``|λ − τ| ≤ tol``.

    Raises if any band is more than ``tol`` below the ridge: that is genuine indefiniteness in the
    operator build, not floating-point noise, and must not be silently clamped away.
    """
    d = np.asarray(evals, dtype=float) - float(tau)
    bad = d < -tol
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} band(s) lie more than tol={tol:.3e} below the ridge tau={tau}; "
            f"most negative lambda-tau = {d.min():.3e}. This is operator indefiniteness, not "
            "round-off — fix the operator rather than widening the tolerance."
        )
    return np.where(np.abs(d) <= tol, 0.0, d)


def band_weights(
    evals: np.ndarray,
    evecs: np.ndarray,
    lap: np.ndarray,
    *,
    tau: float,
    tol_null_rel: float = TOL_NULL_REL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Variance and QV weights per band, plus the live mask and the snapped-band count.

    ``E[QV] = tr(L Ĉ) = Σ_k (λ_k − τ) · u_kᵀ L u_k`` for centred data, so band ``k`` contributes
    ``(λ_k − τ) · u_kᵀ L u_k`` — a quadratic form, unlike the variance weight.

    Returns ``(var_w, qv_w, live, n_snapped)`` with both weight vectors normalised to sum to 1.
    """
    evals = np.asarray(evals, dtype=float).reshape(-1)
    evecs = np.asarray(evecs, dtype=float)
    lap = np.asarray(lap, dtype=float)
    if evecs.shape[1] != evals.size:
        raise ValueError(f"evecs has {evecs.shape[1]} columns but evals has {evals.size} entries")
    if lap.shape != (evecs.shape[0], evecs.shape[0]):
        raise ValueError(f"lap must be {(evecs.shape[0], evecs.shape[0])}; got {lap.shape}")

    tol = float(tol_null_rel) * float(evals.max())
    v = _snap(evals, tau, tol)
    n_snapped = int((v == 0.0).sum())
    if v.sum() <= 0.0:
        raise ValueError("no band carries data variance after the near-ridge snap")

    rayleigh = np.einsum("ik,ij,jk->k", evecs, lap, evecs)
    qv = v * rayleigh
    if qv.sum() <= 0.0:
        raise ValueError("total E[QV] is non-positive; check that lap is PSD and matches evecs")
    return v / v.sum(), qv / qv.sum(), v > 0.0, n_snapped


def f_star(phi_vals: np.ndarray, evals: np.ndarray, *, tau: float) -> np.ndarray:
    """Ridge-aware unit-SNR clock ``F*(λ) = log(2 − τ/λ) / φ(λ)``.

    The exact inverse of :func:`ccd.forward.rate.snr`, whose prefactor ``(λ−τ)/λ`` makes ``F*``
    vanish at ``λ = τ``: a band holding only the ridge carries no signal at any ``F``. The ``τ → 0``
    form ``ln2/φ`` overstates the clock of every near-ridge band and must not be substituted.
    """
    evals = np.asarray(evals, dtype=float).reshape(-1)
    phi_vals = np.asarray(phi_vals, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.clip(2.0 - float(tau) / np.where(evals > 0.0, evals, np.nan), 1e-300, None)
        return np.log(inner) / phi_vals


def coverage(
    phi: Callable[[np.ndarray], np.ndarray],
    evals: np.ndarray,
    evecs: np.ndarray,
    lap: np.ndarray,
    *,
    tau: float,
    F_min: float,
    F_max: float,
    label: str = "",
    arm_class: str = "pinned",
    tol_null_rel: float = TOL_NULL_REL,
) -> CoverageRecord:
    """Bucket the live bands of one clock into below / inside / above the sampler window."""
    if not (0.0 < F_min < F_max):
        raise ValueError(f"need 0 < F_min < F_max; got F_min={F_min}, F_max={F_max}")
    evals = np.asarray(evals, dtype=float).reshape(-1)
    var_w, qv_w, live, n_snapped = band_weights(
        evals, evecs, lap, tau=tau, tol_null_rel=tol_null_rel
    )

    phi_vals = np.asarray(phi(evals), dtype=float).reshape(-1)
    if phi_vals.size != evals.size:
        raise ValueError(f"phi returned {phi_vals.size} values for {evals.size} eigenvalues")
    if not np.all(phi_vals > 0.0):
        raise ValueError("phi must be strictly positive on the spectrum")

    fs = np.where(live, f_star(phi_vals, evals, tau=tau), np.inf)
    below = live & (fs < F_min)
    above = live & (fs > F_max)
    window = live & ~below & ~above

    return CoverageRecord(
        label=label,
        arm_class=arm_class,
        n_live=int(live.sum()),
        n_snapped=n_snapped,
        n_below=int(below.sum()),
        n_window=int(window.sum()),
        n_above=int(above.sum()),
        var_below=float(var_w[below].sum()),
        var_window=float(var_w[window].sum()),
        var_above=float(var_w[above].sum()),
        qv_below=float(qv_w[below].sum()),
        qv_window=float(qv_w[window].sum()),
        qv_above=float(qv_w[above].sum()),
        phi_min=float(phi_vals.min()),
        phi_max=float(phi_vals.max()),
        clock_range=float(phi_vals.max() / phi_vals.min()),
    )


def g0_gate(
    records: Iterable[CoverageRecord],
    *,
    max_share: float = 5e-3,
    max_spread: float = 5e-3,
    expect_R: float | None = None,
    rtol_R: float = 1e-6,
) -> tuple[bool, list[str]]:
    """The G0 decision rule: bounded annihilation, bounded *differential* annihilation, and range.

    The first two clauses matter and the second is the operative one. Annihilation that is uniform
    across clocks is a shared offset; annihilation that differs between clocks enters the clock
    contrast directly and is a confound.

    The **range clause** (R3) is the third. Without it a ladder built at the wrong ``R`` passes G0
    silently — and a common ``R`` is the one property the two-point pin exists to guarantee, the
    property that makes ``Δ_order`` an order contrast rather than a joint order-and-range contrast.
    ``clock_range`` is already measured and carried, so this is a criterion, not a new measurement.

    **The asymmetry that makes it easy to get wrong:** ``a=0`` has ``R = 1`` by definition and does
    **not** share the pinned range, so it must never be tested against ``expect_R``. That carve-out
    keys off ``CoverageRecord.arm_class``, never off a parsed label.

    Applies to colored clocks only — a VP arm has no ``φ`` and hence no ``F*``; its numerical
    adequacy is a solver question, not a coverage question.

    Returns ``(passed, reasons)``; ``reasons`` is empty exactly when ``passed``.
    """
    recs = list(records)
    if not recs:
        raise ValueError("g0_gate needs at least one CoverageRecord")
    if expect_R is not None and not (expect_R > 1.0):
        raise ValueError(f"expect_R must exceed 1 for a pinned ladder; got {expect_R}")
    reasons: list[str] = []

    for r in recs:
        if r.arm_class == "flat":
            # R = 1 exactly: phi is constant, so max/min is a ratio of identical floats.
            if r.clock_range != 1.0:
                reasons.append(
                    f"{r.label or '<unlabelled>'}: arm_class='flat' but clock range is "
                    f"{r.clock_range:.12g}, not exactly 1 — phi is not constant on this spectrum"
                )
        elif expect_R is not None:
            if abs(r.clock_range / expect_R - 1.0) > rtol_R:
                reasons.append(
                    f"{r.label or '<unlabelled>'}: clock range {r.clock_range:.6g} differs from the "
                    f"pinned R={expect_R:.6g} by {abs(r.clock_range / expect_R - 1.0):.3e} "
                    f"(rtol {rtol_R:.1e})"
                )

    for r in recs:
        if r.var_below > max_share:
            reasons.append(
                f"{r.label or '<unlabelled>'}: annihilated variance share {r.var_below:.4%} "
                f"exceeds {max_share:.4%}"
            )
        if r.qv_below > max_share:
            reasons.append(
                f"{r.label or '<unlabelled>'}: annihilated E[QV] share {r.qv_below:.4%} "
                f"exceeds {max_share:.4%}"
            )

    for name, vals in (("variance", [r.var_below for r in recs]),
                       ("E[QV]", [r.qv_below for r in recs])):
        spread = max(vals) - min(vals)
        if spread > max_spread:
            reasons.append(
                f"clock-to-clock {name} annihilation spread {spread:.4%} exceeds {max_spread:.4%}"
            )

    return (not reasons), reasons


def format_table(records: Sequence[CoverageRecord]) -> str:
    """Fixed-width audit table for the A0 measurements record."""
    head = (f"{'clock':>14} {'R':>12} {'live':>6} {'below':>6} {'window':>7} {'above':>6} "
            f"{'var%':>9} {'QV%':>9}")
    lines = [head, "-" * len(head)]
    for r in records:
        lines.append(
            f"{r.label:>14} {r.clock_range:>12,.4g} {r.n_live:>6} {r.n_below:>6} "
            f"{r.n_window:>7} {r.n_above:>6} {100 * r.var_below:>8.4f}% {100 * r.qv_below:>8.4f}%"
        )
    return "\n".join(lines)
