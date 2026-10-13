r"""Blind Chebyshev-degree audit: the smallest ``K`` at which every band function meets a tolerance.

``K`` is a single number shared by **every** arm — never an arm-dependent knob — so it must be
resolved once, before any timing depends on it, and recorded in the seal.

**Why this can be exact.** Both ``p_K(S)`` and ``f(S)`` are functions of ``S`` and hence diagonal in
its eigenbasis, so the operator error is the scalar error on the spectrum:

    err_K(f, arg, clock) = max_i |p_K(λ_i) − f(λ_i)| / max_i |f(λ_i)|
    K* = smallest ladder rung with  max over fns × args × clocks  of err_K  ≤  tol

No matvecs, no operator, no eigendecomposition — the spectrum comes **from the caller**, the same
contract as :mod:`ccd.forward.clock_audit`, which is why this module lives in ``src`` with no
pragma.

**Audit the object that runs.** The callables passed here must be the *same* factories the operator
path uses (:mod:`ccd.forward.bands`), not transcriptions. Auditing a re-typed copy audits the copy.

**The ladder is never silently extended.** Exhausting it raises: on the fMRI operator the *unsplit*
function set fails at every rung through 256 (``c_skip`` 3.9e-1, ``c_out`` 2.1e-2), and the correct
response was to fix the function set — the h-split and the dense dispositions — not to buy a larger
degree. See ``docs/fmri_stage_ab_implementation_plan.md`` §2.1b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

__all__ = ["DegreeRecord", "NoPassingDegreeError", "RungError", "audit_degree", "cheb_eval",
           "default_arg_grid"]

LADDER: tuple[int, ...] = (24, 32, 40, 48, 64, 96, 128, 192, 256)


@dataclass(frozen=True)
class RungError:
    """The worst cell at one ladder rung — reported for **every** rung, not only the passing one.

    §3.3's teeth-control quotes an audited defect size from this table, so a search that returned
    only its answer would leave that control with no calibration.
    """

    K: int
    rel_err: float
    worst_fn: str
    worst_arg: str
    worst_clock: str


class NoPassingDegreeError(ValueError):
    """No rung met tolerance — raised **with the completed table attached**.

    A failed audit is a measurement, not an absence of one: §2.1b-bis's whole finding is the shape
    of the error-vs-degree curve on the arm that does not resolve, and §3.3's teeth-control reads a
    defect size off a rung that never passed. A bare message would leave a caller with the best
    rung's number and nothing else, so callers that record a failure can serialise ``.table``
    exactly as they serialise a pass.

    It subclasses ``ValueError`` so existing handlers keep working, but callers should catch **this**
    and let other ``ValueError``s through: a malformed band function and an unresolvable one are
    different events, and only the second is a result.
    """

    def __init__(self, message: str, *, table: tuple["RungError", ...], ladder: tuple[int, ...],
                 tol: float):
        super().__init__(message)
        self.table = table
        self.ladder = ladder
        self.tol = tol
        self.best = min(table, key=lambda r: r.rel_err) if table else None


@dataclass(frozen=True)
class DegreeRecord:
    K: int
    worst_fn: str
    worst_arg: str
    worst_clock: str
    rel_err: float
    table: tuple[RungError, ...]

    def error_at(self, K: int) -> float:
        for row in self.table:
            if row.K == K:
                return row.rel_err
        raise KeyError(f"rung {K} is not in the audited ladder {[r.K for r in self.table]}")


def cheb_eval(coeffs: torch.Tensor, lam: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Evaluate a Chebyshev series at arbitrary points by the Clenshaw recurrence.

    The scalar twin of :func:`~ccd.operators.chebyshev.apply_fun`'s operator Clenshaw, and it must
    stay that: the audit's whole claim is that the scalar error equals the operator error, which
    holds only if both evaluate the same series the same way.
    """
    a, b = 0.5 * (hi - lo), 0.5 * (hi + lo)
    y = (lam - b) / a
    two_y = 2.0 * y
    d0 = torch.zeros_like(y)
    d1 = torch.zeros_like(y)
    for k in range(len(coeffs) - 1, 0, -1):
        d0, d1 = two_y * d0 - d1 + coeffs[k], d0
    return y * d0 - d1 + coeffs[0]


def default_arg_grid(F_min: float, F_max: float, budgets: Sequence[int] = (200, 400),
                     n_dense: int = 512) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    """Union of the geometric sampler grids, their interval midpoints, and a dense validation grid.

    Grid nodes alone do not cover the audit's obligation: training draws ``F`` **continuously**, so
    a degree resolved only at the sampler's own nodes says nothing about the loss. The midpoints
    catch the worst case *between* nodes, where a Chebyshev fit is least constrained.
    """
    singles: set[float] = set()
    pairs: set[tuple[float, float]] = set()
    for n in budgets:
        grid = np.geomspace(F_min, F_max, int(n) + 1)
        singles.update(float(f) for f in grid)
        singles.update(float(np.sqrt(a * b)) for a, b in zip(grid[:-1], grid[1:]))  # geometric mids
        pairs.update((float(a), float(b)) for a, b in zip(grid[1:], grid[:-1]))     # F_a > F_b
    singles.update(float(f) for f in np.geomspace(F_min, F_max, n_dense))
    return tuple(sorted(singles)), tuple(sorted(pairs))


def audit_degree(
    band_fns: Mapping[str, Callable[..., Callable]],
    pair_fns: Mapping[str, Callable[..., Callable]],
    evals: np.ndarray,
    *,
    lo: float,
    hi: float,
    clocks: Mapping[str, Callable],
    single_args: Sequence[float],
    pair_args: Sequence[tuple[float, float]],
    ladder: Sequence[int] = LADDER,
    tol: float = 1e-4,
    dtype: torch.dtype = torch.float64,
) -> DegreeRecord:
    """Smallest rung meeting ``tol`` across every (function × argument × clock).

    ``band_fns`` map a name to a factory ``(clock, F) -> (λ -> value)``; ``pair_fns`` to
    ``(clock, F_a, F_b) -> (λ -> value)``. Raises if no rung passes — the ladder is not extended.
    """
    if not band_fns and not pair_fns:
        raise ValueError("nothing to audit: both function maps are empty")
    if not clocks:
        raise ValueError("no clocks supplied")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be finite and positive; got {tol!r}")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"need a finite bracket with lo < hi; got [{lo}, {hi}]")
    rungs = [int(k) for k in ladder]
    if rungs != sorted(set(rungs)) or rungs[0] < 1:
        raise ValueError(f"ladder must be strictly increasing and positive; got {list(ladder)}")

    lam = torch.as_tensor(np.asarray(evals, dtype=float).reshape(-1), dtype=dtype)
    if not torch.isfinite(lam).all():
        raise ValueError("evals contains non-finite entries")

    # (name, arg-label, clock-name, λ→value) — built once and reused at every rung, so a rung's
    # error cannot differ from another's because the function was rebuilt differently.
    cases: list[tuple[str, str, str, Callable]] = []
    for cname, clock in clocks.items():
        for fname, factory in band_fns.items():
            for F in single_args:
                cases.append((fname, f"F={F:.6e}", cname, factory(clock, float(F))))
        for fname, factory in pair_fns.items():
            for F_a, F_b in pair_args:
                cases.append((fname, f"pair(F={F_a:.6e}->{F_b:.6e})", cname,
                              factory(clock, float(F_a), float(F_b))))

    from .chebyshev import cheb_coeffs

    if not cases:
        raise ValueError(
            f"the audit constructed ZERO cases from {len(band_fns)} band function(s), "
            f"{len(pair_fns)} pair function(s), {len(single_args)} single argument(s) and "
            f"{len(pair_args)} pair(s). An audit with nothing to check reports rel_err=0 at the "
            "first rung and 'passes' -- which is why this raises instead."
        )

    table: list[RungError] = []
    passing: DegreeRecord | None = None
    for K in rungs:
        worst = RungError(K=K, rel_err=0.0, worst_fn="", worst_arg="", worst_clock="")
        n_scored = 0
        for fname, arg, cname, fn in cases:
            exact = fn(lam)
            # Shape and finiteness are checked SEPARATELY from the zero test. Folding a non-finite
            # denominator into the `denom == 0` branch skipped an all-NaN function as though it were
            # identically zero on the spectrum -- a false pass, not a degenerate case.
            if exact.shape != lam.shape:
                raise ValueError(
                    f"{fname} at {arg} on clock {cname} returned shape {tuple(exact.shape)} for "
                    f"{tuple(lam.shape)} eigenvalues; the audit compares band-for-band"
                )
            if not torch.isfinite(exact).all():
                n_bad = int((~torch.isfinite(exact)).sum())
                raise ValueError(
                    f"{fname} at {arg} on clock {cname} is non-finite on {n_bad} of "
                    f"{exact.numel()} bands. A non-finite exact value cannot be audited; it must "
                    "not be silently skipped."
                )
            denom = float(exact.abs().max())
            if denom == 0.0:
                continue                      # identically zero on the spectrum: no relative error
            c = cheb_coeffs(fn, lo, hi, K, dtype=dtype, device=lam.device)
            err = float((cheb_eval(c, lam, lo, hi) - exact).abs().max()) / denom
            if not np.isfinite(err):
                raise ValueError(f"non-finite error for {fname} at {arg} on clock {cname}")
            n_scored += 1
            if err > worst.rel_err:
                worst = RungError(K=K, rel_err=err, worst_fn=fname, worst_arg=arg,
                                  worst_clock=cname)
        if n_scored == 0:
            raise ValueError(
                f"every one of the {len(cases)} case(s) was identically zero on the spectrum, so "
                "nothing was scored. A rel_err of 0 here means 'not measured', not 'exact'."
            )
        table.append(worst)
        if passing is None and worst.rel_err <= tol:
            passing = DegreeRecord(K=K, worst_fn=worst.worst_fn, worst_arg=worst.worst_arg,
                                   worst_clock=worst.worst_clock, rel_err=worst.rel_err,
                                   table=())

    if passing is None:
        best = min(table, key=lambda r: r.rel_err)
        raise NoPassingDegreeError(
            f"no rung in {rungs} meets tol={tol:.1e}. Best was K={best.K} at {best.rel_err:.3e} "
            f"(worst cell: {best.worst_fn} @ {best.worst_arg} on {best.worst_clock}). The ladder is "
            "NOT extended: a function set that cannot be resolved at an admissible degree must be "
            "fixed (split the draw, or apply it densely), not bought.",
            table=tuple(table), ladder=tuple(rungs), tol=float(tol),
        )
    return DegreeRecord(K=passing.K, worst_fn=passing.worst_fn, worst_arg=passing.worst_arg,
                        worst_clock=passing.worst_clock, rel_err=passing.rel_err,
                        table=tuple(table))


def format_table(rec: DegreeRecord) -> str:
    head = f"{'K':>5} {'rel_err':>11}  {'worst function':<14} {'clock':<10} argument"
    lines = [head, "-" * (len(head) + 18)]
    for r in rec.table:
        mark = " *" if r.K == rec.K else "  "
        lines.append(f"{r.K:>5} {r.rel_err:>11.3e}{mark} {r.worst_fn:<14} {r.worst_clock:<10} {r.worst_arg}")
    lines.append(f"\nK* = {rec.K}  (worst {rec.worst_fn} @ {rec.worst_arg} on {rec.worst_clock})")
    return "\n".join(lines)
