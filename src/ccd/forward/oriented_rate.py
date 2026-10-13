"""Both orientations of the two-point-pinned clock, kept OUT of ``rate.py``.

:class:`~ccd.forward.rate.ShiftedPowerRate` admits only ``a < 0``, and ``rate.py`` stays
byte-identical to ``main``'s: the fMRI seal gates hash that file's bytes, so ANY edit to it — a
docstring included — refuses every record sealed against it, whatever the arithmetic does (owner
ruling, 2026-10-12). The ``a > 0`` support that `77ff994` put into ``rate.py`` for the orientation
probe (hand-off #14 §1.1) therefore lives here, unchanged in its arithmetic:

* :func:`pinned_power_rate` — the constructor to call. ``a < 0`` returns ``rate.py``'s own
  ``ShiftedPowerRate``, CONSTRUCTED rather than re-implemented, so the registered path is ``main``'s
  class bit for bit; ``a > 0`` returns :class:`MirroredShiftedPowerRate`.
* :func:`orientation` and :func:`pinned_at` — at either sign; ``main``'s class has neither.
* :func:`generation_order` — ``rate.py``'s, called through unchanged. It exists so the warning below
  sits on the function its callers import.

🛑 **Which order a sign means is a statement in OPERATOR-EIGENVALUE coordinates.** On a
covariance-like ``S`` (large eigenvalue ↔ smooth mode) ``a<0`` is coarse→fine in graph frequency; on
a Laplacian-like ``S = L_sym + σI`` (large eigenvalue ↔ ROUGH mode) the same ``a<0`` is fine→coarse.
``rate.py``'s docstrings still say "coarsest"/"finest" and label the signs coarse→fine/fine→coarse;
those words are the covariance-like reading only. `scripts/gfs_arms.py` measures and records the
graph-frequency reading on every arm it builds, because the label alone misled an entire probe.
"""

from __future__ import annotations

import math

from torch import Tensor

from ccd.forward.rate import ColouringRate, ShiftedPowerRate
from ccd.forward.rate import generation_order as _rate_generation_order

__all__ = ["MirroredShiftedPowerRate", "generation_order", "orientation", "pinned_at",
           "pinned_power_rate"]


class MirroredShiftedPowerRate(ShiftedPowerRate):
    r"""The ``a > 0`` member of the two-point-pinned family: ``φ(λ) = FLOOR·((λ+s)/(λ_min+s))^a``.

    The pin is MIRRORED (hand-off #14 §1.1): ``φ(λ_min)=FLOOR`` and ``φ(λ_max)=R·FLOOR``, so
    ``min_λ φ = FLOOR`` still holds and the range, the ``NormPhi`` no-op and the common ``F_max`` all
    survive. The shift ``s`` solves the SAME equation as ``ShiftedPowerRate``'s —
    ``(λ_max+s)/(λ_min+s) = R^{1/|a|}`` depends on ``|a|`` only — so at a fixed ``|a|`` the two signs
    share one ``s`` and ``φ_{+|a|}`` is the reflection of ``φ_{−|a|}`` in ``log(λ+s)`` about the
    bracket's midpoint. That is a reflection of the SHAPE, not only a relabelling of the order: the
    decreasing law is steep at the small-``λ`` end and flat at the large-``λ`` end, the increasing law
    the reverse, so which modes get well-separated clocks changes with the sign.

    The ``v_F`` guard is unconditional here — ``φ'>0`` makes ``dv/dλ = 1 + τFφ'e^{−Fφ} > 0`` for every
    ``F`` — so ``tau`` is accepted and not needed.

    Everything except construction and :meth:`range` is inherited unchanged: ``__call__``,
    ``local_a``, ``weighted_a``, ``effective_a``, ``clocks`` and ``theta`` were not touched by the
    sign extension. :meth:`clocks` returns ``(longest, shortest)``; the longest sits at ``λ_min``
    here, which :func:`pinned_at` reports.

    The arithmetic is `77ff994`'s ``a > 0`` branch, expression for expression. Build through
    :func:`pinned_power_rate` unless the sign is known to be positive.
    """

    def __init__(self, a: float, lam_min: float, lam_max: float, *, FLOOR: float = 0.077,
                 F_min: float = 0.02, c: float = 3.0, R: float | None = None,
                 tau: float | None = None) -> None:
        a = float(a)
        lam_min, lam_max = float(lam_min), float(lam_max)
        if not (a > 0.0 and math.isfinite(a)):
            raise ValueError(
                f"MirroredShiftedPowerRate needs a finite a > 0; got a={a}. a < 0 is "
                "ccd.forward.rate.ShiftedPowerRate; pinned_power_rate dispatches on the sign.")
        if not (0.0 < lam_min < lam_max):
            raise ValueError(f"need 0 < lam_min < lam_max; got {lam_min}, {lam_max}")
        R = float(R) if R is not None else math.log(2.0) / (float(c) * float(F_min) * float(FLOOR))
        if R <= 1.0:
            raise ValueError(f"need R > 1; got R={R}")
        # The pin equation depends on |a| only, so the shift is the same at either sign.
        rho = R ** (1.0 / abs(a))
        s = (lam_max - rho * lam_min) / (rho - 1.0)
        if s <= 0.0:
            raise ValueError(
                f"R={R} unreachable at a={a}: the unpinned range is only "
                f"(lam_max/lam_min)^|a| = {(lam_max / lam_min) ** abs(a)}; pinning can only SHRINK "
                "the range, so R must be below it.")
        # tau: φ' > 0 makes dv/dλ = 1 + τFφ'e^{-Fφ} > 0 unconditionally; nothing to guard.
        self.a, self.s, self.R = a, s, R
        self.lam_min, self.lam_max = lam_min, lam_max
        self.floor, self.F_min, self.c = float(FLOOR), float(F_min), float(c)
        # The end where φ is smallest carries the FLOOR pin: λ_min at a > 0.
        self.scale = float(FLOOR) * (lam_min + s) ** (-a)

    def range(self) -> float:
        """Realised clock range ``φ(λ_max)/φ(λ_min) = max_λ φ / min_λ φ`` (should equal ``R``)."""
        return float(((self.lam_max + self.s) / (self.lam_min + self.s)) ** self.a)


def pinned_power_rate(a: float, lam_min: float, lam_max: float, **kw) -> ShiftedPowerRate:
    """The two-point-pinned clock at either sign of ``a``.

    ``a < 0`` constructs ``ccd.forward.rate.ShiftedPowerRate`` itself — ``main``'s class, so the
    registered path cannot drift from it. ``a > 0`` constructs :class:`MirroredShiftedPowerRate`.
    ``a = 0`` and non-finite ``a`` are refused, as `77ff994`'s single class refused them. Keyword
    arguments (``FLOOR``, ``F_min``, ``c``, ``R``, ``tau``) pass through.
    """
    a = float(a)
    if a == 0.0 or not math.isfinite(a):
        raise ValueError(
            f"pinned_power_rate needs a finite a != 0; got a={a}. Isotropic φ is constant (R=1 by "
            "definition) and cannot be two-point pinned — use PhiLogPoly.from_power(0.0, …).")
    if a < 0.0:
        return ShiftedPowerRate(a, lam_min, lam_max, **kw)
    return MirroredShiftedPowerRate(a, lam_min, lam_max, **kw)


def _require_pinned(phi) -> None:
    if not isinstance(phi, ShiftedPowerRate):
        raise TypeError(f"needs a two-point-pinned ShiftedPowerRate; got {type(phi).__name__}")


def orientation(phi: ShiftedPowerRate) -> str:
    """``"decreasing"`` (``a<0``: ``F*`` largest at ``λ_max``, generated first) or ``"increasing"``.

    In OPERATOR-eigenvalue coordinates — see :func:`generation_order` before reading it as an order
    in graph frequency.
    """
    _require_pinned(phi)
    return "decreasing" if phi.a < 0.0 else "increasing"


def pinned_at(phi: ShiftedPowerRate) -> float:
    """The spectrum end carrying ``φ = FLOOR`` (the longest clock): ``λ_max`` for ``a<0``, ``λ_min`` for ``a>0``."""
    _require_pinned(phi)
    return phi.lam_max if phi.a < 0.0 else phi.lam_min


def generation_order(phi: ColouringRate, evals: Tensor) -> Tensor:
    r"""`ccd.forward.rate.generation_order`, called through unchanged: mode indices in generation order.

    Bands are generated in order of decreasing ``F* = ln2/φ(λ)``: ascending ``λ`` when ``φ``
    increases (``a>0``), descending ``λ`` when it decreases (``a<0``), degenerate at ``a=0``.

    🛑 **``λ`` here is the OPERATOR's eigenvalue, and "coarse→fine" is a reading, not a fact of the
    sign.** Descending ``λ`` (``a<0``) is coarse→fine in graph frequency only when a large
    eigenvalue of ``S`` belongs to a smooth mode — true for a covariance-like ``S`` (``p(L)²``,
    ``Ĉ``, ``(L+γI)⁻¹``) and FALSE for ``S = L_sym + σI``, whose largest eigenvalues are its
    roughest modes; there ``a<0`` generates fine→coarse. Package A's registered ``lap`` arm ran
    exactly that way (state doc §6.10), and the label in ``rate.py``'s docstring — "(fine→coarse)"
    for ``a>0``, "(coarse→fine)" for ``a<0``, which stays there because that file may not change —
    is what let it pass unnoticed. Read the orientation off the operator — `gfs_arms.build_arm`
    records it per built arm — rather than off ``a``.
    """
    return _rate_generation_order(phi, evals)
