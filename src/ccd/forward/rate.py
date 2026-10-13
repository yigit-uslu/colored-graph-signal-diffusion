"""Colouring rate ``φ`` and per-band SNR for the colored forward family (Phase I1).

Fixing the stationary law to ``N(0, S)`` ties the drift/diffusion together and leaves a
single positive per-eigendirection rate ``φ(λ)`` (the family knob). The power-law case
``φ(λ)=λ^a`` recovers: ``a=0`` whitened-VP (isotropic mean, no spectral order),
``a=1`` the original ``S``-colouring (fine-to-coarse), ``a=-1`` coarse-to-fine.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

ColouringRate = Callable[[Tensor], Tensor]


def phi_power(a: float) -> ColouringRate:
    """Power-law colouring rate ``φ(λ) = λ^a`` (``a=0`` → ``φ≡1`` whitened-VP)."""
    a = float(a)
    return lambda lam: lam ** a


class NormPhi(nn.Module):
    r"""Rescale any rate so ``min_λ φ(λ) = FLOOR`` on a fixed spectrum — the sampler-budget pin.

    ``F_max = −2 ln(ε_prior) / FLOOR`` is set by the *smallest* per-band rate, so pinning that
    minimum is what makes one ``F``-window mean the same amount of work on every arm. Without it a
    clock contrast is confounded with a budget contrast.

    Exactly **scale-invariant in the base rate**: multiplying ``base`` by any positive constant
    leaves the output unchanged, because the constant cancels between ``b`` and ``scale``. That is
    what makes a min-variance objective over ``φ`` well-posed rather than degenerating to ``φ→0``,
    and it is also why wrapping an already-pinned rate is a **×1 no-op** — :class:`ShiftedPowerRate`
    satisfies ``min_λ φ = FLOOR`` by construction, so ``NormPhi`` must not move it. That equality is
    what makes ``F_max`` common across the ladder; a version that rescaled a pinned arm would
    silently give it a different budget.

    Args:
        base: the rate to wrap (anything callable on a ``(d,)`` tensor of eigenvalues).
        spec: the **fixed** spectrum the minimum is taken over. Registered as a buffer, so it moves
            with the module and is not re-derived per call.
        floor: the pinned minimum ``FLOOR``.

    .. note::
       Promoted into ``src`` at implementation-plan step 4b. It previously existed as two
       independent script-side copies (``viz_gen_real.py``, ``viz_minvar_phi.py``) while five
       docstrings in this module referred to "the script-side ``NormPhi``" — so the ``a=0`` arm, one
       half of the fMRI benchmark's G4 headline contrast, depended on a class living only inside a
       PBMC-hardwired driver. Both copies now import this one; ``tests/forward/test_norm_phi.py``
       pins that it reproduces them bit-identically on the fMRI spectrum.
    """

    def __init__(self, base, spec: Tensor, floor: float):
        super().__init__()
        self.base = base
        self.register_buffer("spec", spec)
        self.floor = float(floor)

    def forward(self, lam: Tensor) -> Tensor:
        b = self.base(lam)
        scale = (self.floor / self.base(self.spec).min()).to(device=b.device, dtype=b.dtype)
        return b * scale                                  # min_lam phi = FLOOR, on lam's device

    def effective_a(self):
        return self.base.effective_a()

    @property
    def theta(self):
        return self.base.theta


class PhiLogPoly(nn.Module):
    r"""Learnable colouring rate ``φ_θ(λ) = exp(Σ_{k=0}^{deg} θ_k û^k)``, ``û = (log λ − μ)/s``.

    A positive, smooth, differentiable member of the colouring family carrying only ``deg+1``
    scalar parameters. It is a :class:`ColouringRate` (callable ``λ ↦ φ(λ)``), so it drops into
    :class:`~ccd.forward.marginals.ColoredForward` wherever :func:`phi_power` goes, and — because it
    is only ever evaluated at the Chebyshev nodes — autograd flows into ``θ`` through
    :func:`ccd.operators.apply_fun` (no eigendecomposition; the coefficient DCT and Clenshaw
    recurrence are both differentiable).

    Design:

    * **Contains the power law exactly.** The degree-1 case reproduces ``φ(λ)=λ^a`` (``θ=(a·μ, a·s,
      0, …)``): :meth:`from_power` initialises the head to any fixed ``a``, so joint training *moves*
      the schedule away from a known baseline rather than from noise.
    * **Positive by construction** (``exp``), so ``Σ_F = λ(1−e^{−Fφ})`` and ``v_F = λ − τe^{−Fφ}``
      stay valid for *any* ``θ`` — no positivity constraint to enforce.
    * **Curvature decouples order from cost.** ``μ, s`` are *fixed* standardisers (from the spectrum
      bracket, not trained), so ``û`` is O(1) across the band and the ``θ`` are comparably scaled.
      With ``deg ≥ 2`` the schedule can be steep in the spectral bulk yet flatten in the tails,
      lowering ``F_max ∝ 1/min_λ φ`` while keeping the generation *order* (the sign of the slope).

    Args:
        mu, s: fixed standardisers of ``log λ`` (``s`` must be > 0).
        degree: polynomial degree in ``û`` (``deg=1`` ≡ the power-law family; default 2).
    """

    def __init__(
        self,
        mu: float,
        s: float,
        *,
        degree: int = 2,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if not (s > 0.0):
            raise ValueError(f"s must be > 0 (log-λ scale); got {s}")
        if degree < 1:
            raise ValueError(f"degree must be >= 1 (deg 1 is the power-law family); got {degree}")
        self.mu = float(mu)
        self.s = float(s)
        self.degree = int(degree)
        self.theta = nn.Parameter(torch.zeros(self.degree + 1, dtype=dtype, device=device))

    @classmethod
    def from_power(
        cls,
        a: float,
        lo: float,
        hi: float,
        *,
        degree: int = 2,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> "PhiLogPoly":
        """Head initialised to the fixed power law ``φ(λ)=λ^a`` on the bracket ``[lo, hi]``.

        Sets ``μ, s`` to the midpoint/half-width of ``log[lo, hi]`` (so ``û ∈ [−1, 1]`` across the
        bracket) and ``θ = (a·μ, a·s, 0, …)``, which is ``φ(λ)=λ^a`` exactly (``log λ = μ + s·û``).
        """
        if not (0.0 < lo < hi):
            raise ValueError(f"need 0 < lo < hi; got lo={lo}, hi={hi}")
        mu = 0.5 * (math.log(lo) + math.log(hi))
        s = 0.5 * (math.log(hi) - math.log(lo))
        head = cls(mu, s, degree=degree, dtype=dtype, device=device)
        with torch.no_grad():
            head.theta.zero_()
            head.theta[0] = float(a) * mu
            head.theta[1] = float(a) * s
        return head

    def effective_a(self) -> float:
        """Local power-law exponent at the spectral centre ``û=0``: ``d log φ / d log λ = θ₁/s``."""
        return float(self.theta[1].detach()) / self.s

    def forward(self, lam: Tensor) -> Tensor:
        lam = torch.as_tensor(lam)
        th = self.theta.to(device=lam.device, dtype=lam.dtype)      # follow the query's device/dtype
        u = (torch.log(lam) - self.mu) / self.s
        p = torch.zeros_like(u)
        for k in range(self.degree, -1, -1):                        # Horner: Σ_k θ_k u^k
            p = p * u + th[k]
        return torch.exp(p)


class TiledRate:
    r"""Rule-based colouring rate that *tiles* the per-band unit-SNR clocks across the sampler budget.

    Given the importance-carrying eigenvalues ``λ_1 > … > λ_m`` (the factor lines, well separated from the
    structureless bulk), the tiling rule stretches their unit-SNR clocks ``F*_g = ln2/φ(λ_g)`` **log-uniformly**
    across the available range: the coarsest clock pinned at ``ln2/FLOOR`` (the ``NormPhi`` budget top) and the
    finest at ``c·F_min`` (just clearing the sampler floor). ``φ`` is the monotone PCHIP interpolant in
    ``(log λ, log φ)`` through the ``m`` knots, held **flat** (a shelf) below ``λ_m`` (the structureless bulk
    inherits the finest factor's clock) and above ``λ_1``.

    Unlike :class:`PhiLogPoly` it carries **no learnable parameters**: the schedule is *computed* from the
    spectrum, the budget ``FLOOR`` and the sampler floor ``F_min`` — not trained. It is a :data:`ColouringRate`
    (a plain callable ``λ ↦ φ``, strictly positive), so it drops into :class:`~ccd.forward.marginals.ColoredForward`
    wherever :func:`phi_power` / :class:`PhiLogPoly` go, and (like them) wraps in the script-side ``NormPhi``
    budget pin as a ≈×1 no-op when ``φ(λ_max)=FLOOR``.

    The factor eigenvalues must be supplied by the caller (a script-side spectral decomposition of ``S``, outside
    the no-eig ``src`` guard — this class never touches the operator, only interpolates). ``effective_a`` / ``theta`` are
    logging shims (the local ``dlogφ/dlogλ`` at the spectral centre) for parity with :class:`PhiLogPoly`.

    Args:
        lam_factors: the importance-carrying eigenvalues (any order; sorted descending internally).
        FLOOR: the min-φ budget pin ``φ(λ_1)=FLOOR`` (so the coarsest clock is ``ln2/FLOOR``).
        F_min: the sampler floor; the finest clock is set to ``c·F_min``.
        c: finest-clock headroom above the floor (default 1.4, inverted from the measured crossing).
    """

    def __init__(self, lam_factors, *, FLOOR: float = 0.077, F_min: float = 0.02, c: float = 1.4, weights=None):
        from scipy.interpolate import PchipInterpolator             # scipy allowed in src (cf. operators/rational.py)

        lam_in = np.asarray(lam_factors, dtype=float).reshape(-1)
        order = np.argsort(lam_in)[::-1]                            # strictly descending λ
        lam = lam_in[order]
        if lam.size < 2:
            raise ValueError(f"need >= 2 factor eigenvalues; got {lam.size}")
        if not (lam[:-1] > lam[1:]).all():
            raise ValueError("factor eigenvalues must be distinct")
        top = math.log(2.0) / float(FLOOR)                          # coarsest clock (NormPhi budget top)
        fine = float(c) * float(F_min)                             # finest clock (just clearing the sampler floor)
        if weights is None:                                        # equal atoms -> log-uniform tiling (C3)
            pos = np.linspace(0.0, 1.0, lam.size)
        else:                                                       # importance-weighted survival placement
            w = np.clip(np.asarray(weights, dtype=float).reshape(-1)[order], 0.0, None)  # desc-λ aligned
            prefix = np.concatenate([[0.0], np.cumsum(w)[:-1]])     # Σ_{h<g} w_h (survival mass above λ_g)
            pos = prefix / prefix[-1] if prefix[-1] > 0 else np.linspace(0.0, 1.0, lam.size)
        self.Fstar = np.exp(np.log(top) + pos * (np.log(fine) - np.log(top)))   # desc-λ order (≡ linspace if equal)
        phi_knots = math.log(2.0) / self.Fstar                      # φ_g = ln2/F*_g (grows as λ decreases)
        ll = np.log(lam)[::-1]                                       # ascending log λ (interpolator wants x asc)
        lp = np.log(phi_knots)[::-1]                                 # matched log φ (decreasing in λ)
        self._pchip = PchipInterpolator(ll, lp, extrapolate=False)
        self._ll_lo, self._ll_hi = float(ll[0]), float(ll[-1])      # [log λ_m, log λ_1]
        self._lp_lo, self._lp_hi = float(lp[0]), float(lp[-1])      # shelf log-φ at the fine / coarse ends
        self.lam_factors = lam
        self.floor = float(FLOOR)
        self.F_min = float(F_min)

    def __call__(self, lam: Tensor) -> Tensor:
        lam = torch.as_tensor(lam)
        ll = np.log(np.clip(lam.detach().cpu().numpy().reshape(-1), 1e-300, None))
        llc = np.clip(ll, self._ll_lo, self._ll_hi)                 # flat shelf beyond both knots (no extrapolation)
        lp = self._pchip(llc)
        lp = np.where(llc <= self._ll_lo, self._lp_lo, lp)          # belt-and-braces shelf at the fine end
        lp = np.where(llc >= self._ll_hi, self._lp_hi, lp)          # …and the coarse end
        out = np.exp(lp).reshape(tuple(lam.shape))
        return torch.as_tensor(out, dtype=lam.dtype, device=lam.device)

    @classmethod
    def from_spectral_model(
        cls, sm, *, G: int | None = None, mode: str = "atoms",
        FLOOR: float = 0.077, F_min: float = 0.02, c: float = 1.4, weights=None,
    ) -> "TiledRate":
        r"""Build the rule from a :class:`ccd.spectral.SpectralModel` (matvec **or** eigh backend).

        ``mode="atoms"`` (the validated instantiation) tiles the detected factor lines
        ``λ_1 > … > λ_G`` exactly as :meth:`__init__` does — so with the ``eigh`` backend this
        reproduces C3 bit-for-bit, and with the matvec backend it is **C3-MF**. ``mode="density"``
        (experimental for generation) places knots on the KPM survival curve instead: it samples
        ``Ḡ(λ)`` on a log-λ grid across the bracket and keeps the ``G`` levels where the importance
        mass is deposited, so the clock steepens where the (continuum) spectral measure concentrates.
        """
        lam = sm.top_g(G)
        if mode == "atoms":
            return cls(lam, FLOOR=FLOOR, F_min=F_min, c=c, weights=weights)
        if mode == "density":
            n = lam.size
            grid = np.exp(np.linspace(math.log(sm.lo), math.log(sm.hi), 256))
            gbar = np.asarray(sm.survival(grid), dtype=float)            # decreasing in λ
            # G knot λ's at equal survival-mass quantiles (the density analogue of the lines)
            levels = (np.arange(n) + 0.5) / n                            # mid-quantiles in (0, 1)
            knots = np.interp(levels, gbar[::-1], grid[::-1])           # survival is monotone ↓
            return cls(knots, FLOOR=FLOOR, F_min=F_min, c=c)
        raise ValueError(f"mode must be 'atoms' or 'density'; got {mode!r}")

    def clocks(self) -> np.ndarray:
        """Per-factor unit-SNR clocks ``F*_g = ln2/φ(λ_g)`` in descending-λ order (for the §4.1 clock audit)."""
        return self.Fstar.copy()

    def effective_a(self) -> float:
        """Local power-law exponent ``dlogφ/dlogλ`` at the spectral centre (logging parity with PhiLogPoly)."""
        c = 0.5 * (self._ll_lo + self._ll_hi)
        h = min(1e-3, 0.25 * (self._ll_hi - self._ll_lo))
        lo, hi = max(c - h, self._ll_lo), min(c + h, self._ll_hi)
        return float((self._pchip(hi) - self._pchip(lo)) / (hi - lo))

    @property
    def theta(self) -> Tensor:
        return torch.tensor([self.effective_a(), 0.0, 0.0])


class ShiftedPowerRate:
    r"""Two-point-pinned shifted power law ``φ(λ) = FLOOR·((λ+s)/(λ_max+s))^a`` (``a<0``).

    **Why.** A plain power law ``λ^a`` is pinned at only *one* end — ``NormPhi`` fixes
    ``φ(λ_max)=FLOOR`` — so its clock **range** ``R = (λ_max/λ_min)^{|a|}`` is whatever the spectrum
    and ``a`` happen to produce (177 / 2360 / 31400 for ``a=−1/−1.5/−2`` on the PBMC ``d=256``
    bracket). Since ``λ_min(S)=τ`` whenever ``Ĉ_n`` is rank-deficient, that range is set by the
    *ridge*, not by the data. Bands whose unit-SNR clock ``F*=ln2/φ`` falls below the sampler floor
    ``F_min`` are annihilated before the first step and are never seen partially noised.
    :class:`TiledRate` already avoids this by pinning **both** ends (``rate.py`` ``top``/``fine``);
    this class gives the power-law family the same two-point budget:

        ``φ(λ_max) = FLOOR``            (coarsest clock ``ln2/FLOOR``; ``NormPhi`` is then a ×1 no-op)
        ``φ(λ_min) = R·FLOOR``          (finest clock ``c·F_min``, i.e. ``R = ln2/(c·F_min·FLOOR)``)

    so every pinned arm shares the C3 arms' clock range by construction, and ``a`` controls only the
    *shape* of ``φ`` between two fixed endpoints. The shift is closed form: with ``ρ = R^{1/|a|}``,

        ``s = (λ_max − ρ·λ_min)/(ρ − 1)``.

    **Why a shift and not a clamp.** ``max(λ, λ_lo)^a`` achieves the same two-point pin but is only
    ``C⁰``; :func:`~ccd.operators.chebyshev.apply_fun` is Chebyshev, so the kink converges
    *algebraically* and stalls (~1.7e-3 at ``K=160``). ``(λ+s)^a`` has its only singularity at
    ``λ=−s<0``, outside ``[lo,hi]``, hence it is analytic there and converges **geometrically** — at
    the production ``K=40`` it is ~800× more accurate at ``F=F_min`` than the plain ``λ^a`` arm it
    replaces (float32; float64 gap is larger). It is also **strictly decreasing**, so ``f_star`` /
    :func:`generation_order` stay well posed instead of tying on a shelf.

    **``v_F`` monotonicity.** :meth:`~ccd.forward.marginals.ColoredForward._v_bracket` assumes
    ``v_F(λ)=λ−τe^{−Fφ}`` is monotone. Since ``dv/dλ = 1 − τF|φ'|e^{−Fφ}`` and
    ``sup_F F e^{−Fφ} = 1/(eφ)``, a sufficient condition **uniform in F** is
    ``τ|a|/(e(λ_min+s)) < 1``; pass ``tau`` to have it checked at construction.

    Args:
        a: power-law exponent, must be ``< 0`` (isotropic ``a=0`` has ``R=1`` by definition and
            cannot be two-point pinned; ``a>0`` is increasing and breaks the pin's orientation).
        lam_min, lam_max: the spectrum's extremes (the pin is defined on the supplied **spectrum**,
            not on the wider Chebyshev bracket ``[lo,hi]``).
        FLOOR, F_min, c: the budget pins, matching :class:`TiledRate`'s.
        R: override the derived range ``ln2/(c·F_min·FLOOR)``.
        tau: if given, enforce the ``v_F``-monotonicity guard above.
    """

    def __init__(self, a: float, lam_min: float, lam_max: float, *, FLOOR: float = 0.077,
                 F_min: float = 0.02, c: float = 3.0, R: float | None = None,
                 tau: float | None = None) -> None:
        a = float(a)
        lam_min, lam_max = float(lam_min), float(lam_max)
        if a >= 0.0:
            raise ValueError(
                f"ShiftedPowerRate needs a < 0; got a={a}. Isotropic φ is constant (R=1 by "
                "definition) and cannot be two-point pinned — use PhiLogPoly.from_power(0.0, …).")
        if not (0.0 < lam_min < lam_max):
            raise ValueError(f"need 0 < lam_min < lam_max; got {lam_min}, {lam_max}")
        R = float(R) if R is not None else math.log(2.0) / (float(c) * float(F_min) * float(FLOOR))
        if R <= 1.0:
            raise ValueError(f"need R > 1; got R={R}")
        rho = R ** (1.0 / abs(a))
        s = (lam_max - rho * lam_min) / (rho - 1.0)
        if s <= 0.0:
            raise ValueError(
                f"R={R} unreachable at a={a}: the unpinned range is only "
                f"(lam_max/lam_min)^|a| = {(lam_max / lam_min) ** abs(a)}; pinning can only SHRINK "
                "the range, so R must be below it.")
        if tau is not None:
            margin = float(tau) * abs(a) / (math.e * (lam_min + s))
            if margin >= 1.0:
                raise ValueError(
                    f"v_F may be non-monotone (tau*|a|/(e*(lam_min+s)) = {margin:.4g} >= 1), which "
                    "violates ColoredForward._v_bracket's precondition; lower |a| or raise R.")
        self.a, self.s, self.R = a, s, R
        self.lam_min, self.lam_max = lam_min, lam_max
        self.floor, self.F_min, self.c = float(FLOOR), float(F_min), float(c)
        self.scale = float(FLOOR) * (lam_max + s) ** (-a)

    @classmethod
    def from_spectrum(cls, a: float, evals, **kw) -> "ShiftedPowerRate":
        """Build from a spectrum array/tensor, taking ``lam_min``/``lam_max`` as its extremes."""
        ev = np.asarray(evals, dtype=float).reshape(-1)
        return cls(a, float(ev.min()), float(ev.max()), **kw)

    def __call__(self, lam: Tensor) -> Tensor:
        lam = torch.as_tensor(lam)
        return self.scale * (lam + self.s) ** self.a

    def effective_a(self) -> float:
        """The **nominal** ``a`` (the ``λ≫s`` asymptote and the arm label).

        Deliberately *not* :class:`TiledRate`'s bracket-centre slope: at the PBMC spectral centre the
        three pinned arms all realise ≈−1.0, so a local slope would collapse them into one cell in
        every reader table. Use :meth:`local_a` / :meth:`weighted_a` for the realised slope, and
        report :meth:`range` alongside so a pinned arm is never confused with a plain one.
        """
        return self.a

    def local_a(self, lam: Tensor) -> Tensor:
        """Realised local slope ``dlogφ/dlogλ = a·λ/(λ+s)`` (→ ``a`` as ``λ→∞``, → 0 as ``λ→0``)."""
        lam = torch.as_tensor(lam)
        return self.a * lam / (lam + self.s)

    def weighted_a(self, evals, weights) -> float:
        """Importance-weighted realised slope ``Σ w_i·local_a(λ_i) / Σ w_i``."""
        ev = torch.as_tensor(np.asarray(evals, dtype=float).reshape(-1))
        w = torch.as_tensor(np.asarray(weights, dtype=float).reshape(-1)).clamp_min(0.0)
        tot = float(w.sum())
        if tot <= 0.0:
            return float("nan")
        return float((w * self.local_a(ev)).sum() / tot)

    def range(self) -> float:
        """Realised clock range ``φ(λ_min)/φ(λ_max)`` (should equal ``R``)."""
        return float(((self.lam_min + self.s) / (self.lam_max + self.s)) ** self.a)

    def clocks(self) -> tuple[float, float]:
        """``(coarsest, finest)`` unit-SNR clocks ``ln2/φ`` — i.e. ``(ln2/FLOOR, ln2/(R·FLOOR))``."""
        return math.log(2.0) / self.floor, math.log(2.0) / (self.R * self.floor)

    @property
    def theta(self) -> Tensor:
        return torch.tensor([self.a, 0.0, 0.0])


class PhiLogPolyFit(PhiLogPoly):
    r"""Smooth log-polynomial (candidate **B′**) whose coefficients are **fit — not trained — to a rule's
    log-clock** in the importance measure ``ν = μ + ε_fit·p`` (matvec-colouring M4, §2.3 path (a)/(b)).

    Where :class:`TiledRate` (C3) is a PCHIP staircase through the factor knots, B′ is a degree-``K`` member
    of the *same* :class:`PhiLogPoly` family (``φ = exp(q)``) that **approximates** that log-clock smoothly.
    The fit weights the ``G`` factor lines by 1 (the importance atoms carry the accuracy) and a set of bulk
    quadrature nodes by ``ε_fit`` times their KPM-density mass (the principled form of offline-B's shelf
    pseudo-points), then solves a weighted ridge least-squares with a smoothness penalty ``λ_s Σ_k k² η_k²``
    (also the operator-cost proxy — a smoother ``q`` lowers the ApplyFun degree of ``e^{-Fφ}``). It carries
    **no trained parameters** and drops into :class:`~ccd.forward.marginals.ColoredForward` exactly like C3 /
    offline-B. Two bases: ``"log"`` (default; ``q`` a poly in standardised ``log λ`` — a native
    :class:`PhiLogPoly`) and ``"raw"`` (``q = Σ η_k T_k(λ̃)``, the pure-continuum fallback whose Gram comes
    from the KPM moments ``M_{jk}=½(μ_{j+k}+μ_{|j-k|})``).

    Diagnostics (attributes after construction): ``gram_cond`` (condition number of the unregularised Gram),
    ``atoms_resid`` (``max_g |ln φ_{B'}(λ_g) − ln φ*(λ_g)|`` — the §6.5(i) clock-placement error).
    """

    _raw = None                                                     # (coeffs, lo, hi) when basis="raw"

    @classmethod
    def from_spectral_model(
        cls, sm, *, G: int | None = None, degree: int = 2, eps_fit: float = 0.1,
        lam_s: float = 1e-3, FLOOR: float = 0.077, F_min: float = 0.02, c: float = 3.0,
        basis: str = "log", n_bulk: int = 32,
        dtype: torch.dtype = torch.float32, device: str | torch.device | None = None,
    ) -> "PhiLogPolyFit":
        """Fit B′ to the C3 log-clock of ``sm`` (the validated generation driver) in the ``ν`` measure."""
        lam = np.asarray(sm.top_g(G), dtype=float)                  # G factor lines (the importance atoms)
        target = TiledRate(lam, FLOOR=FLOOR, F_min=F_min, c=c)      # the rule ln φ* is C3's log-clock
        lo, hi = float(sm.lo), float(sm.hi)
        mu = 0.5 * (math.log(lo) + math.log(hi))
        s = 0.5 * (math.log(hi) - math.log(lo))

        edges = np.exp(np.linspace(math.log(lo), math.log(hi), n_bulk + 1))
        centres = np.sqrt(edges[:-1] * edges[1:])
        if getattr(sm, "mu", None) is not None:
            sv = np.asarray(sm.survival(edges), dtype=float)        # Ḡ at edges (↓ in λ)
            binmass = np.clip(sv[:-1] - sv[1:], 0.0, None)          # mass in each (edge_i, edge_{i+1}]
        else:
            binmass = np.full(n_bulk, 1.0 / n_bulk)                 # atoms-only: flat shelf anchor
        smooth = np.diag((np.arange(degree + 1) ** 2).astype(float))            # k² penalty (k=0 free)

        if basis == "log":
            # path (a): importance measure ν = atoms(w=1) + ε_fit·p — a node quadrature, Gram = BᵀWB
            node_lam = np.concatenate([lam, centres])
            node_w = np.concatenate([np.ones(lam.size), eps_fit * binmass])
            y = np.log(target(torch.as_tensor(node_lam)).cpu().numpy())
            u = (np.log(node_lam) - mu) / s
            B = np.vander(u, degree + 1, increasing=True)          # columns û^0 … û^K
            BtW = B.T * node_w
            gram = BtW @ B
            eta = np.linalg.solve(gram + lam_s * smooth, BtW @ y)   # matrix-free-ok: (deg+1)² Gram, d-independent
        elif basis == "raw":
            # path (b): pure-continuum p-measure fit — Gram from the exact moments, load by p-quadrature
            y = np.log(target(torch.as_tensor(centres)).cpu().numpy())
            a_sh, b_sh = 2.0 / (hi - lo), -(hi + lo) / (hi - lo)
            x = np.clip(a_sh * centres + b_sh, -1.0, 1.0)
            B = np.cos(np.arccos(x)[:, None] * np.arange(degree + 1)[None, :])   # T_k(λ̃)
            gram = _cheb_moment_gram(np.asarray(sm.mu, dtype=float), degree)     # M_{jk}=½(μ_{j+k}+μ_{|j−k|})
            load = (B.T * binmass) @ y                              # ∫ T_k ln φ* dp  (density quadrature)
            eta = np.linalg.solve(gram + lam_s * smooth, load)      # matrix-free-ok: (deg+1)² Gram, d-independent
        else:
            raise ValueError(f"basis must be 'log' or 'raw'; got {basis!r}")
        gram_cond = float(np.linalg.cond(gram))

        head = cls(mu, s, degree=degree, dtype=dtype, device=device)
        with torch.no_grad():
            head.theta.copy_(torch.as_tensor(eta, dtype=head.theta.dtype))
        if basis == "raw":
            head._raw = (torch.as_tensor(eta, dtype=dtype), lo, hi)
        # §6.5(i) clock placement: max |ln φ_{B'}(λ_g) − ln φ*(λ_g)| over the atoms
        with torch.no_grad():
            fit_at_atoms = torch.log(head(torch.as_tensor(lam))).cpu().numpy()
        head.gram_cond = gram_cond
        head.atoms_resid = float(np.abs(fit_at_atoms - np.log(target(torch.as_tensor(lam)).cpu().numpy())).max())
        head.basis = basis
        return head

    def forward(self, lam: Tensor) -> Tensor:
        if self._raw is None:
            return super().forward(lam)
        coeffs, lo, hi = self._raw                                  # exp(Σ η_k T_k(λ̃)) via Clenshaw
        lam = torch.as_tensor(lam)
        cf = coeffs.to(device=lam.device, dtype=lam.dtype)
        a = 2.0 / (hi - lo)
        b = -(hi + lo) / (hi - lo)
        x = torch.clamp(a * lam + b, -1.0, 1.0)
        b1 = torch.zeros_like(x)
        b2 = torch.zeros_like(x)
        for k in range(cf.numel() - 1, 0, -1):
            b1, b2 = 2.0 * x * b1 - b2 + cf[k], b1
        q = x * b1 - b2 + cf[0]
        return torch.exp(q)


def _cheb_moment_gram(mu: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev Gram from spectral moments: ``M_{jk} = ½(μ_{j+k} + μ_{|j−k|})`` (path (b), needs μ to 2K)."""
    idx = np.arange(degree + 1)
    return 0.5 * (mu[np.abs(idx[:, None] - idx[None, :])] + mu[idx[:, None] + idx[None, :]])


def snr(lam: Tensor, F: float, phi: ColouringRate, tau: float) -> Tensor:
    """Per-band signal-to-noise ratio with the ``(λ−τ)/λ`` prefactor (correctness fix #1).

    ``SNR_λ(F) = ((λ−τ)/λ) · 1/(e^{Fφ(λ)} − 1)`` — the ratio of signal power ``m_t²Σ_0``
    to noise power ``Σ_t`` per band. Dropping the prefactor is the marginal-preserving
    (``τ→0``) approximation only.

    Evaluated in the **overflow-safe** form ``1/(e^{Fφ}−1) = e^{−Fφ}/(1−e^{−Fφ})`` (only ever
    exponentiates a *negative* argument). This is value-identical to the naïve ``1/(e^{Fφ}−1)``
    away from overflow, but ``e^{+Fφ}`` overflows to ``inf`` for wide spectra at large ``F``
    (``Fφ ≳ 88`` in float32): the naïve forward masks it (``1/(inf−1)=0``) but its **backward is
    ``inf/inf = NaN``**, which breaks autograd through a *learnable* ``φ`` (the FiLM SNR feature).
    """
    lam = torch.as_tensor(lam)
    x = F * phi(lam)                                          # Fφ ≥ 0 on a valid forward process
    return ((lam - tau) / lam) * torch.exp(-x) / (-torch.expm1(-x))


def unit_crossing_F(phi_val: float) -> float:
    """Integrated rate at which the SNR crosses 1 in the ``τ→0`` limit: ``F* = ln2 / φ``."""
    return math.log(2.0) / float(phi_val)


def f_star(phi: ColouringRate, lam: Tensor) -> Tensor:
    """Per-band unit-SNR crossing ``F*(λ) = ln2 / φ(λ)`` (vectorised ``τ→0`` form).

    The band ``λ`` is signal-dominated while ``F < F*(λ)`` and noise-dominated beyond it
    (cf. :func:`snr`); ``F*`` is therefore the natural per-eigendirection "clock" of the
    colouring. For ``φ(λ)=λ^a`` it is ``F*(λ)=ln2·λ^{−a}``.
    """
    lam = torch.as_tensor(lam)
    return math.log(2.0) / phi(lam)


def generation_order(phi: ColouringRate, evals: Tensor) -> Tensor:
    """Reverse-generation order of the spectral bands: indices of ``evals`` by decreasing ``F*``.

    The reverse (generative) process runs ``F`` from large down to ``0``; a band's signal
    re-emerges from noise as ``F`` drops below its ``F*(λ)`` (:func:`f_star`), so bands are
    *generated* in order of **decreasing** ``F*``. This is the colouring's generation-order
    signature: for ``φ(λ)=λ^a`` it is ascending ``λ`` when ``a>0`` (fine→coarse), descending
    ``λ`` when ``a<0`` (coarse→fine), and degenerate (all ``F*`` equal) at ``a=0``.
    """
    fs = f_star(phi, torch.as_tensor(evals))
    return torch.argsort(fs, descending=True)
