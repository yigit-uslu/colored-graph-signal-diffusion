#!/usr/bin/env python
"""Package-B loss weight and noise-conditioning feature — **one definition, every arm.**

🛑 **Why this file exists, and why it changes the COVARIANCE arm too.**

The reviewing agent found that `ccd.denoiser.denoiser.edm_weight` and `VNNBody._cond_feature` still
compute from the commuting model, and that `MisspecifiedColoredForward` does not override either —
so a graph arm would train under the wrong loss weighting and the wrong FiLM conditioning. Correct,
and my claim that "only the marginal and four EDM preconditioners need to change" was incomplete.

Measuring it (`scratchpad/metr_weight_probe.py`, fold 0) turned up something larger than the
finding. Both sealed definitions are **grid means of per-band scalars on a UNIFORM λ-grid**:

    edm_weight   = 1 / mean_grid c_out²(λ)
    ν(F)         = log mean_grid SNR(λ)

METR-LA's covariance spectrum has κ ≈ 1885, so a uniform grid on `[λ_min, λ_max]` puts almost all of
its mass where almost no eigenvalues live. Against the exact band mean, on the **covariance** arm —
the commuting case, where the sealed code is supposed to be exact:

| F | 0.06 | 0.20 | 1.00 | 4.00 | 9.00 |
|---|---|---|---|---|---|
| `w_grid / w_band` | 0.3842 | 0.2671 | 0.1131 | 0.0622 | 0.0500 |
| `ν_grid − ν_band` (nats) | +3.12 | +3.36 | +3.61 | +3.55 | +3.39 |

A factor of **2.6× to 20×** in the loss weight, and **3.1–3.6 nats** in the conditioning feature.

📌 The `ν` row is re-derived (`scratchpad/metr_nu_table.py`) and each entry is 0.01-0.02 below what
this table carried before: the original probe evaluated the grid mean at **`ngrid = 64`** while
`VNNBody._cond_feature` uses its own `cond_ngrid = 257`. Same conclusion, production grid.

**Severity, scoped honestly (principle 6).** Neither is a correctness bug in the usual sense: the EDM
weight is a per-`F` multiplier that does not move the optimum at fixed `F`, and `ν` is an input to a
learned FiLM MLP, which absorbs a monotone reparametrisation. What matters is that **the ratio's
`F`-dependence differs by arm** — 0.3842→0.0500 on the covariance arm against 0.4936→0.0941 on the
graph arm — so the sealed definitions impose an *arm-dependent effective loss schedule*. That is precisely
the "a shared setting is arm-dependent in effect" trap `docs/experiment_conventions.md` §1 already
records for gradient clipping.

**The resolution is constrained, though not unique — "forced" overstates it.** What is forced is
that Package B must define *a* pair of these quantities itself; *which* pair is a registered policy
choice, justified by the commuting reduction plus one-definition-across-arms, and other extensions
with the same reduction exist. Q2 is therefore an external conceptual replication, not a
regime-identical reproduction of fMRI. Sealed files may not be edited, so `edm_weight` cannot be
corrected in place; and using the sealed form for `cov` and a trace form for `graph` would be the
arm-dependent-method confound `fmri_arms` exists to prevent. That leaves exactly one self-consistent
option: **Package B defines both quantities itself, in a form that is well-defined for commuting and
non-commuting `Σ_0` alike, and every arm uses it.**

⚠️ **DECLARED DEVIATION, needs sign-off.** These are NOT the fMRI regime's definitions. Cross-panel
comparison of loss curves or FiLM behaviour against fMRI is void; Q1/Q2 remain internally valid
because all three arms share one definition. State it wherever Q2 is reported, beside the `ā = −1.30`
shape caveat (state §12.2).

Both forms below reduce to the band means of the sealed scalars in the commuting case, so the change
is **grid-mean → band-mean, not a new quantity**. Measured on fold 0 across the whole `F` window
(`scratchpad/metr_reduction_probe.py`): `edm_weight` agrees with `1/mean_λ c_out²(λ)` to 2e-16-1e-13,
and `snr_bands` agrees with `mean_λ SNR_λ` to **3.0e-16-2.8e-14** across all five folds.

⚠️ That second figure was recorded as "1.5e-8-5e-8 (the eigenbasis rotation's own accuracy)" and was
wrong by seven orders of magnitude — it was **float32 rounding inside the probe's own logging**
(`torch.log(torch.tensor(x))`; `torch.tensor` of a Python float defaults to float32, costing 4.6e-08
relative), which I then attributed to the rotation. Found only because a review refused an acceptance
bound *calibrated from that observation* and required one derived from inputs; the derived bound
`δ·W + 4γ_d` comes out at ~7e-14, at which point 1.5e-8 was impossible. State doc §13.16.1 D6.

🛑 **`snr_bands` REPLACES an earlier `snr_trace`, which did NOT have that property — my error, and
the seventh number to come out wrong in this arc.** **How it got in, since the mechanism is more useful than the apology:** the probe that justified
this file computed the band mean for the `cov` arm (`((λ−τ)/λ)·e/(1−e)`, averaged over the
eigenvalues — correct) and the *trace ratio* for the `graph` arm (`tr(mCm)/tr(Σ_F)`), then reported
both under one heading. I implemented the second for both arms and wrote a test asserting the trace
ratio equals a ratio of sums — my formula against my own code, which could only ever agree.

`snr_trace` returned the *ratio of traces*
`tr(m Σ_0 m)/tr(Σ_F)`, and a ratio of sums is not the mean of the per-band ratios: measured against
`mean_λ SNR_λ` it was off by **0.19 nats at F = 0.02 and 3.76 nats at F = 102**, and its gap to the
sealed grid mean *changed sign* mid-window (+2.84 → −1.84) rather than sitting at the flat ~+3.4 the
table above records. So the file implemented something other than what state §13.14 F2 registered,
while its docstring claimed the reduction as a property of "the trace forms" — true of the weight,
false of the feature. The band form restores the registered quantity; the justification table above
now describes the change actually being made.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["BandSNR", "SNR_FLOOR", "cond_feature", "edm_weight", "snr_bands"]

#: The additive floor inside the feature's logarithm. **One definition**, shared by the reference
#: :func:`cond_feature` and the optimised :class:`BandSNR`, so the two cannot drift apart at the one
#: place where a difference would be invisible (`log` of a vanishing mean at large `F`).
SNR_FLOOR: float = 1e-30


def edm_weight(blocks: dict[str, Tensor]) -> float:
    """`w(F) = 1 / (tr(c_out²)/d)` — the reciprocal MEAN-OVER-BANDS posterior variance.

    `blocks` is one `MisspecifiedColoredForward._blocks(F)` entry, so this costs a trace on an
    already-computed matrix. Reduces exactly to the band mean of `ccd.forward.bands.c_out_sq_fn`
    when `Σ_0 = S − τI`; differs from `ccd.denoiser.denoiser.edm_weight` because that one averages
    on a uniform λ-grid rather than over the spectrum's actual bands.

    🛑 **The `floor=1e-8` argument is gone, and it was worse than useless.** It returned
    `1/max(mean, floor)`, so a **negative** mean posterior variance — which is not a small number but
    a *defect*, the conditional covariance having gone indefinite — was converted into a weight of
    `1e+08`. The largest possible reward for the worst possible input.

    It also never fired. Measured (`scratchpad/metr_review6_floor.py`) on all five registered folds,
    both colored arms, 40 log-spaced points across the whole registered window `F ∈ [0.02, 102]`:
    the smallest mean posterior variance anywhere is **7.17e-02**, which is **7.2e+06×** above the
    floor. So it could only ever have masked a defect, never rescued a computation. Non-positive
    means now raise.
    """
    # float64 for the reduction regardless of the block's apply dtype: `c_out²` collapses towards 0
    # as `F -> 0` and this is its reciprocal, so the one place precision is free is the one place a
    # float32 rounding would be amplified.
    diag = torch.diagonal(blocks["c_out_sq"]).double()
    if not torch.isfinite(diag).all():
        raise ValueError(f"c_out^2 has non-finite diagonal entries "
                         f"({int((~torch.isfinite(diag)).sum())} of {diag.numel()})")
    mean = float(diag.mean())
    if not (mean > 0.0):
        raise ValueError(f"mean posterior variance is {mean:.6e}, which is not positive. c_out^2 is "
                         "a conditional covariance; a non-positive mean is a defect upstream, not a "
                         "small number to floor. Registered folds give >= 7.17e-02 across the whole "
                         "F window.")
    return 1.0 / mean


def snr_bands(blocks: dict[str, Tensor], sigma0: Tensor, evecs: Tensor) -> float:
    r"""`mean_j (u_jᵀ m Σ_0 m u_j) / (u_jᵀ Σ_F u_j)` over `S`'s eigen-directions `u_j`.

    The generalisation of the sealed `ccd.forward.rate.snr`, whose per-band
    `SNR_λ = m²(λ)Σ_0(λ)/Σ_F(λ)` has no counterpart once `Σ_0` stops commuting with `S` — there are
    no per-band scalars. What survives is the *direction*: `S`'s eigenvectors still exist, `m` and
    `Σ_F` are still diagonal in them, and only `Σ_0` is not. Taking the signal and noise powers
    along each of those directions and averaging the ratios reduces **exactly** to `mean_λ SNR_λ`
    when `Σ_0 = S − τI`, which is what makes the deviation from the sealed regime the narrow one
    that was registered: the *grid* becomes the spectrum, and nothing else moves.

    ⚠️ NOT `tr(m Σ_0 m)/tr(Σ_F)`. That ratio-of-traces is also basis-free and also well defined, and
    it is **not** this quantity: a ratio of sums is not the mean of the ratios, and on this panel the
    two differ by up to 3.76 nats. See the module docstring.

    Args:
        blocks: one `MisspecifiedColoredForward._blocks(F)` entry.
        sigma0: `(d, d)` the data covariance, float64.
        evecs: `(d, d)` **`S`'s** eigenvectors — the directions the average is taken over. For an
            isotropic arm any orthonormal basis gives the same answer, `m` and `Σ_F` being scalar.
    """
    # Promoted, and for a second reason besides precision: `MisspecifiedColoredForward` keeps
    # `sigma0` in float64 while casting the cached blocks to the apply dtype, so `m @ sigma0` is a
    # dtype mismatch unless one of them moves. It moves here, where the cost is a d x d cast.
    m = blocks["m"].double()
    sig = blocks["sigma"].double()
    U = evecs.double()
    num = torch.diagonal(U.transpose(-2, -1) @ (m @ sigma0.double() @ m) @ U)
    den = torch.diagonal(U.transpose(-2, -1) @ sig @ U)
    if not (torch.isfinite(num).all() and torch.isfinite(den).all()):
        raise ValueError("degenerate SNR: non-finite signal or noise power")
    if float(den.min()) <= 0.0:
        raise ValueError(f"degenerate SNR: noise power {float(den.min()):.3e} <= 0 on some band of "
                         "S; Sigma_F must be positive definite for F > 0")
    # 🛑 Signal power too, and BEFORE the caller takes a logarithm. `m Sigma_0 m` is PSD for a real
    # `Sigma_0`, so a materially negative diagonal in S's eigenbasis means `Sigma_0` was not a
    # covariance. Left unchecked this returned a negative SNR that `cond_feature` handed to
    # `math.log`, dying with a bare "math domain error" that named neither the arm, the level, nor
    # the quantity.
    #
    # ⚠️ The threshold is a DERIVED roundoff tolerance, not zero. As `F` grows `m -> 0`, so
    # `diag(U^T m Sigma_0 m U)` is a small positive number computed as a difference of larger ones
    # and legitimately lands at ~-2e-16. A hard `< 0` rejected the registered `cov` arm at large `F`
    # — the same error as clamping material negativity, in the opposite direction. Same scale- and
    # dtype-derived form as `metr_marginals.negativity_tolerance`.
    neg_tol = 32.0 * float(torch.finfo(torch.float64).eps) \
        * max(1.0, float(num.abs().max())) * int(num.numel())
    if float(num.min()) < -neg_tol:
        raise ValueError(f"degenerate SNR: signal power {float(num.min()):.3e} is below the "
                         f"roundoff tolerance -{neg_tol:.3e} on some band of S. m Sigma_0 m is PSD "
                         "for any covariance Sigma_0, so this means Sigma_0 is not one.")
    return float((num.clamp_min(0.0) / den).mean())


def cond_feature(blocks: dict[str, Tensor], sigma0: Tensor, evecs: Tensor, *,
                 dtype: torch.dtype | None = None, device=None) -> Tensor:
    """`ν(F) = log(mean_j SNR_j(F))`, shape `(1,)` — **the REFERENCE implementation.**

    Monotone decreasing in `F` (signal decays, noise grows), which is the only property the FiLM MLP
    needs from it.

    📌 **R3 (2026-09-02): this is no longer the path a training run takes.** It is kept, unchanged,
    as the dense reference the optimised :class:`BandSNR` is checked against and as the definition
    the algebra is stated in; `metr_arms.MetrColoredBody` now evaluates through `BandSNR`. Keeping
    the reference rather than deleting it is the point — an optimisation whose only description is
    its own code cannot be compared to anything.

    `dtype`/`device` default to `sigma0`'s, which is **not** what a caller wiring this into a body
    wants: `sigma0` is float64 by construction while the FiLM MLP runs at the training dtype. Pass
    the body's own parameter dtype and device — `metr_arms` does.
    """
    return torch.tensor([math.log(snr_bands(blocks, sigma0, evecs) + SNR_FLOOR)],
                        dtype=sigma0.dtype if dtype is None else dtype,
                        device=sigma0.device if device is None else device)


class BandSNR:
    r"""⭐ **R3: the conditioning feature as an `O(d)` reduction, with the guards discharged once.**

    `_blocks` builds `m = U diag(√e) Uᵀ` and `Σ_F = U diag(λ(1−e)) Uᵀ` on the *same* `U` the average
    is taken over, so in that basis both are diagonal and the band quantities collapse:

        num_j = (Uᵀ m Σ₀ m U)_jj = e_j · d_j,   d_j := (Uᵀ Σ₀ U)_jj   ← **`F`-INDEPENDENT**
        den_j = (Uᵀ Σ_F U)_jj    = λ_j (1 − e_j)
        SNR_j = e_j d_j / (λ_j (1 − e_j)),      e_j = exp(−F φ(λ_j))

    `d_j` is the whole `Σ₀` dependence and it does not move with `F`, so it is computed **once**
    here. What remains per evaluation is three `(d,)` elementwise ops and a mean — replacing four
    dense float64 `d × d` products, and needing **no `_blocks` call at all**.

    🛑 **The per-evaluation guards are DISCHARGED, not dropped.** `snr_bands` checks finiteness,
    positive noise power and non-negative signal power on every call, each through a `float(...)`
    that forces a host synchronisation. Every one of those follows from `F`-independent facts plus
    `F` lying in the registered window, so each is proved **here, at construction**, where the
    synchronisation is paid once:

    * `den_j = λ_j(1 − e_j) > 0` needs `λ_j > 0` (checked) and `e_j < 1` **strictly in float64**.
      `e_j = exp(−F φ_j) < 1` iff `F φ_j` exceeds the float64 rounding wall; the constructor checks
      `F_min · min_j φ_j` against a **four-ulp** wall (`4u = 8.88e-16`) and stores the achieved
      ratio in `roundoff_margin`. On the registered window the product is `0.02 × 0.077 = 1.54e-03`
      and the margin is **1.734e+12×**, so the case is not close.
    * `num_j = e_j d_j ≥ 0` needs `d_j ≥ 0`, checked here under the same derived roundoff tolerance
      `snr_bands` uses — and applied to `d_j` rather than to `e_j d_j`, which is *stricter*, since
      `e_j ≤ 1`.
    * finiteness of `e_j` follows from `φ_j` finite and `F` bounded, both checked.

    Removing a check that has been proved redundant is a different act from removing one that is
    merely inconvenient, and the difference is that this docstring can say which facts do the work.

    ⚠️ **It computes `1 − e`, not `−expm1(−x)`, and that is deliberate.** The accurate form would be
    a *better* approximation to an idealised `Σ_F` — but `_blocks` builds the arm's actual noise
    covariance as `λ · (1 − e)`, and the feature must describe the forward process the arm really
    runs. Measured over the registered window the two differ by at most **1.6e-14** relative, which
    is far inside the acceptance bound either way.
    """

    def __init__(self, evals: Tensor, evecs: Tensor, sigma0: Tensor, phi, *,
                 f_window: tuple[float, float]) -> None:
        lam = torch.as_tensor(evals).double().reshape(-1)
        U = torch.as_tensor(evecs).double()
        s0 = torch.as_tensor(sigma0).double()
        d = int(lam.numel())
        if U.shape != (d, d) or s0.shape != (d, d):
            raise ValueError(f"shapes disagree: evals ({d},), evecs {tuple(U.shape)}, "
                             f"sigma0 {tuple(s0.shape)}")
        if not torch.isfinite(lam).all():
            raise ValueError("eigenvalues are not all finite")
        if float(lam.min()) <= 0.0:
            raise ValueError(f"eigenvalues must be positive; min is {float(lam.min()):.3e}. The "
                             "noise power is lam*(1-e), so a non-positive eigenvalue makes it "
                             "non-positive at every level.")
        if not (torch.isfinite(U).all() and torch.isfinite(s0).all()):
            raise ValueError("evecs or sigma0 has non-finite entries")

        lo, hi = float(f_window[0]), float(f_window[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
            raise ValueError(f"f_window must be a finite 0 < lo <= hi; got {f_window!r}")
        phi_lam = torch.as_tensor(phi(lam)).double().reshape(-1)
        if phi_lam.numel() != d or not torch.isfinite(phi_lam).all():
            raise ValueError("phi(evals) is not a finite (d,) vector")
        if float(phi_lam.min()) <= 0.0:
            raise ValueError(f"phi must be positive on every band; min is "
                             f"{float(phi_lam.min()):.3e}. A non-positive rate gives e >= 1 and a "
                             "non-positive noise power.")
        # The rounding wall: `exp(-x) < 1` in float64 requires x above ~u = 2^-53. Four ulps of
        # margin, and the achieved margin is recorded rather than assumed.
        x_min = lo * float(phi_lam.min())
        wall = 4.0 * float(torch.finfo(torch.float64).eps)
        if not (x_min > wall):
            raise ValueError(f"F_min*min(phi) = {x_min:.3e} is at or below the float64 rounding "
                             f"wall {wall:.3e}: `1 - exp(-F*phi)` can round to zero inside the "
                             f"registered window, so the noise power is not provably positive and "
                             f"the per-evaluation check cannot be discharged here.")

        diag = torch.diagonal(U.transpose(-2, -1) @ s0 @ U)
        if not torch.isfinite(diag).all():
            raise ValueError("diag(U^T sigma0 U) has non-finite entries")
        # Same derived form as `snr_bands`'s signal-power tolerance, applied one step earlier: as
        # `F` grows `m -> 0` and the dense `num` is a difference of larger numbers landing at
        # ~-2e-16, so a hard `< 0` would reject the registered cov arm at large F.
        neg_tol = 32.0 * float(torch.finfo(torch.float64).eps)             * max(1.0, float(diag.abs().max())) * d
        if float(diag.min()) < -neg_tol:
            raise ValueError(f"diag(U^T sigma0 U) has an entry {float(diag.min()):.3e} below the "
                             f"roundoff tolerance -{neg_tol:.3e}. sigma0 is PSD for any covariance, "
                             "so this means it is not one.")
        self.d = d
        self.f_window = (lo, hi)
        self.evals = lam
        self.phi_lam = phi_lam
        self.sigma0_diag = diag.clamp_min(0.0)
        self.roundoff_margin = x_min / wall
        self.neg_tol = neg_tol

    @classmethod
    def from_forward(cls, fwd, *, f_window: tuple[float, float]) -> "BandSNR":
        """Build from a forward's own `(evals, evecs, sigma0, phi)`. **Takes no arm.**

        There is no `arm` or `name` argument anywhere in this class, and that is the mechanism
        rather than the intent: a feature that cannot see which arm it is conditioning cannot be
        computed differently for one of them. One definition for every arm is what justified the
        band-mean deviation in the first place (state §13.14 F2), so an arm-conditional fast path
        would give back exactly what the deviation bought.
        """
        missing = [a for a in ("evals", "evecs", "sigma0", "phi") if not hasattr(fwd, a)]
        if missing:
            raise ValueError(f"{type(fwd).__name__} has no {missing}: the band form needs S's own "
                             f"eigensystem and the data covariance, which the sealed commuting "
                             f"forward does not carry. This is a wiring error upstream.")
        return cls(fwd.evals, fwd.evecs, fwd.sigma0, fwd.phi, f_window=f_window)

    def snr_mean(self, F: float) -> Tensor:
        """`mean_j SNR_j(F)` as a **0-dim tensor**, with no device-to-host transfer.

        Returning a tensor rather than a `float` is the point: `float(...)` on a CUDA tensor is a
        synchronisation, and this runs on every body evaluation.
        """
        F = float(F)
        lo, hi = self.f_window
        if not (math.isfinite(F) and lo <= F <= hi):
            raise ValueError(f"F={F!r} is outside the window [{lo:g}, {hi:g}] this object "
                             f"discharged its guards on. Outside it the noise power is not "
                             "provably positive here, and evaluating anyway would be trusting a "
                             "proof of a different statement.")
        e = torch.exp(-F * self.phi_lam)
        return (self.sigma0_diag * e / (self.evals * (1.0 - e))).mean()

    def feature(self, F: float, *, dtype: torch.dtype | None = None, device=None) -> Tensor:
        """`ν(F) = log(mean_j SNR_j + SNR_FLOOR)`, shape `(1,)` — same value as `cond_feature`."""
        nu = torch.log(self.snr_mean(F) + SNR_FLOOR)
        return nu.to(dtype=self.evals.dtype if dtype is None else dtype,
                     device=self.evals.device if device is None else device).reshape(1)
