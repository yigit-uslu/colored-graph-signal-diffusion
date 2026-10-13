"""Shared machinery for the example training runs in `examples/`, built from `src/ccd` alone.

The two scripts beside this file, `train_fmri.py` and `train_metr_la.py`, each train one colored
covariance diffusion model on a real panel, draw samples with the two samplers the paper reports,
and score them. Everything they need is here and in `src/ccd`. The one exception is METR-LA's graph
arm (`--gso laplacian`), which uses the paper's own `scripts/metr_marginals.py` and
`scripts/metr_regime.py`.

**These are examples, not the paper's experiment drivers.** The drivers in `scripts/` run under
sealed protocols: their random streams derive from a protocol id, they refuse to start without one,
and they gate every arm before it trains. The examples keep the model, the clock rule and the
training recipe of those runs and drop the protocol machinery, so their numbers are not comparable
to the paper's digit for digit. Every `run.json` says so (`is_study_result: false`) and lists how
the run differs from the paper's (`deviations_from_paper`).

A run, in order:

1. `S = Ĉ + τI` from the training frames (unbiased covariance), and its eigensystem.
2. The clock: `ShiftedPowerRate`, pinned at `φ(λ_max) = FLOOR` with range `R`. It is labelled by its
   realised `(R, ā, spearman)` rather than the nominal exponent (`docs/experiment_conventions.md`
   §1), and the run reports how many bands reach unit SNR outside `[F_min, F_max]`.
3. The dense colored arm: one validated `SpectralBackend` behind the forward process, the EDM
   preconditioners and both samplers, and a `VNNBody` (3 TAGConv layers, K = 5, width 96, Mehler
   envelope on).
4. Training on the colored EDM loss: Adam, one log-uniform `F` per minibatch, gradient clipping,
   and an EMA of the weights.
5. Sampling with the EMA weights: `ancestral` (the exponential reverse SDE) and `heun_pfode` (the
   probability-flow ODE), on a geometric `F` grid, from `N(0, S)`, plus the endpoint residual
   `c_out(S, F_min) ξ`.
6. Scoring: `cov_err = ‖Ĉ_gen − Ĉ_ref‖_F / ‖Ĉ_ref‖_F` for each cloud, beside two baselines: exact
   Gaussian clouds `N(0, Ĉ_train)` of the same size (the floor: what perfect second moments score,
   i.e. pure sampling noise), and the untrained analytic head on the same random draws (what the
   colored prior and the sampler give for free). `cov_err` is a second-moment check. The colored
   prior gets second moments right by construction, so a trained model near the floor shows the
   pipeline works, not that training learned anything; the paper's tables add descriptor MMDs for
   that.

`--family vp` trains the paper's standard-diffusion baseline instead (`ccd.vp`), as its drivers
build it (`scripts/fmri_lowstep_train.py`, `scripts/metr_arms.py`): the same body reading the same
operator, seeded identically, but scalar preconditioning `D = α x + σ body(x)`, the loss `vp_loss`,
and no clock. Each colored `F` is mapped to VP's noise level by `VPLevelMap`, which matches the flat
colored clock's population signal-to-noise, so both families train and sample over one window and
one grid. Sampling starts from `N(0, I)` and adds `σ(r_min) ξ`; `z` and `ξ` are the same draws a
colored run of the same seed uses. VP's untrained network is `α x`, which is not a Gaussian
reference, so its paired row samples with `GaussianOracleDenoiser(Ĉ_train)` instead.

`--gso laplacian` (METR-LA only) replaces `S = Ĉ + τI` with the paper's graph prior
`S = σ²(L_sym + γI)⁻¹`, `σ²` matched to `tr(Ĉ + τI)`. The data still have covariance `Ĉ`, which
does not commute with this `S`, so the colored arm takes `Ĉ` explicitly: the exact marginals and
preconditioners of `metr_marginals.MisspecifiedColoredForward` / `MisspecifiedColoredEDM`, and
`metr_regime`'s band-mean loss weight and FiLM feature, as the paper's driver wires them. Its clock
is weighted by the training variance along `S`'s eigenvectors. A VP run on this operator only reads
it as its graph.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# This checkout's `src`, ahead of any installed copy, so an example runs the code it ships with.
sys.path.insert(0, str(ROOT / "src"))

from ccd.denoiser import (  # noqa: E402
    ColoredDenoiser,
    ColoredEDM,
    VNNBody,
    cout_sq_bands,
    edm_loss,
    sample_F_loguniform,
)
from ccd.forward import ColoredForward, ShiftedPowerRate, snr  # noqa: E402
from ccd.forward.clock_audit import band_weights, f_star  # noqa: E402
from ccd.operators import CovarianceOperator  # noqa: E402
from ccd.operators.backends import SpectralBackend  # noqa: E402
from ccd.reverse import LearnedReversePFODE, LearnedReverseSDE, reverse_f_grid  # noqa: E402
from ccd.vp import (  # noqa: E402
    GaussianOracleDenoiser,
    GraphOnlyForward,
    VPAncestralSampler,
    VPBody,
    VPDenoiser,
    VPLevelMap,
    VPReversePFODE,
    vp_loss,
    vp_sigma,
)

FAMILIES = ("colored", "vp")
#: The graph shift operator: the covariance `Ĉ + τI`, or METR-LA's Laplacian prior.
GSOS = ("covariance", "laplacian")
SAMPLERS = ("ancestral", "heun_pfode")
LEVEL_AVERAGES = ("grid", "bands")
#: Recipe fields that set the colored clock and loss weighting; VP has neither.
COLORED_ONLY = {"a": "--a", "R": "--R", "level_average": "--level-average"}
#: float32 bound for the eigensystem check against the operator; the paper's fMRI arms use the same.
VALIDATE_RTOL = 1e-6

#: Every run differs from the paper's in these ways; each script appends its panel's own.
COMMON_DEVIATIONS = (
    "One training seed, one model and its final EMA weights, one cloud per sampler and step "
    "count. The paper's tables aggregate over seeds (and, on METR-LA, folds) and over many clouds.",
    "Random streams derive from --seed alone, not from a sealed protocol id, so no number here is "
    "bit-comparable with a paper run.",
    "Scored by cov_err only, beside an exact-Gaussian floor and the exact Gaussian denoiser; the "
    "paper's tables also report aMMD over graph descriptors.",
)

#: How each family's wiring differs from the paper driver that built it.
FAMILY_DEVIATIONS = {
    "colored": (
        "The arm is wired from src/ccd directly (on a graph operator, also from the paper's "
        "scripts/metr_marginals.py and scripts/metr_regime.py), with the same dense SpectralBackend "
        "construction as the paper's drivers but without their gates (dataset binding, object-graph "
        "and coverage checks).",
    ),
    "vp": (
        "The VP arm is wired from src/ccd directly, as scripts/fmri_lowstep_train.py and "
        "scripts/metr_arms.py build it, without those drivers' gates.",
    ),
}


@dataclass(frozen=True)
class Recipe:
    """Everything that defines a run. The defaults are the paper's fMRI configuration."""

    family: str = "colored"             # "colored", or "vp" for the standard-diffusion baseline
    gso: str = "covariance"             # "covariance" (S = Ĉ + τI) or "laplacian" (METR-LA's prior)
    gamma: float | None = None          # the Laplacian prior's γ in σ²(L_sym + γI)⁻¹; None otherwise
    a: float = -1.0                     # nominal clock exponent: a < 0, or 0 for the flat clock
    R: float = 120.0                    # clock range max φ / min φ (R = 1 when a = 0)
    floor: float = 0.077                # Φ: the rate at the smoothest band, λ_max
    F_min: float = 0.02
    F_max: float = -2.0 * math.log(0.02) / 0.077
    level_average: str = "grid"         # "grid" (fMRI's regime) or "bands" (METR-LA's)
    width: int = 96
    n_layers: int = 3
    K_gf: int = 5
    mehler_envelope: bool = True
    k0: float = 2.0
    updates: int = 2000
    batch: int = 256
    lr: float = 1e-3
    grad_clip: float = 500.0
    ema_decay: float = 0.999
    seed: int = 0


@dataclass
class Panel:
    """A dataset as the examples see it."""

    name: str
    x_train: np.ndarray                 # (n, d) the frames the model trains on
    references: dict[str, np.ndarray]   # name -> (m, d) frames a generated cloud is scored against
    tau: float
    source: dict[str, Any]              # where the frames came from, recorded in run.json
    L_sym: np.ndarray | None = None     # (d, d) normalised Laplacian, for --gso laplacian


def stream_seed(seed: int, *parts: object) -> int:
    """A 63-bit seed for one named random stream, so adding a stream never shifts another."""
    key = repr((int(seed),) + tuple(str(p) for p in parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)


# --------------------------------------------------------------------------- operator and clock


@dataclass
class Operator:
    op: CovarianceOperator
    tau: float              # the operator's ridge: S − τI is what the forward process treats as signal
    mean: np.ndarray        # (d,) training mean; generated clouds are shifted back by it
    C: np.ndarray           # (d, d) unbiased training covariance, float64
    evals: np.ndarray       # (d,) eigenvalues of S, ascending, float64
    evecs: np.ndarray       # (d, d)
    Xc: torch.Tensor        # (d, n) centred training frames on the device: the minibatch source
    C_factor: np.ndarray    # (d, d) with C = C_factor C_factorᵀ: draws exact N(0, C) clouds
    gso: str = "covariance"
    graph: dict[str, Any] | None = None             # the Laplacian prior's scalars
    band_variance: np.ndarray | None = None         # the Laplacian clock's band weights, S's order


def _centred(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(mean, xc, Ĉ)`: the training mean, the `(d, n)` centred frames and their unbiased covariance."""
    x = np.asarray(x_train, dtype=np.float64)
    mean = x.mean(0)
    xc = (x - mean).T                                       # (d, n)
    return mean, xc, xc @ xc.T / (x.shape[0] - 1)


def build_operator(x_train: np.ndarray, tau: float, *, device: torch.device | str,
                   dtype: torch.dtype = torch.float32) -> Operator:
    """`S = Ĉ + τI` with `Ĉ` the unbiased covariance of the training frames."""
    mean, xc, C = _centred(x_train)
    d, n = xc.shape
    evals, evecs = np.linalg.eigh(C + tau * np.eye(d))
    # `CovarianceOperator` applies X Xᵀ/n + τI, the BIASED covariance. Scaling its factor by
    # √(n/(n−1)) makes it apply the unbiased Ĉ above. Fed the raw frames it would build a different
    # matrix (off by 1/n relative), and the dense route would then be exact for the wrong one.
    factor = torch.as_tensor(xc * math.sqrt(n / (n - 1)), dtype=dtype, device=device)
    return Operator(op=CovarianceOperator(factor, tau=tau), tau=float(tau), mean=mean, C=C,
                    evals=evals, evecs=evecs,
                    Xc=torch.as_tensor(xc, dtype=dtype, device=device),
                    # Ĉ = U diag(λ − τ) Uᵀ up to rounding, so this factor needs no Cholesky, which a
                    # rank-deficient Ĉ (fMRI: rank 246 of 360) does not have.
                    C_factor=evecs * np.sqrt(np.clip(evals - tau, 0.0, None)))


#: `eigh` cannot tell apart eigenvalues closer than its backward error, `~d·u·‖L‖₂`, and returns an
#: arbitrary basis inside such a cluster. The paper's rule groups consecutive gaps below
#: `8·d·u·‖L‖₂`; on METR-LA's `L_sym` that isolates its two null modes (gap 7.9e-16) and nothing
#: else (smallest true gap 1.5e-4).
DEGENERACY_FACTOR = 8.0


def degenerate_clusters(evals: np.ndarray) -> list[slice]:
    """Runs of ASCENDING eigenvalues within `eigh`'s resolution of one another."""
    w = np.asarray(evals, dtype=np.float64)
    if np.any(np.diff(w) < 0.0):
        raise ValueError("eigenvalues must be ascending, as numpy.linalg.eigh returns them")
    tol = DEGENERACY_FACTOR * w.size * 2.0 ** -53 * float(np.abs(w).max())
    out, i = [], 0
    while i < w.size:
        j = i + 1
        while j < w.size and w[j] - w[j - 1] <= tol:
            j += 1
        if w[j - 1] - w[i] > tol:
            # A chain of small steps wider than the tolerance: where to cut it would be arbitrary.
            raise ValueError(f"eigenvalues [{i}:{j}] chain across {w[j - 1] - w[i]:.3e} > {tol:.3e}")
        out.append(slice(i, j))
        i = j
    return out


def build_graph_operator(x_train: np.ndarray, tau: float, L_sym: np.ndarray, gamma: float, *,
                         device: torch.device | str,
                         dtype: torch.dtype = torch.float32) -> Operator:
    """METR-LA's graph prior `S = σ²(L_sym + γI)⁻¹`, with `σ² = tr(Ĉ + τI) / tr((L_sym + γI)⁻¹)`.

    Matching traces makes the operator's ridge `τ_g = (tr S − tr Ĉ)/d` equal `τ` in exact arithmetic;
    it must lie in `(0, λ_min(S))` so that `S − τ_g I` is a covariance and the operator factors.
    `Ĉ` stays the data's covariance, which this `S` does not commute with.

    The clock's band weights are the training variance along each eigenvector of `S`,
    `diag(Uᵀ Ĉ U)`, averaged within `L_sym`'s degenerate eigenspaces. Inside one, `eigh` picks an
    arbitrary basis and the raw variances split arbitrarily; their mean does not.
    """
    mean, xc, C = _centred(x_train)
    d = xc.shape[0]
    L = np.asarray(L_sym, dtype=np.float64)
    if L.shape != (d, d) or not np.isfinite(L).all():
        raise ValueError(f"L_sym must be a finite ({d}, {d}) matrix; got shape {L.shape}")
    Lw, Lv = np.linalg.eigh(0.5 * (L + L.T))
    if not float(Lw.min()) + gamma > 0.0:
        raise ValueError(f"L_sym + gamma I is not positive definite: lambda_min(L_sym) = "
                         f"{float(Lw.min()):.3e}, gamma = {gamma}")
    g = 1.0 / (Lw + gamma)
    sigma2 = float((np.trace(C) + d * tau) / g.sum())
    raw = np.einsum("ij,ik,kj->j", Lv, C, Lv)               # diag(Lvᵀ Ĉ Lv)
    pooled = raw.copy()
    clusters = degenerate_clusters(Lw)
    for sl in clusters:
        pooled[sl] = raw[sl].mean()
    order = np.argsort(g, kind="stable")                   # ascending in S, as eigh returns it
    evals, evecs, band_variance = sigma2 * g[order], Lv[:, order], pooled[order]
    tau_g = float((evals.sum() - np.trace(C)) / d)
    if not 0.0 < tau_g < float(evals.min()):
        raise ValueError(f"tau_g = {tau_g:.6g} is not in (0, lambda_min(S) = {float(evals.min()):.6g}); "
                         "S - tau_g I would not be a covariance")
    # n = d columns, so the operator's X Xᵀ/n + τ_g I is U diag(λ − τ_g) Uᵀ + τ_g I = S exactly.
    factor = torch.as_tensor(evecs * np.sqrt((evals - tau_g) * d), dtype=dtype, device=device)
    c_evals, c_evecs = np.linalg.eigh(C)
    graph = {"operator": "S = sigma2 (L_sym + gamma I)^-1, tr S = tr(C_train + tau I)",
             "gamma": float(gamma), "sigma2": sigma2, "tau_g": tau_g, "tau_g_minus_tau": tau_g - tau,
             "lam_min": float(evals.min()), "lam_max": float(evals.max()),
             "kappa": float(evals.max() / evals.min()),
             "n_null_modes": int((Lw < 1e-9).sum()),
             "degenerate_cluster_sizes": [s.stop - s.start for s in clusters if s.stop - s.start > 1]}
    return Operator(op=CovarianceOperator(factor, tau=tau_g), tau=tau_g, mean=mean, C=C,
                    evals=evals, evecs=evecs, Xc=torch.as_tensor(xc, dtype=dtype, device=device),
                    C_factor=c_evecs * np.sqrt(np.clip(c_evals, 0.0, None)),
                    gso="laplacian", graph=graph, band_variance=band_variance)


@dataclass
class Clock:
    phi: Callable[[torch.Tensor], torch.Tensor]
    labels: dict[str, Any]
    coverage: dict[str, Any]


def build_clock(recipe: Recipe, S: Operator) -> Clock:
    """The clock, its `(R, ā, spearman)` label, and where each live band reaches unit SNR."""
    from scipy.stats import spearmanr

    lam_min, lam_max = float(S.evals.min()), float(S.evals.max())
    if recipe.a == 0.0:
        floor = recipe.floor

        def phi(lam: torch.Tensor) -> torch.Tensor:          # flat: R = 1 by definition
            return torch.full_like(torch.as_tensor(lam), floor)
        R, shift = 1.0, None
    elif recipe.a < 0.0:
        phi = ShiftedPowerRate(recipe.a, lam_min, lam_max, FLOOR=recipe.floor,
                               F_min=recipe.F_min, R=recipe.R, tau=S.tau)
        R, shift = phi.R, phi.s
    else:
        raise ValueError(f"a must be < 0 (coarse-to-fine on S = Ĉ + τI) or 0; got {recipe.a}")

    phi_vals = phi(torch.as_tensor(S.evals, dtype=torch.float64)).numpy()
    if S.gso == "laplacian":
        # The data's variance along each of S's directions, which here is not λ − τ. Unit SNR along
        # direction j, e_j·var_j / (λ_j (1 − e_j)) = 1 with e_j = exp(−F φ_j), is reached at
        # F*_j = ln(1 + var_j/λ_j)/φ_j; with var_j = λ_j − τ that is the covariance arm's f_star.
        # Normalised to shares, as `band_weights` returns them: the coverage report sums them.
        var_w = S.band_variance / S.band_variance.sum()
        live = var_w > 0.0
        fs_all = np.log1p(S.band_variance / S.evals) / phi_vals
        weights_note = ("training variance along S's eigenvectors, averaged within L_sym's "
                        "degenerate eigenspaces")
    else:
        # The covariance arm's band variance is λ − τ, snapped to zero on ridge-only bands. The
        # identity stands in for the Laplacian argument, which only the QV weights (unused here)
        # depend on.
        var_w, _, live, _ = band_weights(S.evals, S.evecs, np.eye(S.evals.size), tau=S.tau)
        fs_all = f_star(phi_vals, S.evals, tau=S.tau)
        weights_note = "band variance lambda - tau, snapped to 0 on ridge-only bands"
    abar = 0.0 if recipe.a == 0.0 else phi.weighted_a(S.evals, var_w)
    rho = (None if recipe.a == 0.0
           else float(spearmanr(phi_vals[live], var_w[live])[0]))
    labels = {
        "gso": S.gso, "a_nominal": recipe.a, "R": R, "abar": abar, "spearman_phi_bandvar": rho,
        "shift_s": shift, "floor": recipe.floor, "F_min": recipe.F_min, "F_max": recipe.F_max,
        "lam_min": lam_min, "lam_max": lam_max, "kappa": lam_max / lam_min,
        "abar_weights": weights_note, "graph": S.graph,
    }

    fs = fs_all[live]
    w = var_w[live]
    below, above = fs < recipe.F_min, fs > recipe.F_max
    coverage = {
        "bands": int(S.evals.size), "live_bands": int(live.sum()),
        "below_F_min": int(below.sum()), "above_F_max": int(above.sum()),
        "variance_share_below_F_min": float(w[below].sum()),
        "variance_share_above_F_max": float(w[above].sum()),
        "F_star_min": float(fs.min()), "F_star_max": float(fs.max()),
    }
    return Clock(phi=phi, labels=labels, coverage=coverage)


# --------------------------------------------------------------------------- the model


class BandMeanVNNBody(VNNBody):
    """`VNNBody` whose FiLM feature averages the SNR over the operator's bands, not a λ-grid.

    `VNNBody` takes `ν(F) = log mean SNR_λ(F)` over a uniform grid on `[λ_min, λ_max]`. On METR-LA
    (κ ≈ 1,885) that grid puts nearly all of its points where no eigenvalue lies, so the paper's
    METR-LA runs average this feature and the EDM loss weight over the actual bands instead
    (`scripts/metr_regime.py`). On a covariance arm the formula is unchanged; only the points
    averaged over move.
    """

    def __init__(self, fwd: ColoredForward, band_evals: np.ndarray, **kw: Any):
        super().__init__(fwd, **kw)
        self.band_evals = torch.as_tensor(np.asarray(band_evals), dtype=torch.float64)

    def _cond_feature(self, F: float) -> torch.Tensor:
        p = self.encoder.weight
        s = snr(self.band_evals.to(p.device), F, self.fwd.phi, self.fwd.op.tau)
        return torch.log(s.mean() + 1e-30).to(p.dtype).reshape(1)


def _metr_modules():
    """The paper's graph-arm modules, `scripts/metr_marginals.py` and `scripts/metr_regime.py`.

    Imported only when the graph arm is built, and from this checkout's `scripts/`, which is put on
    the path here rather than at import time so no other example sees the drivers' module names.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import metr_marginals
    import metr_regime
    return metr_marginals, metr_regime


class RegimeVNNBody(VNNBody):
    """`VNNBody` whose FiLM feature is `metr_regime`'s, for a forward whose `Σ_0` is not `S − τI`.

    `ν(F) = log mean_j SNR_j(F)` over `S`'s eigen-directions, with `SNR_j = e_j·(Uᵀ Σ_0 U)_jj /
    (λ_j (1 − e_j))`: the band mean of `BandMeanVNNBody`, which it reduces to when `Σ_0 = S − τI`, but
    defined when `Σ_0` does not commute with `S`. The paper's graph arm builds it the same way
    (`metr_regime.BandSNR`, valid on the training window only).
    """

    def __init__(self, fwd: ColoredForward, *, f_window: tuple[float, float], **kw: Any):
        super().__init__(fwd, **kw)
        self.f_window = (float(f_window[0]), float(f_window[1]))

    @property
    def band_snr(self):
        # Built on first use rather than in __init__, as the paper's body does, so the constructor
        # never reaches for the forward's eigensystem before the forward is fully wired.
        cached = getattr(self, "_band_snr", None)
        if cached is None:
            cached = _metr_modules()[1].BandSNR.from_forward(self.fwd, f_window=self.f_window)
            self._band_snr = cached
        return cached

    def _cond_feature(self, F: float) -> torch.Tensor:
        p = self.encoder.weight
        return self.band_snr.feature(F, dtype=p.dtype, device=p.device)


@dataclass
class Model:
    fwd: ColoredForward
    edm: ColoredEDM
    body: VNNBody
    denoiser: ColoredDenoiser
    sde: LearnedReverseSDE
    pfode: LearnedReversePFODE
    S_sqrt: torch.Tensor                # U diag(√λ) Uᵀ: the terminal draw is S_sqrt @ z
    level_average: str
    band_evals: torch.Tensor            # float64, CPU
    gso: str = "covariance"


def build_model(recipe: Recipe, S: Operator, clock: Clock, *, device: torch.device | str,
                dtype: torch.dtype = torch.float32) -> Model:
    """The dense colored arm: one validated eigensystem behind every spectral apply."""
    if recipe.level_average not in LEVEL_AVERAGES:
        raise ValueError(f"level_average must be one of {LEVEL_AVERAGES}; got {recipe.level_average!r}")
    evals = torch.as_tensor(S.evals, dtype=dtype, device=device)
    evecs = torch.as_tensor(S.evecs, dtype=dtype, device=device)
    # The backend checks UᵀU ≈ I and S U ≈ U diag(λ) against the operator before anything uses it.
    # On the fMRI operator (κ ≈ 19,700) the matrix-free Chebyshev default does not resolve these
    # functions, so leaving any injection point on it would lose accuracy silently.
    backend = SpectralBackend(evals, evecs, validate_against=S.op, rtol=VALIDATE_RTOL)
    lam_min, lam_max = float(S.evals.min()), float(S.evals.max())
    kw = dict(width=recipe.width, n_layers=recipe.n_layers, K_gf=recipe.K_gf,
              mehler_envelope=recipe.mehler_envelope, k0=recipe.k0)
    if S.gso == "laplacian":
        if recipe.level_average != "bands":
            raise ValueError("the graph arm's loss weight and FiLM feature are metr_regime's band "
                             "means; the grid forms assume S = C + tau I")
        marginals, _ = _metr_modules()
        # Σ_0 = Ĉ explicitly. The dense blocks are built in float64 (the class refuses anything
        # else) and cast to the operator's dtype on the way out.
        f64 = dict(dtype=torch.float64, device=device)
        fwd = marginals.MisspecifiedColoredForward(
            S.op, lam_min, lam_max, phi=clock.phi, sigma0=torch.as_tensor(S.C, **f64),
            evals=torch.as_tensor(S.evals, **f64), evecs=torch.as_tensor(S.evecs, **f64),
            backend=backend)
        edm = marginals.MisspecifiedColoredEDM(fwd)
        torch.manual_seed(stream_seed(recipe.seed, "body_init"))
        body = RegimeVNNBody(fwd, f_window=(recipe.F_min, recipe.F_max), **kw)
    else:
        fwd = ColoredForward(S.op, lam_min, lam_max, phi=clock.phi, backend=backend)
        edm = ColoredEDM(fwd, precond_backend=backend, solve_backend=backend)
        torch.manual_seed(stream_seed(recipe.seed, "body_init"))
        body = (BandMeanVNNBody(fwd, S.evals, **kw) if recipe.level_average == "bands"
                else VNNBody(fwd, **kw))
    denoiser = ColoredDenoiser(edm, body)
    return Model(
        fwd=fwd, edm=edm, body=body, denoiser=denoiser,
        # `c_in_kw={}`: ColoredEDM refuses solver kwargs once a solve backend is set.
        sde=LearnedReverseSDE(denoiser, c_in_kw={}, backend=backend),
        pfode=LearnedReversePFODE(denoiser, c_in_kw={}, backend=backend),
        S_sqrt=(evecs * evals.clamp_min(0.0).sqrt()) @ evecs.T,
        level_average=recipe.level_average,
        band_evals=torch.as_tensor(S.evals, dtype=torch.float64),
        gso=S.gso,
    )


@dataclass
class VPModel:
    """The standard-diffusion (VP) baseline: `D(x, r) = α(r) x + σ(r) body(x, r)`, `c_in = 1`."""

    body: VPBody
    denoiser: VPDenoiser
    level_map: VPLevelMap               # colored F -> VP's noise-to-signal ratio r
    ancestral: VPAncestralSampler
    pfode: VPReversePFODE
    C: torch.Tensor                     # Ĉ_train, float64, CPU: the Gaussian denoiser's covariance


def build_vp_model(recipe: Recipe, S: Operator, *, device: torch.device | str,
                   dtype: torch.dtype = torch.float32) -> VPModel:
    """VP as the paper's drivers build it (`scripts/fmri_lowstep_train.py`, `scripts/metr_arms.py`)."""
    # The body reads the covariance operator as its graph, as the colored body does, and is seeded
    # from the same stream, so the two families start from identical weights and differ in the
    # diffusion alone.
    gof = GraphOnlyForward(S.op, float(S.evals.min()), float(S.evals.max()))
    torch.manual_seed(stream_seed(recipe.seed, "body_init"))
    body = VPBody(gof, width=recipe.width, n_layers=recipe.n_layers, K_gf=recipe.K_gf,
                  mehler_envelope=recipe.mehler_envelope, k0=recipe.k0).to(device=device, dtype=dtype)
    denoiser = VPDenoiser(body)
    # φ₀ = FLOOR because the flat colored clock is the constant FLOOR; the traces are the panel's own.
    level_map = VPLevelMap(phi0=recipe.floor, tr_C=float(np.trace(S.C)), tr_S=float(S.evals.sum()))
    return VPModel(body=body, denoiser=denoiser, level_map=level_map,
                   ancestral=VPAncestralSampler(denoiser), pfode=VPReversePFODE(denoiser, level_map),
                   C=torch.as_tensor(S.C, dtype=torch.float64))


def loss_weight(model: Model, F: float) -> float | None:
    """`None` keeps `edm_loss`'s grid weight; `"bands"` uses `1 / mean_k c_out²(λ_k, F)`.

    On the graph arm there are no per-band `c_out²(λ)`, and the weight is `metr_regime`'s
    `1 / (tr c_out²(F) / d)` on the exact posterior covariance, which reduces to the band mean when
    `Σ_0 = S − τI`.
    """
    if model.gso == "laplacian":
        return _metr_modules()[1].edm_weight(model.fwd._blocks(float(F)))
    if model.level_average == "grid":
        return None
    mean = float(cout_sq_bands(model.fwd, F, model.band_evals).mean())
    if not mean > 0.0:
        raise ValueError(f"mean posterior variance {mean:.3e} at F={F} is not positive")
    return 1.0 / mean


# --------------------------------------------------------------------------- training


class EMA:
    """Exponential moving average of the body's weights, with the usual `(1+t)/(10+t)` warm-up.

    It keeps a separate shadow copy, so training is the same with or without it; only what is
    sampled and saved changes.
    """

    def __init__(self, params: list[torch.Tensor], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must lie in (0, 1); got {decay}")
        self.decay, self.steps = float(decay), 0
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self, params: list[torch.Tensor]) -> None:
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for s, p in zip(self.shadow, params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        self.steps += 1

    @contextlib.contextmanager
    def applied(self, params: list[torch.Tensor]):
        """Swap the averaged weights in for a block, then restore the raw ones exactly."""
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, s in zip(params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)


def _loss(model: Model | VPModel, x0: torch.Tensor, F: float,
          generator: torch.Generator) -> torch.Tensor:
    """The colored EDM loss at `F`, or VP's at the matched level `r(F)`."""
    if isinstance(model, VPModel):
        return vp_loss(model.denoiser, x0, model.level_map.r_of_F(F), generator=generator)
    return edm_loss(model.denoiser, x0, F, generator=generator, weight=loss_weight(model, F))


def train(model: Model | VPModel, S: Operator, recipe: Recipe, *, device: torch.device | str,
          log_every: int = 200, log: Callable[[str], None] = print) -> tuple[dict[str, Any], EMA]:
    """`recipe.updates` Adam updates, one log-uniform `F` per minibatch, on the family's own loss.

    Minibatches walk a fresh permutation of the frames each epoch, so every frame is seen once per
    epoch; the last batch of an epoch is the remainder.
    """
    params = [p for p in model.body.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=recipe.lr)
    ema = EMA(params, recipe.ema_decay)
    g_order = torch.Generator().manual_seed(stream_seed(recipe.seed, "minibatch_order"))
    g_level = torch.Generator().manual_seed(stream_seed(recipe.seed, "noise_level"))
    g_noise = torch.Generator(device=device).manual_seed(stream_seed(recipe.seed, "corruption"))
    n = int(S.Xc.shape[1])
    batch = min(int(recipe.batch), n)

    losses: list[float] = []
    norms: list[float] = []
    rows: list[dict[str, Any]] = []
    pending: list[torch.Tensor] = []
    model.denoiser.train()
    t0 = time.perf_counter()
    for step in range(1, recipe.updates + 1):
        if not pending:
            pending = list(torch.randperm(n, generator=g_order).split(batch))
        idx = pending.pop(0)
        x0 = S.Xc[:, idx.to(S.Xc.device)]
        F = sample_F_loguniform(recipe.F_min, recipe.F_max, generator=g_level)
        loss = _loss(model, x0, F, g_noise)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at update {step} (F={F:.4g})")
        opt.zero_grad(set_to_none=True)
        loss.backward(inputs=params)
        norm = float(torch.nn.utils.clip_grad_norm_(params, recipe.grad_clip))
        if not math.isfinite(norm):
            raise RuntimeError(f"non-finite gradient norm at update {step} (F={F:.4g})")
        opt.step()
        ema.update(params)
        losses.append(float(loss.detach()))
        norms.append(norm)
        if step % log_every == 0 or step == recipe.updates:
            window = slice(len(losses) - min(log_every, len(losses)), None)
            w_loss, w_norm = np.asarray(losses[window]), np.asarray(norms[window])
            row = {"update": step, "loss_median": float(np.median(w_loss)),
                   "loss_max": float(w_loss.max()), "grad_norm_median": float(np.median(w_norm)),
                   "clipped": int((w_norm > recipe.grad_clip).sum()),
                   "seconds": time.perf_counter() - t0}
            rows.append(row)
            log(f"  update {step:>7,}/{recipe.updates:,}   loss median {row['loss_median']:.4g} "
                f"(max {row['loss_max']:.3g})   pre-clip |grad| median {row['grad_norm_median']:.3g}"
                f"   clipped {row['clipped']}/{len(w_loss)}   {row['seconds']:.0f} s")
    n_clipped = int((np.asarray(norms) > recipe.grad_clip).sum())
    if n_clipped:
        # Measured, not assumed: of the paper's 25 colored fMRI runs (protocol 135f8e67...), 13 had
        # at least one clipped update over 100,000 updates and 2 within their first 1,000.
        log(f"  {n_clipped} of {recipe.updates:,} updates had a pre-clip gradient norm above "
            f"{recipe.grad_clip:g}. The paper's colored runs spike like this too: 13 of 25 fMRI runs "
            "clipped at least once over 100,000 updates.")
    history = {
        "updates": recipe.updates, "seconds": time.perf_counter() - t0, "log": rows,
        "clipped_updates": n_clipped,
        "note": "loss is the family's weighted loss at a random F per update, so it is noisy by "
                "design; windows report the median and the max because rare spikes dominate a mean",
    }
    return history, ema


# --------------------------------------------------------------------------- sampling and scoring


@torch.no_grad()
def sample(model: Model | VPModel, recipe: Recipe, sampler: str, steps: int, n_samples: int, *,
           device: torch.device | str) -> np.ndarray:
    """One `(n_samples, d)` cloud, float64, in the centred coordinates the model trained on."""
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}; got {sampler!r}")
    vp = isinstance(model, VPModel)
    d = int(model.C.shape[0]) if vp else int(model.S_sqrt.shape[0])
    dtype = next(model.body.parameters()).dtype if vp else model.S_sqrt.dtype
    g = torch.Generator(device=device).manual_seed(stream_seed(recipe.seed, "sample", sampler, steps))
    z = torch.randn(d, n_samples, generator=g, dtype=dtype, device=device)
    xi = torch.randn(d, n_samples, generator=g, dtype=dtype, device=device)
    # Geometric spacing: the PF-ODE is unstable on arc-length grids for a < 0, and a φ-dependent
    # grid would give every clock a different one. VP walks the same grid, mapped to its own r.
    grid = [float(f) for f in reverse_f_grid(recipe.F_max, recipe.F_min, steps, spacing="geometric")]
    model.denoiser.eval()
    if vp:
        x_T = z                                             # VP's terminal law is N(0, I)
        if sampler == "ancestral":
            x = model.ancestral.integrate(x_T, model.level_map.r_grid(grid), generator=g)
        else:
            x = model.pfode.integrate(x_T, grid, scheme="heun")
        # The same endpoint policy as the colored family, with VP's scalar residual σ(r_min).
        x = x + vp_sigma(model.level_map.r_of_F(grid[-1])) * xi
    else:
        x_T = model.S_sqrt @ z                              # the terminal law N(0, S), not N(0, I)
        if sampler == "ancestral":
            x = model.sde.integrate(x_T, grid, scheme="ancestral", generator=g)
        else:
            x = model.pfode.integrate(x_T, grid, scheme="heun")
        # Both integrators end on the denoised mean D(x_Fmin, F_min). The paper's endpoint policy
        # adds the posterior residual c_out(S, F_min) ξ so the cloud keeps the variance left there.
        x = x + model.edm.c_out(xi, grid[-1])
    cloud = x.T.contiguous().to(torch.float64).cpu().numpy()
    if not np.isfinite(cloud).all():
        raise RuntimeError(f"non-finite samples from {sampler} at {steps} steps")
    return cloud


@contextlib.contextmanager
def analytic_head(model: Model):
    """Sample with `F_θ ≡ 0`, so the denoiser is the analytic head `c_skip(S) x`.

    That head is the exact posterior mean for Gaussian data with covariance `Ĉ`, so its clouds show
    what the colored prior and the sampler give before any training. On `cov_err` it already sits
    near the Gaussian floor at 400 steps; a trained model matching it there has learned nothing that
    `cov_err` can see, and the gap at low step counts is the sampler's, not the model's.
    """
    model.body.forward = lambda u, F: torch.zeros_like(u)
    try:
        yield
    finally:
        del model.body.forward                               # back to the class's method


@contextlib.contextmanager
def gaussian_denoiser(model: Model | VPModel):
    """Sample with the exact posterior mean for Gaussian data of covariance `Ĉ_train`.

    On the colored family that is the analytic head (:func:`analytic_head`). On VP it is not: VP's
    zero-body denoiser is `α x`, the scalar preconditioner, which on a correlated panel is close to
    white noise. So VP swaps in `GaussianOracleDenoiser(Ĉ_train)`, and both families' paired rows
    answer the same question: what exact second moments give with this sampler and these draws.
    """
    if not isinstance(model, VPModel):
        with analytic_head(model):
            yield
        return
    oracle = GaussianOracleDenoiser(model.C)
    model.denoiser.forward = lambda x, r: oracle(x, r)
    try:
        yield
    finally:
        del model.denoiser.forward                           # back to the class's method


GAUSSIAN_DENOISER = {
    "colored": "the analytic head c_skip(S) x (zero body), the exact posterior mean for N(0, C_train)",
    "vp": "GaussianOracleDenoiser(C_train), the exact posterior mean for N(0, C_train); VP's own "
          "zero-body denoiser alpha x is not a Gaussian reference",
}


def cov_err(cloud: np.ndarray, C_ref: np.ndarray) -> float:
    """`‖Ĉ_cloud − C_ref‖_F / ‖C_ref‖_F`, with `Ĉ_cloud` the unbiased covariance (as the paper)."""
    C = np.cov(np.asarray(cloud, dtype=np.float64), rowvar=False)
    return float(np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref))


def gaussian_floor(S: Operator, refs: dict[str, np.ndarray], n_samples: int, n_clouds: int,
                   seed: int) -> dict[str, dict[str, float]]:
    """`cov_err` of exact `N(0, Ĉ_train)` clouds of the same size, against each reference."""
    rng = np.random.default_rng(stream_seed(seed, "gaussian_floor"))
    factor = S.C_factor                                      # Ĉ_train, whatever S the arm uses
    vals: dict[str, list[float]] = {k: [] for k in refs}
    for _ in range(n_clouds):
        cloud = (factor @ rng.standard_normal((factor.shape[1], n_samples))).T
        for k, C_ref in refs.items():
            vals[k].append(cov_err(cloud, C_ref))
    return {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)), "n_clouds": int(n_clouds)}
            for k, v in vals.items()}


# --------------------------------------------------------------------------- command line and run


def add_arguments(ap: argparse.ArgumentParser, defaults: Recipe, *, n_samples_help: str) -> None:
    """The flags both scripts share, with `defaults` supplying each panel's paper configuration."""
    ap.add_argument("--out", type=Path, required=True, help="output directory (created)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run in --out")
    ap.add_argument("--device", default="auto", help="auto (CUDA if available), cpu, cuda, cuda:1, ...")
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--family", choices=FAMILIES, default=defaults.family,
                    help="colored, or vp for the standard-diffusion baseline on the same data, "
                         f"budget and draws (default {defaults.family})")
    # The three colored-only flags default to None so that passing one with --family vp is caught.
    ap.add_argument("--a", type=float, default=None,
                    help=f"colored: nominal clock exponent, < 0 or exactly 0 (default {defaults.a})")
    ap.add_argument("--R", type=float, default=None,
                    help=f"colored: clock range max(phi)/min(phi) (default {defaults.R:.6g})")
    ap.add_argument("--level-average", choices=LEVEL_AVERAGES, default=None,
                    help="colored: where the FiLM feature and loss weight average, a uniform "
                         f"lambda-grid or the operator's bands (default {defaults.level_average})")
    ap.add_argument("--updates", type=int, default=defaults.updates)
    ap.add_argument("--batch", type=int, default=defaults.batch)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--ema", type=float, default=defaults.ema_decay, help="EMA decay")
    ap.add_argument("--width", type=int, default=defaults.width, help="VNN channel width")
    ap.add_argument("--samplers", nargs="+", choices=SAMPLERS, default=list(SAMPLERS))
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 400],
                    help="reverse transitions per cloud; one cloud per sampler and step count")
    ap.add_argument("--n-samples", type=int, default=None, help=n_samples_help)
    ap.add_argument("--floor-clouds", type=int, default=20,
                    help="exact-Gaussian clouds behind the cov_err floor")
    ap.add_argument("--log-every", type=int, default=200)


def recipe_from_args(args: argparse.Namespace, defaults: Recipe) -> Recipe:
    given = {field: getattr(args, field) for field in COLORED_ONLY}
    if args.family == "vp":
        passed = [COLORED_ONLY[k] for k, v in given.items() if v is not None]
        if passed:
            raise SystemExit(f"{', '.join(passed)} set the colored clock and loss weighting; "
                             "--family vp has neither. Drop them, or use --family colored.")
    colored = {k: (getattr(defaults, k) if v is None else v) for k, v in given.items()}
    return Recipe(**{**asdict(defaults), **colored, "family": args.family,
                     "updates": args.updates, "batch": args.batch, "lr": args.lr,
                     "ema_decay": args.ema, "width": args.width, "seed": args.seed})


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _environment(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "device": str(device),
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "git_commit": commit}


def run(panel: Panel, recipe: Recipe, *, out: Path, device: torch.device, samplers: list[str],
        steps: list[int], n_samples: int, floor_clouds: int, deviations: list[str],
        overwrite: bool = False, log_every: int = 200) -> dict[str, Any]:
    """Train, sample, score and write `run.json`, `model.pt` and one `.npy` per cloud to `out`."""
    out = Path(out)
    if (out / "run.json").exists() and not overwrite:
        raise SystemExit(f"{out / 'run.json'} exists; pass --overwrite or choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    n, d = panel.x_train.shape
    print(f"{panel.name}: {n:,} training frames, d = {d}, tau = {panel.tau}; device {device}")

    if recipe.family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {recipe.family!r}")
    if recipe.gso not in GSOS:
        raise ValueError(f"gso must be one of {GSOS}; got {recipe.gso!r}")
    vp = recipe.family == "vp"
    if recipe.gso == "laplacian":
        if panel.L_sym is None or recipe.gamma is None:
            raise ValueError("gso='laplacian' needs the panel's L_sym and the recipe's gamma")
        S = build_graph_operator(panel.x_train, panel.tau, panel.L_sym, recipe.gamma, device=device)
        gr = S.graph
        print(f"operator: S = sigma2 (L_sym + gamma I)^-1, gamma = {gr['gamma']}, sigma2 = "
              f"{gr['sigma2']:.6g}, tau_g = {gr['tau_g']:.6g} (tau_g - tau = {gr['tau_g_minus_tau']:.2e}), "
              f"kappa = {gr['kappa']:.4g}, {gr['n_null_modes']} null modes of L_sym")
    else:
        S = build_operator(panel.x_train, panel.tau, device=device)
    if vp:
        model = build_vp_model(recipe, S, device=device)
        lm = model.level_map
        # VP has no clock. What stands in for its label is the noise window the shared F ladder maps
        # onto, and the check behind its scalar preconditioning: unit variances.
        lab = {"family": "vp", "gso": S.gso, "graph": S.graph,
               "phi0": lm.phi0, "tr_C": lm.tr_C, "tr_S": lm.tr_S,
               "F_min": recipe.F_min, "F_max": recipe.F_max,
               "r_at_F_min": lm.r_of_F(recipe.F_min), "r_at_F_max": lm.r_of_F(recipe.F_max),
               "level_map": "VPLevelMap: F -> r matching the flat colored clock's population SNR",
               "max_abs_diag_C_minus_1": float(np.abs(np.diag(S.C) - 1.0).max()),
               "unit_variance_note": "VP's scalar preconditioning assumes unit variances; the "
                                     "panel is used as built, as the paper's drivers do",
               "recipe_fields_not_used": sorted(COLORED_ONLY)}
        cov = None
        print(f"VP: no clock; noise-to-signal ratio r from {lab['r_at_F_min']:.4g} to "
              f"{lab['r_at_F_max']:.4g} (the F window [{recipe.F_min}, {recipe.F_max:.4g}] through "
              f"VPLevelMap, phi0 = {lm.phi0}); max |diag C - 1| = {lab['max_abs_diag_C_minus_1']:.2e}")
    else:
        clock = build_clock(recipe, S)
        lab, cov = clock.labels, clock.coverage
        rho = lab["spearman_phi_bandvar"]
        print(f"clock: R = {lab['R']:.6g}, abar = {lab['abar']:.3f}, spearman(phi, band variance) = "
              f"{'n/a (flat clock)' if rho is None else f'{rho:.3f}'}, nominal a = {lab['a_nominal']}; "
              f"F in [{recipe.F_min}, {recipe.F_max:.4g}]")
        print(f"coverage: {cov['below_F_min']} of {cov['live_bands']} live bands reach unit SNR below "
              f"F_min (variance share {cov['variance_share_below_F_min']:.2e}), {cov['above_F_max']} "
              f"above F_max (share {cov['variance_share_above_F_max']:.2e})")
        model = build_model(recipe, S, clock, device=device)

    n_params = sum(p.numel() for p in model.body.parameters() if p.requires_grad)
    print(f"model: {recipe.family}, {type(model.body).__name__} width {recipe.width}, "
          f"{recipe.n_layers} layers, K = {recipe.K_gf}, {n_params:,} trainable parameters"
          + ("" if vp else f"; level average over the {recipe.level_average}"))
    print(f"training: {recipe.updates:,} updates, batch {recipe.batch}, Adam lr {recipe.lr}, "
          f"clip {recipe.grad_clip}, EMA {recipe.ema_decay}")
    history, ema = train(model, S, recipe, device=device, log_every=log_every)

    refs = {k: np.cov(np.asarray(v, dtype=np.float64), rowvar=False)
            for k, v in panel.references.items()}
    floor = gaussian_floor(S, refs, n_samples, floor_clouds, recipe.seed)
    params = [p for p in model.body.parameters() if p.requires_grad]
    raw_state = {k: v.detach().cpu().clone() for k, v in model.body.state_dict().items()}
    clouds: list[dict[str, Any]] = []
    with ema.applied(params):
        ema_state = {k: v.detach().cpu().clone() for k, v in model.body.state_dict().items()}
        for sampler in samplers:
            for N in steps:
                t0 = time.perf_counter()
                cloud = sample(model, recipe, sampler, int(N), n_samples, device=device)
                errs = {k: cov_err(cloud, C_ref) for k, C_ref in refs.items()}
                # Same random streams as the trained cloud, so the two are paired draws.
                with gaussian_denoiser(model):
                    gauss = sample(model, recipe, sampler, int(N), n_samples, device=device)
                name = f"samples_{sampler}_N{N}.npy"
                np.save(out / name, cloud + S.mean)
                clouds.append({
                    "sampler": sampler, "steps": int(N), "n_samples": int(n_samples),
                    "weights": "ema", "file": name, "seconds": time.perf_counter() - t0,
                    "cov_err": errs,
                    "cov_err_over_gaussian_floor": {k: errs[k] / floor[k]["median"] for k in errs},
                    "gaussian_denoiser_cov_err": {k: cov_err(gauss, C_ref)
                                                  for k, C_ref in refs.items()},
                })
    torch.save({"body_ema": ema_state, "body_raw": raw_state, "recipe": asdict(recipe),
                "family": recipe.family, "clock" if not vp else "vp_level_map": lab,
                "panel": panel.name}, out / "model.pt")

    doc = {
        "is_study_result": False,
        "what_this_is": "an example run built from src/ccd; not a paper measurement",
        "panel": panel.name, "family": recipe.family, "data": panel.source,
        "recipe": asdict(recipe),
        "clock": None if vp else lab, "coverage": cov, "vp_level_map": lab if vp else None,
        "trainable_parameters": int(n_params),
        "training": history, "gaussian_floor": floor,
        "gaussian_denoiser": GAUSSIAN_DENOISER[recipe.family], "clouds": clouds,
        "samples_are_in": "the units of the frames read, with the training mean added back",
        "deviations_from_paper": (list(COMMON_DEVIATIONS) + list(FAMILY_DEVIATIONS[recipe.family])
                                  + list(deviations)),
        "environment": _environment(device), "seconds": time.perf_counter() - t_start,
    }
    (out / "run.json").write_text(json.dumps(doc, indent=1) + "\n")

    print(f"\ncov_err (lower is better), EMA weights, {n_samples:,} samples per cloud. The floor is "
          f"the median over {floor_clouds} exact N(0, C_train) clouds of the same size; 'Gaussian' "
          "is the exact Gaussian denoiser on the same random draws.")
    for k in refs:
        print(f"  against '{k}':   floor {floor[k]['median']:.4f} "
              f"(10-90%: {floor[k]['p10']:.4f}-{floor[k]['p90']:.4f})")
        for c in clouds:
            print(f"    sampler {c['sampler']:<10}  {c['steps']:>4} steps   cov_err "
                  f"{c['cov_err'][k]:.4f} ({c['cov_err_over_gaussian_floor'][k]:.2f}x the floor)"
                  f"   Gaussian {c['gaussian_denoiser_cov_err'][k]:.4f}")
    if vp:
        print("  cov_err checks second moments only. VP's untrained network is alpha x, not the "
              "Gaussian denoiser,\n  so a VP model must learn the correlations that row is given.")
    else:
        print("  cov_err checks second moments only. The colored prior gets those right by "
              "construction: the\n  Gaussian denoiser here is the untrained analytic head, so it "
              "already sits near the floor at\n  high step counts and training is not expected to "
              "beat it.")
    print(f"\nwrote {out}/run.json, model.pt and {len(clouds)} sample file(s) "
          f"in {doc['seconds']:.0f} s")
    return doc
