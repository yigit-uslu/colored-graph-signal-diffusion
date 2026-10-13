"""Standalone variance-preserving (VP) diffusion — the **out-of-family** baseline.

This module is deliberately *not* part of the colouring machinery. It exists because the project's
headline ("colouring beats the isotropic baseline") is measured against the family's ``a=0`` arm, and
``a=0`` is **whitened-VP**, not standard VP:

    a = 0   (whitened-VP):  prior N(0, S),  per-mode noise  λ(1 − e^{−Fφ})   -- noise ∝ λ
    true VP              :  prior N(0, I),  per-mode noise  σ² = 1 − α²      -- noise λ-independent

``ColoredEDM`` *cannot* express VP: it hard-couples the stationary variance into the noise
(``Σ_F = λ(1−e^{−Fφ})`` with ``λ = Σ_0 + τ``). VP is the ``(a,b)=(0,0)`` point of the ``S^a``-drift /
``S^b``-diffusion family, which lies off the ``2b−a=1`` slice the colouring family lives on
(``scripts/viz_spectrogram.py:105-113``). Hence: standalone.

Everything here assumes the data has been **standardised** (per-gene mean 0, variance 1), which is
standard VP practice and is what makes the preconditioning exact and scalar.

Parameterisation
----------------
The time variable is the **noise-to-signal ratio** ``r = σ/α ∈ (0, ∞)``, monotone increasing with
noise — the direct analogue of the colored ``F``, and positive, so it drops straight into
:class:`~ccd.denoiser.vnn.VNNBody`'s ``F > 0`` contract. Log-SNR is ``ℓ = −2 log r``, and

    α(r) = (1+r²)^{-1/2},   σ(r) = r (1+r²)^{-1/2},   α² = sigmoid(ℓ),  α² + σ² = 1.

Preconditioning
---------------
With per-dim data variance ``s² = 1`` and ``α²+σ² = 1`` the EDM coefficients collapse to

    V = α²s² + σ² = 1,   c_in = 1,   c_skip = α,   c_out = σ
    =>  D(x, r) = α·x + σ·body(x, r)

— the same ``c_skip·x + c_out·body(c_in·x)`` shape as the colored denoiser, so the body plays an
identical role and the VP-vs-colouring contrast is controlled. Two consequences worth stating:

* The loss weight ``1/c_out²`` is the scalar ``1/σ²``, and the objective is then *exactly*
  v-prediction MSE: ``(D−x_0)/σ = body − (σ x_0 − α ε)``, so minimising ``‖D−x_0‖²/σ²`` is
  ``‖body − v‖²`` with ``v = σ x_0 − α ε``. (Proof in :func:`vp_loss`.)
* **This is scalar, not per-band, preconditioning.** Standardising sets each *diagonal* to 1, not each
  eigenvalue, so ``c_skip = α`` is *not* the Wiener/MMSE head — unlike the colored family, whose
  ``c_skip`` is exact per band. This is the conventional and honest VP choice (DDPM/EDM-VP do exactly
  this), but it means a VP body must learn the linear Gaussian structure that a colored body gets for
  free. Registered as an interpretation caveat, not silently absorbed.

Nothing here forms a matrix exponential or an eigendecomposition; the only operator is the graph shift
inside the reused ``VNNBody``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from ccd.denoiser.mlp import MLPBody
from ccd.denoiser.vnn import VNNBody


# --------------------------------------------------------------------------- standardisation

def standardise_stats(Xg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene ``(mu, sd)`` from a gene-major ``(d, n)`` cloud, with **dead genes left unscaled**.

    On the PBMC ``d=256`` panel, **22 genes are exactly constant across the 12,000 operator cells**
    (``diag(Ĉ_n) ≤ 2e-7`` there) — the concrete source of the documented ``rank(Ĉ_n) = 235/256``
    deficiency. Dividing them by a small floor is not merely arbitrary, it is actively harmful: 9 of the
    22 *are* variable in the held-out ``Xg_real``, so a ``1e-6`` floor inflates them by ~3.5e6 and a
    "smallest live sd" floor still by ~271×, wrecking the standardised-space metric.

    So dead genes get ``sd = 1`` (no scaling). They are then identically zero in standardised space
    (``x − mu = 0`` exactly on the set the stats came from), carry **zero loading** in the standardised
    eigenbasis (a zero row of ``Z`` makes ``e_g`` a null eigenvector of ``Z Zᵀ``, so the top-``k``
    eigenvectors are orthogonal to it), and un-standardise back to their constant — which is the
    correct generative output for a gene with no variance to model.

    These statistics must come from ``Xg_op``: it is both the set defining ``S`` and the set the body
    trains on, whereas ``Xg_real`` is the held-out labelled cloud the metric scores against.
    """
    if Xg.ndim != 2:
        raise ValueError(f"Xg must be gene-major (d, n); got {Xg.shape}")
    mu = Xg.mean(1).astype(np.float64)
    sd = Xg.std(1).astype(np.float64)
    return mu, np.where(sd > 0.0, sd, 1.0)


def standardise(Xg: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Gene space -> standardised space."""
    return (Xg - mu[:, None]) / sd[:, None]


def unstandardise(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardised space -> gene space (the inverse of :func:`standardise`, exact to round-off)."""
    return Z * sd[:, None] + mu[:, None]


# --------------------------------------------------------------------------- schedule

def vp_alpha(r: float) -> float:
    """Signal coefficient ``α(r) = (1+r²)^{-1/2}``."""
    return 1.0 / math.sqrt(1.0 + r * r)


def vp_sigma(r: float) -> float:
    """Noise coefficient ``σ(r) = r (1+r²)^{-1/2}``; ``σ/α = r`` and ``α²+σ² = 1`` exactly."""
    return r / math.sqrt(1.0 + r * r)


def vp_ell(r: float) -> float:
    """Log-SNR ``ℓ = log(α²/σ²) = −2 log r`` (decreasing in noise)."""
    return -2.0 * math.log(r)


def vp_r(ell: float) -> float:
    """Inverse of :func:`vp_ell`: ``r = e^{−ℓ/2}``."""
    return math.exp(-0.5 * ell)


def ell_of_F(F: float, *, phi0: float, tr_C: float, tr_S: float) -> float:
    """Population log-SNR of the colored ``a=0`` arm at step ``F`` — the **matching map**.

    The ``a=0`` arm has a constant clock ``φ ≡ phi0``, so its forward draw is
    ``x_F = e^{−Fφ₀/2} x_0 + Σ_F^{1/2} ε`` with ``Σ_F = S(1 − e^{−Fφ₀})``, and its *population* SNR
    (expected signal power over expected noise power) is

        SNR(F) = e^{−Fφ₀} tr(Ĉ) / [ (1 − e^{−Fφ₀}) tr(S) ] .

    A standardised VP process has population SNR ``α²·tr(Ĉ_std)/(σ²·tr(I)) = α²/σ² = e^ℓ`` exactly
    (the traces cancel because the standardised covariance has unit diagonal). Setting the two equal
    gives ``ℓ(F)``, which is how this module matches VP's *noise-level distribution* and *sampler grid*
    to the colored arms instead of picking a free ``ℓ`` range: push the colored arms' own log-uniform
    ``F`` sampling and their own geometric ``F`` grid through this map. That removes "VP was trained /
    sampled over a different SNR window" as a confound.

    Monotone decreasing in ``F``. Well conditioned for ``F ≥ F_min > 0`` (at ``F φ₀ ~ 1.5e-3`` the
    ``1 − e^{−Fφ₀}`` cancellation still carries ~13 significant digits in float64).
    """
    if F <= 0.0:
        raise ValueError(f"ell_of_F needs F > 0; got {F}")
    e = math.exp(-F * phi0)
    denom = -math.expm1(-F * phi0)                      # 1 − e^{−Fφ₀}, exact for small Fφ₀
    if denom <= 0.0:
        raise ValueError(f"degenerate colored SNR at F={F} (φ₀={phi0})")
    return math.log(e * tr_C / (denom * tr_S))


def sample_r_matched(F_min: float, F_max: float, *, phi0: float, tr_C: float, tr_S: float,
                     generator: torch.Generator | None = None) -> float:
    """Draw one training ``r`` by pushing a log-uniform colored ``F`` through :func:`ell_of_F`.

    Gives VP *exactly* the ``a=0`` arm's distribution over population log-SNR (one level per
    minibatch, as in :func:`ccd.denoiser.sample_F_loguniform`).
    """
    u = float(torch.rand((), generator=generator,
                         device=None if generator is None else generator.device))
    F = math.exp(math.log(F_min) + u * (math.log(F_max) - math.log(F_min)))
    return vp_r(ell_of_F(F, phi0=phi0, tr_C=tr_C, tr_S=tr_S))


def sample_r_uniform_ell(ell_min: float, ell_max: float,
                         generator: torch.Generator | None = None) -> float:
    """Draw one training ``r`` uniformly in log-SNR — the plain VP-practice alternative to
    :func:`sample_r_matched`, kept as a registered ablation."""
    u = float(torch.rand((), generator=generator,
                         device=None if generator is None else generator.device))
    return vp_r(ell_min + u * (ell_max - ell_min))


class VPLevelMap:
    r"""The matching map ``F <-> r <-> t`` and its Jacobian — VP's schedule on a *shared* ``F`` ledger.

    :func:`ell_of_F` gives VP the colored arms' population-log-SNR window; this class packages that map
    with its **inverse** and its **derivative**, so a sampler indexed by the shared level ``F`` can be
    written without re-deriving either. Everything here is scalar: no matrix, no operator, no solve.

    The four quantities, all closed form (nothing is inverted numerically):

        r(F)     = e^{-ell(F)/2},                       ell from :func:`ell_of_F`
        F(r)     = log1p(r^2 tr(C)/tr(S)) / phi0        -- exact inverse of the above
        t(F)     = log(1 + r(F)^2) = -log alpha^2       -- VP's own integrated time
        dt/dF    = sigma_F^2 phi0 / (1 - e^{-F phi0}) = sigma_F^2 a_F

    ``F(r)`` is obtained by solving ``ell = log(u tr C/((1-u) tr S))`` for ``u = e^{-F phi0}``, giving
    ``u = tr S/(r^2 tr C + tr S)``; it agrees with ``hmclock_pfode_geometry.VPNativeTimePFODE``'s
    ``F_of_t`` composed with ``t(r)``, which is separately gated at 1.7e-16.

    :meth:`a_of_F` is the **one place** the folded score coefficient is written down. It is deliberately
    *not* duplicated in :class:`VPReversePFODE`: the two would then have to be kept in step by hand, and
    ``a_F`` is the coefficient every drift error routes through.

    Note ``a_F(lam) = phi(lam)/(1 - e^{-F phi(lam)})`` is exactly
    :meth:`ccd.reverse.learned._LearnedReverse._a_fn` evaluated at a **constant** clock ``phi = phi0``.
    That is not a coincidence: VP is the flat-clock member as far as the *score coefficient* is
    concerned, and only the first drift coefficient distinguishes it (``sigma_F^2 a_F`` here versus the
    colored ``(Sigma_F/lam) a_F = phi``).

    Args:
        phi0: the colored-flat rate the window is matched to (``1`` under the section 5.3 pin).
        tr_C: trace of the data covariance.
        tr_S: trace of ``S = C + tau I``. Must be ``>= tr_C`` (the ridge is non-negative).
    """

    def __init__(self, *, phi0: float, tr_C: float, tr_S: float):
        for name, v in (("phi0", phi0), ("tr_C", tr_C), ("tr_S", tr_S)):
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"VPLevelMap needs a finite {name}; got {v}")
            if v <= 0.0:
                raise ValueError(f"VPLevelMap needs {name} > 0; got {v}")
        if float(tr_S) < float(tr_C):
            raise ValueError(f"tr_S must be >= tr_C (the ridge is non-negative); "
                             f"got tr_S={tr_S}, tr_C={tr_C}")
        self.phi0, self.tr_C, self.tr_S = float(phi0), float(tr_C), float(tr_S)

    def __repr__(self) -> str:
        return f"VPLevelMap(phi0={self.phi0:g}, tr_C={self.tr_C:g}, tr_S={self.tr_S:g})"

    @staticmethod
    def _check_F(F: float, *, positive: bool = False) -> float:
        F = float(F)
        if not math.isfinite(F):
            raise ValueError(f"F must be finite; got {F}")
        if positive and F <= 0.0:
            raise ValueError(f"F must be > 0 here (a_F diverges at F=0); got {F}")
        if F < 0.0:
            raise ValueError(f"F must be >= 0; got {F}")
        return F

    def r_of_F(self, F: float) -> float:
        """Noise-to-signal level ``r = sigma/alpha`` at shared level ``F``; ``r(0) = 0`` by definition."""
        F = self._check_F(F)
        if F == 0.0:
            return 0.0
        return vp_r(ell_of_F(F, phi0=self.phi0, tr_C=self.tr_C, tr_S=self.tr_S))

    def F_of_r(self, r: float) -> float:
        """Exact inverse of :meth:`r_of_F` — closed form, no numerical solve."""
        r = float(r)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError(f"r must be finite and >= 0; got {r}")
        return math.log1p(r * r * self.tr_C / self.tr_S) / self.phi0

    def t_of_F(self, F: float) -> float:
        """VP's own integrated time ``t = log(1 + r^2) = -log alpha^2`` at shared level ``F``."""
        r = self.r_of_F(F)
        return math.log1p(r * r)

    def a_of_F(self, F: float) -> float:
        r"""Folded score coefficient ``a_F = phi0/(1 - e^{-F phi0})``. Requires ``F > 0``.

        ``-expm1`` keeps the small-``F phi0`` cancellation exact. Diverges as ``1/F`` at the origin,
        which is why the learned drift stops at a positive ``F_min`` — and why the drift near a small
        ``F_min`` carries ``~a_F`` times the round-off of the mid-window. That is conditioning, not a
        defect, and it is shared verbatim with the colored :class:`~ccd.reverse.LearnedReversePFODE`.
        """
        F = self._check_F(F, positive=True)
        return self.phi0 / (-math.expm1(-F * self.phi0))

    def dt_dF(self, F: float) -> float:
        r"""``dt_VP/dF = sigma_F^2 a_F`` — the Jacobian of the matching map.

        At ``F = 0`` the expression is ``0/0``; the limit ``phi0 tr(S)/tr(C)`` is returned explicitly.
        As ``F -> inf`` it tends to ``phi0``. Measured on the ``d=3`` target it stays inside
        ``[1.0000, 1.0029]`` across the whole window, so it is *not* the term that conditions the drift.
        """
        F = self._check_F(F)
        if F == 0.0:
            return self.phi0 * self.tr_S / self.tr_C
        s = vp_sigma(self.r_of_F(F))
        return s * s * self.a_of_F(F)

    def r_grid(self, F_grid) -> list[float]:
        """Push a shared-``F`` grid through :meth:`r_of_F`, for feeding :class:`VPAncestralSampler`.

        The two VP samplers take different grids on purpose — the ancestral one is natural in ``r``, the
        PF-ODE must be in ``F`` so that its discretisation is matched to the analytic comparator. This
        exists so a driver converts **once, explicitly**, and both are driven off one ``F`` ledger.
        """
        return [self.r_of_F(F) for F in F_grid]


# --------------------------------------------------------------------------- body / denoiser

class GraphOnlyForward:
    """Minimal stand-in for :class:`~ccd.forward.marginals.ColoredForward`.

    :class:`~ccd.denoiser.vnn.VNNBody` touches its ``fwd`` only for ``op`` (the graph shift ``S̃``, and
    ``d``/dtype/device) and the bracket ``[lo, hi]`` — plus ``phi``, which lives exclusively in the two
    hooks :class:`VPBody` overrides. There *is* no colored forward process for VP, so fabricating a
    ``ColoredForward`` with a meaningless ``φ`` would be dishonest; this shim makes the absence explicit
    (``phi = None`` will raise loudly if anything unexpected reaches for it).
    """

    def __init__(self, op, lo: float, hi: float):
        self.op, self.lo, self.hi = op, float(lo), float(hi)
        self.phi, self.K = None, None


class VPBody(VNNBody):
    """:class:`~ccd.denoiser.vnn.VNNBody` with its two colour-specific hooks replaced by VP analogues.

    Everything structural is inherited unchanged — the graph-filter stack in ``S̃``, scale-only FiLM,
    no biases on the signal path, ``F_θ(0)=0``. Only the two places where the colouring ``φ`` enters
    are swapped, and in both cases VP's isotropy makes the replacement *simpler*, not different in
    kind:

    * **Conditioning.** The colored feature is ``ν(F) = log⟨SNR_λ(F)⟩``, a φ-weighted average over the
      spectrum. VP's per-mode SNR is ``α² s_λ²/σ²``, whose λ-dependence is a *fixed* additive constant
      in logs, so the whole informative content of the noise level is the scalar ``ℓ = −2 log r``.
      Passing ``ℓ`` directly is the exact analogue (monotone, same units, same scalar shape).
    * **Mehler envelope.** The colored gate is the operator ``η(S) = m_F(S)^{k_0}``, needed because the
      per-band mean decays at band-dependent rates. VP's mean decay ``α`` is λ-independent, so the same
      gate is the *scalar* ``α^{k_0}`` — no Chebyshev apply at all.
    """

    def _cond_feature(self, r: float) -> Tensor:
        p = self.encoder.weight
        return torch.full((1,), vp_ell(r), dtype=p.dtype, device=p.device)

    def _apply_envelope(self, y: Tensor, r: float) -> Tensor:
        return (vp_alpha(r) ** self.k0) * y


class VPMLPBody(MLPBody):
    """:class:`~ccd.denoiser.mlp.MLPBody` with the same two hooks replaced — VP's fully-connected arm.

    Exactly :class:`VPBody`, one class up: the two overrides below are *character-identical* to
    ``VPBody``'s, because the hook contract is shared between the graph body and the MLP body and neither
    hook touches the signal path. The justification for each is in :class:`VPBody`'s docstring and is
    unchanged — ``ℓ = −2 log r`` is the exact scalar analogue of the colored ``ν(F) = log⟨SNR_λ(F)⟩``, and
    VP's λ-independent mean decay makes the Mehler gate the scalar ``α^{k_0}`` rather than an operator.

    **Why this is what makes the Tier-3 arms comparable.** Nothing here owns a parameter, so the four arms
    (``flat``, ``c2f``, ``f2c`` via :class:`~ccd.denoiser.mlp.MLPBody`; ``vp`` via this class) have
    *exactly* identical parameter shapes in the same construction order. Seeded identically they start
    from bit-identical weights, so "no arm-specific architecture or budget tuning" is structural rather
    than a claim to be trusted. ``tests/targets/test_hmclock_mlp_body.py`` gates both properties, and a
    mutation control checks the gates fail if an arm's initialisation is perturbed.
    """

    def _cond_feature(self, r: float) -> Tensor:
        p = self.encoder.weight
        return torch.full((1,), vp_ell(r), dtype=p.dtype, device=p.device)

    def _apply_envelope(self, y: Tensor, r: float) -> Tensor:
        return (vp_alpha(r) ** self.k0) * y


class VPDenoiser(nn.Module):
    """``D(x, r) = α(r)·x + σ(r)·body(x, r)`` — scalar-preconditioned VP denoiser.

    ``c_in = 1`` (standardised data), so unlike :class:`~ccd.denoiser.denoiser.ColoredDenoiser` there is
    no whitening solve and no ``c_in_kw``.
    """

    def __init__(self, body: VPBody | VPMLPBody):
        super().__init__()
        self.body = body

    def forward(self, x: Tensor, r: float) -> Tensor:
        return vp_alpha(r) * x + vp_sigma(r) * self.body(x, r)

    def head(self, x: Tensor, r: float) -> Tensor:
        """The zero-body denoiser ``α·x``. NOTE: unlike the colored head this is **not** the Gaussian
        MMSE denoiser (see the module docstring) — it is the scalar VP preconditioner."""
        return vp_alpha(r) * x


def vp_loss(den: VPDenoiser, x0: Tensor, r: float, *, eps: Tensor | None = None,
            generator: torch.Generator | None = None) -> Tensor:
    r"""VP denoising loss ``w(r)·E‖D(x_r, r) − x_0‖²`` with the scalar weight ``w = 1/c_out² = 1/σ²``.

    The exact analogue of :func:`ccd.denoiser.edm_loss`'s scalar weight ``1/⟨c_out²⟩``: here ``c_out``
    is already the scalar ``σ``, so the spectral average is a no-op.

    Equivalently plain **v-prediction** MSE. With ``x_r = α x_0 + σ ε`` and ``D = α x_r + σ·body``,

        (D − x_0)/σ = (α² − 1)x_0/σ + α ε + body = −σ x_0 + α ε + body = body − (σ x_0 − α ε),

    using ``α²−1 = −σ²``. So the loss is ``E‖body − v‖²`` with ``v = σ x_0 − α ε`` — a standard,
    well-conditioned target at every noise level.
    """
    if eps is None:
        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
    a, s = vp_alpha(r), vp_sigma(r)
    x_r = a * x0 + s * eps
    return (den(x_r, r) - x0).pow(2).mean() / (s * s)


# --------------------------------------------------------------------------- reverse sampler

def vp_ancestral_coeffs(r_a: float, r_b: float) -> tuple[float, float, float]:
    r"""Coefficients of the ancestral (data-prediction) step ``r_a → r_b`` with ``r_b ≤ r_a``:

        x_b = c_x · x_a + c_d · D(x_a, r_a) + q^{1/2} ξ

    This is the reverse posterior ``q(x_b | x_a, x_0)`` with the point estimate ``x_0 ← D``. Writing
    ``α_{a|b} = α_a/α_b`` for the transition mean and ``σ_{a|b}² = σ_a² − α_{a|b}²σ_b²`` for the
    transition variance,

        c_x = α_{a|b} σ_b²/σ_a² ,   c_d = σ_{a|b}² α_b/σ_a² ,   q = σ_{a|b}² σ_b²/σ_a² .

    Structurally identical to the colored
    :meth:`~ccd.reverse.LearnedReverseSDE._ancestral_coeffs` (``c_x = T Σ_b/Σ_a``, ``c_d = R m_b/Σ_a``,
    ``q = R Σ_b/Σ_a``) with the operator bands ``(m_F, Σ_F)`` replaced by the scalars ``(α, σ²)`` — so
    the sampler is a controlled analogue, not a different algorithm.

    **Exact given the true ``x_0``**: substituting ``x_a = α_a x_0 + σ_a ε`` gives mean coefficient
    ``c_x α_a + c_d = α_b`` and total variance ``c_x²σ_a² + q = σ_b²``, i.e. the correct ``r_b``
    marginal. Limits: ``r_b → r_a`` ⇒ ``(1, 0, 0)`` (no move); ``r_b → 0`` ⇒ ``(0, 1, 0)`` (``x_b → D``,
    the built-in final denoise). With the point estimate it under-injects by ``c_d²·Var(x_0|x_a)``, the
    standard data-prediction under-dispersion — exactly as in the colored sampler, and exactly what the
    R3 read-out re-adds.
    """
    if r_b > r_a:
        raise ValueError(f"reverse step needs r_b <= r_a; got r_a={r_a}, r_b={r_b}")
    a_a, s_a = vp_alpha(r_a), vp_sigma(r_a)
    a_b, s_b = vp_alpha(r_b), vp_sigma(r_b)
    a_ab = a_a / a_b                                     # transition mean α_{a|b}
    s_ab_sq = s_a * s_a - a_ab * a_ab * s_b * s_b        # transition variance σ_{a|b}²
    inv = 1.0 / (s_a * s_a)
    return a_ab * s_b * s_b * inv, s_ab_sq * a_b * inv, math.sqrt(max(s_ab_sq * s_b * s_b * inv, 0.0))


class VPAncestralSampler:
    """Ancestral reverse sampler for :class:`VPDenoiser` over a decreasing ``r`` grid.

    Mirrors :meth:`ccd.reverse.LearnedReverseSDE.integrate` with ``scheme="ancestral"``: same step
    family, same ``denoise_final`` convention (the returned cloud is ``D(x, r_min)``, the clean
    estimate, not the last noisy iterate).
    """

    def __init__(self, denoiser: VPDenoiser):
        self.denoiser = denoiser

    def step(self, x: Tensor, r_a: float, r_b: float, *,
             generator: torch.Generator | None = None) -> Tensor:
        if r_b == r_a:
            return x
        with torch.no_grad():
            d = self.denoiser(x, r_a)
            c_x, c_d, q_std = vp_ancestral_coeffs(r_a, r_b)
            xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
            return c_x * x + c_d * d + q_std * xi

    def integrate(self, x_T: Tensor, r_grid, *, generator: torch.Generator | None = None,
                  denoise_final: bool = True) -> Tensor:
        rs = [float(v) for v in r_grid]
        x = x_T
        for r_a, r_b in zip(rs[:-1], rs[1:]):
            x = self.step(x, r_a, r_b, generator=generator)
        if denoise_final:
            with torch.no_grad():
                x = self.denoiser(x, rs[-1])
        return x


class VPReversePFODE:
    r"""Deterministic probability-flow ODE for :class:`VPDenoiser`, on the **shared** ``F`` ledger.

    The VP analogue of :class:`ccd.reverse.LearnedReversePFODE`, matched deliberately: same scheme
    vocabulary (``"euler"`` / ``"heun"``, **heun default**), same ``denoise_final`` convention (the
    returned cloud is the clean estimate ``D(x, r(F_min))``), same decreasing-grid contract. It exists
    because Tier 3 needs a *learned* VP row beside the colored ones and VP previously had only the
    stochastic :class:`VPAncestralSampler`.

    **The drift.** Tweedie for ``x_F = alpha x_0 + sigma eps`` gives ``s = (alpha D - x)/sigma^2``, and
    the PF-ODE ``-1/2 (dt/dF) x - 1/2 (dt/dF) s`` then regroups, using ``sigma^2 - 1 = -alpha^2``, to

        dx/dF = -1/2 a_F alpha ( D(x, r(F)) - alpha x ),        a_F = phi0/(1 - e^{-F phi0}).

    Written this way rather than as the two-term ``-1/2 (dt/dF) x - 1/2 a_F (alpha D - x)`` that mirrors
    the colored file line for line. **The two are the same field** — they differ only in floating-point
    conditioning, and the regrouped one is strictly better behaved: with a zeroed body ``D = alpha x``,
    so ``D - alpha x`` is **structurally** zero and the drift is bit-exactly ``0.0`` at every level,
    where the two-term form leaves ~5.7e-13 near ``F_min = 1e-3``. The equivalence is *gated*
    (``tests/targets/test_vp_pfode.py``), not asserted here.

    ⚠️ That zero-body collapse is a **structural cancellation / regression** gate, not a correctness
    gate: any wrong multiplier or sign in front of ``(D - alpha x)`` also returns zero. What pins
    correctness is agreement with the exact-denoiser analytic comparator, drift-wise and after
    integration.

    Why it collapses to zero at all — and why the colored one does not: the zero-body VP denoiser
    ``alpha x`` implies the score ``(alpha^2 - 1)x/sigma^2 = -x``, which is the score of ``N(0, I)``,
    VP's stationary law; a stationary law has identically zero probability flow. The colored head
    instead collapses to the *nonzero* analytic drift ``c_F = phi tau e^{-F phi}/(2 v_F)``, because its
    stationary law is ``N(0, S)`` and the ridge leaves residual drift.

    Everything here is scalar arithmetic around one denoiser call — no operator apply, no matrix, no
    eigendecomposition — so the matrix-free invariant holds by construction rather than by pragma. There
    is deliberately no ``backend=``: that routes *function-of-S* applies through an exact eigensystem
    and VP has none to route. There is deliberately no ``rk4``: the colored learned PF-ODE offers only
    euler/heun, and the Tier-3 reference rows are registered as Heun at a common ``N_ref`` on both the
    learned and analytic sides, with RK4-4096 kept as a separate continuum-validation row.

    Args:
        denoiser: callable ``D(x, r)`` — a :class:`VPDenoiser`, or any duck-typed stand-in. The gate
            drives it with an exact-mixture adapter, which is why this is not typed to ``VPDenoiser``.
        level_map: the :class:`VPLevelMap` carrying ``r(F)`` and ``a_F``. Not rebuilt internally, so a
            driver can share one map between this and :class:`VPAncestralSampler`.
    """

    def __init__(self, denoiser, level_map: VPLevelMap):
        if not isinstance(level_map, VPLevelMap):
            raise TypeError(f"level_map must be a VPLevelMap; got {type(level_map).__name__}")
        if not callable(denoiser):
            raise TypeError("denoiser must be callable as D(x, r)")
        self.denoiser = denoiser
        self.map = level_map

    def __repr__(self) -> str:
        return f"VPReversePFODE({self.map!r})"

    def drift(self, x: Tensor, F: float) -> Tensor:
        """``dx/dF = -1/2 a_F alpha (D(x, r(F)) - alpha x)``. Requires ``F > 0``."""
        a = self.map.a_of_F(F)                       # validates F > 0; the single definition of a_F
        r = self.map.r_of_F(F)
        al = vp_alpha(r)
        return -0.5 * a * al * (self.denoiser(x, r) - al * x)

    def step(self, x: Tensor, F_a: float, F_b: float, *, scheme: str = "heun") -> Tensor:
        """One deterministic reverse step ``F_a -> F_b`` (``F_b <= F_a``); no noise, no generator.

        ``"heun"`` (default) is the 2nd-order predictor–corrector: a Euler predictor
        ``x~ = x - dF b(x, F_a)`` then the average-slope corrector ``x - dF (b(x,F_a) + b(x~,F_b))/2``.
        ``"euler"`` is the explicit first-order step. Both need ``F_b > 0``.
        """
        if scheme not in ("euler", "heun"):
            raise ValueError(f"scheme must be 'euler' or 'heun'; got {scheme!r}")
        F_a, F_b = float(F_a), float(F_b)
        # BOTH endpoints, explicitly, and BEFORE the no-move shortcut. Relying on the drift call to
        # validate is not enough: `euler` never evaluates the drift at F_b, so a review found
        # `step(x, 1.0, 0.0, scheme="euler")` silently returning a tensor (and `F_b = -1.0` likewise)
        # while `heun` raised, because only its corrector touches the endpoint.
        for name, v in (("F_a", F_a), ("F_b", F_b)):
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"a PF-ODE step needs both endpoints finite and > 0 "
                                 f"(a_F diverges at F=0); got {name}={v}")
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        with torch.no_grad():
            dF = F_a - F_b                                            # > 0
            b_a = self.drift(x, F_a)
            x_euler = x - dF * b_a
            if scheme == "euler":
                return x_euler
            return x - 0.5 * dF * (b_a + self.drift(x_euler, F_b))

    def integrate(self, x_T: Tensor, F_grid, *, scheme: str = "heun",
                  denoise_final: bool = True) -> Tensor:
        """Integrate down a decreasing ``F_grid``; ``x_T`` is the draw at ``F_grid[0]``.

        With ``denoise_final=True`` (default) the output is the clean estimate ``D(x, r(F_grid[-1]))``
        — the production read-out, matching ``AnalyticScorePFODE``'s ``endpoint="denoise"``. With
        ``False`` it is the raw last iterate, matching ``endpoint="none"``; that is the comparison that
        tests the *trajectory*, since a final denoise can mask an upstream discrepancy.

        Deterministic: no injected noise and no ``generator``, so repeats are bit-identical on CPU.
        """
        Fs = [float(f) for f in F_grid]
        if len(Fs) < 2:
            raise ValueError(f"F_grid needs at least two levels; got {len(Fs)}")
        if any(b > a for a, b in zip(Fs[:-1], Fs[1:])):
            raise ValueError("F_grid must be non-increasing")
        if Fs[-1] <= 0.0:
            raise ValueError(f"F_grid must end at a positive F_min (a_F diverges at 0); got {Fs[-1]}")
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme=scheme)
        if denoise_final:
            with torch.no_grad():
                x = self.denoiser(x, self.map.r_of_F(Fs[-1]))
        return x


# --------------------------------------------------------------------------- oracle (gate only)

class GaussianOracleDenoiser:
    """Exact MMSE denoiser for Gaussian data of covariance ``C``: ``D*(x) = α C (α²C + σ²I)^{-1} x``.

    **Gate/test use only** — it eigendecomposes a dense ``C``. Its purpose is to make the sampler
    falsifiable end to end: run :class:`VPAncestralSampler` with this in place of the learned
    denoiser, start from ``N(0, I)``, and the generated cloud must have covariance ``C``. That tests
    the schedule, the ancestral coefficients, the terminal law and the read-out convention together,
    with no learning in the loop.
    """

    def __init__(self, C: Tensor):
        w, U = torch.linalg.eigh(C.double())  # matrix-free-ok: gate-only oracle, NOT a size bound — this is d×d by design, and it never runs on the sampler's matrix-free path
        self.w, self.U, self.dtype = w.clamp_min(0.0), U, C.dtype

    def __call__(self, x: Tensor, r: float) -> Tensor:
        a, s = vp_alpha(r), vp_sigma(r)
        g = (a * self.w / (a * a * self.w + s * s)).to(dtype=x.dtype, device=x.device)
        U = self.U.to(dtype=x.dtype, device=x.device)
        return U @ (g[:, None] * (U.transpose(0, 1) @ x))


class GaussianOraclePosteriorSampler(GaussianOracleDenoiser):
    """Exact posterior **sample** ``x_0 ~ p(x_0 | x_r)`` instead of the posterior **mean**.

    **Gate/test use only.** The ancestral step injects the noise of ``q(x_b | x_a, x_0)`` for a *known*
    ``x_0``; feeding it the conditional mean therefore under-disperses by ``c_d²·Var(x_0|x_a)`` per step
    — the standard data-prediction deficit, shared verbatim with the colored sampler. Substituting this
    class makes the chain exactly correct, which is what turns "is the schedule right?" into a
    decisive test: any residual bias here is a real implementation error, whereas the deficit seen with
    the mean is discretisation and must vanish as ``O(1/N_steps)``.

    Posterior covariance in the eigenbasis of ``C``: ``w σ²/(α²w + σ²)``.

    ⭐ **``generator`` — pass the ledger's.** The default builds a private CPU generator seeded 99,
    which has two consequences a benchmark cannot accept and a standalone gate script never noticed:
    its randomness is **outside the seed ledger** entirely (so the cloud is not reproducible from a
    protocol, and pairing against any other arm is accidental), and a CPU generator **cannot draw on
    CUDA** — ``torch.randn(..., generator=<cpu gen>, device='cuda')`` raises, so the class simply
    could not run on the device the benchmark uses. Supplying a generator fixes both; the default is
    kept only so the existing ``scripts/vp_preflight.py`` call site keeps working unchanged.
    """

    def __init__(self, C: Tensor, seed: int = 99, *, generator: torch.Generator | None = None):
        super().__init__(C)
        self._g = generator if generator is not None else torch.Generator().manual_seed(int(seed))

    def __call__(self, x: Tensor, r: float) -> Tensor:
        a, s = vp_alpha(r), vp_sigma(r)
        v = (self.w * s * s / (a * a * self.w + s * s)).to(dtype=x.dtype, device=x.device)
        U = self.U.to(dtype=x.dtype, device=x.device)
        if self._g.device.type != x.device.type:
            raise ValueError(
                f"the posterior oracle's generator is on {self._g.device} but x is on {x.device}. "
                "torch.randn cannot draw across devices, so this would raise deeper in the stack "
                "with no mention of the ledger. Pass `generator=` from the seed ledger, on the "
                "device the sampler runs on."
            )
        z = torch.randn(x.shape, generator=self._g, dtype=x.dtype, device=x.device)
        return super().__call__(x, r) + U @ (v.sqrt()[:, None] * z)
