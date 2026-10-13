"""The hierarchical sign-chain Gaussian mixture — target for the ``d=3`` clock experiment (Tier 0).

The target of ``notes/hierarchical-mixture-clock-handoff.md`` §1: a three-level Markov *sign chain*
observed through a diagonal Gaussian channel. It is deliberately small and fully analytic — sampling,
density, posterior sign-pattern probabilities, covariance and fourth moments are all closed form — so
every later tier can be gated against exact ground truth rather than against Monte Carlo.

    s_1 ~ Rad(±1),   s_g = s_{g-1} η_g,   Pr(η_g = +1) = p,   r := E η_g = 2p − 1
    x_g = A_g (√ρ_g s_g + √(1−ρ_g) ε_g),   ε ~ N(0, I)

Conditional on a sign pattern ``s`` the observation is Gaussian, ``x | s ~ N(μ_s, D)`` with
``μ_s = diag(A)diag(√ρ)s`` and ``D = diag(A_g²(1−ρ_g))`` — so the marginal is an **eight-component**
mixture with *shared diagonal* covariance, and the population covariance is analytic:

    C_gg = A_g²,     C_gh = A_g A_h √(ρ_g ρ_h) r^{|g−h|}   (g ≠ h).

**Why this shape.** The variance hierarchy ``A = (4, 1, 0.25)`` and the sign chain are the mechanism
under test: the chain runs *coarse → fine* (the dominant coordinate anchors the finer ones), so a
coarse-to-fine clock can condition on it and a fine-to-coarse clock cannot. The hierarchy is
intentional and must **not** be removed by per-coordinate standardisation in the primary arm.

**Two coordinate views, one construction** (handoff §1.3). The primary view is the target above. The
``"standardized"`` view applies the *diagonal* map ``H = diag(C)^{-1/2}``, giving ``y = Hx`` with
covariance the correlation matrix ``R = HCH``, ``μ_s^{(R)} = Hμ_s``, ``D^{(R)} = HDH``. This is
diagonal standardisation, **not** whitening: ``R ≠ I`` in general, so the dependence structure and the
sign hierarchy survive while the marginal scale differences do not. It exists as a pre-registered
*attribution check*, never as an alternative result to select after seeing outcomes. Because ``H`` is
diagonal, a view is a pure relabelling of the same latent draw: :meth:`sample_paired` returns ``x`` and
``y = Hx`` from **one** stream of ``(labels, ε)``, which is what makes the paired invariants exact
rather than statistical.

⚠️ ``R`` and ``S_R = R + τ_R I`` belong to the standardized *samples*. Applying either as the operator
for untransformed ``x`` would break the identity ``Cov(data) = S − τI`` that the whole colored
machinery rests on.

This module is **eig-free and solve-free** — ``D`` is diagonal, so the density and the posterior need
only reciprocals — so it lives inside the ``src/ccd`` matrix-free guard. The ridge ``τ``, the operator
``S``, the spectrum and the clock tables are *not* here: they need an eigendecomposition and belong to
the script-side configuration (``scripts/hmclock_config.py``).

Conventions: **feature-major** ``(3, B)`` samples, ``(B,)`` labels and ``(8, B)`` posterior
probabilities, matching the rest of the repository (``BlockGMM.sample`` etc.). Component order is
:attr:`sign_patterns`, lexicographic in ``(s_1, s_2, s_3)`` with ``s_1`` slowest.
"""

from __future__ import annotations

import itertools
import math
from functools import cached_property

import torch
from torch import Tensor

__all__ = ["HierMixtureTarget"]

_VIEWS = ("primary", "standardized")


class HierMixtureTarget:
    """Eight-component hierarchical sign-chain Gaussian mixture in ``R^3`` (handoff §1).

    Args:
        A: per-coordinate amplitudes, all ``> 0``. The default ``(4, 1, 0.25)`` is the intentional
            variance hierarchy.
        rho: per-coordinate signal fractions, each strictly in ``(0, 1)``. ``ρ_g = 1`` is rejected,
            not silently accepted: it makes component ``g`` degenerate (``D_gg = 0``) and the
            component density a distribution rather than a function.
        p: sign-chain alignment probability ``Pr(η_g = +1) ∈ [0, 1]``; ``r = 2p − 1``. ``p = 0.5``
            gives independent signs, ``p = 1`` a two-pattern target (six priors exactly zero).
        normalize: apply the single global scalar ``x ← x/√(Tr C/3)`` of §1.2 (default ``True``).
            It is applied to ``A``, and ``C, μ_s, D`` are recomputed from the scaled ``A`` — one
            global scalar, never a per-coordinate rescaling.
        view: ``"primary"`` (original relative scales — the scientific arm) or ``"standardized"``
            (the §1.3 correlation-standardized attribution check).
        dtype: ``torch.float64`` by default. ``float32`` is *supported*, not silently substituted:
            it runs and agrees with float64 to ~1e-6, but the oracle-grade algebraic tolerances of
            the later tiers are only claimed in float64.

    All matrices are derived properties recomputed from the primitive inputs (cached per instance),
    so a serialized configuration need only carry ``A, rho, p, normalize, view, dtype, device``.
    """

    def __init__(
        self,
        A=(4.0, 1.0, 0.25),
        rho=(0.8, 0.8, 0.8),
        p: float = 0.95,
        *,
        normalize: bool = True,
        view: str = "primary",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        A_t = torch.as_tensor(A, dtype=torch.float64).reshape(-1)
        rho_t = torch.as_tensor(rho, dtype=torch.float64).reshape(-1)
        if A_t.numel() != 3 or rho_t.numel() != 3:
            raise ValueError(f"this target is defined for exactly 3 coordinates; got "
                             f"len(A)={A_t.numel()}, len(rho)={rho_t.numel()}")
        if not bool(torch.isfinite(A_t).all()) or not bool(torch.isfinite(rho_t).all()):
            raise ValueError(f"A and rho must be finite; got A={A}, rho={rho}")
        if not bool((A_t > 0).all()):
            raise ValueError(f"every A_g must be > 0; got {A}")
        if not bool(((rho_t > 0) & (rho_t < 1)).all()):
            raise ValueError(
                f"every rho_g must lie strictly in (0, 1); got {rho}. rho_g=1 gives D_gg=0, a "
                "singular component density — rejected rather than treated as an ordinary density.")
        if not math.isfinite(float(p)) or not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"p must be a finite probability in [0, 1]; got {p}")
        if view not in _VIEWS:
            raise ValueError(f"view must be one of {_VIEWS}; got {view!r}")

        self._A_raw = A_t
        self.rho = rho_t
        self.p = float(p)
        self.normalize = bool(normalize)
        self.view = str(view)
        self.dtype = dtype
        self.device = torch.device(device)
        self.G = 3

    # ------------------------------------------------------------------ primitives → base quantities

    def __repr__(self) -> str:                                   # primitives only, mirrors the config
        return (f"HierMixtureTarget(A={self._A_raw.tolist()}, rho={self.rho.tolist()}, p={self.p}, "
                f"normalize={self.normalize}, view={self.view!r}, dtype={self.dtype})")

    @property
    def r(self) -> float:
        """Sign-chain correlation ``r = E η_g = 2p − 1``."""
        return 2.0 * self.p - 1.0

    @cached_property
    def scale(self) -> float:
        """The single global normaliser ``√(Tr C/3)`` of §1.2 (``1.0`` when ``normalize=False``)."""
        if not self.normalize:
            return 1.0
        return float(torch.sqrt(self._cov_from(self._A_raw).diagonal().sum() / self.G))

    @cached_property
    def A(self) -> Tensor:
        """Amplitudes after the global normalisation, ``(3,)`` float64."""
        return self._A_raw / self.scale

    def _cov_from(self, A: Tensor) -> Tensor:
        g = torch.arange(self.G, dtype=torch.float64)
        rr = torch.as_tensor(self.r, dtype=torch.float64)
        # r^|g-h| with 0^0 = 1: guard the r=0 (p=1/2, independent signs) case, where torch gives 0**0=1
        C = torch.outer(A, A) * torch.sqrt(torch.outer(self.rho, self.rho)) \
            * rr.abs() ** (g[:, None] - g[None, :]).abs()
        if self.r < 0:                                            # sign of r^|g-h| for p < 1/2
            C = C * torch.where(((g[:, None] - g[None, :]).abs() % 2) == 1, -1.0, 1.0)
        return C - torch.diag(C.diagonal()) + torch.diag(A ** 2)

    # ------------------------------------------------------------------ analytic structure

    @cached_property
    def C(self) -> Tensor:
        """Population covariance in the **primary** view, ``(3, 3)``."""
        return self._cov_from(self.A)

    @cached_property
    def D_C(self) -> Tensor:
        """``diag(C_11, C_22, C_33)`` — the diagonal of the primary covariance, ``(3, 3)``."""
        return torch.diag(self.C.diagonal())

    @cached_property
    def H(self) -> Tensor:
        """The §1.3 diagonal standardiser ``H = D_C^{-1/2}``, ``(3, 3)``."""
        return torch.diag(self.C.diagonal().rsqrt())

    @cached_property
    def R(self) -> Tensor:
        """Correlation matrix ``R = H C H``, ``(3, 3)``. Note ``R ≠ I``: this is not whitening."""
        return self.H @ self.C @ self.H

    @cached_property
    def sign_patterns(self) -> Tensor:
        """The eight sign patterns, ``(8, 3)``, lexicographic in ``(s_1, s_2, s_3)``, ``s_1`` slowest."""
        return torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=torch.float64)

    @cached_property
    def pi(self) -> Tensor:
        """Exact mixture weights ``π_s``, ``(8,)``, aligned with :attr:`sign_patterns`."""
        s = self.sign_patterns
        agree = (s[:, 1:] == s[:, :-1])                           # (8, 2)
        # explicit float64 scalars: torch.where with *Python* floats infers the default dtype
        # (float32), which would silently cap every downstream weight at ~1e-8 relative.
        pp = torch.as_tensor(self.p, dtype=torch.float64)
        return 0.5 * torch.where(agree, pp, 1.0 - pp).prod(dim=1)

    @cached_property
    def alignment_probs(self) -> Tensor:
        """``(Pr(s_2 = s_1), Pr(s_3 = s_2)) = (p, p)`` — the §5.4 alignment targets, ``(2,)``."""
        return torch.full((2,), self.p, dtype=torch.float64)

    # ---- view-dependent: the quantities that differ between primary and standardized coordinates ----

    @cached_property
    def cov(self) -> Tensor:
        """Population covariance **in this view**: ``C`` (primary) or ``R`` (standardized)."""
        return self.C if self.view == "primary" else self.R

    @cached_property
    def means(self) -> Tensor:
        """Component means ``μ_s`` in this view, ``(8, 3)`` (``Hμ_s`` in standardized coordinates)."""
        return self._means_for(self.view)

    @cached_property
    def comp_var(self) -> Tensor:
        """Diagonal of the shared component covariance in this view, ``(3,)``.

        ``D_gg = A_g²(1−ρ_g)`` primary; ``(HDH)_gg = D_gg/C_gg`` standardized (``H`` is diagonal, so
        the component covariance stays diagonal — which is what keeps the density a product form).
        """
        return self._var_for(self.view)

    @cached_property
    def comp_cov(self) -> Tensor:
        """The shared component covariance as a matrix, ``(3, 3)`` (diagonal by construction)."""
        return torch.diag(self.comp_var)

    def standardized(self) -> "HierMixtureTarget":
        """The paired target in the §1.3 standardized view — same primitives, same sample stream."""
        return HierMixtureTarget(self._A_raw.tolist(), self.rho.tolist(), self.p,
                                 normalize=self.normalize, view="standardized",
                                 dtype=self.dtype, device=self.device)

    def primary(self) -> "HierMixtureTarget":
        """The paired target in the primary view."""
        return HierMixtureTarget(self._A_raw.tolist(), self.rho.tolist(), self.p,
                                 normalize=self.normalize, view="primary",
                                 dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ sampling

    def _cast(self, t: Tensor) -> Tensor:
        return t.to(dtype=self.dtype, device=self.device)

    def _draw(self, batch: int, generator: torch.Generator | None):
        """One latent draw shared by both views: ``(labels (B,), ε (3, B) float64)``.

        Always drawn in **float64**, whatever :attr:`dtype` is, and cast only in :meth:`_assemble`.
        ``torch.randn`` consumes a different number of generator words per element at float32, so a
        dtype-dependent draw would make the same seed yield a *different sample stream* — silently
        breaking any float32/float64 cross-check. Drawing once in float64 keeps seeds portable across
        dtype (costless at ``d=3``).
        """
        if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
            raise ValueError(f"batch must be a positive int; got {batch!r}")
        labels = torch.multinomial(self.pi.to(self.device), batch, replacement=True,
                                   generator=generator)
        eps = torch.randn(self.G, batch, generator=generator, dtype=torch.float64,
                          device=self.device)
        return labels, eps

    def _means_for(self, view: str) -> Tensor:
        mu = self.sign_patterns * (self.A * torch.sqrt(self.rho))[None, :]
        return mu if view == "primary" else mu @ self.H

    def _var_for(self, view: str) -> Tensor:
        d = self.A ** 2 * (1.0 - self.rho)
        return d if view == "primary" else d * self.H.diagonal() ** 2

    def _assemble(self, labels: Tensor, eps: Tensor, view: str) -> Tensor:
        mu = self._means_for(view).to(self.device)
        sd = self._var_for(view).sqrt().to(self.device)
        return self._cast(mu[labels].transpose(0, 1) + sd[:, None] * eps)   # (3, B), cast last

    def sample(self, batch: int, *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Draw ``batch`` samples **in this view**: ``(x (3, B), labels (B,))``.

        ``labels`` index :attr:`sign_patterns` / :attr:`pi`. Pass an explicit ``torch.Generator`` for
        reproducibility — identical seeds reproduce samples, labels and every downstream quantity.
        """
        labels, eps = self._draw(batch, generator)
        return self._assemble(labels, eps, self.view), labels

    def sample_paired(
        self, batch: int, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One draw, both views: ``(x (3, B), y (3, B), labels (B,))`` with ``y = Hx`` **exactly**.

        The paired stream the handoff requires: because ``H`` is diagonal and the same ``(labels, ε)``
        drive both views, the standardisation identity holds to round-off rather than in distribution.
        """
        labels, eps = self._draw(batch, generator)
        return (self._assemble(labels, eps, "primary"),
                self._assemble(labels, eps, "standardized"), labels)

    # ------------------------------------------------------------------ density and posterior

    def _log_component_weights(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(log π_s + log N(x; μ_s, D)) (8, B)`` and the Mahalanobis distances ``q (8, B)``."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.dim() != 2 or x.shape[0] != self.G:
            raise ValueError(f"x must be feature-major (3, B); got {tuple(x.shape)}")
        var = self._cast(self.comp_var)
        mu = self._cast(self.means)
        diff = x[None, :, :] - mu[:, :, None]                     # (8, 3, B)
        q = (diff * diff / var[None, :, None]).sum(dim=1)         # (8, B)
        log_norm = -0.5 * (float(torch.log(var).sum()) + self.G * math.log(2.0 * math.pi))
        log_pi = torch.log(self._cast(self.pi))                   # −inf where π_s = 0 (p = 1)
        return log_pi[:, None] - 0.5 * q + log_norm, q

    def log_prob(self, x: Tensor) -> Tensor:
        """Exact log density by an eight-component ``logsumexp``; ``x (3, B)`` → ``(B,)``."""
        logw, _ = self._log_component_weights(x)
        return torch.logsumexp(logw, dim=0)

    def posterior_sign_pattern_probs(self, x: Tensor) -> Tensor:
        """``Pr(s | x)`` for clean data, ``(8, B)``, aligned with :attr:`sign_patterns`.

        Uses the target's own fixed ``μ_s`` and diagonal ``D``; it deliberately accepts **no**
        caller-supplied means or covariance — the time-dependent version with ``(M_Fμ_s, K_F)`` is the
        Tier-1 oracle's, and letting this method take them would blur the two.

        Numerically it normalises against the *nearest* component (``q − min_s q``) rather than
        against the raw log-weights, so the far tail returns a valid one-hot-like distribution instead
        of ``0/0``: at ``50σ`` the raw weights all underflow, but the differences stay finite.
        """
        logw, q = self._log_component_weights(x)
        rel = torch.log(self._cast(self.pi))[:, None] - 0.5 * (q - q.min(dim=0, keepdim=True).values)
        if not bool(torch.isfinite(rel.max(dim=0).values).all()):
            raise ValueError("posterior_sign_pattern_probs: every component underflowed — |x| is far "
                             "beyond the representable tail; the posterior is not defined there.")
        return torch.softmax(rel, dim=0)

    # ------------------------------------------------------------------ analytic moment gates (§6.3)

    @cached_property
    def fourth_moments(self) -> Tensor:
        """``E[X_i² X_j²]`` in this view, ``(3, 3)`` — the §6.3 covariance-gate ingredient.

        Diagonal ``E[X_i⁴] = Σ_s π_s(μ_si⁴ + 6μ_si²D_ii + 3D_ii²)``; off-diagonal
        ``E[X_i²X_j²] = Σ_s π_s(μ_si² + D_ii)(μ_sj² + D_jj)`` — exact because, *given* the pattern,
        the coordinates are independent Gaussians.
        """
        mu, d = self.means, self.comp_var
        m2 = mu ** 2 + d[None, :]                                 # (8, 3) E[X_i²|s]
        off = torch.einsum("s,si,sj->ij", self.pi, m2, m2)
        diag = (self.pi[:, None] * (mu ** 4 + 6.0 * mu ** 2 * d[None, :] + 3.0 * d[None, :] ** 2)).sum(0)
        return off - torch.diag(off.diagonal()) + torch.diag(diag)

    def second_moment_se(self, n: int) -> Tensor:
        """``SE(Ĉ_ij) = √((E[X_i²X_j²] − C_ij²)/n)`` for the population-zero estimator, ``(3, 3)``.

        The mean is known to be **zero**, so the gate uses ``Ĉ = (1/n)Σ x xᵀ`` — not a mean-subtracted
        sample covariance, whose sampling law differs.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n <= 1:
            raise ValueError(f"n must be an int > 1; got {n!r}")
        return ((self.fourth_moments - self.cov ** 2).clamp_min(0.0) / n).sqrt()

    def second_moment(self, x: Tensor) -> Tensor:
        """Population-zero second moment ``Ĉ = (1/n) Σ_ℓ x_ℓ x_ℓᵀ`` of a ``(3, B)`` cloud, ``(3, 3)``."""
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.dim() != 2 or x.shape[0] != self.G:
            raise ValueError(f"x must be feature-major (3, B); got {tuple(x.shape)}")
        return (x @ x.transpose(0, 1)) / x.shape[1]

    def covariance_zscores(self, x: Tensor) -> Tensor:
        """Entrywise ``z_ij = (Ĉ_ij − C_ij)/SE(Ĉ_ij)`` — the §6.3 sampling-error **gate**, ``(3, 3)``."""
        return (self.second_moment(x) - self.cov) / self.second_moment_se(x.shape[1])

    def covariance_scale_error(self, x: Tensor) -> float:
        """``E_scale = max_ij |[Δ^{-1/2}(Ĉ − C)Δ^{-1/2}]_ij|`` — the scale-balanced **diagnostic**.

        ``Δ`` is the **active view's** covariance diagonal: ``D_C`` in the primary view, and the unit
        diagonal of ``R`` in the standardized view (where the map is therefore the identity and
        ``E_scale`` reduces to ``max|Ĉ − R|``). Each view is thus judged in its own units.

        Reported alongside, never in place of, :meth:`covariance_zscores`: it equalises the coordinate
        scales so a ``4²:0.25²`` variance hierarchy cannot hide a fine-band error, but it carries no
        sampling-error calibration.
        """
        Hd = torch.diag(self.cov.diagonal().rsqrt())
        return float((Hd @ (self.second_moment(x) - self.cov) @ Hd).abs().max())
