"""Generation-quality metrics for the end-to-end reverse-sampler gates — TEST ONLY.

Shared by the a=0 gate (``test_v3b_generation``), the a=−1 gate (``test_am1_generation``) and the deterministic
PF-ODE check (``test_pf_ode_learned``) so the block-factor / shape / distribution readouts are defined once.
Importable from any test subdir via the ``tests/conftest.py`` ``sys.path`` insert (same convention as
``_oracles``). ``block_avg`` is duck-typed on the toy (needs ``.B`` and ``.gsize``) to keep this helper free of
the ``ccd.denoiser`` / PyG import.
"""

from __future__ import annotations

import torch


def block_avg(toy, x: torch.Tensor) -> torch.Tensor:
    """Per-block factor estimate ``f̂_g = mean over block-g nodes of x`` — shape ``(G, N)``.

    The idiosyncratic noise averages down by ``√gsize``, so ``f̂_g ≈ f_g`` and its marginal inherits the
    factor's (bi)modality: the linear projection that exposes the non-Gaussian structure.
    """
    return (toy.B.T @ x) / toy.gsize


def excess_kurtosis(v: torch.Tensor) -> torch.Tensor:
    """Central excess kurtosis over the last (sample) axis — ``< 0`` for a sub-Gaussian/bimodal marginal.

    Mean-centred (not the about-zero fourth-moment ratio): so a *shifted unimodal* marginal reads ``≈ 0``, not
    spuriously negative — the kurtosis gate is then a correct bimodality test on its own, not only in concert
    with the std-factor / covariance gates. (For this mean-zero-symmetric data the centring shifts the value
    by ``O(1/N)``.)
    """
    vc = v - v.mean(-1, keepdim=True)
    m2, m4 = vc.pow(2).mean(-1), vc.pow(4).mean(-1)
    return m4 / m2.pow(2) - 3.0


def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Székely energy distance between two sample sets (columns are samples): ``2⟨‖x−y‖⟩ − ⟨‖x−x'‖⟩ −
    ⟨‖y−y'‖⟩`` — ``≥ 0``, zero iff the distributions match. Computed in float64."""
    X, Y = X.double(), Y.double()

    def pairwise_mean(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        d2 = A.pow(2).sum(0)[:, None] + B.pow(2).sum(0)[None, :] - 2.0 * (A.T @ B)
        return d2.clamp_min(0).sqrt().mean()

    return float(2 * pairwise_mean(X, Y) - pairwise_mean(X, X) - pairwise_mean(Y, Y))


def standardize(f: torch.Tensor) -> torch.Tensor:
    """Per-row (per-block) zero-mean, unit-variance — strips the covariance scale so an energy distance on the
    result measures pure *shape* (bi)modality, not the second moment."""
    return (f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(1e-12)
