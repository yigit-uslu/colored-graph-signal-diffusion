"""Validation for the split noise draw (`src/ccd/forward/split_noise.py`, plan §2.1b Remedy A).

The split's whole claim is that it is **exact in law** -- so the tests compare against the analytic
covariance, not against the direct route's own output. Where a sample covariance is unavoidable the
tolerance comes from the sampling law rather than from a hand-picked number: two 1190-sample clouds
from the same law differ by ~7% relative Frobenius, so a tight sample-covariance bar would be
unpassable by construction (the calibration rule this project already broke once).

Run: conda run -n torch_env python -m pytest tests/forward/test_split_noise.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ccd.forward import bands
from ccd.forward.rate import ShiftedPowerRate, phi_power
from ccd.forward.split_noise import SplitNoise
from ccd.operators.backends import SpectralBackend
from ccd.operators.linop import CovarianceOperator

D, N, TAU = 16, 300, 0.05


def _setup(seed=0, d=D, n=N, tau=TAU):
    torch.manual_seed(seed)
    Xc = torch.randn(d, n, dtype=torch.float64)
    Xc = Xc - Xc.mean(dim=1, keepdim=True)                       # centred frames
    S = Xc @ Xc.transpose(0, 1) / (n - 1) + tau * torch.eye(d, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(S)
    return Xc, S, evals, evecs


def _gens(a=1, b=2):
    return torch.Generator().manual_seed(a), torch.Generator().manual_seed(b)


# ------------------------------------------------------------------ the exact N(0, S) half


def test_the_prior_draw_has_covariance_S_analytically():
    """`Cov(Xc z/sqrt(n-1) + sqrt(tau) w) = Xc Xc^T/(n-1) + tau I = S`, exactly.

    Checked as an operator identity on the DRAW MAP rather than by sampling: the map is linear in
    (z, w) with independent standard normal blocks, so its covariance is `A A^T` for
    `A = [Xc/sqrt(n-1), sqrt(tau) I]` -- no Monte Carlo needed, and no tolerance to argue about.
    """
    Xc, S, _, evecs = _setup()
    A = torch.cat([Xc / (N - 1) ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)
    torch.testing.assert_close(A @ A.transpose(0, 1), S, rtol=1e-12, atol=1e-12)


def test_ddof_one_is_the_load_bearing_constant():
    """`sqrt(n)` would draw from ((n-1)/n) Chat + tau I -- the same ddof trap as B1."""
    Xc, S, _, _ = _setup()
    wrong = torch.cat([Xc / N ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)
    rel = float((wrong @ wrong.transpose(0, 1) - S).norm() / S.norm())
    assert rel > 1e-3, f"the two conventions must be distinguishable; got {rel:.2e}"


def test_prior_draw_is_deterministic_given_its_generators():
    Xc, S, evals, evecs = _setup()
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi_power(1.0))
    a = noise.draw_prior(5, **dict(zip(("g_factor", "g_shared"), _gens())))
    b = noise.draw_prior(5, **dict(zip(("g_factor", "g_shared"), _gens())))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# ------------------------------------------------------------------ the split identity


@pytest.mark.parametrize("F", [0.02, 1.0, 50.0])
def test_split_forward_draw_matches_sigma_F_in_law(F):
    """`h^{1/2}(S) N(0,S)` must have covariance `S h(S) = Sigma_F` exactly.

    Verified through the draw map again: the split map is `h^{1/2}(S) A`, so its covariance is
    `h^{1/2} A A^T h^{1/2} = h^{1/2} S h^{1/2}`, and that must equal `Sigma_F` band for band.
    """
    Xc, S, evals, evecs = _setup()
    phi = phi_power(1.0)
    backend = SpectralBackend(evals, evecs)
    A = torch.cat([Xc / (N - 1) ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)

    split_cov = backend.apply(bands.h_sqrt_fn(phi, F),
                              backend.apply(bands.h_sqrt_fn(phi, F), A @ A.transpose(0, 1)).T)
    sigma = evecs @ torch.diag(bands.sigma_fn(phi, F)(evals)) @ evecs.transpose(0, 1)
    rel = float((split_cov - sigma).norm() / sigma.norm())
    assert rel < 1e-11, f"split covariance differs from Sigma_F by {rel:.2e}"


@pytest.mark.parametrize("F_a,F_b", [(1.0, 0.9), (50.0, 10.0), (0.05, 0.02)])
def test_split_ancestral_draw_matches_q_in_law(F_a, F_b):
    """Same identity for the pair-indexed `q = lambda * g`."""
    Xc, S, evals, evecs = _setup()
    phi = phi_power(1.0)
    backend = SpectralBackend(evals, evecs)
    A = torch.cat([Xc / (N - 1) ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)
    g_half = bands.g_sqrt_fn(phi, F_a, F_b)

    split_cov = backend.apply(g_half, backend.apply(g_half, A @ A.transpose(0, 1)).T)
    q = evecs @ torch.diag(bands.q_fn(phi, F_a, F_b)(evals)) @ evecs.transpose(0, 1)
    rel = float((split_cov - q).norm() / q.norm())
    assert rel < 1e-11, f"split covariance differs from q by {rel:.2e}"


def test_the_split_agrees_with_the_direct_route_where_the_direct_route_is_accurate():
    """On a well-conditioned toy operator the unsplit `sqrt(lambda*h)` is fine, so the two must
    coincide. This is what makes the split a change of METHOD and not of distribution."""
    Xc, S, evals, evecs = _setup()
    phi, F = phi_power(1.0), 1.0
    backend = SpectralBackend(evals, evecs)
    direct = evecs @ torch.diag(bands.sigma_sqrt_fn(phi, F)(evals)) @ evecs.transpose(0, 1)
    A = torch.cat([Xc / (N - 1) ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)
    split_cov = backend.apply(bands.h_sqrt_fn(phi, F),
                              backend.apply(bands.h_sqrt_fn(phi, F), A @ A.transpose(0, 1)).T)
    torch.testing.assert_close(split_cov, direct @ direct.transpose(0, 1), rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------ the two traps


def test_passing_the_operator_factor_is_refused():
    """B1 re-entering through the noise path: `op.X` is already scaled by sqrt(n/(n-1))."""
    Xc, S, evals, evecs = _setup()
    Xop = Xc * (N / (N - 1)) ** 0.5
    op = CovarianceOperator(Xop, tau=TAU)
    backend = SpectralBackend(evals, evecs)

    with pytest.raises(ValueError, match="OPERATOR's factor"):
        SplitNoise(op.X, TAU, backend, phi=phi_power(1.0), operator_factor=op.X)
    SplitNoise(Xc, TAU, backend, phi=phi_power(1.0), operator_factor=op.X)      # raw frames pass


def test_the_operator_factor_mistake_would_inflate_the_covariance_measurably():
    """The refusal must be worth having: quantify what it prevents."""
    Xc, S, _, _ = _setup()
    Xop = Xc * (N / (N - 1)) ** 0.5
    wrong = torch.cat([Xop / (N - 1) ** 0.5, TAU ** 0.5 * torch.eye(D, dtype=torch.float64)], dim=1)
    inflated = wrong @ wrong.transpose(0, 1)
    data_part_ratio = float(((inflated - TAU * torch.eye(D, dtype=torch.float64)).norm())
                            / ((S - TAU * torch.eye(D, dtype=torch.float64)).norm()))
    assert data_part_ratio == pytest.approx(N / (N - 1), rel=1e-9)


def test_two_generators_are_independent_streams():
    """The (n,batch) and (d,batch) blocks must not share a generator, or the split silently
    reindexes the shared block that every arm -- including VP -- consumes identically."""
    Xc, S, evals, evecs = _setup()
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi_power(1.0))
    gf, gs = _gens()
    base = noise.draw_prior(4, g_factor=gf, g_shared=gs)
    gf2, gs2 = torch.Generator().manual_seed(99), torch.Generator().manual_seed(2)
    only_factor_changed = noise.draw_prior(4, g_factor=gf2, g_shared=gs2)
    gf3, gs3 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(99)
    only_shared_changed = noise.draw_prior(4, g_factor=gf3, g_shared=gs3)
    assert not torch.allclose(base, only_factor_changed)
    assert not torch.allclose(base, only_shared_changed)


def test_consume_like_advances_both_generators_exactly_as_a_draw_does():
    """The teeth-control reads and discards, and must still pair bit-for-bit afterwards."""
    Xc, S, evals, evecs = _setup()
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi_power(1.0))
    gf_a, gs_a = _gens()
    noise.draw_forward(1.0, 6, g_factor=gf_a, g_shared=gs_a)
    after_draw = noise.draw_prior(3, g_factor=gf_a, g_shared=gs_a)

    gf_b, gs_b = _gens()
    noise.consume_like(6, g_factor=gf_b, g_shared=gs_b)
    after_discard = noise.draw_prior(3, g_factor=gf_b, g_shared=gs_b)
    torch.testing.assert_close(after_draw, after_discard, rtol=0, atol=0)


def test_constructor_and_step_guards():
    Xc, S, evals, evecs = _setup()
    backend = SpectralBackend(evals, evecs)
    with pytest.raises(ValueError, match=r"\(d, n\)"):
        SplitNoise(Xc[0], TAU, backend, phi=phi_power(1.0))
    with pytest.raises(ValueError, match="n >= 2"):
        SplitNoise(Xc[:, :1], TAU, backend, phi=phi_power(1.0))
    with pytest.raises(ValueError, match="tau"):
        SplitNoise(Xc, 0.0, backend, phi=phi_power(1.0))
    noise = SplitNoise(Xc, TAU, backend, phi=phi_power(1.0))
    with pytest.raises(ValueError, match="F_a > F_b"):
        noise.draw_ancestral(0.5, 0.9, 4, **dict(zip(("g_factor", "g_shared"), _gens())))


# ------------------------------------------------------------------ sampling sanity


def test_sample_covariance_is_consistent_at_a_tolerance_the_sampling_law_sets():
    """A loose check, deliberately: two clouds from the SAME law differ by O(sqrt(d/batch)).

    The bar is derived, not chosen -- the reference is a same-size draw from the target law, so the
    test asks only that the split is no further from the truth than an honest draw is.
    """
    Xc, S, evals, evecs = _setup()
    phi, F, batch = phi_power(1.0), 1.0, 20_000
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi)
    y = noise.draw_forward(F, batch, **dict(zip(("g_factor", "g_shared"), _gens())))
    emp = y @ y.transpose(0, 1) / batch
    sigma = evecs @ torch.diag(bands.sigma_fn(phi, F)(evals)) @ evecs.transpose(0, 1)

    L = torch.linalg.cholesky(sigma + 1e-12 * torch.eye(D, dtype=torch.float64))
    ref = L @ torch.randn(D, batch, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(7))
    ref_emp = ref @ ref.transpose(0, 1) / batch
    reference_gap = float((ref_emp - sigma).norm() / sigma.norm())
    split_gap = float((emp - sigma).norm() / sigma.norm())
    assert split_gap < 3 * reference_gap, f"split {split_gap:.4f} vs reference {reference_gap:.4f}"


# ------------------------------------------------------------------ review round 6


def test_the_implied_covariance_check_binds_the_frames_to_the_operator():
    """The `operator_factor` check was NOT "by construction" -- it is optional, so a caller who
    omits it gets nothing. `op=` verifies the drawn law against what the operator applies."""
    Xc, S, evals, evecs = _setup()
    backend = SpectralBackend(evals, evecs)
    op = CovarianceOperator(Xc * (N / (N - 1)) ** 0.5, tau=TAU)

    SplitNoise(Xc, TAU, backend, phi=phi_power(1.0), op=op)                 # raw frames: pass
    with pytest.raises(ValueError, match="do not imply the operator"):
        SplitNoise(op.X, TAU, backend, phi=phi_power(1.0), op=op)           # the B1 mistake
    with pytest.raises(ValueError, match="do not imply the operator"):
        SplitNoise(Xc * 1.01, TAU, backend, phi=phi_power(1.0), op=op)      # any rescaling
    with pytest.raises(ValueError, match="do not imply the operator"):
        SplitNoise(Xc + 0.05, TAU, backend, phi=phi_power(1.0), op=op)      # a different centring


def test_consume_like_returns_the_shared_block_so_pairing_can_be_asserted():
    """"The next draw is aligned" is weaker than "the same shared innovation was used".

    An implementation could burn the right number of words and still inject a different `w`, and
    the alignment-only test would pass. This asserts the block itself.
    """
    Xc, S, evals, evecs = _setup()
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi_power(1.0))

    gf_a, gs_a = _gens()
    z_a, w_a = noise.draw_blocks(6, g_factor=gf_a, g_shared=gs_a)
    gf_b, gs_b = _gens()
    w_b = noise.consume_like(6, g_factor=gf_b, g_shared=gs_b)
    torch.testing.assert_close(w_a, w_b, rtol=0, atol=0)          # SAME shared innovation

    # and both generators are left in the same state afterwards
    torch.testing.assert_close(noise.draw_prior(3, g_factor=gf_a, g_shared=gs_a),
                               noise.draw_prior(3, g_factor=gf_b, g_shared=gs_b), rtol=0, atol=0)


def test_draw_prior_is_built_from_the_exposed_blocks():
    """The blocks accessor and the draw must not drift apart."""
    Xc, S, evals, evecs = _setup()
    noise = SplitNoise(Xc, TAU, SpectralBackend(evals, evecs), phi=phi_power(1.0))
    gf, gs = _gens()
    z, w = noise.draw_blocks(4, g_factor=gf, g_shared=gs)
    gf2, gs2 = _gens()
    y = noise.draw_prior(4, g_factor=gf2, g_shared=gs2)
    torch.testing.assert_close(y, (Xc @ z) / (N - 1) ** 0.5 + TAU ** 0.5 * w, rtol=0, atol=0)
