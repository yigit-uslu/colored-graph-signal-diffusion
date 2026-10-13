"""Validation for the application backends (`src/ccd/operators/backends.py`, plan §2.1b Remedy C).

The dense route exists because a measurement said the matrix-free one cannot work here, so the
load-bearing tests are (a) the two agree wherever Chebyshev *does* resolve — otherwise "dense" would
be a change of subject rather than a change of method — and (b) the dense route actually fixes the
functions it was adopted for.

Run: conda run -n torch_env python -m pytest tests/operators/test_backends.py -v
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from ccd.denoiser.edm import ColoredEDM
from ccd.forward import bands
from ccd.forward.marginals import ColoredForward
from ccd.forward.rate import ShiftedPowerRate, phi_power
from ccd.operators.backends import ChebyshevBackend, SpectralBackend
from ccd.operators.linop import CovarianceOperator

FMRI = pathlib.Path(__file__).resolve().parents[2] / "data" / "fmri"
have_data = (FMRI / "S_ccd.npy").is_file()


def _toy(d=24, n=400, tau=0.05, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(d, n, dtype=torch.float64)
    op = CovarianceOperator(X, tau=tau)
    S = X @ X.transpose(0, 1) / n + tau * torch.eye(d, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(S)
    return op, evals, evecs, S


# ------------------------------------------------------------------ agreement


def test_dense_and_chebyshev_agree_where_chebyshev_resolves():
    """A well-conditioned operator: both routes must give the same answer to float64 precision."""
    op, evals, evecs, _ = _toy()
    lo, hi = float(evals.min()) * 0.99, float(evals.max()) * 1.01
    spec = SpectralBackend(evals, evecs, validate_against=op)
    cheb = ChebyshevBackend(op, lo, hi, K=200)
    v = torch.randn(24, 7, dtype=torch.float64)

    for fn in (bands.m_fn(phi_power(1.0), 0.5),
               bands.h_sqrt_fn(phi_power(1.0), 0.5),
               bands.v_fn(phi_power(1.0), 0.05, 0.5)):
        torch.testing.assert_close(spec.apply(fn, v), cheb.apply(fn, v), rtol=1e-9, atol=1e-9)


def test_dense_apply_is_exact_against_the_explicit_matrix_function():
    op, evals, evecs, S = _toy()
    spec = SpectralBackend(evals, evecs, validate_against=op)
    v = torch.randn(24, 3, dtype=torch.float64)
    fn = bands.m_fn(phi_power(1.0), 0.7)
    truth = evecs @ torch.diag(fn(evals)) @ evecs.transpose(0, 1) @ v
    torch.testing.assert_close(spec.apply(fn, v), truth, rtol=1e-13, atol=1e-13)


def test_backend_is_linear_and_shape_preserving():
    op, evals, evecs, _ = _toy()
    spec = SpectralBackend(evals, evecs)
    fn = bands.m_fn(phi_power(1.0), 0.3)
    a, b = torch.randn(24, 4, dtype=torch.float64), torch.randn(24, 4, dtype=torch.float64)
    torch.testing.assert_close(spec.apply(fn, 2.0 * a + b),
                               2.0 * spec.apply(fn, a) + spec.apply(fn, b), rtol=1e-12, atol=1e-12)
    assert spec.apply(fn, a).shape == a.shape


# ------------------------------------------------------------------ the validation clause


def test_a_wrong_eigensystem_is_rejected_by_matvec():
    """Exact for the wrong matrix is the failure mode this clause exists for."""
    op, evals, evecs, _ = _toy()
    permuted = evecs[:, torch.tensor([1, 0] + list(range(2, evals.numel())))]
    with pytest.raises(ValueError, match="does not diagonalise"):
        SpectralBackend(evals, permuted, validate_against=op)
    with pytest.raises(ValueError, match="does not diagonalise"):
        SpectralBackend(evals + 1e-3, evecs, validate_against=op)
    SpectralBackend(evals, evecs, validate_against=op)             # the correct one passes


def test_malformed_eigensystems_are_rejected():
    _, evals, evecs, _ = _toy()
    with pytest.raises(ValueError, match="square"):
        SpectralBackend(evals[:-1], evecs)
    with pytest.raises(ValueError, match="non-finite"):
        SpectralBackend(evals * float("nan"), evecs)


def test_validation_tolerance_is_dtype_scoped():
    """Plan §1.9 S3: 1e-12 is a float64 tolerance; float32 needs ~1e-6 and must say so."""
    op, evals, evecs, _ = _toy()
    op32 = CovarianceOperator(op.X.to(torch.float32), tau=0.05)
    with pytest.raises(ValueError, match="does not diagonalise"):
        SpectralBackend(evals, evecs, validate_against=op32, rtol=1e-12)
    SpectralBackend(evals, evecs, validate_against=op32, rtol=1e-5)


# ------------------------------------------------------------------ the reason it exists


@pytest.mark.skipif(not have_data, reason="fMRI bundle not built")
def test_the_dense_route_fixes_the_two_functions_remedy_c_was_adopted_for():
    """`c_skip` and `c_out` on the real ridge-dominated operator: 3.9e-1 / 2.1e-2 at K=256.

    The dense route makes them exact. Without this the benchmark would run its EDM head through an
    approximation with O(1) relative error.
    """
    from ccd.operators.cheb_audit import cheb_eval
    from ccd.operators.chebyshev import cheb_coeffs

    evals = torch.tensor(np.load(FMRI / "evals.npy"))
    evecs = torch.tensor(np.load(FMRI / "evecs.npy"))
    tau, lam = 0.01, evals
    lo, hi = float(evals.min()), float(evals.max())
    phi = ShiftedPowerRate(-4.0, lo, hi, FLOOR=0.077, F_min=0.02, R=120.0, tau=tau)
    spec = SpectralBackend(evals, evecs)
    v = torch.randn(360, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    # The eigensystem diagonalises the frozen S -- so "exact against U diag(f) U^T" is a statement
    # about the operator, not just about the backend's own arithmetic.
    Sd = torch.tensor(np.load(FMRI / "S_ccd.npy"))
    assert float((Sd @ evecs - evecs * lam[None, :]).norm() / (evecs * lam[None, :]).norm()) < 1e-12

    for factory, floor in ((bands.c_skip_fn, 1e-1), (bands.c_out_fn, 1e-2)):
        fn = factory(phi, tau, 0.02)
        truth = evecs @ torch.diag(fn(lam)) @ evecs.transpose(0, 1) @ v
        rel_dense = float((spec.apply(fn, v) - truth).norm() / truth.norm())
        assert rel_dense < 1e-12, f"{factory.__name__}: dense must be exact, got {rel_dense:.2e}"

        # ...and the route it replaces is NOT, at the top of the admissible ladder. Without this
        # comparison "dense is exact" would be a tautology about how SpectralBackend is defined.
        exact_band = fn(lam)
        approx = cheb_eval(cheb_coeffs(fn, lo, hi, 256, dtype=torch.float64), lam, lo, hi)
        rel_cheb = float((approx - exact_band).abs().max() / exact_band.abs().max())
        assert rel_cheb > floor, (
            f"{factory.__name__}: expected the K=256 Chebyshev route to fail above {floor:.0e}, "
            f"got {rel_cheb:.2e} -- if this ever passes, Remedy C's premise has changed"
        )


# ------------------------------------------------------------------ wiring


def test_colored_edm_defaults_to_the_matrix_free_path_unchanged():
    """Nothing outside a benchmark that opts in may change behaviour."""
    op, evals, evecs, _ = _toy()
    fwd = ColoredForward(op, float(evals.min()) * 0.99, float(evals.max()) * 1.01,
                         phi=phi_power(1.0), K=200)
    edm = ColoredEDM(fwd)
    assert edm.precond_backend is None
    v = torch.randn(24, 5, dtype=torch.float64)
    edm_dense = ColoredEDM(fwd, precond_backend=SpectralBackend(evals, evecs, validate_against=op))
    torch.testing.assert_close(edm.c_skip(v, 0.5), edm_dense.c_skip(v, 0.5), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(edm.c_out(v, 0.5), edm_dense.c_out(v, 0.5), rtol=1e-8, atol=1e-8)


def test_c_in_is_never_routed_through_the_precond_backend():
    """`c_in` is a Krylov solve, not a polynomial apply; routing it dense would be a silent
    change of algorithm rather than of accuracy."""
    op, evals, evecs, _ = _toy()
    fwd = ColoredForward(op, float(evals.min()) * 0.99, float(evals.max()) * 1.01,
                         phi=phi_power(1.0), K=200)

    class Tripwire(SpectralBackend):
        def apply(self, fn, v):
            self.used = True
            return super().apply(fn, v)

    backend = Tripwire(evals, evecs)
    backend.used = False
    edm = ColoredEDM(fwd, precond_backend=backend)
    v = torch.randn(24, 2, dtype=torch.float64)
    edm.c_in(v, 0.5)
    assert not backend.used, "c_in must not reach the preconditioner backend"
    edm.c_skip(v, 0.5)
    assert backend.used, "c_skip must reach it"


# ------------------------------------------------------------------ review round 6


@pytest.mark.parametrize("shape", [(24,), (24, 1), (24, 7)])
def test_apply_preserves_input_rank(shape):
    """`f[:, None]` broadcast a (d,) input into a (d, d) matrix -- silently, with a plausible
    result. Both ranks are now exercised."""
    op, evals, evecs, S = _toy()
    spec = SpectralBackend(evals, evecs, validate_against=op)
    v = torch.randn(*shape, dtype=torch.float64)
    out = spec.apply(lambda lam: lam, v)
    assert out.shape == v.shape
    torch.testing.assert_close(out, S @ v, rtol=1e-11, atol=1e-11)


def test_apply_rejects_a_wrong_leading_dimension():
    op, evals, evecs, _ = _toy()
    spec = SpectralBackend(evals, evecs)
    with pytest.raises(ValueError, match="leading dimension"):
        spec.apply(lambda lam: lam, torch.randn(25, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"\(d,\) or \(d, batch\)"):
        spec.apply(lambda lam: lam, torch.randn(24, 3, 2, dtype=torch.float64))


def test_vector_and_batch_paths_agree():
    op, evals, evecs, _ = _toy()
    spec = SpectralBackend(evals, evecs)
    fn = bands.m_fn(phi_power(1.0), 0.4)
    cols = [torch.randn(24, dtype=torch.float64) for _ in range(3)]
    stacked = torch.stack(cols, dim=1)
    torch.testing.assert_close(torch.stack([spec.apply(fn, c) for c in cols], dim=1),
                               spec.apply(fn, stacked), rtol=1e-12, atol=1e-12)


def test_rescaled_eigenvectors_are_rejected_by_the_orthonormality_clause():
    """Correspondence alone accepts them: scaling a column preserves `S u = lambda u` exactly,
    while `U diag(f) U^T` is then wrong by that column's scale squared."""
    op, evals, evecs, S = _toy()
    scaled = evecs.clone()
    scaled[:, 0] *= 2.0
    lhs = op.matvec(scaled)
    rhs = scaled * evals[None, :]
    assert float((lhs - rhs).norm() / rhs.norm()) < 1e-12, "correspondence really does still hold"

    with pytest.raises(ValueError, match="not orthonormal"):
        SpectralBackend(evals, scaled, validate_against=op)

    # and the reconstruction really would have been wrong
    bad = scaled @ torch.diag(evals) @ scaled.transpose(0, 1)
    assert float((bad - S).norm() / S.norm()) > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_validation_runs_on_the_device_the_eigensystem_lives_on():
    """⭐ Found by the first GPU run this backend ever had, and it was a hard failure.

    `_validate`'s orthonormality clause built its identity with `torch.eye(d, dtype=...)`, which
    defaults to CPU. On a CUDA eigensystem that raised `Expected all tensors to be on the same
    device` — so the dense primary path, the *primary* route for the whole fMRI benchmark, could not
    be constructed on GPU at all.

    Nothing caught it because every test and every smoke run built this backend on CPU, where the
    default device happens to be the right one. A defect that is invisible off-accelerator needs a
    test that is on one; asserting it on CPU would be asserting nothing.
    """
    op, evals, evecs, _ = _toy()
    dev = torch.device("cuda:0")
    spec = SpectralBackend(evals.to(dev), evecs.to(dev),
                           validate_against=CovarianceOperator(op.X.to(dev), tau=op.tau))
    assert spec.evecs.device.type == "cuda"

    # and the clause still bites there — a CUDA-only pass would be a check that cannot fail
    scaled = evecs.clone()
    scaled[:, 0] *= 2.0
    with pytest.raises(ValueError, match="not orthonormal"):
        SpectralBackend(evals.to(dev), scaled.to(dev),
                        validate_against=CovarianceOperator(op.X.to(dev), tau=op.tau))


# ------------------------------------------------------------------ round 6: the dense primary path


def test_the_dense_path_makes_the_unsplit_forms_correct_again():
    """The ruling's premise, pinned.

    The split was invented to rescue the MATRIX-FREE route, not the law. Once every apply is exact,
    the direct `Sigma_F^{1/2} eps` and `q^{1/2} xi` forms -- the ones §2.1b rejected -- are correct,
    and the primary path needs no split, no factor blocks and no second generator.
    """
    op, evals, evecs, S = _toy()
    spec = SpectralBackend(evals, evecs, validate_against=op)
    phi, F = phi_power(1.0), 0.7

    # direct map: Sigma_F^{1/2}; split map: h^{1/2} L with S = L L^T. Both have covariance
    # `map @ map.T`, and h^{1/2} S h^{1/2} = S h = Sigma_F, so the two must agree.
    direct = spec.apply(bands.sigma_sqrt_fn(phi, F), torch.eye(24, dtype=torch.float64))
    split = spec.apply(bands.h_sqrt_fn(phi, F), torch.linalg.cholesky(S))
    torch.testing.assert_close(direct @ direct.transpose(0, 1),
                               split @ split.transpose(0, 1), rtol=1e-10, atol=1e-10)


def test_forward_and_reverse_accept_a_backend_and_default_to_matrix_free():
    from ccd.denoiser.denoiser import ColoredDenoiser
    from ccd.reverse.learned import LearnedReverseSDE

    op, evals, evecs, _ = _toy()
    lo, hi = float(evals.min()) * 0.99, float(evals.max()) * 1.01
    spec = SpectralBackend(evals, evecs, validate_against=op)

    fwd_free = ColoredForward(op, lo, hi, phi=phi_power(1.0), K=200)
    fwd_dense = ColoredForward(op, lo, hi, phi=phi_power(1.0), K=200, backend=spec)
    assert fwd_free.backend is None and fwd_dense.backend is spec

    v = torch.randn(24, 3, dtype=torch.float64)
    for meth in ("mean", "sigma_half", "v_marginal"):
        torch.testing.assert_close(getattr(fwd_free, meth)(v, 0.5),
                                   getattr(fwd_dense, meth)(v, 0.5), rtol=1e-8, atol=1e-8)

    class _Body(torch.nn.Module):
        def __init__(self, fwd):
            super().__init__()
            self.fwd = fwd

        def forward(self, x, F, **kw):
            return torch.zeros_like(x)

    den = ColoredDenoiser(ColoredEDM(fwd_dense, precond_backend=spec), _Body(fwd_dense))
    rev = LearnedReverseSDE(den, K=200, backend=spec)
    assert rev.backend is spec
    assert LearnedReverseSDE(den, K=200).backend is None


def test_dense_c_in_is_exact_and_refuses_solver_kwargs():
    op, evals, evecs, _ = _toy()
    lo, hi = float(evals.min()) * 0.99, float(evals.max()) * 1.01
    spec = SpectralBackend(evals, evecs, validate_against=op)
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0), K=200)
    v = torch.randn(24, 3, dtype=torch.float64)

    krylov = ColoredEDM(fwd).c_in(v, 0.5, tol=1e-12, maxiter=2000)
    dense = ColoredEDM(fwd, solve_backend=spec).c_in(v, 0.5)
    torch.testing.assert_close(krylov, dense, rtol=1e-6, atol=1e-6)

    truth = evecs @ torch.diag(bands.v_fn(phi_power(1.0), 0.05, 0.5)(evals) ** -0.5) \
        @ evecs.transpose(0, 1) @ v
    torch.testing.assert_close(dense, truth, rtol=1e-11, atol=1e-11)

    with pytest.raises(ValueError, match="dense backend, which runs no solver"):
        ColoredEDM(fwd, solve_backend=spec).c_in(v, 0.5, tol=1e-9)
