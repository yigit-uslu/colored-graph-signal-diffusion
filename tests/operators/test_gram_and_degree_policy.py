"""`ccd.operators.gram` and `ccd.degree_policy` — every behaviour with a mutant that shows it bites."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ccd.degree_policy import (  # noqa: E402
    DegreePolicy, PolicyColoredEDM, PolicyLearnedReversePFODE, policy_apply,
)
from ccd.denoiser.denoiser import ColoredDenoiser  # noqa: E402
from ccd.denoiser.edm import ColoredEDM  # noqa: E402
from ccd.denoiser.vnn import VNNBody  # noqa: E402
from ccd.forward.marginals import ColoredForward  # noqa: E402
from ccd.operators.backends import SpectralBackend  # noqa: E402
from ccd.operators.chebyshev import apply_fun, chebyshev_degree  # noqa: E402
from ccd.operators.gram import DenseGramOperator  # noqa: E402
from ccd.operators.linop import CovarianceOperator  # noqa: E402
from ccd.reverse.learned import LearnedReversePFODE  # noqa: E402

F = 0.7


def _X(d=24, n=40, seed=3, dtype=torch.float64):
    return torch.randn(d, n, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _fwd(op, K=None, backend=None):
    S = CovarianceOperator(op.X.double(), op.tau).matvec(torch.eye(op.d, dtype=torch.float64))
    w = torch.linalg.eigvalsh(0.5 * (S + S.T))
    return ColoredForward(op, float(w.min()), float(w.max()) * 1.01, phi=None, K=K, backend=backend)


def _probe(d=24, B=3, seed=5, dtype=torch.float64):
    return torch.randn(d, B, generator=torch.Generator().manual_seed(seed), dtype=dtype)


# ------------------------------------------------------------------ lever 1: the dense Gram

@pytest.mark.parametrize("shape", [(24,), (24, 5)])
def test_the_gram_matvec_equals_the_factor_matvec(shape):
    X = _X()
    v = torch.randn(*shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    a = DenseGramOperator(X, 0.05).matvec(v)
    b = CovarianceOperator(X, 0.05).matvec(v)
    assert a.shape == b.shape
    assert float((a - b).norm() / b.norm()) < 1e-12


def test_a_gram_built_WITHOUT_the_ridge_would_fail_that_agreement():
    """Mutant: the agreement above is tight enough to see a missing τ."""
    X, v = _X(), _probe()
    op = DenseGramOperator(X, 0.05)
    no_tau = op.S - 0.05 * torch.eye(op.d, dtype=op.S.dtype)
    b = CovarianceOperator(X, 0.05).matvec(v)
    assert float((no_tau @ v - b).norm() / b.norm()) > 1e-3


def test_the_gram_is_a_covariance_operator_keeps_X_counts_and_checks_shape():
    X = _X()
    op = DenseGramOperator(X, 0.05)
    assert isinstance(op, CovarianceOperator) and op.X is X
    assert op.shape == (24, 24) and op.dtype == X.dtype
    op.matvec(_probe()); op.matvec(_probe())
    assert op.matvec_count == 2
    with pytest.raises(ValueError, match="leading dim"):
        op.matvec(torch.zeros(7, dtype=torch.float64))


def test_the_stored_S_is_exactly_symmetric_even_in_float32():
    op = DenseGramOperator(_X(dtype=torch.float32), 0.05)
    assert op.S.dtype == torch.float32 and torch.equal(op.S, op.S.T)


def test_the_float32_gram_matches_the_float32_factor_route_to_rounding():
    X = _X(dtype=torch.float32)
    v = _probe(dtype=torch.float32)
    a = DenseGramOperator(X, 0.05).matvec(v)
    b = CovarianceOperator(X, 0.05).matvec(v)
    assert float((a - b).norm() / b.norm()) < 1e-5


def test_from_operator_and_the_flops_ratio():
    base = CovarianceOperator(_X(d=24, n=40), 0.05)
    op = DenseGramOperator.from_operator(base)
    assert op.X is base.X and op.tau == base.tau
    assert op.flops_ratio_vs_factor == pytest.approx(24 / 80)


def test_a_chebyshev_apply_is_unchanged_by_swapping_in_the_gram():
    X, v = _X(), _probe()
    gram, factor = DenseGramOperator(X, 0.05), CovarianceOperator(X, 0.05)
    fwd = _fwd(factor)
    f = lambda lam: torch.exp(-0.5 * lam)                             # noqa: E731
    a = apply_fun(f, gram, v, fwd.lo, fwd.hi, 32)
    b = apply_fun(f, factor, v, fwd.lo, fwd.hi, 32)
    assert float((a - b).norm() / b.norm()) < 1e-10


# ------------------------------------------------------------------ the policy object

@pytest.mark.parametrize("kw,match", [({"c_out": 0}, "c_out"), ({"smooth": 0}, "smooth"),
                                      ({"tol": 0.0}, "tol"), ({"k_max": 0}, "k_max"),
                                      ({"terminal": 0}, "terminal"),
                                      ({"degree_dtype": torch.float16}, "degree_dtype")])
def test_a_policy_REFUSES_nonsense(kw, match):
    with pytest.raises(ValueError, match=match):
        DegreePolicy(**kw)


def test_c_out_is_fixed_and_everything_else_takes_smooth():
    p = DegreePolicy(c_out=200, smooth=None)
    assert p.degree_for("c_out") == 200
    assert p.degree_for("c_skip") is None and p.degree_for("m_F") is None
    assert p.degree_for("terminal") is None
    assert DegreePolicy(c_out=64, smooth=12).degree_for("a_F") == 12
    assert DegreePolicy(c_out=64, smooth=12).degree_for("terminal") == 12


def test_a_fixed_terminal_degree_governs_the_terminal_ALONE():
    p = DegreePolicy(c_out=64, smooth=None, terminal=256)
    assert p.degree_for("terminal") == 256
    assert p.degree_for("c_skip") is None and p.degree_for("phi") is None
    f = lambda lam: torch.sqrt(lam)                                   # noqa: E731
    assert p.resolve("terminal", f, 0.1, 2.0, dtype=torch.float32, device="cpu") == 256


def _tight_case():
    """A smooth function whose float32 coefficient tail sits above tol=1e-7 where float64's does not."""
    f = lambda lam: torch.exp(-3.0 * lam)                             # noqa: E731
    return f, 0.05, 2.0, 1e-7


def test_degree_dtype_reads_the_tail_in_that_dtype_and_None_keeps_the_applys():
    f, lo, hi, tol = _tight_case()
    p64 = DegreePolicy(tol=tol, degree_dtype=torch.float64)
    assert p64.resolve("m_F", f, lo, hi, dtype=torch.float32, device="cpu") == max(
        chebyshev_degree(f, lo, hi, tol=tol, dtype=torch.float64), 1)
    p = DegreePolicy(tol=tol)
    assert p.resolve("m_F", f, lo, hi, dtype=torch.float32, device="cpu") == max(
        chebyshev_degree(f, lo, hi, tol=tol, dtype=torch.float32), 1)


def test_at_a_tight_tol_the_float32_rule_OVER_RESOLVES_and_float64_does_not():
    """Mutant for the field above: the two rules really differ where the docstring says they do."""
    f, lo, hi, tol = _tight_case()
    k32 = DegreePolicy(tol=tol).resolve("m_F", f, lo, hi, dtype=torch.float32, device="cpu")
    k64 = DegreePolicy(tol=tol, degree_dtype=torch.float64).resolve(
        "m_F", f, lo, hi, dtype=torch.float32, device="cpu")
    assert k64 < k32


def test_degree_dtype_changes_the_degree_but_not_the_applys_dtype():
    f, lo, hi, tol = _tight_case()
    X = _X(dtype=torch.float32)
    op = CovarianceOperator(X, 0.05)
    fwd = _fwd(op)
    v = _probe(dtype=torch.float32)
    p = DegreePolicy(tol=tol, degree_dtype=torch.float64)
    want = p.resolve("m_F", f, fwd.lo, fwd.hi, dtype=torch.float32, device="cpu")
    op.reset_count()
    out = policy_apply(p, "m_F", f, op, v, fwd.lo, fwd.hi)
    assert op.matvec_count == want and out.dtype == torch.float32
    assert torch.equal(out, apply_fun(f, op, v, fwd.lo, fwd.hi, want))


def test_resolve_IS_the_trees_adaptive_rule():
    fwd = _fwd(CovarianceOperator(_X(), 0.05))
    f = lambda lam: torch.exp(-3.0 * lam)                             # noqa: E731
    got = DegreePolicy().resolve("c_skip", f, fwd.lo, fwd.hi, dtype=torch.float64, device="cpu")
    assert got == max(chebyshev_degree(f, fwd.lo, fwd.hi, dtype=torch.float64), 1)


def test_policy_apply_adaptive_is_BITWISE_apply_fun_with_K_None_and_logs():
    op = CovarianceOperator(_X(), 0.05)
    fwd, v = _fwd(op), _probe()
    f = lambda lam: torch.exp(-3.0 * lam)                             # noqa: E731
    log: list = []
    a = policy_apply(DegreePolicy(), "m_F", f, op, v, fwd.lo, fwd.hi, log=log, F=1.5)
    b = apply_fun(f, op, v, fwd.lo, fwd.hi, None)
    assert torch.equal(a, b)
    assert log == [{"name": "m_F", "F": 1.5, "K": max(chebyshev_degree(f, fwd.lo, fwd.hi,
                                                                        dtype=torch.float64), 1)}]


# ------------------------------------------------------------------ PolicyColoredEDM

def test_c_out_spends_EXACTLY_its_fixed_degree_and_c_skip_its_adaptive_one():
    op = CovarianceOperator(_X(), 0.05)
    fwd, v = _fwd(op), _probe()
    edm = PolicyColoredEDM(fwd, policy=DegreePolicy(c_out=48))
    op.reset_count(); edm.c_out(v, F)
    assert op.matvec_count == 48
    want = DegreePolicy().resolve("c_skip", edm._cskip_fn(F), fwd.lo, fwd.hi,
                                  dtype=torch.float64, device="cpu")
    op.reset_count(); edm.c_skip(v, F)
    assert op.matvec_count == want < 48


@pytest.mark.parametrize("name", ["c_skip", "c_out", "c_out_sq"])
def test_at_a_UNIFORM_degree_the_policy_edm_IS_the_parent(name):
    op = CovarianceOperator(_X(), 0.05)
    v = _probe()
    parent = ColoredEDM(_fwd(op, K=40))
    policy = PolicyColoredEDM(_fwd(op, K=None), policy=DegreePolicy(c_out=40, smooth=40))
    assert torch.equal(getattr(policy, name)(v, F), getattr(parent, name)(v, F))


def test_a_uniform_policy_with_a_DIFFERENT_degree_is_not_the_parent():
    """Mutant: the identity above is not vacuous."""
    op = CovarianceOperator(_X(), 0.05)
    v = _probe()
    parent = ColoredEDM(_fwd(op, K=40))
    other = PolicyColoredEDM(_fwd(op), policy=DegreePolicy(c_out=12, smooth=40))
    assert not torch.equal(other.c_out(v, F), parent.c_out(v, F))


def test_the_parents_shared_apply_is_DISABLED():
    edm = PolicyColoredEDM(_fwd(CovarianceOperator(_X(), 0.05)))
    with pytest.raises(RuntimeError, match="NAMED degree"):
        edm._apply_precond(lambda lam: lam, _probe())
    assert torch.equal(edm.head(_probe(), F), edm.c_skip(_probe(), F))


def test_every_parent_caller_of_apply_precond_is_overridden_by_name():
    """If ColoredEDM grows a new `_apply_precond` caller, this fails before the RuntimeError can."""
    callers = {n for n, m in vars(ColoredEDM).items()
               if callable(m) and n != "_apply_precond" and "_apply_precond(" in inspect.getsource(m)}
    assert callers == {"c_skip", "c_out", "c_out_sq"}
    assert callers <= set(vars(PolicyColoredEDM))


def test_the_policy_edm_REFUSES_a_dense_backend():
    op = CovarianceOperator(_X(), 0.05)
    S = op.matvec(torch.eye(24, dtype=torch.float64))
    w, U = torch.linalg.eigh(0.5 * (S + S.T))
    fwd = _fwd(op, backend=SpectralBackend(w, U, validate_against=op))
    with pytest.raises(ValueError, match="matrix-free-route"):
        PolicyColoredEDM(fwd)


def test_the_edm_logs_the_degree_each_preconditioner_used():
    log: list = []
    op = CovarianceOperator(_X(), 0.05)
    edm = PolicyColoredEDM(_fwd(op), policy=DegreePolicy(c_out=30), degree_log=log)
    edm.c_out(_probe(), F); edm.c_skip(_probe(), F)
    assert [r["name"] for r in log] == ["c_out", "c_skip"]
    assert log[0]["K"] == 30 and log[0]["F"] == F


# ------------------------------------------------------------------ PolicyLearnedReversePFODE

def _denoiser(op, *, policy):
    fwd = _fwd(op)
    body = VNNBody(fwd, width=4, n_layers=1, K_gf=2)
    return ColoredDenoiser(PolicyColoredEDM(fwd, policy=policy), body)


def test_at_a_UNIFORM_degree_the_policy_sampler_drift_IS_the_parents():
    op = CovarianceOperator(_X(), 0.05)
    den = _denoiser(op, policy=DegreePolicy(c_out=40, smooth=40))
    x = _probe()
    a = PolicyLearnedReversePFODE(den, policy=DegreePolicy(c_out=40, smooth=40)).drift(x, F)
    b = LearnedReversePFODE(den, K=40).drift(x, F)
    assert torch.equal(a, b)


def test_the_sampler_names_each_drift_band_and_resolves_each_adaptively():
    op = CovarianceOperator(_X(), 0.05)
    log: list = []
    den = _denoiser(op, policy=DegreePolicy(c_out=40))
    s = PolicyLearnedReversePFODE(den, policy=DegreePolicy(c_out=40), degree_log=log)
    s.drift(_probe(), F)
    names = [r["name"] for r in log]
    assert names == ["m_F", "phi", "a_F"]
    assert log[0]["K"] == max(chebyshev_degree(s._mean_fn(F), s.lo, s.hi, dtype=torch.float64), 1)


def test_tagging_the_drift_bands_does_not_change_their_values():
    op = CovarianceOperator(_X(), 0.05)
    den = _denoiser(op, policy=DegreePolicy())
    s, p = PolicyLearnedReversePFODE(den), LearnedReversePFODE(den)
    lam = torch.linspace(s.lo, s.hi, 33, dtype=torch.float64)
    assert torch.equal(s._mean_fn(F)(lam), p._mean_fn(F)(lam))
    assert torch.equal(s._a_fn(F)(lam), p._a_fn(F)(lam))


def test_a_fixed_smooth_degree_spends_exactly_that_on_each_drift_band():
    op = CovarianceOperator(_X(), 0.05)
    den = _denoiser(op, policy=DegreePolicy(c_out=40, smooth=9))
    s = PolicyLearnedReversePFODE(den, policy=DegreePolicy(c_out=40, smooth=9))
    op.reset_count()
    s._apply(s._mean_fn(F), _probe())
    assert op.matvec_count == 9
