"""`ccd.factored_cin` — the factored `c_in`, tested on the library alone (no script imports).

* `FactoredCinEDM.c_in` IS `factored_c_in`, spends iterations + K_g matvecs and logs both.
* The factored `c_in` equals the dense `c_in`.
* Every refusal bites: g's degree at the policy's cap, a v_F that crosses zero more than once, and a
  non-converged solve.

The bitwise pin to the function the split measured (`gfs_matvec_cin_split.factorised_cin`) lives with
that script's tests, in `tests/test_gfs_matvec_cin_split.py`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ccd.degree_policy import DegreePolicy  # noqa: E402
from ccd.factored_cin import (  # noqa: E402
    FactoredCinEDM, FactoredCinError, factored_c_in, singular_root,
)
from ccd.forward.marginals import ColoredForward  # noqa: E402
from ccd.forward.rate import phi_power  # noqa: E402
from ccd.operators.backends import SpectralBackend  # noqa: E402
from ccd.operators.gram import DenseGramOperator  # noqa: E402
from ccd.operators.linop import CovarianceOperator  # noqa: E402

TAU = 0.05
POLICY = DegreePolicy(c_out=256, smooth=None, tol=1e-7, terminal=256, degree_dtype=torch.float64)
KW = {"tol": 1e-5, "maxiter": 500, "method": "zolotarev"}


def _setup(dtype, *, d=16, n=24, seed=3):
    g = torch.Generator().manual_seed(seed)
    X64 = torch.randn(d, n, generator=g, dtype=torch.float64)
    S = CovarianceOperator(X64, TAU).matvec(torch.eye(d, dtype=torch.float64))
    evals, evecs = torch.linalg.eigh(0.5 * (S + S.T))
    op = DenseGramOperator(X64.to(dtype), TAU)
    fwd = ColoredForward(op, float(evals.min()), float(evals.max()), phi=phi_power(1.0), K=None)
    bk = SpectralBackend(evals.to(dtype), evecs.to(dtype))
    v = torch.randn(d, 3, generator=g, dtype=torch.float64).to(dtype)
    return fwd, op, bk, v


def _rel(a, b):
    return float((a - b).norm()) / float(b.norm())


def _dense_c_in(fwd, backend, v, F):
    """`ColoredEDM.c_in`'s dense (`solve_backend`) branch."""
    v_min, _ = fwd._require_positive_bracket(F, "c_in")
    vfn = fwd._vmarg_fn(F)
    return backend.apply(lambda lam: vfn(lam).clamp_min(v_min) ** -0.5, v)


def test_the_edm_c_in_IS_factored_c_in_spends_iterations_plus_K_g_and_logs_both():
    fwd, op, _, v = _setup(torch.float32)
    log: list = []
    edm = FactoredCinEDM(fwd, policy=POLICY, degree_log=log)
    op.reset_count()
    out = edm.c_in(v, 0.3, **KW)
    spent = op.matvec_count
    ref, info = factored_c_in(fwd, v, 0.3, policy=POLICY, **KW)
    assert torch.equal(out, ref)
    assert [r["name"] for r in log] == ["c_in_solve", "c_in_g"]
    assert log[0]["K"] == info["iters"] and log[1]["K"] == info["K_g"] and log[0]["mu"] == info["mu"]
    assert spent == info["iters"] + info["K_g"]


@pytest.mark.parametrize("F", [0.02, 0.5, 10.0])
def test_the_factored_c_in_equals_the_dense_c_in(F):
    fwd, _, bk, v = _setup(torch.float64)
    out, _ = factored_c_in(fwd, v, F, policy=POLICY, tol=1e-10, maxiter=1000)
    assert _rel(out, _dense_c_in(fwd, bk, v, F)) < 1e-6


def test_ANY_shift_below_the_spectrum_gives_the_same_c_in():
    fwd, _, bk, v = _setup(torch.float64)
    mu = singular_root(fwd.phi, TAU, fwd.lo, 0.1) - 0.3 * fwd.lo
    out, info = factored_c_in(fwd, v, 0.1, policy=POLICY, tol=1e-10, maxiter=1000, mu=mu)
    assert info["mu"] == mu and _rel(out, _dense_c_in(fwd, bk, v, 0.1)) < 1e-6


def test_a_g_degree_AT_THE_CAP_is_refused():
    fwd, _, _, v = _setup(torch.float64)
    capped = DegreePolicy(tol=1e-7, degree_dtype=torch.float64, k_max=2)
    with pytest.raises(FactoredCinError, match="cap"):
        factored_c_in(fwd, v, 2.0, policy=capped, tol=1e-8, maxiter=500)


def test_the_same_call_under_the_default_cap_passes_ABOVE_that_degree():
    """Mutant for the refusal: g really needs more than the capped policy allows."""
    fwd, _, _, v = _setup(torch.float64)
    _, info = factored_c_in(fwd, v, 2.0, policy=POLICY, tol=1e-8, maxiter=500)
    assert 2 < info["K_g"] < POLICY.k_max


def _twice_crossing_clock(F: float):
    """φ with v_F(λ) = λ − (0.12 + 0.1 cos 40λ): three sign changes on [0, 0.3]."""
    return lambda lam: -torch.log((0.12 + 0.1 * torch.cos(40.0 * lam)) / TAU) / F


def test_a_v_F_crossing_zero_more_than_once_is_refused():
    F = 1.0
    phi = _twice_crossing_clock(F)
    lam = torch.linspace(0.0, 0.3, 100001, dtype=torch.float64)
    v = lam - TAU * torch.exp(-F * phi(lam))
    assert int((torch.diff(torch.sign(v)) != 0).sum()) == 3                 # the fixture is what it says
    with pytest.raises(FactoredCinError, match="changes sign 3 times"):
        singular_root(phi, TAU, 0.3, F)


def test_a_single_crossing_clock_returns_its_root():
    """Mutant for the refusal: the scan does not refuse a clock that crosses once."""
    mu = singular_root(phi_power(1.0), TAU, 0.3, 1.0)
    assert 0.0 < mu < 0.3
    assert abs(mu - TAU * math.exp(-1.0 * mu)) < 1e-12


def test_an_underflowing_v_F_at_zero_returns_zero():
    steep = lambda lam: 1e4 + 0.0 * lam                                     # noqa: E731
    assert singular_root(steep, TAU, 0.3, 1.0) == 0.0


def test_a_shift_AT_or_above_the_spectrum_is_refused():
    fwd, _, _, v = _setup(torch.float64)
    with pytest.raises(FactoredCinError, match="below the spectrum"):
        factored_c_in(fwd, v, 0.1, policy=POLICY, tol=1e-8, maxiter=100, mu=fwd.lo)


def test_a_non_converged_solve_is_refused_unless_asked_not_to():
    fwd, _, _, v = _setup(torch.float64)
    with pytest.raises(FactoredCinError, match="did not converge"):
        factored_c_in(fwd, v, 0.05, policy=POLICY, tol=1e-12, maxiter=1)
    _, info = factored_c_in(fwd, v, 0.05, policy=POLICY, tol=1e-12, maxiter=1, require_converged=False)
    assert info["converged"] is False


def test_the_factored_edm_refuses_a_dense_backend():
    fwd, op, bk, _ = _setup(torch.float64)
    dense = ColoredForward(op, fwd.lo, fwd.hi, phi=fwd.phi, K=None, backend=bk)
    with pytest.raises(ValueError, match="matrix-free-route"):
        FactoredCinEDM(dense)
