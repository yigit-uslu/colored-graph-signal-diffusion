"""Oracle tests for the Phase I0 matrix-free spectral operators.

Every matvec-only primitive is hard-gated against a dense eigendecomposition / matrix
exponential baseline (in :mod:`_oracles`). Tolerances are sized for float32: the
baseline is float64, so what we measure is the float32 implementation error.
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import tokenize

import pytest
import torch

from ccd.operators import (
    ChebyshevBasis,
    apply_fun,
    chebyshev_degree,
    make_synthetic_operator,
    spectrum_bracket,
)
from _oracles import dense_apply, dense_expm_apply, rel_err

# float32 ⇒ accuracy floor ~1e-6; gate the matrix-free error well above it but tight.
TOL = 1e-3


def _op(d=48, n=300, tau=0.05, decay=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return make_synthetic_operator(n, d, tau=tau, decay=decay, generator=g)


def _vec(d, seed=1, batch=None):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=g) if batch is None else torch.randn(d, batch, generator=g)


def _bracket(op, seed=2):
    return spectrum_bracket(op, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------- matvec oracle

def test_matvec_matches_dense():
    op = _op()
    v = _vec(op.d)
    exact = op._dense().double() @ v.double()
    assert rel_err(op.matvec(v), exact) < 1e-4


def test_matvec_batched():
    op = _op()
    V = _vec(op.d, batch=4)
    exact = op._dense().double() @ V.double()
    assert rel_err(op.matvec(V), exact) < 1e-4


# ----------------------------------------------------------------- spectrum bracket

def test_bracket_contains_spectrum():
    op = _op()
    lo, hi = _bracket(op)
    evals = torch.linalg.eigvalsh(op._dense().double())
    assert lo == pytest.approx(op.tau)
    assert lo <= float(evals.min()) + 1e-6
    assert hi >= float(evals.max())


def test_bracket_not_loose():
    op = _op()
    lo, hi = _bracket(op)
    lam_max = float(torch.linalg.eigvalsh(op._dense().double()).max())
    assert hi <= 2.0 * lam_max


def test_too_small_bracket_blows_up():
    # A Chebyshev filter on a too-small interval leaves part of spec(S̃) outside [-1,1],
    # so the Clenshaw iterates grow — a silently mis-bracketed run must be detectable.
    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    bad_hi = lo + 0.4 * (hi - lo)  # below the top of the spectrum
    f = lambda x: torch.exp(-x)
    approx = apply_fun(f, op, v, lo, bad_hi, K=60)
    exact = dense_apply(lambda x: torch.exp(-x), op, v)
    assert (not torch.isfinite(approx).all()) or rel_err(approx, exact) > 1.0


# ----------------------------------------------------------------- apply_fun vs dense

def test_apply_fun_exp_vs_dense():
    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    f = lambda x: torch.exp(-x)
    approx = apply_fun(f, op, v, lo, hi)                 # adaptive K
    assert rel_err(approx, dense_apply(lambda x: torch.exp(-x), op, v)) < TOL
    assert rel_err(approx, dense_expm_apply(1.0, op, v)) < TOL  # independent expm baseline


def test_apply_fun_noise_factor_vs_dense():
    # Σ_t^{1/2} = sqrt(λ (1 - e^{-Fλ})) — the colored noise factor.
    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    F = 2.0
    f = lambda x: torch.sqrt(x * (1.0 - torch.exp(-F * x)))
    approx = apply_fun(f, op, v, lo, hi)
    exact = dense_apply(lambda x: torch.sqrt(x * (1.0 - torch.exp(-F * x))), op, v)
    assert rel_err(approx, exact) < TOL


def test_apply_fun_batched():
    op = _op()
    V = _vec(op.d, batch=5)
    lo, hi = _bracket(op)
    f = lambda x: torch.exp(-x)
    approx = apply_fun(f, op, V, lo, hi)
    exact = dense_apply(lambda x: torch.exp(-x), op, V)
    assert rel_err(approx, exact) < TOL


def test_funm_crosscheck():
    scipy_linalg = pytest.importorskip("scipy.linalg")
    import numpy as np

    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    S_np = op._dense().double().numpy()
    M = scipy_linalg.funm(S_np, lambda x: np.exp(-x))
    exact = torch.from_numpy(np.real(M)) @ v.double()
    approx = apply_fun(lambda x: torch.exp(-x), op, v, lo, hi)
    assert rel_err(approx, exact) < TOL


# ----------------------------------------------------------------- convergence / efficiency

def test_geometric_convergence_in_degree():
    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    f = lambda x: torch.exp(-20.0 * x)                  # steep ⇒ needs several terms
    exact = dense_apply(lambda x: torch.exp(-20.0 * x), op, v)
    errs = [rel_err(apply_fun(f, op, v, lo, hi, K=K), exact) for K in (4, 8, 16, 24)]
    assert errs[0] > errs[1] > errs[2]                  # decreasing while above the float32 floor
    assert errs[3] < 1e-3
    assert chebyshev_degree(f, lo, hi) < 64             # adaptive degree stays small (efficient)


def test_basis_amortisation_and_matvec_count():
    op = _op()
    v = _vec(op.d)
    lo, hi = _bracket(op)
    K = 20
    F = 2.0
    funcs = [
        lambda x: torch.exp(-F * x / 2.0),                       # m_t
        lambda x: torch.sqrt(x * (1.0 - torch.exp(-F * x))),     # Σ_t^{1/2}
        lambda x: x - op.tau * torch.exp(-F * x),                # V_t
    ]

    # Amortised path: build the basis once (K matvecs), contract all three with no more.
    op.reset_count()
    basis = ChebyshevBasis(op, v, lo, hi, K)
    assert op.matvec_count == K
    shared = [basis.contract(f) for f in funcs]
    assert op.matvec_count == K                                   # contraction is matvec-free

    # Independent path: each apply_fun costs K matvecs (K, not 2K, per function).
    op.reset_count()
    separate = [apply_fun(f, op, v, lo, hi, K=K) for f in funcs]
    assert op.matvec_count == 3 * K

    for a, b in zip(shared, separate):
        assert rel_err(a, b) < 1e-4                              # same result, fewer matvecs


# ----------------------------------------------------------------- matrix-free guarantee

def _code_only(src_text):
    """Source with docstrings and comments blanked, line numbering preserved.

    Prose cannot violate an invariant, and matching it produced pure noise — the guard used to flag
    ``spectral.py``'s own "Nothing here calls ``torch.linalg.eigh``" note as a violation.
    """
    lines = src_text.splitlines()
    for node in ast.walk(ast.parse(src_text)):                     # bare string statements = docstrings
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = ""
    for tok in tokenize.generate_tokens(io.StringIO(src_text).readline):
        if tok.type == tokenize.COMMENT:
            r, c = tok.start
            if lines[r - 1]:
                lines[r - 1] = lines[r - 1][:c]
    return lines


def test_no_eigendecomposition_in_src():
    """The matrix-free invariant: never eig/expm/dense-solve an **operator-sized** matrix in ``src/ccd``.

    Scoped to *code* (docstrings/comments stripped), and to matrices whose size grows with ``d``.
    Exceptions must say so **in place** with a ``# matrix-free-ok:`` pragma, so each one justifies
    itself where it lives rather than by a file-level allowlist. Two kinds are permitted:

    1. **Small, ``d``-independent linear algebra** — a Lanczos ``m×m`` tridiagonal, a ``(deg+1)²``
       coefficient fit. The pragma names the size bound.
    2. **Gate-only oracles that are ``d``-sized by design** and never run on a matrix-free path —
       currently just ``vp.GaussianOracleDenoiser``, whose sole caller is ``scripts/vp_preflight.py``
       and whose whole purpose is to be the dense ground truth the sampler is checked against. The
       pragma names the caller instead of a bound. This category is deliberately narrow: it exempts
       code from the invariant rather than satisfying it, so anything landing here should be
       reachable only from gates, never from ``src``'s own call graph.

    (``qr`` stays out of the pattern entirely: ``make_synthetic_operator`` uses it to build synthetic
    *data*, not to apply an operator.)

    Note the pattern is ``\\beigh`` — anchored at the *start* only, so ``eigh_tridiagonal`` is caught
    (the previous ``\\beigh\\b`` silently exempted the one genuine eigensolve in ``src``) while
    ``weight`` is not (dropping the boundary entirely matches ``w-eigh-t``).
    """
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "ccd"
    forbidden = re.compile(
        r"\beigh|eigvalsh|eigvals|matrix_exp|\bexpm\b|\bfunm\b|linalg\.eig\b|linalg\.solve"
        r"|linalg\.inv\b|torch\.inverse|\.svd\(|svdvals|cholesky|lu_factor|lu_solve"
    )
    offenders = []
    for p in sorted(src.rglob("*.py")):
        raw = p.read_text().splitlines()
        for i, line in enumerate(_code_only(p.read_text())):
            m = forbidden.search(line)
            if m and "matrix-free-ok:" not in raw[i]:              # pragma read from the ORIGINAL line
                offenders.append(f"{p.relative_to(src)}:{i + 1} [{m.group(0)}] {line.strip()[:70]}")
    assert not offenders, (
        "operator-sized eigendecomposition/solve leaked into src/ccd:\n  " + "\n  ".join(offenders)
        + "\nIf the matrix is small and d-independent, mark it `# matrix-free-ok: <size bound>`."
    )
