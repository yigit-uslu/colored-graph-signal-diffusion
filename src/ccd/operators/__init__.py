"""Matrix-free spectral operators for colored covariance diffusion (Phase I0).

The forward colored-diffusion operators are all functions ``f(S)`` of the single
covariance shift operator ``S = Ĉ_n + τI``. This package supplies the two
ingredients needed to apply any such ``f(S)`` to a vector **without forming S,
without an eigendecomposition, and without a matrix exponential**:

* :class:`CovarianceOperator` — ``S`` as a matvec oracle ``v ↦ (1/n)X(Xᵀv)+τv``;
* :func:`spectrum_bracket` — a matvec-only enclosing interval ``[τ, λ_hi] ⊇ spec(S)``;
* :func:`apply_fun` / :class:`ChebyshevBasis` — the Chebyshev–Clenshaw application
  of ``f(S)v`` using only matvecs with ``S``.
"""

from ccd.operators.linop import CovarianceOperator, make_synthetic_operator
from ccd.operators.circulant import CirculantOperator
from ccd.operators.bracket import spectrum_bracket
from ccd.operators.chebyshev import (
    ChebyshevBasis,
    apply_fun,
    cheb_coeffs,
    chebyshev_degree,
)
from ccd.operators.solvers import cg, shifted_cg
from ccd.operators.rational import invsqrt_poles, n_poles_for

__all__ = [
    "CovarianceOperator",
    "CirculantOperator",
    "make_synthetic_operator",
    "spectrum_bracket",
    "apply_fun",
    "cheb_coeffs",
    "chebyshev_degree",
    "ChebyshevBasis",
    "cg",
    "shifted_cg",
    "invsqrt_poles",
    "n_poles_for",
]
