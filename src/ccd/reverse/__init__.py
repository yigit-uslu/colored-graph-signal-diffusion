"""Reverse-time (generative) samplers on the matrix-free colored marginals (Phase I3).

The forward marginals (:mod:`ccd.forward`) are the corruption; generation is the reverse
transport back to the data law. In the **Gaussian-scoped** case the analytic score ``−V_F^{-1}x``
makes both reverse dynamics smooth functions of ``S`` applied by Chebyshev–Clenshaw — **no
eigendecomposition and no explicit inverse**:

* I3a — :class:`GaussianReversePFODE`, the deterministic probability-flow ODE (per-band rescaling
  by ``√(v_{F_b}/v_{F_a})``);
* I3b-1 — :class:`GaussianReverseSDE`, the stochastic reverse-time SDE (exact ancestral transition
  ``μ(S) x + √q(S) ξ``, plus a Euler–Maruyama scheme).

A learned score and the steep-inverse (shifted-CG / Zolotarev) preconditioners are later increments.
"""

from ccd.reverse.pf_ode import GaussianReversePFODE, reverse_f_grid
from ccd.reverse.sde import GaussianReverseSDE
from ccd.reverse.learned import LearnedReversePFODE, LearnedReverseSDE

__all__ = [
    "GaussianReversePFODE",
    "GaussianReverseSDE",
    "LearnedReversePFODE",
    "LearnedReverseSDE",
    "reverse_f_grid",
]
