"""Colored forward-diffusion marginals and draws (Phase I1).

Built entirely on the matrix-free :mod:`ccd.operators` engine: every forward object is
a smooth function of ``S`` applied by Chebyshev–Clenshaw, so the colored process is
constructed with **no matrix exponential and no eigendecomposition**. The colouring
rate ``φ`` selects the member of the colouring family (``φ(λ)=λ^a``).
"""

from ccd.forward.marginals import ColoredForward
from ccd.forward.rate import (
    NormPhi,
    PhiLogPoly,
    ShiftedPowerRate,
    TiledRate,
    f_star,
    generation_order,
    phi_power,
    snr,
    unit_crossing_F,
)
from ccd.forward.sampling import forward_draw, terminal_draw, whiten

__all__ = [
    "ColoredForward",
    "phi_power",
    "NormPhi",
    "PhiLogPoly",
    "ShiftedPowerRate",
    "TiledRate",
    "snr",
    "unit_crossing_F",
    "f_star",
    "generation_order",
    "forward_draw",
    "terminal_draw",
    "whiten",
]
