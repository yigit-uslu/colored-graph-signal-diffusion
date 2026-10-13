"""Learned denoiser components (Phase V).

The EDM denoiser ``D_θ = c_skip(S) x + c_out(S) F_θ(c_in(S) x; F)`` for the colored diffusion, built on
the matrix-free :mod:`ccd.operators` engine. V0 provides the closed-form colored-EDM preconditioners
(``c_skip, c_out, c_in``) and the analytic Gaussian head — the exact MMSE denoiser for Gaussian data and
the zero-parameter baseline for the (later) learned VNN body ``F_θ``.
"""

from ccd.denoiser.edm import ColoredEDM
from ccd.denoiser.vnn import VNNBody
from ccd.denoiser.denoiser import (
    ColoredDenoiser,
    cout_sq_bands,
    edm_loss,
    edm_loss_perband,
    edm_weight,
    sample_F_loguniform,
)
from ccd.denoiser.toy import BlockGMM, GMMinDCT, dct_matrix

__all__ = [
    "ColoredEDM",
    "VNNBody",
    "ColoredDenoiser",
    "edm_loss",
    "edm_loss_perband",
    "cout_sq_bands",
    "edm_weight",
    "sample_F_loguniform",
    "BlockGMM",
    "GMMinDCT",
    "dct_matrix",
]
