"""The learned fully-connected body ``F_θ`` — the Tier-3 hierarchical-mixture architecture.

Why a second body class
-----------------------
:class:`~ccd.denoiser.vnn.VNNBody` is a *graph filter* stack in ``S̃``: every layer mixes coordinates
only through polynomials of the operator, which encodes the prior "the data's non-Gaussianity is
organised along ``S``'s eigenbasis". That prior is the thing under test in the fMRI arc, and it is a
deliberate restriction of the hypothesis class.

The ``d = 3`` hierarchical-mixture clock experiment needs the opposite: a body with **no structural
opinion about the clock or the covariance**, so that any difference between arms is attributable to the
forward process rather than to an architecture that happens to suit one colouring. The brief (§5.1) asks
for a plain MLP, ``d → width → width → width → d``, common to all four arms. That is this module.

What is shared with ``VNNBody``, deliberately
---------------------------------------------
The two classes present the **same contract**, because the same denoiser wrappers and the same
subclass-two-hooks trick must work for both:

* ``forward(u, F)`` takes the pre-whitened signal ``u = c_in(S) x`` of shape ``(d,)`` or ``(d, B)`` and
  returns the residual of the same shape;
* **no biases anywhere on the signal path** and an activation with ``act(0) = 0``, so ``F_θ(0) = 0``
  exactly — a structural invariant, not a learned one;
* **scale-only FiLM** from a scalar level feature: a per-channel positive gain ``γ_l = exp(head_l(embed))``
  with no additive shift, zero-initialised so ``γ_l ≡ 1`` at init;
* the two colour-specific hooks ``_cond_feature`` / ``_apply_envelope``, which a subclass replaces to
  obtain the VP analogue (:class:`ccd.vp.VPMLPBody`, mirroring :class:`ccd.vp.VPBody`).

What is different
-----------------
The signal path is dense: ``Linear(d, width) → [Linear(width, width)] × (n_layers−1) → Linear(width, d)``,
with FiLM and the activation after each hidden map. There is no graph, no shift operator, and **no PyG
import** — which is also why the two hooks below are a narrow *copy* of ``VNNBody``'s rather than an
import from it: importing :mod:`ccd.denoiser.vnn` would pull ``torch_geometric`` into every run of an
experiment that has no graph in it. The copy is closed by
``tests/targets/test_hmclock_mlp_body.py::test_the_copied_hooks_agree_with_VNNBodys_originals``, which
constructs both bodies on the same forward process and asserts the two hooks agree to float64 — so the
duplication cannot drift silently.

Parameter count is identical across all four Tier-3 arms by construction: the colored arms and VP differ
only in the two hooks, neither of which owns a parameter. Constructed under the same seed, the four arms'
initial weights are therefore **bit-identical**, which is what makes a paired seed genuinely paired at
initialisation (plan D1/D8).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ccd.forward.rate import snr
from ccd.operators.chebyshev import apply_fun


class MLPBody(nn.Module):
    """The learned score body ``F_θ(u; F)`` — a conditioned fully-connected stack.

    Args:
        fwd: the forward process, touched only for ``op`` (``d``/dtype/device), the spectral bracket
            ``[lo, hi]``, and — inside the two hooks alone — ``phi``, ``op.tau`` and ``backend``. A
            :class:`~ccd.forward.marginals.ColoredForward` for the colored arms; VP passes the
            :class:`ccd.vp.GraphOnlyForward` shim, whose ``phi=None`` raises loudly if anything
            unexpected reaches for it.
        width: hidden width (the brief's 64).
        n_layers: number of hidden representations of size ``width``. ``n_layers=3`` gives the brief's
            ``d → width → width → width → d``: one encoder, ``n_layers−1`` hidden maps, one decoder.
        act: pointwise nonlinearity. **Must satisfy ``act(0)=0``** (SiLU/tanh/ReLU/identity do) or
            ``F_θ(0)=0`` fails. Default :class:`~torch.nn.SiLU`, the repository-standard smooth choice.
        condition: if ``True`` (default) apply scale-only FiLM from the scalar level feature.
        mehler_envelope: if ``True`` multiply the output by ``η(S)=m_F(S)^{k_0}``. **Default ``False``,
            and registered OFF for Tier 3** (ruling Q13): the envelope is a clock-dependent structural
            gate, so enabling it would make the architecture itself differ by arm in a mechanism
            experiment.
        k0: leading non-Gaussian order for the envelope.
        cond_ngrid: grid size for the scalar log-SNR conditioning feature.

    ``F_θ(0) = 0`` holds exactly for any ``act`` with ``act(0)=0``, because every map on the signal path
    is bias-free and FiLM is multiplicative. The envelope, if on, is linear in ``y`` and preserves it.
    """

    def __init__(
        self,
        fwd,
        *,
        width: int = 64,
        n_layers: int = 3,
        act: nn.Module | None = None,
        condition: bool = True,
        mehler_envelope: bool = False,
        k0: float = 2.0,
        cond_ngrid: int = 257,
    ):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError(f"MLPBody needs n_layers >= 1; got {n_layers}")
        if int(width) < 1:
            raise ValueError(f"MLPBody needs width >= 1; got {width}")
        self.fwd = fwd
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.condition = bool(condition)
        self.mehler_envelope = bool(mehler_envelope)
        self.k0 = float(k0)
        self.cond_ngrid = int(cond_ngrid)
        self.act = act if act is not None else nn.SiLU()

        d = fwd.op.d
        # Signal path: d → width → ... → width → d. NO biases anywhere here, so F_θ(0)=0.
        self.encoder = nn.Linear(d, self.width, bias=False)
        self.hidden = nn.ModuleList(
            [nn.Linear(self.width, self.width, bias=False) for _ in range(self.n_layers - 1)]
        )
        self.decoder = nn.Linear(self.width, d, bias=False)

        # Scale-only FiLM, exactly as in VNNBody: a shared conditioner ν(F) → embedding, then per-layer
        # heads → log-gain, zero-initialised so γ_l ≡ 1 at init. The biases here are on the CONDITIONING
        # path (they set the gain), not the signal path, so they do not break F_θ(0)=0.
        self.cond_dim = 1
        if self.condition:
            self.cond_mlp = nn.Sequential(nn.Linear(self.cond_dim, self.width), nn.SiLU())
            self.film_heads = nn.ModuleList(
                [nn.Linear(self.width, self.width) for _ in range(self.n_layers)]
            )
            for h in self.film_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

        self.to(device=fwd.op.device, dtype=fwd.op.dtype)

    # ------------------------------------------------------------------ conditioning
    def _cond_feature(self, F: float) -> Tensor:
        """Scalar φ-aware log-SNR feature ``ν(F) = log(mean_λ SNR_λ(F))`` on a grid over ``[lo, hi]``.

        A narrow copy of :meth:`ccd.denoiser.vnn.VNNBody._cond_feature` — see the module docstring for
        why it is copied rather than imported, and for the gate that keeps the two identical.
        """
        p = self.encoder.weight
        lam = torch.linspace(self.fwd.lo, self.fwd.hi, self.cond_ngrid, dtype=p.dtype, device=p.device)
        s = snr(lam, F, self.fwd.phi, self.fwd.op.tau)
        nu = torch.log(s.mean() + 1e-30)
        return nu.reshape(1)

    def _film_gains(self, F: float) -> list[Tensor]:
        """Per-layer per-channel positive gains ``γ_l(ν) = exp(head_l(embed))`` (scale-only FiLM)."""
        embed = self.cond_mlp(self._cond_feature(F))
        return [torch.exp(h(embed)) for h in self.film_heads]

    # ------------------------------------------------------------------ envelope
    def _apply_envelope(self, y: Tensor, F: float) -> Tensor:
        """Multiply by the Mehler high-noise gate ``η(S) = exp(−½ k_0 F φ(S)) = m_F(S)^{k_0}``.

        A narrow copy of :meth:`ccd.denoiser.vnn.VNNBody._apply_envelope`. Registered OFF for Tier 3;
        it exists so the contract matches ``VNNBody``'s and so the gate that compares the two hooks has
        something to compare.
        """
        k0 = self.k0
        eta = lambda lam: torch.exp(-0.5 * k0 * F * self.fwd.phi(lam))
        if self.fwd.backend is not None:
            return self.fwd.backend.apply(eta, y)
        tol = 1e-4 if self.fwd.op.dtype == torch.float32 else 1e-9
        return apply_fun(eta, self.fwd.op, y, self.fwd.lo, self.fwd.hi, None, tol=tol)

    # ------------------------------------------------------------------ forward
    def forward(self, u: Tensor, F: float) -> Tensor:
        if F <= 0.0:
            raise ValueError(f"MLPBody requires F > 0 (log-SNR conditioning is undefined at F=0); got F={F}")
        single = u.dim() == 1
        U = u.unsqueeze(1) if single else u                   # (d, B)
        if U.dim() != 2 or U.shape[0] != self.fwd.op.d:
            raise ValueError(f"u must be (d,) or (d, B) with d={self.fwd.op.d}; got {tuple(u.shape)}")

        X = U.transpose(0, 1)                                 # (B, d) — samples on the batch axis
        gammas = self._film_gains(F) if self.condition else None

        H = self.encoder(X)                                   # (B, width)
        if self.condition:
            H = gammas[0].unsqueeze(0) * H
        H = self.act(H)
        for l, lin in enumerate(self.hidden):
            H = lin(H)
            if self.condition:
                H = gammas[l + 1].unsqueeze(0) * H
            H = self.act(H)
        Y = self.decoder(H)                                   # (B, d)

        y = Y.transpose(0, 1)                                 # (d, B)
        if self.mehler_envelope:
            y = self._apply_envelope(y, F)
        return y.squeeze(1) if single else y
