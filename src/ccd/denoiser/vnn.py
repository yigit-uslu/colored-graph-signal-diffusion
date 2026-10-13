"""The learned VNN body ``F_θ`` — a graph filter polynomial in ``S`` (Phase V1).

The EDM denoiser is ``D_θ(x, F) = c_skip(S) x + c_out(S) F_θ(c_in(S) x; F)`` (V0 supplies the
preconditioners and the analytic head; V2 wires the loss). This module is the learned residual
``F_θ`` — the *nonlinear closure of the analytic filter class*:

* The exact Gaussian score ``−V_F^{-1} x`` is itself a function of ``S`` (a graph filter), so the
  family ``{polynomial filters in S}`` is exactly where the analytic machinery lives. ``F_θ`` learns a
  stack of such filters ``Σ_k S̃^k H Θ_k`` (PyG :class:`TAGConv`, ``normalize=False``) with a pointwise
  nonlinearity ``σ`` between layers — the ``σ`` supplies the polynomial-in-``x`` (cumulant) expressivity
  a single linear filter lacks. This is a *prior* that the data's non-Gaussianity is organised along
  ``S``'s eigenbasis (the covariance-structured, near-Gaussian regime — the S-aligned GMM toy is
  in-model); it is misspecified if the higher cumulants are unrelated to ``S``.

Design choices (locked; see the plan memo ``vnn-denoiser-plan``):

* **Graph shift operator** ``S̃ = (2S − (hi+lo)I)/(hi−lo)`` (``a S + b I``, spectrum ``[-1,1]``) — a
  function of ``S`` so it shares ``S``'s eigenbasis and commutes with it (zero theory / expressivity
  cost), but its bounded spectrum pre-conditions the learned *monomial* filter (raw ``λ^k`` over
  ``[lo,hi]`` is Vandermonde-ill-conditioned; ``x^k`` over ``[-1,1]`` is bounded). This is the *same*
  affine map :func:`ccd.operators.apply_fun` uses internally. ``S̃`` is not PSD — fine, only the
  covariance ``S`` must be; the GNN shift need not. For V1–V3 ``op`` is dense ``Ĉ_n+τI`` → a **complete
  graph** materialised once through the public matvec oracle (``op.matvec(I)``, a scoped dense departure
  confined to this body; toy ``d~32–64`` ⇒ ``d²`` edges). Sparsification, if ever, must happen at the
  operator/dataset level so the one shift operator stays shared (SPD-safe tapering / graphical lasso),
  never per-sample — that is a separate future workstream.

* **Conditioning** — the input ``c_in x`` is ``F``-invariant in scale (EDM whitening) and ``c_out(S,F)``
  only rescales per band, so the body cannot infer ``F`` from its input; it needs an explicit ``F``
  feature. We use **scale-only FiLM** off a scalar φ-aware log-SNR ``ν(F)``: a per-channel positive gain
  ``γ_l(ν) = exp(·) ⊙ H`` with **no additive shift** and **no biases anywhere on the signal path**.
  Scale-only is what preserves the two structural invariants: (i) ``F_θ(0)=0`` (a node-constant shift
  ``β`` — or any signal-path bias — would let ``F_θ`` emit nonzero from zero input, wrong for centred
  data), and (ii) with a linear ``σ`` the whole body stays a *pure* polynomial in ``S̃`` and commutes
  with ``S`` (an additive ``β`` makes it affine and breaks that). FiLM modulates the *channel* axis, so
  it never mixes nodes and leaves the ``S``-alignment intact. ``condition=False`` ablates it (γ≡1).

* **Optional Mehler envelope** — the non-Gaussian body must *vanish at high noise*: the k-th Hermite
  mode decays as ``m_F(λ)^k``, so the leading correction decays as ``m_F(λ)^{k_0}`` (``k_0`` the leading
  non-Gaussian order). Neither FiLM (agnostic to the F→∞ limit) nor ``c_out`` (which → ``√Σ_0``, a
  constant, not 0) bakes this in. ``mehler_envelope=True`` multiplies the body output by the fixed
  function-of-``S`` gate ``η(S) = m_F(S)^{k_0} = exp(−½ k_0 F φ(S))`` (commutes, ``S``-aligned) so the
  correct high-noise decay is structural rather than learned. Default off (V2 ablates it on/off).

Everything on the signal path is a function of ``S`` or a per-channel/per-node linear map, so no
eigendecomposition and no matrix exponential is ever formed; PyG is imported only here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch_geometric.nn import TAGConv                      # PyG import localised to ccd.denoiser.vnn

from ccd.forward.marginals import ColoredForward
from ccd.forward.rate import snr
from ccd.operators.chebyshev import apply_fun


class VNNBody(nn.Module):
    """The learned score body ``F_θ(u; F)`` — a conditioned stack of graph filters in ``S̃``.

    Args:
        fwd: the :class:`ColoredForward` (carries ``op``, the bracket ``[lo, hi]``, the colouring ``φ``
            and the Chebyshev degree ``K`` used by the optional envelope).
        width: hidden channel width.
        n_layers: number of ``TAGConv`` layers ``L``.
        K_gf: graph-filter order per layer (hops ``k=0..K_gf`` of ``S̃``).
        act: pointwise nonlinearity between layers. **Must satisfy ``act(0)=0``** (SiLU/tanh/ReLU/identity
            do; sigmoid/GELU-with-bias do not) so that ``F_θ(0)=0`` holds. Default :class:`~torch.nn.SiLU`.
        condition: if ``True`` (default) apply scale-only FiLM from ``ν(F)``; ``False`` ablates it (γ≡1,
            so the body is ``F``-agnostic unless the envelope is on).
        mehler_envelope: if ``True`` multiply the output by ``η(S)=m_F(S)^{k_0}`` (default ``False``).
        k0: leading non-Gaussian order for the envelope (``∈{2,3,4}``; default ``2``).
        cond_ngrid: grid size for the scalar log-SNR conditioning feature.

    The forward takes the *pre-whitened* signal ``u = c_in(S) x`` of shape ``(d,)`` or ``(d, B)`` and the
    diffusion step ``F > 0``, and returns the residual of the same shape. Whitening (``c_in``) and the
    ``c_skip``/``c_out`` wrapping live in V0/V2 — this module is only the body.
    """

    def __init__(
        self,
        fwd: ColoredForward,
        *,
        width: int = 64,
        n_layers: int = 3,
        K_gf: int = 4,
        act: nn.Module | None = None,
        condition: bool = True,
        mehler_envelope: bool = False,
        k0: float = 2.0,
        cond_ngrid: int = 257,
    ):
        super().__init__()
        self.fwd = fwd
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.K_gf = int(K_gf)
        self.condition = bool(condition)
        self.mehler_envelope = bool(mehler_envelope)
        self.k0 = float(k0)
        self.cond_ngrid = int(cond_ngrid)
        self.act = act if act is not None else nn.SiLU()

        # Graph shift operator S̃ = a S + b I on [-1, 1] as a complete graph (self-loops included so
        # Ŝ = S̃ exactly, hence Ŝ^k = S̃^k). Registered as buffers so `.to(dtype/device)` moves them
        # (the long edge_index is left integer; only the float edge_weight is cast).
        edge_index, edge_weight = self._build_graph()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # ⭐ The SAME S̃, dense — what `forward` actually applies (see `_dense_conv`). `persistent=
        # False` keeps `state_dict()` byte-identical to what it was before this route existed, so
        # checkpoints written either side of the change interchange.
        self.register_buffer("shift_dense", self._dense_shift(edge_index, edge_weight),
                             persistent=False)

        # Signal path: encoder → L × [TAGConv → (scale-only FiLM) → σ] → decoder. NO biases anywhere on
        # this path (encoder/conv/decoder all bias=False) so F_θ(0)=0 and the linear-σ body is a pure
        # polynomial in S̃.
        self.encoder = nn.Linear(1, self.width, bias=False)
        self.convs = nn.ModuleList(
            [TAGConv(self.width, self.width, K=self.K_gf, bias=False, normalize=False)
             for _ in range(self.n_layers)]
        )
        self.decoder = nn.Linear(self.width, 1, bias=False)

        # Scale-only FiLM: a shared conditioner ν(F) → embedding, then per-layer heads → log-gain.
        # γ_l = exp(head_l(embed)); heads zero-initialised so γ_l ≡ 1 at init (the body starts as the
        # un-modulated filter stack). Biases here are on the CONDITIONING path (they set the gain), not
        # the signal path, so they do not break F_θ(0)=0 or commutation.
        self.cond_dim = 1                                    # scalar log-SNR (spectral variant swaps this)
        if self.condition:
            self.cond_mlp = nn.Sequential(nn.Linear(self.cond_dim, self.width), nn.SiLU())
            self.film_heads = nn.ModuleList([nn.Linear(self.width, self.width) for _ in range(self.n_layers)])
            for h in self.film_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

        # Sync the whole module (Linear/TAGConv params *and* the graph buffers) to the operator's dtype
        # AND device — the buffers are already on op.device, so the params must follow or a forward on a
        # non-CPU operator would split cpu/cuda and crash in the first matmul.
        self.to(device=fwd.op.device, dtype=fwd.op.dtype)

    # ------------------------------------------------------------------ graph construction
    def _build_graph(self) -> tuple[Tensor, Tensor]:
        """Materialise ``S̃`` densely through the public matvec oracle and emit a complete-graph
        ``(edge_index, edge_weight)`` with ``edge_weight[e] = S̃[i, j]`` for edge ``e=(j→i)``.

        ``S̃`` is symmetric, so orientation is immaterial; the diagonal (self-loop) edges carry
        ``S̃[i,i]`` so that the sparse propagation matrix ``Ŝ`` equals ``S̃`` and ``Ŝ^k = S̃^k``.
        """
        op = self.fwd.op
        d = op.d
        lo, hi = self.fwd.lo, self.fwd.hi
        a = 2.0 / (hi - lo)
        b = -(hi + lo) / (hi - lo)
        eye = torch.eye(d, dtype=op.dtype, device=op.device)
        S = op.matvec(eye)                                   # dense S via the matvec oracle (scoped)
        S_tilde = a * S + b * eye                            # spectrum in [-1, 1]
        idx = torch.arange(d, device=op.device)
        src = idx.repeat_interleave(d)                       # (d²,)
        dst = idx.repeat(d)                                  # (d²,)
        edge_index = torch.stack([src, dst], dim=0).long()   # (2, d²)
        edge_weight = S_tilde[src, dst].contiguous()         # (d²,); = S̃[i,j] (symmetric ⇒ Ŝ = S̃)
        return edge_index, edge_weight

    def _dense_shift(self, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """Scatter the complete-graph edge list back into the ``(d, d)`` matrix it came from.

        Built from the buffers rather than recomputed from ``op`` so the two representations cannot
        disagree: whatever ``_build_graph`` decided — orientation, self-loops, the ``[-1, 1]``
        rescaling — is what the dense route applies, by construction rather than by a second
        derivation that has to be kept in step.
        """
        d = self.fwd.op.d
        S = torch.zeros(d, d, dtype=edge_weight.dtype, device=edge_weight.device)
        S[edge_index[0], edge_index[1]] = edge_weight
        return S

    def _dense_conv(self, conv: TAGConv, H: Tensor, B: int) -> Tensor:
        """``TAGConv`` evaluated as matmuls: ``lins[0](H) + Σ_k lins[k](S̃^k H)``.

        ⭐ **Plan revision 22, and it is a memory fix, not a speed one.** ``TAGConv`` is
        message-passing, so PyG materialises one ``(E, C)`` tensor per hop — and this graph is
        **complete**, ``E = B·d²``. At the registered ``batch=256``, ``d=360``, ``width=96`` that is
        **11.87 GiB for a single hop**, and one training step measured **0.0982 GiB per sample**:
        ≈25.2 GiB against a 23.69 GiB card and a sealed 20 GiB cap. The step-8 probe could not run.

        On a complete graph the propagation *is* a dense matmul, so ``S̃ @ H`` computes the same
        quantity in ``(B·d, C)`` — smaller by a factor of exactly ``d``, i.e. 0.494 GiB where the
        sparse form needed 25.2. Nothing about the model changes: same parameters, same optimiser
        state, same ``state_dict`` keys, same architecture. It is the identical sum, reassociated.

        The reduction ORDER differs, so float32 results agree to ~1e-7 rather than bit-exactly —
        inside the sealed ``Numerics.backend_rtol`` of 1e-6, and float64 agrees to ~1e-15. This is
        the same argument §2.1b-ter already made for routing the spectral operations densely.

        ``conv.lins`` is reused directly, never re-implemented: this reads TAGConv's own parameters
        in TAGConv's own order, so a PyG upgrade that changed either would be caught by the
        equivalence tests rather than silently diverging.
        """
        x = H.reshape(B, self.fwd.op.d, -1)
        out = conv.lins[0](x)
        for lin in conv.lins[1:]:
            x = self.shift_dense @ x                         # (d,d) @ (B,d,C) broadcasts over B
            out = out + lin(x)
        if conv.bias is not None:                            # constructed bias=False; kept honest
            out = out + conv.bias
        return out.reshape(H.shape[0], -1)

    def _pyg_conv(self, conv: TAGConv, H: Tensor, B: int) -> Tensor:
        """The message-passing route, retained as the **test oracle** for `_dense_conv`.

        Not dead code and not a fallback: it is the independent implementation the equivalence
        tests compare against, which is the only thing that keeps "same convolution" a measured
        claim rather than an assertion. Never called by `forward` — see `_dense_conv` for why.
        """
        ei, ew = self._batched_graph(B)
        return conv(H, ei, ew)

    def _batched_graph(self, B: int) -> tuple[Tensor, Tensor]:
        """Disjoint union of ``B`` copies of the shift graph (block-diagonal, ``N = B·d`` nodes).

        Used only by `_pyg_conv`, i.e. only by the tests. ``B·d²`` edges is why.
        """
        ei, ew = self.edge_index, self.edge_weight
        if B == 1:
            return ei, ew
        d, E = self.fwd.op.d, ei.shape[1]
        offs = (d * torch.arange(B, device=ei.device)).view(1, B, 1)
        ei_b = (ei.view(2, 1, E) + offs).reshape(2, B * E)
        ew_b = ew.repeat(B)
        return ei_b, ew_b

    # ------------------------------------------------------------------ conditioning
    def _cond_feature(self, F: float) -> Tensor:
        """Scalar φ-aware log-SNR feature ``ν(F) = log(mean_λ SNR_λ(F))`` on a grid over ``[lo, hi]``.

        Monotone decreasing in ``F`` and φ-aware (φ enters through ``SNR``), so the FiLM MLP can place
        the noise level. Returns shape ``(cond_dim,)``; the per-band ``SNR_λ(F)`` feed is the drop-in
        spectral variant (deferred noise-dependent-taps ablation).
        """
        p = self.encoder.weight
        lam = torch.linspace(self.fwd.lo, self.fwd.hi, self.cond_ngrid, dtype=p.dtype, device=p.device)
        s = snr(lam, F, self.fwd.phi, self.fwd.op.tau)       # per-band SNR ≥ 0
        nu = torch.log(s.mean() + 1e-30)
        return nu.reshape(1)

    def _film_gains(self, F: float) -> list[Tensor]:
        """Per-layer per-channel positive gains ``γ_l(ν) = exp(head_l(embed))`` (scale-only FiLM)."""
        embed = self.cond_mlp(self._cond_feature(F))
        return [torch.exp(h(embed)) for h in self.film_heads]

    # ------------------------------------------------------------------ envelope
    def _apply_envelope(self, y: Tensor, F: float) -> Tensor:
        """Multiply by the Mehler high-noise gate ``η(S) = exp(−½ k_0 F φ(S)) = m_F(S)^{k_0}``.

        Uses an *adaptive* Chebyshev degree, **not** ``fwd.K``: the envelope is a smooth exponential
        (adaptive degree ≈ a handful), whereas ``fwd.K`` is sized for the steep preconditioners
        (``K~200–400`` near the ``λ=τ`` boundary layer). Inheriting it would force ~30× wasted matvecs.

        .. warning::
           "Smooth exponential ⇒ a handful of terms" is a statement about ``η`` as a function of
           ``Fφ``, and it fails when ``φ`` itself is near-singular. **Measured on the fMRI ladder**
           (``d=360``, ``R=120``, ``k_0=2``), worst relative error over a 60-point ``F`` grid:

           ===========  ========  ========  ========
           clock        ``K=32``  ``K=64``  ``K=256``
           ===========  ========  ========  ========
           ``a=0``      5.6e-15   1.3e-14   6.0e-14
           ``ā=−0.50``  3.3e-01   1.5e-01   **2.3e-03**
           ``ā=−0.83``  7.8e-04   1.5e-06   5.3e-14
           ``ā=−1.68``  2.5e-09   1.2e-14   5.0e-14
           ===========  ========  ========  ========

           ``η`` inherits ``φ``'s branch point at ``λ=−s``, so the mild two-point-pinned arm misses
           ``tol`` at **every** admissible degree — and :func:`chebyshev_degree` *caps* silently at
           ``k_max`` rather than raising, so the shortfall is invisible. This is a **generative**
           error: the body output is multiplied by a wrong gate, arm-dependently.

           With a ``backend`` on the wrapped :class:`ColoredForward` the envelope is exact and the
           whole issue is moot — which is why the fMRI benchmark's primary path injects one
           (implementation plan §2.1b-ter). ``backend=None`` keeps the historical route unchanged.
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
            raise ValueError(f"VNNBody requires F > 0 (log-SNR conditioning is undefined at F=0); got F={F}")
        single = u.dim() == 1
        U = u.unsqueeze(1) if single else u                  # (d, B)
        if U.dim() != 2 or U.shape[0] != self.fwd.op.d:
            raise ValueError(f"u must be (d,) or (d, B) with d={self.fwd.op.d}; got {tuple(u.shape)}")
        d, B = U.shape

        x = U.transpose(0, 1).reshape(B * d, 1)              # node features: block b = signal b
        gammas = self._film_gains(F) if self.condition else None

        H = self.encoder(x)                                  # (B·d, width)
        for l, conv in enumerate(self.convs):
            H = self._dense_conv(conv, H, B)                 # rev 22; `_pyg_conv` is the oracle
            if self.condition:
                H = gammas[l].unsqueeze(0) * H               # scale-only FiLM (per-channel, over nodes)
            H = self.act(H)
        y = self.decoder(H)                                  # (B·d, 1)

        y = y.reshape(B, d).transpose(0, 1)                  # (d, B)
        if self.mehler_envelope:
            y = self._apply_envelope(y, F)
        return y.squeeze(1) if single else y
