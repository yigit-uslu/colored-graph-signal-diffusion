"""Structural / equivariance gates for the learned VNN body ``F_θ`` (Phase V1).

The body is a stack of graph filters ``Σ_k S̃^k H Θ_k`` (PyG ``TAGConv``, ``normalize=False``) on the
affinely-rescaled shift ``S̃=(2S−(hi+lo)I)/(hi−lo)``, with scale-only FiLM conditioning and an optional
Mehler high-noise envelope. These tests pin the *design invariants* (V1 has no training yet):

* the graph reproduces the monomial filter ``Σ_k c_k S̃^k`` (``Ŝ = S̃``);
* a **linear** ``σ`` makes the body a pure polynomial in ``S̃`` ⇒ it commutes with ``S``; a nonlinearity
  breaks that (the σ is what buys non-graph-filter expressivity);
* ``F_θ(0)=0`` (scale-only FiLM + no signal-path bias — the centred-data / origin symmetry);
* scale-only FiLM and the envelope are each wired to ``F``, and both are ablatable;
* disjoint-union batching does not leak across signals.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ccd.denoiser import VNNBody
from ccd.forward import ColoredForward, phi_power
from ccd.operators import make_synthetic_operator, spectrum_bracket
from _oracles import dense_apply, rel_err

torch.manual_seed(0)
PHI_POLY = lambda lam: 0.5 + lam + 0.3 * lam ** 2      # non-power-law φ (φ-decoupling of the body)


def _fwd(d=10, n=80, tau=0.05, seed=0, phi=None, dtype=torch.float64, K=200):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    return ColoredForward(op, lo, hi, phi=phi or phi_power(1.0), K=K), op


def _body(fwd, **kw):
    kw.setdefault("width", 16)
    kw.setdefault("n_layers", 3)
    kw.setdefault("K_gf", 4)
    return VNNBody(fwd, **kw).double()


# --------------------------------------------------------- (1) the graph reproduces the monomial filter
def test_reproduces_graph_filter():
    from torch_geometric.nn import TAGConv
    fwd, op = _fwd()
    d = op.d
    lo, hi = fwd.lo, fwd.hi
    a, b = 2.0 / (hi - lo), -(hi + lo) / (hi - lo)
    S_tilde = a * op._dense().double() + b * torch.eye(d, dtype=torch.float64)
    body = _body(fwd)
    K = 4
    coeffs = torch.tensor([0.7, -1.3, 0.5, 0.2, -0.4], dtype=torch.float64)
    tag = TAGConv(1, 1, K=K, bias=False, normalize=False).double()
    with torch.no_grad():
        for k in range(K + 1):
            tag.lins[k].weight.copy_(coeffs[k].reshape(1, 1))
    u = torch.randn(d, 1, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    got = tag(u, body.edge_index, body.edge_weight)
    ref, Sk = torch.zeros_like(u), u.clone()
    for k in range(K + 1):
        ref = ref + coeffs[k] * Sk
        Sk = S_tilde @ Sk
    assert rel_err(got, ref) < 1e-12                    # Ŝ = S̃ and TAGConv computes monomial powers


# ----------------------------------------------------- (2) linear σ ⇒ graph filter ⇒ commutes with S
@pytest.mark.parametrize("phi", [phi_power(1.0), phi_power(0.0), PHI_POLY])
def test_commutes_when_linear(phi):
    fwd, op = _fwd(phi=phi)
    body = _body(fwd, act=nn.Identity())
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    F = 1.3
    assert rel_err(body(op.matvec(v), F), op.matvec(body(v, F))) < 1e-10


def test_commutes_when_linear_with_envelope():
    # the envelope η(S)=m_F(S)^{k0} is a function of S ⇒ commutation still holds with it on.
    fwd, op = _fwd()
    body = _body(fwd, act=nn.Identity(), mehler_envelope=True, k0=2.0)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    F = 0.8
    assert rel_err(body(op.matvec(v), F), op.matvec(body(v, F))) < 1e-9


def test_nonlinearity_breaks_commutation():
    fwd, op = _fwd()
    body = _body(fwd, act=nn.Tanh())
    with torch.no_grad():                               # perturb so tanh actually bends the map
        for p in body.parameters():
            p.add_(0.5 * torch.randn_like(p))
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    assert rel_err(body(op.matvec(v), 1.3), op.matvec(body(v, 1.3))) > 1e-2


# ------------------------------------------------------------------- (3) F_θ(0) = 0 (origin symmetry)
@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("condition", [False, True])
def test_zero_input_zero_output(envelope, condition):
    fwd, op = _fwd()
    body = _body(fwd, condition=condition, mehler_envelope=envelope)
    with torch.no_grad():                               # even a "trained" (perturbed) body maps 0 -> 0
        for p in body.parameters():
            p.add_(0.4 * torch.randn_like(p))
    z = torch.zeros(op.d, dtype=torch.float64)
    assert float(body(z, 1.0).norm()) < 1e-10


# --------------------------------------------------------------------- (4) batching does not leak
def test_batched_matches_columnwise():
    fwd, op = _fwd()
    body = _body(fwd, mehler_envelope=True)
    with torch.no_grad():                               # non-trivial params so the check has teeth
        for p in body.parameters():
            p.add_(0.3 * torch.randn_like(p))
    U = torch.randn(op.d, 5, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    F = 1.1
    batched = body(U, F)
    cols = torch.stack([body(U[:, j], F) for j in range(U.shape[1])], dim=1)
    assert rel_err(batched, cols) < 1e-10


# ------------------------------------------------------------- (5) conditioning / envelope wired to F
def test_condition_off_is_F_invariant():
    fwd, op = _fwd()
    body = _body(fwd, condition=False, mehler_envelope=False)
    U = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    assert rel_err(body(U, 0.3), body(U, 4.0)) < 1e-12   # no F dependence anywhere


def test_condition_on_uses_F():
    fwd, op = _fwd()
    body = _body(fwd, condition=True, mehler_envelope=False)
    with torch.no_grad():                               # FiLM heads are zero-init (γ≡1) ⇒ perturb them
        for h in body.film_heads:
            h.weight.add_(0.3 * torch.randn_like(h.weight))
            h.bias.add_(0.3 * torch.randn_like(h.bias))
    U = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    assert rel_err(body(U, 0.3), body(U, 4.0)) > 1e-3


def test_envelope_uses_F():
    fwd, op = _fwd()
    body = _body(fwd, condition=False, mehler_envelope=True, k0=2.0)
    U = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    assert rel_err(body(U, 0.3), body(U, 4.0)) > 1e-3


def test_envelope_decays_with_noise():
    # η=m_F(S)^{k0} → 0 as F→∞: the body's contribution vanishes at high noise (the Mehler asymptotic).
    fwd, op = _fwd()
    body = _body(fwd, condition=False, mehler_envelope=True, k0=2.0)
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    assert float(body(v, 0.2).norm()) > float(body(v, 20.0).norm())


@pytest.mark.parametrize("k0", [2.0, 3.0])
def test_envelope_matches_dense(k0):
    # Pin the EXACT gate η(λ)=exp(−½ k0 F φ(λ)) = m_F(λ)^{k0} against the dense oracle — decay/F-dependence
    # tests alone would pass for any exp(−c F φ), so this fixes the ½ constant AND the k0 knob (k0=3 is
    # non-default: dropping the ½ or hardcoding k0=2 both fail here).
    fwd, op = _fwd()
    body = _body(fwd, mehler_envelope=True, k0=k0)
    y = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    F = 0.7
    got = body._apply_envelope(y, F)
    want = dense_apply(lambda lam: torch.exp(-0.5 * k0 * F * fwd.phi(lam)), op, y)
    assert rel_err(got, want) < 1e-8


# --------------------------------------------------------------------------- shapes, dtypes, grads
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_and_dtype(dtype):
    op = make_synthetic_operator(80, 10, tau=0.05, generator=torch.Generator().manual_seed(0), dtype=dtype)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0), K=64)
    body = VNNBody(fwd, width=16, n_layers=2, K_gf=3, mehler_envelope=True).to(dtype)
    u1 = torch.randn(op.d, dtype=dtype)
    assert body(u1, 1.0).shape == (op.d,) and body(u1, 1.0).dtype == dtype
    uB = torch.randn(op.d, 6, dtype=dtype)
    assert body(uB, 1.0).shape == (op.d, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_gpu():
    # The model passes must run on GPU: a body built from a CUDA operator must place every parameter AND
    # buffer (the graph) on cuda — the __init__ device-sync — and forward+backward must run entirely
    # on-device. Guards against the dtype-only .to() regression (buffers on cuda, params on cpu → crash).
    dev = "cuda"
    op = make_synthetic_operator(80, 32, tau=0.05, generator=torch.Generator(device=dev).manual_seed(0),
                                 dtype=torch.float32, device=dev)
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0), K=64)
    body = VNNBody(fwd, width=32, n_layers=3, K_gf=4, mehler_envelope=True)
    assert {p.device.type for p in body.parameters()} | {b.device.type for b in body.buffers()} == {"cuda"}
    U = torch.randn(op.d, 8, device=dev)
    y = body(U, 1.3)
    assert y.device.type == "cuda" and y.shape == (op.d, 8)
    y.pow(2).sum().backward()
    grads = [p.grad for p in body.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_grad_flows():
    # Every parameter must receive a finite gradient — and crucially the conditioning subgraph
    # (cond_mlp + film_heads) must be trainable, not detached. We perturb the zero-init FiLM heads so
    # the conditioning path is active, then require NON-None finite grads for ALL params (no None-filter,
    # which would silently hide a detached FiLM path) and a nonzero total over the conditioning params.
    fwd, op = _fwd()
    body = _body(fwd, n_layers=2, K_gf=3, mehler_envelope=True)
    with torch.no_grad():
        for h in body.film_heads:
            h.weight.add_(0.3 * torch.randn_like(h.weight))
            h.bias.add_(0.3 * torch.randn_like(h.bias))
    U = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    body(U, 1.3).pow(2).sum().backward()
    named = dict(body.named_parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values())
    cond_grad = sum(float(p.grad.abs().sum()) for n, p in named.items()
                    if n.startswith("cond_mlp") or n.startswith("film_heads"))
    assert cond_grad > 0.0                                 # the scale-only FiLM path actually trains


# ------------------------------------------------------------------------------------- guards
def test_rejects_nonpositive_F():
    fwd, op = _fwd()
    body = _body(fwd)
    u = torch.randn(op.d, dtype=torch.float64)
    for F in (0.0, -1.0):
        with pytest.raises(ValueError):
            body(u, F)


def test_rejects_wrong_shape():
    fwd, op = _fwd()
    body = _body(fwd)
    with pytest.raises(ValueError):
        body(torch.randn(op.d + 1, dtype=torch.float64), 1.0)


# ------------------------------------------------- (rev 22) the dense route IS the message-passing one
#
# `forward` applies `_dense_conv` (matmuls); `_pyg_conv` is retained as the independent oracle. The
# swap is a MEMORY fix: on a complete graph PyG materialises one `(B·d², C)` message tensor per hop,
# which is 11.87 GiB at the registered `batch=256, d=360, width=96` and made step 8 un-runnable. The
# dense form is the same sum reassociated, so the whole burden is showing that "same" is measured.


def _both_routes(body, H, B):
    return [(body._dense_conv(c, H, B), body._pyg_conv(c, H, B)) for c in body.convs]


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-6)])
@pytest.mark.parametrize("B", [1, 4])
def test_dense_conv_matches_pyg_forward(dtype, tol, B):
    """Both dtypes and both batching paths — `B=1` skips `_batched_graph` entirely, so it is a
    genuinely different code path in the oracle and not a redundant case.

    The float32 bound is the sealed `Numerics.backend_rtol`; the reassociation is what costs the
    bits, and float64 pins that the difference really is reduction order and not a formula.
    """
    fwd, op = _fwd(dtype=dtype)
    body = VNNBody(fwd, width=16, n_layers=3, K_gf=4).to(dtype)
    H = torch.randn(B * op.d, 16, dtype=dtype)
    for dense, pyg in _both_routes(body, H, B):
        assert rel_err(dense, pyg) < tol, f"{rel_err(dense, pyg):.3e}"


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-11), (torch.float32, 1e-5)])
def test_dense_and_pyg_agree_on_input_and_parameter_gradients(dtype, tol):
    """Forward agreement is not enough: the body is *trained*, so the backward pass is what the
    experiment actually consumes. A reassociation that got the adjoint wrong would leave the
    forward exact and silently train a different model."""
    fwd, op = _fwd(dtype=dtype)
    body = VNNBody(fwd, width=16, n_layers=3, K_gf=4).to(dtype)
    B = 3
    u0 = torch.randn(op.d, B, dtype=dtype)                 # ONE input, both routes

    def run(route):
        u = u0.clone().requires_grad_(True)
        H = body.encoder(u.transpose(0, 1).reshape(B * op.d, 1))
        out = H
        for conv in body.convs:
            out = route(conv, out, B)
        loss = body.decoder(out).square().sum()
        grads = torch.autograd.grad(loss, [u] + list(body.parameters()), allow_unused=True)
        return grads

    for gd, gp in zip(run(body._dense_conv), run(body._pyg_conv)):
        if gd is None or gp is None:
            assert gd is None and gp is None                # unused either way, or neither
            continue
        assert rel_err(gd, gp) < tol, f"{rel_err(gd, gp):.3e}"


@pytest.mark.parametrize("condition", [False, True])
@pytest.mark.parametrize("envelope", [False, True])
def test_whole_body_forward_matches_the_pyg_route(condition, envelope):
    """End to end, with conditioning and the envelope on and off — the swap must not disturb the
    FiLM gains or the Mehler gate, both of which sit *around* the convolutions."""
    fwd, op = _fwd()
    body = _body(fwd, condition=condition, mehler_envelope=envelope)
    u = torch.randn(op.d, 5, dtype=torch.float64)

    got = body(u, 0.7)
    dense_conv = body._dense_conv
    try:                                                   # swap in the oracle for one call
        body._dense_conv = body._pyg_conv
        want = body(u, 0.7)
    finally:
        body._dense_conv = dense_conv
    assert rel_err(got, want) < 1e-12


def test_single_and_batched_inputs_agree_under_the_dense_route():
    """`u` of shape `(d,)` and one column of `(d, B)` must give the same answer — the reshape in
    `_dense_conv` is exactly where a batching bug would hide."""
    fwd, op = _fwd()
    body = _body(fwd)
    cols = [torch.randn(op.d, dtype=torch.float64) for _ in range(4)]
    stacked = torch.stack(cols, dim=1)
    single = torch.stack([body(c, 0.5) for c in cols], dim=1)
    assert rel_err(single, body(stacked, 0.5)) < 1e-12


def test_the_dense_shift_is_the_edge_list_it_replaced():
    """Derived from the buffers, not re-derived from `op`: whatever `_build_graph` decided about
    orientation, self-loops and the `[-1, 1]` rescaling is what the dense route applies."""
    fwd, op = _fwd()
    body = _body(fwd)
    scattered = torch.zeros_like(body.shift_dense)
    scattered[body.edge_index[0], body.edge_index[1]] = body.edge_weight
    assert torch.equal(body.shift_dense, scattered)
    assert body.shift_dense.shape == (op.d, op.d)
    assert float(body.shift_dense.abs().max()) > 0.0        # not a zero matrix passing vacuously


def test_the_state_dict_layout_is_unchanged_by_the_dense_route():
    """A checkpoint written before this change must still load, and one written after must not
    carry a new tensor — hence `persistent=False` on `shift_dense`. `Training.updates=2000` per
    (arm, seed) is too expensive to re-run over a serialisation detail."""
    fwd, _ = _fwd()
    body = _body(fwd)
    sd = body.state_dict()
    assert "shift_dense" not in sd
    assert {"edge_index", "edge_weight", "encoder.weight", "decoder.weight"} <= set(sd)
    assert body.load_state_dict(sd, strict=True) is not None      # round-trips strictly

    reloaded = _body(_fwd()[0])
    reloaded.load_state_dict(sd, strict=True)
    u = torch.randn(fwd.op.d, 2, dtype=torch.float64)
    assert rel_err(reloaded(u, 0.5), body(u, 0.5)) < 1e-12


def test_the_dense_route_is_the_one_forward_actually_uses():
    """The mutation that keeps every test above honest.

    All of them compare two routes; none of them establishes which one `forward` calls. Break the
    dense route and `forward` must break with it — otherwise the equivalence suite could be green
    while production quietly ran the 25.2 GiB path.
    """
    fwd, op = _fwd()
    body = _body(fwd)
    u = torch.randn(op.d, 2, dtype=torch.float64)
    baseline = body(u, 0.5)

    body._dense_conv = lambda conv, H, B: torch.zeros_like(H)
    assert float(body(u, 0.5).abs().max()) == 0.0
    assert float(baseline.abs().max()) > 0.0
