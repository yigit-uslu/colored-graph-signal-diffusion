#!/usr/bin/env python
"""GPU environment smoke test — does this workstation's `torch_env` actually run CCD?

Run this FIRST on any newly cloned machine, before launching real work. It answers one
question: *is the Python/conda environment on this box wired correctly for colored
covariance diffusion on the GPU?* It is deliberately not a benchmark and not a
reproducibility check — nothing here is seeded for cross-machine agreement, no number is
compared against a reference, and passing says only "the code runs".

    conda run --no-capture-output -n torch_env python scripts/smoke_gpu.py

⚠️  The `--no-capture-output` is not optional. `conda run` without it buffers stdout until
exit, so a hang looks identical to silence. (And never pipe this script into `conda run
python -` via a heredoc: that combination exits 0 without ever reaching Python.)

**Validating an environment you must not disturb.** This script needs `ccd` importable, but
NOT installed — run it against an untouched environment with:

    PYTHONPATH=/path/to/colored-covariance-diffusion/src python scripts/smoke_gpu.py

That is the same `sys.path` entry `pip install -e .` would create (the editable install is a
one-line .pth file holding exactly that directory), so a pass here predicts a pass after
installing, while writing nothing. See the git-versioning skill §1 for the full
snapshot -> dry-run -> install -> diff procedure.

Seven stages, each isolated so a failure names the broken dependency rather than a symptom
20 frames downstream:

    1  interpreter + package versions          which python, torch, PyG, drivers
    2  CUDA is genuinely usable                allocation + kernel + synchronize
    3  PyTorch Geometric works on the device    TAGConv forward on GPU
    4  the `ccd` package imports                i.e. `pip install -e . --no-deps` was run
    5  colored forward marginals on device      Chebyshev matvec engine, no eigendecomposition
    6  VNN denoiser training step  ← the point  loss -> backward -> Adam, params must move
    7  learned reverse PF-ODE                   the generative path, a few steps

Stage 6 is the one that matters. Stages 1-5 exist so that when it fails you already know
whether to blame the driver, the PyG wheels, the editable install, or the numerics.

**Why stage 3 is separate from stage 6.** PyTorch Geometric enters this codebase through
exactly one import — `TAGConv` in `ccd.denoiser.vnn` — so a smoke test that trained any
*other* denoiser would pass with PyG entirely absent. Stage 3 pins PyG on its own, and
stage 6 routes through `VNNBody` specifically to keep it on the tested path. PyG is by far
the most fragile part of this stack to install: its wheels are built against a specific
(torch, CUDA) pair, and a mismatch typically imports fine and then dies inside the first
scatter on the device.

Exit code is 0 only if every stage passes; any failure exits 1 and prints the traceback.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass, field

# Many small tensor ops on a busy host: uncapped intra-op threads have cost >25x wall-clock
# here before (see AGENTS.md "Host hazards"). Set before torch is imported.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch  # noqa: E402


# --------------------------------------------------------------------------- stage plumbing

@dataclass
class Result:
    name: str
    ok: bool
    seconds: float
    detail: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)


RESULTS: list[Result] = []
WARNINGS: list[str] = []


def stage(name: str, fn, *, fatal: bool = True) -> Result:
    """Run one stage, capture its outcome, and keep going unless it is fatal."""
    print(f"\n[{len(RESULTS) + 1}/7] {name}")
    print("-" * 78)
    notes: list[str] = []
    t0 = time.perf_counter()
    try:
        detail = fn(notes) or ""
        res = Result(name, True, time.perf_counter() - t0, detail, notes=notes)
        print(f"  PASS  ({res.seconds:.2f}s)")
    except Exception:
        res = Result(name, False, time.perf_counter() - t0,
                     error=traceback.format_exc(), notes=notes)
        print(f"  FAIL  ({res.seconds:.2f}s)")
        print(_indent(res.error))
    RESULTS.append(res)
    if not res.ok and fatal:
        _summary()
        print("\nStopping: the stages after this one depend on it and would fail for the "
              "wrong reason.")
        raise SystemExit(1)
    return res


def _indent(text: str, pad: str = "        ") -> str:
    return "\n".join(pad + line for line in text.rstrip().splitlines())


def say(notes: list[str], line: str) -> None:
    notes.append(line)
    print(f"        {line}")


def _declared_python_floor() -> tuple[int, ...] | None:
    """Read `requires-python` out of pyproject.toml rather than restating it here.

    A version number duplicated into a message is precisely the hand-maintained fact that
    drifts silently (`docs/evidence_practices.md` principle 3) — this script would happily
    keep citing 3.11 for years after the floor moved. `tomllib` is itself 3.11+, which this
    script must not require, so the value is regexed deliberately rather than parsed.
    Returns None if pyproject.toml is unreadable, which the caller reports rather than
    silently treating as "no floor".
    """
    pp = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pp.read_text()
    except OSError:
        return None
    m = re.search(r'^\s*requires-python\s*=\s*["\'][^0-9]*([0-9]+(?:\.[0-9]+)*)',
                  text, re.MULTILINE)
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def warn(notes: list[str], line: str) -> None:
    """An advisory: this configuration is off the tested path but not known to be broken.

    Deliberately not a failure. The stages below are the actual evidence — a hard version
    gate would short-circuit the empirical answer this script exists to give. Warnings are
    re-printed in the summary so a pass on an untested configuration never reads as a pass
    on a supported one.
    """
    notes.append(f"WARNING: {line}")
    WARNINGS.append(line)
    print(f"        !! {line}")


# --------------------------------------------------------------------------- 1. environment

def s1_environment(notes) -> str:
    say(notes, f"python      {platform.python_version()}  ({sys.executable})")
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    say(notes, f"conda env   {conda or '(none — are you inside torch_env?)'}")
    say(notes, f"host        {platform.node()}  {platform.platform()}")
    say(notes, f"torch       {torch.__version__}  (built against CUDA {torch.version.cuda})")

    # The reference environment is torch 2.5.1 / PyG 2.5.2. Older majors are not known to
    # be broken -- the stages below are the test -- but they are far off the tested path,
    # so say so rather than let a pass imply support.
    try:
        torch_major = int(torch.__version__.split(".")[0])
    except (ValueError, IndexError):
        torch_major = -1
    if 0 <= torch_major < 2:
        warn(notes,
             f"torch {torch.__version__} is a 1.x release; this code is developed against "
             f"2.5.x. This is a much larger deviation than the Python version. If a later "
             f"stage fails, suspect this first -- and do NOT upgrade torch to fix it.")

    floor = _declared_python_floor()
    if floor is None:
        warn(notes, "could not read requires-python from pyproject.toml; the interpreter "
                    "floor was not checked")
    elif sys.version_info[:len(floor)] < floor:
        shown = ".".join(str(p) for p in floor)
        warn(notes,
             f"Python {platform.python_version()} is below pyproject.toml's "
             f"requires-python >= {shown}. No syntax in src/ needs above Python 3.9 "
             f"(measured by parsing every module against the 3.9 grammar), so this may "
             f"still work -- the stages below decide. NOTE: `pip install -e .` will be "
             f"REFUSED by pip on this interpreter regardless of --no-deps; run via "
             f"PYTHONPATH=<repo>/src instead.")

    try:
        pyg = importlib.import_module("torch_geometric")
        say(notes, f"PyG         {pyg.__version__}")
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(
            "torch_geometric is not importable. It is required: ccd.denoiser.vnn imports "
            "TAGConv from it. Install the wheel matching this exact torch build "
            f"({torch.__version__}) — a mismatched wheel is the usual cause."
        ) from exc

    # Optional companions. Modern PyG does not need these, so absence is informational.
    for mod in ("torch_scatter", "torch_sparse", "torch_cluster"):
        try:
            m = importlib.import_module(mod)
            say(notes, f"{mod:<12}{getattr(m, '__version__', 'present')}")
        except Exception:                                     # noqa: BLE001
            say(notes, f"{mod:<12}absent (optional for PyG >= 2.3)")

    say(notes, f"cuda.is_available()  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        say(notes, f"device count         {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            say(notes, f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GiB  sm_{p.major}{p.minor}")
    return "environment reported"


# --------------------------------------------------------------------------- 2. CUDA usable

def s2_cuda(device: str, notes) -> str:
    if device == "cpu":
        say(notes, "device is cpu (--allow-cpu): skipping the GPU-specific assertions")
        return "cpu mode"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. torch is installed but cannot see a GPU — "
            "usually a CPU-only build, or a driver older than the CUDA runtime torch was "
            f"built against (CUDA {torch.version.cuda}). Check `nvidia-smi`.")

    # is_available() alone is not enough: it can be True while every kernel launch fails on
    # a driver/runtime mismatch. Force a real allocation, a real kernel, and a sync.
    #
    # Resolve to an EXPLICIT index. torch 2.x accepts a bare torch.device("cuda") in the
    # cuda.* helpers; torch 1.12 raises ValueError from _get_device_index for exactly that.
    # Being explicit is correct on every version and costs nothing.
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(512, 512, device=dev)
    b = torch.randn(512, 512, device=dev)
    c = a @ b
    torch.cuda.synchronize(dev)
    if not torch.isfinite(c).all():
        raise RuntimeError("a 512x512 matmul on the GPU produced non-finite values")

    # mem_get_info is a reporting nicety and is absent on older torch. Never let this
    # script fail on an API *it* chose -- that would report an environment problem the
    # environment does not have.
    # Catch Exception deliberately, not a curated tuple: this line reports a number nobody
    # gates on, and the whole point of the stage is the kernel launch above, which has
    # already succeeded by here. An earlier version guarded (AttributeError, RuntimeError)
    # and still failed the stage on torch 1.12's ValueError -- a narrow guard on a
    # cosmetic call bought a false negative on a working GPU.
    try:
        free, total = torch.cuda.mem_get_info(dev)
        say(notes, f"matmul on {dev} ok; {free / 1024**3:.1f} GiB free of "
                   f"{total / 1024**3:.1f} GiB")
    except Exception as exc:                                  # noqa: BLE001
        say(notes, f"matmul on {dev} ok (free-memory reporting unavailable on torch "
                   f"{torch.__version__}: {type(exc).__name__})")
    say(notes, f"driver-visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    del a, b, c
    torch.cuda.empty_cache()
    return f"{device} verified by kernel launch"


# --------------------------------------------------------------------------- 3. PyG on device

def s3_pyg(device: str, dtype: torch.dtype, notes) -> str:
    from torch_geometric.nn import TAGConv

    # A tiny complete graph, matching how ccd.denoiser.vnn builds its shift operator.
    d, width, K = 8, 4, 3
    idx = torch.arange(d, device=device)
    src = idx.repeat_interleave(d)
    dst = idx.repeat(d)
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_weight = torch.randn(d * d, device=device, dtype=dtype)

    conv = TAGConv(1, width, K=K, bias=False, normalize=False).to(device=device, dtype=dtype)
    x = torch.randn(d, 1, device=device, dtype=dtype)
    out = conv(x, edge_index, edge_weight)

    if out.shape != (d, width):
        raise RuntimeError(f"TAGConv returned {tuple(out.shape)}, expected {(d, width)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("TAGConv produced non-finite output")
    if out.device.type != torch.device(device).type:
        raise RuntimeError(f"TAGConv output landed on {out.device}, expected {device}")

    # The scatter/gather underneath is where a mismatched PyG wheel actually dies.
    out.sum().backward()
    if conv.lins[0].weight.grad is None:
        raise RuntimeError("no gradient reached TAGConv weights")

    say(notes, f"TAGConv(K={K}) forward+backward on {device}, {d * d} edges, dtype {dtype}")
    return "PyG functional on device"


# --------------------------------------------------------------------------- 4. ccd imports

def s4_ccd_import(notes) -> str:
    try:
        import ccd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`import ccd` failed. The package is not installed in this environment. From "
            "the repo root run:  pip install -e . --no-deps\n"
            "The --no-deps is mandatory (pyproject.toml explains why): without it pip may "
            "replace the pinned torch in torch_env."
        ) from exc

    say(notes, f"ccd from {os.path.dirname(ccd.__file__)}")
    # Import the subpackages the later stages need, so a broken module is named here.
    for mod in ("ccd.operators", "ccd.forward", "ccd.denoiser", "ccd.reverse"):
        importlib.import_module(mod)
        say(notes, f"  {mod} ok")
    return "ccd importable"


# --------------------------------------------------------------------------- 5. forward pass

def s5_forward(cfg, notes) -> dict:
    from ccd.denoiser import ColoredEDM
    from ccd.forward import ColoredForward, ShiftedPowerRate, forward_draw
    from ccd.operators import make_synthetic_operator, spectrum_bracket

    device, dtype = cfg.device, cfg.dtype

    # The generator MUST live on the target device: make_synthetic_operator draws with
    # torch.randn(..., generator=gen, device=device), and a CPU generator paired with
    # device='cuda' raises. This has bitten the codebase before (see ccd/vp.py).
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    op = make_synthetic_operator(cfg.n, cfg.d, tau=cfg.tau, generator=gen,
                                 dtype=dtype, device=device)
    say(notes, f"operator S = C_n + {cfg.tau}I  on {op.device}, d={cfg.d}, n={cfg.n}")

    lo, hi = spectrum_bracket(op, generator=gen)
    if not (0.0 < lo < hi):
        raise RuntimeError(f"degenerate spectrum bracket: lo={lo}, hi={hi}")
    say(notes, f"spectrum bracket [{lo:.4g}, {hi:.4g}]  (power iteration, matvec-only)")

    # ShiftedPowerRate is the experiment default (a two-point-pinned clock). Plain power
    # laws are avoided repo-wide; see docs/experiment_conventions.md. Nothing here depends
    # on the choice -- it is used so the smoke path matches the real one.
    #
    # R must be derived, not hardcoded. The class's default range is
    # R = ln2/(c*F_min*FLOOR) ~= 150, but two-point pinning can only SHRINK a clock's
    # range, so R is capped by the unpinned span (lam_max/lam_min)^|a| -- which this
    # synthetic operator sets to roughly (1+tau)/tau, far below 150. Asking for more
    # raises. Taking the geometric midpoint of the feasible interval keeps the stage valid
    # for any --dim / --samples / --dtype the caller passes.
    span = (hi / lo) ** abs(cfg.a)
    if span <= 1.0:
        raise RuntimeError(
            f"spectrum span {span:.4g} leaves no room for a colored clock; the operator is "
            f"degenerate (lo={lo:.4g}, hi={hi:.4g})")
    R = span ** 0.5
    phi = ShiftedPowerRate(a=cfg.a, lam_min=lo, lam_max=hi, R=R)
    fwd = ColoredForward(op, lo, hi, phi=phi, K=cfg.K)
    say(notes, f"clock ShiftedPowerRate(a={cfg.a}, R={R:.4g} of max {span:.4g}), "
               f"Chebyshev K={cfg.K}")

    x0 = torch.randn(cfg.d, cfg.batch, generator=gen, dtype=dtype, device=device)
    x_F = forward_draw(fwd, x0, torch.randn_like(x0), cfg.F_max)
    if not torch.isfinite(x_F).all():
        raise RuntimeError(f"forward_draw at F={cfg.F_max} produced non-finite values")
    say(notes, f"forward_draw at F={cfg.F_max} -> {tuple(x_F.shape)} on {x_F.device}")

    # c_in / c_out / c_skip go through the Chebyshev + rational solvers; exercise them.
    edm = ColoredEDM(fwd)
    u = edm.c_in(x_F, cfg.F_max)
    if not torch.isfinite(u).all():
        raise RuntimeError("ColoredEDM.c_in produced non-finite values (solver did not converge)")
    say(notes, "ColoredEDM preconditioners c_in/c_out/c_skip applied")

    return {"op": op, "fwd": fwd, "edm": edm, "gen": gen, "x0": x0, "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- 6. training

def s6_train(cfg, ctx, notes) -> str:
    from ccd.denoiser import ColoredDenoiser, VNNBody, edm_loss, sample_F_loguniform

    fwd, edm, gen, x0 = ctx["fwd"], ctx["edm"], ctx["gen"], ctx["x0"]

    # VNNBody moves itself to fwd.op.device/dtype in __init__, so it follows the operator.
    body = VNNBody(fwd, width=cfg.width, n_layers=cfg.layers, K_gf=cfg.K_gf)
    denoiser = ColoredDenoiser(edm, body)

    params = [p for p in body.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    if n_par == 0:
        raise RuntimeError("VNNBody exposed no trainable parameters")
    bad = [n for n, p in body.named_parameters() if p.device.type != torch.device(cfg.device).type]
    if bad:
        raise RuntimeError(f"parameters did not follow the operator onto {cfg.device}: {bad}")
    say(notes, f"VNNBody(width={cfg.width}, n_layers={cfg.layers}, K_gf={cfg.K_gf}) "
               f"-> {n_par} params on {cfg.device}")

    opt = torch.optim.Adam(params, lr=cfg.lr)
    before = [p.detach().clone() for p in params]

    # sample_F_loguniform draws through torch.rand on the CPU, so it needs a CPU generator
    # even when everything else is on the GPU.
    cpu_gen = torch.Generator().manual_seed(cfg.seed + 1)

    losses = []
    for step in range(cfg.steps):
        F = sample_F_loguniform(cfg.F_min, cfg.F_max, generator=cpu_gen)
        loss = edm_loss(denoiser, x0, F, generator=gen)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss went non-finite at step {step} (F={F:.4g})")
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Older torch returned a float here rather than a Tensor; as_tensor covers both.
        gnorm = torch.as_tensor(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"gradient norm went non-finite at step {step} (F={F:.4g})")
        if float(gnorm) == 0.0:
            raise RuntimeError(
                f"gradient norm is exactly zero at step {step}: no gradient reached the "
                "VNN body, so nothing was actually trained")
        opt.step()
        losses.append(float(loss))
        say(notes, f"  step {step}  F={F:<10.4g} loss={float(loss):<12.6g} |grad|={float(gnorm):.4g}")

    moved = sum(1 for b, p in zip(before, params) if not torch.equal(b, p.detach()))
    if moved == 0:
        raise RuntimeError(
            "no parameter changed after Adam steps — the optimizer ran but the training "
            "loop is inert")
    say(notes, f"{moved}/{len(params)} parameter tensors changed after {cfg.steps} Adam steps")

    if cfg.device != "cpu":
        # Reporting only — never fail a passing training step over a memory statistic.
        try:
            peak = torch.cuda.max_memory_allocated(cfg.device) / 1024**2
            say(notes, f"peak GPU memory {peak:.1f} MiB")
        except Exception as exc:                              # noqa: BLE001
            say(notes, f"peak GPU memory unavailable ({type(exc).__name__})")

    ctx["denoiser"] = denoiser
    return f"{cfg.steps} training steps, loss {losses[0]:.4g} -> {losses[-1]:.4g}"


# --------------------------------------------------------------------------- 7. reverse

def s7_reverse(cfg, ctx, notes) -> str:
    from ccd.reverse import LearnedReversePFODE, reverse_f_grid

    denoiser = ctx["denoiser"]

    # Geometric spacing: arc-length grids are unstable for a < 0 on the PF-ODE.
    grid = reverse_f_grid(cfg.F_max, cfg.F_min, cfg.rev_steps, spacing="geometric")
    grid = [float(f) for f in grid]

    rev = LearnedReversePFODE(denoiser)
    x = torch.randn(cfg.d, cfg.batch, dtype=cfg.dtype, device=cfg.device)

    with torch.no_grad():
        for F_a, F_b in zip(grid[:-1], grid[1:]):
            x = rev.step(x, F_a, F_b, scheme="heun")
            if not torch.isfinite(x).all():
                raise RuntimeError(f"reverse PF-ODE diverged on step {F_a:.4g} -> {F_b:.4g}")

    say(notes, f"{len(grid) - 1} heun steps on a geometric grid "
               f"F={grid[0]:.4g} -> {grid[-1]:.4g}")
    say(notes, f"sample {tuple(x.shape)} on {x.device}, "
               f"std={float(x.std()):.4g} (finite, not checked against any reference)")
    return f"{len(grid) - 1}-step reverse sample drawn"


# --------------------------------------------------------------------------- summary

def _summary() -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for i, r in enumerate(RESULTS, 1):
        mark = "PASS" if r.ok else "FAIL"
        print(f"  {mark}  {i}. {r.name:<46} {r.seconds:6.2f}s  {r.detail}")
    n_ok = sum(r.ok for r in RESULTS)
    print("-" * 78)
    print(f"  {n_ok}/{len(RESULTS)} stages passed")
    if WARNINGS:
        print(f"\n  {len(WARNINGS)} ADVISORY/ADVISORIES — this environment is off the "
              f"tested path:")
        for w in WARNINGS:
            print(f"    !! {w}")


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="GPU environment smoke test for colored covariance diffusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:1, or cpu")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="do not fail when no GPU is present (defeats the purpose; for "
                         "checking the non-GPU half of the environment only)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--steps", type=int, default=5, help="training steps")
    ap.add_argument("--rev-steps", type=int, default=4, help="reverse PF-ODE steps")
    ap.add_argument("-d", "--dim", dest="d", type=int, default=32, help="ambient dimension")
    ap.add_argument("-n", "--samples", dest="n", type=int, default=256,
                    help="samples backing the empirical covariance")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0,
                    help="for a self-consistent run only; NOT for cross-machine agreement")
    args = ap.parse_args()

    torch.set_num_threads(2)

    # Seed the GLOBAL RNG too, not just the two explicit generators below. VNNBody's
    # parameter init draws from the global stream, so without this the loss trajectory
    # differed run to run on ONE machine -- which would have made any cross-machine loss
    # comparison meaningless while looking like a version effect. This buys same-machine,
    # same-version repeatability only; it is NOT a cross-machine reproducibility claim,
    # and nothing in this script is compared against a reference.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            print("ERROR: --device is a GPU but torch.cuda.is_available() is False.\n"
                  "       This is the failure the smoke test exists to catch. Re-run with\n"
                  "       --allow-cpu only if you deliberately want the CPU-only checks.",
                  file=sys.stderr)
            return 1
        print("WARNING: no GPU visible; falling back to CPU because --allow-cpu was given.\n"
              "         A pass here says NOTHING about GPU readiness.")
        device = "cpu"

    cfg = argparse.Namespace(
        device=device,
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        d=args.d, n=args.n, batch=args.batch, seed=args.seed,
        tau=0.05, a=-1.0, K=40, F_min=0.02, F_max=2.0,
        width=16, layers=2, K_gf=3, lr=1e-3, grad_clip=500.0,
        steps=args.steps, rev_steps=args.rev_steps,
    )

    print("=" * 78)
    print("CCD GPU SMOKE TEST")
    print("=" * 78)
    print(f"device={cfg.device}  dtype={args.dtype}  d={cfg.d}  n={cfg.n}  "
          f"batch={cfg.batch}  steps={cfg.steps}")
    print("This checks that the environment RUNS the code. It is not a benchmark and")
    print("makes no reproducibility claim — no output is compared against a reference.")

    stage("interpreter and package versions", s1_environment)
    stage("CUDA is genuinely usable", lambda nt: s2_cuda(cfg.device, nt))
    stage("PyTorch Geometric works on the device",
          lambda nt: s3_pyg(cfg.device, cfg.dtype, nt))
    stage("the ccd package imports", s4_ccd_import)

    ctx: dict = {}

    def _fwd(nt):
        ctx.update(s5_forward(cfg, nt))
        return "forward marginals ok"

    stage("colored forward marginals on device", _fwd)
    stage("VNN denoiser training step", lambda nt: s6_train(cfg, ctx, nt))
    stage("learned reverse PF-ODE", lambda nt: s7_reverse(cfg, ctx, nt), fatal=False)

    _summary()
    ok = all(r.ok for r in RESULTS)
    if not ok:
        print("\nEnvironment is NOT ready. Fix the failing stage above before running "
              "experiments.")
    elif cfg.device == "cpu":
        # Scope the verdict to what actually ran: a CPU pass exercised torch, PyG and the
        # ccd numerics, but nothing about this machine's GPU.
        print("\nCPU checks passed: torch, PyTorch Geometric and the ccd numerics all run "
              "here.\nThis says NOTHING about GPU readiness — no GPU was exercised. Re-run "
              "without\n--allow-cpu on a machine with a visible GPU before launching work.")
    elif WARNINGS:
        print(f"\nAll stages passed on {cfg.device}, but with {len(WARNINGS)} advisory/ies "
              f"above:\nthe code RUNS here, on a configuration that is not the tested one. "
              f"That is a real\nresult, not a rubber stamp — if numerical work on this box "
              f"later disagrees with\nanother machine, start with the advisories.")
    else:
        print(f"\nEnvironment looks good on {cfg.device} — you can launch real work on "
              f"this machine.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
