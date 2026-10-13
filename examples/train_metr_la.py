"""Train an example colored covariance diffusion model on METR-LA traffic speeds and sample from it.

The panel is 153 Los Angeles road sensors (the well-observed subset of METR-LA's 207) with one
graph signal per five-minute frame, cut into five blocked folds whose test set is one held-out week.
Build the folds once; the script downloads the public, md5-pinned sources (Zenodo record 5724362,
CC-BY-4.0, and the DCRNN sensor graph) and takes about a minute:

    python scripts/gen_metr.py                       # writes data/metr_la/fold0 ... fold4

Then train on one fold, sample and score:

    python examples/train_metr_la.py --fold 0 --out outputs/examples/metr_la_fold0

The defaults are the paper's METR-LA covariance arm except the training budget: the clock
`ShiftedPowerRate` with nominal `a = -3.3` (realised `ā ≈ -1.30`), `R = ln2/(3·0.02·0.077) ≈ 150.03`,
`Φ = 0.077`, `F ∈ [0.02, 102.0]`, and the FiLM feature and loss weight averaged over the operator's
bands rather than a uniform λ-grid (on this spectrum, κ ≈ 1,885, a uniform grid puts nearly all of
its points where no eigenvalue lies).

`--family vp` trains the paper's standard-diffusion baseline ("VP, cov." in its table) on the same
fold, budget and random draws; see `example_common.py`.

**Held-out.** The operator `S = Ĉ_train + τI` and the model see only the fold's training frames.
Each cloud is scored against both the training frames and the unseen test week; against the test
week, the exact-Gaussian floor also carries the train-to-test covariance shift.

Frames are z-scored per sensor on the fold's training statistics, and the saved samples are in
those units; multiply by the fold's `sd.npy` and add `mu.npy` for miles per hour.

**Cost.** Measured at the defaults on 2026-10-12: 84 s on one RTX 3090 that another training job was
also using (20 ms per update). The paper trained for 80,000 updates (`--updates 80000`).

Outputs, all in `--out`: `run.json`, `model.pt` and `samples_<sampler>_N<steps>.npy`, as in
`train_fmri.py`. Nothing written is a paper measurement (`is_study_result: false`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import example_common as ex  # noqa: E402

DEFAULT_DATA = ex.ROOT / "data" / "metr_la"

#: The paper's METR-LA covariance arm (`scripts/metr_clocks.py`, `scripts/metr_regime.py`): the range
#: bound `R = ln2/(c·F_min·Φ)` saturated at c = 3, and `F_max` from its prior-term budget 7.8543.
DEFAULTS = replace(
    ex.Recipe(), a=-3.3, R=math.log(2.0) / (3.0 * 0.02 * 0.077), F_max=7.8543 / 0.077,
    level_average="bands",
)

DEVIATIONS = [
    "Default budget 2,000 updates; the paper's METR-LA runs trained 80,000. --updates 80000 "
    "restores it.",
    "One fold (--fold, default 0) of the five the paper trains.",
]

#: How the colored arm alone differs; a `--family vp` run does not carry these.
COLORED_DEVIATIONS = [
    "Nominal a is fixed at -3.3. The paper solved it per fold so that the realised abar is -1.30 "
    "(fold 0: a = -3.303); run.json records the abar this run realised.",
    "The paper built its covariance arm from scripts/metr_marginals.py's dense misspecified "
    "classes, so that one numerical method served its covariance and graph arms alike. These "
    "reduce to the standard ColoredForward and ColoredEDM used here when Sigma_0 = S - tau I, which "
    "holds on the covariance arm; that reduction is not re-checked numerically here.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(data: Path, fold: int) -> ex.Panel:
    """One fold: its training frames, its held-out test week, and the bundle's `τ`."""
    d = data / f"fold{fold}"
    files = {"train": d / "x_train.npy", "test": d / "x_test.npy"}
    config = d / "fold_config.json"
    missing = [str(p) for p in list(files.values()) + [config] if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}. Build the folds first: python scripts/gen_metr.py")
    x = {k: np.load(p).astype(np.float64) for k, p in files.items()}
    tau = float(json.loads(config.read_text())["tau"])
    return ex.Panel(
        name=f"metr_la_fold{fold}", x_train=x["train"], references=x, tau=tau,
        source={"dir": str(d), "fold": fold, "frames": "x_train (the model); x_test (held out)",
                "sha256": {p.name: _sha256(p) for p in list(files.values()) + [config]}},
    )


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the gen_metr.py output root")
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ex.add_arguments(ap, DEFAULTS,
                     n_samples_help="samples per cloud (default: the fold's test-week size)")
    args = ap.parse_args(argv)
    recipe = ex.recipe_from_args(args, DEFAULTS)            # refuses bad flags before any data loads
    panel = load_panel(args.data, args.fold)
    return ex.run(panel, recipe, out=args.out,
                  device=ex.resolve_device(args.device), samplers=args.samplers, steps=args.steps,
                  n_samples=args.n_samples or panel.references["test"].shape[0],
                  floor_clouds=args.floor_clouds,
                  deviations=DEVIATIONS + (COLORED_DEVIATIONS if recipe.family == "colored" else []),
                  overwrite=args.overwrite, log_every=args.log_every)


if __name__ == "__main__":
    main()
