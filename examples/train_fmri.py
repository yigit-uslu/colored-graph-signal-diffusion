"""Train an example colored covariance diffusion model on the fMRI panel and sample from it.

The panel is 1,190 resting-state frames over 360 brain regions (one graph signal per frame), the
dataset of Schmidt & Mateos's flow-stability paper. Build it once, from a public 3 MB file:

    python external/fmri_stability/fetch_data.py
    python scripts/gen_fmri.py                       # writes data/fmri

Then train, sample and score (a GPU is strongly recommended; see "Cost" below):

    python examples/train_fmri.py --out outputs/examples/fmri

The defaults are the paper's fMRI configuration except the training budget: the clock
`ShiftedPowerRate` with nominal `a = -1`, `R = 120`, `Φ = 0.077`, `F ∈ [0.02, 101.6]`, the FiLM
feature and loss weight averaged on a uniform λ-grid, and the body the paper uses (194,208
parameters). That clock is the paper's arm `ab=-0.83`; weighting by the band variances of all
1,190 frames, this script computes `ā = -0.84` and records it in `run.json`. The paper's ladder also
ran nominal `a ∈ {-0.5, -2, -4}` (arms `ab=-0.50, -1.22, -1.68`) and the flat clock `a = 0`; pass
`--a` to train one of those. `--family vp` trains the paper's standard-diffusion baseline instead,
on the same frames, budget and random draws; see `example_common.py`.

**In-sample.** The operator `S = Ĉ + τI` is built on all 1,190 frames, the model trains on them,
and each cloud is scored against them, as in the paper's in-sample sweep. The paper's Table 1 holds
data out instead: five folds, each training on 952 frames and scored on the other 238. For a held-out
evaluation here, see `train_metr_la.py`, which scores against an unseen test week.

**Cost.** The forward pass of one update applies a dense 360 x 360 shift operator 15 times
(3 layers of 5 taps) to 256 x 96 signals. Measured at the defaults on 2026-10-12: 4 minutes on one
RTX 3090 that another training job was also using (56 ms per update), against about 1.1 s per update
on 8 CPU threads. `--updates 200 --steps 20` is a quick CPU check. The paper trained 20,000 epochs:
80,000 updates per fold for Table 1, and 100,000 on all frames in the in-sample sweep
(`--updates 100000`).

**Loss spikes.** Expect an occasional update whose loss is orders of magnitude above the rest. The
paper's colored runs have them too (13 of 25 clipped at least once over 100,000 updates); the
gradient clip at 500 bounds the step, and the log reports each window's median, maximum and clipped
count so a single spike is visible without hiding the trend.

Outputs, all in `--out`: `run.json` (configuration, clock labels, training log, scores),
`model.pt` (EMA and raw body weights) and `samples_<sampler>_N<steps>.npy`, one `(n, 360)` cloud per
sampler and step count. Nothing written is a paper measurement (`is_study_result: false`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import example_common as ex  # noqa: E402

DEFAULT_DATA = ex.ROOT / "data" / "fmri"
SPLITS = ("train", "val", "test")

#: The paper's fMRI configuration (`protocol.json` schedule and training blocks), at a short budget.
DEFAULTS = ex.Recipe()

DEVIATIONS = [
    "Default budget 2,000 updates. The paper trained 20,000 epochs: 100,000 updates on all 1,190 "
    "frames in its in-sample sweep (--updates 100000 restores that), and 80,000 per fold for "
    "Table 1.",
    "In-sample: the operator, the model and the scores all use all 1,190 frames. Table 1's fMRI rows "
    "hold out data: five folds, each training on 952 frames and scored against the other 238.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(data: Path) -> ex.Panel:
    """All 1,190 frames (train, val and test splits concatenated) and the bundle's `τ`."""
    files = [data / f"x_{s}.npy" for s in SPLITS]
    config = data / "fmri_config.json"
    missing = [str(p) for p in files + [config] if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}. Build the bundle first: "
                         "python external/fmri_stability/fetch_data.py && python scripts/gen_fmri.py")
    x = np.concatenate([np.load(p) for p in files]).astype(np.float64)
    tau = float(json.loads(config.read_text())["our_choices"]["tau"])
    return ex.Panel(
        name="fmri", x_train=x, references={"all_frames": x}, tau=tau,
        source={"dir": str(data), "frames": "x_train + x_val + x_test (in-sample, as the paper)",
                "sha256": {p.name: _sha256(p) for p in files + [config]}},
    )


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the gen_fmri.py bundle")
    ex.add_arguments(ap, DEFAULTS, n_samples_help="samples per cloud (default: one per frame, 1,190)")
    args = ap.parse_args(argv)
    recipe = ex.recipe_from_args(args, DEFAULTS)            # refuses bad flags before any data loads
    panel = load_panel(args.data)
    return ex.run(panel, recipe, out=args.out,
                  device=ex.resolve_device(args.device), samplers=args.samplers, steps=args.steps,
                  n_samples=args.n_samples or panel.x_train.shape[0],
                  floor_clouds=args.floor_clouds, deviations=DEVIATIONS,
                  overwrite=args.overwrite, log_every=args.log_every)


if __name__ == "__main__":
    main()
