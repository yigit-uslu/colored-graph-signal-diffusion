#!/usr/bin/env python
"""Build the blocked-fold bundles for the METR-LA generalization protocol (Package B, plan §5.B).

Fetches and digest-verifies the METR-LA release, cuts it to the registered well-observed sensor
subset, and writes ``data/metr_la/fold{k}/`` for k = 0..N_FOLDS-1:

    x_train.npy      (n_train, 153)  standardised on TRAIN statistics
    x_test.npy       (n_test, 153)   standardised on the SAME train statistics
    train_index.npy  (n_train,)      int64 row indices into the kept-frame matrix
    test_index.npy   (n_test,)       int64
    mu.npy, sd.npy   (153,)          the train standardisation, recorded so it is invertible
    C_train.npy      (153, 153)      unbiased covariance of x_train, centred on the TRAIN mean
    S_ccd.npy        (153, 153)      C_train + tau*I  -- the covariance arm's operator
    W.npy            (153, 153)      symmetrised road adjacency, no self-loops
    L.npy            (153, 153)      combinatorial Laplacian  D - W
    S_lap.npy        (153, 153)      normalised Laplacian  I - D^-1/2 W D^-1/2  (the Laplacian GSO)
    fold_config.json                 every free parameter, source digests, and measured facts

⭐ THE GRAPH IS EXTERNAL, NOT DATA-DERIVED. Unlike the fMRI panel — whose descriptor graph is a
correlation graph and therefore had to be rebuilt per fold from train frames only — the road network
comes from sensor-to-sensor driving distances. It is identical in every fold and cannot leak test
signal, because it never sees a signal. That is the whole reason this dataset can answer Q1.

Registered design decisions, all owner-ratified 2026-08-29 (see docs/metr_state.md §3):

* **Sensor cut at <9% missing → d = 153.** The plan's literal rule ("drop timestamps with any
  zero-speed sentinel") keeps 5,206 of 34,272 frames, n/d = 25, destroying the abundant-samples
  property that motivated a traffic dataset. Measured frontier in scratchpad/metr_frontier_probe.py.
  ⚠️ The cut is computed over the WHOLE record, not per fold: the sensor set must be identical
  across folds or the graph changes between them. It selects on a data-QUALITY statistic (sentinel
  rate), never on the target statistic (speed covariance). Declared, not hidden.
* **max(A, Aᵀ) symmetrisation.** The DCRNN adjacency is directed (max|A-Aᵀ| = 0.996). GAD's own
  choice is unrecoverable — its paper documents no preprocessing and its shipped code has no script
  that produces the L_sym.npy it loads (baselines/gad/PROVENANCE.md, defect 5).
* **Blocked folds on the WALL CLOCK.** Test blocks are 7 consecutive days with a 1-day purge each
  side. After the completeness cut the retained rows are irregularly spaced, so blocks are selected
  by timestamp, never by row index — a row-index block would silently span different durations.

Nothing here edits a sealed file. The bundle is a function of (FOLD_SEED, N_FOLDS, the registered
constants below, the digest-pinned raw inputs) and nothing else, so it re-derives.

Run:  conda run -n rrm --no-capture-output python scripts/gen_metr.py
      conda run -n rrm --no-capture-output python scripts/gen_metr.py --verify   # digests only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from fmri_provenance import source_fingerprint  # noqa: E402

__all__ = [
    "FOLD_ARRAYS", "FOLD_SEED", "N_FOLDS", "SENSOR_MAX_MISSING", "TAU", "SOURCES",
    "MetrError", "block_folds", "build_fold", "fetch_sources", "kept_sensors",
    "load_panel", "road_graph", "write_fold",
]

# --------------------------------------------------------------------------- registered constants
N_FOLDS: int = 5
FOLD_SEED: int = 20260829
TEST_BLOCK_DAYS: int = 7
PURGE_DAYS: int = 1
#: Sensors whose sentinel rate is >= this are dropped. 0.09 -> d = 153, n = 21,286, n/d = 139.
SENSOR_MAX_MISSING: float = 0.09
#: Ridge on the covariance operator. Matches the fMRI panel's; the data is standardised so
#: diag(C) ~ 1 and tau is directly comparable across the two panels.
TAU: float = 0.01
#: DCRNN's thresholded-Gaussian-kernel parameter. Reproduces the shipped adj_mx.pkl to float32.
ADJ_THRESHOLD: float = 0.1
SENTINEL: float = 0.0

DATA_ROOT = ROOT / "data" / "metr_la"
RAW_ROOT = DATA_ROOT / "raw"

#: Raw inputs, pinned to the digests their PUBLISHERS published. We record our own sha256 of each
#: file in the config; we verify against the publisher's md5, because that is the external
#: authority. A self-computed digest pinned against itself would check nothing.
_DCRNN = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
SOURCES: dict[str, dict[str, str]] = {
    "METR-LA.csv": {
        "url": "https://zenodo.org/api/records/5724362/files/METR-LA.csv/content",
        "md5": "7c463fcbd28ecc6c9cd93e1ca120aaaa",
        "authority": "Zenodo record 5724362, doi:10.5281/zenodo.5724362, CC-BY-4.0",
    },
    "adj_mx.pkl": {
        "url": f"{_DCRNN}/adj_mx.pkl",
        "md5": "55996dfce3e39cc981e6e42e3f427da8",
        "authority": "github.com/liyaguang/DCRNN (MIT); md5 identical to Zenodo 5724362's "
                     "adj_mx_METR-LA.pkl, which is what authenticates the Zenodo mirror",
    },
    "graph_sensor_ids.txt": {
        "url": f"{_DCRNN}/graph_sensor_ids.txt",
        "md5": "5cbbcadfc83f0c1141fee5d853eb7004",
        "authority": "github.com/liyaguang/DCRNN (MIT)",
    },
    "distances_la_2012.csv": {
        "url": f"{_DCRNN}/distances_la_2012.csv",
        "md5": "aca7e68cde930a10083d6ab2c389a2c1",
        "authority": "github.com/liyaguang/DCRNN (MIT)",
    },
}

FOLD_ARRAYS: tuple[str, ...] = (
    "x_train.npy", "x_test.npy", "train_index.npy", "test_index.npy",
    "mu.npy", "sd.npy", "C_train.npy", "S_ccd.npy", "W.npy", "L.npy", "S_lap.npy",
    # Binding to the source and to the wall clock. Without these a loader cannot verify the
    # purge -- the one property that makes a blocked split a blocked split -- nor check that the
    # frames are the panel's rows at the recorded positions. `train_index`/`test_index` index
    # into these kept-frame arrays, NOT into the original 34,272-row panel.
    "kept_frames.npy", "kept_time.npy", "sensor_index.npy",
)
FOLD_CONFIG = "fold_config.json"
METR_ENTRY: tuple[str, ...] = ("gen_metr",)


class MetrError(RuntimeError):
    pass


# --------------------------------------------------------------------------- sources
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sources(root: Path = RAW_ROOT, *, allow_download: bool = True) -> dict[str, str]:
    """Ensure every raw input is present and matches its publisher's md5. Returns our sha256s.

    Idempotent: a file already matching its pinned md5 is never re-fetched. A file present but
    mismatched is an error, never a silent re-download — a changed byte upstream is a fact the
    operator must see.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name, spec in SOURCES.items():
        path = root / name
        if not path.exists():
            if not allow_download:
                raise MetrError(f"{path} missing and downloads disabled")
            print(f"  fetching {name} ...", flush=True)
            urllib.request.urlretrieve(spec["url"], path)  # noqa: S310 (pinned https, md5-checked)
        got = _md5(path)
        if got != spec["md5"]:
            raise MetrError(
                f"{name}: md5 {got} != pinned {spec['md5']} ({spec['authority']}). "
                "Refusing to build on an unverified input."
            )
        out[name] = _sha256(path)
    return out


# --------------------------------------------------------------------------- panel
def load_panel(root: Path = RAW_ROOT) -> tuple[pd.DataFrame, list[str]]:
    """The full 34,272 x 207 speed panel and the authoritative sensor-id order."""
    ids = (Path(root) / "graph_sensor_ids.txt").read_text().strip().split(",")
    df = pd.read_csv(Path(root) / "METR-LA.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    if list(df.columns) != ids:
        raise MetrError("panel columns do not match graph_sensor_ids.txt in order; the adjacency "
                        "is indexed by that order, so a mismatch would silently mis-wire the graph")
    return df, ids


def kept_sensors(df: pd.DataFrame, *, max_missing: float = SENSOR_MAX_MISSING) -> np.ndarray:
    """Boolean mask of sensors whose sentinel rate is below the registered threshold."""
    rate = (df.to_numpy(np.float64) == SENTINEL).mean(0)
    return rate < float(max_missing)


def complete_frames(df: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    """Row indices with no sentinel among the KEPT sensors."""
    v = df.to_numpy(np.float64)[:, keep]
    return np.flatnonzero((v == SENTINEL).sum(1) == 0).astype(np.int64)


# --------------------------------------------------------------------------- graph
def _distance_matrix(root: Path, ids: list[str]) -> np.ndarray:
    """The (d, d) road-distance matrix, inf where no route is listed."""
    id2ind = {int(x): i for i, x in enumerate(ids)}
    dmx = np.full((len(ids), len(ids)), np.inf, dtype=np.float64)
    dist = pd.read_csv(Path(root) / "distances_la_2012.csv", dtype={"from": int, "to": int})
    for a, b, c in dist.itertuples(index=False, name=None):
        if a in id2ind and b in id2ind:
            dmx[id2ind[a], id2ind[b]] = c
    return dmx


def _kernel_adjacency(dmx: np.ndarray) -> tuple[np.ndarray, float]:
    """DCRNN's thresholded Gaussian kernel. Returns the DIRECTED adjacency and sigma."""
    sigma = float(dmx[~np.isinf(dmx)].flatten().std())
    A = np.exp(-np.square(dmx / sigma))
    A[A < ADJ_THRESHOLD] = 0.0
    return A, sigma


def road_graph(root: Path, keep: np.ndarray, ids: list[str]) -> dict[str, Any]:
    """The symmetrised road adjacency and its two Laplacians, restricted to kept sensors.

    Rebuilt from ``distances_la_2012.csv`` with DCRNN's documented kernel rather than read from
    ``adj_mx.pkl``, so the construction is derived rather than trusted; ``verify_adjacency``
    pins it against the shipped array.
    """
    A, sigma = _kernel_adjacency(_distance_matrix(root, ids))
    np.fill_diagonal(A, 0.0)                       # no self-loops before symmetrising
    W = np.maximum(A, A.T)[np.ix_(keep, keep)]     # registered symmetrisation
    deg = W.sum(1)
    if (deg == 0).any():
        raise MetrError(f"{int((deg == 0).sum())} isolated node(s) after the sensor cut; the "
                        "normalised Laplacian is undefined there")
    L = np.diag(deg) - W
    dinv = 1.0 / np.sqrt(deg)
    S_lap = np.eye(len(deg)) - (dinv[:, None] * W * dinv[None, :])
    return {"W.npy": W, "L.npy": _sym(L), "S_lap.npy": _sym(S_lap), "sigma": sigma}


def verify_adjacency(root: Path, ids: list[str]) -> dict[str, Any]:
    """Cross-check the rebuilt full adjacency against the shipped adj_mx.pkl.

    The shipped array is float32, so agreement is to float32 precision, NOT bitwise. Reporting it
    as exact would overstate what was measured. The dtype is read BEFORE any cast of ours -- a
    cast-then-report would have written our own dtype into the record as if it were theirs.
    """
    with open(Path(root) / "adj_mx.pkl", "rb") as fh:
        shipped_ids, _, ref_raw = pickle.load(fh, encoding="latin1")
    shipped_dtype = str(np.asarray(ref_raw).dtype)
    ref = np.asarray(ref_raw, np.float64)
    A, _ = _kernel_adjacency(_distance_matrix(root, ids))
    return {
        "ids_match": list(shipped_ids) == ids,
        "shipped_dtype": shipped_dtype,
        "max_abs_diff": float(np.abs(A - ref).max()),
        "nnz_rebuilt": int((A > 0).sum()),
        "nnz_shipped": int((ref > 0).sum()),
        "note": "shipped array is float32; agreement is to float32 precision, not bitwise",
    }


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


# --------------------------------------------------------------------------- folds
def block_folds(times: pd.DatetimeIndex, *, n_folds: int = N_FOLDS, seed: int = FOLD_SEED,
                test_days: int = TEST_BLOCK_DAYS, purge_days: int = PURGE_DAYS
                ) -> list[dict[str, np.ndarray]]:
    """Contiguous wall-clock test blocks with a purge either side.

    Returns, per fold, the TEST rows and the TRAIN rows (everything outside test ∪ purge). Blocks
    are chosen on the day grid so that no two test blocks — nor a block and another's purge —
    overlap; the seed picks which days start them.
    """
    day0 = times.normalize().min()
    day_of = (times.normalize() - day0).days.to_numpy()
    n_days = int(day_of.max()) + 1
    stride = test_days + 2 * purge_days
    n_slots = n_days // stride
    if n_slots < n_folds:
        raise MetrError(f"{n_days} days give {n_slots} disjoint {stride}-day slots; need {n_folds}")
    rng = np.random.default_rng(int(seed))
    slots = np.sort(rng.choice(n_slots, size=n_folds, replace=False))
    folds = []
    for s in slots:
        start = int(s) * stride + purge_days
        stop = start + test_days
        test = (day_of >= start) & (day_of < stop)
        purge = (day_of >= start - purge_days) & (day_of < stop + purge_days)
        folds.append({
            "test_index": np.flatnonzero(test).astype(np.int64),
            "train_index": np.flatnonzero(~purge).astype(np.int64),
            "test_day_start": int(start), "test_day_stop": int(stop),
        })
    return folds


def build_fold(x_kept: np.ndarray, fold: dict[str, np.ndarray], graph: dict[str, Any],
               *, tau: float = TAU, kept_frames: np.ndarray | None = None,
               kept_time: np.ndarray | None = None,
               sensor_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every array of one fold. Pure; no I/O.

    Standardisation is computed on TRAIN rows only and applied unchanged to test — the practitioner's
    version, which the fMRI panel could not do because it arrived pre-standardised over all frames.
    """
    tr, te = fold["train_index"], fold["test_index"]
    mu = x_kept[tr].mean(0)
    sd = x_kept[tr].std(0, ddof=1)
    if (sd <= 0).any():
        raise MetrError(f"{int((sd <= 0).sum())} sensor(s) constant over train; z-scoring undefined")
    x_train = (x_kept[tr] - mu) / sd
    x_test = (x_kept[te] - mu) / sd
    C = _sym(np.cov(x_train, rowvar=False, bias=False))
    n_kept, d = x_kept.shape
    return {
        "x_train.npy": x_train, "x_test.npy": x_test,
        "train_index.npy": tr, "test_index.npy": te,
        "mu.npy": mu, "sd.npy": sd,
        "C_train.npy": C, "S_ccd.npy": C + tau * np.eye(C.shape[0]),
        "W.npy": graph["W.npy"], "L.npy": graph["L.npy"], "S_lap.npy": graph["S_lap.npy"],
        "kept_frames.npy": (np.arange(n_kept, dtype=np.int64) if kept_frames is None
                            else np.asarray(kept_frames, np.int64)),
        "kept_time.npy": (np.zeros(n_kept, dtype=np.int64) if kept_time is None
                          else np.asarray(kept_time, np.int64)),
        "sensor_index.npy": (np.arange(d, dtype=np.int64) if sensor_index is None
                             else np.asarray(sensor_index, np.int64)),
    }


def commutator(C: np.ndarray, L: np.ndarray) -> float:
    """‖[C, L]‖_F / ‖C‖_F — how far the signal is from graph-stationary w.r.t. L.

    Zero iff C and L share an eigenbasis, i.e. iff the Laplacian arm's stationarity assumption
    holds exactly. The fMRI panel measured 0.118.
    """
    return float(np.linalg.norm(C @ L - L @ C, "fro") / np.linalg.norm(C, "fro"))


def _facts(arrays: dict[str, np.ndarray], tau: float) -> dict[str, Any]:
    C, L, W = arrays["C_train.npy"], arrays["L.npy"], arrays["W.npy"]
    ev = np.linalg.eigvalsh(C)[::-1]
    Ct = _sym(np.cov(arrays["x_test.npy"], rowvar=False, bias=False))
    lev = np.linalg.eigvalsh(L)
    return {
        "n_train": int(arrays["x_train.npy"].shape[0]),
        "n_test": int(arrays["x_test.npy"].shape[0]),
        "d": int(C.shape[0]),
        "test_block_retention": None,   # filled by the driver: kept / possible timestamps
        "cov_train_rank": int((ev > 1e-10 * ev[0]).sum()),
        "cov_train_eig_max": float(ev[0]),
        "cov_train_trace": float(ev.sum()),
        "cov_test_trace": float(np.trace(Ct)),
        "cov_test_rank": int((np.linalg.eigvalsh(Ct) > 1e-10 * ev[0]).sum()),
        "diag_dev_max": float(np.abs(np.diag(C) - 1.0).max()),
        "S_ccd_cond": float((ev[0] + tau) / tau),
        "lap_eig_max": float(lev[-1]),
        "lap_zero_eigs": int((lev < 1e-9).sum()),
        "deg_min": float(W.sum(1).min()),
        "deg_max": float(W.sum(1).max()),
        "commutator_C_L": commutator(C, L),
        "commutator_C_Slap": commutator(C, arrays["S_lap.npy"]),
    }


def write_fold(root: Path, arrays: dict[str, np.ndarray], config: dict[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in FOLD_ARRAYS:
        np.save(root / name, np.ascontiguousarray(arrays[name]))
    (root / FOLD_CONFIG).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- driver
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW_ROOT)
    ap.add_argument("--out", type=Path, default=DATA_ROOT)
    ap.add_argument("--verify", action="store_true", help="check source digests and exit")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(argv)

    print("sources:")
    sha = fetch_sources(args.raw, allow_download=not args.no_download)
    for k, v in sha.items():
        print(f"  {k:24s} sha256 {v[:16]}…  md5 pinned OK")
    if args.verify:
        return 0

    df, ids = load_panel(args.raw)
    keep = kept_sensors(df)
    rows = complete_frames(df, keep)
    x_kept = df.to_numpy(np.float64)[np.ix_(rows, keep)]
    times = df.index[rows]
    adj_check = verify_adjacency(args.raw, ids)
    graph = road_graph(args.raw, keep, ids)
    print(f"panel: {df.shape} -> kept {int(keep.sum())} sensors x {len(rows):,} complete frames "
          f"(n/d = {len(rows)/int(keep.sum()):.1f})")
    print(f"adjacency: rebuilt vs shipped max|Δ| = {adj_check['max_abs_diff']:.3e} "
          f"({adj_check['shipped_dtype']}), nnz {adj_check['nnz_rebuilt']}/{adj_check['nnz_shipped']}")

    folds = block_folds(times)
    fingerprint = source_fingerprint(METR_ENTRY)
    kept_time = (times.values.astype("datetime64[s]").astype(np.int64))      # epoch seconds, UTC-naive
    sensor_index = np.flatnonzero(keep).astype(np.int64)
    for k, fold in enumerate(folds):
        arrays = build_fold(x_kept, fold, graph, kept_frames=rows,
                            kept_time=kept_time, sensor_index=sensor_index)
        config = {
            "artifact": "metr_generalization_fold",
            "fold": k, "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
            "split_rule": "blocked_wallclock",
            "test_block_days": TEST_BLOCK_DAYS, "purge_days": PURGE_DAYS,
            "test_day_start": fold["test_day_start"], "test_day_stop": fold["test_day_stop"],
            "test_time_first": str(times[arrays["test_index.npy"]][0]),
            "test_time_last": str(times[arrays["test_index.npy"]][-1]),
            "sensor_max_missing": SENSOR_MAX_MISSING,
            "sensor_cut_scope": ("computed over the WHOLE record, not per fold: the sensor set must "
                                 "be identical across folds or the graph differs between them. "
                                 "Selects on sentinel rate (data quality), never on the speed "
                                 "covariance (the target). Declared limitation."),
            "sentinel": SENTINEL,
            "standardisation": "per-sensor z-score, mean and sd from TRAIN rows only",
            "symmetrisation": "max(A, A^T), self-loops removed before symmetrising",
            "adj_threshold": ADJ_THRESHOLD, "adj_sigma": float(graph["sigma"]),
            "adj_vs_shipped": adj_check,
            "graph_scope": ("EXTERNAL road network; identical in every fold and derived from no "
                            "signal, so it cannot leak test data"),
            "operator": "S_ccd = C_train + tau*I, C_train centred on the train mean",
            "tau": TAU,
            "grade": "built locally by scripts/gen_metr.py",
            "sources": {k2: {**v2, "sha256": sha[k2]} for k2, v2 in SOURCES.items()},
            "measured": _facts(arrays, TAU),
            "source_fingerprint": fingerprint,
        }
        possible = TEST_BLOCK_DAYS * 24 * 12          # 5-minute grid
        config["measured"]["test_block_retention"] = float(
            config["measured"]["n_test"] / possible)
        write_fold(Path(args.out) / f"fold{k}", arrays, config)
        m = config["measured"]
        print(f"fold{k}: train {m['n_train']:,} | test {m['n_test']:,} "
              f"({100*m['test_block_retention']:.0f}% of the block) | rank {m['cov_train_rank']}"
              f" | days {fold['test_day_start']}-{fold['test_day_stop']}"
              f" | comm(C,L) {m['commutator_C_L']:.3f} comm(C,S_lap) {m['commutator_C_Slap']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
