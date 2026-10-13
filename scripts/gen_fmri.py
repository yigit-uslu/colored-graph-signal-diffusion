"""Build the fMRI graph-signal dataset used by the Schmidt & Mateos stability paper, plus the
covariance-native operator the CCD arm needs.

Emits a bundle under ``data/fmri`` (override with ``--tag``): the split signals, the CCD operator
``S_ccd = C_train + tau*I`` with its spectrum, the authors' Laplacian GSOs for the reproduction
path, optional subsample-perturbed operators, and a sidecar ``fmri_config.json`` recording every
free parameter.

**This is a re-derivation of a partially-specified setup.** What comes from where:

  FROM THE PAPER   360 regions x 1190 timepoints, one graph signal per timepoint; 80/20
                   trainval/test then 20% of trainval held for validation; the test-time
                   perturbation sweeps the fraction of samples used to estimate the graph.
  FROM THE AUTHORS (partial source supplied directly, not in the paper) the graph construction --
                   ``A = max(corr, 0)`` with zero diagonal, normalized Laplacian as the GSO and
                   combinatorial Laplacian alongside -- and the perturbation, which subsamples
                   TIMEPOINTS without replacement and rebuilds the graph.
  OUR CHOICE       the train/val/test split RULE (the paper states proportions only), whether the
                   covariance is estimated on train or on everything, and tau.

Provenance, verified properties and the measured defects of the authors' evaluation path are in
``external/fmri_stability/README.md``. Fetch the data first:

    conda run -n torch_env python external/fmri_stability/fetch_data.py

The data arrives ALREADY per-region standardised (row mean <= 9.2e-17, row std in [0.9996,
0.9996]), so no signal normalisation is applied here and none is implied. One consequence worth
knowing: the raw BOLD scale is absent from the file, so relative regional variance is not
recoverable from this dataset.

The two operators
-----------------
The paper's GSO and ours are near-inverses of each other on this data. Measured:
``|<top eigenvector of C_n, smoothest Laplacian eigenvector>| = 0.9951`` (against 0.0010 for the
roughest), and ``Spearman(eig(C_n), Laplacian smoothness of that eigenvector) = -0.67``. High
variance <=> smooth. So a schedule that is coarse-to-fine on ``C_n + tau*I`` reads as
fine-to-coarse on their Laplacian: **any cross-paper statement about the colouring exponent must
flip sign.** Both operators are emitted so either path stays available.

Their pipeline needs no ridge -- a GNN touches the GSO only through forward powers ``S^p``, so its
zero eigenvalue never bites. CCD uses S as a covariance (needs ``S^{1/2}``, ``S^a``), so tau is
structural, not a tuning knob: ``C_n`` here has rank ~250/360.

Which signals define C_n
------------------------
Default ``--cov-on all``: the operator is estimated from every signal, matching the authors, whose
nominal graph is likewise built from all of them and perturbed only at test time. The operator is
treated as given STRUCTURE -- the analogue of being handed the graph -- while the generative model
still trains on the train split alone.

``--cov-on train`` is the leak-free variant, and it is the honest choice for any held-out
generative claim. It is not the default because of a measured cost: on the contiguous train block
``C_n`` has rank **168/360**, against **246** for a random block of the same size and 246 for all
1190 signals. A contiguous window has far fewer effective independent samples -- the very
autocorrelation that motivates the contiguous split. Under ``train`` roughly 78 of the data's ~246
directions would reach the model as pure ridge, and that deficiency would be indistinguishable
from a colouring effect. Since this arm exists to compare colouring exponents rather than to make
a held-out generative claim, that confound costs more than the leakage does. Recorded either way
in the sidecar.

Choosing tau
------------
Because the data is per-region standardised, ``trace(C_n) = d`` exactly, so the ridge fraction is
exactly ``tau/(1+tau)`` -- no calibration needed. At the default ``tau = 0.01``: ~1% ridge, 221
modes above tau carrying 99.96% of the variance, 14.3 octaves of spectral span, and ~7 orders of
magnitude of clearance above the numerical-zero floor (lambda_250 ~ 9e-15) so the ~110 null
directions are cleanly tau-dominated rather than noise-dominated.

tau is FIXED, not swept alongside the colouring exponent: the two are not separately identified
(the range enters as tau^a). A sensitivity ablation over tau varies it deliberately and checks
that the RANKING over ``a`` is stable, not that ``a*`` is unchanged.

Run:  conda run -n torch_env python scripts/gen_fmri.py
      conda run -n torch_env python scripts/gen_fmri.py --est-frac 1 5 10 25 50 90
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC_MAT = REPO / "external" / "fmri_stability" / "data" / "RS_TCs.mat"

TAU = 0.01                 # ridge; see "Choosing tau" above
TEST_FRAC = 0.20           # paper: 20% test
VAL_FRAC = 0.20            # paper: 20% OF the remaining trainval
SPLIT = "contiguous"       # our choice; see split_signals()
COV_ON = "all"             # see "Which signals define C_n" above; "train" is the leak-free variant
SEED = 0
MIN_SUBSAMPLE = 2          # a correlation needs >= 2 samples; the authors' max(1, ...) allows 1


# --------------------------------------------------------------------------- loading
def load_signals(path: pathlib.Path = SRC_MAT) -> np.ndarray:
    """Return the (n_regions, n_timepoints) array, with the guards the pipeline depends on."""
    import scipy.io as sio

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it first:\n"
            f"  conda run -n torch_env python external/fmri_stability/fetch_data.py"
        )
    sig = np.asarray(sio.loadmat(str(path))["RS_TCs"], dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {sig.shape}")
    if not np.isfinite(sig).all():
        raise ValueError("non-finite entries in the signal matrix")
    return sig


# --------------------------------------------------------------------------- graph (authors')
def construct_graph(signals_subset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The authors' ``construct_graph``, reimplemented in numpy.

    Theirs calls ``nx.normalized_laplacian_matrix`` / ``nx.laplacian_matrix``, which are broken
    under this repo's env (networkx 2.8.8 calls ``sp.errstate``, removed in scipy 1.15). This is
    the same computation; ``tests/test_gen_fmri.py`` pins it against networkx where networkx runs.

    Args:
        signals_subset: (n_regions, n_timepoints). Correlation is taken ACROSS REGIONS, so
            timepoints are the samples -- matching ``np.corrcoef``'s rows-as-variables convention.

    Returns:
        (A, S_lap, L) -- rectified weighted adjacency, normalized Laplacian (their GSO),
        combinatorial Laplacian (their TV operator and degree source).
    """
    if signals_subset.shape[1] < MIN_SUBSAMPLE:
        raise ValueError(
            f"need >= {MIN_SUBSAMPLE} timepoints to form a correlation, "
            f"got {signals_subset.shape[1]}"
        )
    const = np.isclose(signals_subset.std(axis=1), 0.0)
    if const.any():
        raise ValueError(
            f"{int(const.sum())} region(s) are constant over the selected timepoints; "
            "np.corrcoef would emit NaN. Increase the subsample fraction."
        )

    corr = np.corrcoef(signals_subset)
    np.fill_diagonal(corr, 0.0)          # no self-loops
    A = np.where(corr > 0, corr, 0.0)    # POSITIVE PART -- not |corr|, and not thresholded

    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0   # networkx's isolated-node convention
    S_lap = np.eye(len(deg)) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    return A, _symmetrise(S_lap), _symmetrise(L)


def _symmetrise(M: np.ndarray) -> np.ndarray:
    """Kill the ~1e-18 asymmetry that the outer-product scaling leaves behind."""
    return 0.5 * (M + M.T)


# --------------------------------------------------------------------------- operator (ours)
def ccd_operator(x: np.ndarray, tau: float = TAU) -> tuple[np.ndarray, np.ndarray]:
    """CCD's covariance-native structuring matrix.

    Args:
        x: (n_samples, n_regions) -- one graph signal per row, i.e. the TRANSPOSE of the layout
            ``construct_graph`` takes. The two conventions are opposite; do not mix them.
        tau: ridge.

    Returns:
        (C, S) with ``C`` the sample covariance and ``S = C + tau*I``.
    """
    if x.shape[0] < MIN_SUBSAMPLE:
        raise ValueError(f"need >= {MIN_SUBSAMPLE} samples for a covariance, got {x.shape[0]}")
    C = _symmetrise(np.cov(x, rowvar=False, bias=False))
    return C, C + tau * np.eye(C.shape[0])


# --------------------------------------------------------------------------- splitting
def split_signals(n: int, *, rule: str = SPLIT, seed: int = SEED) -> tuple[np.ndarray, ...]:
    """Index sets for train / val / test.

    The paper fixes the PROPORTIONS (20% test; then 20% of the rest for validation) but not the
    rule. We default to CONTIGUOUS blocks because BOLD timeseries are strongly autocorrelated: a
    random split puts temporally adjacent -- hence highly correlated -- frames on both sides of
    the boundary, which inflates apparent test performance. ``random`` is available to match a
    driver that shuffles, and is recorded in the sidecar either way.
    """
    n_test = int(round(TEST_FRAC * n))
    n_val = int(round(VAL_FRAC * (n - n_test)))
    if rule == "contiguous":
        idx = np.arange(n)
    elif rule == "random":
        idx = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"unknown split rule {rule!r} (expected 'contiguous' or 'random')")
    n_train = n - n_test - n_val
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def subsample_indices(n: int, pct: float, *, seed: int = SEED) -> np.ndarray:
    """The authors' test-time perturbation: draw ``pct``% of timepoints WITHOUT replacement.

    Theirs floors at 1 sample (``max(1, ...)``), which cannot define a correlation; we floor at 2
    and raise below that rather than emit a silently degenerate operator.
    """
    k = int((pct / 100.0) * n)
    if k < MIN_SUBSAMPLE:
        raise ValueError(
            f"est-frac {pct}% of {n} timepoints is {k} sample(s); "
            f"need >= {MIN_SUBSAMPLE} to form a correlation"
        )
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="", help="suffix for the output dir: data/fmri<tag>")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--split", default=SPLIT, choices=("contiguous", "random"))
    ap.add_argument("--cov-on", default=COV_ON, choices=("train", "all"),
                    help="signals defining C_n: 'train' (ours, no leakage) or 'all' (authors')")
    ap.add_argument("--est-frac", type=float, nargs="*", default=[],
                    help="perturbation percentages; emits S_ccd_p<P>.npy and S_lap_p<P>.npy")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = REPO / "data" / f"fmri{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = load_signals()                       # (n_regions, n_timepoints)
    n_regions, n_time = sig.shape
    x_all = sig.T                              # (n_timepoints, n_regions): one signal per TR

    tr, va, te = split_signals(n_time, rule=args.split, seed=args.seed)
    x_train, x_val, x_test = x_all[tr], x_all[va], x_all[te]

    # --- CCD operator (covariance-native) ---
    cov_src = x_train if args.cov_on == "train" else x_all
    C_train, S_ccd = ccd_operator(cov_src, tau=args.tau)
    evals, evecs = np.linalg.eigh(S_ccd)
    evals, evecs = evals[::-1].copy(), evecs[:, ::-1].copy()   # descending

    # --- authors' Laplacian GSOs, from ALL signals (their nominal-graph convention) ---
    A_all, S_lap, L_comb = construct_graph(sig)
    C_all = _symmetrise(np.cov(x_all, rowvar=False, bias=False))

    for name, arr in [
        ("x_train", x_train), ("x_val", x_val), ("x_test", x_test),
        ("C_train", C_train), ("S_ccd", S_ccd), ("evals", evals), ("evecs", evecs),
        ("C_all", C_all), ("A_all", A_all), ("S_lap", S_lap), ("L", L_comb),
    ]:
        np.save(out_dir / f"{name}.npy", arr)

    # --- perturbed operators ---
    # Each operator is perturbed by subsampling the SAME pool that defined its nominal value --
    # `cov_src` for S_ccd, all signals for S_lap (the authors' nominal-graph convention). Drawing
    # the CCD subsample from `x_all` while the nominal came from `x_train` would fold a fixed
    # train-vs-all population gap into every rel_dS_ccd and put a floor under the sweep.
    perturb: dict[str, dict] = {}
    for pct in args.est_frac:
        idx = subsample_indices(n_time, pct, seed=args.seed)
        _, S_lap_p, _ = construct_graph(sig[:, idx])
        idx_c = subsample_indices(cov_src.shape[0], pct, seed=args.seed)
        _, S_ccd_p = ccd_operator(cov_src[idx_c], tau=args.tau)
        key = f"{pct:g}"
        np.save(out_dir / f"S_lap_p{key}.npy", S_lap_p)
        np.save(out_dir / f"S_ccd_p{key}.npy", S_ccd_p)
        perturb[key] = {
            "n_timepoints": int(len(idx)),
            "n_samples_ccd": int(len(idx_c)),
            "rel_dS_lap": float(np.linalg.norm(S_lap_p - S_lap, 2) / np.linalg.norm(S_lap, 2)),
            "rel_dS_ccd": float(np.linalg.norm(S_ccd_p - S_ccd, 2) / np.linalg.norm(S_ccd, 2)),
        }

    deg = A_all.sum(axis=1)
    ev_lap = np.linalg.eigvalsh(S_lap)
    ev_cov = np.linalg.eigvalsh(C_train)[::-1]
    rank = int((ev_cov > 1e-10 * ev_cov[0]).sum())
    n_above = int((ev_cov > args.tau).sum())

    cfg = {
        "provenance": {
            "purpose": "fMRI graph-signal arm: covariance-native CCD operator + the authors' Laplacian GSO",
            "status": "RE-DERIVATION of a partially-specified setup",
            "paper": "arXiv:2607.07510v1 (Schmidt & Mateos, Stability of Flow Models for Graph Signals)",
            "data_source": "external/fmri_stability/data/RS_TCs.mat (mdeff/pygsp_tutorial_graphsip)",
            "authors_source": "construct_graph + perturbation snippet supplied directly by the authors",
            "notes": "external/fmri_stability/README.md",
            "generator": "scripts/gen_fmri.py",
        },
        "from_paper": {
            "n_regions": n_regions,
            "n_timepoints": n_time,
            "signal_convention": "one graph signal per timepoint",
            "test_frac": TEST_FRAC,
            "val_frac_of_trainval": VAL_FRAC,
            "perturbation": "subsample timepoints, re-estimate the graph (1%-90%, log scale)",
        },
        "from_authors": {
            "adjacency": "A = max(corr, 0), zero diagonal, NO significance threshold",
            "gso": "normalized Laplacian; combinatorial Laplacian kept for TV and degrees",
            "subsample": "np.random.choice(..., replace=False) over timepoints",
        },
        "our_choices": {
            "tau": args.tau,
            "operator": "S_ccd = C_n + tau*I (covariance-native; option (a))",
            "cov_on": args.cov_on,
            "split_rule": args.split,
            "split_rationale": "contiguous by default: BOLD is autocorrelated, so a random split leaks",
            "seed": args.seed,
            "min_subsample": MIN_SUBSAMPLE,
            "signal_normalisation": "none -- the source file is already per-region standardised",
        },
        "derived": {
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "cov_diag_min": float(np.diag(C_train).min()),
            "cov_diag_max": float(np.diag(C_train).max()),
            "cov_rank": rank,
            "cov_eig_max": float(ev_cov[0]),
            "cov_top1_var_frac": float(ev_cov[0] / ev_cov.sum()),
            "cov_top10_var_frac": float(ev_cov[:10].sum() / ev_cov.sum()),
            "ridge_fraction": float(args.tau / (1.0 + args.tau)),
            "n_modes_above_tau": n_above,
            "var_frac_above_tau": float(ev_cov[:n_above].sum() / ev_cov.sum()),
            "S_ccd_cond": float(evals[0] / evals[-1]),
            "S_ccd_span_octaves": float(np.log2(evals[0] / evals[-1])),
            "lap_density": float((A_all > 0).mean()),
            "lap_n_isolated": int((deg == 0).sum()),
            "lap_deg_min": float(deg.min()), "lap_deg_max": float(deg.max()),
            "lap_eig_max": float(ev_lap.max()),
            "perturbation": perturb,
        },
        "arrays": {
            "x_train": list(x_train.shape), "x_val": list(x_val.shape), "x_test": list(x_test.shape),
            "C_train": list(C_train.shape), "S_ccd": list(S_ccd.shape),
            "evals": list(evals.shape), "evecs": list(evecs.shape),
            "C_all": list(C_all.shape), "A_all": list(A_all.shape),
            "S_lap": list(S_lap.shape), "L": list(L_comb.shape),
            "dtype": "float64",
            "note": "signals are (n_samples, n_regions); construct_graph takes the TRANSPOSE",
        },
    }
    (out_dir / "fmri_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"wrote {out_dir}")
    print(f"  signals {x_all.shape} -> train {x_train.shape[0]} / val {x_val.shape[0]} "
          f"/ test {x_test.shape[0]}  ({args.split})")
    print(f"  C_n on {args.cov_on}: rank {rank}/{n_regions}, top-1 var frac "
          f"{cfg['derived']['cov_top1_var_frac']:.3f}")
    print(f"  S_ccd = C_n + {args.tau}I: ridge {cfg['derived']['ridge_fraction']:.4f}, "
          f"{n_above} modes > tau ({cfg['derived']['var_frac_above_tau']:.4f} of var), "
          f"span {cfg['derived']['S_ccd_span_octaves']:.2f} octaves")
    print(f"  S_lap: density {cfg['derived']['lap_density']:.3f}, "
          f"{cfg['derived']['lap_n_isolated']} isolated, eig_max {ev_lap.max():.4f}")
    for k, v in perturb.items():
        print(f"  perturb {k}%: n={v['n_timepoints']}, rel dS_ccd {v['rel_dS_ccd']:.4f}, "
              f"rel dS_lap {v['rel_dS_lap']:.4f}")


if __name__ == "__main__":
    main()
