# Colored Graph Signal Diffusion Models

Code for *Colored Graph Signal Diffusion Models*, by Yiğit Berkay Uslu, Samar Hadou, Shirin Saeedi
Bidokhti and Alejandro Ribeiro (University of Pennsylvania).

We build diffusion models whose noise is shaped by the data's covariance, or by a graph, instead of
being the same in every direction. This lets us choose the order in which a model builds up coarse
and fine structure. A closed-form part of the model captures the data's correlations, so the neural
network only learns what is left. In our experiments, these models reach the same sample quality as
standard diffusion in far fewer sampling steps.

## Layout

| path | contents |
|---|---|
| `src/ccd/` | the package: `operators/`, `forward/`, `reverse/`, `denoiser/`, `diffusion/`, and the standard-diffusion baseline in `vp.py` |
| `examples/` | example training runs on fMRI and METR-LA |
| `scripts/` | the data builders and an environment check |
| `external/fmri_stability/` | the fMRI data fetcher |
| `tests/` | tests, mirroring the package |

Every function of the shift operator goes through one of two backends in `src/ccd/operators/`.
`SpectralBackend` applies it exactly from an eigendecomposition. `ChebyshevBackend` needs only
matrix-vector products with the operator. The examples use the dense spectral route, as the paper's
experiments do.

## Environment

Python 3.10 or 3.11, PyTorch, PyTorch Geometric, NumPy 1.x, SciPy and pandas. Verified with
PyTorch 1.12 and 2.5 and PyTorch Geometric 2.3 to 2.7 on CUDA GPUs. `pyproject.toml` declares no
dependencies, so installing never changes your PyTorch:

    pip install -e . --no-deps
    python scripts/smoke_gpu.py      # exit 0 means this machine runs the code

## Data

Built locally into `data/` from public, checksum-verified sources:

    python external/fmri_stability/fetch_data.py && python scripts/gen_fmri.py   # fMRI, 360 regions
    python scripts/gen_metr.py                                                   # METR-LA, 153 sensors

fMRI: `github.com/mdeff/pygsp_tutorial_graphsip`. METR-LA: Zenodo record 5724362 (CC-BY-4.0) and
the DCRNN sensor graph (MIT).

## Examples

    python examples/train_fmri.py --out outputs/examples/fmri
    python examples/train_metr_la.py --fold 0 --out outputs/examples/metr_la

Each trains one model (minutes on one GPU), samples it and writes its scores to `run.json`. The
defaults follow the paper's configuration with a shorter training budget; see each script's
docstring. Add `--family vp` to train the standard diffusion we compare against, on the same data,
budget and random draws.

## Tests

    python -m pytest              # about 5 minutes on a 48-core machine
    python -m pytest -m slow      # the 30 long end-to-end checks, about 30 minutes

Tests that need the built data skip until it exists.

## License

MIT. See `LICENSE`.
