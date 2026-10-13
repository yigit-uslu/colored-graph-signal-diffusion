"""Fetch the fMRI resting-state time courses used by the Schmidt & Mateos stability paper.

The paper says only "a single subject from the HCP dataset, with data in R^{360x1190}". The actual
file is `data/RS_TCs.mat` from Michael Defferrard's PyGSP/GraphSIP tutorial repo -- a small, openly
downloadable redistribution. **No ConnectomeDB account or Data Use Terms acceptance is needed.**

Verified on download (2026-08-04): shape (360, 1190) float64, matching the paper's stated dimensions
exactly; already per-region standardised (row mean ~1e-17, row std ~0.9996), so no additional signal
normalisation is implied.

Run:  conda run -n torch_env python external/fmri_stability/fetch_data.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

URL = "https://github.com/mdeff/pygsp_tutorial_graphsip/raw/master/data/RS_TCs.mat"
SHA256 = "702c0652342610265190fb35553c6f7f1a141f2bb4a0c11f23bcc1e2aa0da968"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data", "RS_TCs.mat")
EXPECTED_SHAPE = (360, 1190)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> str:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and not force:
        got = sha256_of(DEST)
        if got == SHA256:
            print(f"already present, checksum OK: {DEST}")
            return DEST
        print(f"checksum mismatch ({got[:12]}... != {SHA256[:12]}...), re-downloading")

    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    got = sha256_of(DEST)
    if got != SHA256:
        raise RuntimeError(
            f"checksum mismatch after download:\n  expected {SHA256}\n  got      {got}\n"
            "The upstream file may have changed; verify before trusting it."
        )
    print(f"saved {DEST} ({os.path.getsize(DEST)} bytes), checksum OK")
    return DEST


def verify(path: str) -> None:
    """Load and check the array matches the paper's stated dimensions."""
    import numpy as np
    import scipy.io as sio

    sig = sio.loadmat(path)["RS_TCs"]
    assert sig.shape == EXPECTED_SHAPE, f"expected {EXPECTED_SHAPE}, got {sig.shape}"
    assert np.isfinite(sig).all(), "non-finite entries present"
    n_const = int((sig.std(axis=1) == 0).sum())
    assert n_const == 0, f"{n_const} constant region(s) -> corrcoef would emit NaN"
    print(f"verified: shape {sig.shape}, finite, no constant regions")
    print(f"  per-region mean |mu| = {np.abs(sig.mean(1)).max():.2e}   "
          f"per-region std in [{sig.std(1).min():.4f}, {sig.std(1).max():.4f}]")


if __name__ == "__main__":
    path = fetch(force="--force" in sys.argv)
    verify(path)
