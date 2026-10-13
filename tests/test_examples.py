"""The example training runs in `examples/`: wiring checks with teeth, and a toy end-to-end run.

The examples are documentation that executes, so the failure that matters is silent drift: a change
in `src/ccd` that leaves them importable and wrong, or not runnable at all. Each wiring check below
is paired with the wrong construction it must reject. Nothing here reads real data or needs a GPU.

Run: conda run -n torch_env python -m pytest tests/test_examples.py -v
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import example_common as ex  # noqa: E402
import train_fmri  # noqa: E402
import train_metr_la  # noqa: E402
from ccd.denoiser import cout_sq_bands, edm_weight  # noqa: E402
from ccd.forward import snr  # noqa: E402
from ccd.operators import CovarianceOperator  # noqa: E402
from ccd.vp import vp_alpha, vp_loss, vp_sigma  # noqa: E402

TAU = 0.01


def _frames(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Frames with a spread spectrum (variances 9 down to 0.01), so a pinned clock has room."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) @ np.diag(np.geomspace(3.0, 0.1, d)) + 0.5


def _toy_model(level_average: str):
    S = ex.build_operator(_frames(40, 6), TAU, device="cpu", dtype=torch.float64)
    recipe = ex.Recipe(a=-1.0, R=4.0, level_average=level_average, width=4)
    clock = ex.build_clock(recipe, S)
    return S, clock, ex.build_model(recipe, S, clock, device="cpu", dtype=torch.float64)


def _dense(op: CovarianceOperator) -> np.ndarray:
    return op.matvec(torch.eye(op.d, dtype=op.dtype)).numpy()


def test_the_operator_applies_the_unbiased_covariance_plus_the_ridge():
    x = _frames(40, 6)
    S = ex.build_operator(x, TAU, device="cpu", dtype=torch.float64)
    target = np.cov(x, rowvar=False) + TAU * np.eye(6)
    np.testing.assert_allclose(_dense(S.op), target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose((S.evecs * S.evals) @ S.evecs.T, target, rtol=1e-12, atol=1e-12)
    # The construction the √(n/(n−1)) scaling exists to prevent: the raw centred frames make the
    # operator apply the BIASED covariance, 1/n = 2.5 % off here.
    assert not np.allclose(_dense(CovarianceOperator(S.Xc, tau=TAU)), target, rtol=1e-6)


def test_the_band_mean_feature_averages_the_snr_over_the_operators_bands():
    S, clock, model = _toy_model("bands")
    F = 0.7
    bands = torch.as_tensor(S.evals, dtype=torch.float64)
    want = math.log(float(snr(bands, F, clock.phi, TAU).mean()) + 1e-30)
    assert float(model.body._cond_feature(F)) == pytest.approx(want, rel=1e-12)
    # Teeth: the stock body averages on a uniform λ-grid instead, and on this spread spectrum that
    # is a different number. If the override stopped being called, this is what would come back.
    _, _, grid = _toy_model("grid")
    assert float(grid.body._cond_feature(F)) != pytest.approx(want, rel=1e-3)


def test_the_band_mean_loss_weight_is_the_reciprocal_mean_posterior_variance_over_bands():
    S, _, model = _toy_model("bands")
    F = 0.7
    bands = torch.as_tensor(S.evals, dtype=torch.float64)
    want = 1.0 / float(cout_sq_bands(model.fwd, F, bands).mean())
    assert ex.loss_weight(model, F) == pytest.approx(want, rel=1e-12)
    # Teeth: the grid weight `edm_loss` falls back to is a different number on this spectrum, and
    # the grid regime must hand `None` to `edm_loss` so that it does fall back to it.
    assert edm_weight(model.fwd, F) != pytest.approx(want, rel=1e-3)
    _, _, grid = _toy_model("grid")
    assert ex.loss_weight(grid, F) is None


def test_the_analytic_head_zeroes_the_body_only_inside_its_block():
    _, _, model = _toy_model("grid")
    x = torch.randn(6, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    F = 0.7
    head = model.edm.head(x, F).detach().numpy()
    with ex.analytic_head(model):
        np.testing.assert_allclose(model.denoiser(x, F).detach().numpy(), head, rtol=1e-12,
                                   atol=1e-12)
    # Teeth: once the block exits, the body must be back. A head that leaked out of its block
    # would silently turn every later cloud, and any further training, into the untrained head.
    assert not np.allclose(model.denoiser(x, F).detach().numpy(), head, rtol=1e-6)


def test_the_clock_is_labelled_by_its_realised_range_and_slope_not_the_nominal_exponent():
    S, clock, _ = _toy_model("grid")
    lab = clock.labels
    assert lab["R"] == pytest.approx(4.0)
    phi = clock.phi(torch.as_tensor(S.evals, dtype=torch.float64)).numpy()
    assert phi.max() / phi.min() == pytest.approx(4.0, rel=1e-12)       # the range really is R
    assert phi[np.argmax(S.evals)] == pytest.approx(0.077, rel=1e-12)    # pinned at λ_max
    # ā is a variance-weighted average of the local slope a·λ/(λ+s), so it is strictly shallower
    # than the nominal a whenever the shift s is positive. Equality would mean it was not computed.
    assert lab["shift_s"] > 0 and -1.0 < lab["abar"] < 0.0
    assert lab["spearman_phi_bandvar"] == pytest.approx(-1.0)


def _toy_fmri(root: Path) -> list[str]:
    root.mkdir()
    for split, n in (("train", 30), ("val", 10), ("test", 10)):
        np.save(root / f"x_{split}.npy", _frames(n, 8, seed=n))
    (root / "fmri_config.json").write_text(json.dumps({"our_choices": {"tau": TAU}}))
    return ["--data", str(root)]


def _toy_metr(root: Path) -> list[str]:
    (root / "fold0").mkdir(parents=True)
    np.save(root / "fold0" / "x_train.npy", _frames(50, 8, seed=1))
    np.save(root / "fold0" / "x_test.npy", _frames(20, 8, seed=2))
    (root / "fold0" / "fold_config.json").write_text(json.dumps({"tau": TAU}))
    return ["--data", str(root), "--fold", "0"]


@pytest.mark.parametrize("family", ex.FAMILIES)
@pytest.mark.parametrize("script,toy", [(train_fmri, _toy_fmri), (train_metr_la, _toy_metr)],
                         ids=["fmri", "metr_la"])
def test_each_example_runs_end_to_end_on_a_toy_bundle(tmp_path, script, toy, family):
    out = tmp_path / "out"
    argv = toy(tmp_path / "data") + [
        "--out", str(out), "--device", "cpu", "--family", family, "--updates", "3",
        "--batch", "16", "--width", "4", "--steps", "3", "--n-samples", "24",
        "--floor-clouds", "3", "--log-every", "1",
    ] + (["--R", "4"] if family == "colored" else [])
    doc = script.main(argv)

    on_disk = json.loads((out / "run.json").read_text())
    assert on_disk["is_study_result"] is False and on_disk["deviations_from_paper"]
    assert on_disk["family"] == family and on_disk["recipe"]["family"] == family
    assert on_disk["training"]["updates"] == 3 and (out / "model.pt").exists()
    assert {(c["sampler"], c["steps"]) for c in on_disk["clouds"]} == {
        (s, 3) for s in ex.SAMPLERS}, "every score must name its sampler and step count"
    refs = set(doc["gaussian_floor"])
    for c in on_disk["clouds"]:
        assert set(c["cov_err"]) == set(c["gaussian_denoiser_cov_err"]) == refs
        assert all(math.isfinite(v) and v > 0
                   for v in list(c["cov_err"].values()) + list(c["gaussian_denoiser_cov_err"].values()))
        assert np.load(out / c["file"]).shape == (24, 8)

    # Each family carries its own label and its own deviations, and never the other family's.
    deviations = set(on_disk["deviations_from_paper"])
    other = "vp" if family == "colored" else "colored"
    assert set(ex.FAMILY_DEVIATIONS[family]) <= deviations
    assert deviations.isdisjoint(ex.FAMILY_DEVIATIONS[other])
    colored_only = set(getattr(script, "COLORED_DEVIATIONS", []))
    assert colored_only <= deviations if family == "colored" else deviations.isdisjoint(colored_only)
    if family == "vp":
        assert on_disk["clock"] is None and on_disk["coverage"] is None
        window = on_disk["vp_level_map"]
        assert window["r_at_F_max"] > window["r_at_F_min"] > 0
    else:
        assert on_disk["vp_level_map"] is None and on_disk["clock"]["R"] == pytest.approx(4.0)

    # A second run into the same directory refuses rather than mixing two runs' files.
    with pytest.raises(SystemExit, match="exists"):
        script.main(argv)


# --------------------------------------------------------------------------- the VP family


def _toy_vp(seed: int = 0):
    S = ex.build_operator(_frames(40, 6), TAU, device="cpu", dtype=torch.float64)
    return S, ex.build_vp_model(ex.Recipe(family="vp", width=4, seed=seed), S, device="cpu",
                                dtype=torch.float64)


def test_the_vp_arm_reads_the_colored_arms_operator_and_starts_from_its_weights():
    _, _, colored = _toy_model("grid")
    _, vp = _toy_vp()
    # The same graph shift S̃, so the two families differ in the diffusion and not in the network.
    torch.testing.assert_close(vp.body.shift_dense, colored.body.shift_dense, rtol=0, atol=0)
    c_state, v_state = colored.body.state_dict(), vp.body.state_dict()
    assert c_state.keys() == v_state.keys()
    for k in c_state:
        torch.testing.assert_close(v_state[k], c_state[k], rtol=0, atol=0)
    # Teeth: another seed draws other weights, so the equality above is the shared seed at work.
    _, other = _toy_vp(seed=1)
    assert not torch.equal(other.body.encoder.weight, colored.body.encoder.weight)


def test_the_vp_noise_level_matches_the_flat_colored_clocks_signal_to_noise():
    S, vp = _toy_vp()
    frames = _frames(40, 6)
    tr_C = float(np.trace(np.cov(frames, rowvar=False)))       # from the frames, not from S
    tr_S = tr_C + 6 * TAU
    phi0 = ex.Recipe().floor
    for F in (0.05, 1.0, 30.0):
        # The flat colored clock's population SNR, written out; VP's at level r is α²/σ² = 1/r².
        want = math.exp(-F * phi0) * tr_C / (-math.expm1(-F * phi0) * tr_S)
        assert 1.0 / vp.level_map.r_of_F(F) ** 2 == pytest.approx(want, rel=1e-12)
        # Teeth: the traces swapped, the mistake a two-trace constructor invites, is a different map.
        assert want * (tr_S / tr_C) ** 2 != pytest.approx(want, rel=1e-6)

    # Training uses that level: the example's loss at F is vp_loss at r(F), not at F.
    x0 = S.Xc[:, :16]
    F = 0.7
    got = ex._loss(vp, x0, F, torch.Generator().manual_seed(3))
    want_loss = vp_loss(vp.denoiser, x0, vp.level_map.r_of_F(F), generator=torch.Generator().manual_seed(3))
    assert float(got) == pytest.approx(float(want_loss), rel=1e-12)
    wrong = vp_loss(vp.denoiser, x0, F, generator=torch.Generator().manual_seed(3))
    assert float(wrong) != pytest.approx(float(want_loss), rel=1e-6)


def test_the_vp_gaussian_denoiser_is_the_exact_posterior_mean_only_inside_its_block():
    S, vp = _toy_vp()
    x = torch.randn(6, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    r = 0.7
    a, s = vp_alpha(r), vp_sigma(r)
    # The Gaussian posterior mean E[x0 | a x0 + s ε = x] for x0 ~ N(0, Ĉ), written out.
    want = a * S.C @ np.linalg.solve(a * a * S.C + s * s * np.eye(6), x.numpy())
    with ex.gaussian_denoiser(vp):
        np.testing.assert_allclose(vp.denoiser(x, r).detach().numpy(), want, rtol=1e-10, atol=1e-12)
    # Teeth: VP's own zero-body head α x is a different denoiser on this correlated spectrum, which
    # is why the row does not use it; and the trained network is back once the block exits.
    assert not np.allclose(a * x.numpy(), want, rtol=1e-3)
    after = vp.denoiser(x, r).detach().numpy()
    np.testing.assert_allclose(after, (a * x + s * vp.body(x, r)).detach().numpy(), rtol=1e-12)
    assert not np.allclose(after, want, rtol=1e-6)


@pytest.mark.parametrize("sampler", ex.SAMPLERS)
def test_vp_sampling_starts_from_N0I_walks_the_shared_grid_and_adds_sigma_r_min_xi(monkeypatch,
                                                                                  sampler):
    _, vp = _toy_vp()
    recipe, N, n = ex.Recipe(family="vp"), 4, 7
    seen = {}

    def fake(x_T, grid, **kw):                      # record what the sampler is handed, return 0
        seen.update(x_T=x_T.clone(), grid=list(grid))
        return torch.zeros_like(x_T)

    monkeypatch.setattr(vp.ancestral if sampler == "ancestral" else vp.pfode, "integrate", fake)
    cloud = ex.sample(vp, recipe, sampler, N, n, device="cpu")

    # The draws, regenerated from their named stream: z first, then ξ.
    g = torch.Generator().manual_seed(ex.stream_seed(recipe.seed, "sample", sampler, N))
    z = torch.randn(6, n, generator=g, dtype=torch.float64)
    xi = torch.randn(6, n, generator=g, dtype=torch.float64)
    F_grid = np.geomspace(recipe.F_max, recipe.F_min, N + 1)
    torch.testing.assert_close(seen["x_T"], z, rtol=0, atol=0)        # VP's terminal law N(0, I)
    want_grid = [vp.level_map.r_of_F(F) for F in F_grid] if sampler == "ancestral" else F_grid
    np.testing.assert_allclose(seen["grid"], want_grid, rtol=1e-12)
    r_min = vp.level_map.r_of_F(recipe.F_min)
    np.testing.assert_allclose(cloud, (vp_sigma(r_min) * xi).T.numpy(), rtol=1e-12, atol=0)


@pytest.mark.parametrize("flag", [["--a", "-2"], ["--R", "10"], ["--level-average", "bands"]],
                         ids=["a", "R", "level_average"])
def test_colored_only_flags_are_refused_with_family_vp_before_any_data_loads(tmp_path, flag):
    argv = ["--data", str(tmp_path / "absent"), "--out", str(tmp_path / "out"), *flag]
    with pytest.raises(SystemExit, match="--family vp has neither"):
        train_fmri.main(argv + ["--family", "vp"])
    # Control: the same flag on the colored family passes that check and stops at the missing data.
    with pytest.raises(SystemExit, match="missing"):
        train_fmri.main(argv + ["--family", "colored"])
