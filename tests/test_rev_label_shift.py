"""Protocol 16 label-shift audit: synthetic logits only — no development outcome is used here."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from src.rev import label_shift as ls
from src.rev import probe_audit as pa


def _logits(n=600, c=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g, dtype=torch.float64) * 2.0


def _masks(n=600):
    fit = torch.zeros(n, dtype=torch.bool)
    fit[::2] = True
    return fit, ~fit


def test_ideal_label_shift_is_explained_by_class_only_correction() -> None:
    z = _logits()
    w = torch.linspace(-2, 2, z.shape[1], dtype=torch.float64)
    fit, ev = _masks()
    res = ls.residuals(z, z + w, fit, ev)  # exact prior reweighting
    assert res["D"] > 0.1
    assert res["R"] < 1e-6 and res["Rfit"] < 1e-6
    assert res["fit_info"]["converged"]
    assert torch.allclose(res["v"] - res["v"].mean(), w - w.mean(), atol=1e-3)


def test_concept_shift_is_not_explained() -> None:
    z = _logits()
    g = torch.Generator().manual_seed(3)
    swap = z.clone()
    rows = torch.arange(z.shape[0]) % 3 == 0  # x-dependent change: on a third of inputs, class 0 gains mass
    swap[rows, 0] += 4.0
    fit, ev = _masks()
    res = ls.residuals(z, swap, fit, ev)
    assert res["R"] > 0.2 * res["D"] and res["R"] > 0.05
    del g


def test_gauge_and_identity() -> None:
    z = _logits()
    fit, ev = _masks()
    same = ls.residuals(z, z, fit, ev)
    assert same["D"] == pytest.approx(0.0, abs=1e-12) and same["R"] == pytest.approx(0.0, abs=1e-9)
    shifted = ls.residuals(z, z + 7.0, fit, ev)  # adding a constant to all logits changes nothing
    assert shifted["D"] == pytest.approx(0.0, abs=1e-9) and shifted["R"] == pytest.approx(0.0, abs=1e-9)


def test_heldout_residual_not_below_insample_for_noise_fit() -> None:
    z = _logits(n=400, c=50)
    noise = _logits(n=400, c=50, seed=9) * 0.3
    fit, ev = _masks(400)
    res = ls.residuals(z, z + noise, fit, ev)
    assert res["R"] >= res["Rfit"] * 0.9  # optimism of the in-sample fit is measurable, not hidden


def test_no_label_inputs_anywhere() -> None:
    for fn in (ls.residuals, ls.fit_correction, ls.kl_rows, ls.analyse_seed, ls.compute_logits, ls.probe_images):
        assert not any("label" in p or p in ("y", "targets") for p in inspect.signature(fn).parameters), fn
    assert inspect.signature(pa.fit_or_eval).parameters.keys() == {"seed", "absolute_index"}


def test_manifest_loader_rejects_label_fields(tmp_path: Path) -> None:
    d = tmp_path / "results_rev/pilot/label_shift"
    d.mkdir(parents=True)
    (d / "probe_manifest_1.json").write_text(json.dumps({"version": 2, "probes": [{"absolute_train_index": 3, "split": "fit"}]}))
    assert ls.load_manifest(tmp_path, "1") == [{"absolute_train_index": 3, "split": "fit"}]
    (d / "probe_manifest_2.json").write_text(json.dumps({"version": 2, "probes": [{"absolute_train_index": 3, "split": "fit", "label": 7}]}))
    with pytest.raises(AssertionError):
        ls.load_manifest(tmp_path, "2")


def test_split_is_deterministic_and_balanced() -> None:
    a = [pa.fit_or_eval("123", i) for i in range(2000)]
    assert a == [pa.fit_or_eval("123", i) for i in range(2000)]
    assert 0.45 < a.count("fit") / len(a) < 0.55
    assert a != [pa.fit_or_eval("2026", i) for i in range(2000)]


def _per(values):
    """values[(cond, client)] = (pre_R, post_R, pre_D, post_D) -> synthetic per-round store."""
    per = {}
    for (cond, client), (pre_r, post_r, pre_d, post_d) in values.items():
        for r in ls.ROUNDS:
            post = r in ls.POST
            jitter = 0.001 * (r % 3)
            R = (post_r if post else pre_r) + jitter
            D = (post_d if post else pre_d) + jitter
            subsets = {name: {"R": R, "D": D} for name in ("eval_random_half_a", "eval_random_half_b",
                                                          "eval_predclass_0_49", "eval_predclass_50_99")}
            per[f"{cond}/{client}/{r}"] = {"R": R, "D": D, "Rfit": R * 0.95, "subsets": subsets}
    return per


def test_verdicts_on_a_constructed_supporting_case() -> None:
    vals = {}
    for cond in ls.RUNS:
        for i in range(5):
            vals[(cond, f"client_00{i}")] = (0.10, 0.10, 0.30, 0.30)
    vals[("relabel", "client_002")] = (0.10, 0.30, 0.30, 0.60)
    vals[("badnets", "client_002")] = (0.10, 0.40, 0.30, 0.70)
    vals[("drift", "client_002")] = (0.10, 0.11, 0.30, 0.60)
    v = ls.verdicts({"s": {"per": _per(vals)}})
    assert v["A_correction_reduces_drift_residual"] == "PASS"
    assert v["B_primary_drift_below_relabel"] == "PASS" and v["F_heldout"] == "CONSISTENT"
    assert v["C_badnets_sensitivity"] == "PASS" and v["E_probe_composition"] == "NO ORDER CHANGE"
    assert v["G_uncorrected_baseline"] == "CORRECTION ADDS SEPARATION" and v["H_LS_core"].startswith("NOT FALSIFIED")


def test_verdicts_on_a_constructed_failing_case() -> None:
    vals = {(cond, f"client_00{i}"): (0.10, 0.10, 0.30, 0.30) for cond in ls.RUNS for i in range(5)}
    vals[("relabel", "client_002")] = (0.10, 0.20, 0.30, 0.60)
    vals[("drift", "client_002")] = (0.10, 0.25, 0.30, 0.60)  # drift residual above relabel-only
    vals[("badnets", "client_002")] = (0.10, 0.22, 0.30, 0.70)
    v = ls.verdicts({"s": {"per": _per(vals)}})
    assert v["B_primary_drift_below_relabel"] == "FAIL" and v["H_LS_core"] == "FALSIFIED"


def test_missing_values_make_criteria_not_evaluable() -> None:
    vals = {(cond, f"client_00{i}"): (0.1, 0.1, 0.3, 0.3) for cond in ls.RUNS for i in range(5)}
    per = _per(vals)
    per["drift/client_002/25"] = None
    v = ls.verdicts({"s": {"per": per}})
    assert v["B_primary_drift_below_relabel"] == "NOT EVALUABLE" and v["H_LS_core"] == "FALSIFIED"
