"""NDSS E2 -- milder attack-severity robustness check: post-processing script.

PREPARED BUT NOT EXECUTED. Reads ONLY the (future) raw outputs of the two E2
runs under results_ndss/E2_attack_severity/{gaussian_rho050_client002_round20_seed123,
directional_c050_client002_round20_seed123}/, plus the ALREADY-EXISTING,
read-only baseline traces for the rho=1.0 Gaussian run (EXP_016,
results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123/)
and the c=0.0 directional run (EXP_018,
results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123/).
Neither baseline is rerun; both are read exactly as already saved.

Scores every trace with the existing, frozen, unmodified
src/analysis/fedtrace_detector.py module -- no formula, threshold,
persistence value, or calibration window is changed or reimplemented here.
No training, no experiment, and no GPU job is run by this script; it
performs only file reads, frozen-detector scoring (pure CPU computation over
already-saved CSV/JSON data), and file writes under
results_ndss/E2_attack_severity/. No existing raw run output (E1, the
primary-validation series, or any other prior experiment) is read for
write purposes or modified in any way.

If a requested quantity cannot be reconstructed from the saved artifacts,
the corresponding field is written as the literal string "UNAVAILABLE" --
never inferred, estimated, or fabricated. A failed or delayed detection is
reported exactly as computed; this script contains no logic that adjusts,
retries, or reinterprets the frozen detector's output based on whether the
result is favorable.

Usage (NOT executed by this task -- prepared for manual review only):
    python scripts/ndss_e2_postprocess.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis import fedtrace_detector as ft  # noqa: E402  (frozen, unmodified module)

E2_ROOT = REPO_ROOT / "results_ndss" / "E2_attack_severity"
E2_RUN_DIRS = {
    "gaussian_0.5": E2_ROOT / "gaussian_rho050_client002_round20_seed123",
    "directional_0.5": E2_ROOT / "directional_c050_client002_round20_seed123",
}
# Existing, already-completed, read-only baseline runs -- NEVER rerun, NEVER
# written to. These paths belong to the frozen primary-validation series.
BASELINE_RUN_DIRS = {
    "gaussian_1.0": REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123",
    "directional_0.0": REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123",
}
E2_WRAPPER_LOG = E2_ROOT / "e2_wrapper_log.txt"  # only if the user's own launch wrapper writes one

UNAVAILABLE = "UNAVAILABLE"

EXPECTED_T_REF = 20
EXPECTED_THRESHOLD = 3.0
EXPECTED_PERSISTENCE = 3
assert ft.DEFAULT_T_REF == EXPECTED_T_REF, (
    f"Frozen T_ref drifted: expected {EXPECTED_T_REF}, module has {ft.DEFAULT_T_REF}."
)
assert ft.DEFAULT_THRESHOLD == EXPECTED_THRESHOLD, (
    f"Frozen tau drifted: expected {EXPECTED_THRESHOLD}, module has {ft.DEFAULT_THRESHOLD}."
)
assert ft.DEFAULT_PERSISTENCE == EXPECTED_PERSISTENCE, (
    f"Frozen P drifted: expected {EXPECTED_PERSISTENCE}, module has {ft.DEFAULT_PERSISTENCE}."
)
COMPONENTS = ("m", "t_res")

ROUND_WINDOW_18_22 = list(range(18, 23))

# Static guard: E2 output paths must never coincide with any prior
# experiment tree. Checked at import time, not only at config-authoring
# time, so accidental reuse of this script against a relocated/renamed E2
# tree is still caught.
_ALL_PRIOR_ROOTS = [
    REPO_ROOT / "results" / "federated",
    REPO_ROOT / "results_ndss" / "E1_10clients",
]
for _run_dir in E2_RUN_DIRS.values():
    for _prior_root in _ALL_PRIOR_ROOTS:
        assert not str(_run_dir).startswith(str(_prior_root)), (
            f"E2 output path {_run_dir} would overlap prior experiment root {_prior_root}."
        )


class E2AnalysisError(RuntimeError):
    """Raised when a required raw artifact is missing or structurally invalid --
    fails loudly rather than silently proceeding with an incomplete analysis."""


# ---------------------------------------------------------------------------
# Raw-artifact loading (read-only)
# ---------------------------------------------------------------------------


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise E2AnalysisError(f"summary.json not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_trace(run_dir: Path) -> list[dict]:
    """Frozen-detector-compatible loader (subset of columns needed for scoring)."""
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E2AnalysisError(f"client_round_trace.csv not found: {path}")
    return ft.load_trace_rows(path)


def load_raw_csv_rows(run_dir: Path) -> list[dict]:
    """Full raw CSV read (all columns), for diagnostic-only fields the frozen
    loader does not parse (delta_attack_norm_ratio, realized_cosine_to_honest).
    Never used to feed compute_all_scores_v3 / evaluate_detector_v3."""
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E2AnalysisError(f"client_round_trace.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_wrapper_runtime(run_dir: Path, condition_label: str) -> str | int:
    if not E2_WRAPPER_LOG.exists():
        return UNAVAILABLE
    text = E2_WRAPPER_LOG.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"--- {condition_label} run end:.*?elapsed_s=(\d+) ---", text, re.IGNORECASE)
    return int(match.group(1)) if match else UNAVAILABLE


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(run_key: str, run_dir: Path, rows: list[dict], summary: dict, expected_rho: float | None, expected_c: float | None) -> dict:
    checks: dict[str, tuple[bool, str]] = {}

    client_ids = sorted({row["client_id"] for row in rows})
    checks["exactly_5_distinct_clients"] = (len(client_ids) == 5, f"found {len(client_ids)}: {client_ids}")

    round_ids = sorted({row["round_id"] for row in rows})
    checks["50_rounds_present"] = (
        round_ids == list(range(50)),
        f"found {len(round_ids)} rounds" if round_ids else "no rounds found",
    )

    full_participation = all(
        sum(1 for r in rows if r["round_id"] == rid and r.get("selected")) == 5 for rid in round_ids
    )
    checks["5_of_5_full_participation_every_round"] = (
        full_participation, "5/5 clients selected in every round" if full_participation
        else "at least one round had != 5 selected clients",
    )

    checks["seed_123"] = (summary.get("seed") == 123, f"summary seed={summary.get('seed')}")

    attack_info = summary.get("attack", {})
    checks["malicious_client_exactly_client_002"] = (
        attack_info.get("malicious_client_id") == "client_002",
        f"summary malicious_client_id={attack_info.get('malicious_client_id')}",
    )
    onset = attack_info.get("attack_start_round")
    checks["attack_onset_exactly_round_20"] = (onset == 20, f"summary attack_start_round={onset}")

    pre_onset_active = [r for r in rows if r["round_id"] < 20 and r.get("attack_active")]
    checks["no_attack_active_before_round_20"] = (
        len(pre_onset_active) == 0,
        f"{len(pre_onset_active)} pre-round-20 row(s) with attack_active=True" if pre_onset_active
        else "0 pre-round-20 active-attack rows",
    )

    attacker = attack_info.get("malicious_client_id")
    honest_active = [
        r for r in rows if r["client_id"] != attacker and r.get("attack_active")
    ]
    checks["no_honest_client_ever_attack_active"] = (
        len(honest_active) == 0,
        f"{len(honest_active)} honest-client row(s) with attack_active=True" if honest_active
        else "0 honest-client active-attack rows",
    )

    checks["T_ref_matches_frozen_default"] = (ft.DEFAULT_T_REF == EXPECTED_T_REF, f"ft.DEFAULT_T_REF={ft.DEFAULT_T_REF}")
    checks["tau_matches_frozen_default"] = (ft.DEFAULT_THRESHOLD == EXPECTED_THRESHOLD, f"ft.DEFAULT_THRESHOLD={ft.DEFAULT_THRESHOLD}")
    checks["P_matches_frozen_default"] = (ft.DEFAULT_PERSISTENCE == EXPECTED_PERSISTENCE, f"ft.DEFAULT_PERSISTENCE={ft.DEFAULT_PERSISTENCE}")

    if expected_rho is not None:
        rho = attack_info.get("attack_strength_rho")
        checks[f"gaussian_rho_exactly_{expected_rho}"] = (rho == expected_rho, f"summary attack_strength_rho={rho}")

    if expected_c is not None:
        c = attack_info.get("directional_target_cosine")
        checks[f"directional_target_cosine_exactly_{expected_c}"] = (c == expected_c, f"summary directional_target_cosine={c}")

        raw_rows = load_raw_csv_rows(run_dir)
        ratios = [
            float(r["delta_attack_norm_ratio"]) for r in raw_rows
            if r["client_id"] == attacker and int(r["round_id"]) >= 20
            and r.get("delta_attack_norm_ratio") not in (None, "")
        ]
        if ratios:
            max_rel_err = max(abs(x - 1.0) for x in ratios)
            checks["directional_norm_preservation"] = (
                max_rel_err < 1e-6, f"max |delta_attack_norm_ratio - 1| = {max_rel_err:.3e} over {len(ratios)} rows",
            )
        else:
            checks["directional_norm_preservation"] = (False, "no delta_attack_norm_ratio values found post-onset")

        cosines = [
            float(r["realized_cosine_to_honest"]) for r in raw_rows
            if r["client_id"] == attacker and int(r["round_id"]) >= 20
            and r.get("realized_cosine_to_honest") not in (None, "")
        ]
        if cosines:
            max_cos_err = max(abs(x - expected_c) for x in cosines)
            checks["realized_cosine_matches_target_c"] = (
                max_cos_err < 1e-6, f"max |realized_cosine_to_honest - {expected_c}| = {max_cos_err:.3e} over {len(cosines)} rows",
            )
        else:
            checks["realized_cosine_matches_target_c"] = (False, "no realized_cosine_to_honest values found post-onset -- UNAVAILABLE, not fabricated")

    checks["no_ground_truth_used_as_detector_input"] = (
        True,
        "static property of frozen compute_all_scores_v3/apply_persistence_rule_v3 "
        "(verified by code inspection, not a per-run runtime check)",
    )
    checks["output_path_does_not_overlap_prior_experiment"] = (
        True, f"{run_dir} is under results_ndss/E2_attack_severity/, disjoint from results/federated/ and results_ndss/E1_10clients/",
    )

    return checks


# ---------------------------------------------------------------------------
# Per-run FedTrace analysis (shared by E2 runs AND the two read-only baselines)
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path, severity_label: str, severity_value: float) -> dict:
    rows = load_trace(run_dir)
    summary = load_summary(run_dir)
    scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
    result = ft.evaluate_detector_v3(
        rows, severity_label, COMPONENTS,
        t_ref=EXPECTED_T_REF, threshold=EXPECTED_THRESHOLD, persistence=EXPECTED_PERSISTENCE,
    )

    true_attacker = result.true_malicious_client
    true_onset = result.attack_start_round
    predicted = result.predicted_client
    client_ids = sorted({r["client_id"] for r in rows})
    honest_ids = [c for c in client_ids if c != true_attacker]

    honest_fp_ids = list(result.false_positive_clients)
    max_streaks: dict[str, int] = {}
    for cid in client_ids:
        client_scored = sorted((s for s in scored if s.client_id == cid), key=lambda s: s.round_id)
        streak = best = 0
        for s in client_scored:
            combined = s.combined_score(COMPONENTS)
            anomalous = combined is not None and combined[0] > EXPECTED_THRESHOLD
            streak = streak + 1 if anomalous else 0
            best = max(best, streak)
        max_streaks[cid] = best

    dominance = {"m": 0, "t_res": 0, "tie": 0}
    if true_attacker is not None and true_onset is not None:
        mal_scored = sorted((s for s in scored if s.client_id == true_attacker), key=lambda s: s.round_id)
        for s in mal_scored:
            if s.round_id < true_onset:
                continue
            combined = s.combined_score(COMPONENTS)
            if combined is None or combined[0] <= EXPECTED_THRESHOLD:
                continue
            if s.z_m is not None and s.z_t_res is not None and abs(s.z_m - s.z_t_res) < 1e-12:
                dominance["tie"] += 1
            else:
                dominance[combined[1]] += 1
    total_anomalous_post_onset = sum(dominance.values())

    scores_18_22 = {}
    for rid in ROUND_WINDOW_18_22:
        mal_row = next((s for s in scored if s.client_id == true_attacker and s.round_id == rid), None) if true_attacker else None
        mal_combined = mal_row.combined_score(COMPONENTS) if mal_row else None
        honest_vals = []
        for cid in honest_ids:
            hr = next((s for s in scored if s.client_id == cid and s.round_id == rid), None)
            if hr is None:
                continue
            hc = hr.combined_score(COMPONENTS)
            if hc is not None:
                honest_vals.append((hc[0], cid))
        max_honest = max(honest_vals) if honest_vals else (None, None)
        scores_18_22[rid] = {
            "malicious_z_m": mal_row.z_m if mal_row else UNAVAILABLE,
            "malicious_z_t_res": mal_row.z_t_res if mal_row else UNAVAILABLE,
            "malicious_S": mal_combined[0] if mal_combined else UNAVAILABLE,
            "max_honest_S": max_honest[0] if max_honest[0] is not None else UNAVAILABLE,
            "max_honest_client": max_honest[1] if max_honest[1] is not None else UNAVAILABLE,
        }

    predicted_result = result.per_client.get(predicted) if predicted else None

    return {
        "condition": severity_label,
        "severity_value": severity_value,
        "seed": summary.get("seed"),
        "true_attacker": true_attacker,
        "true_onset": true_onset,
        "predicted_attacker": predicted,
        "estimated_onset": predicted_result.onset_estimate if predicted_result else UNAVAILABLE,
        "confirmation_round": predicted_result.confirmation_round if predicted_result else UNAVAILABLE,
        "onset_error": result.onset_localization_error if result.onset_localization_error is not None else UNAVAILABLE,
        "confirmation_delay": result.detection_delay if result.detection_delay is not None else UNAVAILABLE,
        "honest_persistent_fp_count": len(honest_fp_ids),
        "honest_persistent_fp_ids": honest_fp_ids,
        "max_streak_by_honest_client": {cid: max_streaks[cid] for cid in honest_ids},
        "magnitude_dominant_count": dominance["m"],
        "temporal_residual_dominant_count": dominance["t_res"],
        "tie_count": dominance["tie"],
        "total_anomalous_post_onset_count": total_anomalous_post_onset,
        "scores_rounds_18_22": scores_18_22,
        "final_top1_pct": summary["final_test"]["top1"] * 100 if "final_test" in summary else UNAVAILABLE,
        "total_experiment_seconds": summary.get("total_experiment_seconds", UNAVAILABLE),
    }


# ---------------------------------------------------------------------------
# Comparison table (E2 new conditions vs. existing, read-only baselines)
# ---------------------------------------------------------------------------


def dominant_signal_label(r: dict) -> str:
    m, t, tie, total = r["magnitude_dominant_count"], r["temporal_residual_dominant_count"], r["tie_count"], r["total_anomalous_post_onset_count"]
    if total == 0:
        return UNAVAILABLE
    if m == total:
        return f"z^m ({m}/{total})"
    if t == total:
        return f"z^t_res ({t}/{total})"
    return f"mixed (z^m {m}/{total}, z^t_res {t}/{total}, ties {tie})"


def write_comparison_csv(all_results: dict[str, dict], path: Path) -> None:
    header = [
        "Attack", "Severity", "Predicted attacker", "Estimated onset", "Confirmation",
        "Onset error", "Delay", "Persistent honest FP", "Dominant signal",
    ]
    rows_out = []
    for key, attack_label, severity in [
        ("gaussian_1.0", "Gaussian", "rho=1.0 (existing baseline, EXP_016)"),
        ("gaussian_0.5", "Gaussian", "rho=0.5 (E2-A, new)"),
        ("directional_0.0", "Directional", "c=0.0 (existing baseline, EXP_018)"),
        ("directional_0.5", "Directional", "c=0.5 (E2-B, new)"),
    ]:
        r = all_results[key]
        rows_out.append([
            attack_label, severity, r["predicted_attacker"], r["estimated_onset"],
            r["confirmation_round"], r["onset_error"], r["confirmation_delay"],
            r["honest_persistent_fp_count"], dominant_signal_label(r),
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows_out)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_e2_summary_json(e2_results: dict, baseline_results: dict, sanity: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"e2_new_conditions": e2_results, "existing_baselines_read_only": baseline_results, "sanity_checks": sanity},
            handle, indent=2, default=str,
        )


def write_e2_run_log(e2_results: dict, sanity: dict, path: Path) -> None:
    lines = ["NDSS E2 post-processing run log", "=" * 40, ""]
    for key, r in e2_results.items():
        lines.append(f"--- {key} ---")
        for k, v in r.items():
            lines.append(f"  {k}: {v}")
        lines.append("  sanity checks:")
        for name, (passed, detail) in sanity[key].items():
            lines.append(f"    [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_severity_comparison_figure(all_results: dict, run_dirs_for_scoring: dict[str, Path], path: Path) -> None:
    """Prepared but NOT invoked by this task. Two panels: Gaussian
    (rho=1.0 vs rho=0.5) and Directional (c=0.0 vs c=0.5), malicious
    client's S_{k,r} only (not all clients, to avoid the visual clutter the
    task explicitly asked to watch for), with severity distinguished by
    line style, plus tau/onset reference lines. No significance/robustness
    claim is drawn on the figure itself."""
    from generate_paper_figures import MALICIOUS_COLOR, THRESHOLD_COLOR, ONSET_COLOR  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    panel_groups = [
        ("Gaussian", [("gaussian_1.0", "rho=1.0 (baseline)", "-"), ("gaussian_0.5", "rho=0.5 (E2-A)", "--")]),
        ("Directional", [("directional_0.0", "c=0.0 (baseline)", "-"), ("directional_0.5", "c=0.5 (E2-B)", "--")]),
    ]
    for ax, (title, conditions) in zip(axes, panel_groups):
        for key, label, style in conditions:
            run_dir = run_dirs_for_scoring[key]
            rows = load_trace(run_dir)
            scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
            attacker = all_results[key]["true_attacker"]
            mal_scored = sorted((s for s in scored if s.client_id == attacker), key=lambda s: s.round_id)
            xs, ys = [], []
            for s in mal_scored:
                combined = s.combined_score(COMPONENTS)
                if combined is None:
                    continue
                xs.append(s.round_id)
                ys.append(combined[0])
            ax.plot(xs, ys, style, color=MALICIOUS_COLOR, lw=1.8, label=label)
        ax.axhline(EXPECTED_THRESHOLD, color=THRESHOLD_COLOR, lw=1.0, ls=":", label=r"$\tau=3$")
        ax.axvline(20, color=ONSET_COLOR, lw=1.0, ls=":", label="onset (r=20)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("communication round")
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel(r"malicious-client $S_{k,r}$")
    fig.suptitle(
        "NDSS E2: attack-severity comparison, malicious-client combined score\n"
        "(two single-point severity checks, not a sweep; no significance claim)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for key, run_dir in E2_RUN_DIRS.items():
        if not run_dir.exists():
            raise E2AnalysisError(f"Expected E2 run directory not found: {run_dir}")
    for key, run_dir in BASELINE_RUN_DIRS.items():
        if not run_dir.exists():
            raise E2AnalysisError(f"Expected existing baseline run directory not found: {run_dir}")

    e2_results = {
        "gaussian_0.5": analyze_run(E2_RUN_DIRS["gaussian_0.5"], "gaussian_0.5", 0.5),
        "directional_0.5": analyze_run(E2_RUN_DIRS["directional_0.5"], "directional_0.5", 0.5),
    }
    baseline_results = {
        "gaussian_1.0": analyze_run(BASELINE_RUN_DIRS["gaussian_1.0"], "gaussian_1.0", 1.0),
        "directional_0.0": analyze_run(BASELINE_RUN_DIRS["directional_0.0"], "directional_0.0", 0.0),
    }
    all_results = {**e2_results, **baseline_results}
    all_run_dirs = {**E2_RUN_DIRS, **BASELINE_RUN_DIRS}

    sanity = {}
    for key, run_dir in E2_RUN_DIRS.items():
        rows = load_trace(run_dir)
        summary = load_summary(run_dir)
        expected_rho = 0.5 if "gaussian" in key else None
        expected_c = 0.5 if "directional" in key else None
        sanity[key] = run_sanity_checks(key, run_dir, rows, summary, expected_rho, expected_c)

    E2_ROOT.mkdir(parents=True, exist_ok=True)
    write_e2_summary_json(e2_results, baseline_results, sanity, E2_ROOT / "e2_summary.json")
    write_e2_run_log(e2_results, sanity, E2_ROOT / "e2_run_log.txt")
    write_comparison_csv(all_results, E2_ROOT / "e2_severity_comparison.csv")
    make_severity_comparison_figure(all_results, all_run_dirs, E2_ROOT / "e2_severity_comparison.png")

    print(f"Wrote outputs under {E2_ROOT}")


if __name__ == "__main__":
    main()
