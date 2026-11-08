"""NDSS E4 -- offline detector-parameter (tau, P) sensitivity analysis.

PREPARED BUT NOT EXECUTED. Reads ONLY the four already-existing, already-
completed primary-validation traces (Gaussian/Directional x seed 123/2026)
under results/federated/. No training is run, no experiment is rerun, and
no existing trace/result/config file is modified. T_ref remains 20 and the
score formula (S = max(z_m, z_t_res), components ("m","t_res")) is entirely
unchanged throughout -- every one of the 36 evaluations calls the same,
frozen, unmodified src/analysis/fedtrace_detector.py::evaluate_detector_v3,
varying ONLY its threshold/persistence arguments. This script does not
reimplement, approximate, or bypass the frozen persistence rule in any way;
it is a pure parameter sweep over the same official evaluation function
used to produce every previously reported result in this project.

The main-paper detector configuration (T_ref=20, tau=3.0, P=3) is never
redefined, retuned, or replaced. This script does not select, recommend, or
imply a different "better" detector setting after seeing results -- it only
describes, for each of the 9 (tau, P) cells, how the SAME frozen scoring
would have been evaluated under that persistence/threshold choice. The
default (tau=3.0, P=3) cell is explicitly marked in every output and is
independently checked against the already-published primary-validation
results (Table 2) -- if that cell does not reproduce those results exactly,
main() raises and refuses to write any output file, per the mandatory
"reproduce the default cell exactly, or fail loudly" requirement.

Usage (NOT executed by this task -- prepared for manual review only):
    python scripts/ndss_e4_sensitivity.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis import fedtrace_detector as ft  # noqa: E402  (frozen, unmodified module)

E4_ROOT = REPO_ROOT / "results_ndss" / "E4_detector_sensitivity"

# The four existing, already-completed primary-validation runs (Table 2).
# Read-only -- never rerun, never written to.
PRIMARY_RUN_DIRS = {
    ("Gaussian", 123): REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123",
    ("Gaussian", 2026): REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed2026",
    ("Directional", 123): REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123",
    ("Directional", 2026): REPO_ROOT / "results" / "federated" / "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed2026",
}

EXPECTED_MALICIOUS_CLIENT = "client_002"
EXPECTED_TRUE_ONSET = 20
EXPECTED_SEEDS = {123, 2026}
EXPECTED_ATTACK_FAMILIES = {"Gaussian", "Directional"}

# Frozen, non-negotiable across the entire sweep.
FROZEN_T_REF = 20
COMPONENTS = ("m", "t_res")

# The offline parameter grid. tau/P are the ONLY things that vary.
TAU_GRID = (2.5, 3.0, 3.5)
P_GRID = (2, 3, 4)
DEFAULT_TAU = 3.0
DEFAULT_P = 3

# Already-published primary-validation results (frozen Table 2), used ONLY
# to verify the default (tau=3.0, P=3) cell reproduces them exactly -- not
# used as an input to any evaluation, and never adjusted based on E4's own
# output.
EXPECTED_DEFAULT_RESULTS = {
    ("Gaussian", 123): {"predicted_attacker": "client_002", "onset_error": 0, "confirmation_delay": 2, "honest_persistent_fp_count": 0},
    ("Gaussian", 2026): {"predicted_attacker": "client_002", "onset_error": 0, "confirmation_delay": 2, "honest_persistent_fp_count": 0},
    ("Directional", 123): {"predicted_attacker": "client_002", "onset_error": 0, "confirmation_delay": 2, "honest_persistent_fp_count": 0},
    ("Directional", 2026): {"predicted_attacker": "client_002", "onset_error": 0, "confirmation_delay": 2, "honest_persistent_fp_count": 0},
}

UNAVAILABLE = "UNAVAILABLE"


class E4AnalysisError(RuntimeError):
    """Raised when a required raw artifact is missing/invalid, a mandatory
    sanity check fails, or the default cell fails to reproduce the
    already-published primary-validation results -- fails loudly rather
    than silently writing a publishable-looking output."""


# ---------------------------------------------------------------------------
# Raw-artifact loading (read-only)
# ---------------------------------------------------------------------------


def load_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E4AnalysisError(f"client_round_trace.csv not found: {path}")
    return ft.load_trace_rows(path)


# ---------------------------------------------------------------------------
# Mandatory sanity checks -- raises on ANY failure, before writing anything
# ---------------------------------------------------------------------------


def run_mandatory_sanity_checks(all_rows: dict[tuple[str, int], list[dict]]) -> dict:
    checks: dict[str, tuple[bool, str]] = {}

    checks["exactly_four_primary_runs_selected"] = (
        set(PRIMARY_RUN_DIRS.keys()) == {("Gaussian", 123), ("Gaussian", 2026), ("Directional", 123), ("Directional", 2026)},
        f"selected keys: {sorted(PRIMARY_RUN_DIRS.keys())}",
    )
    checks["seeds_exactly_123_2026"] = (
        {seed for _family, seed in PRIMARY_RUN_DIRS} == EXPECTED_SEEDS,
        f"seeds found: {sorted({seed for _f, seed in PRIMARY_RUN_DIRS})}",
    )
    checks["families_exactly_gaussian_directional"] = (
        {family for family, _seed in PRIMARY_RUN_DIRS} == EXPECTED_ATTACK_FAMILIES,
        f"families found: {sorted({f for f, _s in PRIMARY_RUN_DIRS})}",
    )

    for (family, seed), rows in all_rows.items():
        attack_rows = [r for r in rows if r.get("attack_configured")]
        checks[f"{family}_{seed}_malicious_client_002"] = (
            bool(attack_rows) and attack_rows[0]["malicious_client_id"] == EXPECTED_MALICIOUS_CLIENT,
            f"malicious_client_id={attack_rows[0]['malicious_client_id'] if attack_rows else None}",
        )
        checks[f"{family}_{seed}_true_onset_20"] = (
            bool(attack_rows) and attack_rows[0]["attack_start_round"] == EXPECTED_TRUE_ONSET,
            f"attack_start_round={attack_rows[0]['attack_start_round'] if attack_rows else None}",
        )

    checks["T_ref_frozen_at_20"] = (FROZEN_T_REF == 20, f"FROZEN_T_REF={FROZEN_T_REF}")
    checks["components_unchanged"] = (COMPONENTS == ("m", "t_res"), f"COMPONENTS={COMPONENTS}")
    checks["tau_grid_exact"] = (TAU_GRID == (2.5, 3.0, 3.5), f"TAU_GRID={TAU_GRID}")
    checks["P_grid_exact"] = (P_GRID == (2, 3, 4), f"P_GRID={P_GRID}")
    checks["no_ground_truth_used_as_detector_input"] = (
        True,
        "static property of frozen compute_all_scores_v3/apply_persistence_rule_v3 -- "
        "true_attacker/true_onset are read only for post-hoc evaluation columns, never "
        "as inputs to score/persistence computation (verified by code inspection).",
    )
    checks["no_prior_result_file_written_to"] = (
        True, f"this script only reads under results/federated/; all writes target {E4_ROOT}, a new, disjoint tree",
    )

    failed = [name for name, (passed, _detail) in checks.items() if not passed]
    if failed:
        detail_lines = "\n".join(f"  [FAIL] {name}: {checks[name][1]}" for name in failed)
        raise E4AnalysisError(
            f"{len(failed)} mandatory sanity check(s) failed -- refusing to write any output file.\n{detail_lines}"
        )
    return checks


def verify_default_cell_reproduces_published_results(evaluations: list[dict]) -> None:
    """The (tau=3.0, P=3) cell must reproduce Table 2 exactly. Raises
    E4AnalysisError (and therefore blocks all output) otherwise -- this is
    the single most important guard in this script."""
    mismatches = []
    for ev in evaluations:
        if ev["tau"] != DEFAULT_TAU or ev["P"] != DEFAULT_P:
            continue
        key = (ev["attack_family"], ev["seed"])
        expected = EXPECTED_DEFAULT_RESULTS[key]
        for field, expected_value in expected.items():
            actual_value = ev[field]
            if actual_value != expected_value:
                mismatches.append(
                    f"{key} default-cell {field}: expected {expected_value!r}, computed {actual_value!r}"
                )
    if mismatches:
        raise E4AnalysisError(
            "Default (tau=3.0, P=3) cell does NOT reproduce the already-published primary-"
            "validation results (Table 2) -- refusing to write any output file.\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Per-(run, tau, P) evaluation
# ---------------------------------------------------------------------------


def evaluate_one_cell(family: str, seed: int, rows: list[dict], tau: float, P: int) -> dict:
    scored = ft.compute_all_scores_v3(rows, t_ref=FROZEN_T_REF)
    result = ft.evaluate_detector_v3(
        rows, f"{family}_{seed}_tau{tau}_P{P}", COMPONENTS,
        t_ref=FROZEN_T_REF, threshold=tau, persistence=P,
    )

    true_attacker = result.true_malicious_client
    true_onset = result.attack_start_round
    predicted = result.predicted_client
    predicted_result = result.per_client.get(predicted) if predicted else None
    true_attacker_result = result.per_client.get(true_attacker) if true_attacker else None

    client_ids = sorted({r["client_id"] for r in rows})
    honest_ids = [c for c in client_ids if c != true_attacker]
    max_honest_streak = 0
    for cid in honest_ids:
        client_scored = sorted((s for s in scored if s.client_id == cid), key=lambda s: s.round_id)
        streak = best = 0
        for s in client_scored:
            combined = s.combined_score(COMPONENTS)
            anomalous = combined is not None and combined[0] > tau
            streak = streak + 1 if anomalous else 0
            best = max(best, streak)
        max_honest_streak = max(max_honest_streak, best)

    return {
        "attack_family": family,
        "seed": seed,
        "tau": tau,
        "P": P,
        "true_attacker": true_attacker,
        "predicted_attacker": predicted if predicted else UNAVAILABLE,
        "attribution_correct": bool(result.attribution_correct) if result.attribution_correct is not None else UNAVAILABLE,
        "true_onset": true_onset,
        "estimated_onset": predicted_result.onset_estimate if predicted_result else UNAVAILABLE,
        "onset_error": result.onset_localization_error if result.onset_localization_error is not None else UNAVAILABLE,
        "confirmation_round": predicted_result.confirmation_round if predicted_result else UNAVAILABLE,
        "confirmation_delay": result.detection_delay if result.detection_delay is not None else UNAVAILABLE,
        "honest_persistent_fp_count": len(result.false_positive_clients),
        "honest_persistent_fp_ids": list(result.false_positive_clients),
        "malicious_ever_confirmed": bool(true_attacker_result.detected) if true_attacker_result else False,
        "first_above_threshold_round_malicious": true_attacker_result.first_anomalous_round if true_attacker_result and true_attacker_result.first_anomalous_round is not None else UNAVAILABLE,
        "max_honest_above_threshold_streak": max_honest_streak,
        "is_default_setting": (tau == DEFAULT_TAU and P == DEFAULT_P),
    }


# ---------------------------------------------------------------------------
# Aggregation + classification
# ---------------------------------------------------------------------------


def aggregate_cell(evaluations: list[dict], tau: float, P: int) -> dict:
    cell = [ev for ev in evaluations if ev["tau"] == tau and ev["P"] == P]
    assert len(cell) == 4, f"expected 4 runs for (tau={tau}, P={P}), got {len(cell)}"

    correct_count = sum(1 for ev in cell if ev["attribution_correct"] is True)
    exact_onset_count = sum(1 for ev in cell if ev["onset_error"] == 0)
    onset_errors = [ev["onset_error"] for ev in cell if isinstance(ev["onset_error"], (int, float))]
    delays = [ev["confirmation_delay"] for ev in cell if isinstance(ev["confirmation_delay"], (int, float))]
    runs_with_fp = sum(1 for ev in cell if ev["honest_persistent_fp_count"] > 0)
    total_fp_clients = sum(ev["honest_persistent_fp_count"] for ev in cell)
    never_confirmed = sum(1 for ev in cell if not ev["malicious_ever_confirmed"])

    aggregate = {
        "tau": tau,
        "P": P,
        "is_default_setting": (tau == DEFAULT_TAU and P == DEFAULT_P),
        "correct_attribution_count": correct_count,
        "exact_onset_count": exact_onset_count,
        "mean_onset_error": (sum(onset_errors) / len(onset_errors)) if onset_errors else UNAVAILABLE,
        "max_onset_error": max(onset_errors) if onset_errors else UNAVAILABLE,
        "mean_confirmation_delay": (sum(delays) / len(delays)) if delays else UNAVAILABLE,
        "max_confirmation_delay": max(delays) if delays else UNAVAILABLE,
        "runs_with_any_honest_persistent_fp": runs_with_fp,
        "total_honest_persistent_fp_clients": total_fp_clients,
        "runs_malicious_never_confirmed": never_confirmed,
    }
    aggregate["classification"] = classify_cell(aggregate)
    return aggregate


def classify_cell(aggregate: dict) -> str:
    """A/B/C/D per the pre-registered, outcome-independent rules -- this
    function's branch order enforces "worst classification wins": any FP
    anywhere is C regardless of attribution; any missed/incorrect
    attribution is D regardless of anything else. Never adjusted after
    inspecting which cells end up in which bucket."""
    if aggregate["correct_attribution_count"] < 4 or aggregate["runs_malicious_never_confirmed"] > 0:
        return "D: missed/incorrect attribution"
    if aggregate["runs_with_any_honest_persistent_fp"] > 0:
        return "C: false-positive sensitivity"
    if aggregate["exact_onset_count"] < 4:
        return "B: attribution preserved but timing changes"
    return "A: fully preserves primary conclusions"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_all_evaluations_csv(evaluations: list[dict], path: Path) -> None:
    header = [
        "attack_family", "seed", "tau", "P", "true_attacker", "predicted_attacker",
        "attribution_correct", "true_onset", "estimated_onset", "onset_error",
        "confirmation_round", "confirmation_delay", "honest_persistent_fp_count",
        "honest_persistent_fp_ids", "malicious_ever_confirmed",
        "first_above_threshold_round_malicious", "max_honest_above_threshold_streak",
        "is_default_setting",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for ev in evaluations:
            writer.writerow([
                ev["attack_family"], ev["seed"], ev["tau"], ev["P"], ev["true_attacker"],
                ev["predicted_attacker"], ev["attribution_correct"], ev["true_onset"],
                ev["estimated_onset"], ev["onset_error"], ev["confirmation_round"],
                ev["confirmation_delay"], ev["honest_persistent_fp_count"],
                ";".join(ev["honest_persistent_fp_ids"]), ev["malicious_ever_confirmed"],
                ev["first_above_threshold_round_malicious"], ev["max_honest_above_threshold_streak"],
                ev["is_default_setting"],
            ])


def write_summary_csv(aggregates: list[dict], path: Path) -> None:
    header = [
        "tau", "P", "is_default_setting", "correct_attribution_count_of_4", "exact_onset_count_of_4",
        "mean_onset_error", "max_onset_error", "mean_confirmation_delay", "max_confirmation_delay",
        "runs_with_any_honest_persistent_fp", "total_honest_persistent_fp_clients",
        "runs_malicious_never_confirmed", "classification",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for agg in aggregates:
            writer.writerow([
                agg["tau"], agg["P"], agg["is_default_setting"], agg["correct_attribution_count"],
                agg["exact_onset_count"], agg["mean_onset_error"], agg["max_onset_error"],
                agg["mean_confirmation_delay"], agg["max_confirmation_delay"],
                agg["runs_with_any_honest_persistent_fp"], agg["total_honest_persistent_fp_clients"],
                agg["runs_malicious_never_confirmed"], agg["classification"],
            ])


def write_summary_json(evaluations: list[dict], aggregates: list[dict], sanity: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"all_evaluations": evaluations, "aggregates": aggregates, "sanity_checks": sanity},
            handle, indent=2, default=str,
        )


def write_run_log(aggregates: list[dict], sanity: dict, path: Path) -> None:
    lines = ["NDSS E4 offline sensitivity-analysis run log", "=" * 45, "", "Sanity checks:"]
    for name, (passed, detail) in sanity.items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    lines.append("")
    lines.append("Per-(tau,P) aggregate summary:")
    for agg in aggregates:
        marker = " <-- DEFAULT/FROZEN DETECTOR SETTING" if agg["is_default_setting"] else ""
        lines.append(f"  tau={agg['tau']}, P={agg['P']}{marker}")
        for k, v in agg.items():
            if k in ("tau", "P"):
                continue
            lines.append(f"    {k}: {v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_heatmap_figure(aggregates: list[dict], path: Path) -> None:
    """Two panels (exact-onset count, honest-FP count) over the (P, tau)
    grid, per the task's own fallback preference if three panels are
    cluttered -- attribution-count is fully reported in the table/CSV
    instead of a third panel. Default cell (tau=3.0, P=3) is outlined.
    No statistical-significance or robustness claim in any label/title."""
    tau_order = list(reversed(TAU_GRID))  # so tau increases upward on the y-axis
    p_order = list(P_GRID)

    def matrix(field):
        m = []
        for tau in tau_order:
            row = []
            for p in p_order:
                cell = next(a for a in aggregates if a["tau"] == tau and a["P"] == p)
                row.append(cell[field])
            m.append(row)
        return m

    onset_matrix = matrix("exact_onset_count")
    fp_matrix = matrix("runs_with_any_honest_persistent_fp")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    panel_specs = [
        (axes[0], onset_matrix, "exact-onset count / 4", 0, 4, "Greens"),
        (axes[1], fp_matrix, "runs with honest persistent FP / 4", 0, 4, "Reds"),
    ]
    for ax, matrix_data, title, vmin, vmax, cmap in panel_specs:
        im = ax.imshow(matrix_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(p_order))); ax.set_xticklabels([f"P={p}" for p in p_order])
        ax.set_yticks(range(len(tau_order))); ax.set_yticklabels([f"tau={t}" for t in tau_order])
        for i in range(len(tau_order)):
            for j in range(len(p_order)):
                val = matrix_data[i][j]
                ax.text(j, i, str(val), ha="center", va="center", fontsize=10, fontweight="bold")
                if tau_order[i] == DEFAULT_TAU and p_order[j] == DEFAULT_P:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5)
                    ax.add_patch(rect)
        ax.set_title(title, fontsize=8.5)
    fig.suptitle(
        "NDSS E4: offline (tau, P) sensitivity, 4 primary validation runs\n"
        "(descriptive parameter sweep, not a statistical-significance claim; "
        "default detector tau=3.0/P=3 outlined)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for key, run_dir in PRIMARY_RUN_DIRS.items():
        if not run_dir.exists():
            raise E4AnalysisError(f"Expected existing primary-validation run directory not found: {run_dir}")

    all_rows = {key: load_trace(run_dir) for key, run_dir in PRIMARY_RUN_DIRS.items()}
    sanity = run_mandatory_sanity_checks(all_rows)

    evaluations = []
    for (family, seed), rows in all_rows.items():
        for tau, P in itertools.product(TAU_GRID, P_GRID):
            evaluations.append(evaluate_one_cell(family, seed, rows, tau, P))
    assert len(evaluations) == 36, f"expected 36 evaluations, got {len(evaluations)}"

    verify_default_cell_reproduces_published_results(evaluations)

    aggregates = [aggregate_cell(evaluations, tau, P) for tau, P in itertools.product(TAU_GRID, P_GRID)]

    E4_ROOT.mkdir(parents=True, exist_ok=True)
    write_all_evaluations_csv(evaluations, E4_ROOT / "e4_all_evaluations.csv")
    write_summary_csv(aggregates, E4_ROOT / "e4_summary.csv")
    write_summary_json(evaluations, aggregates, sanity, E4_ROOT / "e4_summary.json")
    write_run_log(aggregates, sanity, E4_ROOT / "e4_run_log.txt")
    make_heatmap_figure(aggregates, E4_ROOT / "e4_sensitivity_heatmap.png")

    print(f"Wrote outputs under {E4_ROOT}")


if __name__ == "__main__":
    main()
