"""NDSS E1 -- 10-client scaling test: post-processing / analysis script.

Reads ONLY the existing, already-completed raw outputs of the three E1 runs
under results_ndss/E1_10clients/{clean_seed123, gaussian_client002_round20_seed123,
directional_client002_round20_seed123}/ (each produced by the existing,
unmodified src.experiments.run_federated_cifar100 runner). Scores every
trace with the existing, frozen, unmodified src/analysis/fedtrace_detector.py
module -- no formula, threshold, persistence value, or calibration window is
changed or reimplemented here. No training, no experiment, and no GPU job is
run by this script; it performs only file reads, frozen-detector scoring
(pure CPU computation over already-saved CSV/JSON data), and file writes
under results_ndss/E1_10clients/. No existing raw run output is modified.

If a requested quantity cannot be reconstructed from the saved artifacts
(e.g. peak GPU memory, which the existing runner never logs), the
corresponding field is written as the literal string "UNAVAILABLE" -- never
inferred, estimated, or fabricated.

Usage (NOT executed by this task -- prepared for manual review only):
    python scripts/ndss_e1_postprocess.py
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

E1_ROOT = REPO_ROOT / "results_ndss" / "E1_10clients"
RUN_DIRS = {
    "clean": E1_ROOT / "clean_seed123",
    "gaussian": E1_ROOT / "gaussian_client002_round20_seed123",
    "directional": E1_ROOT / "directional_client002_round20_seed123",
}
WRAPPER_LOG = E1_ROOT / "e1_wrapper_log.txt"

UNAVAILABLE = "UNAVAILABLE"

# Frozen detector parameters -- must match src/analysis/fedtrace_detector.py's
# own module constants exactly. Asserted (not silently trusted) so that any
# future drift in the frozen module is caught loudly rather than silently
# inherited by this NDSS analysis.
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

ROUND_WINDOW_18_22 = list(range(18, 23))


class E1AnalysisError(RuntimeError):
    """Raised when a required raw artifact is missing or structurally invalid --
    fails loudly rather than silently proceeding with an incomplete analysis."""


# ---------------------------------------------------------------------------
# Raw-artifact loading (read-only)
# ---------------------------------------------------------------------------


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise E1AnalysisError(f"summary.json not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E1AnalysisError(f"client_round_trace.csv not found: {path}")
    return ft.load_trace_rows(path)


def load_raw_csv_rows(run_dir: Path) -> list[dict]:
    """Read client_round_trace.csv directly (all columns), bypassing
    ft.load_trace_rows -- the frozen loader only parses the subset of
    columns the detector's own scoring needs and silently drops
    diagnostic-only columns such as delta_attack_norm_ratio /
    realized_cosine_to_honest. Used ONLY for auxiliary sanity checks that
    are not part of the frozen detector's scoring path; never used to feed
    compute_all_scores_v3 / evaluate_detector_v3."""
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E1AnalysisError(f"client_round_trace.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_partition_summary(run_dir: Path) -> dict[str, int]:
    path = run_dir / "client_partition_summary.csv"
    if not path.exists():
        raise E1AnalysisError(f"client_partition_summary.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["client_id"]: int(row["num_samples"]) for row in csv.DictReader(handle)}


def parse_wrapper_runtime(condition_label: str) -> str | int:
    """Best-effort cross-check of wall-clock runtime from the wrapper log's
    own start/end markers (written by the shell wrapper that launched all
    three runs sequentially). Returns UNAVAILABLE if the log or the matching
    markers are absent -- never estimated from partial data."""
    if not WRAPPER_LOG.exists():
        return UNAVAILABLE
    text = WRAPPER_LOG.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"--- {condition_label} run end:.*?elapsed_s=(\d+) ---", text, re.IGNORECASE
    )
    return int(match.group(1)) if match else UNAVAILABLE


# ---------------------------------------------------------------------------
# Sanity checks (per run)
# ---------------------------------------------------------------------------


def run_sanity_checks(run_key: str, run_dir: Path, rows: list[dict], summary: dict) -> dict:
    """Returns {check_name: (bool_pass, detail_string)}. Never raises -- a
    failed check is reported, not hidden, so the caller can print a PASS/FAIL
    table without the whole script aborting on the first failure."""
    checks: dict[str, tuple[bool, str]] = {}

    client_ids = sorted({row["client_id"] for row in rows})
    checks["exactly_10_distinct_clients"] = (
        len(client_ids) == 10,
        f"found {len(client_ids)}: {client_ids}",
    )

    round_ids = sorted({row["round_id"] for row in rows})
    checks["50_rounds_present"] = (
        round_ids == list(range(50)),
        f"found {len(round_ids)} rounds, range [{min(round_ids)},{max(round_ids)}]"
        if round_ids else "no rounds found",
    )

    full_participation = all(
        sum(1 for r in rows if r["round_id"] == rid and r.get("selected")) == 10
        for rid in round_ids
    )
    checks["full_participation_every_round"] = (
        full_participation, "10/10 clients selected in every round" if full_participation
        else "at least one round had != 10 selected clients",
    )

    checks["num_clients_10_in_summary"] = (
        summary.get("num_clients") == 10, f"summary.json num_clients={summary.get('num_clients')}",
    )
    checks["clients_per_round_10_in_summary"] = (
        summary.get("clients_per_round") == 10,
        f"summary.json clients_per_round={summary.get('clients_per_round')}",
    )

    checks["T_ref_matches_frozen_default"] = (
        ft.DEFAULT_T_REF == EXPECTED_T_REF, f"ft.DEFAULT_T_REF={ft.DEFAULT_T_REF}",
    )
    checks["tau_matches_frozen_default"] = (
        ft.DEFAULT_THRESHOLD == EXPECTED_THRESHOLD, f"ft.DEFAULT_THRESHOLD={ft.DEFAULT_THRESHOLD}",
    )
    checks["P_matches_frozen_default"] = (
        ft.DEFAULT_PERSISTENCE == EXPECTED_PERSISTENCE, f"ft.DEFAULT_PERSISTENCE={ft.DEFAULT_PERSISTENCE}",
    )

    attack_info = summary.get("attack", {})
    attack_type = attack_info.get("attack_type")

    if run_key == "clean":
        checks["no_attack_configured"] = (
            attack_type == "none" and not any(r.get("attack_configured") for r in rows),
            f"summary attack_type={attack_type}; any attack_configured row: "
            f"{any(r.get('attack_configured') for r in rows)}",
        )
    else:
        onset = attack_info.get("attack_start_round")
        checks["attack_onset_exactly_round_20"] = (
            onset == 20, f"summary attack_start_round={onset}",
        )
        pre_onset_active = [
            r for r in rows if r["round_id"] < 20 and r.get("attack_active")
        ]
        checks["no_attack_active_before_round_20"] = (
            len(pre_onset_active) == 0,
            f"{len(pre_onset_active)} pre-round-20 row(s) with attack_active=True"
            if pre_onset_active else "0 pre-round-20 active-attack rows",
        )
        post_onset_malicious_active = all(
            r.get("attack_active")
            for r in rows
            if r["client_id"] == attack_info.get("malicious_client_id") and r["round_id"] >= 20
        )
        checks["attack_active_every_post_onset_round_for_attacker"] = (
            post_onset_malicious_active, "checked all round_id>=20 rows for the malicious client",
        )

    if run_key == "gaussian":
        rho = attack_info.get("attack_strength_rho")
        checks["gaussian_rho_exactly_1.0"] = (rho == 1.0, f"summary attack_strength_rho={rho}")

    if run_key == "directional":
        c = attack_info.get("directional_target_cosine")
        checks["directional_target_cosine_exactly_0.0"] = (
            c == 0.0, f"summary directional_target_cosine={c}",
        )
        attacker = attack_info.get("malicious_client_id")
        raw_rows = load_raw_csv_rows(run_dir)
        ratios = [
            float(r["delta_attack_norm_ratio"])
            for r in raw_rows
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

    # No ground-truth attacker identity is used as detector input: verified
    # by construction, not by a runtime probe -- compute_all_scores_v3 and
    # apply_persistence_rule_v3 (src/analysis/fedtrace_detector.py) never
    # read "malicious_client_id" or "attack_start_round" from the row
    # mapping anywhere in their bodies; those fields are read only by
    # evaluate_detector_v3's OWN post-hoc evaluation code (attribution
    # correctness / onset error), strictly after the attribution decision
    # (predicted_client) has already been computed. This is a static
    # property of the frozen module, not something a per-run trace can
    # demonstrate at runtime, so it is recorded here as a fixed statement
    # rather than a per-run pass/fail.
    checks["no_ground_truth_used_as_detector_input"] = (
        True,
        "static property of frozen compute_all_scores_v3/apply_persistence_rule_v3 "
        "(verified by code inspection, not a per-run runtime check)",
    )

    return checks


# ---------------------------------------------------------------------------
# Per-run FedTrace analysis
# ---------------------------------------------------------------------------


def analyze_attacked_run(run_key: str, run_dir: Path) -> dict:
    rows = load_trace(run_dir)
    summary = load_summary(run_dir)
    scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
    result = ft.evaluate_detector_v3(
        rows, run_key, ("m", "t_res"),
        t_ref=EXPECTED_T_REF, threshold=EXPECTED_THRESHOLD, persistence=EXPECTED_PERSISTENCE,
    )

    true_attacker = result.true_malicious_client
    true_onset = result.attack_start_round
    predicted = result.predicted_client
    client_ids = sorted({r["client_id"] for r in rows})
    honest_ids = [c for c in client_ids if c != true_attacker]

    # Persistent honest false positives + max streak per honest client.
    honest_fp_ids = list(result.false_positive_clients)
    max_streaks: dict[str, int] = {}
    for cid in client_ids:
        client_scored = sorted((s for s in scored if s.client_id == cid), key=lambda s: s.round_id)
        streak = 0
        best = 0
        for s in client_scored:
            combined = s.combined_score(("m", "t_res"))
            anomalous = combined is not None and combined[0] > EXPECTED_THRESHOLD
            streak = streak + 1 if anomalous else 0
            best = max(best, streak)
        max_streaks[cid] = best

    # Dominance counts: malicious client, round_id >= true_onset, S > tau.
    dominance = {"m": 0, "t_res": 0, "tie": 0}
    if true_attacker is not None and true_onset is not None:
        mal_scored = sorted((s for s in scored if s.client_id == true_attacker), key=lambda s: s.round_id)
        for s in mal_scored:
            if s.round_id < true_onset:
                continue
            combined = s.combined_score(("m", "t_res"))
            if combined is None or combined[0] <= EXPECTED_THRESHOLD:
                continue
            if s.z_m is not None and s.z_t_res is not None and abs(s.z_m - s.z_t_res) < 1e-12:
                dominance["tie"] += 1
            else:
                dominance[combined[1]] += 1

    # Malicious-client and max-honest scores at rounds 18-22.
    scores_18_22 = {}
    for rid in ROUND_WINDOW_18_22:
        mal_row = next((s for s in scored if s.client_id == true_attacker and s.round_id == rid), None) \
            if true_attacker else None
        mal_combined = mal_row.combined_score(("m", "t_res")) if mal_row else None
        honest_vals = []
        for cid in honest_ids:
            hr = next((s for s in scored if s.client_id == cid and s.round_id == rid), None)
            if hr is None:
                continue
            hc = hr.combined_score(("m", "t_res"))
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
        "run_key": run_key,
        "seed": summary.get("seed"),
        "num_clients": summary.get("num_clients"),
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
        "dominance_post_onset": dominance,
        "scores_rounds_18_22": scores_18_22,
        "final_top1_pct": summary["final_test"]["top1"] * 100 if "final_test" in summary else UNAVAILABLE,
        "total_experiment_seconds_summary_json": summary.get("total_experiment_seconds", UNAVAILABLE),
        "total_experiment_seconds_wrapper_log": parse_wrapper_runtime(run_key.upper()),
        "peak_gpu_memory": UNAVAILABLE,  # never logged by the existing runner -- see report Section D
    }


def analyze_clean_run(run_dir: Path) -> dict:
    rows = load_trace(run_dir)
    summary = load_summary(run_dir)
    scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
    result = ft.evaluate_detector_v3(
        rows, "clean", ("m", "t_res"),
        t_ref=EXPECTED_T_REF, threshold=EXPECTED_THRESHOLD, persistence=EXPECTED_PERSISTENCE,
    )
    client_ids = sorted({r["client_id"] for r in rows})

    max_scores: dict[str, float | str] = {}
    max_streaks: dict[str, int] = {}
    for cid in client_ids:
        client_scored = sorted((s for s in scored if s.client_id == cid), key=lambda s: s.round_id)
        vals = [s.combined_score(("m", "t_res"))[0] for s in client_scored if s.combined_score(("m", "t_res")) is not None]
        max_scores[cid] = max(vals) if vals else UNAVAILABLE
        streak = 0
        best = 0
        for s in client_scored:
            combined = s.combined_score(("m", "t_res"))
            anomalous = combined is not None and combined[0] > EXPECTED_THRESHOLD
            streak = streak + 1 if anomalous else 0
            best = max(best, streak)
        max_streaks[cid] = best

    confirmed_clients = {cid: r.confirmation_round for cid, r in result.per_client.items() if r.detected}

    return {
        "run_key": "clean",
        "seed": summary.get("seed"),
        "num_clients": summary.get("num_clients"),
        "any_client_confirmed": bool(confirmed_clients),
        "confirmed_clients_and_rounds": confirmed_clients,
        "persistent_false_positive_ids": list(confirmed_clients.keys()),
        "max_combined_score_by_client": max_scores,
        "max_streak_by_client": max_streaks,
        "final_top1_pct": summary["final_test"]["top1"] * 100 if "final_test" in summary else UNAVAILABLE,
        "total_experiment_seconds_summary_json": summary.get("total_experiment_seconds", UNAVAILABLE),
        "total_experiment_seconds_wrapper_log": parse_wrapper_runtime("CLEAN"),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_e1_summary_csv(results: dict, path: Path) -> None:
    header = [
        "run", "seed", "num_clients", "true_attacker", "true_onset", "predicted_attacker",
        "estimated_onset", "confirmation_round", "onset_error", "confirmation_delay",
        "honest_persistent_fp_count", "honest_persistent_fp_ids", "final_top1_pct",
        "total_experiment_seconds", "peak_gpu_memory",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key in ("gaussian", "directional"):
            r = results[key]
            writer.writerow([
                key, r["seed"], r["num_clients"], r["true_attacker"], r["true_onset"],
                r["predicted_attacker"], r["estimated_onset"], r["confirmation_round"],
                r["onset_error"], r["confirmation_delay"], r["honest_persistent_fp_count"],
                ";".join(r["honest_persistent_fp_ids"]) if r["honest_persistent_fp_ids"] else "",
                r["final_top1_pct"], r["total_experiment_seconds_summary_json"], r["peak_gpu_memory"],
            ])
        c = results["clean"]
        writer.writerow([
            "clean", c["seed"], c["num_clients"], UNAVAILABLE, UNAVAILABLE,
            ";".join(c["persistent_false_positive_ids"]) if c["persistent_false_positive_ids"] else "None",
            UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, len(c["persistent_false_positive_ids"]),
            ";".join(c["persistent_false_positive_ids"]) if c["persistent_false_positive_ids"] else "",
            c["final_top1_pct"], c["total_experiment_seconds_summary_json"], UNAVAILABLE,
        ])


def write_e1_summary_json(results: dict, sanity: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "sanity_checks": sanity}, handle, indent=2, default=str)


def write_e1_roundwise_scores_csv(rows_out: list[list], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "client_id", "round_id", "z_m", "z_t_res", "S", "is_malicious"])
        writer.writerows(rows_out)


def write_e1_client_partitions_csv(results: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "client_id", "num_samples"])
        for key, run_dir in RUN_DIRS.items():
            counts = load_partition_summary(run_dir)
            for cid, n in sorted(counts.items()):
                writer.writerow([key, cid, n])


def write_e1_run_log(results: dict, sanity: dict, path: Path) -> None:
    lines = ["NDSS E1 post-processing run log", "=" * 40, ""]
    for key in ("clean", "gaussian", "directional"):
        lines.append(f"--- {key} ---")
        for k, v in results[key].items():
            lines.append(f"  {k}: {v}")
        lines.append("  sanity checks:")
        for name, (passed, detail) in sanity[key].items():
            lines.append(f"    [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(path: Path) -> None:
    from generate_paper_figures import MALICIOUS_COLOR, HONEST_COLOR, THRESHOLD_COLOR, ONSET_COLOR, CLIENT_MARKERS  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [("gaussian", "Gaussian (client_002, seed 123)"), ("directional", "Directional (client_002, seed 123)")]
    for ax, (key, title) in zip(axes, panels):
        rows = load_trace(RUN_DIRS[key])
        scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
        summary = load_summary(RUN_DIRS[key])
        attacker = summary["attack"]["malicious_client_id"]
        onset = summary["attack"]["attack_start_round"]
        client_ids = sorted({r["client_id"] for r in rows})
        for cid in client_ids:
            client_scored = sorted((s for s in scored if s.client_id == cid), key=lambda s: s.round_id)
            xs, ys = [], []
            for s in client_scored:
                combined = s.combined_score(("m", "t_res"))
                if combined is None:
                    continue
                xs.append(s.round_id)
                ys.append(combined[0])
            marker = CLIENT_MARKERS.get(cid, "o")
            if cid == attacker:
                ax.plot(xs, ys, color=MALICIOUS_COLOR, lw=2.2, marker="^", ms=4, label=f"{cid} (malicious)", zorder=5)
            else:
                ax.plot(xs, ys, color=HONEST_COLOR, lw=0.9, alpha=0.75, marker=marker, ms=2.5, label=cid)
        ax.axhline(EXPECTED_THRESHOLD, color=THRESHOLD_COLOR, lw=1.0, ls="--", label=r"$\tau=3$")
        ax.axvline(onset, color=ONSET_COLOR, lw=1.0, ls=":", label=f"onset (r={onset})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("communication round")
    axes[0].set_ylabel(r"$S_{k,r}=\max(z^m_{k,r},\,z^t_{\mathrm{res},k,r})$")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=6, fontsize=6.5)
    fig.suptitle(
        "NDSS E1: FedTrace combined score, 10-client federation, rounds 0-49\n"
        "(single run per attack family, seed 123 -- descriptive, not a generalization claim)",
        fontsize=8.5, y=1.18,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for key, run_dir in RUN_DIRS.items():
        if not run_dir.exists():
            raise E1AnalysisError(f"Expected E1 run directory not found: {run_dir}")

    results = {
        "clean": analyze_clean_run(RUN_DIRS["clean"]),
        "gaussian": analyze_attacked_run("gaussian", RUN_DIRS["gaussian"]),
        "directional": analyze_attacked_run("directional", RUN_DIRS["directional"]),
    }

    sanity = {}
    for key in ("clean", "gaussian", "directional"):
        rows = load_trace(RUN_DIRS[key])
        summary = load_summary(RUN_DIRS[key])
        sanity[key] = run_sanity_checks(key, RUN_DIRS[key], rows, summary)

    roundwise_rows = []
    for key in ("clean", "gaussian", "directional"):
        rows = load_trace(RUN_DIRS[key])
        summary = load_summary(RUN_DIRS[key])
        attacker = summary.get("attack", {}).get("malicious_client_id")
        scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
        for s in scored:
            combined = s.combined_score(("m", "t_res"))
            roundwise_rows.append([
                key, s.client_id, s.round_id, s.z_m, s.z_t_res,
                combined[0] if combined else "",
                s.client_id == attacker,
            ])

    write_e1_summary_csv(results, E1_ROOT / "e1_summary.csv")
    write_e1_summary_json(results, sanity, E1_ROOT / "e1_summary.json")
    write_e1_roundwise_scores_csv(roundwise_rows, E1_ROOT / "e1_roundwise_scores.csv")
    write_e1_client_partitions_csv(results, E1_ROOT / "e1_client_partitions.csv")
    write_e1_run_log(results, sanity, E1_ROOT / "e1_run_log.txt")
    make_figure(E1_ROOT / "e1_10client_scaling.png")

    print(f"Wrote outputs under {E1_ROOT}")


if __name__ == "__main__":
    main()
