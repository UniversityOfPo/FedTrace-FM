"""NDSS E3 -- intermittent attack-window robustness check: post-processing script.

PREPARED BUT NOT EXECUTED. Reads ONLY the (future) raw outputs of the two E3
runs under results_ndss/E3_intermittent/{gaussian_client002_seed123,
directional_client002_seed123}/. Scores every trace with the existing,
frozen, unmodified src/analysis/fedtrace_detector.py module -- no formula,
threshold, persistence value, or calibration window is changed or
reimplemented here. No training, no experiment, and no GPU job is run by
this script; it performs only file reads, frozen-detector scoring (pure CPU
computation over already-saved CSV/JSON data), and file writes under
results_ndss/E3_intermittent/. No existing raw run output (E1, E2, the
primary-validation series, or any other prior experiment) is read for write
purposes or modified in any way.

Global attribution/onset/confirmation/delay/false-positive fields are taken
EXACTLY as returned by evaluate_detector_v3 -- this script does not
reimplement or approximate the persistence rule for those top-level fields.
The additional burst/gap diagnostics this script computes are clearly
separate, explicitly labeled quantities:
  - "local_3_in_a_row_within_burst": a simple, burst-scoped scan of that
    burst's OWN above-threshold rounds only -- NOT the frozen detector's
    real cross-round persistence state, and never presented as such.
  - "global_confirmation_falls_in_this_window": whether the ONE real,
    global confirmation round (from evaluate_detector_v3, which may have
    been produced by a streak spanning a window boundary) happens to fall
    inside this burst's round range -- a fact about the real algorithm's
    output, not a separate computation.
Residual post-burst evidence in a gap is reported as a raw score fact only;
it is never reinterpreted as a new ground-truth attack event.

If a requested quantity cannot be reconstructed from the saved artifacts,
it is written as the literal string "UNAVAILABLE" -- never inferred,
estimated, or fabricated. A failed, delayed, or missed detection (globally,
per-burst, or per-gap) is reported exactly as computed; this script contains
no logic that adjusts, retries, suppresses, or reinterprets the frozen
detector's output based on whether the result is favorable. If any
mandatory sanity check fails, main() raises rather than writing any output
file (see run_sanity_checks / the "fail loudly" requirement).

Usage (NOT executed by this task -- prepared for manual review only):
    python scripts/ndss_e3_postprocess.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis import fedtrace_detector as ft  # noqa: E402  (frozen, unmodified module)

E3_ROOT = REPO_ROOT / "results_ndss" / "E3_intermittent"
E3_RUN_DIRS = {
    "gaussian": E3_ROOT / "gaussian_client002_seed123",
    "directional": E3_ROOT / "directional_client002_seed123",
}

UNAVAILABLE = "UNAVAILABLE"

EXPECTED_T_REF = 20
EXPECTED_THRESHOLD = 3.0
EXPECTED_PERSISTENCE = 3
assert ft.DEFAULT_T_REF == EXPECTED_T_REF, f"Frozen T_ref drifted: {ft.DEFAULT_T_REF}"
assert ft.DEFAULT_THRESHOLD == EXPECTED_THRESHOLD, f"Frozen tau drifted: {ft.DEFAULT_THRESHOLD}"
assert ft.DEFAULT_PERSISTENCE == EXPECTED_PERSISTENCE, f"Frozen P drifted: {ft.DEFAULT_PERSISTENCE}"
COMPONENTS = ("m", "t_res")

TRUE_WINDOWS = [(20, 24), (30, 34), (40, 44)]
TRUE_GAPS = [(25, 29), (35, 39), (45, 49)]
EXPECTED_ACTIVE_ROUNDS = {r for start, end in TRUE_WINDOWS for r in range(start, end + 1)}
EXPECTED_MALICIOUS_CLIENT = "client_002"

_ALL_PRIOR_ROOTS = [
    REPO_ROOT / "results" / "federated",
    REPO_ROOT / "results_ndss" / "E1_10clients",
    REPO_ROOT / "results_ndss" / "E2_attack_severity",
]
for _run_dir in E3_RUN_DIRS.values():
    for _prior_root in _ALL_PRIOR_ROOTS:
        assert not str(_run_dir).startswith(str(_prior_root)), (
            f"E3 output path {_run_dir} would overlap prior experiment root {_prior_root}."
        )


class E3AnalysisError(RuntimeError):
    """Raised when a required raw artifact is missing/invalid, or a mandatory
    sanity check fails -- fails loudly rather than silently proceeding to
    write a publishable-looking output."""


# ---------------------------------------------------------------------------
# Raw-artifact loading (read-only)
# ---------------------------------------------------------------------------


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise E3AnalysisError(f"summary.json not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E3AnalysisError(f"client_round_trace.csv not found: {path}")
    return ft.load_trace_rows(path)


def load_raw_csv_rows(run_dir: Path) -> list[dict]:
    """Full raw CSV read, for diagnostic-only columns the frozen loader
    doesn't parse (delta_attack_norm_ratio, realized_cosine_to_honest).
    Never fed to compute_all_scores_v3 / evaluate_detector_v3."""
    path = run_dir / "client_round_trace.csv"
    if not path.exists():
        raise E3AnalysisError(f"client_round_trace.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Mandatory sanity checks -- raises E3AnalysisError on ANY failure
# ---------------------------------------------------------------------------


def run_mandatory_sanity_checks(run_key: str, run_dir: Path, rows: list[dict], summary: dict, is_gaussian: bool) -> dict:
    checks: dict[str, tuple[bool, str]] = {}

    client_ids = sorted({row["client_id"] for row in rows})
    checks["exactly_5_distinct_clients"] = (len(client_ids) == 5, f"found {len(client_ids)}: {client_ids}")

    round_ids = sorted({row["round_id"] for row in rows})
    checks["exactly_50_rounds"] = (round_ids == list(range(50)), f"found {len(round_ids)} rounds")

    full_participation = all(
        sum(1 for r in rows if r["round_id"] == rid and r.get("selected")) == 5 for rid in round_ids
    )
    checks["5_of_5_participation_every_round"] = (full_participation, "checked all 50 rounds")

    checks["seed_123"] = (summary.get("seed") == 123, f"summary seed={summary.get('seed')}")

    attack_info = summary.get("attack", {})
    checks["malicious_client_exactly_client_002"] = (
        attack_info.get("malicious_client_id") == EXPECTED_MALICIOUS_CLIENT,
        f"summary malicious_client_id={attack_info.get('malicious_client_id')}",
    )

    actual_active_rounds = {
        r["round_id"] for r in rows if r["client_id"] == EXPECTED_MALICIOUS_CLIENT and r.get("attack_active")
    }
    checks["exact_active_malicious_round_set"] = (
        actual_active_rounds == EXPECTED_ACTIVE_ROUNDS,
        f"expected {sorted(EXPECTED_ACTIVE_ROUNDS)}, got {sorted(actual_active_rounds)}",
    )

    off_window_active = actual_active_rounds - EXPECTED_ACTIVE_ROUNDS
    checks["client_002_inactive_outside_windows"] = (
        len(off_window_active) == 0, f"unexpected active rounds outside windows: {sorted(off_window_active)}",
    )

    honest_active = [r for r in rows if r["client_id"] != EXPECTED_MALICIOUS_CLIENT and r.get("attack_active")]
    checks["all_honest_clients_attack_active_false_always"] = (
        len(honest_active) == 0, f"{len(honest_active)} honest-client row(s) with attack_active=True",
    )

    if is_gaussian:
        rho = attack_info.get("attack_strength_rho")
        checks["gaussian_rho_exactly_1.0"] = (rho == 1.0, f"summary attack_strength_rho={rho}")
    else:
        c = attack_info.get("directional_target_cosine")
        checks["directional_target_cosine_exactly_0.0"] = (c == 0.0, f"summary directional_target_cosine={c}")

        raw_rows = load_raw_csv_rows(run_dir)
        ratios = [
            float(r["delta_attack_norm_ratio"]) for r in raw_rows
            if r["client_id"] == EXPECTED_MALICIOUS_CLIENT and int(r["round_id"]) in EXPECTED_ACTIVE_ROUNDS
            and r.get("delta_attack_norm_ratio") not in (None, "")
        ]
        if ratios:
            max_rel_err = max(abs(x - 1.0) for x in ratios)
            checks["directional_norm_preservation"] = (
                max_rel_err < 1e-6, f"max |delta_attack_norm_ratio - 1| = {max_rel_err:.3e} over {len(ratios)} active rows",
            )
        else:
            checks["directional_norm_preservation"] = (False, "no delta_attack_norm_ratio values found on active rows")

        cosines = [
            float(r["realized_cosine_to_honest"]) for r in raw_rows
            if r["client_id"] == EXPECTED_MALICIOUS_CLIENT and int(r["round_id"]) in EXPECTED_ACTIVE_ROUNDS
            and r.get("realized_cosine_to_honest") not in (None, "")
        ]
        if cosines:
            max_cos_err = max(abs(x - 0.0) for x in cosines)
            checks["realized_directional_cosine_matches_0.0"] = (
                max_cos_err < 1e-6, f"max |realized_cosine_to_honest - 0.0| = {max_cos_err:.3e} over {len(cosines)} active rows",
            )
        else:
            checks["realized_directional_cosine_matches_0.0"] = (False, "no realized_cosine_to_honest values found on active rows")

    checks["T_ref_matches_frozen_default"] = (ft.DEFAULT_T_REF == EXPECTED_T_REF, f"{ft.DEFAULT_T_REF}")
    checks["tau_matches_frozen_default"] = (ft.DEFAULT_THRESHOLD == EXPECTED_THRESHOLD, f"{ft.DEFAULT_THRESHOLD}")
    checks["P_matches_frozen_default"] = (ft.DEFAULT_PERSISTENCE == EXPECTED_PERSISTENCE, f"{ft.DEFAULT_PERSISTENCE}")

    checks["no_prior_result_path_overlap"] = (
        True, f"{run_dir} is under results_ndss/E3_intermittent/, disjoint from all prior experiment roots (asserted at import time)",
    )
    checks["no_ground_truth_used_as_detector_input"] = (
        True,
        "static property of frozen compute_all_scores_v3/apply_persistence_rule_v3 "
        "(verified by code inspection, not a per-run runtime check)",
    )

    failed = [name for name, (passed, _detail) in checks.items() if not passed]
    if failed:
        detail_lines = "\n".join(f"  [FAIL] {name}: {checks[name][1]}" for name in failed)
        raise E3AnalysisError(
            f"{run_key}: {len(failed)} mandatory sanity check(s) failed -- refusing to write "
            f"any output file for this run.\n{detail_lines}"
        )

    return checks


# ---------------------------------------------------------------------------
# Global FedTrace analysis (frozen, unmodified, full trace)
# ---------------------------------------------------------------------------


def analyze_global(run_dir: Path) -> tuple[dict, list]:
    rows = load_trace(run_dir)
    scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
    result = ft.evaluate_detector_v3(
        rows, run_dir.name, COMPONENTS,
        t_ref=EXPECTED_T_REF, threshold=EXPECTED_THRESHOLD, persistence=EXPECTED_PERSISTENCE,
    )
    true_attacker = result.true_malicious_client
    predicted = result.predicted_client
    predicted_result = result.per_client.get(predicted) if predicted else None

    global_summary = {
        "true_attacker": true_attacker,
        "first_true_malicious_round": result.attack_start_round,
        "predicted_attacker": predicted,
        "estimated_onset": predicted_result.onset_estimate if predicted_result else UNAVAILABLE,
        "first_confirmation_round": predicted_result.confirmation_round if predicted_result else UNAVAILABLE,
        "onset_error": result.onset_localization_error if result.onset_localization_error is not None else UNAVAILABLE,
        "confirmation_delay": result.detection_delay if result.detection_delay is not None else UNAVAILABLE,
        "honest_persistent_fp_count": len(result.false_positive_clients),
        "honest_persistent_fp_ids": list(result.false_positive_clients),
        "honest_confirmations": {
            cid: r.confirmation_round for cid, r in result.per_client.items()
            if r.detected and cid != true_attacker
        },
    }
    return global_summary, scored


# ---------------------------------------------------------------------------
# Burst-level and gap-level diagnostics
# ---------------------------------------------------------------------------


def _score_lookup(scored: list, client_id: str) -> dict[int, tuple]:
    return {
        s.round_id: (s.z_m, s.z_t_res, s.combined_score(COMPONENTS))
        for s in scored if s.client_id == client_id
    }


def analyze_bursts(scored: list, client_ids: list[str], true_attacker: str, global_summary: dict) -> list[dict]:
    honest_ids = [c for c in client_ids if c != true_attacker]
    mal_lookup = _score_lookup(scored, true_attacker)
    burst_rows = []
    for start, end in TRUE_WINDOWS:
        per_round = []
        above_threshold_flags = []
        for rid in range(start, end + 1):
            z_m, z_t_res, combined = mal_lookup.get(rid, (UNAVAILABLE, UNAVAILABLE, None))
            s_val = combined[0] if combined else UNAVAILABLE
            above = combined is not None and combined[0] > EXPECTED_THRESHOLD
            above_threshold_flags.append(above)
            per_round.append({"round_id": rid, "z_m": z_m, "z_t_res": z_t_res, "S": s_val, "above_threshold": above})

        first_above = next((p["round_id"] for p in per_round if p["above_threshold"]), UNAVAILABLE)
        local_3_in_a_row = any(
            all(above_threshold_flags[i:i + 3]) for i in range(0, max(0, len(above_threshold_flags) - 2))
        ) if len(above_threshold_flags) >= 3 else False

        conf = global_summary["first_confirmation_round"]
        global_conf_in_window = isinstance(conf, int) and start <= conf <= end

        s_values = [p["S"] for p in per_round if isinstance(p["S"], (int, float))]
        max_mal_s = max(s_values) if s_values else UNAVAILABLE

        max_honest_s = UNAVAILABLE
        for cid in honest_ids:
            lookup = _score_lookup(scored, cid)
            for rid in range(start, end + 1):
                _, _, combined = lookup.get(rid, (None, None, None))
                if combined is not None and (max_honest_s == UNAVAILABLE or combined[0] > max_honest_s):
                    max_honest_s = combined[0]

        burst_rows.append({
            "window_start": start,
            "window_end": end,
            "per_round": per_round,
            "first_above_threshold_round": first_above,
            "local_3_in_a_row_within_burst": local_3_in_a_row,
            "global_confirmation_falls_in_this_window": global_conf_in_window,
            "confirmation_round_if_attributable_to_this_burst": conf if global_conf_in_window else "NOT_APPLICABLE (global confirmation, if any, did not occur in this window)",
            "max_malicious_S_in_burst": max_mal_s,
            "max_honest_S_in_same_rounds": max_honest_s,
        })
    return burst_rows


def analyze_gaps(scored: list, client_ids: list[str], true_attacker: str, global_summary: dict) -> list[dict]:
    honest_ids = [c for c in client_ids if c != true_attacker]
    mal_lookup = _score_lookup(scored, true_attacker)
    gap_rows = []
    for start, end in TRUE_GAPS:
        per_round = []
        for rid in range(start, end + 1):
            z_m, z_t_res, combined = mal_lookup.get(rid, (UNAVAILABLE, UNAVAILABLE, None))
            s_val = combined[0] if combined else UNAVAILABLE
            per_round.append({
                "round_id": rid, "z_m": z_m, "z_t_res": z_t_res, "S": s_val,
                "above_threshold": (combined is not None and combined[0] > EXPECTED_THRESHOLD),
            })

        rounds_above_after_stop = 0
        for p in per_round:
            if p["above_threshold"]:
                rounds_above_after_stop += 1
            else:
                break
        residual_present = per_round[0]["above_threshold"] if per_round else False
        rounds_to_drop_below_tau = rounds_above_after_stop if residual_present else 0

        max_honest_s = UNAVAILABLE
        honest_confirmed_in_gap = []
        for cid in honest_ids:
            lookup = _score_lookup(scored, cid)
            for rid in range(start, end + 1):
                _, _, combined = lookup.get(rid, (None, None, None))
                if combined is not None and (max_honest_s == UNAVAILABLE or combined[0] > max_honest_s):
                    max_honest_s = combined[0]
        for cid, conf_round in global_summary["honest_confirmations"].items():
            if isinstance(conf_round, int) and start <= conf_round <= end:
                honest_confirmed_in_gap.append(cid)

        gap_rows.append({
            "gap_start": start,
            "gap_end": end,
            "per_round": per_round,
            "residual_evidence_present_at_gap_start": residual_present,
            "rounds_for_S_to_drop_below_tau": rounds_to_drop_below_tau if residual_present else "NOT_APPLICABLE (no residual evidence at gap start)",
            "max_honest_S_in_gap": max_honest_s,
            "any_honest_client_persistent_confirmation_in_gap": honest_confirmed_in_gap,
        })
    return gap_rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_e3_summary_csv(results: dict, path: Path) -> None:
    header = [
        "run", "true_attacker", "first_true_malicious_round", "predicted_attacker",
        "estimated_onset", "first_confirmation_round", "onset_error", "confirmation_delay",
        "honest_persistent_fp_count", "honest_persistent_fp_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, r in results.items():
            g = r["global"]
            writer.writerow([
                key, g["true_attacker"], g["first_true_malicious_round"], g["predicted_attacker"],
                g["estimated_onset"], g["first_confirmation_round"], g["onset_error"], g["confirmation_delay"],
                g["honest_persistent_fp_count"], ";".join(g["honest_persistent_fp_ids"]),
            ])


def write_burst_diagnostics_csv(results: dict, path: Path) -> None:
    header = [
        "run", "window_start", "window_end", "round_id", "z_m", "z_t_res", "S", "above_threshold",
        "first_above_threshold_round", "local_3_in_a_row_within_burst",
        "global_confirmation_falls_in_this_window", "max_malicious_S_in_burst", "max_honest_S_in_same_rounds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, r in results.items():
            for burst in r["bursts"]:
                for p in burst["per_round"]:
                    writer.writerow([
                        key, burst["window_start"], burst["window_end"], p["round_id"], p["z_m"], p["z_t_res"],
                        p["S"], p["above_threshold"], burst["first_above_threshold_round"],
                        burst["local_3_in_a_row_within_burst"], burst["global_confirmation_falls_in_this_window"],
                        burst["max_malicious_S_in_burst"], burst["max_honest_S_in_same_rounds"],
                    ])


def write_gap_diagnostics_csv(results: dict, path: Path) -> None:
    header = [
        "run", "gap_start", "gap_end", "round_id", "z_m", "z_t_res", "S", "above_threshold",
        "residual_evidence_present_at_gap_start", "rounds_for_S_to_drop_below_tau",
        "max_honest_S_in_gap", "any_honest_client_persistent_confirmation_in_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, r in results.items():
            for gap in r["gaps"]:
                for p in gap["per_round"]:
                    writer.writerow([
                        key, gap["gap_start"], gap["gap_end"], p["round_id"], p["z_m"], p["z_t_res"],
                        p["S"], p["above_threshold"], gap["residual_evidence_present_at_gap_start"],
                        gap["rounds_for_S_to_drop_below_tau"], gap["max_honest_S_in_gap"],
                        ";".join(gap["any_honest_client_persistent_confirmation_in_gap"]),
                    ])


def write_e3_summary_json(results: dict, sanity: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "sanity_checks": sanity}, handle, indent=2, default=str)


def write_e3_run_log(results: dict, sanity: dict, path: Path) -> None:
    lines = ["NDSS E3 post-processing run log", "=" * 40, ""]
    for key, r in results.items():
        lines.append(f"--- {key} ---")
        lines.append(f"  global: {r['global']}")
        lines.append("  sanity checks:")
        for name, (passed, detail) in sanity[key].items():
            lines.append(f"    [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(results: dict, path: Path) -> None:
    """Two panels (Gaussian intermittent, Directional intermittent):
    malicious-client S over all 50 rounds, tau line, true windows lightly
    shaded (gaps left unshaded -- never visually implied to be malicious),
    max-honest-S as a subdued reference curve. No significance/
    generalization claim in the title."""
    from generate_paper_figures import MALICIOUS_COLOR, HONEST_COLOR, THRESHOLD_COLOR  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, key, title in zip(axes, ("gaussian", "directional"), ("Gaussian (intermittent)", "Directional (intermittent)")):
        r = results[key]
        rows = load_trace(E3_RUN_DIRS[key])
        scored = ft.compute_all_scores_v3(rows, t_ref=EXPECTED_T_REF)
        attacker = r["global"]["true_attacker"]
        client_ids = sorted({row["client_id"] for row in rows})
        honest_ids = [c for c in client_ids if c != attacker]

        mal_lookup = _score_lookup(scored, attacker)
        xs = sorted(mal_lookup.keys())
        ys = [mal_lookup[x][2][0] if mal_lookup[x][2] else None for x in xs]
        ax.plot(xs, ys, color=MALICIOUS_COLOR, lw=1.8, marker="^", ms=3, label=f"{attacker} (malicious)")

        max_honest_per_round = {}
        for cid in honest_ids:
            lookup = _score_lookup(scored, cid)
            for rid, (_, _, combined) in lookup.items():
                if combined is None:
                    continue
                if rid not in max_honest_per_round or combined[0] > max_honest_per_round[rid]:
                    max_honest_per_round[rid] = combined[0]
        hx = sorted(max_honest_per_round.keys())
        hy = [max_honest_per_round[x] for x in hx]
        ax.plot(hx, hy, color=HONEST_COLOR, lw=1.0, alpha=0.6, label="max honest client S (reference)")

        for start, end in TRUE_WINDOWS:
            ax.axvspan(start - 0.5, end + 0.5, color=MALICIOUS_COLOR, alpha=0.10, lw=0)
        ax.axhline(EXPECTED_THRESHOLD, color=THRESHOLD_COLOR, lw=1.0, ls=":", label=r"$\tau=3$")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("communication round")
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel(r"$S_{k,r}$")
    fig.suptitle(
        "NDSS E3: FedTrace combined score under intermittent attack windows\n"
        "(shaded = true malicious windows 20-24/30-34/40-44; single seed, descriptive only)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for key, run_dir in E3_RUN_DIRS.items():
        if not run_dir.exists():
            raise E3AnalysisError(f"Expected E3 run directory not found: {run_dir}")

    sanity = {}
    results = {}
    for key, run_dir in E3_RUN_DIRS.items():
        rows = load_trace(run_dir)
        summary = load_summary(run_dir)
        sanity[key] = run_mandatory_sanity_checks(key, run_dir, rows, summary, is_gaussian=(key == "gaussian"))

        global_summary, scored = analyze_global(run_dir)
        client_ids = sorted({r["client_id"] for r in rows})
        bursts = analyze_bursts(scored, client_ids, global_summary["true_attacker"], global_summary)
        gaps = analyze_gaps(scored, client_ids, global_summary["true_attacker"], global_summary)
        results[key] = {"global": global_summary, "bursts": bursts, "gaps": gaps}

    E3_ROOT.mkdir(parents=True, exist_ok=True)
    write_e3_summary_csv(results, E3_ROOT / "e3_summary.csv")
    write_e3_summary_json(results, sanity, E3_ROOT / "e3_summary.json")
    write_burst_diagnostics_csv(results, E3_ROOT / "e3_burst_diagnostics.csv")
    write_gap_diagnostics_csv(results, E3_ROOT / "e3_gap_diagnostics.csv")
    write_e3_run_log(results, sanity, E3_ROOT / "e3_run_log.txt")
    make_figure(results, E3_ROOT / "e3_intermittent_scores.png")

    print(f"Wrote outputs under {E3_ROOT}")


if __name__ == "__main__":
    main()
