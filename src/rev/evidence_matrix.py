"""Development evidence matrix: clean / BadNets / relabel-only / benign drift x seeds 123, 2026 (read-only, CPU).

    python -m src.rev.evidence_matrix --seeds 123 2026 [--with-logs]

CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. Reads sealed traces, pair analyses and (with --with-logs)
update logs; writes only ``results_rev/pilot/evidence_matrix/``. Nothing is fitted: no threshold, weight or classifier
is estimated from these outcomes. Detector thresholds are the Phase S values (1.75, 11.75) and frozen v3.

Two kinds of quantities are kept apart:
    deployable   computable online from submitted updates and the client's own history (no counterfactual);
    diagnostic   uses the matched clean counterfactual run (explanation only, never a detector input).

Definitions (diagnostic):
    norm_ratio_r25_35              mean over rounds 25-35 of ||update|| intervened / matched clean, intervened client.
    self_cos_change_r21_29         mean over rounds 21-29 of self-cosine intervened - matched clean, intervened client.
    signed_residual_change_r21_29  mean over rounds 21-29 of the signed calibrated v3 residual z, intervened - clean.
    reference_movement_on_others   per other client k, mean over rounds 20-29 (protocol 13, E6) of
                                   [med(t_a) - med(t_a with the intervened client's t replaced by its clean value)] / sd_k,
                                   med = inclusive 5-client median, sd_k = client k's clean calibration sd (rounds < 20).
                                   Positive = the intervened client pushed k's reference up (raises k's v3 residual).
Accuracy: ``final_top1`` is the training-time final test evaluation (final_test_metrics.json); ``pilot_top1_r49`` is
the pilot's re-evaluation of the round-49 global model; both are reported, and the change uses the former.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev import phase_s as ps
from src.rev.diagnostics import norm_ratios
from src.rev.online import online_detect_v3
from src.rev.reference_audit import clients_of, decompose, inclusive_median, update_geometry
from src.rev.relabel_compare import self_cos_change, signed_residual_change

CLIENT, ONSET = "client_002", 20
CONDITIONS = {"clean": ("X1_clean_d{s}", False), "badnets": ("X2_badnets_d{s}", True),
              "relabel": ("X3_relabel_d{s}", True), "drift": ("X4_drift_d{s}", False)}
CONFIRMATION_CONDITIONS = {"clean": ("{s}_clean", False), "badnets": ("{s}_badnets", True), "drift": ("{s}_drift", False)}
FIXED = {"ZP": 1.75, "CUSUM": 11.75}
NORM_ROUNDS, MOVE_ROUNDS = range(25, 36), range(ONSET, ONSET + 10)


def _pair_dir(root: Path, seed: str, condition: str, conditions=CONDITIONS) -> Path | None:
    run = conditions[condition][0].format(s=seed)
    clean = conditions["clean"][0].format(s=seed)
    for d in (root / "results_rev/pilot/reanalysis_target_rate" / f"{run}__vs__{clean}",
              root / "results_rev/pilot" / f"{run}__vs__{clean}"):
        if (d / "summary.json").exists():
            return d
    return None


def _final_effect(pair: Path | None, run: str) -> dict[str, float] | None:
    """Last row of the pair's effect table (round 49); None if the pair analysis is absent (optional artifact)."""
    if pair is None or not (pair / f"effect_{run}.csv").exists():
        return None
    last = list(csv.DictReader((pair / f"effect_{run}.csv").open(encoding="utf-8")))[-1]
    out: dict[str, float] = {"round_id": float(last["round_id"])}
    out.update({k: float(last[k]) for k in ("asr", "untriggered_target_rate", "clean_top1") if last.get(k) not in (None, "")})
    return out


def _decisions(rows, malicious: str | None) -> dict[str, Any]:
    out = {}
    online = online_detect_v3(rows)
    alarms = {c: (e.alarm_round, e.estimated_onset, e.streak_mean_score_at_alarm or 0.0)
              for c, e in online.per_client.items() if e.alarm_round is not None}
    for name, a in (("v3", alarms),
                    ("zp_norm", ps.q_alarms(rows, "ZP", "m", FIXED["ZP"])[0]),
                    ("cusum_norm", ps.q_alarms(rows, "CUSUM", "m", FIXED["CUSUM"])[0])):
        ev = ps.evaluate_run(a, malicious, ONSET if malicious else None)
        out[name] = {"alarms": ev["alarm_rounds"], "detected_any": bool(ev["alarm_rounds"]),
                     "first_attributed": ev["attributed"], "honest_false_accusations": ev["honest_false_alarms"],
                     "n_honest_false_accusations": ev["n_honest_false_alarms"],
                     "attacker_detected": ev.get("attacker_detected"), "attribution_correct": ev.get("attribution_correct")}
    out["v3"]["online_wrapper_attributed"] = online.attributed_client  # consistency check against the scoring order
    return out


def _mean_or_none(values: list[float | None], n: int) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if len(vals) == n else None


def classifier_concentration(log: Path, client: str, rounds, num_classes: int = 100) -> dict[int, dict[str, float]]:
    """Per round: share of the classifier-weight update norm carried by its largest class row (and that class),
    and the energy concentration sum_c (||W_c||^2 / ||W||^2)^2. Deployable: uses only the submitted update."""
    out = {}
    for r, delta in update_geometry(log, client, rounds).items():
        w = delta["fc.weight"].reshape(num_classes, -1)
        row = w.norm(dim=1)
        energy = (row ** 2) / (row ** 2).sum()
        out[r] = {"max_row_share": float(row.max() / w.norm()), "max_row_class": int(row.argmax()),
                  "energy_concentration": float((energy ** 2).sum())}
    return out


def _diagnostic(clean_rows, rows) -> dict[str, Any]:
    ratios = norm_ratios(clean_rows, rows).get(CLIENT, {})
    missing = sorted({(r["round_id"], r["client_id"]) for x in (clean_rows, rows) for r in x
                      if r["round_id"] in MOVE_ROUNDS and r.get("self_prev_round_cosine") is None})
    if missing:  # the decomposition needs every client's self-cosine in both runs: undefined, not imputed
        movement, undefined = {}, sorted({c for c in clients_of(rows) if c != CLIENT})
    else:
        dec = decompose(clean_rows, rows, CLIENT, MOVE_ROUNDS, inclusive_median)
        movement = {k: statistics.mean(x["attacker_induced"] for x in v) for k, v in dec.items() if k != CLIENT and v}
        undefined = sorted(k for k, v in dec.items() if k != CLIENT and not v)
    signed = max(movement.items(), key=lambda kv: (kv[1], kv[0])) if movement else None
    absolute = max(movement.items(), key=lambda kv: (abs(kv[1]), kv[0])) if movement else None
    return {
        "norm_ratio_r25_35": _mean_or_none([ratios.get(r) for r in NORM_ROUNDS], len(NORM_ROUNDS)),
        "self_cos_change_r21_29": self_cos_change(clean_rows, rows),
        "signed_residual_change_r21_29": signed_residual_change(clean_rows, rows),
        "reference_movement_on_others": movement,
        "reference_movement_undefined_clients": undefined,  # calibration sd undefined or zero, or missing self-cosine
        "reference_movement_missing_inputs": [list(m) for m in missing],
        "reference_movement_max_signed": {"client": signed[0], "value": signed[1]} if signed else None,
        "reference_movement_max_abs": {"client": absolute[0], "value": absolute[1]} if absolute else None,
    }


def _trajectory(clean_rows, rows) -> dict[int, dict[str, float | None]]:
    ratios = norm_ratios(clean_rows, rows).get(CLIENT, {})
    tc = {r["round_id"]: r.get("self_prev_round_cosine") for r in clean_rows if r["client_id"] == CLIENT}
    tx = {r["round_id"]: r.get("self_prev_round_cosine") for r in rows if r["client_id"] == CLIENT}
    return {r: {"norm_ratio": ratios.get(r),
                "self_cos_change": tx[r] - tc[r] if tx.get(r) is not None and tc.get(r) is not None else None}
            for r in range(ONSET, 36)}


def build(root: Path, seeds, with_logs: bool, out_dir: Path | None = None, conditions=CONDITIONS) -> dict[str, Any]:
    """``conditions``: development arms (default) or CONFIRMATION_CONDITIONS (protocol 26; no relabel-only arm)."""
    effect_pair = "relabel" if "relabel" in conditions else "badnets"  # pair whose effect table includes the clean run
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY", "rows": {}, "trajectories": {}}
    for s in seeds:
        rows = {c: load_trace_rows(root / "results_rev" / v[0].format(s=s) / "client_round_trace.csv") for c, v in conditions.items()}
        clean_run = conditions["clean"][0].format(s=s)
        clean_top1 = json.loads((root / "results_rev" / clean_run / "final_test_metrics.json").read_text())["top1"]
        for cond, (pattern, malicious) in conditions.items():
            run = pattern.format(s=s)
            entry: dict[str, Any] = {"run": run, "condition": cond, "malicious": malicious,
                                     "malicious_client": CLIENT if malicious else None,
                                     "intervened_client": None if cond == "clean" else CLIENT,
                                     "onset": None if cond == "clean" else ONSET}
            if cond != "clean":
                entry["diagnostic"] = _diagnostic(rows["clean"], rows[cond])
                report["trajectories"][f"{s}/{cond}"] = _trajectory(rows["clean"], rows[cond])
            entry["decisions"] = _decisions(rows[cond], CLIENT if malicious else None)
            top1 = json.loads((root / "results_rev" / run / "final_test_metrics.json").read_text())["top1"]
            entry["final_top1"] = top1
            entry["top1_change_vs_clean_pp"] = 100 * (top1 - clean_top1)
            # the clean run's effect table is read from the relabel pair (same matched clean run, same trigger)
            effect = _final_effect(_pair_dir(root, s, cond if cond != "clean" else effect_pair, conditions), run)
            entry["final_effect"] = effect
            entry["pilot_top1_r49"] = effect.get("clean_top1") if effect else None
            report["rows"][f"{s}/{cond}"] = entry
            if with_logs:
                conc = classifier_concentration(root / "logs_rev" / run / "update_log", CLIENT, range(15, 31))
                pre = [conc[r]["max_row_share"] for r in range(15, 20)]
                post = [conc[r]["max_row_share"] for r in range(21, 30)]
                entry["deployable_classifier_concentration"] = {
                    "pre_onset_mean_max_row_share_r15_19": statistics.mean(pre),
                    "post_mean_max_row_share_r21_29": statistics.mean(post),
                    "post_minus_pre": statistics.mean(post) - statistics.mean(pre),
                    "post_dominant_classes": sorted({conc[r]["max_row_class"] for r in range(21, 30)}),
                    "post_mean_energy_concentration": statistics.mean(conc[r]["energy_concentration"] for r in range(21, 30)),
                }
    out = out_dir or root / "results_rev" / "pilot" / "evidence_matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence_matrix.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def reconcile(report: dict[str, Any], root: Path) -> list[str]:
    """Cross-check against previously written reports (Phase S, benign drift); returns mismatch descriptions."""
    issues: list[str] = []
    phase = root / "results_rev/pilot/phase_s/phase_s.json"
    if phase.exists():
        methods = json.loads(phase.read_text())["methods"]
        for key, entry in report["rows"].items():
            if entry["condition"] == "drift":
                continue
            for name, method in (("v3", "V-inclusive_median"), ("zp_norm", "Q-ZP-m"), ("cusum_norm", "Q-CUSUM-m")):
                saved = methods.get(method, {}).get(key)
                if saved is not None and (saved["attributed"] != entry["decisions"][name]["first_attributed"]
                                          or saved["honest_false_alarms"] != entry["decisions"][name]["honest_false_accusations"]):
                    issues.append(f"{key} {name}: phase_s {saved['attributed']}/{saved['honest_false_alarms']}")
    drift = root / "results_rev/pilot/benign_drift/benign_drift.json"
    if drift.exists():
        per_seed = json.loads(drift.read_text())["per_seed"]
        for s, saved in per_seed.items():
            entry = report["rows"].get(f"{s}/drift")
            if entry is None:
                continue
            d = entry["decisions"]
            if sorted(saved["E1_v3_confirmed_clients"]) != sorted(d["v3"]["alarms"]):
                issues.append(f"{s}/drift v3 confirmations")
            if saved["E2_zp_norm"]["confirmed_clients"] != sorted(d["zp_norm"]["alarms"]):
                issues.append(f"{s}/drift zp confirmations")
            if saved["E3_cusum_norm"]["confirmed_clients"] != sorted(d["cusum_norm"]["alarms"]):
                issues.append(f"{s}/drift cusum confirmations")
            for k, v in saved["E6_reference_movement_from_drifted_client"].items():
                if abs(v - entry["diagnostic"]["reference_movement_on_others"][k]) > 1e-12:
                    issues.append(f"{s}/drift reference movement {k}")
            if abs(saved["E7_final_top1"]["drift"] - entry["final_top1"]) > 1e-12:
                issues.append(f"{s}/drift top1")
    for key, entry in report["rows"].items():
        v3 = entry["decisions"]["v3"]
        if v3["online_wrapper_attributed"] != v3["first_attributed"]:
            issues.append(f"{key} v3 scoring order differs from the online wrapper")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Development evidence matrix (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    parser.add_argument("--with-logs", action="store_true")
    parser.add_argument("--confirmation", action="store_true", help="protocol 26 runs W1 / W2 (separate output, no reconciliation)")
    args = parser.parse_args(argv)
    if args.confirmation:
        out = Path("results_rev/pilot/evidence_matrix_confirmation")
        seeds = args.seeds if args.seeds != ["123", "2026"] else ["W1", "W2"]
        build(Path("."), seeds, args.with_logs, out, CONFIRMATION_CONDITIONS)
        print(f"written {out / 'evidence_matrix.json'}")
        return 0
    report = build(Path("."), args.seeds, args.with_logs)
    issues = reconcile(report, Path("."))
    print("written results_rev/pilot/evidence_matrix/evidence_matrix.json")
    print("reconciliation:", "OK" if not issues else issues)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
