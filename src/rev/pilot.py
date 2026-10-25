"""Analysis of a matched clean / attacked pair (Stage C pilot). Read-only on run outputs.

    python -m src.rev.pilot --clean X1_clean_d123 --attacked X2_badnets_d123 \
        --malicious client_002 --onset 20 --target-class 0 --trigger-size 3

Writes ``results_rev/pilot/<attacked>__vs__<clean>/``:

    prefix.json            matched-prefix validity: logged global states (initial + rounds < onset), every
                           submitted update of rounds < onset, and all trace / history rows of rounds < onset must be
                           bit-identical; otherwise the pair is INVALID as a scientific comparison
    effect_<run>.csv       per logged global state: clean test top-1 and attack success rate (ASR) on the triggered
                           test view (non-target-class test images only; denominator reported)
    outcome_<run>.json     online FedTrace v3 events and the run outcome (src.rev.metrics.run_outcome); the clean run
                           is scored with no malicious client, so any confirmed client is an honest false alarm
    trajectories_<run>.csv per-client scores: causal (online) scores from t_ref on; offline scores of calibration
                           rounds are included but flagged ``retrospective`` (they use future rounds)
    summary.json           paired quantities (final top-1 difference, final ASR in both arms) and pointers

Nothing here tunes parameters; thresholds stay those of the frozen detector.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from src.analysis.fedtrace_detector import compute_all_scores_v3, load_trace_rows
from src.rev.attacks import PatchTrigger, TriggeredEvaluationDataset, cifar100_white_patch
from src.rev.hashing import load_tensor_state
from src.rev.metrics import attack_success_counts, predict, run_outcome
from src.rev.online import OnlineConfig, causal_scores_v3, online_detect_v3
from src.rev.replay import verify_update_log
from src.rev.update_log import read_manifest


def _index(log: Path) -> tuple[dict[int, str], dict[tuple[int, str], str]]:
    globals_, submitted = {}, {}
    for r in read_manifest(log):
        if r.get("kind") == "global":
            globals_[r["round_id"]] = r["content_sha256"]
        elif r.get("kind") == "submitted":
            submitted[(r["round_id"], r["client_id"])] = r["content_sha256"]
    return globals_, submitted


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matched_prefix(clean_log: Path, attacked_log: Path, onset: int, clean_results: Path | None = None,
                   attacked_results: Path | None = None, check_content: bool = False) -> dict[str, Any]:
    """Bit-identity of everything before the onset round; first differences after it."""
    problems: list[str] = []
    for name, log in (("clean", clean_log), ("attacked", attacked_log)):
        report = verify_update_log(log, check_content=check_content)
        if not report.ok:
            problems.append(f"{name} log failed verification: {report.problems[:3]}")
    g_c, s_c = _index(clean_log)
    g_a, s_a = _index(attacked_log)
    pre_globals = [r for r in sorted(g_c) if r < onset]
    pre_subs = sorted(k for k in s_c if k[0] < onset)
    differing_globals = [r for r in pre_globals if g_a.get(r) != g_c[r]]
    differing_subs = [k for k in pre_subs if s_a.get(k) != s_c[k]]
    if set(k for k in s_a if k[0] < onset) != set(pre_subs):
        problems.append("different sets of pre-onset submissions")
    rows_checked = 0
    for name in ("client_round_trace.csv", "round_history.csv"):
        if clean_results is None or attacked_results is None:
            break
        a, b = _csv_rows(clean_results / name), _csv_rows(attacked_results / name)
        a = [r for r in a if int(r["round_id"]) < onset]
        b = [r for r in b if int(r["round_id"]) < onset]
        if name == "round_history.csv":  # wall-clock columns are not training outputs
            a = [{k: v for k, v in r.items() if k not in ("round_seconds", "cumulative_seconds")} for r in a]
            b = [{k: v for k, v in r.items() if k not in ("round_seconds", "cumulative_seconds")} for r in b]
        rows_checked += len(a)
        if a != b:
            problems.append(f"{name}: pre-onset rows differ")
    all_rounds = sorted(set(g_c) | set(g_a))
    first_global = next((r for r in all_rounds if g_c.get(r) != g_a.get(r)), None)
    first_sub = min((k for k in set(s_c) | set(s_a) if s_c.get(k) != s_a.get(k)), default=None)
    honest_at_onset = sorted(c for (r, c) in s_c if r == onset and s_a.get((r, c)) == s_c[(r, c)])
    valid = not (problems or differing_globals or differing_subs)
    return {
        "valid_matched_pair": valid, "onset": onset, "problems": problems,
        "pre_onset_globals_checked": len(pre_globals), "pre_onset_submissions_checked": len(pre_subs),
        "pre_onset_csv_rows_checked": rows_checked, "differing_pre_onset_globals": differing_globals,
        "differing_pre_onset_submissions": [list(k) for k in differing_subs],
        "first_differing_global_round": first_global,
        "first_differing_submission": list(first_sub) if first_sub else None,
        "submissions_identical_at_onset": honest_at_onset,
    }


def per_round_effect(log: Path, test_dataset: Dataset, trigger: PatchTrigger, target_class: int,
                     model_fn: Callable[[], nn.Module], device: str = "cpu", batch_size: int = 512,
                     rounds: list[int] | None = None) -> list[dict[str, Any]]:
    """Clean top-1 and ASR of every logged global state (``round_id`` -1 = initialisation)."""
    labels = [int(t) for t in test_dataset.targets]  # type: ignore[attr-defined]
    view = TriggeredEvaluationDataset(test_dataset, trigger=trigger, target_class=target_class, labels=labels)
    records = sorted((r for r in read_manifest(log) if r.get("kind") == "global"), key=lambda r: r["round_id"])
    out = []
    model = model_fn().to(device)
    for record in records:
        if rounds is not None and record["round_id"] not in rounds:
            continue
        model.load_state_dict(load_tensor_state(log / record["file"]))
        preds = predict(model, test_dataset, device, batch_size)
        clean = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        non_target = [p for p, y in zip(preds, labels) if y != target_class]
        untriggered_target = sum(int(p == target_class) for p in non_target) / len(non_target)
        asr = attack_success_counts(predict(model, view, device, batch_size), target_class)
        out.append({"round_id": record["round_id"], "clean_top1": clean, "untriggered_target_rate": untriggered_target,
                    "asr": asr.rate, "asr_k": asr.k,
                    "asr_n": asr.n, "global_content_sha256": record["content_sha256"]})
    return out


def forensic_outcome(trace: Path, run_id: str, seed: int, malicious: str | None, onset: int | None,
                     config: OnlineConfig | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = config or OnlineConfig()
    rows = load_trace_rows(trace)
    detection = online_detect_v3(rows, config)
    outcome = run_outcome(run_id, seed, detection, malicious, onset)
    events = {c: asdict(e) for c, e in detection.per_client.items()}
    trajectories = []
    for s in compute_all_scores_v3(rows, t_ref=config.t_ref):
        if s.round_id < config.t_ref:
            combined = s.combined_score(config.components)
            trajectories.append({"round_id": s.round_id, "client_id": s.client_id, "z_m": s.z_m, "z_t_res": s.z_t_res,
                                 "score": combined[0] if combined else None, "kind": "retrospective"})
    for s in causal_scores_v3(rows, config):
        combined = s.combined_score(config.components)
        trajectories.append({"round_id": s.round_id, "client_id": s.client_id, "z_m": s.z_m, "z_t_res": s.z_t_res,
                             "score": combined[0] if combined else None, "kind": "causal"})
    trajectories.sort(key=lambda r: (r["client_id"], r["round_id"]))
    return {"outcome": asdict(outcome), "events": events, "config": asdict(config),
            "retrospective_onset": detection.retrospective_onset}, trajectories


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matched clean/attacked pair analysis (read-only).")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--attacked", required=True)
    parser.add_argument("--malicious", required=True)
    parser.add_argument("--onset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--skip-effect", action="store_true", help="skip the per-round GPU/CPU evaluation")
    parser.add_argument("--out-dir", help="output directory (default results_rev/pilot/<attacked>__vs__<clean>); "
                        "must not already contain results")
    args = parser.parse_args(argv)

    root = Path(".")
    res = {name: root / "results_rev" / name for name in (args.clean, args.attacked)}
    logs = {name: root / "logs_rev" / name / "update_log" for name in (args.clean, args.attacked)}
    out = Path(args.out_dir) if args.out_dir else root / "results_rev" / "pilot" / f"{args.attacked}__vs__{args.clean}"
    if args.out_dir and out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite existing results in {out.as_posix()}")
    out.mkdir(parents=True, exist_ok=True)

    prefix = matched_prefix(logs[args.clean], logs[args.attacked], args.onset, res[args.clean], res[args.attacked])
    (out / "prefix.json").write_text(json.dumps(prefix, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: prefix[k] for k in ("valid_matched_pair", "first_differing_global_round",
                                             "first_differing_submission", "problems")}))
    summary: dict[str, Any] = {"prefix_valid": prefix["valid_matched_pair"]}
    for name, malicious in ((args.clean, None), (args.attacked, args.malicious)):
        result, trajectories = forensic_outcome(res[name] / "client_round_trace.csv", name, args.seed, malicious,
                                                args.onset if malicious else None)
        (out / f"outcome_{name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        _write_csv(out / f"trajectories_{name}.csv", trajectories)
        summary[f"outcome_{name}"] = {k: result["outcome"][k] for k in (
            "detected", "attribution_correct", "alarm_round", "actionable_round", "estimated_onset",
            "onset_abs_error", "delay", "delay_censored", "honest_alarm_clients", "attributed_client")}
        summary[f"final_test_top1_{name}"] = json.loads((res[name] / "final_test_metrics.json").read_text())["top1"]
    summary["paired_final_top1_attacked_minus_clean"] = (summary[f"final_test_top1_{args.attacked}"]
                                                        - summary[f"final_test_top1_{args.clean}"])
    if not args.skip_effect:
        from src.data.cifar100 import build_datasets
        from src.models.resnet18_cifar import resnet18_cifar

        _, test = build_datasets("data", download=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trigger = cifar100_white_patch(args.trigger_size)
        for name in (args.clean, args.attacked):
            rows = per_round_effect(logs[name], test, trigger, args.target_class,
                                    lambda: resnet18_cifar(num_classes=100), device)
            _write_csv(out / f"effect_{name}.csv", rows)
            summary[f"final_asr_{name}"] = rows[-1]["asr"]
            summary[f"asr_denominator_{name}"] = rows[-1]["asr_n"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if prefix["valid_matched_pair"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
