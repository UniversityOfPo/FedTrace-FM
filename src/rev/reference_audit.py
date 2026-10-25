"""Read-only audit of same-round reference contamination (CPU; sealed traces and update logs only).

    python -m src.rev.reference_audit --seeds 123 2026 [--with-logs]

POST-HOC EXPLORATORY, DEVELOPMENT ONLY. Writes ``results_rev/pilot/reference_audit_posthoc/``.

Reference constructions for the temporal residual d(k, r) = ref(k, r) - t(k, r), where t is client k's cosine to
its previous submitted update (all are computable online from round-r submissions and k's own history):

    inclusive_median     median over all clients of round r, including k (frozen v3)
    leave_one_out_median median over the other clients of round r
    trimmed_mean         mean of round r after dropping the minimum and maximum (all clients, including k)
    historical_own_mean  k's own mean of t over rounds [1, T_REF) (no cross-client input)

Each residual is standardised with k's own calibration over rounds [1, T_REF) (v3's convention) and combined with
the unchanged v3 magnitude score z_m under the frozen persistence rule (tau 3, P 3). No threshold is tuned.

The matched clean counterfactual is used only by ``decompose`` (diagnosis); it is never a detector input. Logs are
opened read-only; nothing under ``logs_rev/`` or ``results_rev/<run>/`` is modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.analysis.fedtrace_detector import ScoredRowV3, apply_persistence_rule_v3, load_trace_rows
from src.rev.hashing import load_tensor_state
from src.rev.online import OnlineConfig, causal_scores_v3
from src.rev.update_log import read_manifest

T_REF = 20
CONFIG = OnlineConfig()
Reference = Callable[[Mapping[tuple[int, str], float | None], int, str, Sequence[str]], float | None]


def self_cosines(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float | None]:
    return {(int(r["round_id"]), str(r["client_id"])): r.get("self_prev_round_cosine") for r in rows}


def clients_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r["client_id"]) for r in rows})


def _round_values(t, rnd, clients, exclude=None):
    return [t[(rnd, c)] for c in clients if c != exclude and t.get((rnd, c)) is not None]


def inclusive_median(t, rnd, k, clients):
    vals = _round_values(t, rnd, clients)
    return statistics.median(vals) if len(vals) >= 2 else None


def leave_one_out_median(t, rnd, k, clients):
    vals = _round_values(t, rnd, clients, exclude=k)
    return statistics.median(vals) if len(vals) >= 2 else None


def trimmed_mean(t, rnd, k, clients):
    vals = sorted(_round_values(t, rnd, clients))
    return statistics.mean(vals[1:-1]) if len(vals) >= 3 else None


def historical_own_mean(t, rnd, k, clients, t_ref: int = T_REF):
    vals = [t[(r, k)] for r in range(1, t_ref) if t.get((r, k)) is not None]
    return statistics.mean(vals) if len(vals) >= 2 else None


REFERENCES: dict[str, Reference] = {
    "inclusive_median": inclusive_median,
    "leave_one_out_median": leave_one_out_median,
    "trimmed_mean": trimmed_mean,
    "historical_own_mean": historical_own_mean,
}


def residuals(rows: Sequence[Mapping[str, Any]], reference: Reference, t_ref: int = T_REF) -> dict[str, dict[int, dict[str, float]]]:
    """Per client and round: d = ref - t, signed z with the client's own calibration over rounds [1, t_ref)."""
    t = self_cosines(rows)
    clients = clients_of(rows)
    rounds = sorted({r for r, _ in t})
    out: dict[str, dict[int, dict[str, float]]] = {}
    for k in clients:
        d = {}
        for r in rounds:
            ref = reference(t, r, k, clients)
            if t.get((r, k)) is not None and ref is not None:
                d[r] = ref - t[(r, k)]
        cal = [v for r, v in d.items() if 1 <= r < t_ref]
        if len(cal) < 2 or statistics.stdev(cal) == 0:
            out[k] = {}
            continue
        mu, sd = statistics.mean(cal), statistics.stdev(cal)
        out[k] = {r: {"d": v, "z": (v - mu) / sd, "cal_mu": mu, "cal_sd": sd} for r, v in d.items()}
    return out


def detect(rows: Sequence[Mapping[str, Any]], res: Mapping[str, Mapping[int, Mapping[str, float]]],
           last_round: int | None = None) -> dict[str, Any]:
    """Frozen v3 persistence on max(z_m, max(0, z_ref)) from T_REF; online tie-break (streak mean up to the alarm)."""
    clients = clients_of(rows)
    last = last_round if last_round is not None else max(int(r["round_id"]) for r in rows)
    zm = {(s.round_id, s.client_id): s.z_m for s in causal_scores_v3(rows, CONFIG)}
    scored = [ScoredRowV3(round_id=r, client_id=k, z_m=zm.get((r, k)),
                          z_t_res=max(0.0, res[k][r]["z"]) if r in res.get(k, {}) else None)
              for k in clients for r in range(T_REF, last + 1)]
    per = apply_persistence_rule_v3(scored, ("m", "t_res"), CONFIG.threshold, CONFIG.persistence)
    score = {(s.round_id, s.client_id): (s.combined_score(("m", "t_res")) or (None,))[0] for s in scored}
    alarms = {k: (p.confirmation_round, p.onset_estimate) for k, p in per.items() if p.confirmation_round is not None}
    ranked = sorted(alarms, key=lambda k: (alarms[k][0],
                                           -statistics.mean(score[(r, k)] for r in range(alarms[k][1], alarms[k][0] + 1)), k))
    return {"alarms": alarms, "attributed": ranked[0] if ranked else None, "ranking": ranked}


def decompose(clean: Sequence[Mapping[str, Any]], attacked: Sequence[Mapping[str, Any]], attacker: str,
              rounds: Iterable[int], reference: Reference = inclusive_median) -> dict[str, list[dict[str, float]]]:
    """DIAGNOSIS ONLY. Per client and round, in units of the client's calibration sd (identical in both arms):

        delta_z          = [(ref_a - t_a) - (ref_c - t_c)] / sd
        from_reference   = (ref_a - ref_c) / sd
        from_own_update  = -(t_a - t_c) / sd                       (delta_z = from_reference + from_own_update)
        attacker_induced = (ref_a - ref_a[attacker's t replaced by its clean value]) / sd
    """
    tc, ta = self_cosines(clean), self_cosines(attacked)
    clients = clients_of(attacked)
    cal = residuals(clean, reference)
    out: dict[str, list[dict[str, float]]] = {}
    for k in clients:
        if not cal.get(k):  # calibration sd undefined or zero (e.g. the median client with d == 0): z undefined, as in v3
            out[k] = []
            continue
        sd = next(iter(cal[k].values()))["cal_sd"]
        rows = []
        for r in rounds:
            ref_a, ref_c = reference(ta, r, k, clients), reference(tc, r, k, clients)
            swapped = dict(ta)
            swapped[(r, attacker)] = tc[(r, attacker)]
            ref_swap = reference(swapped, r, k, clients)
            rows.append({
                "round": r,
                "delta_z": ((ref_a - ta[(r, k)]) - (ref_c - tc[(r, k)])) / sd,
                "from_reference": (ref_a - ref_c) / sd,
                "from_own_update": -(ta[(r, k)] - tc[(r, k)]) / sd,
                "attacker_induced": (ref_a - ref_swap) / sd,
            })
        out[k] = rows
    return out


def pre_onset_rank(rows: Sequence[Mapping[str, Any]], client: str, rounds: Iterable[int]) -> list[int]:
    """Rank (1 = lowest) of the client's self-cosine among all clients in each round."""
    t = self_cosines(rows)
    clients = clients_of(rows)
    out = []
    for r in rounds:
        order = sorted(clients, key=lambda c: (t[(r, c)], c))
        out.append(order.index(client) + 1)
    return out


def update_geometry(log: Path, client: str, rounds: Iterable[int]) -> dict[int, dict[str, torch.Tensor]]:
    """Read-only: client's update deltas (float parameters, BN running stats excluded) from a sealed update log."""
    records = read_manifest(log)
    glob = {r["round_id"]: r for r in records if r.get("kind") == "global"}
    sub = {r["round_id"]: r for r in records if r.get("kind") == "submitted" and r["client_id"] == client}
    out = {}
    for r in rounds:
        g = load_tensor_state(log / glob[r - 1]["file"])
        s = load_tensor_state(log / sub[r]["file"])
        out[r] = {k: (s[k] - g[k]).double().reshape(-1) for k in s
                  if s[k].is_floating_point() and not k.endswith(("running_mean", "running_var"))}
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def geometry_summary(attacked: dict[int, dict[str, torch.Tensor]], clean: dict[int, dict[str, torch.Tensor]],
                     classifier_weight: str = "fc.weight", num_classes: int = 100, target: int = 0) -> list[dict[str, float]]:
    keys = sorted(next(iter(attacked.values())))
    flat = lambda d: torch.cat([d[k] for k in keys])
    rows = []
    rounds = sorted(attacked)
    for prev, r in zip(rounds, rounds[1:]):
        row = {"round": r, "self_cos_attacked": _cos(flat(attacked[r]), flat(attacked[prev])),
               "self_cos_clean": _cos(flat(clean[r]), flat(clean[prev]))}
        for name, d in (("attacked", attacked), ("clean", clean)):
            w = d[r][classifier_weight].reshape(num_classes, -1)
            row[f"target_row_norm_share_{name}"] = float(w[target].norm() / w.norm())
            l4 = [k for k in keys if k.startswith("layer4")]
            row[f"layer4_self_cos_{name}"] = _cos(torch.cat([d[r][k] for k in l4]), torch.cat([d[prev][k] for k in l4]))
        rows.append(row)
    return rows


def audit(root: Path, seeds: Sequence[str], attacker: str = "client_002", onset: int = 20, with_logs: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "POST-HOC EXPLORATORY, DEVELOPMENT ONLY; clean counterfactual used for diagnosis only"}
    for seed in seeds:
        clean_run, attacked_run = f"X1_clean_d{seed}", f"X2_badnets_d{seed}"
        rc = load_trace_rows(root / "results_rev" / clean_run / "client_round_trace.csv")
        ra = load_trace_rows(root / "results_rev" / attacked_run / "client_round_trace.csv")
        dec = decompose(rc, ra, attacker, range(onset, onset + 10))
        seed_rep: dict[str, Any] = {
            "attacker_pre_onset_rank": pre_onset_rank(rc, attacker, range(onset - 5, onset)),
            "decomposition_mean": {k: ({x: statistics.mean(r[x] for r in v) for x in ("delta_z", "from_reference", "from_own_update", "attacker_induced")}
                                       if v else "undefined (zero calibration sd)") for k, v in dec.items()},
            "alternatives": {},
        }
        for arm, rows in (("clean", rc), ("attacked", ra)):
            for name, ref in REFERENCES.items():
                res = residuals(rows, ref)
                seed_rep["alternatives"][f"{arm}/{name}"] = detect(rows, res)
        if with_logs:
            a = update_geometry(root / "logs_rev" / attacked_run / "update_log", attacker, range(onset - 2, onset + 11))
            c = update_geometry(root / "logs_rev" / clean_run / "update_log", attacker, range(onset - 2, onset + 11))
            seed_rep["geometry"] = geometry_summary(a, c)
        report[seed] = seed_rep
    out = root / "results_rev" / "pilot" / "reference_audit_posthoc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_audit_module.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reference-contamination audit (read-only, exploratory).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    parser.add_argument("--with-logs", action="store_true", help="also read sealed update logs (update geometry)")
    args = parser.parse_args(argv)
    report = audit(Path("."), args.seeds, with_logs=args.with_logs)
    print(json.dumps({s: {"rank": report[s]["attacker_pre_onset_rank"],
                          "attributed": {k: v["attributed"] for k, v in report[s]["alternatives"].items()}}
                      for s in args.seeds}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
