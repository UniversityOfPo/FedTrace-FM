"""Label-shift-invariant conditional-shift feasibility audit (protocol 16; CPU, read-only, no retraining).

    python -m src.rev.label_shift --stage logits --seeds 123 2026
    python -m src.rev.label_shift --stage analyse --seeds 123 2026

CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. Not a detector; no threshold is fitted; nothing about intent
is inferred. Probe images are the label-free manifest of ``src.rev.probe_audit`` (v2); their labels are never loaded
into, or passed to, any function here.

For client k, round r (protocol 16 §4):
    z_g(x)   logits of the pre-round global model θ_r;    z_k(x) logits of the client's submitted model;
    q = softmax(z).
    D_k(r)   = mean_{x in EVAL} KL(q_k(x) || q_g(x))                                   (uncorrected distance)
    v_hat    = argmin_v mean_{x in FIT} CE(q_k(x), softmax(z_g(x) + v)) + (LAMBDA/2)||v||^2
               (class-only log-prior correction, unsupervised: uses predictions only)
    R_k(r)   = mean_{x in EVAL} KL(q_k(x) || softmax(z_g(x) + v_hat))                    (held-out residual)
    Rfit_k(r)= same on FIT (in-sample, optimistic)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

LAMBDA = 1e-6
ROUNDS, PRE, POST = range(16, 30), range(16, 20), range(21, 30)
RUNS = {"clean": "X1_clean_d{s}", "badnets": "X2_badnets_d{s}", "relabel": "X3_relabel_d{s}", "drift": "X4_drift_d{s}"}
CLIENT = "client_002"
SIZES = (250, 500, 1000)
BATCH = 500


# ---------------------------------------------------------------- probes (no labels)

def load_manifest(root: Path, seed: str) -> list[dict[str, Any]]:
    data = json.loads((root / "results_rev/pilot/label_shift" / f"probe_manifest_{seed}.json").read_text())
    assert data["version"] == 2 and all(set(p) == {"absolute_train_index", "split"} for p in data["probes"])
    return data["probes"]


def probe_images(root: Path, indices: Sequence[int]) -> torch.Tensor:
    """Evaluation-transform images of the given CIFAR-100 training-file indices; labels are discarded."""
    from torchvision import datasets

    from src.data.cifar100 import build_eval_transforms

    ds = datasets.CIFAR100(root=str(root / "data"), train=True, download=False, transform=build_eval_transforms())
    return torch.stack([ds[i][0] for i in indices])


def order_key(seed: str, index: int) -> str:
    """Label-free deterministic order used for nested probe subsets and random halves."""
    return hashlib.sha256(f"FedTrace-probe-order-v1|{seed}|{index}".encode()).hexdigest()


# ---------------------------------------------------------------- stage 1: logits

def compute_logits(root: Path, seed: str, out: Path) -> dict[str, Any]:
    from src.models.resnet18_cifar import resnet18_cifar
    from src.rev.hashing import load_tensor_state, state_content_sha256
    from src.rev.update_log import read_manifest

    torch.set_num_threads(8)
    probes = load_manifest(root, seed)
    x = probe_images(root, [p["absolute_train_index"] for p in probes])
    model = resnet18_cifar(num_classes=100)
    index: dict[str, Any] = {"probe_indices": [p["absolute_train_index"] for p in probes], "states": {}}
    for cond, pattern in RUNS.items():
        run = pattern.format(s=seed)
        log = root / "logs_rev" / run / "update_log"
        records = read_manifest(log)
        glob = {r["round_id"]: r for r in records if r.get("kind") == "global"}
        sub = {(r["round_id"], r["client_id"]): r for r in records if r.get("kind") == "submitted"}
        wanted = {f"global_r{r - 1}": glob[r - 1] for r in ROUNDS}
        wanted.update({f"r{r}_{c}": rec for (r, c), rec in sub.items() if r in ROUNDS})
        (out / run).mkdir(parents=True, exist_ok=True)
        for name, rec in sorted(wanted.items()):
            target = out / run / f"{name}.pt"
            state = load_tensor_state(log / rec["file"])
            ok = state_content_sha256(state) == rec["content_sha256"]
            index["states"][f"{run}/{name}"] = {"content_sha256": rec["content_sha256"], "hash_ok": ok}
            if not ok or target.exists():
                continue
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                z = torch.cat([model(x[i:i + BATCH]) for i in range(0, len(x), BATCH)])
            partial = target.with_suffix(".partial")
            torch.save(z.to(torch.float32), partial)
            partial.replace(target)  # atomic: an interrupted run never leaves a truncated logits file
    (out / f"logits_index_{seed}.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


# ---------------------------------------------------------------- correction and residuals

def kl_rows(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Per-row KL(q || p) from log-probabilities (float64)."""
    return (log_q.exp() * (log_q - log_p)).sum(dim=1)


def fit_correction(z_ref: torch.Tensor, log_q: torch.Tensor, lam: float = LAMBDA) -> tuple[torch.Tensor, dict[str, Any]]:
    """Class-only log-prior correction v minimising mean CE(q, softmax(z_ref + v)) + lam/2 ||v||^2 (convex)."""
    z_ref, q = z_ref.double(), log_q.double().exp()
    v = torch.zeros(z_ref.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=1000, history_size=20, tolerance_grad=1e-10,
                            tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def objective() -> torch.Tensor:
        opt.zero_grad()
        loss = -(q * torch.log_softmax(z_ref + v, dim=1)).sum(dim=1).mean() + 0.5 * lam * (v * v).sum()
        loss.backward()
        return loss

    opt.step(objective)
    loss = objective()
    grad = float(v.grad.abs().max())
    return v.detach(), {"grad_max": grad, "converged": grad < 1e-6, "objective": float(loss.detach())}


def residuals(z_ref: torch.Tensor, z_client: torch.Tensor, fit: torch.Tensor, ev: torch.Tensor) -> dict[str, Any]:
    """Uncorrected distance and corrected residuals for one client model (boolean masks fit / ev over probes)."""
    z_ref, z_client = z_ref.double(), z_client.double()
    log_q = torch.log_softmax(z_client, dim=1)
    v, info = fit_correction(z_ref[fit], log_q[fit])
    raw = kl_rows(log_q, torch.log_softmax(z_ref, dim=1))
    corr = kl_rows(log_q, torch.log_softmax(z_ref + v, dim=1))
    return {"raw_rows": raw, "corr_rows": corr, "v": v, "fit_info": info,
            "D": float(raw[ev].mean()), "R": float(corr[ev].mean()), "Rfit": float(corr[fit].mean()),
            "D_fit": float(raw[fit].mean())}


# ---------------------------------------------------------------- stage 2: analysis

def _masks(seed: str, probes: list[dict[str, Any]], pred_class_r16: torch.Tensor) -> dict[str, torch.Tensor]:
    ev = torch.tensor([p["split"] == "eval" for p in probes])
    fit = ~ev
    keys = [order_key(seed, p["absolute_train_index"]) for p in probes]
    rank = {k: i for i, k in enumerate(sorted(keys))}
    eval_order = sorted((rank[k], i) for i, k in enumerate(keys) if ev[i])
    masks = {"fit": fit, "eval": ev}
    for n in SIZES:  # nested eval subsets by the label-free order
        m = torch.zeros(len(probes), dtype=torch.bool)
        m[[i for _, i in eval_order[:n]]] = True
        masks[f"eval_{n}"] = m
    bit = torch.tensor([int(k[1], 16) % 2 == 0 for k in keys])
    masks["eval_random_half_a"], masks["eval_random_half_b"] = ev & bit, ev & ~bit
    low = pred_class_r16 < 50  # label-free composition split: θ_16's predicted class
    masks["eval_predclass_0_49"], masks["eval_predclass_50_99"] = ev & low, ev & ~low
    fit_order = sorted((rank[k], i) for i, k in enumerate(keys) if fit[i])
    for n in SIZES:
        m = torch.zeros(len(probes), dtype=torch.bool)
        m[[i for _, i in fit_order[:n]]] = True
        masks[f"fit_{n}"] = m
    return masks


def analyse_seed(root: Path, seed: str, store: Path) -> dict[str, Any]:
    probes = load_manifest(root, seed)
    index = json.loads((store / f"logits_index_{seed}.json").read_text())
    bad = sorted(k for k, v in index["states"].items() if not v["hash_ok"])
    z16 = torch.load(store / RUNS["clean"].format(s=seed) / "global_r15.pt")
    masks = _masks(seed, probes, z16.argmax(dim=1))
    out: dict[str, Any] = {"hash_failures": bad, "n_fit": int(masks["fit"].sum()), "n_eval": int(masks["eval"].sum()),
                           "per": {}}
    for cond, pattern in RUNS.items():
        run = pattern.format(s=seed)
        for r in ROUNDS:
            zg = store / run / f"global_r{r - 1}.pt"
            for c in (f"client_00{i}" for i in range(5)):
                zc = store / run / f"r{r}_{c}.pt"
                key = f"{cond}/{c}/{r}"
                if f"{run}/global_r{r - 1}" in bad or f"{run}/r{r}_{c}" in bad or not zg.exists() or not zc.exists():
                    out["per"][key] = None
                    continue
                g, k = torch.load(zg), torch.load(zc)
                res = residuals(g, k, masks["fit"], masks["eval"])
                rec = {"D": res["D"], "R": res["R"], "Rfit": res["Rfit"], "D_fit": res["D_fit"],
                       "explained_fraction": 1 - res["R"] / res["D"] if res["D"] > 0 else None,
                       "fit_info": res["fit_info"], "v_max": float(res["v"].max()), "v_min": float(res["v"].min()),
                       "top1_disagreement": float((g.argmax(1) != k.argmax(1))[masks["eval"]].double().mean()),
                       "subsets": {name: {"D": float(res["raw_rows"][m].mean()), "R": float(res["corr_rows"][m].mean())}
                                   for name, m in masks.items() if name.startswith("eval_")}}
                if c == CLIENT and r in POST:  # probe-size sensitivity of the fit
                    rec["fit_size"] = {}
                    for n in SIZES:
                        sub = residuals(g, k, masks[f"fit_{n}"], masks["eval"])
                        rec["fit_size"][str(n)] = {"R": sub["R"], "converged": sub["fit_info"]["converged"]}
                out["per"][key] = rec
    return out


def _series(per: Mapping[str, Any], cond: str, client: str, field: str, subset: str | None = None) -> dict[int, float | None]:
    out = {}
    for r in ROUNDS:
        rec = per.get(f"{cond}/{client}/{r}")
        if rec is None:
            out[r] = None
        elif subset is None:
            out[r] = rec[field]
        else:
            out[r] = rec["subsets"][subset][field]
    return out


def _mean(values) -> float | None:
    values = list(values)
    return None if any(v is None for v in values) else statistics.mean(values)


def excess(per, cond: str, client: str, field: str, subset: str | None = None) -> float | None:
    """Deployable own-history excess: mean over rounds 21..29 minus mean over rounds 16..19."""
    s = _series(per, cond, client, field, subset)
    post, pre = _mean(s[r] for r in POST), _mean(s[r] for r in PRE)
    return None if post is None or pre is None else post - pre


def matched(per, cond: str, client: str, field: str) -> float | None:
    """Diagnostic excess over the matched clean run (uses the counterfactual)."""
    a, c = _series(per, cond, client, field), _series(per, "clean", client, field)
    return _mean((a[r] - c[r]) if a[r] is not None and c[r] is not None else None for r in POST)


def clean_scale(per, field: str) -> float | None:
    """Pooled within-client sd of the per-round value over rounds 21..29 in the matched clean run (all 5 clients)."""
    devs = []
    for i in range(5):
        s = _series(per, "clean", f"client_00{i}", field)
        vals = [s[r] for r in POST]
        if any(v is None for v in vals):
            return None
        m = statistics.mean(vals)
        devs += [v - m for v in vals]
    return (sum(d * d for d in devs) / (len(devs) - 5)) ** 0.5


def verdicts(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Protocol 16 §8 criteria A-G (fixed before outcomes)."""
    v: dict[str, Any] = {"per_seed": {}}
    for s, a in per_seed.items():
        per = a["per"]
        E = {(cond, f): excess(per, cond, CLIENT, f) for cond in RUNS for f in ("R", "D", "Rfit")}
        S = {f: clean_scale(per, f) for f in ("R", "D")}
        drift_series = _series(per, "drift", CLIENT, "R")
        drift_raw = _series(per, "drift", CLIENT, "D")
        rec = {"E": {f"{c}/{f}": x for (c, f), x in E.items()}, "S_clean": S,
               "M": {f"{c}/{f}": matched(per, c, CLIENT, f) for c in ("badnets", "relabel", "drift") for f in ("R", "D")}}
        rec["A"] = all(drift_series[r] is not None and drift_raw[r] is not None and drift_series[r] < drift_raw[r] for r in POST) \
            and rec["M"]["drift/R"] is not None and rec["M"]["drift/D"] is not None and rec["M"]["drift/R"] < rec["M"]["drift/D"]
        rec["B"] = None if None in (E[("drift", "R")], E[("relabel", "R")]) else E[("drift", "R")] < E[("relabel", "R")]
        rec["B_insample"] = None if None in (E[("drift", "Rfit")], E[("relabel", "Rfit")]) else E[("drift", "Rfit")] < E[("relabel", "Rfit")]
        rec["B_uncorrected"] = None if None in (E[("drift", "D")], E[("relabel", "D")]) else E[("drift", "D")] < E[("relabel", "D")]
        rec["C"] = None if None in (E[("badnets", "R")], E[("drift", "R")], S["R"]) else E[("badnets", "R")] - E[("drift", "R")] >= 2 * S["R"]
        honest = [excess(per, cond, f"client_00{i}", "R") for cond in ("clean", "relabel") for i in range(5)
                  if not (cond == "relabel" and i == 2)]
        rec["honest_abs_excess_max"] = max(abs(x) for x in honest if x is not None)
        rec["D_similar"] = E[("relabel", "R")] is not None and rec["honest_abs_excess_max"] >= E[("relabel", "R")]
        halves = {}
        for sub in ("eval_random_half_a", "eval_random_half_b", "eval_predclass_0_49", "eval_predclass_50_99"):
            d, rl = excess(per, "drift", CLIENT, "R", sub), excess(per, "relabel", CLIENT, "R", sub)
            halves[sub] = None if None in (d, rl) else d < rl
        rec["E_halves_B"] = halves
        rec["E_strong_dependence"] = any(x is None or x != rec["B"] for x in halves.values())
        sep = {f: (E[("relabel", f)] - E[("drift", f)]) / S[f] if None not in (E[("relabel", f)], E[("drift", f)], S[f]) and S[f] else None
               for f in ("R", "D")}
        rec["separation_index"] = sep
        v["per_seed"][s] = rec
    seeds = v["per_seed"].values()
    v["A_correction_reduces_drift_residual"] = "PASS" if all(r["A"] for r in seeds) else "FAIL"
    v["B_primary_drift_below_relabel"] = "PASS" if all(r["B"] is True for r in seeds) else ("NOT EVALUABLE" if any(r["B"] is None for r in seeds) else "FAIL")
    v["C_badnets_sensitivity"] = "PASS" if all(r["C"] is True for r in seeds) else "FAIL"
    v["D_clean_clients_similar_residuals"] = "SIMILAR" if any(r["D_similar"] for r in seeds) else "NOT SIMILAR"
    v["E_probe_composition"] = "STRONG DEPENDENCE" if any(r["E_strong_dependence"] for r in seeds) else "NO ORDER CHANGE"
    v["F_heldout"] = ("SEPARATION DISAPPEARS ON HELD-OUT" if all(r["B_insample"] for r in seeds) and not all(r["B"] for r in seeds)
                      else "CONSISTENT")
    v["G_uncorrected_baseline"] = ("CORRECTION ADDS SEPARATION" if all(r["separation_index"]["R"] is not None and r["separation_index"]["D"] is not None
                                                                       and r["separation_index"]["R"] > r["separation_index"]["D"] for r in seeds)
                                   else "BASELINE EQUALLY GOOD OR BETTER")
    core = v["A_correction_reduces_drift_residual"] == "PASS" and v["B_primary_drift_below_relabel"] == "PASS" and v["F_heldout"] == "CONSISTENT"
    v["H_LS_core"] = "NOT FALSIFIED (2 development seeds)" if core else "FALSIFIED"
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Protocol 16 label-shift feasibility audit (read-only).")
    parser.add_argument("--stage", choices=("logits", "analyse"), required=True)
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    root = Path(".")
    store = root / "results_rev/pilot/label_shift/logits"
    if args.stage == "logits":
        for s in args.seeds:
            idx = compute_logits(root, s, store)
            print(s, "states", len(idx["states"]), "hash failures", sum(not v["hash_ok"] for v in idx["states"].values()))
        return 0
    per_seed = {s: analyse_seed(root, s, store) for s in args.seeds}
    result = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY (protocol 16)",
              "per_seed": per_seed, "verdicts": verdicts(per_seed)}
    out = root / "results_rev/pilot/label_shift/label_shift.json"
    out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result["verdicts"].items() if k != "per_seed"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
