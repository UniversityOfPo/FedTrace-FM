"""Generate the manuscript figures and tables from frozen artifacts only (read-only; CPU).

    python manuscript_c4ai4c27/scripts/make_assets.py          (run from the repository root)

Inputs (all hashed in paper_plan/31_evidence_manifest.sha256):
    results_rev/X2_badnets_d2026/client_round_trace.csv, results_rev/W1_badnets/client_round_trace.csv
    results_rev/pilot/unified_outcomes/unified_outcomes.json            (development + archive)
    results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json
Outputs: manuscript_c4ai4c27/figures/*.pdf, manuscript_c4ai4c27/tables/*.tex, manuscript_c4ai4c27/ASSET_PROVENANCE.json
Nothing is fitted; detector outcomes are read, not recomputed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from src.analysis.fedtrace_detector import load_trace_rows  # noqa: E402
from src.rev.reference_audit import inclusive_median, residuals, self_cosines  # noqa: E402

OUT = Path("manuscript_c4ai4c27")
INPUTS = {
    "trace_2026_badnets": Path("results_rev/X2_badnets_d2026/client_round_trace.csv"),
    "trace_W1_badnets": Path("results_rev/W1_badnets/client_round_trace.csv"),
    "unified_dev": Path("results_rev/pilot/unified_outcomes/unified_outcomes.json"),
    "unified_conf": Path("results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json"),
}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def band_series(rows, j="client_002", rounds=range(12, 36)):
    t = self_cosines(rows)
    clients = sorted({r["client_id"] for r in rows})
    out = {"r": [], "t_j": [], "lo": [], "hi": [], "c": []}
    for r in rounds:
        others = sorted(t[(r, c)] for c in clients if c != j)
        out["r"].append(r)
        out["t_j"].append(t[(r, j)])
        out["lo"].append(others[1])   # K = 5: the others' middle band [s_2, s_3] (Lemma A)
        out["hi"].append(others[2])
        out["c"].append(statistics.median([t[(r, c)] for c in clients]))
    return out


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for ax, key, title in ((axes[0], "trace_2026_badnets", "(a) seed 2026 (dev.): median client crosses up"),
                           (axes[2], "trace_W1_badnets", "(c) seed W1 (conf.): starts above band")):
        s = band_series(load_trace_rows(INPUTS[key]))
        ax.fill_between(s["r"], s["lo"], s["hi"], color="0.85", label="others' middle band")
        ax.plot(s["r"], s["c"], color="0.3", lw=0.8, ls="--", label="inclusive median $c(r)$")
        ax.plot(s["r"], s["t_j"], color="C3", lw=1.0, label="client_002 $t_j(r)$")
        ax.axvline(20, color="k", lw=0.5, ls=":")
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("round")
        ax.set_ylabel("self-cosine")
    axes[0].legend(fontsize=5, loc="upper left", frameon=False)
    rows = load_trace_rows(INPUTS["trace_2026_badnets"])
    res = residuals(rows, inclusive_median)
    for c, col in (("client_001", "C0"), ("client_004", "C2"), ("client_002", "C3")):
        rr = [r for r in range(12, 36) if r in res[c]]
        axes[1].plot(rr, [max(0.0, res[c][r]["z"]) for r in rr], color=col, lw=0.9, label=c)
    axes[1].axhline(3.0, color="k", lw=0.6, ls="--", label=r"$\tau=3$")
    axes[1].axvline(20, color="k", lw=0.5, ls=":")
    axes[1].set_title("(b) seed 2026: one-sided residual score $z^t_{res}$", fontsize=6.5)
    axes[1].set_xlabel("round")
    axes[1].legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures/fig1_reference_contamination.pdf")
    plt.close(fig)


CODES = {"ok": (0, "ok"), "ok_fa": (1, "ok+FA"), "wrong": (2, "wrong"), "miss": (3, "miss"),
         "miss_fa": (4, "miss+FA"), "fa": (4, "FA"), "silent": (5, "-")}
LEGEND = ("ok = attacker confirmed first, no honest alarm; ok+FA = attacker first, an honest client also confirmed; "
          "wrong = an honest client confirmed first; miss = attacker never confirmed; FA = honest client(s) confirmed")


def outcome(row):
    fa = row["honest_false_accusations"]
    if row["malicious"]:
        if row["attribution_correct"]:
            return "ok_fa" if fa else "ok"
        if row["attacker_detected"]:
            return "wrong"
        return "miss_fa" if fa else "miss"
    return "fa" if row["any_confirmation"] else "silent"


def load_rows():
    rows = json.loads(INPUTS["unified_dev"].read_text())["rows"] + json.loads(INPUTS["unified_conf"].read_text())["rows"]
    return {(r["trace"], r["detector"]): r for r in rows}


RUN = {("123", "BadNets"): "X2_badnets_d123", ("2026", "BadNets"): "X2_badnets_d2026", ("W1", "BadNets"): "W1_badnets",
       ("W2", "BadNets"): "W2_badnets", ("123", "drift"): "X4_drift_d123", ("2026", "drift"): "X4_drift_d2026",
       ("W1", "drift"): "W1_drift", ("W2", "drift"): "W2_drift", ("123", "clean"): "X1_clean_d123",
       ("2026", "clean"): "X1_clean_d2026", ("W1", "clean"): "W1_clean", ("W2", "clean"): "W2_clean"}
SEEDS = ("123", "2026", "W1", "W2")
ARMS = ("BadNets", "drift", "clean")
DETS = ("v3", "v3 magnitude-only", "v3 t_res-only", "ZP-norm", "CUSUM-norm")


def fig2(table):
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#2e7d32", "#9ccc65", "#f9a825", "#9e9e9e", "#c62828", "#ffffff"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.7), sharey=True)
    for ax, arm in zip(axes, ARMS):
        grid = [[CODES[outcome(table[(RUN[(s, arm)], d)])][0] for s in SEEDS] for d in DETS]
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=5.5, aspect="auto")
        ax.set_xticks(range(len(SEEDS)), [s + ("*" if s.startswith("W") else "") for s in SEEDS])
        ax.set_yticks(range(len(DETS)), DETS)
        ax.set_title({"BadNets": "BadNets (malicious)", "drift": "benign drift", "clean": "clean"}[arm], fontsize=6.5)
        for i in range(len(DETS)):
            for k in range(len(SEEDS)):
                ax.text(k, i, CODES[outcome(table[(RUN[(SEEDS[k], arm)], DETS[i])])][1], ha="center",
                        va="center", fontsize=4.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures/fig2_reliability_grid.pdf")
    plt.close(fig)


def fmt(row):
    fa = row["honest_false_accusations"]
    fa_s = (r"; FA\," + ",".join(c[-1] for c in fa)) if fa else ""
    if row["malicious"]:
        if row["attribution_correct"]:
            return rf"\checkmark\ {row['onset_error']}/{row['delay']}{fa_s}"
        if row["attacker_detected"]:
            return rf"$\times$ first {row['first_attributed'][-1]}{fa_s}"
        return "missed" + fa_s
    return "--" if not row["any_confirmation"] else r"\textbf{acc.\,}" + ",".join(c[-1] for c in fa)


def table1(table):
    lines = [r"% GENERATED by manuscript_c4ai4c27/scripts/make_assets.py from frozen artifacts -- do not edit by hand.",
             r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrr" + "l" * len(DETS) + "}", r"\hline",
             "Arm & Seed & ASR & $\\Delta$top-1 & " + " & ".join(d.replace("_", r"\_") for d in DETS) + r" \\", r"\hline"]
    for arm in ARMS:
        for s in SEEDS:
            r0 = table[(RUN[(s, arm)], "v3")]
            asr = r0["attack_effect"]["asr"]
            top = r0["attack_effect"]["final_top1"]
            ctop = table[(RUN[(s, "clean")], "v3")]["attack_effect"]["final_top1"]
            dtop = "" if arm == "clean" else f"{100 * (top - ctop):+.2f}"
            seed = s + ("$^\\ast$" if s.startswith("W") else "")
            lines.append(f"{arm} & {seed} & {'' if asr is None else f'{100 * asr:.1f}\\%'} & {dtop} & "
                         + " & ".join(fmt(table[(RUN[(s, arm)], d)]) for d in DETS) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}",
              r"\caption{AUTHOR REVIEW REQUIRED. Seed-level outcomes (no pooling). $^\ast$ preregistered confirmation seeds "
              r"(W1 = 1178015705, W2 = 1463065807); 123, 2026 development seeds. Malicious arm cells: \checkmark\ = attacker "
              r"confirmed first, onset error / delay in rounds; $\times$ = an honest client confirmed first; FA = honest clients "
              r"confirmed (index). Drift / clean cells: acc. = clients confirmed (all honest; in drift client 2 is the "
              r"intervened honest client). ASR = triggered attack success on 9,900 non-target test images; criterion "
              r"$\geq$ 50\%. Thresholds frozen (v3 $\tau=3$, $P=3$; ZP $h=1.75$, CUSUM $h=11.75$ from development clean runs).}",
              r"\label{tab:seedlevel}", r"\end{table*}"]
    (OUT / "tables/table1_seed_level.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for d in ("figures", "tables"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    table = load_rows()
    fig1()
    fig2(table)
    table1(table)
    prov = {"generator": "manuscript_c4ai4c27/scripts/make_assets.py",
            "inputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in INPUTS.items()},
            "outputs": {str(p): sha(p) for p in sorted(list((OUT / "figures").glob("*.pdf")) + list((OUT / "tables").glob("*.tex")))}}
    (OUT / "ASSET_PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
