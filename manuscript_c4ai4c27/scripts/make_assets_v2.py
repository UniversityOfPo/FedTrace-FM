"""Revision-2 manuscript assets, generated only from frozen artifacts (read-only inputs; CPU).

    python manuscript_c4ai4c27/scripts/make_assets_v2.py OUT_DIR          (run from the repository root)

Figures (vector PDF + PNG preview): fig_overview, fig_contamination, fig_calibration, fig_reliability_matrix.
Tables (LaTeX, booktabs): tab_main_outcomes, tab_ablation, tab_archived, tab_sensitivity, tab_calibration.
Algorithm (LaTeX algorithmicx): alg_fedtrace_v3. Data (CSV) behind every figure / table, and PROVENANCE.json.
Nothing is fitted or tuned; detector outcomes are read from the frozen unified tables or recomputed with the frozen code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")  # vector output; the Agg DLL may be blocked by OS application control
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, ".")
from src.analysis.fedtrace_detector import load_trace_rows  # noqa: E402
from src.rev.online import OnlineConfig, causal_scores_v3  # noqa: E402
from src.rev.reference_audit import inclusive_median, residuals, self_cosines  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "manuscript_c4ai4c27/assets_v2")
UNI_DEV = Path("results_rev/pilot/unified_outcomes/unified_outcomes.json")
UNI_CONF = Path("results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json")
E4 = Path("results_ndss/E4_detector_sensitivity/e4_summary.csv")
SEEDS = ("123", "2026", "W1", "W2")
SEED_LABEL = {"123": "123", "2026": "2026", "W1": "W1$^\\ast$", "W2": "W2$^\\ast$"}
RUN = {("badnets", "123"): "X2_badnets_d123", ("badnets", "2026"): "X2_badnets_d2026", ("badnets", "W1"): "W1_badnets",
       ("badnets", "W2"): "W2_badnets", ("relabel", "123"): "X3_relabel_d123", ("relabel", "2026"): "X3_relabel_d2026",
       ("drift", "123"): "X4_drift_d123", ("drift", "2026"): "X4_drift_d2026", ("drift", "W1"): "W1_drift",
       ("drift", "W2"): "W2_drift", ("clean", "123"): "X1_clean_d123", ("clean", "2026"): "X1_clean_d2026",
       ("clean", "W1"): "W1_clean", ("clean", "W2"): "W2_clean"}
DETS = ["v3", "v3 magnitude-only", "v3 t_res-only", "v3 LOO-ref", "v3 trimmed-ref", "v3 historical-ref",
        "ZP-norm", "CUSUM-norm", "PH-norm"]
DET_LABEL = {"v3": "FedTrace v3 (full)", "v3 magnitude-only": "v3: magnitude only", "v3 t_res-only": "v3: temporal only",
             "v3 LOO-ref": "v3: leave-one-out ref.", "v3 trimmed-ref": "v3: trimmed-mean ref.",
             "v3 historical-ref": "v3: client-historical ref.", "ZP-norm": "Norm z-score + persistence (ZP)",
             "CUSUM-norm": "Norm CUSUM", "PH-norm": "Norm Page-Hinkley"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.titlesize": 8.5, "legend.fontsize": 7})
INPUTS: dict[str, Path] = {"unified_dev": UNI_DEV, "unified_conf": UNI_CONF, "e4": E4}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig, name: str) -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    try:  # PNG preview only where the raster backend is available
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
        fig.savefig(OUT / "figures" / f"{name}.png", dpi=250, bbox_inches="tight")
    except ImportError:
        pass
    plt.close(fig)


def table() -> dict:
    rows = json.loads(UNI_DEV.read_text())["rows"] + json.loads(UNI_CONF.read_text())["rows"]
    return {(r["trace"], r["detector"]): r for r in rows}


def code(r) -> str:
    fa = r["honest_false_accusations"]
    if r["malicious"]:
        if r["attribution_correct"]:
            return "ok+FA" if fa else "ok"
        if r["attacker_detected"]:
            return "wrong"
        return "miss+FA" if fa else "miss"
    return "FA" if r["any_confirmation"] else "silent"


def ids(r) -> str:
    return ",".join(str(int(c[-3:])) for c in r["honest_false_accusations"])


# ------------------------------------------------------------------------------------------------ Fig: overview
def box(ax, x, y, w, h, text, fc, ec="#4a4a4a", fs=7.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#333333", style="-|>", lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, lw=lw, color=color, ls=ls))


def fig_overview():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # clients
    ax.text(0.01, 0.95, r"Clients (K = 5), updates $\Delta_{k,r}$", ha="left", fontsize=7.4, weight="bold")
    ys = [0.79, 0.655, 0.505, 0.37, 0.235]
    for i, y in enumerate(ys):
        lab = "client 002\n(intervened)" if i == 2 else f"client 00{i}"
        box(ax, 0.015, y, 0.1, 0.105 if i == 2 else 0.085, lab, "#fde9e7" if i == 2 else "#eef3fb", fs=6.4)
        arrow(ax, 0.128, y + 0.045, 0.175, 0.52 + (2 - i) * 0.06)
    ax.text(0.065, 0.06, "from round 20:\nclean | BadNets\nrelabel | honest drift", ha="center", fontsize=6.0,
            style="italic", linespacing=1.2)
    # server monitor
    ax.add_patch(FancyBboxPatch((0.175, 0.17), 0.57, 0.74, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#fafafa", ec="#888888", lw=0.8, ls="--"))
    ax.text(0.46, 0.865, "Server-side monitor: FedTrace v3 (frozen)", ha="center", fontsize=7.4, weight="bold")
    box(ax, 0.195, 0.55, 0.235, 0.25, "Magnitude channel\n" r"$n_{k,r}=\Vert\Delta_{k,r}\Vert_2$" "\n"
        r"$z^m_{k,r}=|n_{k,r}-\mu^m_k|/\sigma^m_k$", "#e8f1fd", fs=6.5)
    box(ax, 0.195, 0.21, 0.235, 0.28, "Temporal / peer channel\n" r"$t_{k,r}=\cos(\Delta_{k,r},\Delta_{k,r-1})$" "\n"
        r"$d_{k,r}=\mathrm{med}_j\,t_{j,r}-t_{k,r}$" "\n" r"$z^t_{k,r}=[(d_{k,r}-\mu^d_k)/\sigma^d_k]_+$",
        "#eaf6ec", fs=6.5)
    box(ax, 0.465, 0.53, 0.115, 0.18, "Fusion\n" r"$S=\max(z^m,z^t)$", "#fff4dd", fs=6.3)
    box(ax, 0.615, 0.53, 0.11, 0.18, "Persistence\n" r"$S\geq 3$ for" "\n3 rounds", "#fff4dd", fs=6.3)
    arrow(ax, 0.442, 0.675, 0.453, 0.65)
    arrow(ax, 0.442, 0.36, 0.453, 0.56)
    arrow(ax, 0.592, 0.62, 0.603, 0.62)
    box(ax, 0.465, 0.21, 0.26, 0.24, "Baselines (same norm score)\nZP: h = 1.75, 3 rounds\nCUSUM: h = 11.75\n"
        "Page-Hinkley: h = 6.5", "#f2f2f2", fs=6.1)
    # verdict and audit
    box(ax, 0.78, 0.63, 0.205, 0.26, "Forensic verdict\nclient " r"$\hat{k}$" ", onset " r"$\hat{r}$"
        "\nconfirmation round", "#f3ecfa", fs=6.5)
    arrow(ax, 0.737, 0.64, 0.766, 0.72)
    box(ax, 0.78, 0.06, 0.205, 0.48, "Reliability audit\n(reported separately)\n\n1 attack success (ASR)\n2 attacker detected?\n"
        "3 correct attribution?\n4 onset error, delay\n5 honest accused?", "#fdf6e7", fs=6.2)
    arrow(ax, 0.8825, 0.616, 0.8825, 0.554)
    ax.text(0.46, 0.075, "thresholds frozen  |  matched arms identical before round 20\nseeds 123, 2026 (development); W1, W2 (preregistered)",
            ha="center", fontsize=6.0, color="#444444", linespacing=1.25)
    save(fig, "fig_overview")


# ------------------------------------------------------------------------------------------------ Fig: contamination
def band_series(rows, j="client_002", rounds=range(12, 36)):
    t = self_cosines(rows)
    clients = sorted({r["client_id"] for r in rows})
    out = {"r": [], "t": [], "lo": [], "hi": [], "c": []}
    for r in rounds:
        others = sorted(t[(r, c)] for c in clients if c != j)
        out["r"].append(r)
        out["t"].append(t[(r, j)])
        out["lo"].append(others[1])
        out["hi"].append(others[2])
        out["c"].append(statistics.median([t[(r, c)] for c in clients]))
    return out


def fig_contamination():
    p26 = Path("results_rev/X2_badnets_d2026/client_round_trace.csv")
    pw1 = Path("results_rev/W1_badnets/client_round_trace.csv")
    INPUTS.update({"trace_X2_badnets_d2026": p26, "trace_W1_badnets": pw1})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    data_rows = []
    for ax, path, title in ((axes[0], p26, "(a) Seed 2026: median client crosses up"),
                            (axes[2], pw1, "(c) Seed W1*: starts above the band")):
        s = band_series(load_trace_rows(path))
        ax.fill_between(s["r"], s["lo"], s["hi"], color="#d9d9d9", label="other clients' middle band")
        ax.plot(s["r"], s["c"], color="#333333", lw=1.0, ls="--", label=r"inclusive median $c_r$")
        ax.plot(s["r"], s["t"], color="#c0392b", lw=1.4, label=r"client 002 self-cosine $t_{2,r}$")
        ax.axvline(20, color="#555555", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_ylabel("self-cosine")
        for i, r in enumerate(s["r"]):
            data_rows.append({"panel": title[:3], "round": r, "client002_t": s["t"][i], "band_lo": s["lo"][i],
                              "band_hi": s["hi"][i], "median": s["c"][i]})
    rows = load_trace_rows(p26)
    res = residuals(rows, inclusive_median)
    for c, col, lab in (("client_001", "#2c7fb8", "client 001 (honest)"), ("client_004", "#41ab5d", "client 004 (honest)"),
                        ("client_002", "#c0392b", None)):
        rr = [r for r in range(12, 36) if r in res[c]]
        axes[1].plot(rr, [max(0.0, res[c][r]["z"]) for r in rr], color=col, lw=1.3, label=lab)
        for r in rr:
            data_rows.append({"panel": "(b)", "round": r, "client": c, "z_t": max(0.0, res[c][r]["z"])})
    axes[1].axhline(3.0, color="#000000", lw=0.8, ls="--", label=r"threshold $\tau=3$")
    axes[1].axvline(20, color="#555555", lw=0.6, ls=":")
    axes[1].set_title(r"(b) Seed 2026: temporal score $z^t$")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel(r"$z^t_{k,r}$")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=6.8)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_contamination")
    write_csv("fig_contamination_data.csv", data_rows)


# ------------------------------------------------------------------------------------------------ Fig: calibration
def zm_series(path: Path, client: str):
    rows = load_trace_rows(path)
    zm = {(s.round_id, s.client_id): s.z_m for s in causal_scores_v3(rows, OnlineConfig())}
    return {r: zm[(r, client)] for r in range(20, 50) if zm.get((r, client)) is not None}


def fig_calibration(col: bool = False):
    # col=True: single-column variant (panels stacked) for a page-limited layout
    if col:
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.5), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2), sharey=True)
    data_rows = []
    colors = {"123": "#7f7f7f", "2026": "#2c7fb8", "W1": "#c0392b", "W2": "#e67e22"}
    for s in SEEDS:
        p = Path(f"results_rev/{RUN[('badnets', s)]}/client_round_trace.csv")
        INPUTS[f"trace_{RUN[('badnets', s)]}"] = p
        z = zm_series(p, "client_002")
        axes[0].plot(list(z), list(z.values()), color=colors[s], lw=1.3 if s.startswith("W") else 0.9,
                     ls="-" if s.startswith("W") else "--", label=f"seed {s}{'*' if s.startswith('W') else ''}")
        data_rows += [{"panel": "a", "seed": s, "client": "client_002", "round": r, "abs_z_norm": v} for r, v in z.items()]
    for ax in axes:
        ax.axhline(3.0, color="black", lw=0.9, ls="-")
        ax.axhline(1.75, color="black", lw=0.9, ls=":")
        if col:
            ax.set_ylabel(r"$z^m$")
            ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("round")
    if not col:
        axes[0].set_xlabel("round")
    axes[0].text(49.5, 3.08, r"v3: $\tau=3$", ha="right", va="bottom", fontsize=6.8)
    axes[0].text(49.5, 1.83, r"ZP: $h=1.75$", ha="right", va="bottom", fontsize=6.8)
    if not col:
        axes[0].set_ylabel(r"norm score $|n-\mu^m|/\sigma^m$")
    axes[0].set_title("(a) BadNets: attacker (client 002)", fontsize=7.5 if col else None)
    axes[0].legend(ncol=2, frameon=False, loc="lower right", fontsize=6.3 if col else None)
    p = Path("results_rev/W1_clean/client_round_trace.csv")
    INPUTS["trace_W1_clean"] = p
    for c, col in (("client_000", "#9e9e9e"), ("client_001", "#bdbdbd"), ("client_002", "#757575"), ("client_003", "#cfcfcf"),
                   ("client_004", "#6a3d9a")):
        z = zm_series(p, c)
        axes[1].plot(list(z), list(z.values()), color=col, lw=1.4 if c == "client_004" else 0.8,
                     label="client 004" if c == "client_004" else None)
        data_rows += [{"panel": "b", "seed": "W1", "client": c, "round": r, "abs_z_norm": v} for r, v in z.items()]
    axes[1].set_title("(b) W1* clean run (no intervention): all clients", fontsize=7.5 if col else None)
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.3 if col else None)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_calibration_col" if col else "fig_calibration")
    if not col:
        write_csv("fig_calibration_data.csv", data_rows)


# ------------------------------------------------------------------------------------------------ Fig: reliability matrix
COLORS = {"ok": "#2e7d32", "ok+FA": "#9ccc65", "wrong": "#f9a825", "miss": "#9e9e9e", "miss+FA": "#c62828",
          "FA": "#c62828", "silent": "#ffffff"}
TEXT = {"ok": "✓", "ok+FA": "✓+FA", "wrong": "W", "miss": "M", "miss+FA": "M+FA", "FA": "FA", "silent": "–"}


def fig_matrix(T):
    cols = [("badnets", s) for s in SEEDS] + [("relabel", s) for s in ("123", "2026")] + \
           [("drift", s) for s in SEEDS] + [("clean", s) for s in SEEDS]
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    data_rows = []
    for i, d in enumerate(DETS):
        for j, key in enumerate(cols):
            r = T[(RUN[key], d)]
            c = code(r)
            ax.add_patch(plt.Rectangle((j, len(DETS) - 1 - i), 1, 1, fc=COLORS[c], ec="white", lw=1.2))
            ax.text(j + 0.5, len(DETS) - 0.5 - i, TEXT[c], ha="center", va="center", fontsize=6.6,
                    color="white" if c in ("ok", "miss+FA", "FA") else "black", weight="bold" if c != "silent" else "normal")
            data_rows.append({"detector": d, "arm": key[0], "seed": key[1], "outcome": c, "honest_accused": ids(r),
                              "onset_error": r.get("onset_error"), "delay": r.get("delay")})
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(DETS))
    ax.set_xticks([j + 0.5 for j in range(len(cols))], [s + ("*" if s.startswith("W") else "") for _, s in cols], fontsize=7)
    ax.set_yticks([len(DETS) - 0.5 - i for i in range(len(DETS))], [DET_LABEL[d] for d in DETS], fontsize=7)
    ax.tick_params(length=0)
    for x in (4, 6, 10):
        ax.axvline(x, color="black", lw=1.2)
    ax.axhline(3, color="black", lw=1.2)
    for x, lab in ((2, "BadNets"), (5, "Relabel (dev.)"), (8, "Honest drift"), (12, "Clean")):
        ax.text(x, len(DETS) + 0.25, lab, ha="center", va="bottom", fontsize=7.5, weight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    legend = "  ".join(f"{TEXT[k]} {v}" for k, v in (("ok", "correct first attribution"), ("ok+FA", "correct + honest alarm"),
                                                      ("wrong", "honest client first"), ("miss", "attacker missed"),
                                                      ("FA", "honest client accused"), ("silent", "no alarm")))
    ax.text(len(cols) / 2, -1.05, legend, ha="center", va="top", fontsize=6.6)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_reliability_matrix")
    write_csv("fig_reliability_matrix_data.csv", data_rows)


# ------------------------------------------------------------------------------------------------ tables
def cell(r) -> str:
    c = code(r)
    if c in ("ok", "ok+FA"):
        s = f"\\checkmark\\,({r['onset_error']},\\,{r['delay']})"
        return s + (f"\\,+FA\\{{{ids(r)}\\}}" if c == "ok+FA" else "")
    if c == "wrong":
        return f"W: FA\\{{{ids(r)}\\}} first; attacker later ($d{{=}}{r['delay']}$)"
    if c == "miss":
        return "M"
    if c == "miss+FA":
        return f"M+FA\\{{{ids(r)}\\}}"
    if c == "FA":
        return f"FA\\{{{ids(r)}\\}}"
    return "--"


def tab_main(T):
    def eff(arm, s, key):
        return T[(RUN[(arm, s)], "v3")]["attack_effect"][key]

    clean = {s: eff("clean", s, "final_top1") for s in SEEDS}
    L = [r"% GENERATED by manuscript_c4ai4c27/scripts/make_assets_v2.py from frozen artifacts -- do not edit numbers by hand.",
         r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{Seed-level outcomes of the frozen monitor and the norm baselines (no pooling). "
         r"\checkmark\,($e$,\,$d$): attacker confirmed first, onset error $e$ and confirmation delay $d$ in rounds; "
         r"W: an honest client confirmed first; M: attacker never confirmed; FA\{$i$\}: honest client(s) $i$ confirmed; "
         r"--: no alarm; n/r: not run. $^\ast$preregistered confirmation seeds.}",
         r"\label{tab:main}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Arm & Quantity / detector & 123 & 2026 & W1$^\ast$ & W2$^\ast$ \\", r"\midrule",
         "Clean & final top-1 (\\%) & " + " & ".join(f"{100 * clean[s]:.2f}" for s in SEEDS) + r" \\",
         "BadNets & attack success rate (\\%) & " + " & ".join(f"{100 * eff('badnets', s, 'asr'):.1f}" for s in SEEDS) + r" \\",
         " & top-1 change (pp) & " + " & ".join(f"${100 * (eff('badnets', s, 'final_top1') - clean[s]):+.2f}$" for s in SEEDS) + r" \\"]
    for d in ("v3", "ZP-norm", "CUSUM-norm", "PH-norm"):
        L.append(f" & {DET_LABEL[d]} & " + " & ".join(cell(T[(RUN[('badnets', s)], d)]) for s in SEEDS) + r" \\")
    L += [r"\midrule",
          "Relabel-only & attack success rate (\\%) & " + " & ".join(
              f"{100 * eff('relabel', s, 'asr'):.1f}" if ("relabel", s) in RUN else "n/r" for s in SEEDS) + r" \\",
          " & top-1 change (pp) & " + " & ".join(
              f"${100 * (eff('relabel', s, 'final_top1') - clean[s]):+.2f}$" if ("relabel", s) in RUN else "n/r" for s in SEEDS) + r" \\"]
    for d in ("v3", "ZP-norm", "CUSUM-norm", "PH-norm"):
        L.append(f" & {DET_LABEL[d]} & " + " & ".join(cell(T[(RUN[('relabel', s)], d)]) if ("relabel", s) in RUN else "n/r"
                                                     for s in SEEDS) + r" \\")
    L += [r"\midrule", " Honest drift & top-1 change (pp) & " + " & ".join(
        f"${100 * (eff('drift', s, 'final_top1') - clean[s]):+.2f}$" for s in SEEDS) + r" \\"]
    for d in ("v3", "ZP-norm", "CUSUM-norm", "PH-norm"):
        L.append(f" & {DET_LABEL[d]} & " + " & ".join(cell(T[(RUN[('drift', s)], d)]) for s in SEEDS) + r" \\")
    L.append(r"\midrule")
    for k, d in enumerate(("v3", "ZP-norm", "CUSUM-norm", "PH-norm")):
        L.append(("Clean" if k == 0 else "") + f" & {DET_LABEL[d]} & " + " & ".join(cell(T[(RUN[('clean', s)], d)]) for s in SEEDS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("tab_main_outcomes.tex", L)


def short(r) -> str:
    return {"ok": "C", "ok+FA": "C+FA", "wrong": "W", "miss": "M", "miss+FA": "M+FA", "FA": "FA", "silent": "--"}[code(r)]


def tab_ablation(T):
    L = [r"% GENERATED by make_assets_v2.py", r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
         r"\caption{Ablation of the frozen monitor (same thresholds). C: correct first attribution; W: honest client first; "
         r"M: attacker missed; +FA / FA: honest client(s) confirmed; --: no alarm. $^\ast$preregistered.}",
         r"\label{tab:ablation}", r"\begin{tabular}{@{}l cccc cccc cccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{BadNets} & \multicolumn{4}{c}{Honest drift} & \multicolumn{4}{c}{Clean} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"Variant & 123 & 2026 & W1$^\ast$ & W2$^\ast$ & 123 & 2026 & W1$^\ast$ & W2$^\ast$ & 123 & 2026 & W1$^\ast$ & W2$^\ast$ \\",
         r"\midrule"]
    names = {"v3": "Full v3", "v3 magnitude-only": "Magnitude only", "v3 t_res-only": "Temporal only",
             "v3 LOO-ref": "Leave-one-out ref.", "v3 trimmed-ref": "Trimmed-mean ref.", "v3 historical-ref": "Historical ref."}
    for d in DETS[:6]:
        L.append(names[d] + " & " + " & ".join(short(T[(RUN[(a, s)], d)]) for a in ("badnets", "drift", "clean") for s in SEEDS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_ablation.tex", L)


def tab_archived():
    L = [r"% GENERATED by make_assets_v2.py (archived traces; v3 recomputed and equal to the original NDSS tables)",
         r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Archived model-update perturbations (not data poisoning), frozen v3. Entries: onset error / delay "
         r"(rounds); all attributions correct.}", r"\label{tab:archived}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
         r"Setting (FedAvg) & Seeds & Full v3 \\", r"\midrule",
         r"Gaussian noise, $\rho=1$, $K=5$ & 123, 2026 & 0/2, 0/2 \\",
         r"Norm-preserving directional, $c=0$, $K=5$ & 123, 2026 & 0/2, 0/2 \\",
         r"Directional, attacker client 004 & 123 & 0/2 \\",
         r"Directional, onset at round 30 & 123 & 0/2 \\",
         r"Gaussian / directional, $K=10$ & 123 & 0/2, 0/2 \\",
         r"Reduced severity: Gaussian $\rho=0.5$ & 123 & 21/23 \\",
         r"Reduced severity: directional $c=0.5$ & 123 & 0/2 \\",
         r"Intermittent Gaussian / directional & 123 & 0/2, 0/2 \\",
         r"\midrule",
         r"Ablation: magnitude only, directional & 123, 2026 & 27/29, 24/26 \\",
         r"Ablation: temporal only, Gaussian & 123, 2026 & 1/3, 0/2 \\",
         r"\midrule",
         r"Multi-Krum aggregation, clean runs & 123, 2026 & honest client confirmed (r46) \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_archived.tex", L)


def tab_sensitivity():
    rows = list(csv.DictReader(E4.open(encoding="utf-8")))
    L = [r"% GENERATED by make_assets_v2.py from results_ndss/E4_detector_sensitivity/e4_summary.csv",
         r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Archived threshold / persistence sensitivity of v3 on the four archived validation runs "
         r"(Gaussian and directional, seeds 123 and 2026). Correct attributions out of 4 and runs with any honest "
         r"persistent alarm.}", r"\label{tab:sensitivity}", r"\begin{tabular}{@{}cccc@{}}", r"\toprule",
         r"$\tau$ & $P$ & correct / 4 & runs with honest alarm \\", r"\midrule"]
    for r in rows:
        tag = r" (default)" if r["is_default_setting"] == "True" else ""
        L.append(f"{r['tau']} & {r['P']}{tag} & {r['correct_attribution_count_of_4']} & {r['runs_with_any_honest_persistent_fp']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_sensitivity.tex", L)


def tab_calibration():
    L = [r"% GENERATED by make_assets_v2.py (frozen traces; v3 magnitude score = ZP statistic, verified identical)",
         r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Why the full monitor misses BadNets while ZP detects it: both use the identical norm score; only the "
         r"threshold differs. Inflation: $\sigma^m$(rounds 0--19)\,/\,$\sigma^m$(rounds 2--19) for the attacker.}",
         r"\label{tab:calibration}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & 123 & 2026 & W1$^\ast$ & W2$^\ast$ \\", r"\midrule"]
    peak, n3, n175, infl = [], [], [], []
    rows_out = []
    for s in SEEDS:
        p = Path(f"results_rev/{RUN[('badnets', s)]}/client_round_trace.csv")
        z = zm_series(p, "client_002")
        rows = load_trace_rows(p)
        v = [r["update_l2_norm"] for r in sorted(rows, key=lambda r: r["round_id"]) if r["client_id"] == "client_002" and r["round_id"] < 20]
        inf = statistics.stdev(v) / statistics.stdev(v[2:])
        peak.append(f"{max(z.values()):.2f}")
        n3.append(str(sum(x >= 3 for x in z.values())))
        n175.append(str(sum(x > 1.75 for x in z.values())))
        infl.append(f"{inf:.2f}")
        rows_out.append({"seed": s, "peak_abs_z": max(z.values()), "rounds_ge_3": sum(x >= 3 for x in z.values()),
                         "rounds_gt_1.75": sum(x > 1.75 for x in z.values()), "sd_inflation": inf})
    L += ["Attacker peak norm score (r20--49) & " + " & ".join(peak) + r" \\",
          r"Rounds with score $\geq 3$ (v3 $\tau$) & " + " & ".join(n3) + r" \\",
          r"Rounds with score $> 1.75$ (ZP $h$) & " + " & ".join(n175) + r" \\",
          r"Calibration $\sigma^m$ inflation & " + " & ".join(infl) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_calibration.tex", L)
    write_csv("tab_calibration_data.csv", rows_out)


ALGORITHM = r"""% Algorithm box for the frozen monitor. Verified line by line against src/analysis/fedtrace_detector.py
% (compute_all_scores_v3, apply_persistence_rule_v3; frozen at commit 7f6ca8b). Requires: \usepackage{algorithm,algpseudocode}
\begin{algorithm}[t]
\caption{FedTrace v3 forensic monitor (frozen; evaluated, not proposed)}
\label{alg:fedtrace}
\begin{algorithmic}[1]
\Require client updates $\Delta_{k,r}$ for $k=1,\dots,K$; $T_{\mathrm{ref}}=20$; $\tau=3$; $P=3$
\Statex \textbf{Calibration} (once, at $r=T_{\mathrm{ref}}$)
\For{each client $k$}
  \State $\mu^m_k,\sigma^m_k \gets$ sample mean / s.d.\ of $n_{k,r}=\Vert\Delta_{k,r}\Vert_2$, $0\le r<T_{\mathrm{ref}}$
  \State $\mu^d_k,\sigma^d_k \gets$ sample mean / s.d.\ of $d_{k,r}=\mathrm{med}_j\,t_{j,r}-t_{k,r}$, $1\le r<T_{\mathrm{ref}}$
\EndFor
\Statex \textbf{Monitoring} (each round $r\ge T_{\mathrm{ref}}$)
\For{each client $k$}
  \State $t_{k,r}\gets\cos(\Delta_{k,r},\Delta_{k,r-1})$;\; $c_r\gets\mathrm{med}_{j}\,t_{j,r}$ \Comment{inclusive median}
  \State $z^m_{k,r}\gets|n_{k,r}-\mu^m_k|/\sigma^m_k$;\; $z^t_{k,r}\gets\max\{0,(c_r-t_{k,r}-\mu^d_k)/\sigma^d_k\}$
  \State $S_{k,r}\gets\max(z^m_{k,r},z^t_{k,r})$
  \If{$S_{k,r'}\ge\tau$ for $P$ consecutive rounds $r'$ ending at $r$ and $k$ not yet confirmed}
    \State confirm $k$ at round $r$; onset estimate $\hat r_k\gets r-P+1$
  \EndIf
\EndFor
\State \Return attributed client $\hat k$ = earliest confirmed client (ties: larger mean streak score); $\hat r_{\hat k}$; all confirmations
\end{algorithmic}
\end{algorithm}
"""


def write_csv(name, rows):
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / "data" / name).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_tex(name, lines):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = table()
    fig_overview()
    fig_contamination()
    fig_calibration()
    fig_calibration(col=True)
    fig_matrix(T)
    tab_main(T)
    tab_ablation(T)
    tab_archived()
    tab_sensitivity()
    tab_calibration()
    (OUT / "algorithm").mkdir(exist_ok=True)
    (OUT / "algorithm" / "alg_fedtrace_v3.tex").write_text(ALGORITHM, encoding="utf-8", newline="\n")
    prov = {"generator": "manuscript_c4ai4c27/scripts/make_assets_v2.py",
            "inputs": {k: {"path": v.as_posix(), "sha256": sha(v)} for k, v in sorted(INPUTS.items())},
            "outputs": {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "PROVENANCE.json"}}
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(len(prov["outputs"]), "outputs")


if __name__ == "__main__":
    main()
