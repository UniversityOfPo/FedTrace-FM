"""Generate the final publication figure package for the FedTrace-FM paper
(Phase P4).

Reads ONLY existing experiment artifacts under results/ (via the frozen,
unmodified src/analysis/fedtrace_detector.py module for FedTrace scores, and
raw CSV/JSON artifacts directly for everything else). Never trains a model,
runs an attack, or re-tunes any detector parameter. Writes only under
paper_assets/figures/ -- no source/config/result/checkpoint/Research-Database
file is touched.

Usage:
    python scripts/generate_paper_figures.py

Repository-relative paths throughout (no hard-coded absolute paths). Fails
loudly (raises) if a required artifact is missing, rather than silently
skipping or fabricating a figure.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "federated"
FIG_ROOT = REPO_ROOT / "paper_assets" / "figures"
MAIN_DIR = FIG_ROOT / "main"
APPENDIX_DIR = FIG_ROOT / "appendix"
PREVIEW_DIR = FIG_ROOT / "previews"
SOURCE_DATA_DIR = FIG_ROOT / "source_data"

sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis import fedtrace_detector as ft  # noqa: E402  (frozen, unmodified module)

EXP_DIRS = {
    "EXP_008": "cifar100_fedavg_dirichlet_a01_seed123_50rounds",
    "EXP_009": "cifar100_fedavg_dirichlet_a01_seed2026_50rounds",
    "EXP_016": "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123",
    "EXP_017": "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed2026",
    "EXP_018": "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123",
    "EXP_019": "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed2026",
    "EXP_020": "cifar100_multikrum_dirichlet_a01_seed123_50rounds",
    "EXP_021": "cifar100_multikrum_dirichlet_a01_seed2026_50rounds",
    "EXP_022": "cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed123",
    "EXP_023": "cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed2026",
    "EXP_024": "cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed123",
    "EXP_025": "cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed2026",
    "EXP_026": "cifar100_fedavg_dirichlet_a01_direction_attack_client004_round20_seed123",
    "EXP_027": "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round30_seed123",
}


class FigureDataError(RuntimeError):
    """Raised when a required artifact is missing or malformed -- fails
    loudly rather than silently producing an incomplete/fabricated figure."""


def exp_dir(exp_id: str) -> Path:
    if exp_id not in EXP_DIRS:
        raise FigureDataError(f"Unknown experiment id {exp_id!r}")
    path = RESULTS_DIR / EXP_DIRS[exp_id]
    if not path.exists():
        raise FigureDataError(f"{exp_id}: result directory not found: {path}")
    return path


def load_summary(exp_id: str) -> dict:
    path = exp_dir(exp_id) / "summary.json"
    if not path.exists():
        raise FigureDataError(f"{exp_id}: summary.json not found at {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_trace_rows(exp_id: str) -> list[dict]:
    path = exp_dir(exp_id) / "client_round_trace.csv"
    if not path.exists():
        raise FigureDataError(f"{exp_id}: client_round_trace.csv not found at {path}")
    return ft.load_trace_rows(path)


def load_multikrum_rows(exp_id: str) -> list[dict]:
    path = exp_dir(exp_id) / "multikrum_diagnostics.csv"
    if not path.exists():
        raise FigureDataError(f"{exp_id}: multikrum_diagnostics.csv not found at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_id"] = int(row["round_id"])
        row["krum_score"] = float(row["krum_score"])
        row["krum_rank"] = int(row["krum_rank"])
        row["multikrum_rejected"] = row["multikrum_rejected"].strip().lower() == "true"
    return rows


def load_partition_summary(exp_id: str) -> dict[str, int]:
    path = exp_dir(exp_id) / "client_partition_summary.csv"
    if not path.exists():
        raise FigureDataError(f"{exp_id}: client_partition_summary.csv not found at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["client_id"]: int(row["num_samples"]) for row in csv.DictReader(handle)}


def fedtrace_scores(exp_id: str) -> dict[str, list]:
    """Client-id -> sorted list of ScoredRowV3, via the frozen, unmodified detector."""
    rows = load_trace_rows(exp_id)
    scored = ft.compute_all_scores_v3(rows)
    by_client: dict[str, list] = {}
    for row in scored:
        by_client.setdefault(row.client_id, []).append(row)
    for client_rows in by_client.values():
        client_rows.sort(key=lambda r: r.round_id)
    return by_client


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

HONEST_COLOR = "#8c8c8c"
MALICIOUS_COLOR = "#c1272d"
THRESHOLD_COLOR = "#222222"
ONSET_COLOR = "#1f6f43"
CONFIRM_COLOR = "#2255a4"
FEDAVG_COLOR = "#2255a4"
MULTIKRUM_COLOR = "#c1272d"
GAUSSIAN_COLOR = "#2255a4"
DIRECTIONAL_COLOR = "#c1272d"

CLIENT_MARKERS = {
    "client_000": "o", "client_001": "s", "client_002": "^",
    "client_003": "D", "client_004": "v",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "font.family": "sans-serif",
    })


def save_figure(fig, name: str, main: bool) -> tuple[Path, Path]:
    target_dir = MAIN_DIR if main else APPENDIX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{name}.pdf"
    png_path = PREVIEW_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_source_data(name: str, header: list[str], rows: list[list]) -> Path:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DATA_DIR / f"{name}_source_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


PROVENANCE: list[dict] = []


def record_provenance(figure_id, panel, experiment_id, source_file, source_columns,
                       transformation, plotted_metric, notes) -> None:
    PROVENANCE.append({
        "figure_id": figure_id, "panel": panel, "experiment_id": experiment_id,
        "source_file": source_file, "source_columns": source_columns,
        "transformation": transformation, "plotted_metric": plotted_metric, "notes": notes,
    })


# ---------------------------------------------------------------------------
# Figure 1 -- conceptual method overview (no experimental data)
# ---------------------------------------------------------------------------


def fig01_overview() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2", ec="#333333", fontsize=8):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                              fc=fc, ec=ec, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))

    # Explanatory note lives in its own reserved band at the top, well clear
    # of every box below it -- never overlapping, per the visual-inspection
    # fix applied after the first rendering pass.
    ax.text(5.0, 7.15,
            "Cross-client round reference $c_r$ (round $r$'s own median across clients) vs.\n"
            "per-client longitudinal calibration (each client's own history over $[0,T_{\\mathrm{ref}})$)\n"
            "are two structurally different calibration windows -- see branch labels below.",
            ha="center", va="top", fontsize=6.8, style="italic", color="#444444")

    box(3.7, 5.55, 2.6, 0.7, r"Client update trace  $\Delta_{k,r}$", fc="#e8e8e8")
    arrow(5.0, 5.55, 5.0, 5.0)

    box(0.3, 3.75, 3.9, 1.05,
        "Magnitude branch\n" r"$m_{k,r}=\|\Delta_{k,r}\|_2$" "\n"
        "per-client longitudinal calibration\n" r"$z^m_{k,r}$",
        fc="#dbe6f5")
    box(5.8, 3.75, 3.9, 1.05,
        "Temporal residual branch\n" r"$t_{k,r}=\cos(\Delta_{k,r},\Delta_{k,r-1})$" "\n"
        "cross-client round reference " r"$c_r$" " + per-client\n"
        "longitudinal residual calibration " r"$\to z^t_{k,r}$",
        fc="#f5dbdb")

    arrow(2.25, 3.75, 4.6, 3.0)
    arrow(7.75, 3.75, 5.4, 3.0)

    box(3.6, 2.4, 2.8, 0.65, r"$S_{k,r}=\max(z^m_{k,r},\,z^t_{k,r})$", fc="#eeeeee")
    arrow(5.0, 2.4, 5.0, 1.9)

    box(3.1, 1.3, 3.8, 0.65, r"threshold $\tau=3$,  persistence $P=3$", fc="#eeeeee")
    arrow(5.0, 1.3, 5.0, 0.8)

    ax.text(5.0, 0.4, "malicious-client identity  +  attack-onset estimate",
            ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title("FedTrace v3 -- Conceptual Pipeline (no experimental data plotted)", fontsize=10)
    save_figure(fig, "fig01_fedtrace_overview", main=True)
    record_provenance("fig01_fedtrace_overview", "single", "NOT_APPLICABLE",
                       "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE",
                       "NOT APPLICABLE -- conceptual figure, no experimental data plotted.")


# ---------------------------------------------------------------------------
# Figure 2 -- attack-onset trajectory (EXP_018)
# ---------------------------------------------------------------------------


def fig02_onset_trajectory() -> None:
    exp_id = "EXP_018"
    scores = fedtrace_scores(exp_id)
    window = list(range(15, 31))
    malicious = "client_002"
    true_onset, confirm_round = 20, 22

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.2), sharex=True)
    panel_names = [r"z^m", r"z^t_{\mathrm{res}}", r"\max\!\left(z^m,\, z^t_{\mathrm{res}}\right)"]
    panel_labels = [r"$z^m$", r"$z^t_{\mathrm{res}}$", r"$S=\max(z^m,\,z^t_{\mathrm{res}})$"]
    attrs = ["z_m", "z_t_res", None]

    rows_out = []
    for ax, name, label, attr in zip(axes, panel_names, panel_labels, attrs):
        for client_id, client_rows in sorted(scores.items()):
            by_round = {r.round_id: r for r in client_rows}
            xs, ys = [], []
            for r in window:
                if r not in by_round:
                    continue
                row = by_round[r]
                val = row.combined_score(("m", "t_res"))[0] if attr is None else getattr(row, attr)
                if val is None:
                    continue
                xs.append(r)
                ys.append(val)
                rows_out.append([exp_id, name, client_id, r, val])
            if client_id == malicious:
                ax.plot(xs, ys, color=MALICIOUS_COLOR, lw=2.0, marker="^", ms=4,
                        label="client_002 (malicious)", zorder=5)
            else:
                ax.plot(xs, ys, color=HONEST_COLOR, lw=1.0, alpha=0.85,
                        marker=CLIENT_MARKERS[client_id], ms=3, label=client_id)
        ax.axhline(3.0, color=THRESHOLD_COLOR, lw=1.0, ls="--", label=r"$\tau=3$")
        ax.axvline(true_onset, color=ONSET_COLOR, lw=1.0, ls=":", label="true onset (r=20)")
        ax.axvline(confirm_round, color=CONFIRM_COLOR, lw=1.0, ls="-.", label="confirmation (r=22)")
        ax.set_ylabel(label)
        ax.set_title(f"panel: {label}", fontsize=8, loc="left")

    axes[-1].set_xlabel("communication round")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=6.5)
    fig.suptitle("EXP_018 (directional attack, seed 123) -- rounds 15-30 window", fontsize=8.5, y=1.10)
    save_figure(fig, "fig02_onset_trajectory", main=True)
    write_source_data("fig02_onset_trajectory",
                       ["experiment_id", "panel", "client_id", "round_id", "value"], rows_out)
    record_provenance("fig02_onset_trajectory", "z_m/z_t_res/S", exp_id,
                       "results/federated/.../client_round_trace.csv",
                       "update_l2_norm, self_prev_round_cosine",
                       "src/analysis/fedtrace_detector.py::compute_all_scores_v3 (frozen, unmodified)",
                       "z_m, z_t_res, S=max(z_m,z_t_res) per client per round",
                       "Window rounds 15-30 shown deliberately (round 0's known self-inclusive "
                       "calibration artifact, documented in FEDTRACE_MASTER_RESEARCH_STATUS.md "
                       "Section 7, is outside this window by design, not silently cropped -- see caption.")


# ---------------------------------------------------------------------------
# Figure 3 -- signal complementarity (Gaussian vs directional)
# ---------------------------------------------------------------------------


def fig03_signal_complementarity() -> None:
    runs = [("EXP_016", "Gaussian", "seed 123"), ("EXP_017", "Gaussian", "seed 2026"),
            ("EXP_018", "Directional", "seed 123"), ("EXP_019", "Directional", "seed 2026")]
    window = list(range(15, 31))
    malicious = "client_002"

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0), sharex=True, sharey=True)
    rows_out = []
    for row_idx, family in enumerate(["Gaussian", "Directional"]):
        family_runs = [r for r in runs if r[1] == family]
        ax_m, ax_t = axes[row_idx]
        for exp_id, _, seed_label in family_runs:
            scores = fedtrace_scores(exp_id)[malicious]
            by_round = {r.round_id: r for r in scores}
            style = "-" if "123" in seed_label else "--"
            xs_m, ys_m, xs_t, ys_t = [], [], [], []
            for r in window:
                if r not in by_round:
                    continue
                row = by_round[r]
                if row.z_m is not None:
                    xs_m.append(r); ys_m.append(row.z_m)
                    rows_out.append([exp_id, family, seed_label, "z_m", r, row.z_m])
                if row.z_t_res is not None:
                    xs_t.append(r); ys_t.append(row.z_t_res)
                    rows_out.append([exp_id, family, seed_label, "z_t_res", r, row.z_t_res])
            color = GAUSSIAN_COLOR if family == "Gaussian" else DIRECTIONAL_COLOR
            ax_m.plot(xs_m, ys_m, style, color=color, lw=1.6, label=seed_label)
            ax_t.plot(xs_t, ys_t, style, color=color, lw=1.6, label=seed_label)
        for ax in (ax_m, ax_t):
            ax.axhline(3.0, color=THRESHOLD_COLOR, lw=0.9, ls=":")
            ax.axvline(20, color=ONSET_COLOR, lw=0.8, ls=":")
        ax_m.set_ylabel(f"{family} attack\n\n" + r"$z^m$ (client\_002)")
        if row_idx == 0:
            ax_m.set_title(r"magnitude signal $z^m$")
            ax_t.set_title(r"temporal residual signal $z^t_{res}$")
        ax_t.set_ylabel(r"$z^t_{res}$ (client\_002)")
        ax_m.legend(fontsize=6.5, loc="upper left")
    axes[1, 0].set_xlabel("communication round")
    axes[1, 1].set_xlabel("communication round")
    fig.suptitle("Malicious-client (client_002) signal response by attack family, rounds 15-30", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig03_signal_complementarity", main=True)
    write_source_data("fig03_signal_complementarity",
                       ["experiment_id", "attack_family", "seed", "signal", "round_id", "value"], rows_out)
    for exp_id, family, seed_label in runs:
        record_provenance("fig03_signal_complementarity", f"{family}/{seed_label}", exp_id,
                           "results/federated/.../client_round_trace.csv",
                           "update_l2_norm, self_prev_round_cosine",
                           "src/analysis/fedtrace_detector.py::compute_all_scores_v3 (frozen)",
                           "z_m, z_t_res for client_002",
                           "Not claiming fusion beats best single detector -- see caption; "
                           "shows attack-family coverage only.")


# ---------------------------------------------------------------------------
# Figure 4 -- Multi-Krum attack-geometry dichotomy
# ---------------------------------------------------------------------------


def fig04_multikrum_geometry() -> None:
    cells = {
        ("Gaussian", "seed 123"): ("EXP_022", "client_002", 20),
        ("Gaussian", "seed 2026"): ("EXP_023", "client_002", 20),
        ("Directional", "seed 123"): ("EXP_024", "client_002", 20),
        ("Directional", "seed 2026"): ("EXP_025", "client_002", 20),
    }
    rates = {}
    rows_out = []
    for (family, seed_label), (exp_id, attacker, onset) in cells.items():
        rows = load_multikrum_rows(exp_id)
        post = [r for r in rows if r["client_id"] == attacker and r["round_id"] >= onset]
        rejected = sum(1 for r in post if r["multikrum_rejected"])
        rate = rejected / len(post)
        rates[(family, seed_label)] = rate
        rows_out.append([exp_id, family, seed_label, attacker, onset, rejected, len(post), rate])

    fig = plt.figure(figsize=(6.75, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6])

    ax0 = fig.add_subplot(gs[0])
    families = ["Gaussian", "Directional"]
    seeds = ["seed 123", "seed 2026"]
    matrix = [[rates[(f, s)] * 100 for s in seeds] for f in families]
    im = ax0.imshow(matrix, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax0.set_xticks(range(len(seeds))); ax0.set_xticklabels(seeds)
    ax0.set_yticks(range(len(families))); ax0.set_yticklabels(families)
    for i in range(len(families)):
        for j in range(len(seeds)):
            val = matrix[i][j]
            ax0.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if val > 50 else "black")
    ax0.set_title("post-onset attacker\nrejection rate", fontsize=8.5)

    ax1 = fig.add_subplot(gs[1])
    exp_g, exp_d = "EXP_022", "EXP_024"
    rows_g = load_multikrum_rows(exp_g)
    rows_d = load_multikrum_rows(exp_d)
    for label, rows_mk, y_off in [("Gaussian (EXP_022)", rows_g, 0.15), ("Directional (EXP_024)", rows_d, -0.15)]:
        attacker_rows = sorted((r for r in rows_mk if r["client_id"] == "client_002"), key=lambda r: r["round_id"])
        xs = [r["round_id"] for r in attacker_rows]
        ys = [1 + y_off if r["multikrum_rejected"] else 0 + y_off for r in attacker_rows]
        color = GAUSSIAN_COLOR if "Gaussian" in label else DIRECTIONAL_COLOR
        marker = "o" if "Gaussian" in label else "^"
        ax1.step(xs, ys, where="post", color=color, lw=1.3, label=label)
        ax1.scatter(xs, ys, color=color, s=8, marker=marker, zorder=5)
    ax1.axvline(20, color=ONSET_COLOR, lw=0.9, ls=":", label="attack onset (r=20)")
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["selected", "rejected"])
    ax1.set_xlabel("communication round")
    ax1.set_title("client_002 selected/rejected status,\nseed 123, rounds 0-49", fontsize=8.5)
    ax1.legend(fontsize=6.5, loc="center right")

    fig.suptitle("Multi-Krum attacker-rejection dichotomy by attack geometry\n"
                  "(empirical result under the tested configuration -- not a universal theorem)", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure(fig, "fig04_multikrum_geometry", main=True)
    write_source_data("fig04_multikrum_geometry",
                       ["experiment_id", "attack_family", "seed", "attacker", "onset_round",
                        "rejected_count", "post_onset_round_count", "rejection_rate"], rows_out)
    for (family, seed_label), (exp_id, attacker, onset) in cells.items():
        record_provenance("fig04_multikrum_geometry", f"{family}/{seed_label}", exp_id,
                           "results/federated/.../multikrum_diagnostics.csv",
                           "round_id, client_id, multikrum_rejected",
                           "rejection_rate = rejected_rounds / total_post_onset_rounds for the true attacker",
                           "post-onset attacker rejection rate (%)",
                           "Caption states this is an empirical result under the tested "
                           "configuration (n=5, f=1, alpha=0.1, c=0/rho=1.0), not a universal "
                           "theorem about Multi-Krum.")
    record_provenance("fig04_multikrum_geometry", "timeline (right panel)", "EXP_022, EXP_024",
                       "results/federated/.../multikrum_diagnostics.csv",
                       "round_id, multikrum_rejected (client_002 only)",
                       "binary selected/rejected step series",
                       "selected(0)/rejected(1) per round", "Seed 123 only, representative panel.")


# ---------------------------------------------------------------------------
# Figure 5 -- robust aggregation vs forensic attribution
# ---------------------------------------------------------------------------


def fig05_aggregation_vs_attribution() -> None:
    # FedTrace facts (frozen v3, from analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md
    # Section 4 / analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md, re-verified against
    # the frozen detector in this task rather than retyped from memory).
    ft_scores_018 = fedtrace_scores("EXP_018")["client_002"]
    ft_result_018 = ft.evaluate_detector_v3(load_trace_rows("EXP_018"), "H", ("m", "t_res"))
    assert ft_result_018.predicted_client == "client_002"
    ft_onset, ft_confirm = 20, 22

    mk_rows = load_multikrum_rows("EXP_024")
    proxy_confirmations = {}
    for client_id in sorted({r["client_id"] for r in mk_rows}):
        streak = 0
        client_rows = sorted((r for r in mk_rows if r["client_id"] == client_id), key=lambda r: r["round_id"])
        for r in client_rows:
            if r["multikrum_rejected"]:
                streak += 1
                if streak >= 3:
                    proxy_confirmations[client_id] = r["round_id"]
                    break
            else:
                streak = 0
    proxy_client = min(proxy_confirmations, key=lambda c: (proxy_confirmations[c], c))
    proxy_round = proxy_confirmations[proxy_client]
    attacker_ever_rejected = any(r["multikrum_rejected"] for r in mk_rows if r["client_id"] == "client_002")

    fig, ax = plt.subplots(figsize=(6.75, 2.6))
    ax.set_xlim(-1, 51)
    ax.set_ylim(0, 3)
    ax.set_yticks([2, 1])
    ax.set_yticklabels(["FedTrace v3\n(EXP_018)", "Multi-Krum proxy\n(EXP_024)"])
    ax.set_xlabel("communication round")

    ax.axvline(20, color=ONSET_COLOR, lw=1.0, ls=":", label="true attack onset (r=20)")
    ax.scatter([ft_confirm], [2], color=MALICIOUS_COLOR, marker="^", s=70, zorder=5)
    ax.annotate(f"correctly confirms client_002\nat r={ft_confirm} (delay=2)", (ft_confirm, 2),
                xytext=(ft_confirm + 3, 2.35), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))

    ax.scatter([proxy_round], [1], color=HONEST_COLOR, marker="s", s=70, zorder=5)
    ax.annotate(f"confirms honest {proxy_client}\nat r={proxy_round} (before onset)", (proxy_round, 1),
                xytext=(proxy_round + 3, 0.55), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))
    if not attacker_ever_rejected:
        ax.text(35, 1.35, "true attacker client_002: NEVER rejected, any round (0/30 post-onset)",
                fontsize=7, color=MALICIOUS_COLOR, style="italic")

    ax.set_title("Successful per-round exclusion behavior $\\neq$ correct longitudinal attribution\n"
                  "(directional attack, seed 123)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig05_aggregation_vs_attribution", main=True)

    rows_out = [
        ["EXP_018", "FedTrace_v3", "client_002", "client_002", ft_onset, ft_confirm, ft_confirm - ft_onset, "correct"],
        ["EXP_024", "MultiKrum_proxy", "client_002", proxy_client, "NOT_APPLICABLE", proxy_round, "NOT_APPLICABLE",
         "incorrect" if proxy_client != "client_002" else "correct"],
    ]
    write_source_data("fig05_aggregation_vs_attribution",
                       ["experiment_id", "method", "true_attacker", "predicted_client",
                        "true_onset", "confirmation_round", "delay", "attribution_correct"], rows_out)
    record_provenance("fig05_aggregation_vs_attribution", "FedTrace", "EXP_018",
                       "results/federated/.../client_round_trace.csv",
                       "update_l2_norm, self_prev_round_cosine",
                       "src/analysis/fedtrace_detector.py::evaluate_detector_v3 (frozen)",
                       "predicted_client, onset_estimate, confirmation_round",
                       "Cross-checked against analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md.")
    record_provenance("fig05_aggregation_vs_attribution", "Multi-Krum proxy", "EXP_024",
                       "results/federated/.../multikrum_diagnostics.csv",
                       "round_id, client_id, multikrum_rejected",
                       "P_krum=3 consecutive-rejection persistence rule applied to raw diagnostics "
                       "(same rule as Research_Database/sync_master_results.py)",
                       "proxy_predicted_client, proxy_confirmation_round",
                       "Matches Research_Database/FedTrace_Master_Results.xlsx 'MultiKrum Diagnostics "
                       "Summary' sheet computed in Phase P2.")


# ---------------------------------------------------------------------------
# Figure 6 (appendix) -- utility 2x2
# ---------------------------------------------------------------------------


def fig06_utility_2x2() -> None:
    conditions = [
        ("Gaussian", "seed 123", "EXP_008", "EXP_016", "EXP_020", "EXP_022"),
        ("Gaussian", "seed 2026", "EXP_009", "EXP_017", "EXP_021", "EXP_023"),
        ("Directional", "seed 123", "EXP_008", "EXP_018", "EXP_020", "EXP_024"),
        ("Directional", "seed 2026", "EXP_009", "EXP_019", "EXP_021", "EXP_025"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), sharey=True)
    rows_out = []
    for ax, family in zip(axes, ["Gaussian", "Directional"]):
        family_conditions = [c for c in conditions if c[0] == family]
        x = range(4)
        width = 0.35
        fedavg_vals, mk_vals, xticklabels = [], [], []
        for _, seed_label, clean_fa, attack_fa, clean_mk, attack_mk in family_conditions:
            top1_clean_fa = load_summary(clean_fa)["final_test"]["top1"] * 100
            top1_attack_fa = load_summary(attack_fa)["final_test"]["top1"] * 100
            top1_clean_mk = load_summary(clean_mk)["final_test"]["top1"] * 100
            top1_attack_mk = load_summary(attack_mk)["final_test"]["top1"] * 100
            fedavg_vals.extend([top1_clean_fa, top1_attack_fa])
            mk_vals.extend([top1_clean_mk, top1_attack_mk])
            xticklabels.extend([f"{seed_label}\nclean", f"{seed_label}\nattacked"])
            rows_out.append([family, seed_label, "fedavg", "clean", clean_fa, top1_clean_fa])
            rows_out.append([family, seed_label, "fedavg", "attacked", attack_fa, top1_attack_fa])
            rows_out.append([family, seed_label, "multikrum", "clean", clean_mk, top1_clean_mk])
            rows_out.append([family, seed_label, "multikrum", "attacked", attack_mk, top1_attack_mk])
        xs = range(len(fedavg_vals))
        ax.bar([i - width / 2 for i in xs], fedavg_vals, width, color=FEDAVG_COLOR, label="FedAvg")
        ax.bar([i + width / 2 for i in xs], mk_vals, width, color=MULTIKRUM_COLOR, label="Multi-Krum")
        ax.set_xticks(list(xs)); ax.set_xticklabels(xticklabels, fontsize=6.5)
        ax.set_title(family, fontsize=9)
    axes[0].set_ylabel("Test Top-1 accuracy (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Utility: FedAvg vs. Multi-Krum, clean vs. attacked (2 seeds; not a significance claim)",
                 fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "fig06_utility_2x2", main=False)
    write_source_data("fig06_utility_2x2",
                       ["attack_family", "seed", "aggregation", "condition", "experiment_id", "top1_pct"], rows_out)
    for family, seed_label, clean_fa, attack_fa, clean_mk, attack_mk in conditions:
        for exp_id, cond in [(clean_fa, "clean_fedavg"), (attack_fa, "attacked_fedavg"),
                              (clean_mk, "clean_multikrum"), (attack_mk, "attacked_multikrum")]:
            record_provenance("fig06_utility_2x2", f"{family}/{seed_label}/{cond}", exp_id,
                               "results/federated/.../summary.json", "final_test.top1",
                               "read directly, multiplied by 100 for percentage display",
                               "Test Top-1 accuracy (%)",
                               "n=2 seeds; no significance claim made or implied.")


# ---------------------------------------------------------------------------
# Figure 7 (appendix) -- partition / fairness association
# ---------------------------------------------------------------------------


def fig07_partition_fairness() -> None:
    seeds = [("seed 123", "EXP_020", "client_004"), ("seed 2026", "EXP_021", "client_001")]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8), sharey=True)
    rows_out = []
    for ax, (seed_label, exp_id, rejected_client) in zip(axes, seeds):
        counts = load_partition_summary(exp_id)
        clients = sorted(counts.keys())
        values = [counts[c] for c in clients]
        colors = [MALICIOUS_COLOR if c == rejected_client else HONEST_COLOR for c in clients]
        ax.bar(clients, values, color=colors)
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels(clients, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(f"{seed_label}\n(most-rejected: {rejected_client})", fontsize=8)
        for c, v in zip(clients, values):
            rows_out.append([exp_id, seed_label, c, v, c == rejected_client])
    axes[0].set_ylabel("client sample count")
    fig.suptitle("Client partition size vs. persistently-rejected client (clean Multi-Krum)\n"
                  "replicated association under the tested $\\alpha=0.1$ configuration -- not a causal claim",
                 fontsize=8.2)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, "fig07_partition_fairness", main=False)
    write_source_data("fig07_partition_fairness",
                       ["experiment_id", "seed", "client_id", "num_samples", "is_most_rejected_client"], rows_out)
    for seed_label, exp_id, rejected_client in seeds:
        record_provenance("fig07_partition_fairness", seed_label, exp_id,
                           "results/federated/.../client_partition_summary.csv",
                           "client_id, num_samples",
                           "direct read; highlighted client cross-referenced against "
                           "Research_Database MultiKrum Diagnostics Summary pre_attack_rejected_client",
                           "client sample count",
                           "Caption states 'replicated association under the tested configuration', "
                           "not a causal claim.")


# ---------------------------------------------------------------------------
# Figure 8 (appendix) -- generalization check (see report: recommended for P5 table instead)
# ---------------------------------------------------------------------------


def fig08_generalization() -> None:
    # Computed live from each run's own client_round_trace.csv via the frozen,
    # unmodified detector -- not transcribed from prose -- exactly as fig02/03/05.
    run_ids = ["EXP_018", "EXP_026", "EXP_027"]
    conditions = []
    for exp_id in run_ids:
        rows = load_trace_rows(exp_id)
        result = ft.evaluate_detector_v3(rows, "H_fedtrace_v3", ("m", "t_res"))
        true_client = result.true_malicious_client
        true_onset = result.attack_start_round
        predicted = result.predicted_client
        onset_est = result.per_client[predicted].onset_estimate if predicted else None
        delay = result.detection_delay
        honest_fp_count = len(result.false_positive_clients)
        if not (predicted == true_client and result.onset_localization_error == 0 and delay is not None):
            raise FigureDataError(
                f"{exp_id}: unexpected FedTrace v3 result while building fig08 "
                f"(predicted={predicted}, true={true_client}, onset_err="
                f"{result.onset_localization_error}, delay={delay}) -- refusing to plot "
                "an assumption that does not match the frozen detector's live output."
            )
        conditions.append((exp_id, true_client, true_onset, predicted, onset_est, delay, honest_fp_count))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    labels = [c[0] for c in conditions]
    onsets = [c[2] for c in conditions]
    confirms = [c[2] + c[5] for c in conditions]
    ys = range(len(conditions))
    for y, (exp_id, true_c, true_o, pred_c, est_o, delay, fp) in zip(ys, conditions):
        ax.plot([true_o, true_o + delay], [y, y], color=MALICIOUS_COLOR, lw=3, solid_capstyle="butt")
        ax.scatter([true_o], [y], color=ONSET_COLOR, marker="|", s=140, zorder=5)
        ax.scatter([true_o + delay], [y], color=CONFIRM_COLOR, marker="o", s=35, zorder=5)
        ax.text(true_o + delay + 1, y, f"{pred_c}, onset={est_o}, delay={delay}, FP={fp}", fontsize=6.5, va="center")
    ax.set_yticks(list(ys)); ax.set_yticklabels(labels)
    ax.set_xlabel("communication round")
    ax.set_xlim(15, 45)
    ax.set_title("Generalization gate: identity (EXP_026) and\nonset (EXP_027) checks vs. EXP_018 anchor", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig08_generalization_check", main=False)
    write_source_data("fig08_generalization_check",
                       ["experiment_id", "true_attacker", "true_onset", "predicted_client",
                        "onset_estimate", "detection_delay", "honest_persistent_fp"], conditions)
    for exp_id, true_c, true_o, pred_c, est_o, delay, fp in conditions:
        record_provenance("fig08_generalization_check", exp_id, exp_id,
                           "results/federated/.../client_round_trace.csv",
                           "update_l2_norm, self_prev_round_cosine",
                           "src/analysis/fedtrace_detector.py::evaluate_detector_v3 (frozen, unmodified) "
                           "-- computed live, not transcribed from any analysis document",
                           "predicted_client, onset_estimate, detection_delay, honest_persistent_fp_count",
                           "Cross-checked against analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md Section 4 "
                           "(C10/C11) and matches exactly. RECOMMENDATION: this content is inherently "
                           "tabular (7 short fields x 3 rows) and is likely stronger as a P5 table than "
                           "as this figure -- see analysis/FEDTRACE_FINAL_FIGURE_PLAN.md.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    apply_style()
    fig01_overview()
    fig02_onset_trajectory()
    fig03_signal_complementarity()
    fig04_multikrum_geometry()
    fig05_aggregation_vs_attribution()
    fig06_utility_2x2()
    fig07_partition_fairness()
    fig08_generalization()

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    provenance_path = FIG_ROOT / "FIGURE_DATA_PROVENANCE.csv"
    header = ["figure_id", "panel", "experiment_id", "source_file", "source_columns",
              "transformation", "plotted_metric", "notes"]
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(PROVENANCE)

    print(f"Generated {len(list(MAIN_DIR.glob('*.pdf')))} main-paper PDF figures in {MAIN_DIR}")
    print(f"Generated {len(list(APPENDIX_DIR.glob('*.pdf')))} appendix PDF figures in {APPENDIX_DIR}")
    print(f"Generated {len(list(PREVIEW_DIR.glob('*.png')))} PNG previews in {PREVIEW_DIR}")
    print(f"Provenance rows: {len(PROVENANCE)} -> {provenance_path}")


if __name__ == "__main__":
    main()
