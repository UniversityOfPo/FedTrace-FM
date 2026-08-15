"""Generate a horizontal (1x3) layout variant of Figure 2 (attack-onset
trajectory, EXP_018) for a two-column ICLR-style paper.

Reads ONLY the existing, already-generated source-data CSV
(paper_assets/figures/source_data/fig02_onset_trajectory_source_data.csv) --
no experiment result file is re-read, no detector is re-run, no value is
recomputed, estimated, or transcribed by hand. Reuses the original figure's
style constants, colors, and markers from scripts/generate_paper_figures.py
unchanged (imported, not duplicated, so the two figures are provably using
the same visual encoding). Writes two NEW files only
(fig02_onset_trajectory_horizontal.{pdf,png}); never touches the original
fig02_onset_trajectory.{pdf,png} or the source-data CSV itself.

Usage:
    python scripts/generate_fig02_horizontal.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_paper_figures as gpf  # noqa: E402  (main() is guarded by __name__; importing has no side effects)

SOURCE_DATA_PATH = (
    REPO_ROOT / "paper_assets" / "figures" / "source_data" / "fig02_onset_trajectory_source_data.csv"
)
MAIN_DIR = REPO_ROOT / "paper_assets" / "figures" / "main"
PREVIEW_DIR = REPO_ROOT / "paper_assets" / "figures" / "previews"

MALICIOUS = "client_002"
TRUE_ONSET = 20
CONFIRM_ROUND = 22
THRESHOLD = 3.0

# Exact panel identifiers as written into the source-data CSV's "panel"
# column by the original fig02_onset_trajectory() -- must match verbatim.
PANEL_ORDER = [
    "z^m",
    r"z^t_{\mathrm{res}}",
    r"\max\!\left(z^m,\, z^t_{\mathrm{res}}\right)",
]
PANEL_LABELS = {
    "z^m": r"$z^m$",
    r"z^t_{\mathrm{res}}": r"$z^t_{\mathrm{res}}$",
    r"\max\!\left(z^m,\, z^t_{\mathrm{res}}\right)": r"$S=\max(z^m,\,z^t_{\mathrm{res}})$",
}


def load_source_data() -> dict[str, dict[str, list[tuple[int, float]]]]:
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Original fig02 source-data CSV not found: {SOURCE_DATA_PATH}")
    data: dict[str, dict[str, list[tuple[int, float]]]] = {}
    with SOURCE_DATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            panel = row["panel"]
            client_id = row["client_id"]
            round_id = int(row["round_id"])
            value = float(row["value"])
            data.setdefault(panel, {}).setdefault(client_id, []).append((round_id, value))
    for panel_rows in data.values():
        for client_rows in panel_rows.values():
            client_rows.sort(key=lambda pair: pair[0])
    return data


def main() -> None:
    gpf.apply_style()
    data = load_source_data()

    missing_panels = [p for p in PANEL_ORDER if p not in data]
    if missing_panels:
        raise ValueError(f"Source-data CSV is missing expected panel(s): {missing_panels!r}")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharex=True)

    for ax, panel_key in zip(axes, PANEL_ORDER):
        panel_data = data[panel_key]
        for client_id in sorted(panel_data.keys()):
            xs = [r for r, _ in panel_data[client_id]]
            ys = [v for _, v in panel_data[client_id]]
            if client_id == MALICIOUS:
                ax.plot(xs, ys, color=gpf.MALICIOUS_COLOR, lw=2.0, marker="^", ms=4,
                         label="client_002 (malicious)", zorder=5)
            else:
                ax.plot(xs, ys, color=gpf.HONEST_COLOR, lw=1.0, alpha=0.85,
                         marker=gpf.CLIENT_MARKERS[client_id], ms=3, label=client_id)
        ax.axhline(THRESHOLD, color=gpf.THRESHOLD_COLOR, lw=1.0, ls="--", label=r"$\tau=3$")
        ax.axvline(TRUE_ONSET, color=gpf.ONSET_COLOR, lw=1.0, ls=":", label="true onset (r=20)")
        ax.axvline(CONFIRM_ROUND, color=gpf.CONFIRM_COLOR, lw=1.0, ls="-.", label="confirmation (r=22)")
        ax.set_ylabel(PANEL_LABELS[panel_key], labelpad=6)
        ax.set_title(f"panel: {PANEL_LABELS[panel_key]}", fontsize=8, loc="left")
        ax.set_xlabel("communication round")

    fig.subplots_adjust(wspace=0.55)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, fontsize=6.5)
    fig.suptitle("EXP_018 (directional attack, seed 123) -- rounds 15-30 window", fontsize=8.5, y=1.32)

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = MAIN_DIR / "fig02_onset_trajectory_horizontal.pdf"
    png_path = PREVIEW_DIR / "fig02_onset_trajectory_horizontal.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
