"""Build the ICASSP 2027 C4AI4C research export package (copy-only; originals are never modified).

    python scripts/build_export_package.py            (run from the repository root, CPU only)

Creates exports/ICASSP2027_C4AI4C_Research_Package/ and exports/ICASSP2027_C4AI4C_Research_Package.zip.
Every copied file is hash-checked (source == copy). Excluded: update logs, checkpoints (*.pt), label-shift logits,
Git internals, environments, FedTrace-FM material; their paths and hashes are recorded in 12_Evidence_Manifest.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path.cwd()
WIN_ROOT = r"C:\HTU Research\5. Research_FL\FedTrace-FM"
PKG_NAME = "ICASSP2027_C4AI4C_Research_Package"
PKG = ROOT / "exports" / PKG_NAME
ZIP = ROOT / "exports" / f"{PKG_NAME}.zip"
copied: list[tuple[str, str, str]] = []   # (source rel, dest rel, sha256)
EXCLUDE_PARTS = {".git", ".venv", "__pycache__", "checkpoints", "checkpoints_rev", "checkpoints_ndss", "logits"}
EXCLUDE_SUFFIX = {".pt", ".pth", ".ckpt", ".pyc", ".partial"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(p: Path) -> bool:
    return bool(EXCLUDE_PARTS & set(p.parts)) or p.suffix in EXCLUDE_SUFFIX or p.name.startswith(".env") \
        or "fm" in {x.lower() for x in p.parts} or p.name.upper().startswith("FM_") or "FEDTRACE_FM" in p.name.upper()


def copy(src: str | Path, dest_dir: str, keep_tree_from: str | None = None) -> None:
    """Copy one file; keep_tree_from: None = flat, "" = full repository path, prefix = tree below the prefix."""
    s = ROOT / src if not Path(src).is_absolute() else Path(src)
    if not s.is_file() or excluded(s.relative_to(ROOT) if s.is_relative_to(ROOT) else s):
        return
    rel = s.relative_to(ROOT).as_posix() if s.is_relative_to(ROOT) else s.name
    if keep_tree_from is None:
        sub = Path(Path(rel).name)                       # flat
    elif keep_tree_from == "":
        sub = Path(rel)                                  # full repository-relative tree
    else:
        assert rel.startswith(keep_tree_from), (rel, keep_tree_from)
        sub = Path(rel).relative_to(keep_tree_from)      # tree below the given prefix
    if any(d == (PKG / dest_dir / sub).relative_to(PKG).as_posix() for _, d, _ in copied):
        raise RuntimeError(f"duplicate destination {dest_dir}/{sub}")
    d = PKG / dest_dir / sub
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    h1, h2 = sha(s), sha(d)
    assert h1 == h2, f"copy mismatch {rel}"
    copied.append((rel, d.relative_to(PKG).as_posix(), h1))


def copy_glob(pattern: str, dest_dir: str, keep_tree_from: str | None = "") -> None:
    """keep_tree_from as in copy(); default keeps the full repository-relative path."""
    for f in sorted(glob.glob(pattern, recursive=True)):
        if Path(f).is_file():
            copy(f, dest_dir, keep_tree_from)


def write(rel: str, text: str) -> None:
    p = PKG / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def win(rel: str) -> str:
    return WIN_ROOT + "\\" + rel.replace("/", "\\")


# ------------------------------------------------------------------------------------------------ copy plan
def collect() -> None:
    # 01 research documents
    copy_glob("paper_plan/*.md", "01_Research_Documents/paper_plan", "paper_plan")
    copy_glob("research_plans/FEDTRACE_*.md", "01_Research_Documents/research_plans", "research_plans")
    copy_glob("research_plans/MULTIKRUM_*.md", "01_Research_Documents/research_plans", "research_plans")
    copy_glob("research_plans/PHASE*.md", "01_Research_Documents/research_plans", "research_plans")
    copy_glob("analysis/*", "01_Research_Documents/original_fedtrace_analysis", "analysis")
    for f in ("README.md", "ARCHITECTURE.md", "docs/EXPERIMENT_MAP.md", "docs/REPRODUCIBILITY.md", "requirements.txt"):
        copy(f, "01_Research_Documents/project")
    copy_glob("experiments/*.md", "01_Research_Documents/project/experiments", "experiments")
    for f in ("Experiment_Index.csv", "Figure_Index.csv", "Table_Index.csv", "FedTrace_Master_Results.md",
              "FedTrace_Master_Results.xlsx", "README.md"):
        copy(f"Research_Database/{f}", "01_Research_Documents/Research_Database")
    copy_glob("Research_Database/FedTrace_Master_Results/**/*", "01_Research_Documents/Research_Database/FedTrace_Master_Results",
              "Research_Database/FedTrace_Master_Results")
    # 02 original (archived) results: everything in results/ and results_ndss/ (no checkpoints exist there)
    copy_glob("results/**/*", "02_Original_FedTrace_Results/results", "results")
    copy_glob("results_ndss/**/*", "02_Original_FedTrace_Results/results_ndss", "results_ndss")
    # 03 development results (X runs, pairs, analyses)
    copy_glob("results_rev/X[1-4]_*/*", "03_Development_Results/runs", "results_rev")
    copy_glob("results_rev/pilot/X[2-4]_*__vs__*/**/*", "03_Development_Results/pair_analyses", "results_rev/pilot")
    copy_glob("results_rev/pilot/reanalysis_target_rate/**/*", "03_Development_Results/pair_analyses", "results_rev/pilot")
    for d in ("evidence_matrix", "contamination", "unified_outcomes", "source_attribution", "relabel_comparison",
              "reference_audit_posthoc"):
        copy_glob(f"results_rev/pilot/{d}/**/*", "03_Development_Results/analyses", "results_rev/pilot")
    copy_glob("results_rev/pilot/label_shift/*.json", "03_Development_Results/analyses", "results_rev/pilot")
    copy_glob("logs_rev/X[1-4]_*/update_log/manifest.jsonl", "03_Development_Results/update_log_manifests", "logs_rev")
    copy_glob("logs_rev/X[1-4]_*/update_log/SEAL.json", "03_Development_Results/update_log_manifests", "logs_rev")
    # 04 confirmation results
    copy_glob("results_rev/W[12]_*/*", "04_Confirmation_Results/runs", "results_rev")
    copy_glob("results_rev/pilot/W[12]_*__vs__*/**/*", "04_Confirmation_Results/pair_analyses", "results_rev/pilot")
    copy_glob("results_rev/pilot/evidence_matrix_confirmation/*", "04_Confirmation_Results/analyses", "results_rev/pilot")
    for f in ("results_rev/pilot/contamination/contamination_confirmation.json",
              "results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json"):
        copy(f, "04_Confirmation_Results/analyses", "results_rev/pilot")
    copy_glob("logs_rev/W[12]_*/update_log/manifest.jsonl", "04_Confirmation_Results/update_log_manifests", "logs_rev")
    copy_glob("logs_rev/W[12]_*/update_log/SEAL.json", "04_Confirmation_Results/update_log_manifests", "logs_rev")
    for f in ("W_runs.times", "W_gpu.csv", "W_analysis.log", "W_contamination.log", "preflight_W.json",
              "W1_clean.log", "W1_badnets.log", "W1_drift.log", "W2_clean.log", "W2_badnets.log", "W2_drift.log"):
        copy(f"logs_rev/console/{f}", "04_Confirmation_Results/console_logs")
    # 05 baselines
    copy_glob("results_rev/pilot/phase_s/*", "05_Baseline_Results/phase_s", "results_rev/pilot/phase_s")
    copy_glob("results_rev/pilot/benign_drift/*", "05_Baseline_Results/benign_drift", "results_rev/pilot/benign_drift")
    # 06 figures
    copy_glob("manuscript_c4ai4c27/figures/*.pdf", "06_Figures/workshop", "manuscript_c4ai4c27/figures")
    copy_glob("paper_assets/figures/**/*", "06_Figures/original_ndss", "paper_assets/figures")
    copy_glob("results_ndss/*/*.png", "06_Figures/original_ndss_extensions", "results_ndss")
    # 07 tables
    copy("manuscript_c4ai4c27/tables/table1_seed_level.tex", "07_Tables/workshop", None)
    for f in ("28_unified_outcome_table.md", "30_confirmation_outcome_table.md", "29_confirmation_results.md"):
        copy(f"paper_plan/{f}", "07_Tables/workshop")
    copy_glob("paper_assets/tables/**/*", "07_Tables/original_ndss", "paper_assets/tables")
    copy_glob("results_ndss/*/*.csv", "07_Tables/original_ndss_extensions", "results_ndss")
    # 08 scripts / source
    for pat in ("src/__init__.py", "src/analysis/__init__.py", "src/analysis/fedtrace_detector.py", "src/rev/*.py",
                "src/experiments/*.py", "src/federated/**/*.py", "src/models/*.py", "src/data/*.py",
                "src/training/*.py", "src/evaluation/*.py", "src/utils/*.py"):
        copy_glob(pat, "08_Analysis_Scripts", "")
    for pat in ("scripts/ndss_e*.py", "scripts/render_unified_outcomes.py", "scripts/build_evidence_manifest.py",
                "scripts/build_export_package.py", "tests/test_rev_*.py", "tests/rev_toy_federation.py",
                "manuscript_c4ai4c27/scripts/*.py"):
        copy_glob(pat, "08_Analysis_Scripts", "")
    # 09 configurations
    copy_glob("configs/*.yaml", "09_Configurations", "")
    copy_glob("configs/rev/*.yaml", "09_Configurations", "")
    # 10 LaTeX skeleton
    copy_glob("manuscript_c4ai4c27/**/*", "10_LaTeX_Skeleton/manuscript_c4ai4c27", "manuscript_c4ai4c27")
    # 11 references
    copy("manuscript_c4ai4c27/refs.bib", "11_References")
    copy("paper_plan/10_prior_work_checklist.md", "11_References")
    tz = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if tz and tz.exists():
        copy(str(tz.resolve()), "11_References/official_ICASSP2027_template")
    # 12 evidence manifest
    copy("paper_plan/31_evidence_manifest.sha256", "12_Evidence_Manifest")


# ------------------------------------------------------------------------------------------------ previews
def previews() -> None:
    sys.path.insert(0, str(ROOT))
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("m", ROOT / "manuscript_c4ai4c27/scripts/make_assets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = PKG / "06_Figures/workshop_png_previews"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    m.OUT = out
    orig = plt.Figure.savefig
    plt.Figure.savefig = lambda self, p, *a, **k: orig(self, str(p).replace(".pdf", ".png"), dpi=220)
    m.fig1()
    m.fig2(m.load_rows())
    plt.Figure.savefig = orig
    for p in (out / "figures").glob("*.png"):
        p.rename(out / p.name)
    (out / "figures").rmdir()


# ------------------------------------------------------------------------------------------------ derived CSV
def unified_csv() -> None:
    rows = []
    for name, f in (("development+archive", "results_rev/pilot/unified_outcomes/unified_outcomes.json"),
                    ("confirmation", "results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json")):
        for r in json.loads((ROOT / f).read_text())["rows"]:
            rows.append({"source": name, "trace": r["trace"], "family": r["family"], "intervened": r["intervened"],
                         "malicious": r["malicious"], "onset": r["onset"], "detector": r["detector"],
                         "any_confirmation": r["any_confirmation"], "first_attributed": r["first_attributed"],
                         "honest_false_accusations": ";".join(r["honest_false_accusations"]),
                         "attacker_detected": r.get("attacker_detected"), "attribution_correct": r.get("attribution_correct"),
                         "onset_error": r.get("onset_error"), "delay": r.get("delay"),
                         "final_top1": r["attack_effect"]["final_top1"], "asr": r["attack_effect"]["asr"]})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    write("07_Tables/workshop/unified_outcomes_all_traces.csv", buf.getvalue())


# ------------------------------------------------------------------------------------------------ inventories
def pdf_size(p: Path) -> str:
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", p.read_bytes())
    if not m:
        return "n/a"
    w, h = float(m.group(3)) - float(m.group(1)), float(m.group(4)) - float(m.group(2))
    return f"{w:.0f} x {h:.0f} pt ({w / 72 * 25.4:.0f} x {h / 72 * 25.4:.0f} mm)"


def png_size(p: Path) -> str:
    from PIL import Image
    with Image.open(p) as im:
        return f"{im.width} x {im.height} px"


def experiment_inventory() -> str:
    lines = ["# EXPORT FILE INVENTORY — experiments", "",
             "Generated by `scripts/build_export_package.py`. Absolute paths are on the PI's machine (repository root "
             f"`{WIN_ROOT}`); package paths are relative to the package root. Dataset CIFAR-100 and model ResNet-18 "
             "(CIFAR variant) for every run below. SHA-256 = hash of `client_round_trace.csv` (raw per-client, per-round "
             "metrics); full hashes of all files in `12_Evidence_Manifest/`.", "",
             "| Group | Experiment ID / run | Seed | Clients | Attack / intervention | Configuration | Result directory (repo-relative) | Absolute path | Raw metrics | Summary | Log | Supports | Trace SHA-256 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = []
    for p in sorted(glob.glob("results/federated/*/summary.json")) + sorted(glob.glob("results/centralized/*/summary.json")) + \
            sorted(glob.glob("results_ndss/*/*/summary.json")):
        groups.append(("ORIGINAL (NDSS archive)", Path(p).parent))
    for p in sorted(glob.glob("results_rev/X[1-4]_*/summary.json")):
        groups.append(("DEVELOPMENT (rev)", Path(p).parent))
    for p in sorted(glob.glob("results_rev/W[12]_*/summary.json")):
        groups.append(("CONFIRMATION (protocol 26)", Path(p).parent))
    for g, d in groups:
        s = json.loads((d / "summary.json").read_text())
        rel = d.as_posix()
        att = s.get("attack") or {}
        if d.name.startswith(("X", "W")):
            cfg = f"configs/rev/{d.name}.rev.yaml"
            kind = {"clean": "none", "badnets": "BadNets data poisoning (client_002, r20)", "relabel": "relabel-only poisoning (client_002, r20)",
                    "drift": "benign class-composition drift (client_002, r20)"}[d.name.split("_")[1]]
            exp = d.name
        else:
            cfg = next((c for c in (f"configs/{s.get('experiment_name')}.yaml",) if (ROOT / c).exists()), "see 09_Configurations")
            kind = "none" if not att or not att.get("enabled", att.get("type")) else \
                f"{att.get('type', att.get('attack_type', 'attack'))} (client {att.get('malicious_client_id', '?')}, onset {att.get('start_round', att.get('attack_start_round', '?'))})"
            exp = s.get("experiment_id", d.name)
        trace = d / "client_round_trace.csv"
        supports = ("Table 1 / Fig. 2 / claims C-01–C-04, C-07" if g.startswith("CONFIRMATION") else
                    "Table 1 / Figs. 1–2 / claims C-02–C-10" if g.startswith("DEVELOPMENT") else
                    ("NDSS tables / claim C-11" if trace.exists() else "utility context only"))
        lines.append(f"| {g} | {exp} | {s.get('seed')} | {s.get('num_clients', '–')} | {kind} | `{cfg}` | `{rel}` | `{win(rel)}` | "
                     f"{'`client_round_trace.csv`' if trace.exists() else '—'} | `summary.json`, `final_test_metrics.json` | "
                     f"{'`run.log`' if (d / 'run.log').exists() else '—'} | {supports} | {sha(trace)[:16] + '…' if trace.exists() else '—'} |")
    lines += ["", "## Analyses (paths repo-relative; all included in the package)", "",
              "| Analysis | Path | Protocol | Supports |", "|---|---|---|---|",
              "| Phase S baselines (ZP, CUSUM, PH, reference variants), calibration sensitivity | `results_rev/pilot/phase_s/` | 12 | thresholds; L-55–L-58, L-61 |",
              "| Benign drift evaluation | `results_rev/pilot/benign_drift/benign_drift.json` | 13 | C-03 |",
              "| Evidence matrix (dev / confirmation) | `results_rev/pilot/evidence_matrix*/` | — | ASR, accuracy, diagnostics |",
              "| Unified outcome tables | `results_rev/pilot/unified_outcomes/` | 25 §6 | Table 1, Fig. 2 |",
              "| Reference contamination | `results_rev/pilot/contamination/` | 25 (+A1, A2) | C-05, C-06, C-08 |",
              "| Pair analyses (prefix check, ASR per round, outcomes) | `results_rev/pilot/*__vs__*/` | 08, 09, 13, 26 | C-07, matched histories |",
              "| Original ablation (signal) and sensitivity (τ, P) | `paper_assets/tables/main/table03_signal_ablation.csv`, `results_ndss/E4_detector_sensitivity/` | NDSS | C-11 |",
              "| Falsified exploratory analyses | `results_rev/pilot/source_attribution/`, `results_rev/pilot/label_shift/*.json` | 14, 16 | Limitations |"]
    return "\n".join(lines) + "\n"


def figure_inventory() -> str:
    rows = ["# EXPORT FILE INVENTORY — figures", "",
            "| Figure | Package path | Format | Dimensions | Generation script | Source artifacts | Purpose / caption | Verified | Workshop use |",
            "|---|---|---|---|---|---|---|---|---|"]
    meta = {
        "fig1_reference_contamination": ("manuscript_c4ai4c27/scripts/make_assets.py", "X2_badnets_d2026 and W1_badnets traces",
                                         "reference-contamination mechanism (seed 2026) vs W1 (no crossing)", "yes (regenerated from frozen traces; hashes in ASSET_PROVENANCE.json)", "YES — Fig. 1"),
        "fig2_reliability_grid": ("manuscript_c4ai4c27/scripts/make_assets.py", "unified_outcomes*.json",
                                  "detector × seed outcome grid incl. honest alarms (BadNets, drift, clean)", "yes (as above)", "YES — Fig. 2 (if space)"),
    }
    for p in sorted((PKG / "06_Figures").rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".pdf", ".png", ".svg"):
            continue
        rel = p.relative_to(PKG).as_posix()
        key = p.stem
        if key in meta:
            gen, src, purpose, ver, use = meta[key]
        elif "original_ndss/" in rel:
            gen, src = "NDSS figure scripts (see `FIGURE_DATA_PROVENANCE.csv`)", "archived traces (see provenance CSV)"
            purpose, ver = "original NDSS figure (model-update perturbations / Multi-Krum)", "CLAUDE-REPORTED provenance; not re-verified pixel-wise"
            use = "context only (fig02 onset trajectory, fig03 signal complementarity)" if key.startswith(("fig02", "fig03")) else "no (out of scope / space)"
        else:
            gen, src, purpose = "`scripts/ndss_e*_postprocess.py`", "results_ndss traces", "NDSS extension (E1–E4) diagnostic plot"
            ver, use = "CLAUDE-REPORTED", "no (appendix-type)"
        dims = pdf_size(p) if p.suffix.lower() == ".pdf" else png_size(p) if p.suffix.lower() == ".png" else "n/a"
        rows.append(f"| {key} | `{rel}` | {p.suffix[1:].upper()} | {dims} | {gen} | {src} | {purpose} | {ver} | {use} |")
    return "\n".join(rows) + "\n"


def table_inventory() -> str:
    rows = ["# EXPORT FILE INVENTORY — tables", "", "| Table | Package path | Generation | Source data | Protocol | Verified | Manuscript relevance |",
            "|---|---|---|---|---|---|---|"]
    for p in sorted((PKG / "07_Tables").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PKG).as_posix()
        n = p.name
        if n == "table1_seed_level.tex":
            r = ("`manuscript_c4ai4c27/scripts/make_assets.py`", "unified_outcomes*.json", "12, 13, 25, 26", "yes (generated; provenance hashes)", "MAIN Table 1")
        elif n == "unified_outcomes_all_traces.csv":
            r = ("derived by `scripts/build_export_package.py` (flattened JSON, no recomputation)", "unified_outcomes*.json", "25 §6, 26", "yes (lossless flattening)", "source data for Table 1 / Fig. 2")
        elif n.startswith(("28_", "30_")):
            r = ("`scripts/render_unified_outcomes.py`", "unified_outcomes*.json", "25 §6 / 26", "yes", "full detector × trace tables (supplementary)")
        elif n.startswith("29_"):
            r = ("written from artifacts (Prompt 34)", "pair summaries, unified, contamination JSONs", "26", "CLAUDE-REPORTED", "development vs confirmation comparison")
        elif "original_ndss/" in rel:
            r = ("NDSS table scripts (see `TABLE_DATA_PROVENANCE.csv`)", "archived traces / summaries", "NDSS (frozen v3)", "CLAUDE-REPORTED; v3 attributions recomputed 2026-10-26", "context (perturbation results, ablation)")
        else:
            r = ("`scripts/ndss_e*_postprocess.py`", "results_ndss traces", "NDSS extensions", "CLAUDE-REPORTED", "context")
        rows.append(f"| {n} | `{rel}` | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    return "\n".join(rows) + "\n"


CLAIM_MAP = """# CLAIM → FIGURE / TABLE / EVIDENCE MAP

Status labels (this package): **VERIFIED** = checked by Claude against frozen artifacts and holding on the
preregistered confirmation seeds W1 / W2 (CLAUDE-REPORTED; no independent PI verification yet); **DEVELOPMENT-ONLY** =
seeds 123 / 2026 only; **CONDITIONAL** = exact algebra plus an explanation that holds under a stated precondition;
**UNVERIFIED** = not checked against a primary source; **UNSUPPORTED** = must not be claimed. Full wording:
`01_Research_Documents/paper_plan/32_final_claim_matrix.md`.

| Claim | Status | Experiment | Raw data | Summary | Figure | Table | Code | Limitations |
|---|---|---|---|---|---|---|---|---|
| C-01 FedTrace v3 missed BadNets on W1 and W2 (late at 123; honest clients first at 2026) | VERIFIED (W1, W2) + DEVELOPMENT | W1/W2_badnets; X2_badnets_d123/d2026 | `04_.../runs/W*_badnets/client_round_trace.csv`; `03_.../runs/X2_*` | `04_.../analyses/unified_outcomes_confirmation.json`; `29_confirmation_results.md` | Fig. 2 | Table 1 | `08_.../src/analysis/fedtrace_detector.py`, `src/rev/online.py`, `src/rev/unified_outcomes.py` | W2 ASR below criterion; attacker norm z peaked 2.9–3.6 vs τ = 3 |
| C-02 Norm z-score / CUSUM attributed BadNets on all four seeds but also alarmed an honest client on W1 | VERIFIED | X2_*, W*_badnets | traces as above | unified tables; `05_.../phase_s/phase_s.json` | Fig. 2 | Table 1 | `src/rev/baselines.py`, `src/rev/phase_s.py` | thresholds from development clean runs (L-57) |
| C-03 Benign drift caused honest false accusations (ZP, CUSUM on all four seeds; v3 at 2026 and W1) | VERIFIED | X4_drift_*, W*_drift | traces | `05_.../benign_drift/benign_drift.json`; unified tables | Fig. 2 | Table 1 | `src/rev/attacks.py` (BenignClassDriftDataset), `src/rev/benign_drift.py` | one drift type, one client; stress test |
| C-04 Threshold transfer failed: honest alarm in the W1 clean run | VERIFIED (one seed) | W1_clean | `04_.../runs/W1_clean/client_round_trace.csv` | unified confirmation JSON | Fig. 2 | Table 1 | as C-02 | W2 clean silent |
| C-05 Reference-driven misattribution observed only in seed 2026 | DEVELOPMENT-ONLY | X2/X3/X4 d2026; checked on 123, W1, W2 | traces | `03_.../analyses/contamination/contamination.json`, `04_.../analyses/contamination_confirmation.json` | Fig. 1 | Table 1 | `src/rev/contamination.py`, `src/rev/reference_audit.py` | three arms of one seed |
| C-06a Median-band identity, equal transfer, one-sidedness, necessity condition | CONDITIONAL (exact algebra; elementary) | all traces (identity error 0.0) | — | contamination JSONs; `25_reference_contamination_protocol.md` | — | — (equations in `10_LaTeX_Skeleton/main.tex`) | `src/rev/contamination.py` | elementary; not a contribution by itself |
| C-06b Accusations explained only when the intervened client starts inside the band (1 of 4 seeds) | CONDITIONAL | as C-05 | traces | contamination JSONs; `29_confirmation_results.md` §4 | Fig. 1 | — | as C-05 | not exercised on W1 / W2 |
| C-07 Attack effectiveness varied (ASR 43.5 / 64.7 / 63.3 / 44.2 %); two of four below 50 % | VERIFIED (seed-level) | X2_*, W*_badnets | pair `effect_*.csv` | pair `summary.json` | — | Table 1 | `src/rev/pilot.py` | criterion fixed in advance |
| C-08 Deployable contamination proxy did not generalise | VERIFIED (negative) | W runs | — | `contamination_confirmation.json` | — | — | `src/rev/contamination.py` | limitation only |
| C-09 Detection ≠ attribution at 2026 BadNets | DEVELOPMENT-ONLY | X2_badnets_d2026 | trace | unified JSON | Fig. 1(b), Fig. 2 | Table 1 | as C-01 | one run |
| C-10 v3 channel ablation (magnitude-only vs t_res-only) | DEVELOPMENT + VERIFIED misses | X / W runs | traces | `28_*.md`, `30_*.md` | Fig. 2 | Table 1 | `src/rev/contamination.py::v3_alarms` | ablation of the evaluated framework |
| C-11 Original FedTrace on model-update perturbations: attributed, onset error 0, delay 2 (except reduced-severity Gaussian, error 21) | VERIFIED against archived traces (recomputed); not data poisoning | EXP_014–019, 026, 027; E1–E3 | `02_.../results/federated/*/client_round_trace.csv`, `02_.../results_ndss/*/*/` | `07_Tables/original_ndss/main/table02*.csv`, `table05*`, `results_ndss/*summary*` | `06_Figures/original_ndss/main/fig02*, fig03*` | `07_Tables/original_ndss/...` | `src/analysis/fedtrace_detector.py` | dirty-tree provenance for E1–E3 / EXP_020–025 (L-17); same seeds 123 / 2026 |
| C-11b Original signal ablation (magnitude-only late on directional; t_res-only fast) | VERIFIED against archived traces (table reproduced in doc 28) | EXP_016–019 | traces | `table03_signal_ablation.csv` | fig03 | table03 | detector code | perturbation attacks only |
| C-12 Historical-reference variant accuses honest clients in every clean run | VERIFIED | X1, W clean | traces | unified tables | — | `28/30_*.md` | `src/rev/reference_audit.py` | variant, not v3 |
| C-13 Malice not identifiable from updates without restricting benign data | CONDITIONAL (analytic) | — | — | `15_identifiability_and_research_directions.md` (A3–A4) | — | — | — | not a universal theorem |
| FLTracer / FLDetector / DeepSight / FLForensics statements | UNVERIFIED beyond the parts read (see `11_References/10_prior_work_checklist.md`) | not executed | — | — | — | — | — | no numerical comparison permitted |
| Superiority, rates, mechanism confirmed, generalisation, novelty of the decomposition, D1 / label-shift as methods | UNSUPPORTED | — | — | — | — | — | — | see claim matrix U-01–U-07 |
"""


README = """# ICASSP 2027 C4AI4C — Research Evidence Package (FedTrace)

Prepared 2026-10-27 for manuscript preparation by the PI (single target: ICASSP 2027 C4AI4C Satellite Workshop,
deadline 11 Nov 2026). **Evidence only** — no manuscript text, no new experiment, no modified result. Every copied
file is byte-identical to its source (hash-checked at copy time); `12_Evidence_Manifest/PACKAGE_SHA256SUMS.txt` lists
every file in this package.

Start here:
1. `00_README/CLAIM_TO_FIGURE_TABLE_MAP.md` — what may be claimed, with status and evidence.
2. `01_Research_Documents/paper_plan/32_final_claim_matrix.md`, `29_confirmation_results.md`, `33_…`, `34_…`, `35_…`, `36_…`.
3. `07_Tables/workshop/table1_seed_level.tex` and `06_Figures/workshop/*.pdf` (PNG previews alongside).
4. `10_LaTeX_Skeleton/` — the author-ready skeleton with the official ICASSP 2027 template files.

| Folder | Content |
|---|---|
| 00_README | this file, inventories (experiments, figures, tables), claim map, missing items, security scan |
| 01_Research_Documents | planning / evidence documents (paper_plan 01–36), original FedTrace method and analysis documents, Research_Database |
| 02_Original_FedTrace_Results | archived NDSS results: `results/` (26 federated + 2 centralized runs) and `results_ndss/` (E1–E4) |
| 03_Development_Results | 8 development runs (seeds 123 / 2026), pair analyses, analyses, update-log manifests |
| 04_Confirmation_Results | 6 confirmation runs (W1 = 1178015705, W2 = 1463065807), pair analyses, analyses, console logs, update-log manifests |
| 05_Baseline_Results | Phase S statistical baselines and calibration sensitivity; benign-drift evaluation |
| 06_Figures | workshop figures (PDF + PNG previews); original NDSS figures and extension plots |
| 07_Tables | workshop Table 1 (LaTeX), unified outcome tables (Markdown, CSV), original NDSS tables (CSV, LaTeX) |
| 08_Analysis_Scripts | detector, baselines, evaluation, figure / table generators, training pipeline, tests |
| 09_Configurations | every experiment configuration (NDSS, development, confirmation) |
| 10_LaTeX_Skeleton | `manuscript_c4ai4c27/` copy (not compiled — no TeX engine on the source machine) |
| 11_References | `refs.bib` (verified entries; TODO-VERIFY marks), prior-work checklist, official template zip |
| 12_Evidence_Manifest | original-path SHA-256 manifest (370 files), large artifacts not included (paths + hashes), package manifest |

Labels used throughout: CLAUDE-REPORTED (checked by Claude from artifacts), DEVELOPMENT ONLY, CONFIRMATION,
POST-HOC EXPLORATORY. No result is PI-verified. AI-use disclosure is required for the paper (doc 36 §4).
"""


MISSING = """# Missing or incomplete items (recorded, not silently omitted)

| Item | Status | Where / what to do |
|---|---|---|
| Original NDSS FedTrace manuscript (LaTeX / PDF) | **not in the repository** | the PI provides it separately |
| Update logs (fp32 states; 14 runs x ~13 GiB) | excluded (size) | paths + SEAL manifest hashes: `12_Evidence_Manifest/LARGE_ARTIFACTS_NOT_INCLUDED.md`; manifests and seals included |
| Model checkpoints | excluded (size; not evidence for any claim) | paths in the same file |
| Label-shift probe logits (1.3 GiB; falsified exploratory analysis) | excluded | indexes and result JSONs included |
| Compiled PDF of the LaTeX skeleton | **not available** — no TeX engine on the source machine | compile on TeX Live / MiKTeX / Overleaf |
| Bibliography | incomplete: `TODO-VERIFY` fields (DOIs, pages, given names); Krum, ResNet, CIFAR-100, Page-Hinkley **unverified placeholders** | `11_References/refs.bib` |
| FLTracer / FLDetector / DeepSight / FLForensics results | **not executed** (no numbers exist) | qualitative positioning only |
| Relabel-only confirmation runs | not run (protocol 26 had three arms) | development results only |
| ASR for archived perturbation runs | not applicable (no trigger) | — |
| Clean FedAvg traces for EXP_008 / EXP_009 | never logged in the archive (L-18) | development clean runs X1 serve as clean controls |
| Development run X4_drift_d2026 | provenance flag `execution_clean = False` (documentation-only edit during the run; accepted, L-68) | disclose if used |
| ICASSP 2027: deadline time zone, supplementary-material policy, ICASSP dual-submission wording, 2027 workshop topic list | unresolved | `01_Research_Documents/paper_plan/36_publication_policy_audit.md` |
| PI-independent verification of results | none of the results is PI-verified; all are CLAUDE-REPORTED | — |
"""


def large_artifacts() -> str:
    lines = ["# Large artifacts NOT included (exact paths and hashes)", "",
             "Recorded instead of copied. Update-log manifests and seals ARE included (they hash every logged state).", "",
             "| Artifact | Repo-relative path | Absolute path | Size | Integrity reference |", "|---|---|---|---|---|"]
    for d in sorted(glob.glob("logs_rev/[XW]*/update_log")):
        size = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
        seal = json.loads((Path(d) / "SEAL.json").read_text())
        lines.append(f"| update log (fp32 states) | `{d}` | `{win(d)}` | {size / 2**30:.1f} GiB | manifest SHA-256 `{seal['manifest_sha256']}` (SEAL.json) |")
    for d in sorted(glob.glob("checkpoints_rev/[XW]*")):
        size = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
        lines.append(f"| model checkpoints | `{d}` | `{win(d)}` | {size / 2**20:.0f} MiB | not hashed here (not evidence for any claim) |")
    lg = Path("results_rev/pilot/label_shift/logits")
    if lg.exists():
        size = sum(f.stat().st_size for f in lg.rglob("*") if f.is_file())
        idx = {p.name: sha(p) for p in lg.glob("logits_index_*.json")}
        lines.append(f"| label-shift probe logits (falsified exploratory analysis) | `{lg.as_posix()}` | `{win(lg.as_posix())}` | {size / 2**30:.2f} GiB | indexes (copied): {idx} |")
    lines.append(f"| archived NDSS checkpoints | `checkpoints/`, `checkpoints_ndss/` | `{win('checkpoints')}` | large | frozen NDSS archive; not needed for any claim |")
    return "\n".join(lines) + "\n"


SECRET_PATTERNS = [
    (r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{8,}", "credential assignment"),
    (r"sk-[A-Za-z0-9]{20,}", "API key (sk-)"), (r"ghp_[A-Za-z0-9]{30,}", "GitHub token"), (r"AKIA[0-9A-Z]{16}", "AWS key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"), (r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "e-mail address"),
]


def scan() -> tuple[str, list]:
    hits = []
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith(".env") or p.suffix in EXCLUDE_SUFFIX:
            hits.append((p.relative_to(PKG).as_posix(), "forbidden file type", ""))
            continue
        if p.suffix.lower() in (".png", ".pdf", ".xlsx", ".zip", ".eps", ".doc"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat, label in SECRET_PATTERNS:
            for m in re.finditer(pat, text):
                hits.append((p.relative_to(PKG).as_posix(), label, m.group(0)[:80]))
    return "", hits


def main() -> int:
    if PKG.exists():
        shutil.rmtree(PKG)          # only the generated package directory is ever removed
    if ZIP.exists():
        ZIP.unlink()
    PKG.mkdir(parents=True)
    collect()
    previews()
    unified_csv()
    write("00_README/README.md", README)
    write("00_README/EXPORT_FILE_INVENTORY.md", experiment_inventory() + "\n" + figure_inventory() + "\n" + table_inventory())
    write("00_README/CLAIM_TO_FIGURE_TABLE_MAP.md", CLAIM_MAP)
    write("12_Evidence_Manifest/LARGE_ARTIFACTS_NOT_INCLUDED.md", large_artifacts())
    write("00_README/MISSING_ITEMS.md", MISSING)
    write("12_Evidence_Manifest/SOURCE_COPY_VERIFICATION.tsv",
          "source\tpackage_path\tsha256\n" + "\n".join(f"{a}\t{b}\t{c}" for a, b, c in copied) + "\n")
    _, hits = scan()
    report = ["# Security / privacy scan", "", f"Files scanned: {sum(1 for p in PKG.rglob('*') if p.is_file())}", "",
              "| File | Finding | Match (truncated) |", "|---|---|---|"]
    report += [f"| `{f}` | {k} | `{m}` |" for f, k, m in hits] or ["| — | none | — |"]
    write("00_README/SECURITY_SCAN.md", "\n".join(report) + "\n")
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and p.name != "PACKAGE_SHA256SUMS.txt")
    write("12_Evidence_Manifest/PACKAGE_SHA256SUMS.txt",
          "\n".join(f"{sha(p)} *{p.relative_to(PKG).as_posix()}" for p in files) + "\n")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                z.write(p, f"{PKG_NAME}/{p.relative_to(PKG).as_posix()}")
    print(json.dumps({"copied": len(copied), "files_in_package": len(files) + 1, "secret_scan_hits": len(hits),
                      "zip_mb": round(ZIP.stat().st_size / 2**20, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
