"""Generate the final publication table package for the FedTrace-FM paper
(Phase P5).

Reads ONLY existing experiment artifacts under results/ (via the frozen,
unmodified src/analysis/fedtrace_detector.py module for FedTrace-derived
quantities, and raw CSV/JSON/YAML artifacts directly for everything else).
Never trains a model, runs an attack, or re-tunes any detector parameter.
Writes only under paper_assets/tables/ -- no source/config/result/checkpoint/
Research-Database file is touched.

Usage:
    python scripts/generate_paper_tables.py

Repository-relative paths throughout. Fails loudly (raises TableDataError)
if a required artifact is missing or a live-computed value contradicts an
already-published expectation, rather than silently producing an incorrect
table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "federated"
CONFIGS_DIR = REPO_ROOT / "configs"
TABLES_ROOT = REPO_ROOT / "paper_assets" / "tables"
MAIN_DIR = TABLES_ROOT / "main"
APPENDIX_DIR = TABLES_ROOT / "appendix"
SOURCE_DATA_DIR = TABLES_ROOT / "source_data"
LATEX_DIR = TABLES_ROOT / "latex"
REGISTRY_CSV = REPO_ROOT / "analysis" / "FEDTRACE_EXPERIMENT_REGISTRY.csv"

sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis import fedtrace_detector as ft  # noqa: E402  (frozen, unmodified module)

NOT_APPLICABLE = "NOT APPLICABLE"
NOT_RECORDED = "NOT RECORDED"
DASH = "--"  # LaTeX missing-value marker, mapped from NOT_APPLICABLE/NOT_RECORDED

EXP_DIRS = {
    "EXP_001": ("centralized", "cifar100_resnet18"),
    "EXP_002": ("federated", "cifar100_fedavg_iid"),
    "EXP_003": ("federated", "cifar100_fedavg_iid_20rounds"),
    "EXP_004": ("federated", "cifar100_fedavg_iid_50rounds"),
    "EXP_005": ("federated", "cifar100_fedavg_iid_50rounds_cosine"),
    "EXP_006": ("federated", "cifar100_fedavg_dirichlet_a05_50rounds"),
    "EXP_007": ("federated", "cifar100_fedavg_dirichlet_a01_50rounds"),
    "EXP_008": ("federated", "cifar100_fedavg_dirichlet_a01_seed123_50rounds"),
    "EXP_009": ("federated", "cifar100_fedavg_dirichlet_a01_seed2026_50rounds"),
    "EXP_010": ("federated", "cifar100_fedprox_dirichlet_a01_mu001_50rounds"),
    "EXP_011": ("federated", "cifar100_fedprox_dirichlet_a01_mu001_seed123_50rounds"),
    "EXP_012": ("federated", "cifar100_fedprox_dirichlet_a01_mu001_seed2026_50rounds"),
    "EXP_014": ("federated", "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42"),
    "EXP_015": ("federated", "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed42"),
    "EXP_016": ("federated", "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123"),
    "EXP_017": ("federated", "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed2026"),
    "EXP_018": ("federated", "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123"),
    "EXP_019": ("federated", "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed2026"),
    "EXP_020": ("federated", "cifar100_multikrum_dirichlet_a01_seed123_50rounds"),
    "EXP_021": ("federated", "cifar100_multikrum_dirichlet_a01_seed2026_50rounds"),
    "EXP_022": ("federated", "cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed123"),
    "EXP_023": ("federated", "cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed2026"),
    "EXP_024": ("federated", "cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed123"),
    "EXP_025": ("federated", "cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed2026"),
    "EXP_026": ("federated", "cifar100_fedavg_dirichlet_a01_direction_attack_client004_round20_seed123"),
    "EXP_027": ("federated", "cifar100_fedavg_dirichlet_a01_direction_attack_client002_round30_seed123"),
}


class TableDataError(RuntimeError):
    pass


def exp_dir(exp_id: str) -> Path:
    if exp_id not in EXP_DIRS:
        raise TableDataError(f"Unknown experiment id {exp_id!r}")
    category, name = EXP_DIRS[exp_id]
    path = REPO_ROOT / "results" / category / name
    if not path.exists():
        raise TableDataError(f"{exp_id}: result directory not found: {path}")
    return path


def load_summary(exp_id: str) -> dict:
    path = exp_dir(exp_id) / "summary.json"
    if not path.exists():
        raise TableDataError(f"{exp_id}: summary.json not found at {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_config(exp_id: str) -> dict:
    _, name = EXP_DIRS[exp_id]
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise TableDataError(f"{exp_id}: config not found at {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_trace_rows(exp_id: str) -> list[dict]:
    path = exp_dir(exp_id) / "client_round_trace.csv"
    if not path.exists():
        raise TableDataError(f"{exp_id}: client_round_trace.csv not found at {path}")
    return ft.load_trace_rows(path)


def load_multikrum_rows(exp_id: str) -> list[dict]:
    path = exp_dir(exp_id) / "multikrum_diagnostics.csv"
    if not path.exists():
        raise TableDataError(f"{exp_id}: multikrum_diagnostics.csv not found at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_id"] = int(row["round_id"])
        row["multikrum_rejected"] = row["multikrum_rejected"].strip().lower() == "true"
    return rows


def load_partition_summary(exp_id: str) -> dict[str, int]:
    path = exp_dir(exp_id) / "client_partition_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["client_id"]: int(row["num_samples"]) for row in csv.DictReader(handle)}


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}"


PROVENANCE: list[dict] = []


def record_provenance(table_id, row_id, experiment_id, source_file, source_field,
                       transformation, notes="") -> None:
    PROVENANCE.append({
        "table_id": table_id, "row_id": row_id, "experiment_id": experiment_id,
        "source_file": source_file, "source_field": source_field,
        "transformation": transformation, "notes": notes,
    })


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def latex_escape(value) -> str:
    text = str(value)
    return (text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&"))


def write_latex(path: Path, caption: str, label: str, col_spec: str,
                 header: list[str], rows: list[list], bold_cells: set[tuple[int, int]] | None = None,
                 note: str | None = None, wide: bool = False) -> None:
    """``wide=True`` wraps the tabular in a two-column-spanning ``table*``
    environment plus ``\\resizebox{\\textwidth}{!}{...}`` -- reserved for
    tables genuinely too wide for a single ICML/ICLR column (>=12 columns
    here), per this task's "use resizebox only if genuinely necessary"
    instruction; every narrower table is left unscaled."""
    bold_cells = bold_cells or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    env = "table*" if wide else "table"
    lines = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(h) for h in header) + r" \\")
    lines.append(r"\midrule")
    for r_idx, row in enumerate(rows):
        cells = []
        for c_idx, cell in enumerate(row):
            text = DASH if cell in (NOT_APPLICABLE, NOT_RECORDED, None) else latex_escape(cell)
            if (r_idx, c_idx) in bold_cells:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if wide:
        lines.append(r"}")
    if note:
        lines.append(rf"\vspace{{2pt}}\par\raggedright\footnotesize {note}")
    lines.append(rf"\end{{{env}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit(table_id: str, main: bool, caption: str, label: str, col_spec: str,
          header: list[str], rows: list[list], bold_cells: set[tuple[int, int]] | None = None,
          note: str | None = None, wide: bool | None = None) -> None:
    target_dir = MAIN_DIR if main else APPENDIX_DIR
    write_csv(target_dir / f"{table_id}.csv", header, rows)
    if wide is None:
        wide = len(header) >= 12
    write_latex(LATEX_DIR / f"{table_id}.tex", caption, label, col_spec, header, rows, bold_cells, note, wide)


# ---------------------------------------------------------------------------
# Table 1 -- experimental setup
# ---------------------------------------------------------------------------


def table01_experimental_setup() -> None:
    cfg018 = load_config("EXP_018")
    krum_meta020 = None
    mk_diag = exp_dir("EXP_020") / "multikrum_diagnostics.csv"
    with mk_diag.open(newline="", encoding="utf-8") as handle:
        first_round_rows = [r for r in csv.DictReader(handle) if int(r["round_id"]) == 0]
    n_clients_020 = len(first_round_rows)
    # f is a fixed constructor argument (src/federated/aggregation/multikrum.py,
    # f=1 for this project's frozen baseline); m = n - f is read directly from
    # this run's own round-0 selected count, not assumed.
    m_selected = sum(1 for r in first_round_rows if r["multikrum_selected"].strip().lower() == "true")
    f_value = n_clients_020 - m_selected

    rows = [
        ["Dataset", "CIFAR-100"],
        ["Model", cfg018.get("model", NOT_RECORDED)],
        ["Number of clients", cfg018.get("num_clients", NOT_RECORDED)],
        ["Participation", f"{cfg018.get('clients_per_round', NOT_RECORDED)} / {cfg018.get('num_clients', NOT_RECORDED)} (full)"],
        ["Communication rounds", cfg018.get("communication_rounds", NOT_RECORDED)],
        ["Local epochs", cfg018.get("local_epochs", NOT_RECORDED)],
        ["Optimizer", str(cfg018.get("optimizer", NOT_RECORDED)).upper()],
        ["Learning rate", cfg018.get("learning_rate", NOT_RECORDED)],
        ["Batch size", cfg018.get("batch_size", NOT_RECORDED)],
        ["Partition", cfg018.get("partition", NOT_RECORDED)],
        [r"Dirichlet $\alpha$", cfg018.get("dirichlet_alpha", NOT_RECORDED)],
        ["Attack onset (main validation)", cfg018.get("attack_start_round", NOT_RECORDED)],
        ["Attacker (main validation)", cfg018.get("malicious_client_id", NOT_RECORDED)],
        [r"Gaussian attack strength $\rho$", "1.0"],
        [r"Directional target cosine $c$", "0.0"],
        [r"FedTrace threshold $\tau$", ft.DEFAULT_THRESHOLD],
        ["FedTrace persistence $P$", ft.DEFAULT_PERSISTENCE],
        ["FedTrace calibration window $T_{\\mathrm{ref}}$", ft.DEFAULT_T_REF],
        ["Multi-Krum $f$", f_value],
        ["Multi-Krum $m = n-f$", m_selected],
    ]
    emit("table01_experimental_setup", True,
         "Experimental setup shared across the main validation, external-baseline, and "
         "generalization experiments.", "tab:setup", "lX",
         ["Setting", "Value"], rows)
    for setting, value in rows:
        record_provenance("table01_experimental_setup", setting, "EXP_018 (config); EXP_020 (Multi-Krum f/m)",
                           "configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123.yaml; "
                           "results/federated/.../EXP_020/multikrum_diagnostics.csv",
                           "direct read" if setting not in
                           (r"FedTrace threshold $\tau$", "FedTrace persistence $P$",
                            "FedTrace calibration window $T_{\\mathrm{ref}}$") else
                           "src/analysis/fedtrace_detector.py module constant",
                           "Gaussian rho / directional c are fixed attack-construction "
                           "parameters, not read from a single config field, since every "
                           "Gaussian/directional run in the main series uses the same values.")


# ---------------------------------------------------------------------------
# Table 2 -- frozen FedTrace v3 validation (EXP_016-019 only)
# ---------------------------------------------------------------------------


def dominant_signal_for_client(exp_id: str, client_id: str) -> str:
    """Dominant-signal share among the malicious client's POST-ONSET anomalous
    rounds only. Restricting to round_id >= attack_start_round deliberately
    excludes the documented, pre-attack round-0/round-1 self-inclusive
    calibration artifact (analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md Section
    7) from being conflated with which signal actually drove the genuine,
    attack-driven detection -- matching the exact methodology already used in
    every EXP_016-019/026/027 analysis document (e.g. "z_t_res dominant,
    30/30 post-attack rounds")."""
    trace_rows = load_trace_rows(exp_id)
    attack_rows = [r for r in trace_rows if r.get("attack_configured")]
    onset = attack_rows[0]["attack_start_round"] if attack_rows else None
    if onset is None:
        return NOT_RECORDED
    by_client = {}
    for row in ft.compute_all_scores_v3(trace_rows):
        by_client.setdefault(row.client_id, []).append(row)
    rows = sorted(by_client[client_id], key=lambda r: r.round_id)
    m_wins = t_wins = 0
    total = 0
    for row in rows:
        if row.round_id < onset:
            continue
        combined = row.combined_score(("m", "t_res"))
        if combined is None or combined[0] <= ft.DEFAULT_THRESHOLD:
            continue
        total += 1
        if combined[1] == "m":
            m_wins += 1
        else:
            t_wins += 1
    if total == 0:
        return NOT_RECORDED
    if m_wins == total:
        return rf"$z^m$ ({m_wins}/{total}, 100\%)"
    if t_wins == total:
        return rf"$z^t_{{\mathrm{{res}}}}$ ({t_wins}/{total}, 100\%)"
    return rf"mixed ($z^m$ {m_wins}/{total}, $z^t_{{\mathrm{{res}}}}$ {t_wins}/{total})"


def validation_row(exp_id: str, attack_label: str, seed: int) -> tuple[list, bool]:
    rows_data = load_trace_rows(exp_id)
    result = ft.evaluate_detector_v3(rows_data, "H_fedtrace_v3", ("m", "t_res"))
    true_client = result.true_malicious_client
    predicted = result.predicted_client
    per_client = result.per_client[predicted] if predicted else None
    first_anom = per_client.first_anomalous_round if per_client else NOT_RECORDED
    confirm = per_client.confirmation_round if per_client else NOT_RECORDED
    onset_est = per_client.onset_estimate if per_client else NOT_RECORDED
    top1 = pct(load_summary(exp_id)["final_test"]["top1"])
    all_pass = (
        predicted == true_client and result.onset_localization_error == 0
        and result.detection_delay == 2 and len(result.false_positive_clients) == 0
    )
    row = [
        exp_id, attack_label, seed, true_client, predicted, first_anom, confirm,
        result.attack_start_round, onset_est, result.onset_localization_error,
        result.detection_delay, len(result.false_positive_clients),
        dominant_signal_for_client(exp_id, true_client), top1,
    ]
    return row, all_pass


def table02_fedtrace_validation() -> None:
    header = ["Experiment", "Attack", "Seed", "True attacker", "Predicted attacker",
              "First anomaly", "Confirmation", "True onset", "Estimated onset",
              "Onset error", "Detection delay", "Honest persistent FP",
              "Dominant signal", "Final Top-1 (%)"]
    runs = [("EXP_016", "Gaussian", 123), ("EXP_017", "Gaussian", 2026),
            ("EXP_018", "Directional", 123), ("EXP_019", "Directional", 2026)]
    rows, bold = [], set()
    for i, (exp_id, attack, seed) in enumerate(runs):
        row, all_pass = validation_row(exp_id, attack, seed)
        rows.append(row)
        if all_pass:
            bold.add((i, 9))  # onset error column
            bold.add((i, 10))  # detection delay column
            bold.add((i, 11))  # honest FP column
        record_provenance("table02_fedtrace_validation", exp_id, exp_id,
                           "results/federated/.../client_round_trace.csv, summary.json",
                           "update_l2_norm, self_prev_round_cosine, final_test.top1",
                           "src/analysis/fedtrace_detector.py::evaluate_detector_v3 (frozen, unmodified)",
                           "Development runs EXP_014/EXP_015 deliberately excluded -- "
                           "independent frozen-formula validation only.")
    note = ("4/4 runs: correct attribution, onset error 0, detection delay 2 (the "
            "theoretical minimum under $P=3$), zero honest persistent false positives. "
            "``First anomaly''=0 in every row reflects a documented, non-persistent, "
            "self-inclusive calibration artifact at round 0 (not attack-related; see "
            "FEDTRACE\\_MASTER\\_RESEARCH\\_STATUS.md Sec.~7) -- it is reported here, not "
            "hidden, and is why Estimated onset (not First anomaly) is the column that "
            "matches True onset. No mean/std row is reported ($n=2$ seeds per attack "
            "family does not support a pre-registered summary statistic).")
    emit("table02_fedtrace_validation", True,
         "Frozen FedTrace v3, independent validation (unseen seeds, formulas fixed before "
         "these traces existed).", "tab:fedtrace_validation", "lllllccccccccc",
         header, rows, bold, note)


# ---------------------------------------------------------------------------
# Table 3 -- signal ablation / coverage
# ---------------------------------------------------------------------------


def family_summary(exp_a: str, exp_b: str, components: tuple[str, ...], evaluator) -> str:
    results = []
    for exp_id in (exp_a, exp_b):
        rows_data = load_trace_rows(exp_id)
        result = evaluator(rows_data, "tmp", components)
        if not result.evaluable:
            results.append("not evaluable")
            continue
        if result.predicted_client != result.true_malicious_client:
            results.append("wrong client")
            continue
        oe = result.onset_localization_error
        dl = result.detection_delay
        results.append(f"err {oe}/delay {dl}" if oe is not None else NOT_RECORDED)
    if results[0] == results[1]:
        return results[0]
    return f"seed123: {results[0]}; seed2026: {results[1]}"


def table03_signal_ablation() -> None:
    detectors = [
        ("Peer-only", ft.evaluate_detector, ("p",)),
        ("Magnitude-only", ft.evaluate_detector_v3, ("m",)),
        ("Temporal-residual-only", ft.evaluate_detector_v3, ("t_res",)),
        ("Full FedTrace v3 (magnitude + temporal-residual)", ft.evaluate_detector_v3, ("m", "t_res")),
    ]
    header = ["Detector", "Gaussian attribution", "Gaussian onset err/delay",
              "Directional attribution", "Directional onset err/delay",
              "Honest persistent FP (4 runs)", "Interpretation"]
    rows = []
    for name, evaluator, components in detectors:
        gaussian_correct = []
        directional_correct = []
        honest_fp_total = 0
        for exp_id in ("EXP_016", "EXP_017", "EXP_018", "EXP_019"):
            rows_data = load_trace_rows(exp_id)
            result = evaluator(rows_data, "tmp", components)
            if not result.evaluable:
                continue
            honest_fp_total += len(result.false_positive_clients)
            correct = result.predicted_client == result.true_malicious_client
            (gaussian_correct if exp_id in ("EXP_016", "EXP_017") else directional_correct).append(correct)
        gaussian_attr = "Yes (2/2)" if all(gaussian_correct) and gaussian_correct else (
            "Partial" if any(gaussian_correct) else "No (0/2)")
        directional_attr = "Yes (2/2)" if all(directional_correct) and directional_correct else (
            "Partial" if any(directional_correct) else "No (0/2)")
        g_onset = family_summary("EXP_016", "EXP_017", components, evaluator)
        d_onset = family_summary("EXP_018", "EXP_019", components, evaluator)
        if name == "Peer-only":
            interp = "Never contributes a detection in any development or validation run"
        elif name == "Magnitude-only":
            interp = "Fast/correct for Gaussian; correct but far too late for directional"
        elif name == "Temporal-residual-only":
            interp = "Fast/correct for directional; matches or nearly matches full detector for Gaussian"
        else:
            interp = "Matches whichever single signal is informative -- coverage, not fusion"
        rows.append([name, gaussian_attr, g_onset, directional_attr, d_onset, honest_fp_total, interp])
        record_provenance("table03_signal_ablation", name, "EXP_016, EXP_017, EXP_018, EXP_019",
                           "results/federated/.../client_round_trace.csv",
                           "update_l2_norm, self_prev_round_cosine, peer_mean_cosine_similarity",
                           f"src/analysis/fedtrace_detector.py::{evaluator.__name__} (frozen, unmodified), "
                           f"components={components}")
    note = ("Message: the two signals provide complementary attack-family coverage. This table "
            "does NOT support, and does not claim, that the combiner improves over the best "
            "single detector -- in every run checked, the full-detector outcome matches "
            "whichever single component was already informative for that run.")
    emit("table03_signal_ablation", True,
         "Signal ablation across the four independent validation runs.", "tab:ablation",
         "lllllcp{3.2cm}", header, rows, note=note)


# ---------------------------------------------------------------------------
# Table 4 -- Multi-Krum external baseline (EXP_020-025)
# ---------------------------------------------------------------------------


def multikrum_proxy(rows_mk: list[dict]) -> tuple[str, int | str]:
    confirmations = {}
    for client_id in sorted({r["client_id"] for r in rows_mk}):
        streak = 0
        client_rows = sorted((r for r in rows_mk if r["client_id"] == client_id), key=lambda r: r["round_id"])
        for r in client_rows:
            if r["multikrum_rejected"]:
                streak += 1
                if streak >= 3:
                    confirmations[client_id] = r["round_id"]
                    break
            else:
                streak = 0
    if not confirmations:
        return NOT_APPLICABLE, NOT_APPLICABLE
    client = min(confirmations, key=lambda c: (confirmations[c], c))
    return client, confirmations[client]


def table04_multikrum_baseline() -> None:
    header = ["Attack", "Seed", "Clean exp.", "Attacked exp.", "Attacker rejection rate (%)",
              "Persistent honest rejected client (clean)", "Rejection switch?",
              "Proxy predicted client", "Proxy attribution correct?", "FedTrace attribution correct?",
              "Top-1 Multi-Krum (%)", "Top-1 attacked FedAvg (%)", r"$\Delta$ Top-1 (MK $-$ FedAvg)"]
    conditions = [
        ("Gaussian", 123, "EXP_020", "EXP_022", "EXP_016"),
        ("Gaussian", 2026, "EXP_021", "EXP_023", "EXP_017"),
        ("Directional", 123, "EXP_020", "EXP_024", "EXP_018"),
        ("Directional", 2026, "EXP_021", "EXP_025", "EXP_019"),
    ]
    rows = []
    for attack, seed, clean_exp, attacked_exp, fedavg_exp in conditions:
        rows_mk = load_multikrum_rows(attacked_exp)
        summary = load_summary(attacked_exp)
        attacker = summary["attack"]["malicious_client_id"]
        onset = summary["attack"]["attack_start_round"]
        post = [r for r in rows_mk if r["client_id"] == attacker and r["round_id"] >= onset]
        rejected = sum(1 for r in post if r["multikrum_rejected"])
        rate = rejected / len(post) * 100

        clean_rows = load_multikrum_rows(clean_exp)
        pre_counts = {}
        for r in clean_rows:
            if r["multikrum_rejected"]:
                pre_counts[r["client_id"]] = pre_counts.get(r["client_id"], 0) + 1
        persistent_client = max(pre_counts, key=lambda c: (pre_counts[c], c)) if pre_counts else NOT_APPLICABLE

        switch = "Yes" if rate == 100.0 else "No"
        proxy_client, proxy_round = multikrum_proxy(rows_mk)
        proxy_correct = "No" if proxy_client != attacker else "Yes"

        ft_result = ft.evaluate_detector_v3(load_trace_rows(fedavg_exp), "H", ("m", "t_res"))
        ft_correct = "Yes" if ft_result.predicted_client == ft_result.true_malicious_client else "No"

        top1_mk = pct(summary["final_test"]["top1"])
        top1_fedavg = pct(load_summary(fedavg_exp)["final_test"]["top1"])
        delta = f"{float(top1_mk) - float(top1_fedavg):+.2f}"

        rows.append([attack, seed, clean_exp, attacked_exp, f"{rate:.0f}", persistent_client,
                     switch, proxy_client, proxy_correct, ft_correct, top1_mk, top1_fedavg, delta])
        record_provenance("table04_multikrum_baseline", f"{attack}/seed{seed}", attacked_exp,
                           "results/federated/.../multikrum_diagnostics.csv, summary.json",
                           "round_id, client_id, multikrum_rejected, final_test.top1",
                           "P_krum=3 consecutive-rejection persistence proxy applied to raw diagnostics "
                           "(same rule as Research_Database/sync_master_results.py, Phase P2); "
                           "FedTrace column via src/analysis/fedtrace_detector.py::evaluate_detector_v3 "
                           "on the paired FedAvg run")
    note = ("Multi-Krum's rejection proxy is a \\emph{repurposed} full-run persistence rule applied "
            "to a native robust-aggregation mechanism, not a native attribution method. Gaussian: "
            "100\\% attacker rejection at both seeds. Directional: 0\\% attacker rejection at both "
            "seeds. The proxy misattributes the persistent clean-phase honest client in all four "
            "attacked runs; FedTrace v3 attributes correctly in all four paired FedAvg runs.")
    emit("table04_multikrum_baseline", True,
         "Multi-Krum external baseline: attack-geometry dichotomy and repurposed-proxy attribution "
         "failure, contrasted with FedTrace v3 on the paired FedAvg run.", "tab:multikrum",
         "llllccccccccc", header, rows, note=note)


# ---------------------------------------------------------------------------
# Table 5 -- targeted generalization (replaces Figure 8)
# ---------------------------------------------------------------------------


def table05_generalization() -> None:
    header = ["Experiment", "Changed factor", "True attacker", "Predicted attacker",
              "True onset", "Estimated onset", "Confirmation delay", "Honest persistent FP",
              "Dominant signal", "Result"]
    run_specs = [
        ("EXP_018", "baseline validation condition"),
        ("EXP_026", r"attacker identity: client\_002 $\to$ client\_004"),
        ("EXP_027", "attack onset: round 20 $\\to$ round 30"),
    ]
    rows = []
    for exp_id, changed in run_specs:
        rows_data = load_trace_rows(exp_id)
        result = ft.evaluate_detector_v3(rows_data, "H", ("m", "t_res"))
        predicted = result.predicted_client
        onset_est = result.per_client[predicted].onset_estimate if predicted else NOT_RECORDED
        all_pass = (predicted == result.true_malicious_client and result.onset_localization_error == 0
                    and result.detection_delay == 2 and len(result.false_positive_clients) == 0)
        if not all_pass:
            raise TableDataError(f"{exp_id}: generalization row does not match the expected PASS pattern")
        rows.append([exp_id, changed, result.true_malicious_client, predicted,
                     result.attack_start_round, onset_est, result.detection_delay,
                     len(result.false_positive_clients),
                     dominant_signal_for_client(exp_id, result.true_malicious_client), "PASS"])
        record_provenance("table05_generalization", exp_id, exp_id,
                           "results/federated/.../client_round_trace.csv",
                           "update_l2_norm, self_prev_round_cosine",
                           "src/analysis/fedtrace_detector.py::evaluate_detector_v3 (frozen, unmodified)",
                           "Migrated from Figure 8 (Phase P4) per that figure's own recommendation "
                           "that this content is better represented as a table.")
    note = ("Identity generalization (EXP\\_026) and onset generalization (EXP\\_027) each change "
            "exactly one factor relative to the EXP\\_018 anchor and both pass without any "
            "retuning. One targeted test per axis is evidence of generalization within that one "
            "comparison, \\emph{not} a claim of arbitrary-client or arbitrary-onset invariance.")
    emit("table05_generalization", True,
         "Targeted attacker-identity and attack-onset generalization checks (frozen FedTrace v3, "
         "no retuning).", "tab:generalization", "llccccccll", header, rows, note=note)


# ---------------------------------------------------------------------------
# Appendix Table A1 -- full experiment registry
# ---------------------------------------------------------------------------


def table_a1_registry() -> None:
    if not REGISTRY_CSV.exists():
        raise TableDataError(f"Required registry not found: {REGISTRY_CSV}")
    with REGISTRY_CSV.open(newline="", encoding="utf-8") as handle:
        reg_rows = list(csv.DictReader(handle))
    header = ["ID", "Category", "Aggregation", "Seed", "Attack", "Attacker", "Onset",
              "Purpose", "Final Top-1 (%)", "Attribution result", "Scientific status"]
    rows = []
    for r in reg_rows:
        top1 = r["Final_Test_Top1"]
        try:
            top1_fmt = f"{float(top1) * 100:.2f}"
        except ValueError:
            top1_fmt = NOT_APPLICABLE if "APPLICABLE" in top1 else NOT_RECORDED
        attribution = r["Attribution_Result"] if r["Attribution_Result"] not in ("", None) else NOT_RECORDED
        rows.append([
            r["Experiment_ID"], r["Status"], r["Aggregation"], r["Seed"],
            r["Attack_Family"] if r["Attack_Present"] == "Yes" else NOT_APPLICABLE,
            r["Malicious_Client"] if r["Attack_Present"] == "Yes" else NOT_APPLICABLE,
            r["Attack_Onset_Round"] if r["Attack_Present"] == "Yes" else NOT_APPLICABLE,
            r["Purpose"][:60] + ("..." if len(r["Purpose"]) > 60 else ""),
            top1_fmt, attribution, r["Status"],
        ])
        record_provenance("tableA1_experiment_registry", r["Experiment_ID"], r["Experiment_ID"],
                           "analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv",
                           "all columns", "direct reformat/subset of the already-verified registry "
                           "(itself built from results/**/summary.json and analysis documents in "
                           "the FEDTRACE_MASTER_RESEARCH_STATUS.md task)")
    note = ("EXP\\_013 is synthesis-only (no independent training run); every quantitative field "
            "is NOT APPLICABLE, not fabricated.")
    emit("tableA1_experiment_registry", False,
         "Full experiment registry, EXP\\_001--EXP\\_027.", "tab:registry",
         "llllllllccc", header, rows, note=note)


# ---------------------------------------------------------------------------
# Appendix Table A2 -- method evolution (development history, qualitative)
# ---------------------------------------------------------------------------


def table_a2_method_evolution() -> None:
    header = ["Version", "Temporal calibration", "Observed failure", "Fix",
              "Development experiment", "Validation status"]
    rows = [
        ["v1", "Static, per-client, pre-attack-only mean/stdev of $t_{k,r}$ (rounds $<T_{\\mathrm{ref}}$)",
         "Non-stationarity: $t$ declines secularly for every client; 3 honest clients confirmed "
         "persistent late (rounds 43/47/48) on EXP\\_015",
         "Same-round cross-sectional reference (v2)", "EXP\\_014, EXP\\_015 (seed 42)",
         "Superseded -- not used in any validation run"],
        ["v2", "Same-round cross-sectional median/MAD ($c_r$, $\\mathrm{scale}_t(r)=1.4826\\cdot\\mathrm{MAD}$)",
         "Small-sample MAD instability at $n_r=5$: honest client\\_000 confirmed persistent at "
         "round 11, before the true attack begins at round 20 -- attribution-breaking",
         "Longitudinal (per-client, historical) residual-scale calibration (v3)",
         "EXP\\_014, EXP\\_015 (seed 42)", "Superseded -- not used in any validation run"],
        ["v3 (frozen)", "Same-round median reference $c_r$ (kept from v2) + per-client longitudinal "
         "residual calibration $\\mu^d_k, s^d_k$ over $D_k=\\{d_{k,r}: r\\in[1,T_{\\mathrm{ref}})\\}$",
         "None found -- zero honest persistent false positives across an exhaustive 49-round "
         "$\\times$ 5-client check on EXP\\_015", "N/A (final method)",
         "EXP\\_014, EXP\\_015 (seed 42)", "Frozen; independently validated on EXP\\_016--019, "
         "EXP\\_026, EXP\\_027"],
    ]
    for row in rows:
        record_provenance("tableA2_method_evolution", row[0], "EXP_014, EXP_015",
                           "analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md; "
                           "analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md; "
                           "analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md",
                           "qualitative summary fields",
                           "Manually curated from the cited, already-verified retrospective analysis "
                           "documents (re-read during Phase P3); not derived from a single numeric "
                           "artifact field, since this table is a development-history narrative, "
                           "not a results table.")
    note = ("Development history only -- v1 and v2 are shown to explain why v3's calibration design "
            "was necessary, not as competing final methods. v3 is the sole version used in any "
            "independent validation or generalization experiment.")
    emit("tableA2_method_evolution", False,
         "FedTrace temporal-calibration design evolution (development history).",
         "tab:evolution", "lp{3cm}p{3.6cm}p{2.6cm}llp{2.2cm}", header, rows, note=note)


# ---------------------------------------------------------------------------
# Appendix Table A3 -- utility metrics
# ---------------------------------------------------------------------------


def table_a3_utility_metrics() -> None:
    header = ["Experiment", "Loss", "Top-1 (%)", "Top-5 (%)", "Macro Precision",
              "Macro Recall", "Macro F1", "Balanced Accuracy", "Best Val Top-1 (%)"]
    exp_ids = ["EXP_001", "EXP_008", "EXP_009", "EXP_016", "EXP_017", "EXP_018", "EXP_019",
               "EXP_020", "EXP_021", "EXP_022", "EXP_023", "EXP_024", "EXP_025",
               "EXP_026", "EXP_027"]
    rows = []
    for exp_id in exp_ids:
        s = load_summary(exp_id)
        ft_ = s["final_test"]
        bv = s.get("best_validation", {})
        rows.append([
            exp_id, f"{ft_['loss']:.4f}", pct(ft_["top1"]), pct(ft_["top5"]),
            f"{ft_['macro_precision']:.4f}", f"{ft_['macro_recall']:.4f}",
            f"{ft_['macro_f1']:.4f}", f"{ft_['balanced_accuracy']:.4f}",
            pct(bv["top1"]) if "top1" in bv else NOT_RECORDED,
        ])
        record_provenance("tableA3_utility_metrics", exp_id, exp_id,
                           "results/.../summary.json", "final_test.*, best_validation.top1",
                           "direct read, percentages multiplied by 100")
    note = ("Restricted to the centralized reference and the main FedAvg/Multi-Krum/generalization "
            "groups -- not every historical experiment (EXP\\_002--007, EXP\\_010--015 are in "
            "Table~A1 only).")
    emit("tableA3_utility_metrics", False,
         "Detailed utility metrics for the main experimental groups.", "tab:utility",
         "lcccccccc", header, rows, note=note)


def main() -> None:
    table01_experimental_setup()
    table02_fedtrace_validation()
    table03_signal_ablation()
    table04_multikrum_baseline()
    table05_generalization()
    table_a1_registry()
    table_a2_method_evolution()
    table_a3_utility_metrics()

    provenance_path = TABLES_ROOT / "TABLE_DATA_PROVENANCE.csv"
    header = ["table_id", "row_id", "experiment_id", "source_file", "source_field",
              "transformation", "notes"]
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(PROVENANCE)

    print(f"Main tables: {len(list(MAIN_DIR.glob('*.csv')))} in {MAIN_DIR}")
    print(f"Appendix tables: {len(list(APPENDIX_DIR.glob('*.csv')))} in {APPENDIX_DIR}")
    print(f"LaTeX snippets: {len(list(LATEX_DIR.glob('*.tex')))} in {LATEX_DIR}")
    print(f"Provenance rows: {len(PROVENANCE)} -> {provenance_path}")


if __name__ == "__main__":
    main()
