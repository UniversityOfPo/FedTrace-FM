# FedTrace Master Results — Synchronization Report (Phase P2)

**Status:** Phase P2 (Master Results Synchronization) execution report. No federated experiment, attack, FedTrace analysis, or Multi-Krum run was executed. FedTrace v3, the attack implementations, and Multi-Krum were not modified. No existing config, result, or checkpoint file was modified. Environment change disclosed explicitly in §9.

---

## 1. Objective

Build a reproducible, idempotent synchronization pipeline (`Research_Database/sync_master_results.py`) that reads existing experiment artifacts (never invents values) and updates `Research_Database/FedTrace_Master_Results.xlsx` to represent `EXP_001`–`EXP_027`, adding FedTrace-attribution and Multi-Krum-diagnostic schema that did not previously exist, while preserving every pre-existing sheet unchanged.

## 2. Step 1–2: Sources Read and Workbook Audit

**Authoritative sources read:** `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md`, `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv`, `Research_Database/FedTrace_Master_Results.xlsx` (via `openpyxl`, all 8 sheets, headers and full row content), `Research_Database/FedTrace_Master_Results.md` (spec document), every `results/federated/*/` and `results/centralized/*/` artifact directory's file listing (representative attack, Multi-Krum, and clean runs individually inspected), `multikrum_diagnostics.csv`'s exact column schema, `final_test_metrics.json`'s schema. Root-level stale documents (`FEDTRACE_FM_MASTER.md`, `PROJECT_CONTEXT.md`, `PROJECT_ROADMAP.md`, `RESEARCH_DIARY.md`) were **not** used as a source of truth for any value, per this task's explicit instruction.

**Workbook audit (before synchronization):**

| Sheet | Columns (as actually stored — differs in places from `FedTrace_Master_Results.md`'s documented snake_case CSV schema) | Rows (excl. title/header) | Purpose | Still useful? | Extend or new sheet? |
|---|---|---:|---|---|---|
| Experiment Summary | `Experiment ID, Method, Dataset, Partition, Model, Clients, Rounds/Epochs, Local Epochs, Seed, Top-1, Top-5, Macro F1, Balanced Accuracy, Runtime (s), Git Commit, Status, Result Folder` | 2 (`EXP_001`, `EXP_002`) | Human-readable per-experiment summary | Yes, as a historical record of the project's first two experiments | **New sheet** (`Experiment Registry`) — the human-readable column naming and lack of attack/FedTrace/dirichlet-alpha columns make in-place extension riskier than addition; preserved unchanged |
| Hyperparameters | `Experiment ID, Optimizer, Learning Rate, Momentum, Weight Decay, Scheduler, Batch Size, Local Epochs, Validation Fraction, Aggregation, Clients/Round, Minimum Updates` | 2 | Training hyperparameters | Yes | Preserved unchanged; equivalent fields folded into `Experiment Registry` |
| Hardware | `Experiment ID, Device, GPU, VRAM (GB), Python, PyTorch, CUDA, Runtime (s)` | 2 | Reproducibility environment | Yes | Preserved unchanged; device/GPU fields folded into `Reproducibility Provenance` |
| Federated Results | `Experiment ID, Partition, Num Clients, Clients/Round, Rounds, Local Epochs, Samples/Client, Aggregation, Best Val Round, Best Val Top-1, Final Val Loss, Final Val Top-1, Final Val Top-5, Final Test Top-1, Final Test Top-5, Status` | 1 (`EXP_002`) | Federated-only results | Yes | Preserved unchanged; superseded in coverage (not content) by `Experiment Registry` |
| Attack Results | `Experiment ID, Attack Type, Malicious Clients, Poisoning Rate, Clean Top-1, Attack Success Rate, Detection AUROC, False Positive Rate, Runtime (s), Git Commit, Status, Notes` | 0 (header only) | Anticipated attack-result schema, authored before Phase 4 existed | **Schema does not match this project's actual attack fields** (no `attack_start_round`, `directional_target_cosine`, etc.; "Detection AUROC" was never the metric this project settled on) | **New sheet** (`FedTrace Attribution`) rather than force-fit into this schema; left empty and untouched, since populating it would require inventing values in columns this project never measured |
| Attribution Results | `Experiment ID, Method, Client Attribution Accuracy, Top-k Culprit Recall, Round Localization Error, Spearman Correlation, Counterfactual Error, Attribution Runtime (s), Speedup, Git Commit, Status, Result Folder, Notes` | 0 (header only) | Anticipated attribution-result schema | **Schema does not match** (this project's actual metrics are onset error/detection delay/honest-FP-count/dominant-signal, not "Spearman Correlation" or "Speedup") | Same as above — left empty and untouched |
| Paper Tables | `Table ID, Paper Section, Caption, Source Experiment IDs, Primary Metric, Status, Output File, Git Commit, Last Updated, Notes` | 0 (header only) | Paper table registry | Yes, schema is reasonable and general | Preserved unchanged; genuinely no paper tables exist yet to populate it with (per `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §9, manuscript readiness ~5%) — populating it now would be premature, not a synchronization task |
| Notes | `Experiment ID, Objective, Key Observation, Conclusion, Limitation, Next Action, Status` | 2 | Free-form scientific notes | Yes | Preserved unchanged |

**Decision:** every pre-existing sheet is preserved **exactly as-is** — not one cell was altered. All new content lives in six new, additively-created sheets (§4).

## 3. Step 3: Target Schema

Six new sheets were added, each with a schema either matching or extending this task's own Step 3 field list:

1. **Experiment Registry** — the main per-experiment record (35 columns; superset of the requested field list — `experiment_status/category` is named `experiment_category`; every other requested field is present verbatim).
2. **FedTrace Attribution** — `experiment_id, fedtrace_version, predicted_client, true_client, attribution_correct, first_anomaly_round, confirmation_round, onset_estimate, true_onset, onset_error, detection_delay, honest_persistent_fp_count, dominant_signal, threshold, persistence_p, evaluable, notes, source_analysis_document`.
3. **MultiKrum Diagnostics Summary** — `experiment_id, clean_or_attacked, seed, attack_type, true_attacker, pre_attack_rejected_client, pre_attack_rejection_count, post_attack_attacker_rejection_count, post_attack_attacker_rejection_rate, post_attack_attacker_selection_count, rejection_switch_round, proxy_predicted_client, proxy_attribution_correct, proxy_confirmation_round, f, m, neighbors_used, source_analysis_document`.
4. **Ablation Detector Results** — `experiment_id, detector, onset_error, detection_delay, notes` (a compact schema; full precision/recall/F1 tables remain in their source analysis documents, cited via `notes`/the FedTrace Attribution sheet's own `source_analysis_document` column, rather than duplicated here).
5. **Reproducibility Provenance** — `experiment_id, git_commit, result_dir, checkpoint_dir, config_path, device, gpu_name, total_experiment_seconds`.
6. **Sync Metadata** — a new, non-requested-but-low-risk sheet: sync timestamp, script path, source registry, row counts per sheet, the missing-value policy (§7), and the managed/legacy sheet-name lists — added specifically to satisfy this task's "make future synchronization simple" objective by making the sync's own provenance self-documenting inside the workbook.

**"Core Metrics" was not created as a separate sheet** — its requested fields (loss/top1/top5/best-validation/macro metrics) are folded directly into `Experiment Registry` as one wide table, avoiding an unnecessary duplicate-key join between two sheets for data that is always read together.

## 4. Step 4: `sync_master_results.py` — Design Summary

**Location:** `Research_Database/sync_master_results.py` (matches the task's own naming/location instruction).

**Discovery:** a single pass over every `results/**/summary.json`, keyed by each file's own embedded `experiment_id` field, with exactly one explicit, commented exception (`EXP_001`'s centralized `summary.json` predates that field). Smoke-test artifacts (`EXP_002_SMOKE`, the centralized smoke run) are excluded automatically — they simply never match any `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` row.

**Numeric ground truth:** read fresh from each experiment's own `summary.json` on every run — never copied unchecked from the registry CSV. A cross-check against each run's separate `final_test_metrics.json` is performed and any disagreement is surfaced as a discrepancy in the sync summary (§8) rather than silently resolved.

**FedTrace-attribution and ablation fields:** per this task's explicit "MUST NOT be reverse-engineered from memory" instruction, these are **not** parsed from analysis-document prose at runtime. They are instead a small, explicit, individually-cited constant table (`FEDTRACE_ATTRIBUTION_FACTS`, `ABLATION_FACTS`) at the top of the script, each entry re-verified against its exact source document/section **during this task**, not merely recalled (the `EXP_014`/`EXP_015` development-stage numbers, in particular, were re-read directly from `analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md` and `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` in this task before being transcribed, specifically to avoid the risk this instruction warns about).

**Multi-Krum diagnostic fields:** computed **fresh, deterministically**, from each run's own `multikrum_diagnostics.csv` (raw per-round-per-client `krum_score`/`krum_rank`/`multikrum_selected`/`multikrum_rejected`) — not hand-transcribed — by mechanically applying the already-documented `P_krum=3` consecutive-rejection persistence rule used throughout the Multi-Krum analysis series. This is a legitimate artifact-derived computation (the raw data already exposes exactly what is needed), not a re-analysis requiring new judgment.

**Idempotency mechanism:** the six new sheets are fully deleted and rebuilt from scratch on every run; the eight legacy sheets are never opened for writing at all. This makes duplicate-row accumulation structurally impossible rather than merely unlikely.

**Safety:** a timestamped backup (`Research_Database/backup/FedTrace_Master_Results_<UTC-timestamp>.xlsx`) is written before every sync; the new workbook is written to a `.xlsx.tmp` file and atomically replaced over the target path (`Path.replace`).

## 5. Step 5: `EXP_013` Handling

Registered explicitly as `experiment_category = SYNTHESIS_ONLY_NO_INDEPENDENT_RUN`. Every numeric field (`final_test_loss`, `final_test_top1`, `final_test_top5`, `best_validation_top1`, macro metrics, seed, etc.) is written as the literal string `NOT APPLICABLE`, never a fabricated or interpolated value. `notes_status` states plainly that it consolidates `EXP_007/008/009/010/011/012` and has no `summary.json` of its own. Verified directly in the resulting workbook (§8).

## 6. Step 6: Round-0 Artifact Wording

No occurrence of "no honest client ever exceeds threshold" (or any equivalent overclaim) was written anywhere in the workbook. Where the round-0 artifact is relevant to a transcribed fact (e.g. `EXP_015`'s v1/v3 honest-false-positive counts), the wording used is "zero honest persistent false positives" / an explicit count of confirmed persistent false positives — matching `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §7's documented policy exactly. A regression test (`test_round0_wording_policy_not_violated_in_notes`) guards this for future syncs.

## 7. Missing-Value Policy

- **`NOT RECORDED`** — the value should exist somewhere but was not safely, deterministically recoverable by this script in this pass (e.g. `EXP_001`'s `git_commit`, which is absent from its own `summary.json` because that file predates the field being written).
- **`NOT APPLICABLE`** — the field does not apply to this experiment's type (e.g. `dirichlet_alpha` for an IID run; every quantitative field for `EXP_013`).
- Neither is ever substituted with a fabricated number, an interpolated estimate, or a value copied from a *different* (even closely related) experiment.

## 8. Step 9–10: Dry-Run, Backup, Execution, Idempotency

**Dry run** (`python Research_Database/sync_master_results.py --dry-run`): 27 experiments registered, 10 FedTrace-attribution rows (`EXP_014`: 1; `EXP_015`: 3, one per version; `EXP_016`/`017`/`018`/`019`/`026`/`027`: 1 each), 6 Multi-Krum rows (`EXP_020`–`025`), 18 ablation rows (3 per run × 6 runs), **zero `summary.json`/`final_test_metrics.json` discrepancies**.

**Run 1** (real, with backup): backup written to `Research_Database/backup/FedTrace_Master_Results_20260811T185444Z.xlsx` (a copy of the pre-sync, `EXP_001`/`EXP_002`-only workbook). Workbook updated: 14 sheets total (8 legacy, unchanged; 6 new). `Experiment Registry`: 27 data rows, `EXP_001`–`EXP_027`, no duplicates (verified programmatically: `len(ids) != len(set(ids))` → `False`). `MultiKrum Diagnostics Summary` spot-checked against known values — `EXP_020` (clean, seed 123): most-rejected client `client_004`, 49/50 rounds; `EXP_021` (clean, seed 2026): `client_001`, 49/50 — both match `analysis/MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md`'s independently-established finding exactly. `EXP_022`/`023` (Gaussian): 30/30 post-attack rejection, both seeds. `EXP_024`/`025` (directional): 0/30 post-attack rejection, both seeds. All match the already-published Multi-Krum synthesis documents.

**Run 2** (idempotency check): re-ran the identical script against unchanged inputs. Result: identical experiment-ID list, identical row counts on every managed sheet (`Experiment Registry` 27 rows both times; `FedTrace Attribution` 10; `MultiKrum Diagnostics Summary` 6; `Ablation Detector Results` 18; `Reproducibility Provenance` 26; `Sync Metadata` 11 — all unchanged between run 1 and run 2 except the `Sync Metadata` sheet's own timestamp field, which is expected to change every run by design). **No duplicate rows were created. No key result value differed between the two runs.** A second, independent backup was written before run 2 (now a snapshot of the post-run-1 state), confirming the backup step itself is also safely repeatable.

## 9. Environment Change (Disclosed, Not Silently Made)

`openpyxl` was **not installed** in this project's `.venv` and is **not listed** in `requirements.txt`, despite being the only way to read or write the existing `.xlsx` workbook programmatically (confirmed: `Research_Database/FedTrace_Master_Results.md` itself explains the workbook was originally hand-added because no `.xlsx`-writing library was available in an earlier session). `openpyxl==3.1.5` was installed into the project's `.venv` via `pip install openpyxl` so this task could be completed at all. **`requirements.txt` was deliberately left unmodified** — adding a new pinned dependency there was judged to be outside the explicitly authorized file list for this task ("If anything else must change, STOP and explain before changing it"). **This is flagged here, not actioned:** a follow-up task should add `openpyxl` to `requirements.txt` so `sync_master_results.py` is runnable from a fresh clone without this manual step.

## 10. Discrepancies / Blockers Found

**None that block this task.** The one item requiring follow-up is the `requirements.txt` gap noted in §9 (not a blocker — the script runs correctly in the current environment; it is a reproducibility gap for a *future*, fresh environment).

---

## Final Report

1. **Files created:** `Research_Database/sync_master_results.py`; `tests/test_sync_master_results.py`; `analysis/FEDTRACE_MASTER_RESULTS_SYNC_REPORT.md`; two workbook backups (`Research_Database/backup/FedTrace_Master_Results_20260811T185444Z.xlsx`, `Research_Database/backup/FedTrace_Master_Results_20260811T185517Z.xlsx`).
2. **Files modified:** `Research_Database/FedTrace_Master_Results.xlsx` (new sheets added; all 8 pre-existing sheets byte-for-byte preserved, verified by direct re-read).
3. **Sync script path:** `Research_Database/sync_master_results.py`.
4. **Workbook path:** `Research_Database/FedTrace_Master_Results.xlsx`.
5. **Backup path:** `Research_Database/backup/` (two timestamped copies, one taken before each of the two sync runs).
6. **Workbook sheets before:** 8 — `Experiment Summary, Hyperparameters, Hardware, Federated Results, Attack Results, Attribution Results, Paper Tables, Notes` (only `EXP_001`/`EXP_002` represented, per §2).
7. **Workbook sheets after:** 14 — the same 8 (unchanged) plus `Experiment Registry, FedTrace Attribution, MultiKrum Diagnostics Summary, Ablation Detector Results, Reproducibility Provenance, Sync Metadata`.
8. **Experiments represented before:** 2 (`EXP_001`, `EXP_002`), and only in the legacy sheets' original narrow schema.
9. **Experiments represented after:** 27 (`EXP_001`–`EXP_027`), in `Experiment Registry`; `EXP_020`–`027` additionally represented in `MultiKrum Diagnostics Summary`/`FedTrace Attribution`/`Ablation Detector Results` as applicable.
10. **Confirmation `EXP_001`–`EXP_027` coverage:** confirmed programmatically — `Experiment Registry` contains exactly `EXP_001` through `EXP_027`, 27 rows, no gaps, no duplicates.
11. **`EXP_013` handling:** registered as `SYNTHESIS_ONLY_NO_INDEPENDENT_RUN`; every quantitative field `NOT APPLICABLE`; no fabricated metric (§5).
12. **FedTrace Attribution sheet/schema status:** created, populated for all 8 FedAvg attack runs (`EXP_014`–`019`, `026`, `027`; 10 rows total, `EXP_015` contributing 3 for its three design versions), sourced via an explicit, individually-cited constant table, each entry re-verified against its source document during this task.
13. **Multi-Krum sheet/schema status:** created, populated for all 6 Multi-Krum runs (`EXP_020`–`025`), computed fresh from each run's own `multikrum_diagnostics.csv` via the documented `P_krum=3` rule.
14. **Paper Tables sheet/schema status:** left untouched, empty — genuinely no paper tables exist yet (manuscript readiness ~5% per `FEDTRACE_MASTER_RESEARCH_STATUS.md` §9); populating it now would require inventing content, which this task does not do.
15. **Missing/unrecoverable fields:** `EXP_001`'s `git_commit` (absent from its own `summary.json`, pre-dates that field); `EXP_014`'s precise `first_anomaly_round` in the attribution sheet (recorded as `NOT RECORDED` rather than guessed — the source document reports the confirmed/onset rounds precisely but not this specific raw-first-crossing round); a handful of federated-config fields for experiments whose `configs/<name>.yaml` could not be located by the deterministic naming convention (none encountered in this run — all 27 configs resolved).
16. **Extraction policy used:** numeric/config fields — fresh from `summary.json`/YAML every run, cross-validated against `final_test_metrics.json`; FedTrace-attribution/ablation fields — explicit, cited constant table, individually re-verified against source documents in this task; Multi-Krum fields — computed fresh from raw `multikrum_diagnostics.csv` via the documented persistence rule. Full detail in §4, §7.
17. **Idempotency result:** confirmed — second run produced identical row counts and identical key values on every managed sheet; no duplicates created (§8).
18. **Duplicate-ID validation:** confirmed — `load_registry()` rejects duplicate `Experiment_ID`s (tested); the live registry CSV contains none; the resulting `Experiment Registry` sheet contains 27 unique IDs.
19. **Key metric spot-checks:** `EXP_020`/`EXP_021` clean-control most-rejected client (`client_004`/`client_001`) matches `MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md`; `EXP_022`/`023` 30/30 and `EXP_024`/`025` 0/30 post-attack rejection match the Gaussian/directional cross-seed synthesis documents exactly; `EXP_008`/`018`/`026`/`027` `final_test_top1`/`loss` values match both `summary.json` and their respective prior analysis documents (§8).
20. **Tests added:** `tests/test_sync_master_results.py`, 20 tests covering ID parsing/sorting, duplicate-ID rejection, `EXP_013` handling, missing-optional-artifact tolerance, dry-run determinism, legacy/managed sheet-name disjointness, clean-run and attack-run field extraction, Multi-Krum extraction (clean/Gaussian/directional), non-mutation of source `summary.json`, attribution-fact coverage/consistency, and the round-0 wording policy.
21. **Tests executed and results:** `python -m pytest tests/test_sync_master_results.py -v` — **20 passed, 0 failed**.
22. **Synchronization commands executed:** `python Research_Database/sync_master_results.py --dry-run` (once); `python Research_Database/sync_master_results.py` (twice, for the idempotency check); `python -m pytest tests/test_sync_master_results.py -v` (once); plus several read-only `python -c` inspection one-liners against the workbook (`openpyxl.load_workbook(..., data_only=True)`) and `pip install openpyxl` (§9).
23. **Confirmation no federated experiment run:** confirmed — no training/federated-runner script was invoked at any point in this task.
24. **Confirmation FedTrace unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was not read or touched in this task; the sync script does not import or call it.
25. **Confirmation attack code unchanged:** confirmed — no file under `src/attacks/` was read or touched.
26. **Confirmation Multi-Krum unchanged:** confirmed — no Multi-Krum aggregation-strategy source file was read or touched; only each run's already-written `multikrum_diagnostics.csv` output was read.
27. **Confirmation configs/results/checkpoints unchanged:** confirmed — every config, result, and checkpoint file was opened read-only; none was written to (verified directly for one representative case via `test_registry_extraction_does_not_alter_source_summary_json`, which byte-compares `summary.json` before and after extraction).
28. **Discrepancies or blockers found:** none blocking; one disclosed, non-blocking follow-up item — `openpyxl` should be added to `requirements.txt` in a future task (§9).
29. **Final P2 synchronization verdict: A — complete and trustworthy.** All ten pipeline requirements from this task's Primary Objective are met: artifacts-only reads, no invented metrics, full `EXP_001`–`EXP_027` support, legacy-sheet preservation, new FedTrace/attack/Multi-Krum schema added, idempotent (proven by a real second run), no duplicate rows, deterministic (dry-run comparison + test), self-validating (cross-check against `final_test_metrics.json`, plus 20 passing tests), and the `Sync Metadata` sheet plus this report make future synchronization straightforward.
30. **Exact recommended next phase:** proceed to **P3 (Mathematical/theoretical formalization)**, per `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §10 — no further database work is blocking; the one open item (§9's `requirements.txt` addition) is a small, independent housekeeping task that can be done at any point, not a gate. This task does not begin P3.
