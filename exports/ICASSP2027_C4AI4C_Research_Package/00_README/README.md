# ICASSP 2027 C4AI4C — Research Evidence Package (FedTrace)

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
