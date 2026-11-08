# Missing or incomplete items (recorded, not silently omitted)

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
