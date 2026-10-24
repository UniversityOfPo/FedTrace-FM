# 07 — Decision Log

Append-only. Each entry: ID, date, decision, owner, rationale, evidence. Dates are 2026.

## Decisions taken

| ID | Date | Decision | Owner | Rationale / evidence |
|---|---|---|---|---|
| D-001 | 10-20 | NDSS archive frozen and backed up (`<preservation root>/NDSS_freeze_20261020`, 1,017 files, seal `e20c9055…`) | PI | Preserve submitted evidence bit-for-bit |
| D-002 | after NDSS rejection | FedTrace-FM (DINOv2-S + LoRA) paused; FM-00b not started | PI | Focus on reviewer response on ResNet-18/CIFAR-100 |
| D-003 | Phase R2 | Withdrawn: "ρ = 0.5 Gaussian ≈ harmless" (EXP_008 not a matched control), "[10] likely FLForensics", "40–50 rounds mandatory" | PI/Claude | Phase R2 report |
| D-004 | Prompt 13 | All new comparisons within one frozen code revision with fresh matched clean controls; dev seeds {123, 2026}; ≥ 5 fresh validation seeds | PI | Matched-control requirement |
| D-005 | Prompt 13 | Attack taxonomy A (stress) / B (poisoning) / C (detector-aware); fixed-direction attacks are stress tests only | PI | Reviewer #368A |
| D-006 | Prompt 13 | Online semantics: no scoring before T_ref; alarm / actionable / onset / censored delay / retrospective onset separated | PI | Look-ahead in offline v3 |
| D-007 | 10-24 | 41 untracked FM files backed up (`<preservation root>/FM_untracked_20261024`, manifest `5eb86478…`, seal `d77cdcd5…`) | Claude (PI-approved) | Prompt 14 §0 |
| D-008 | 10-24 | `.gitignore` +3 rules: `results_rev/`, `checkpoints_rev/`, `logs_rev/` | PI | Prompt 14 §1 |
| D-009 | 10-24 | Stage A implemented as `src/rev/` without editing protected code (runner hooks + temporary name rebinding in the NDSS script module) | Claude | Prompt 14 §2; audited Prompt 15 |
| D-010 | 10-24 | Stage A: Claude PASS; PI's independent reviewer CONDITIONAL PASS | PI reviewer | Prompt 15 |
| D-011 | 10-24 | Preservation test no longer hard-codes a drive path: `$FEDTRACE_PRESERVATION_ROOT`, default `FedTrace_Preservation` at the repository drive root; manifests checked against their seals; `FEDTRACE_REQUIRE_PRESERVATION=1` turns skips into failures; `.gitignore` check pinned to NDSS base commit `9307b96` | Claude (PI-authorised, Prompt 16) | Prompt 16 §1 |
| D-012 | 10-24 | Surviving mutant R2 (replay peer order) accepted as equivalent for the 3-client toy (P-02 resolved). The order-dependent field feeds the NDSS naive peer baseline and v1 z_p, not v3; archived decision margins ≥ 1.9e-3 relative (L-16) | PI reviewer authorised; Claude analysis | Prompt 16 §1 |
| D-013 | 10-24 | Clean Multi-Krum false alarm (L-11), missing clean FedAvg evidence (L-18), Multi-Krum provenance gap (L-17) recorded as CLAUDE-REPORTED; reproduction ≠ validation (blueprint §10a); direct forensic baselines required (blueprint §10b, C4 MUST-attempt) | PI | Prompt 16 §4 |

## Pending decisions (owner: PI unless stated)

| ID | Decision needed | Needed before | Recommendation |
|---|---|---|---|
| P-01 | Approve the source checkpoint (branch, file list, message without AI co-author trailers per the PI's earlier instruction) | S0 / B-R1 | Approve (see Prompt 15 report, Part D) |
| P-02 | Accept surviving mutant R2 as equivalent for the 3-client toy | Stage B | Accept; add a 4-peer permutation test later (non-blocking) |
| P-03 | Launch R0/R1 with or without a fixed `PYTHONHASHSEED` | B-R0 | Without (mirrors archived launches); archive and permutation evidence show the only order-dependent field was invariant (L-16) |
| P-04 | Validation seeds and derivation procedure | validation batch | SHA-256-derived, sealed in the pre-registration |
| P-05 | Provisional thresholds: backdoor success ≥ 0.5, κ = 2.5, H1 delay ≤ 5 | validation batch | Decide after dev runs, before sealing |
| P-06 | Level B tolerance (final top-1 0.3 pp proposed) | Stage B gate (only if Level A fails) | Decide only if needed |
| P-07 | Provide the submitted PDF (identify [10]) and sources for FLForensics / FedREDefense / FLDetector | C4, related work | As early as possible (Day 1) |
| P-08 | Logged-run subset and external encrypted backup medium | > 400 GiB new data | Subset per `06` disk plan |
| P-09 | X6 (two attackers) go/no-go | Day 4 | Only if engineering ≤ 0.5 day |
| P-10 | Disclose archived dirty-tree runs (E1–E3; EXP_020–025) in the paper | manuscript | Yes |
| P-11 | How to treat L-11 (clean Multi-Krum honest alarm) | manuscript | Report as a specificity finding; investigate drift in C3 |
