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
| D-014 | 10-24 | Local checkpoint commit `d36094a0b3c4f70f63710701f00746936813c71e` on branch `rev/stage-a` (32 files: `.gitignore`, `src/rev/` 12, `configs/rev/` 2, 10 test files, `paper_plan/` 7); not pushed; FM files untracked (P-01 resolved) | PI-authorised; Claude executed | Prompt 16 §2 |
| D-015 | 10-24 | Stage B prepared (uncommitted): `configs/rev/S0a_smoke_ndss.yaml`, `configs/rev/S0b_smoke_logged.rev.yaml`, `src/rev/compare_runs.py`, `src/rev/preflight.py`, `tests/test_rev_compare_preflight.py`; comparison statuses IDENTICAL / NUMERICAL / VALUE / STRUCTURAL / MISSING / EXCLUDED / INFO | Claude | Prompt 16 §3 |

| D-016 | 10-24 | Second local commit `857882a686ac5f6e8f56de60d4d4d945369fe6d8` (Stage B preparation; 8 files); not pushed (P-12 resolved) | PI-approved; Claude executed | Prompt 17 §1 |
| D-017 | 10-24 | Stage B executed under the approved gates (P-13 resolved): S0a, S0b → gate PASS; R0 → Level A PASS vs EXP_016; R1 → bit-identical to R0. PYTHONHASHSEED not fixed (P-03 resolved as recommended). No Level B tolerance was defined or used (P-06 moot). Stage C not launched | PI-approved gates; Claude executed | Prompt 17; ledger L-19–L-21 |
| D-018 | 10-24 | Target venue changed to CVPR 2027 Main Track (blueprint and schedule still reflect the NDSS structure; to be revised) | PI | Prompt 17 |

## Pending decisions (owner: PI unless stated)

| ID | Decision needed | Needed before | Recommendation |
|---|---|---|---|
| P-01 | ~~Approve the source checkpoint~~ — RESOLVED by D-014 | — | — |
| P-14 | Commit the Stage B evidence updates to `paper_plan/` (uncommitted after D-017) | Stage C | Approve with the next checkpoint |
| P-15 | Revise blueprint, section structure and schedule for CVPR 2027 Main Track (page limit, reviewer expectations, venue fit of a security/forensics paper) | Stage C planning | Required |
| P-12 | ~~Approve a second local commit~~ — RESOLVED by D-016. Original text: Approve a second local commit of the Stage B files (D-015) and the paper_plan updates, so S0/R0/R1 run from committed code (preflight `git` check is FAIL until then) | S0 | Approve |
| P-13 | ~~Approve GPU execution~~ — RESOLVED by D-017. Original text: Approve GPU execution of S0a → S0b → R0 → R1 (≈ 3 GPU-h, ≈ 17.6 GiB) | Stage B | Approve after P-12 |
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
