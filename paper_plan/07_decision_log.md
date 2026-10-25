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

| D-019 | 10-24 | Stage B evidence commit `8695cc7a835bc2017d4c959d8225fd4acb6f2583` (4 paper_plan files; P-14 resolved) | PI-approved; Claude executed | Prompt 18 §1 |
| D-020 | 10-24 | CVPR framing adopted (blueprint §0): research question, two primary endpoints, candidates A/B/C as single-swap ablations with a pre-specified adoption rule; no merged method yet (P-15 resolved) | PI | Prompt 18 §2 |
| D-021 | 10-24 | `labels_from_metadata` fixed for nested `Subset` (NDSS client datasets); tested on the real local CIFAR-100 pipeline (L-25) | Claude | Prompt 18 §4 |
| D-022 | 10-24 | First matched pair prepared: `configs/rev/X1_clean_d123.rev.yaml`, `configs/rev/X2_badnets_d123.rev.yaml` (same base config; poisoning the only difference); analysis `src/rev/pilot.py`; not run | Claude | Prompt 18 §3 |

| D-023 | 10-25 | P1 pair executed under conditional P-17 (all 8 pre-GPU checks PASS); no retuning, no further runs; results L-26–L-29 (CLAUDE-REPORTED, DEVELOPMENT ONLY); predeclared 50 % ASR criterion not met and not changed; scientific validation NOT established (PI assessment) | PI-approved; Claude executed | Prompt 19 |

| D-024 | 10-25 | Pilot evidence committed locally: `09e39287…` (rows with the original labels — the label-correction script aborted but the commit ran) and `87992016…` (labels corrected to CLAUDE-REPORTED, DEVELOPMENT ONLY; 50 % ASR criterion recorded as not met and unchanged) | Claude (PI-approved commit) | Prompt 20 Task 1 |
| D-025 | 10-25 | P1 causal failure analysis (L-30–L-33) and POST-HOC EXPLORATORY comparisons (L-34, L-35) recorded; no candidate adopted, no threshold selected | Claude | Prompt 20 Tasks 2–5 |

| D-026 | 10-25 | Root cause of D-024: the label-correction command ran a Python edit step followed by `grep`, `git add` and `git commit` separated by `;`/newlines; the Python step exited 1 but the shell continued and committed unedited files. Practice adopted: every multi-step shell command starts with `set -euo pipefail` and chains with `&&`; scripts exit non-zero on any unmet precondition; commits verify the staged file list and expected content first | Claude | Prompt 21 Part 1 |
| D-027 | 10-25 | PI decisions: G1 (seed-2026 matched pair) conditionally approved; G2 deferred until G1 review; G3 not approved; 50 % ASR criterion retained, P1 BadNets classified **partially effective**, no retuning; preregistered protocol (P-21) required before G1 | PI | Prompt 21 |

| D-028 | 10-25 | G1 preregistered and frozen (`paper_plan/08_G1_preregistration.md`); diagnostics made tracked and tested (`src/rev/diagnostics.py`, `tests/test_rev_diagnostics.py`; reproduces the P1 exploratory numbers exactly); G1 configs `configs/rev/X{1_clean,2_badnets}_d2026.rev.yaml` | Claude | Prompt 21 Parts 1–2 |

| D-029 | 10-25 | G1 executed exactly as preregistered (all gates PASS; sequence exit 0); results L-36–L-43 (CLAUDE-REPORTED, DEVELOPMENT ONLY); no detector selected, no threshold tuned, no candidate adopted; G2/G3/validation not launched | PI-approved; Claude executed | Prompt 21 |

| D-030 | 10-25 | Reference-contamination audit (L-44–L-48, POST-HOC EXPLORATORY); no reference alternative selected or implemented; no GPU run | Claude | Prompt 22 |

| D-031 | 10-25 | P-22 resolved: the artifact-supported reading is accepted — Observation D's preregistered criterion (S1–S4 true, F1/F2 false) is SUPPORTED on seed 2026; the criterion text in `08_G1_preregistration.md` is unchanged; the earlier PI review note (criterion not satisfied) is recorded as incorrect; Observation D does not establish a successful detector (v3 misattributed on the same run) | PI | Prompt 23 |
| D-032 | 10-25 | P-23: onset-30 and the four-run matrix deferred; a relabel-only control is prioritised (specification only, no execution). P-24: method-development protocol approved for writing. P-25: audit script to become a tested read-only module in `src/rev/`. Seeds 123 and 2026 remain DEVELOPMENT and are excluded from independent validation | PI | Prompt 23 |

| D-033 | 10-25 | Method-development protocol and relabel-only specification written (`09_method_development_protocol.md`); prior-work checklist (`10_prior_work_checklist.md`); audit converted to tested read-only module `src/rev/reference_audit.py` (reproduces the scratch audit exactly; adds the attacker-induced term); relabel-only attack variant (`apply_trigger=False`, same samples as BadNets) and configs `X3_relabel_d{123,2026}`; pilot analysis gains the untriggered-target-rate column. Nothing executed on GPU | Claude | Prompt 23 |

| D-034 | 10-25 | P-26 conditionally approved (relabel-only d123/d2026 + analyses only). Protocol 09 Amendment 1 (ASR as manipulation check; ρ descriptive with explicit undefined/near-zero/opposite-sign handling; H-MECH algebra vs causation and the full v3 confirmation condition; primary/secondary/safety metrics). Confirmation seeds sealed (`11_seed_registry.md`: 978310317, 478485063; spare 1082418221). Pair analysis gains a non-overwriting `--out-dir`; comparison module `src/rev/relabel_compare.py` | PI / Claude | Prompt 24 |

| D-035 | 10-25 | Relabel-only control executed under conditional P-26 (all gates PASS, sequence exit 0); results L-49–L-54 (CLAUDE-REPORTED, DEVELOPMENT ONLY); ρ classes: mixed (self-cosine), mixed (norm); no threshold tuned, no method selected; no other runs | PI-approved; Claude executed | Prompt 24 |

| D-036 | 10-25 | Phase S (CPU) executed under frozen protocol 12; results L-55–L-60; one post-hoc cross-seed sensitivity outside the protocol, labelled; prior-work identities verified from primary pages (abstract level); no method adopted, no threshold selected | Claude | Prompt 25 |

## Pending decisions (owner: PI unless stated)

| P-29 | Approve committing L-55–L-60, D-036 and the checklist update | next commit | Approve after review |
| P-30 | Approve reading the full texts and code of FLDetector, FLForensics, FedREDefense (web access, no execution of their code) to complete `10` | before any comparative claim | Approve |

| P-26 | ~~Approve GPU execution of the relabel-only control~~ — CONDITIONALLY APPROVED (D-034). Original text: Approve GPU execution of the relabel-only control (2 runs) and the re-analysis of the two BadNets pairs (≈ 3.0–3.5 GPU-h, ≈ 26 GiB) under `09` §3 | next GPU slot | Approve |
| P-27 | ~~Choose and seal ≥ 2 development-confirmation seeds~~ — RESOLVED (D-034). Original text: Choose and seal ≥ 2 development-confirmation seeds (`09` §6) before any candidate is compared | before Phase C | Approve derivation by a fixed rule (e.g. SHA-256 of a sealed string), recorded before use |
| P-28 | Obtain FLDetector / FLForensics / FedREDefense papers and code to complete `10_prior_work_checklist.md` (P-07) | before any comparative statement | Required |

| P-22 | ~~Clarify~~ — RESOLVED by D-031. Original text: Clarify the review note that the seed-2026 negative-residual persistence criterion was not satisfied: the artifacts show the frozen criteria S1–S4 met (attacker 7 consecutive rounds ≤ −3; no other client any round ≤ −3). Decide whether the rule needs a dated amendment (e.g. honest *positive* runs, decay) for future seeds — not retroactively | before next protocol | Amend prospectively only |
| P-23 | ~~Approve the next development runs~~ — RESOLVED by D-032 (relabel-only prioritised, not executed). Original text: Approve the next development runs (estimate only): BadNets onset 30 on seeds 123 and 2026; relabel-to-target without trigger (needs a small attack variant) on both seeds; optionally targeted label flip and a benign data-shift control (needs code) | next GPU slot | Approve onset-30 + relabel-only first |
| P-24 | ~~Approve writing~~ — RESOLVED by D-032. Original text: Approve writing the method-development protocol that will compare reference constructions (inclusive / leave-one-out / trimmed / historical / two-sided) on development data only, with a fixed decision rule | before any method comparison | Approve |
| P-25 | ~~Decide whether to track~~ — RESOLVED by D-032. Original text: Decide whether to track the reference-audit script as a tested module (`src/rev/`) | next commit | Track, if the PI wants it reproducible from the repository |

| ID | Decision needed | Needed before | Recommendation |
|---|---|---|---|
| P-01 | ~~Approve the source checkpoint~~ — RESOLVED by D-014 | — | — |
| P-14 | ~~Commit the Stage B evidence updates~~ — RESOLVED by D-019 | — | — |
| P-15 | ~~Revise for CVPR 2027~~ — RESOLVED by D-020 (section structure and page budget still to be adapted when writing) | — | — |
| P-19 | ~~Approve the next development GPU matrix~~ — RESOLVED by D-027 (G1 only). Original text: Approve the next development GPU matrix (G1 pair seed 2026, G2 onset 30; G3 only if the attack must meet 50 %) ≈ 4.2–5.6 GPU-h | next GPU slot | Approve G1 + G2 first |
| P-20 | ~~Decide how to treat the unmet 50 % ASR criterion~~ — RESOLVED by D-027 (partially effective; G3 not approved). Original text: Decide how to treat the unmet 50 % ASR criterion: (a) report BadNets as partially effective at poison 0.5, or (b) one predeclared development re-parameterisation (G3, poison 1.0) | before validation | (b), decided before seeing G1 |
| P-21 | ~~Approve writing one pre-registered protocol~~ — RESOLVED: G1 protocol `08_G1_preregistration.md` (D-028); a method-development protocol is still required before any method comparison. Original text: Approve writing one pre-registered method-development protocol (detectors, candidates, thresholds from clean development runs, decision rule) before any further CPU comparison | before CPU comparisons | Approve |
| P-16 | Approve downloading one additional vision dataset (and checking its licence) for CVPR breadth | Day 1 | Decide early; without it the single-dataset limitation stays |
| P-17 | Approve GPU execution of the P1 pair (X1_clean_d123, X2_badnets_d123) and the pair analysis (≈ 3 GPU-h, ≈ 26 GiB) | Stage C | Approve after reviewing this phase |
| P-18 | Approve the candidate adoption rule and the fixed κ₀ = 1 for candidate C before any candidate is evaluated | C5–C7 | Approve |
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
