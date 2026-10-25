# 04 — Claim–Evidence Ledger

Status: living document, created 2026-10-24. Every sentence in the paper that states a result must map to a row
here before submission.

**Status definitions**
- **VERIFIED** — confirmed by the PI or the PI's independent reviewer. *No row currently carries this status:*
  the reviewer's Stage A verdict is CONDITIONAL PASS and no archived result has been re-verified independently in
  this phase. Rows move here only on PI sign-off.
- **CLAUDE-REPORTED** — computed or checked by Claude from hash-sealed artifacts with a recorded command;
  reproducible but not independently verified.
- **HYPOTHESIS** — planned; requires the named experiment.
- **UNSUPPORTED** — must not be claimed (withdrawn, contradicted or untestable in scope).

Archive provenance caveat (applies to rows citing them): EXP_020–025 record commit `12d030a`, but
`src/federated/aggregation/multikrum.py` was first committed in `df4d814`; E1–E3 record `df4d814` but used code
committed in `fd332d2`. Those runs executed uncommitted code (the NDSS runner records no dirty flag). Disclose.

## Archived claims (NDSS evidence)

| ID | Exact wording (paper-ready) | Evidence source | Code / config | Runs | Status | Statistical support | Table/Fig | Reviewer risk |
|---|---|---|---|---|---|---|---|---|
| L-01 | "Under a Gaussian update-noise perturbation (ρ = 1) starting at round 20, frozen FedTrace v3 confirms the perturbed client at round 22 with onset estimate 20 in both validation seeds." | archived traces; `paper_assets/tables/main/table02_fedtrace_validation.csv`; online re-computation (A5) | EXP_016 `7f6ca8b`, EXP_017 `bfe9171`; configs `cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed{123,2026}.yaml` | EXP_016, EXP_017 | CLAUDE-REPORTED | 2/2 (CP 95 % [0.16, 1]) — descriptive only | T3 (Class A row), A7 | Gaussian noise is a geometry stress test, not poisoning |
| L-02 | Same for the norm-preserving directional perturbation (c = 0) | as L-01 | EXP_018/019 `12d030a` | EXP_018, EXP_019 | CLAUDE-REPORTED | 2/2 | T3, A7 | same |
| L-03 | "No honest client is persistently confirmed in the four attacked FedAvg validation runs." | table02 col. "Honest persistent FP" | as above | EXP_016–019 | CLAUDE-REPORTED | 0/16 honest client-runs; not a clean-run FP rate | T3 | Scope: attacked runs only; clean FedAvg K=5 untested |
| L-04 | Development runs (seed 42) gave the same outcome | archived traces | EXP_014 `9aa40d1`, EXP_015 `9bd92ab` | EXP_014, EXP_015 | CLAUDE-REPORTED | 2/2 | A7 | seed 42 was used for development |
| L-05 | "Changing the attacker to client_004 or the onset to round 30 preserves attribution with onset error 0 and delay 2." | table05 | `12d030a` | EXP_026, EXP_027 | CLAUDE-REPORTED | 1 run each | A7 | n = 1 each |
| L-06 | Component ablation: magnitude fires for Gaussian, temporal residual for directional; v3 covers both | table03 | traces | EXP_016–019 | CLAUDE-REPORTED | 2 seeds per attack | T6, F4 | "coverage not fusion" is the honest framing |
| L-07 | "Multi-Krum (f = 1) rejects the Gaussian attacker in every post-attack round and never rejects the directional attacker, at both seeds." | table04; MK synthesis docs | EXP_022–025 (dirty-tree caveat) | EXP_022–025 | CLAUDE-REPORTED | 2 seeds | A6 | provenance caveat |
| L-08 | "Multi-Krum's rejection proxy persistently implicates the largest-partition honest client in clean runs (49/50 rounds) at both seeds." | `analysis/MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md` | EXP_020/021 | EXP_020, EXP_021 | CLAUDE-REPORTED | 2 seeds; mechanism is an association | A6, F5 | provenance caveat |
| L-09 | E1 (K = 10): attribution under Gaussian/directional; clean K = 10 run has no honest alarm | `results_ndss/E1_10clients/` | `df4d814` (dirty-tree caveat) | 3 runs, seed 123 | CLAUDE-REPORTED | n = 1 per condition | A7 | provenance caveat |
| L-10 | ρ = 0.5 Gaussian (E2) detected late (alarm 43, onset estimate 41) | `results_ndss/E2_attack_severity/` + Phase R2 analysis | `df4d814` | 1 run | CLAUDE-REPORTED | n = 1 | A7 | late detection must be shown, not hidden |

## New findings from the reviewer-response phase

| ID | Exact wording | Evidence | Code | Runs | Status | Support | Table/Fig | Risk |
|---|---|---|---|---|---|---|---|---|
| L-11 | "On both clean Multi-Krum runs, frozen FedTrace v3 persistently confirms an honest client (onset 44, confirmation 46), driven by an upward drift of z_m (≈2.4 at round 40 to ≈4.0 at round 49)." | `evaluate_detector_v3` on archived traces (command in Prompt 15 audit) | v3 unchanged | EXP_020, EXP_021 | CLAUDE-REPORTED | 2/2 clean MK runs | F5, A6 | **High**: contradicts any unscoped "no false positive" claim; previously unreported |
| L-12 | "Online (causal) FedTrace yields identical scores at rounds ≥ T_ref and identical alarms and attributions to offline v3 on all 21 archived traces." | `src/rev/online.py`; audit script | rev code (uncommitted at time of check) | 21 archived traces | CLAUDE-REPORTED | exhaustive over archive | A5 | needs independent re-run after commit |
| L-13 | "Offline v3 scores of calibration rounds depend on future rounds (self-inclusive calibration); the attribution tie-break uses post-confirmation scores." | `tests/test_rev_online.py` | rev code | synthetic | CLAUDE-REPORTED | unit tests | method §, A5 | none if disclosed |
| L-14 | "With n self-inclusive calibration rounds, k equal-magnitude outliers reach at most z = √((n−k)(n−1)/(kn)); for n = 20, k ≥ 2 never exceeds τ = 3." | derivation (blueprint §4) + `test_self_inclusive_calibration_masks_…` | — | — | CLAUDE-REPORTED (equal-height case); HYPOTHESIS (general case) | analytic | A8 | needs a written proof checked by PI |
| L-15 | "Logging every submitted update leaves training bit-identical and permits bit-exact replay of aggregation and trace features" | `tests/test_rev_hooks_replay.py`; GPU: L-21 | rev code | CPU toy; GPU ResNet-18 EXP_016 (R1) | CLAUDE-REPORTED (CPU toy and one GPU run) | — | A1 | GPU evidence is one configuration (K = 5, seed 123) |
| L-16 | "peer_mean_cosine_similarity is invariant to the client-set iteration order in archived runs" | 8 FedAvg K=5 processes bit-identical on 100/100 prefix rows; 24-permutation check | — | archived | CLAUDE-REPORTED | archive | A1 note | field not used by v3 — but **is** used by the NDSS naive peer-agreement baseline (`apply_naive_detector`: `naive_threshold`, `naive_flag`, `first_flagged_round`) and by the v1 peer signal z_p; any rev baseline configured on this field inherits the order dependence. Archived decision margins: min \|p − naive_threshold\|/\|threshold\| = 1.9e-3, min \|z_p − 3\| = 0.33 (≈ 13 orders of magnitude above fp64 rounding), so no archived decision can flip. Stage B reports this field in its own category, never ignored |
| L-17 | "Historical Multi-Krum runs EXP_020–025 executed code not contained in their recorded commit" | `git log -- src/federated/aggregation/multikrum.py` (first commit `df4d814`) vs recorded `12d030a` | — | EXP_020–025 | CLAUDE-REPORTED | — | A6 caption, reproducibility statement | must be disclosed; results not invalid, provenance incomplete |
| L-18 | "No clean FedAvg K=5 trace exists in the archive; clean-run specificity in the main setting has never been measured" | EXP_008/009 predate trace logging (no `client_round_trace.csv`) | — | EXP_008, EXP_009 | CLAUDE-REPORTED | — | T3 (clean row PENDING) | the NDSS "no persistent honest-client false positives" claim covers attacked runs only |

## Stage B reproduction evidence (2026-10-24) — engineering evidence, NOT scientific validation

All runs: commit `857882a686ac5f6e8f56de60d4d4d945369fe6d8` (branch `rev/stage-a`, execution_clean = true),
torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, RTX 5060 Laptop GPU, driver 572.97, deterministic algorithms on,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, PYTHONHASHSEED not fixed (mirrors the archived launches), seed 123.
Comparison reports: `results_rev/comparisons/*.json` (git-ignored, local). Statuses from `src/rev/compare_runs.py`.

| ID | Exact wording | Evidence | Config (sha256 prefix) | Runs | Status | Support | Table/Fig | Risk |
|---|---|---|---|---|---|---|---|---|
| L-19 | "A 3-round smoke run through the reviewer-response entry point with update logging is bit-identical to the same run through the unmodified NDSS runner, and its update log verifies and replays bit-exactly." | `S0b-vs-S0a.json` (`5a04246f…`): 33 IDENTICAL, 0 differences, 2 EXCLUDED (timing columns; plots/run.log), 1 INFO | S0a `044e7a22…`, S0b `0ba811af…` | S0a (exit 0, 341 s loop), S0b (exit 0, 308 s loop; log 22 records, manifest `6b7fec15…`) | CLAUDE-REPORTED | n = 1 smoke run | — (gate only) | none: engineering check |
| L-20 | "Re-running EXP_016 with the unmodified NDSS runner at commit 857882a reproduces the archived run bit-for-bit: configuration, partition, client selection, local-training trajectory, all trace fields (including the order-dependent peer field), FedTrace v3 scores and decisions, final global model, server/runner/client checkpoints and test metrics (top-1 0.5650)." | `R0-vs-EXP016.json` (`124cdf33…`): 23 IDENTICAL, 0 differences, 3 EXCLUDED (timing; plots/run.log; no update log), 1 INFO (recorded commits `7f6ca8b` vs `857882a`) | R0 `b781144b…` | R0 (exit 0, 18:41:31→20:00:27 UTC, 4,714 s loop) | CLAUDE-REPORTED | one archived run, same seed | A1 | Level A — no tolerance used. Reproduces one archived Class A stress-test run; adds no new seed or scientific evidence. Per-round tensors of the archived run were never saved, so per-round equality is established through trace features, history and final states, not per-round tensors |
| L-21 | "Logging every submitted update (fp32) and every global state does not change training: the logged run R1 is bit-identical to R0 on every compared artifact; the 351-record log verifies, re-aggregation reproduces all 50 logged global states, the last logged global equals the final model, and the trace recomputed from the log equals the live trace." | `R1-vs-R0.json` (`16c649af…`): 33 IDENTICAL, 0 differences, 2 EXCLUDED, 1 INFO | R1 `f7103517…`; resolved `5b8e1e90…` | R1 (exit 0, 20:01:06→21:24:33 UTC, 4,982 s loop; log 14.68 GiB, manifest `96909a09…`) | CLAUDE-REPORTED | one run | A1 | logging overhead +268 s (+5.7 %) on the training loop |

| L-22 | "The R1 update log has 351 sealed records: 250 submitted client states (5 × 50), 50 pre-transform states of the attacker, 1 initial and 50 post-round global states; the attacker's pre-transform state equals its submitted state in rounds 0–19 and differs from round 20 on." | manifest audit (Prompt 18) | R1 | R1 | CLAUDE-REPORTED | exhaustive over the log | A1 | none |

Distinctions (for the paper and reviews): **bit-identical reproduction** — achieved for EXP_016 (L-20) and for
logging non-interference (L-21); **numerically equivalent reproduction** — not needed, no Level B tolerance was
defined or used; **failed reproduction** — none; **scientific validation** — none; Stage B makes no claim about
FedTrace's effectiveness or novelty.

## Development diagnostics (CVPR framing; development data only, never validation evidence)

| ID | Exact wording | Evidence | Runs | Status | Risk |
|---|---|---|---|---|---|
| L-24 | "In both clean Multi-Krum runs every client's mean update norm in rounds 40–49 exceeds its calibration mean (median ratio 1.037 and 1.041); the client falsely confirmed by v3 has the largest ratio (1.091, 1.100). In the clean K = 10 FedAvg run the median ratio is 1.021." | per-client mean(update_l2_norm, rounds 40–49) / mean(rounds < 20) on archived traces | EXP_020, EXP_021, E1 clean | CLAUDE-REPORTED | development data; motivates H-B; shows that part of the drift is client-specific |
| L-25 | "Before the fix, the BadNets wrapper could not read labels of NDSS client datasets (nested `Subset`), which would have aborted the attacked run at client construction; fixed and tested on the real local CIFAR-100 pipeline." | `tests/test_rev_pilot.py` | — | CLAUDE-REPORTED | implementation note, not a scientific claim |

## Stage C pilot P1 (development seed 123; 2026-10-25) — CLAUDE-REPORTED, DEVELOPMENT ONLY (not scientific validation)

Runs at commit `bb995c1` (execution_clean = true): X1_clean_d123 (exit 0, 22:49:47→00:13:58 UTC, loop 5,027 s),
X2_badnets_d123 (exit 0, 00:13:58→01:30:28 UTC, loop 4,564 s), pair analysis (exit 0, 01:30:28→01:40:02 UTC).
Both: same initial model (content hash `57de4405…`), same partition (`93a81d03…`), 301-record sealed update logs.
Outputs: `results_rev/pilot/X2_badnets_d123__vs__X1_clean_d123/`. Single seed: no statistical claim possible.

| ID | Observation (exact) | Status |
|---|---|---|
| L-26 | Matched prefix valid: 21 global states (initial + after rounds 0–19), 100 submissions (rounds 0–19) and 120 trace/history rows bit-identical; first differing global = after round 20; first differing submission = round 20, client_002; the four honest submissions of round 20 are identical | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-27 | ASR on the 9,900 non-target triggered test images: clean model 0.73 %, BadNets model 43.54 % at round 49 (difference +42.8 pp); attacked ASR stays ≤ 5 % in rounds 20–29, 13.5 % at round 30, 40.3 % at round 40; never ≥ 50 % in rounds 20–49. Final clean top-1: 56.88 % (clean) vs 54.94 % (BadNets), paired −1.94 pp (one seed; seed-to-seed noise not measured for this pair) | CLAUDE-REPORTED, DEVELOPMENT ONLY — the predeclared effectiveness criterion (final ASR ≥ 50 %) is **NOT met**; the criterion stays at 50 % |
| L-28 | Online FedTrace v3 attributes client_002: first exceedance round 25, alarm round 30, actionable 31, estimated onset 28 (true 20; error 8), delay 10 rounds; retrospective (offline) onset also 28. Evidence came from z_m only (z_t_res = 0 for client_002 after round 20); client_002's causal scores in rounds 25–34 lie between 2.85 and 3.26 — a margin of at most 0.26 above τ = 3 | CLAUDE-REPORTED, DEVELOPMENT ONLY — detection is marginal |
| L-29 | No honest client exceeded τ in either run (max causal honest score 2.15 clean, 2.25 attacked); no persistent honest alarm | CLAUDE-REPORTED, DEVELOPMENT ONLY (one seed; not a false-alarm rate) |

## Hypotheses (need experiments; see `02_experiment_matrix.md`)

| ID | Wording (to be confirmed or refuted) | Experiment | Status |
|---|---|---|---|
| H-01 | EXP_016 is bit-reproducible at HEAD; logging does not alter GPU training | B-R0, B-R1 | → L-20, L-21 (CLAUDE-REPORTED; awaiting PI audit) |
| H-02 | Clean FedAvg K=5: 0/5 runs with persistent honest alarm | X1 | HYPOTHESIS (at risk, cf. L-11) |
| H-03 | Unconstrained backdoor is effective and attributed in ≥ 4/5 V seeds, delay ≤ 5 | X2 | HYPOTHESIS (thresholds PROVISIONAL) |
| H-04 | Constrained attacker cannot have both full success and evasion | X3 | HYPOTHESIS |
| H-05 | Calibration contamination degrades detection | X4 | HYPOTHESIS |
| H-06 | FedTrace delay ≤ CUSUM/PH delay at matched clean FPR | C1 | HYPOTHESIS |
| H-07 | Partial participation reduces detection via streak breaks | X5 | HYPOTHESIS |
| H-A | Robust (median/MAD) client-specific calibration detects under calibration contamination without more clean alarms | C5, X4 | HYPOTHESIS |
| H-B | Drift-adjusted magnitude reduces benign-drift alarms without delaying BadNets attribution | C6 (X1, X2, archived MK) | HYPOTHESIS |
| H-C | Causal change-time onset beats the streak start when alarms are late | C7 | HYPOTHESIS |

## Unsupported — do not claim

| ID | Claim | Reason |
|---|---|---|
| U-01 | "No false positives" (unscoped) | L-11; clean FedAvg K=5 untested |
| U-02 | Gaussian/directional perturbations are effective poisoning / "ρ = 0.5 is harmless" | no harmful objective measured; EXP_008 is not a matched control (diverges from round 0) — withdrawn |
| U-03 | [10] is FLForensics | withdrawn; [10] unidentified |
| U-04 | 40–50 rounds are mandatory | withdrawn; detector needs R ≥ T_ref + P for onset at T_ref |
| U-05 | Robust to adaptive attackers | not tested |
| U-06 | Any comparison with FLDetector / FedREDefense / FLForensics | mechanics unverified |
| U-07 | Statistical significance from 2 seeds | insufficient n |
| U-08 | GPU equivalence established by CPU tests | different hardware/model |
