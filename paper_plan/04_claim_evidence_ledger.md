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

### P1 causal failure analysis (CPU, sealed X1/X2 traces; scripts and outputs in `results_rev/pilot/X2_badnets_d123__vs__X1_clean_d123/diagnostics_posthoc/`) — CLAUDE-REPORTED, DEVELOPMENT ONLY

| ID | Observation | Status |
|---|---|---|
| L-30 | The BadNets attacker's update norm **decreased** after onset: 0.955× its clean counterfactual (same client, same seed, X1) at round 20, 0.90× at 21, ≈ 0.83× at 25, ≈ 0.80× at 35; honest clients' norms stayed within 0.998–1.012× of their counterfactuals | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-31 | v3's calibration window includes the first two training rounds, whose norms are far above later ones (client_002: 6.74, 5.91 vs ≈ 5.4 in rounds 2–19). This inflates the calibration sd 1.8–6.3× across clients (client_002: 0.314 with rounds 0–1 vs 0.050 without). With sd 0.314 the attacker's norm must fall below 4.57 to exceed τ = 3; its norms in rounds 25–35 were 4.49–4.62, hence the marginal, intermittent exceedances | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-32 | After onset the attacker's self-cosine **rose** (0.22 at round 20 → 0.45 at 21 → 0.37 at 30) while the round median stayed ≈ 0.22–0.26; the residual d = c(r) − t became strongly negative (signed z = −6.45 at round 21, −4.81 at round 30), but v3's temporal residual is one-sided (max(0, ·)), so z_t_res = 0 from round 21 on | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-33 | Event chain under frozen v3: z_m 1.14 (r20), 2.02–2.75 (r21–24), first exceedance 3.04 (r25), 2.85 / 2.91 (r26–27, streak broken), 3.07 / 3.01 / 3.13 (r28–30) → persistence 3 reached at r30 = alarm; onset = streak start = r28 | CLAUDE-REPORTED, DEVELOPMENT ONLY |

### POST-HOC EXPLORATORY, DEVELOPMENT ONLY comparisons (not method selection; not generalisation evidence)

| ID | Observation | Status |
|---|---|---|
| L-34 | Onset estimated at the v3 alarm by the CUSUM change time on the v3 score (κ₀ ∈ {0.5, 1, 1.5, 2}, all reported): X2 → 20 / 20 / 21 / 21 (v3: 28); archived E2 ρ = 0.5 → 20 / 20 / 20 / 26 (v3: 41); identical to v3 on the 11 other archived traces. Caveat: true onset equals T_ref in 12 of 13 traces and the estimator cannot return a value below T_ref, so these results favour it; only EXP_027 (onset 30) tests it away from that floor | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-35 | Two-sided detectors with thresholds set on the single clean development run X1 (k = 0.5 / δ = 0.5 fixed a priori): CUSUM on standardised norm h = 8.75 → X2 alarm 25, onset 20, correct, but an honest late alarm (client_000, round 48); CUSUM on residual h = 30.5 → alarm 26, onset 21, correct, no honest alarm; Page–Hinkley on norm λ = 6.5 → attacker missed, honest client_000 attributed at 49; Page–Hinkley on residual λ = 0.25 (degenerate threshold) → three clients alarmed at 21, wrong attribution. A threshold calibrated on one clean run is unreliable | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |

## G1 — preregistered replication on development seed 2026 (2026-10-25) — CLAUDE-REPORTED, DEVELOPMENT ONLY

Protocol `paper_plan/08_G1_preregistration.md` (frozen in commit `a7290fc`, before the runs). Runs at `a7290fc`
(execution_clean = true): X1_clean_d2026 (exit 0, 04:24:39→05:36:40 UTC, loop 4,297 s), X2_badnets_d2026 (exit 0,
05:36:40→06:47:51, loop 4,247 s), pair analysis (exit 0, 10 min), diagnostics (exit 0). Same initial model
(`c74c2616…`) and partition (`1d7757a2…`) in both arms. Outputs `results_rev/pilot/X2_badnets_d2026__vs__X1_clean_d2026/`.

| ID | Endpoint / observation | Preregistered verdict | Status |
|---|---|---|---|
| L-36 | Matched prefix valid: 21 globals, 100 submissions, 120 trace/history rows bit-identical; first difference = round 20 (client_002); the four honest round-20 submissions identical | (prerequisite) VALID | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-37 | E1: attacker norm / counterfactual over rounds 25–35 = 0.791; honest 1.007–1.012 | REPLICATED | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-38 | E2/Observation D, three separate facts. (a) **Magnitude:** attacker signed residual z = 0.38 (r20), −19.5 (r21), −11.8, −11.4, −9.4, −8.2, −6.0, −3.86 (r27), −2.77 (r28), −1.99 (r29); median rounds 20–29 = −7.09. (b) **Preregistered criteria as written:** S1 (min z r20–22 = −19.5 ≤ −3) true; S2 (median −7.09 ≤ −3) true; S3/S4/F2 are *absence* conditions — no client in the clean run and no honest client in the attacked run has any round with z ≤ −3 — true; F1 false. The attacker itself had 7 consecutive rounds with z ≤ −3 (r21–27), after which the shift decays (z > −3 from r28; ≈ −0.31 from r30). (c) **Limits of the rule:** it tests negative residuals only; honest client_001 and client_004 had 17 and 27 consecutive rounds with z ≥ +3 in the attacked run (the false-alarm mechanism, L-43), which the rule does not examine. | Rule verdict SUPPORTED (frozen rule, unchanged). This is NOT detector performance (v3 misattributed, L-41). A PI review note stating that the persistence criterion was not satisfied was not reproduced by the artifacts; the PI accepted the correction (D-031); criterion text unchanged | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-39 | E3: honest signed residuals, rounds ≥ 20 — clean run within [−1.11, +1.64], none beyond ±3; attacked run: 38.3 % of honest client-rounds ≥ +3 (max +4.87) | (reported) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-40 | E4: sd(rounds 0–19)/sd(rounds 2–19) = 6.48, 4.01, 7.05, 2.36, 2.83 (clients 000–004); exploratory b = 1/3/5: 1.68–2.58 / 2.71–8.96 / 2.89–9.26 | REPLICATED (5/5 clients ≥ 1.5) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-41 | E5 frozen v3 (online): clean run — no alarm; attacked run — client_001 alarm 23 (onset est. 21), client_004 alarm 23 (21), client_002 alarm 24 (22); attribution = client_004 (tie at 23 broken by streak mean 4.24 vs 3.56) → **WRONG**; honest false alarms: client_001, client_004 (both via z_t_res); attacker detected via z_m at 24 (actionable 25), onset error 2, delay 4 | (reported) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-42 | E6: final ASR 64.67 % (attacked) vs 0.27 % (clean), +64.4 pp; ASR 1.8 % at r20 → 21.3 % at r30 → 54.2 % at r40 (logged-global evaluation); final clean top-1 57.94 % vs 57.00 % (−0.94 pp; NDSS final test evaluation with AMP — the logged-global re-evaluation without AMP gives 57.92 % vs 56.99 %) | EFFECTIVE (≥ 50 %) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-43 | Mechanism of the misattribution (descriptive, not preregistered): before onset client_002 was the median-consistency client (its self-cosine equalled the round median, 0.265–0.269); after onset its self-cosine rose to 0.43, so the 5-client median moved to the next client (0.257 clean counterfactual vs 0.333 attacked at round 21). v3's same-round reference includes all clients (no leave-one-out), so the two least self-consistent honest clients (self-cosine ≈ 0.19–0.21) acquired positive residuals > 3σ and were confirmed first | — | CLAUDE-REPORTED, DEVELOPMENT ONLY — mechanism observed on one seed |

Contradictions with seed 123 (both development): v3 attribution correct at 123 (alarm 30) but wrong at 2026 (honest
alarms at 23); BadNets partially effective at 123 (43.5 %) but effective at 2026 (64.7 %); no honest alarm at 123
versus two at 2026. Consistent across seeds: attacker norm drop (0.82 vs 0.79), attacker self-consistency rise
(Observation D), calibration-spread inflation (5/5 clients both seeds).

### Reference-contamination audit (Prompt 22; CPU, sealed logs of both development seeds) — POST-HOC EXPLORATORY, DEVELOPMENT ONLY

Script and outputs: `results_rev/pilot/reference_audit_posthoc/` (local, git-ignored). Clean counterfactuals used for diagnosis only.

| ID | Observation | Status |
|---|---|---|
| L-44 | Seed 2026: v3's round reference is the median over all five clients including the candidate (verified numerically). Pre-onset the attacker *was* the median client (self-cosine = median, 0.269 at r19). At r21 its self-cosine rose to 0.432 and the median moved to 0.333 (clean counterfactual 0.257). client_001 (t ≈ 0.19–0.21, calibration μ 0.018, σ 0.034) and client_004 (μ 0.016, σ 0.026) reached z_t_res 3.46 / 4.00 at r21 and stayed > 3 → confirmed at r23; the attacker's own evidence was z_m only (3.01 at r22, confirmed r24); its z_t_res was 0 because its residual was negative. Honest z_m stayed < 0.6 — no norm contribution to the false alarms | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-45 | Decomposition of the post-onset residual change (attacked − matched clean, mean r20–29, z units): seed 2026 honest clients +1.30 / +1.84 / +1.47 / +2.44 (000/001/003/004), of which +1.25 / +1.79 / +1.38 / +2.31 from reference movement and +0.05 / +0.05 / +0.09 / +0.12 from their own updates; seed 123 honest clients −0.10 … −0.72 (reference moved down; no false alarms) | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-46 | The attacker's residual-calibration σ at seed 2026 is 0.0051 (it was the median client, so d ≈ 0 during calibration); its signed z (−19.5) is inflated by this, while its raw self-cosine rise vs counterfactual (+0.115 mean r21–29; seed 123: +0.120) is the scale-free observation | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-47 | Reference alternatives with frozen τ = 3, P = 3 and z_m unchanged (no tuning): inclusive median — 123 correct / 2026 wrong (2 honest alarms); leave-one-out median — 123 **wrong** (honest client_001 alarm r26 before attacker r30) / 2026 correct; trimmed mean (inclusive) — same pattern as leave-one-out; client-historical reference — attribution correct in both attacked runs but late honest alarms in clean runs (123: three clients r44–49; 2026: two clients r42–48) and attacked runs (r43–49). No alternative avoided false alarms on both seeds | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-48 | Attacker update geometry (logs): raw self-cosine +0.12 above counterfactual in both seeds; honest clients −0.002 … −0.004 (global-trajectory effect on honest consistency negligible); class-0 row of the classifier update carries 0.45–0.59 of the classifier update norm (clean 0.06–0.10); consistency also rose in layer4 (e.g. 0.37 vs 0.21 at r21, seed 2026); the attacker holds **no class-0 samples** in either seed (53 and 57 classes present) | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |

## Relabel-only control (protocol 09 §3 + Amendment 1; 2026-10-25) — CLAUDE-REPORTED, DEVELOPMENT ONLY

Runs at `3f2f3b4` (execution_clean): X3_relabel_d123 (exit 0, 15:25:30→16:39:42 UTC, loop 4,425 s), X3_relabel_d2026
(exit 0, 16:39:42→17:48:59, loop 4,132 s); pair analyses, diagnostics, BadNets re-analyses (separate folder
`results_rev/pilot/reanalysis_target_rate/`, original BadNets outputs untouched), comparison — all exit 0. Outputs:
`results_rev/pilot/relabel_comparison/relabel_comparison.json`.

| ID | Observation | Status |
|---|---|---|
| L-49 | Matched prefix valid for both relabel pairs (21 globals, 100 submissions, 120 rows; first difference r20 client_002; honest round-20 submissions identical); same poisoned indices as BadNets (3,601 / 4,586), no trigger | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-50 | Attacker raw self-cosine change vs clean (r21–29): BadNets +0.1195 / +0.1146, relabel +0.0750 / +0.0836 → ρ = 0.63 / 0.73 (MID, MID) → preregistered class **mixed**; norm effect 1 − ratio: BadNets 0.181 / 0.209, relabel 0.126 / 0.140 → ρ = 0.70 / 0.67 → **mixed**. Reading: relabel-only reproduces part (≈ 2/3) of both changes in both seeds — descriptive, not causal proof | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-51 | Frozen v3 on relabel-only: seed 123 — no alarm at all (attacker missed; max causal attacker score 2.43); seed 2026 — attacker missed (score > 3 only at r20), honest client_001 and client_004 confirmed at r23 (same pattern as BadNets), attributed client = client_004 → false accusation without detection. Attacker-induced reference movement for those honest clients: +1.97 / +2.55 z (BadNets +1.79 / +2.31) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-52 | Relabel-only effects on the model: triggered ASR 0.72 % / 0.39 % (clean 0.73 % / 0.27 %) — trigger absent as intended; untriggered target-class rate only transiently above clean (peak 1.5 % / 1.3 % vs clean ≤ 1.3 % / 0.5 %; final 0.68 % / 0.35 % vs 0.74 % / 0.25 %) — no lasting target bias; final clean top-1 55.64 % / 56.33 % (clean 56.86 % / 57.92 %; BadNets 54.92 % / 56.99 %, re-evaluated without AMP) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-53 | Observation D rule (G1 diagnostics module) applied to relabel-only: seed 123 SUPPORTED (S1 −3.55, S2 −3.51); seed 2026 FALSIFIED by F1 (median r20–29 = −0.31), although the raw self-cosine rose +0.084 — the attacker became the median client again in later rounds (residual ≈ 0 relative to itself; its calibration σ is 0.0051). Signed residuals are unreliable for the median client; the raw self-cosine change is the scale-free measure | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-54 | Seed-123 attacker-induced reference terms are identical in the BadNets and relabel arms to 6 decimals; verified: the term is non-zero only in round 20 (first poisoned update below the median in both arms, honest round-20 submissions identical), zero in rounds 21–29 (attacker above the median) | CLAUDE-REPORTED, DEVELOPMENT ONLY (verification note) |

## Phase S — standard baselines (protocol `12_phase_s_protocol.md`, frozen in `f706150` before results) — CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY

Harness `src/rev/phase_s.py`; outputs `results_rev/pilot/phase_s/phase_s.json`. Six development runs; four attacked.

| ID | Observation | Status |
|---|---|---|
| L-55 | Group V (fixed τ = 3, P = 3): inclusive median (frozen v3) — attribution correct 1/4 attacked runs (123 BadNets), honest false accusations in 2/4 runs (2026 BadNets, 2026 relabel), relabel attacker missed in both seeds; leave-one-out and trimmed mean — correct 1/4 (2026 BadNets), honest accusation of client_001 in both seed-123 runs, 2026 relabel missed; historical reference — correct 2/4 (both BadNets), honest alarms in all four attacked runs (late, r43–49) and in both clean runs (3 and 2 clients, from r42) | CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY |
| L-56 | Group Q, thresholds from the two clean development runs (§4): ZP on norm (h = 1.75) — attribution correct 4/4, no honest alarm, attacker alarm r22–23, onset 20–21; CUSUM on norm (h = 11.75) — correct 4/4, no honest alarm, alarm r25–27, onset 20; ZP and CUSUM on residuals dA/dB/dC — honest alarms in most attacked runs (1–3 clients); PH on residuals — degenerate thresholds (0.25–0.75), alarms at r21 with 0–4 honest clients; PH on norm — attacker missed in all four, honest alarms at r49; ZP on raw self-cosine and on dD — attacker missed in all four; CUSUM on t and on dD — NOT CALIBRATABLE on the grid; PH on t / dD — correct 3/4, late (r39–47), relabel 123 missed | CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY |
| L-57 | Calibration caveat: clean calibration runs share seeds and partitions with the attacked runs, and honest clients behave almost identically in both (norm ratios ≈ 1.00–1.01), so group-Q honest false-alarm counts are optimistically biased; clean-run alarms of group Q are zero by construction | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-58 | Post-hoc cross-seed sensitivity (outside protocol 12; `posthoc_cross_seed_sensitivity.json`): ZP-norm calibrated on one seed's clean run (h = 1.75 for both) stays correct with no honest alarm on the other seed's runs; CUSUM-norm calibrated on seed 123 (h = 8.75) raises honest alarms on seed 2026 (clean: client_002; BadNets: client_000, client_001); PH-t calibrated on seed 2026 raises an honest alarm on seed 123 and misses relabel | POST-HOC EXPLORATORY, DEVELOPMENT ONLY |
| L-59 | Reference contamination (diagnosis only): attacker-induced movement of honest references — inclusive median: seed 123 −0.09 … −0.72 z, seed 2026 +1.25 … +2.55 z; leave-one-out and trimmed mean: seed 2026 reduced to +0.51 … +0.71 z but seed 123 **increased** to +0.30 … +2.30 z (client_001); historical: 0 by construction. A four-value leave-one-out median averages the middle pair, which the attacker can occupy; whether an order-statistic reference is contaminated depends on the attacker's rank | CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY |
| L-60 | Prior work, identities verified from primary pages (abstract level only): FLDetector (KDD 2022) detects clients whose updates are inconsistent across iterations (history-based prediction, L-BFGS) and removes them; FLForensics (NeurIPS 2025) post-hoc traces malicious clients from a misclassified target input; FedREDefense (ICML 2024) filters clients by update reconstruction error without cross-client information | VERIFIED IDENTITY / ABSTRACT; full methods TO VERIFY |

## Calibration sensitivity and prior-work verification (Prompt 26; CPU) — CLAUDE-REPORTED, DEVELOPMENT ONLY

| ID | Observation | Status |
|---|---|---|
| L-61 | Norm-feature calibration (`results_rev/pilot/phase_s/calibration_sensitivity.json`): z-score threshold 1.75 from seed 123 alone, seed 2026 alone and both; CUSUM threshold 8.75 (seed 123), 11.75 (seed 2026, both). The binding clean statistic for the z-score is client_002 at seed 2026 (1.719; margin 0.031). Lag-1 autocorrelation of clean norm z-scores 0.82–0.91 per client (AR(1) effective n ≈ 1.5–2.9 of 30 rounds); mean pairwise cross-client correlation 0.94 (seed 123) and 0.95 (seed 2026): 300 nominal clean client-rounds carry only a handful of effectively independent observations, with 2 independent runs | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-62 | Prior work from full text (FLDetector, FLForensics) and code (FedREDefense; paper text not readable here): none estimates attack onset; FLDetector and FLForensics rely on cross-client normalisation/clustering; FedREDefense uses a fixed per-client threshold plus a cross-group polarity check; all are evaluated with 100–300 clients; FLDetector's history-based update-consistency score is the closest prior to FedTrace's temporal evidence (details and sections in `10_prior_work_checklist.md` §A1) | VERIFIED FROM FULL TEXT (FLDetector, FLForensics) / CODE (FedREDefense) |

## Benign client-specific drift control (protocol 13 + Amendment 1; 2026-10-25) — CLAUDE-REPORTED, DEVELOPMENT ONLY

Runs at `934c9c0`: X4_drift_d123 (exit 0, 20:23:21→21:37:27 UTC, loop 4,422 s), X4_drift_d2026 (exit 0,
21:37:27→22:46:48, loop 4,140 s); pair analyses with `--benign` and `src.rev.benign_drift` exit 0. All clients honest.
Outputs: `results_rev/pilot/X4_drift_d{seed}__vs__X1_clean_d{seed}/`, `results_rev/pilot/benign_drift/benign_drift.json`.

| ID | Observation | Status |
|---|---|---|
| L-63 | Matched prefix valid for both seeds (21 globals, 100 submissions, 120 rows bit-identical; first difference round 20 at client_002; honest round-20 submissions identical) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-64 | **H-BEN FALSIFIED for all three preregistered detectors.** Seed 123: v3 — no alarm; z-score on norm (h 1.75) — drifted client_002 first crosses at r21, confirmed r23; CUSUM on norm (h 11.75) — client_002 confirmed r28. Seed 2026: v3 — honest client_001 and client_004 first cross at r21, confirmed r23, attribution would name client_004; z-score — client_001 (not drifted) confirmed r49; CUSUM — client_002 confirmed r37. A single crossing of client_000 at r47 (seed 123) was not confirmed | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-65 | Signatures: drifted client norm ratio (r25–35) 0.872 (seed 123; class DECREASE → norm signature recorded **not specific to poisoning**) and 0.917 (seed 2026; NO MATERIAL CHANGE by the 0.90 rule); other clients 1.002–1.011; raw self-cosine change +0.064 / +0.067 (below the preregistered +0.075 → self-cosine signature **not** recorded as non-specific, although its size is ≈ 0.85–0.89 of the smallest poisoning value); signed residual change of client_002 −2.37 / −0.24 | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-66 | Reference contamination by benign drift: seed 2026 drift-induced movement of the other clients' reference +1.12 / +1.59 / +1.23 / +2.06 z (client_000/001/003/004; ≥ +1 → **recorded**), honest standardised residual change +1.09 … +2.13; the resulting v3 confirmations (client_001, client_004 at r23) reproduce the BadNets / relabel-only pattern of the same seed; seed 123 movement −0.09 … −0.72 (no contamination) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-67 | Final clean top-1: 55.13 % vs 56.88 % clean (seed 123, −1.75 pp); 56.15 % vs 57.94 % (seed 2026, −1.79 pp). Intervention: 25/53 and 30/57 classes retained, 44.1 % / 47.2 % of samples replaced, 4,027 / 4,842 unique samples after onset, length and optimiser steps unchanged | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-68 | **Provenance deviation:** X4_drift_d2026 recorded `execution_clean = False` because the tracked file `paper_plan/10_prior_work_checklist.md` was edited (documentation only) while X4_drift_d123 was running; recorded `executed_uncommitted_modules = []`, `used_uncommitted_files = []`, and the recorded diff hash equals the diff of that single Markdown file; executed code and configs are those of `934c9c0`. The flag is left as recorded. P-34 audit (2026-10-25): recorded diff hash = hash of the checklist-only diff; the file was last modified 20:24 UTC, before the run started (21:37 UTC); no code reads `paper_plan/` (only docstrings mention it); all executed modules and the three config files equal commit `934c9c0` (config hashes identical at HEAD); packages, dataset files, initial model and partition unchanged; matched prefix bit-identical | **SCIENTIFICALLY ACCEPTED WITH DOCUMENTED PROVENANCE DEVIATION** (execution_clean=False preserved) |

## Development evidence matrix and identifiability audit (Prompts 28–29; CPU, read-only) — CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY

| ID | Claim | Status |
|---|---|---|
| L-69 | Eight-run matrix (`src/rev/evidence_matrix.py`; clean / BadNets / relabel-only / benign drift × seeds 123, 2026; intervened client_002, onset 20; malicious = BadNets and relabel-only only). Detector decisions reconcile exactly with `phase_s.json` and `benign_drift.json`; v3 scoring order equals the online wrapper's attribution in all 8 runs; the 142 run-artifact files are hash-identical before/after. Norm detectors (z-score h 1.75, CUSUM h 11.75) attribute client_002 in all 4 attacked runs and accuse an honest client in both drift runs; v3 detects but misattributes (client_004 first, 2 honest accusations) at seed 2026 in BadNets, relabel-only and drift, and misses relabel-only / drift at seed 123. Clean runs: no confirmation under any detector. Final top-1 change vs clean (training-time evaluation): BadNets −1.94 / −0.94 pp, relabel −1.26 / −1.60 pp, drift −1.75 / −1.79 pp; ASR BadNets 43.5 / 64.7 %, all other arms ≤ 0.7 % | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-70 | Integrity review: (i) seed-2026 BadNets accuracy sign verified negative (0.5700 − 0.5794 = −0.94 pp; pilot re-evaluation of the round-49 model 0.5699 − 0.5792 = −0.93 pp; the two evaluations differ by ≤ 0.03 pp in all 8 runs); (ii) "reference movement" = per other client k, mean over rounds 20–29 of [inclusive median of the intervened run − same median with client_002's self-cosine replaced by its matched-clean value] / k's clean calibration sd (diagnostic; uses the counterfactual); (iii) **correction to the Prompt 28 report:** its "largest reference movement" column was the signed maximum; at seed 123 the largest-magnitude movement is −0.72 (client_001), not −0.09; the matrix now reports signed and absolute maxima separately; (iv) seed-123 movement is identical in all three arms because only round 20 contributes (client_002 is already above the median in the clean run and stays on the same side afterwards); (v) missing self-cosines now make the movement undefined instead of raising | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-71 | Identifiability (post hoc, n = 2 per arm, nothing fitted): update norm, raw self-cosine (relabel vs drift), signed residual and shared-reference deviation are **demonstrably non-specific** between poisoning and benign drift; client-historical deviation inherits the norm result and its calibration (effective n ≈ 1.5–2.9) is not evaluable. Classifier-row concentration of client_002 (rounds 21–29): poisoning 0.375–0.52 on class 0 every round; drift 0.29–0.40; honest clean clients 0.19–0.39 (client_000, seed 2026, one class throughout) — magnitude overlaps; "dominant class outside the client's own pre-onset update footprint" separated 4 of 5 drift classes — **potentially discriminative but unvalidated**, observed after the fact, and equally produced by a benign client that acquires a new class. Intent is not identifiable from the logged evidence | CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY |

## D1 explaining-away source localization (protocol 14, frozen in `f66cc62` before outcomes; CPU) — CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY

| ID | Claim | Status |
|---|---|---|
| L-72 | **F1 FALSIFIED (motivating case).** B-MED (leave-one-source-out influence, round-29 cumulative) ranks client_002: seed 2026 BadNets **5th of 5** (top client_001), relabel 1st, drift 1st; seed 123 BadNets 2nd (top client_000), relabel 1st, drift 1st. Full orders (B-MED): 123 clean 1-4-3-2-0; 123 BadNets 0-2-1-3-4; 123 relabel 2-0-1-3-4; 123 drift 2-0-1-3-4; 2026 clean 2-1-4-3-0; 2026 BadNets 1-4-0-3-2; 2026 relabel 2-1-4-0-3; 2026 drift 2-1-4-0-3 (client index). F5: estimator-dependent at seed-2026 relabel (TRIM, MEAN → client_001; also drift) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-73 | F2: **no evidence that D1 adds information beyond median influence** (vs ABS: better 3 / worse 1; vs SIGNED: better 5 / worse 1; the worse case is the motivating seed-2026 BadNets run). Rank of client_002 (6 intervened runs, 123 B/R/D, 2026 B/R/D): ABS 4,4,4,1,1,1; SIGNED 5,5,5,3,3,3; leave-one-out A 1,2,2,1,1,1; own self-cosine change 4,5,5,3,5,5; own norm change 1,1,1,1,1,1 (non-specific: drift included; clean-run winners exist) | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| L-74 | Tautology / instability (post hoc mechanism checks): (i) **attacker-residual leakage** — at seed 2026 BadNets, 19.2 of client_001's score 22.4 (and 19.1 of client_004's 21.1) comes from client_002's own residual (z ≈ −19.5 at r21) re-entering as an "other" client when another candidate is excluded, so the true source is penalised the more anomalous it is; (ii) **median-position mechanism** — in both clean runs the per-round top is the median-position client in 10/10 rounds (default winner client_001 at seed 123 and **client_002 at seed 2026**); at seed-2026 relabel / drift client_002 sits at the median after onset (z ≈ −0.31 constant) and is selected by the same mechanism (position 3 in 7/10 rounds); (iii) seed-123 localizations rest on round 20 alone (per-round I = +12.48, negative in rounds 21–29; per-round top client_000; F4 UNSTABLE for 123 relabel and drift, 1/10 rounds); (iv) no exact or near ties for B; ABS / SIGNED have exact ties at the top in both clean runs (same-side candidates, as predicted by the K = 5 median algebra); (v) F3: clean top-second margins 0.157 / 0.132 below the smallest correct margin 0.230 | CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY |
| L-75 | Outcomes kept separate: source localization (L-72); anomaly detection — the existing frozen detectors (L-69), and descriptively the B top score exceeds both clean-run top scores in all six intervened runs, drift included; **malicious attribution — none**: localization of client_002 in the drift runs is localization of a benign change, and no D1 quantity distinguishes drift from poisoning | CLAUDE-REPORTED, DEVELOPMENT ONLY |

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
| H-MECH | Honest false accusation by v3 arises from attacker-induced reference movement exceeding the honest client's pre-onset margin, with small own-behaviour change (P1–P3, `09` §2) | relabel-only control, later runs | HYPOTHESIS (descriptive mechanism, not a method) |

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
