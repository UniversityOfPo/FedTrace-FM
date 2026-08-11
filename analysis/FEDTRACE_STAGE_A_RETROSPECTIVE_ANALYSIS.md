# FedTrace Stage A: Retrospective Offline Evaluation on EXP_014 / EXP_015

**Status:** Stage A only (retrospective, offline, on already-collected data). No new experiment was run. No FedTrace method redesign occurred as a result of these findings — every number below was produced by applying the method exactly as frozen in `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`.

---

## 1. Purpose and Scope

This document evaluates the frozen FedTrace multi-signal detector against the two malicious-client traceability pilot runs already on disk — `EXP_014` (`gaussian_update_noise`, ρ=1.0) and `EXP_015` (`norm_preserving_directional`, target_cosine=0.0), both attacking `client_002` starting at round 20, seed 42. The evaluation is purely retrospective: it re-reads each run's already-saved `client_round_trace.csv`, computes the frozen calibration/score/persistence pipeline, and reports what the method would have concluded — nothing about the method itself was changed based on what these numbers turned out to be.

The governing question is **not** "can FedTrace detect the EXP_015 attack" (already effectively known from `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`'s discovery that `self_prev_round_cosine` collapses to ≈0 post-attack). The governing question is whether the frozen multi-signal *combination* — calibrated once, applied identically to both attack families, never tuned per-attack — adds genuine detection value over the best single signal, or only inherits coverage from whichever signal happens to be discriminative for a given attack.

## 2. Restrictions Honored

No federated-learning experiment or training was run. No `EXP_016` was created. No experiment config was created or modified. `src/attacks/traceability.py` and all federated training code are unmodified (verified: this task only added new files under `src/analysis/` and `tests/`, and this document). `Research_Database`, `README.md`, and every other pre-existing document are unmodified. The method design (`research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`) was not altered after seeing these results — every calibration rule, threshold, persistence value, and combiner choice below matches that document exactly, decided before this evaluation was run. No shell, PowerShell, or Bash command was executed at any point in this task (see §17 and the final report).

## 3. Data Availability Verification

Both trace CSVs were read in full (all 250 rows: 5 clients × 50 rounds) for this evaluation.

| Signal | Field | `EXP_014` (`gaussian_update_noise`) | `EXP_015` (`norm_preserving_directional`) |
|---|---|---|---|
| `m` (magnitude) | `update_l2_norm` | **Available**, all 250 rows | **Available**, all 250 rows |
| `p` (peer-cosine) | `peer_mean_cosine_similarity` | **Available**, all 250 rows | **Available**, all 250 rows |
| `t` (temporal) | `self_prev_round_cosine` | **Absent** — column does not exist in the header (14-field schema) | **Available**, all rows with `round_id >= 1` (round 0 has no prior-round delta for any client, by construction) |

`EXP_014`'s header is confirmed, by direct read, to be exactly the 14 original fields (`round_id` … `first_flagged_round`); the four `norm_preserving_directional`-era fields (`directional_target_cosine`, `delta_attack_norm_ratio`, `realized_cosine_to_honest`, `self_prev_round_cosine`) do not exist in that file at all — not blank, not zero, simply not a column. This is a **logging-generation artifact** (the code that produces `self_prev_round_cosine` postdates `EXP_014`'s run), not evidence that the temporal signal is inapplicable to Gaussian-noise attacks — that question cannot be answered from existing data, and this document does not attempt to answer it.

Rounds 0–19 (the honest, pre-attack window) are bit-identical between the two runs for `update_l2_norm` and `peer_mean_cosine_similarity` (verified directly: e.g. `client_000` round 0 `update_l2_norm = 7.727471901133038` in both files) — expected, since both runs share seed 42 and code up to the point the attack activates at round 20.

**Consequence for the detector matrix:** any detector requiring `t` (`C_temporal_only`, `E_magnitude_temporal`, `F_peer_temporal`) is **NOT EVALUABLE on `EXP_014`** — `src/analysis/fedtrace_detector.py`'s `evaluate_detector` returns `evaluable=False` for these on that run, per its `field_available` check, rather than silently substituting a fabricated 0/undefined value. `G_full_fedtrace` remains technically evaluable on `EXP_014` (the frozen method excludes missing components rather than failing outright), but with `t` undefined for every row it is **numerically identical to `D_magnitude_peer`** on that run — this is reported explicitly in §10, not presented as an independent full-method result.

## 4. Method Implementation Summary

The frozen method is implemented at `src/analysis/fedtrace_detector.py`, deliberately placed outside the empty `src/attribution/` placeholder (reserved, per `traceability.py`'s own docstring, for a future production implementation only once Stage A/B validate it). No part of the calibration, scoring, or persistence logic was invented for this task — every formula matches `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §6–9:

- `z_m`, `z_t`: per-client longitudinal calibration, mean/std over each client's own rows with `round_id < T_ref` (T_ref=20), sample standard deviation (`statistics.stdev`, matching `apply_naive_detector`'s existing convention), excluded (not zeroed) if fewer than 2 observations or zero variance.
- `z_p`: per-round cross-sectional calibration, mean/std over that round's own selected clients' `peer_mean_cosine_similarity` (including the malicious client itself — no leave-one-out, no historical window), same exclusion rule.
- `FedTraceScore(k,r) = max` over whichever of `{z_m, z_p, z_t}` are defined for that row — a missing signal is never treated as 0.
- Raw-anomalous: score > 3. Persistence: P=3 consecutive anomalous rounds. Attribution: earliest-confirmed client. Onset estimate: first round of the confirmed persistent run (not the confirmation round). `attack_start_round` is read only for the post-hoc metrics in `evaluate_detector`, never inside `apply_persistence_rule`.

Unit tests exercising this module (calibration windowing, the three-way persistence distinction, missing-component exclusion, NOT-EVALUABLE detection, attribution tie-breaking, precision/recall) are written at `tests/test_analysis_fedtrace_detector.py`, per this task's restriction, **not executed**.

## 5. Calibration Statistics

**`m` (update_l2_norm), per-client longitudinal, rounds 0–19 (n=20), identical for both runs:**

| Client | mean | std |
|---|---|---|
| `client_000` | 6.5156 | 0.3150 |
| `client_001` | 5.8221 | 0.4836 |
| `client_002` | 6.0034 | 0.2690 |
| `client_003` | 6.0395 | 0.3421 |
| `client_004` | 5.9329 | 0.2770 |

**`t` (self_prev_round_cosine), per-client longitudinal, rounds 1–19 (n=19), `EXP_015` only:**

| Client | mean | std |
|---|---|---|
| `client_000` | 0.3015 | 0.0599 |
| `client_001` | 0.3194 | 0.0376 |
| `client_002` | 0.3119 | 0.0385 |
| `client_003` | 0.3136 | 0.0413 |
| `client_004` | 0.3131 | 0.0405 |

**`p` (peer_mean_cosine_similarity), per-round cross-sectional (n=5 per round, both runs):** computed per round rather than once; representative values reported in §9 alongside the rounds they were used for. A methodological note on sampling density is given in §16.

## 6. Persistence Rule Mechanics

Three distinct quantities are reported per detected client, exactly as specified — conflating them would misstate detection speed:

- **First anomalous round**: the first single round (raw, score > 3), regardless of whether it was ever confirmed persistent. Can be an isolated flicker.
- **Confirmation round**: the round at which the 3rd consecutive anomalous round is observed — the round the detector actually "fires."
- **Onset estimate**: the first round of that confirmed 3-round run — always `confirmation_round − 2` when the run is contiguous, and the quantity compared against ground-truth `attack_start_round` for localization error. This is a retroactive label: it never uses information from a future round to score an earlier one, only delays the *decision* to act on an already-computed score until persistence is confirmed.

## 7. Detector A — Existing Peer-Cosine Baseline

Read directly from each run's already-computed `naive_flag`/`first_flagged_round` columns (no new computation; this is `apply_naive_detector` from `src/attacks/traceability.py`, already run at experiment time). Across every row of both runs' trace CSVs, `naive_flag` is `False` for every client, every round — confirmed by direct inspection of all 250 rows in each file. `first_flagged_round` is empty (`None`) for every client in both runs.

**Result: Detector A never detects `client_002` in either run.** This matches the already-documented finding in `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` and `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md` — the per-client longitudinal threshold on raw peer-cosine is defeated by the shared secular decline in peer-cosine that affects every client (honest and malicious) as training converges, which the frozen FedTrace `p` calibration was specifically designed to fix (§9 tests whether it succeeds).

## 8. Detector B — Magnitude-Only (`z_m`)

**`EXP_014`:** `client_002`'s `update_l2_norm` jumps from ~5.9–6.0 (pre-attack) to 8.48–9.46 (post-attack) — `z_m(client_002, r=20) = |8.4816 − 6.0034| / 0.2690 = 9.21`. First anomalous round = 20; confirmed at round 22 (rounds 20, 21, 22 all > 3); onset estimate = 20. **Onset localization error = 0. Detection delay = 2 rounds.** No honest client's `z_m` exceeds ~1.29 anywhere in the run (max observed: `client_004`, z≈1.29 at round 49) — **zero honest false positives.**

**`EXP_015`:** because the attack is norm-preserving by construction (`delta_attack_norm_ratio ≈ 1.0` exactly, verified in the trace), `client_002`'s `update_l2_norm` does **not** jump at round 20 (`z_m(r=20) = 0.03`). It instead drifts upward gradually over the run as the perturbed global-model trajectory diverges from the honest counterfactual, only crossing z=3 much later: an isolated single-round flicker at round 40 (`z=3.04`, immediately followed by round 41 dropping back to `z=2.73`), then a sustained run from round 43 onward. Confirmed persistent at round 45 (rounds 43, 44, 45); onset estimate = 43. **Onset localization error = 23 rounds. Detection delay = 25 rounds.** No honest client's `z_m` exceeds ~1.21 anywhere in this run — **zero honest false positives**, but detection is late and imprecise.

## 9. Detector C — Temporal-Only (`z_t`)

**`EXP_014`: NOT EVALUABLE** — `self_prev_round_cosine` does not exist in this run's trace file (§3).

**`EXP_015`:** `client_002`'s `self_prev_round_cosine` collapses from ~0.24–0.38 (pre-attack) to ≈0 (post-attack, |t| < 0.0007 for every round 20–49) — a direct consequence of the directional attack replacing the honest delta's direction with an independent, target-cosine-0 rotation, destroying round-to-round self-consistency. `z_t(client_002, r=20) = |−0.00025 − 0.3119| / 0.0385 = 8.12`, and stays ≈8 for every subsequent round. First anomalous round = 20; confirmed at round 22; onset estimate = 20. **Onset localization error = 0. Detection delay = 2 rounds** — matching Detector B's performance on `EXP_014` exactly, but on the attack family Detector B handles poorly.

**Honest-client false positives (the central finding of this stage):** `self_prev_round_cosine` has its own secular decline across training, shared by every client — the same phenomenon that defeats Detector A's raw peer-cosine baseline, but here affecting the *temporal* signal instead. Because `t`'s calibration window is pre-attack-only (rounds 1–19), it cannot see this decline continuing past round 20, and **three of the four honest clients cross z_t > 3 and confirm persistent detection later in the same run**:

| Honest client | First anomalous round | Confirmed at | Onset estimate |
|---|---|---|---|
| `client_004` | 41 | 43 | 41 |
| `client_003` | 42 (isolated; breaks at 44) → 45 | 47 | 45 |
| `client_001` | 46 | 48 | 46 |
| `client_000` | never crosses 3 (peaks at z≈2.98, round 49) | — | — |

`client_002`'s confirmation (round 22) still precedes every honest client's false alarm by a wide margin (21+ rounds), so **attribution (earliest-detected client) remains correct**. But by round 49, Detector C has 4 confirmed-detected clients total, only 1 of them actually malicious.

## 10. Detectors D, E, F, G — Combined Signals

Computed via `max(z_m, z_p, z_t)` over whichever components each detector uses. `z_p` (§ below, computed at representative rounds) never exceeds ≈1.8 for any client in either run — it never wins the max, never contributes to a detection or a false positive anywhere in this evidence. Consequently:

- **`D_magnitude_peer` (`m`+`p`)** is numerically identical to Detector B on both runs (`z_p` never dominates or adds sensitivity): `EXP_014` onset=20/confirm=22/0 FP; `EXP_015` onset=43/confirm=45/0 FP.
- **`E_magnitude_temporal` (`m`+`t`), `EXP_015` only** (NOT EVALUABLE on `EXP_014`, `t` absent): identical to Detector C — `z_t` reaches threshold far earlier than `z_m` ever does for `client_002` (round 20 vs round 40+), and `z_m` never exceeds `z_t` at the honest clients' false-alarm rounds either. Onset=20/confirm=22 for `client_002`; same 3 honest false positives as Detector C.
- **`F_peer_temporal` (`p`+`t`), `EXP_015` only**: identical to Detector C for the same reason (`z_p` inert).
- **`G_full_fedtrace` (`m`+`p`+`t`)**: on `EXP_015`, identical to Detector C/E/F (onset=20/confirm=22, same 3 honest false positives). On `EXP_014`, `t` is undefined for every row, so `G` reduces to exactly Detector D (= Detector B): onset=20/confirm=22/0 FP. **`G` is never observed, in this evidence, to outperform whichever single signal happens to dominate for a given run.**

## 11. Full Metrics Suite

Attribution correctness is against the single true malicious client (`client_002`); precision/recall/F1 treat "confirmed-detected by end of run" as a positive prediction per client.

| Detector | Run | Evaluable | Attribution correct | Onset error | Delay | Precision | Recall | F1 | Honest FP |
|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | EXP_014 | yes | No (never detects) | — | — | — (0 predictions) | 0.0 | — | 0 |
| A (baseline) | EXP_015 | yes | No (never detects) | — | — | — | 0.0 | — | 0 |
| B (magnitude) | EXP_014 | yes | Yes | 0 | 2 | 1.0 | 1.0 | 1.0 | 0 |
| B (magnitude) | EXP_015 | yes | Yes | 23 | 25 | 1.0 | 1.0 | 1.0 | 0 |
| C (temporal) | EXP_014 | **NOT EVALUABLE** | — | — | — | — | — | — | — |
| C (temporal) | EXP_015 | yes | Yes | 0 | 2 | 0.25 | 1.0 | 0.40 | 3 |
| D (mag+peer) | EXP_014 | yes | Yes | 0 | 2 | 1.0 | 1.0 | 1.0 | 0 |
| D (mag+peer) | EXP_015 | yes | Yes | 23 | 25 | 1.0 | 1.0 | 1.0 | 0 |
| E (mag+temporal) | EXP_014 | **NOT EVALUABLE** | — | — | — | — | — | — | — |
| E (mag+temporal) | EXP_015 | yes | Yes | 0 | 2 | 0.25 | 1.0 | 0.40 | 3 |
| F (peer+temporal) | EXP_014 | **NOT EVALUABLE** | — | — | — | — | — | — | — |
| F (peer+temporal) | EXP_015 | yes | Yes | 0 | 2 | 0.25 | 1.0 | 0.40 | 3 |
| G (full FedTrace) | EXP_014 | yes (degenerate, = D) | Yes | 0 | 2 | 1.0 | 1.0 | 1.0 | 0 |
| G (full FedTrace) | EXP_015 | yes (= C/E/F) | Yes | 0 | 2 | 0.25 | 1.0 | 0.40 | 3 |

## 12. Max-Combiner Diagnostics

Across every round of both runs where any detector's combined score exceeds 3 (client_002's confirmed run plus the three honest false alarms in `EXP_015`), the dominant (max-winning) component was:

- **`EXP_014`:** `z_m` — 100% of the time. `z_p` never won a single round; `z_t` was unavailable.
- **`EXP_015`:** `z_t` — effectively 100% of the time, for both the true detection (`client_002`, every round 20–49) and all three honest false alarms. `z_m` approached but only barely crossed threshold in isolation (round 40) and never won a round where `z_t` was also being evaluated (`z_t` was already far above 3 by then). `z_p` never won a single round in either run.

**`z_p` (peer-cosine, cross-sectionally calibrated) never contributed to a single successful detection or a single false alarm anywhere in this Stage A evidence**, despite being specifically redesigned (relative to Detector A's per-client longitudinal calibration) to fix the shared-secular-trend problem. At the representative rounds computed (both runs, rounds 20, 21, 22/23, 25, 30, 40, 45/49), `client_002`'s `z_p` peaked at ≈1.65–1.77 in both runs — well short of 3 — because the cross-sectional spread across only 5 clients per round is itself large enough (std ≈ 0.0035–0.015) to absorb the malicious client's deviation. The cross-sectional fix did solve the secular-trend confound (representative honest-client z_p values are equally small, not systematically inflated), but the resulting signal is simply too weak, at n=5 clients, to ever cross the fixed threshold=3 bar for either attack family tested.

## 13. Coverage Advantage vs. Detection-Performance Advantage

This is the central scientific question this stage was designed to answer, and the evidence gives an unambiguous answer: **in every case observed, `G_full_fedtrace` performs identically to whichever single signal already dominates for that run — never better, never worse.** On `EXP_014`, `G` = Detector B exactly (peer and temporal contribute nothing). On `EXP_015`, `G` = Detector C exactly (magnitude and peer contribute nothing, including neither mitigating nor adding to `z_t`'s false-positive problem). There is no round, in either run, where combining two or three signals produced a different (earlier, more accurate, or lower-FP) outcome than the best available single signal alone would have.

What the max-combiner *does* provide, within this evidence, is **coverage**: a single fixed detector configuration that did not need to be told in advance which signal would matter — `z_m` happened to work for `EXP_014`, `z_t` happened to work for `EXP_015`, and `max()` picked up whichever was informative without per-attack retuning. This is a real, useful property for a detector meant to generalize across unknown future attack families. But it is a claim about *not needing to guess which signal to deploy*, not a claim that *combining signals outperforms deploying the single best one*. On the current evidence, a practitioner who had somehow known in advance to deploy magnitude-only for `EXP_014` and temporal-only for `EXP_015` would have gotten exactly the same detection performance as full FedTrace — and, on `EXP_015`, exactly the same false-positive problem too. **The combination has not yet been shown to buy anything beyond what its best individual component already provides; it also does not average away or otherwise mitigate that component's weakness.**

## 14. Discovered Weakness: Temporal-Signal False Positives on Honest Clients

This is reported as required — not tuned away. The `t` calibration rule (pre-attack-only per-client longitudinal window, `T_ref=20`) shares a structural blind spot with Detector A's already-known failure mode: both `peer_mean_cosine_similarity` (Detector A) and `self_prev_round_cosine` (`z_t`) exhibit a **secular decline across training that is common to honest and malicious clients alike**, and a calibration window drawn only from early training cannot anticipate how far that decline continues by round 49. Detector A's fix (Detector A itself failing outright, never flagging anyone) accidentally avoided false positives by being useless; `z_t`'s longitudinal-but-pre-attack-only calibration is sensitive enough to catch the true attack immediately, but that same sensitivity, applied to a signal with its own secular trend, produces exactly the false-alarm behavior that `p`'s *cross-sectional* (not longitudinal) redesign was specifically built to avoid for the peer-cosine signal. **The temporal signal was given the peer-cosine signal's old (longitudinal) calibration strategy, not its new (cross-sectional) one, and inherited the old strategy's known weakness as a result.** This is a property of the frozen method as specified, present in `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` itself, not an implementation bug in `src/analysis/fedtrace_detector.py`.

## 15. Honest-Client False Positive Analysis

Zero honest-client false positives for Detector A (trivially, since it never flags anyone), Detector B, and Detector D on both runs — magnitude-based detection stayed comfortably within ±1.3σ of each honest client's own calibrated baseline throughout both 50-round runs. Detector C (and, by inheritance, E/F/G) produced exactly 3 honest-client false positives on `EXP_015` (`client_001`, `client_003`, `client_004`; only `client_000` narrowly avoided crossing threshold, peaking at z≈2.98). All 3 false alarms occurred well after `client_002`'s true confirmation (round 22), so single-prediction attribution accuracy was unaffected in this n=1 instance — but a deployment that continues monitoring past the first detection (as any real system would) would end this run believing 4 of 5 clients were malicious, using the full-signal detector.

## 16. Non-Overclaiming and Limitations

- **n=1 per attack family.** Every number above comes from a single seed (42) run of each attack. No claim here should be read as "FedTrace will behave this way in general" — only "FedTrace, applied exactly as frozen, behaved this way on these two specific runs." Stage B (seed replication) is the mechanism for testing whether these findings hold, not this document.
- **`z_p` sampling.** The `p` cross-sectional calibration was computed and verified by hand at 8 representative rounds per run (20, 21, 22/23, 25, 30, 40, 45, 49) rather than exhaustively at all 30 post-attack rounds — a deliberate scope tradeoff given this task's manual-arithmetic, no-code-execution constraint. The sampled values are consistent and non-borderline (peaking at ≈1.65–1.77 against a threshold of 3, roughly 1.2–1.4σ of headroom), giving reasonable confidence in the qualitative conclusion (`z_p` is inert in both runs), but this is not a literal 100-round-observation audit. `src/analysis/fedtrace_detector.py`'s `compute_cross_sectional_calibration` is written to compute this exhaustively and correctly if run in a future stage.
- **`t`'s unavailability on `EXP_014` is a logging artifact, not a finding about Gaussian-noise attacks.** This document does not know, and does not claim to know, whether the temporal signal would also have false-alarm or true-detect properties against a magnitude-based attack — that data does not exist yet.
- **The false-negative-toward-zero risk of `z_p`.** A signal that never crosses threshold in either tested run cannot yet be distinguished, from this evidence alone, between "genuinely uninformative at n=5 clients" and "would be informative at a larger client population" — this dataset has exactly 5 clients per round, always.

## 17. No Post-Hoc Tuning Confirmation

The threshold (3), persistence value (P=3), `T_ref` (20), calibration windowing rules, minimum-observation count (2), and `max` combiner were fixed in `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` before this evaluation began and are unchanged here. No alternative threshold, no alternative P, no per-signal weighting, no new derived feature, and no different calibration window was tried or reported — the weaknesses in §12–15 are reported as found, not resolved or hidden by adjusting the method. No shell, PowerShell, or Bash command was executed at any point during this task; every number in this document was produced by manual arithmetic applied to the CSV rows read via the `Read`/`Grep` tools, following exactly the formulas implemented (but not executed) in `src/analysis/fedtrace_detector.py`.

## 18. Decision Gate

**Decision: (B) — Methodological weakness found; must be resolved before proceeding to Stage B seed validation.**

Justification, strictly from the evidence above: attribution correctness held in both n=1 runs (§11), and the coverage property in §13 is real and valuable. But §9/§14/§15 establish a reproducible, structural false-positive problem in the temporal signal's calibration rule — not a fluke of one run, but the same secular-trend failure mode already known from Detector A, now reintroduced through `z_t`'s calibration choice. Three of four honest clients would be confirmed-flagged as malicious by the end of a single 50-round run under the full FedTrace detector. Advancing directly to Stage B (replicating this exact method across seeds 123/2026) would faithfully reproduce this same false-positive behavior at every seed, since it is a property of the calibration rule, not of seed-42-specific noise — that would not be new information, and would spend the seed-replication budget confirming a weakness already established here rather than testing the method's core discriminative claims under noise. **This is not a recommendation to invent a fix now** (out of scope for this document, and post-hoc tuning is explicitly prohibited, §17) — it is a recommendation that whoever owns the method plan decide how `t`'s calibration window should account for its own secular trend (e.g., an analogous cross-sectional or trend-adjusted calibration, mirroring what already worked for `p`) before the frozen method is spent on a 3-seed validation run that would only reconfirm this same finding.
