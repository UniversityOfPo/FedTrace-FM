# FedTrace-FM — Master Research Status

**Purpose of this document:** the authoritative, current-state project handoff record. A new researcher or a new AI conversation with zero prior context should be able to read this file plus the specific artifacts it cites and reconstruct the complete research state: what was built, why, every major experiment, what succeeded, what failed and what each failure taught, how FedTrace evolved v1→v2→v3, what evidence supports which claim, what remains untested, why the essential experimental phase is now considered complete, and what should happen next.

**Status of this document:** produced by directly reading repository artifacts (`summary.json`/`client_round_trace.csv` files, configs, every `analysis/*.md` and `research_plans/*.md` file, `Research_Database/*`, and the root-level project documents) — not reconstructed from memory. No code, config, test, result, checkpoint, README, Research Database, or master-workbook file was created or modified in producing it. No experiment was run. No FedTrace formula, threshold, persistence rule, or combiner was changed. Companion file: `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` (one row per experiment, `EXP_001`–`EXP_027`).

**⚠ Important documentation-landscape note, established directly in this task (see §11):** this repository also contains an older set of root-level "master" documents — `FEDTRACE_FM_MASTER.md`, `PROJECT_CONTEXT.md`, `PROJECT_ROADMAP.md`, `RESEARCH_DIARY.md` — which describe themselves as the project's authoritative memory. **They are severely stale**, frozen at `EXP_004` (dated 2026-08-02 to 2026-08-04), describing Phase 4 (attack simulation) and Phase 5 (FedTrace attribution) as "Not Started." Twenty-three additional experiments and the entire FedTrace v1–v3 + Multi-Krum + generalization-gate research program (this document's actual subject) postdate them and are **not reflected there at all**. **This document — `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` — is the current authoritative status record; the root-level documents should be treated as a historical record of Phases 0–3 only until they are explicitly updated (a task not performed here, per this task's restriction against modifying files outside `analysis/`).**

---

## 1. What We Built and Why (Narrative Reconstruction)

**Research objective** (`FEDTRACE_FM_MASTER.md`, `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`): investigate whether malicious, corrupted, or otherwise anomalous client contributions in a federated learning (FL) system leave detectable, attributable signatures — and specifically, whether such attribution and onset-localization remains possible under **strong, realistic client heterogeneity** (Dirichlet non-IID partitioning), where honest clients already disagree with one another enough to defeat naive anomaly detection.

**The arc, phase by phase, as it actually unfolded:**

1. **Centralized + IID federated infrastructure (`EXP_001`–`EXP_005`).** A leakage-free centralized ResNet-18/CIFAR-100 baseline, then a full federated stack (`FederatedClient`/`FederatedServer`/pluggable `AggregationStrategy`/`FedAvg`/`FederatedRunner`), validated first under IID partitioning across a round-budget sweep (`EXP_002`→`EXP_004`: 10→20→50 rounds), then a cosine-scheduler variant (`EXP_005`). **No attack, attribution, or robustness code existed yet at this stage.**
2. **Heterogeneity characterization (`EXP_006`–`EXP_009`).** Non-IID Dirichlet partitioning introduced at `α=0.5` (`EXP_006`) then the project's chosen "strong" heterogeneity level `α=0.1` (`EXP_007`, single seed), then replicated at two further seeds (`EXP_008`/seed123, `EXP_009`/seed2026). **This 3-seed set is the quantitative foundation for everything that follows**: it established a **2.85-percentage-point honest seed-to-seed noise floor** and up to **~31% client sample-count coefficient of variation** at `α=0.1` — the concrete, measured reason a naive per-round anomaly detector cannot simply flag "the client that looks different."
3. **FedProx side-branch (`EXP_010`–`EXP_013`).** FedProx (`μ=0.01`) tested as an optimization-side response to the `α=0.1` degradation, across the same three paired seeds. Result: mixed/near-neutral (mean Top-1 Δ −0.11pp, no metric improving in all three seeds) — **FedProx was not adopted**; FedAvg was retained as the sole aggregation strategy for every subsequent phase, deliberately decoupling the optimization question from the traceability question.
4. **Threat model, attack design, naive-baseline failure (`PHASE5` plan, `EXP_014`/`EXP_015`, development seed 42).** A single-attacker, local-knowledge-only, honest-server threat model; a first attack family (additive Gaussian update noise, `EXP_014`) and a second, deliberately harder family (norm-preserving directional rotation, `EXP_015`) designed specifically to defeat a magnitude-based shortcut; a naive peer-cosine 3-sigma baseline, pre-registered to fail and confirmed to fail on both attacks.
5. **FedTrace method design, three iterations, each diagnosed from real failure (`EXP_014`/`EXP_015` reused as development data throughout — see §2 for full detail).** v1 → (Stage A retrospective finds v1's flaw) → v2 → (v2 retrospective finds a *different*, worse flaw) → v3 → (v3 retrospective finds zero flaws on the same data) → **frozen**.
6. **Independent validation, Stage B (`EXP_016`–`EXP_019`, unseen seeds 123/2026).** Frozen v3 applied, unmodified, to two attack families × two unseen seeds. **4/4 PASS** on every pre-registered criterion.
7. **External baseline, Multi-Krum branch (`EXP_020`–`EXP_025`).** Because Stage B's own synthesis identified "no external comparison point" as the single biggest remaining publication-risk gap, Krum/Multi-Krum was selected (FoolsGold explicitly rejected on threat-model-mismatch grounds) and evaluated across a full 2×2 (attack family × seed) factorial plus clean controls — six runs total.
8. **Generalization gate (`EXP_026`/`EXP_027`).** The `ICML_ICLR_EVIDENCE_GAP_AUDIT.md` (see §11) identified one essential remaining evidence gap: every attack run through `EXP_025` used the identical attacker (`client_002`) at the identical onset (round 20). `EXP_026` varied identity alone; `EXP_027` varied onset alone. **Both passed.** Gate decision: **A** (see §6).
9. **This document.** Freezes and consolidates the above before manuscript construction begins.

## 2. Method Development History — v1 → v2 → v3

**Governing discipline, stated once here and applicable throughout this section:** `EXP_014`/`EXP_015` (seed 42) are **development data only**. They were used to design and diagnose every one of v1/v2/v3. **No result from them is ever cited as independent validation evidence anywhere in this project** — that role belongs exclusively to `EXP_016`–`EXP_019` (§ below), whose formulas were frozen before those traces existed.

### FedTrace v1 (`research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`)

- **Problem it attempted to solve:** the naive peer-cosine baseline fails under honest Non-IID heterogeneity; a multi-signal, honestly-calibrated detector was needed.
- **Signals:** magnitude `m(k,r)=‖Δ_{k,r}‖₂`; peer-direction `p(k,r)=peer_mean_cosine_similarity`; temporal self-consistency `t(k,r)=cos(Δ_{k,r},Δ_{k,r-1})`.
- **Calibration:** `m`/`t` — per-client longitudinal (mean/stdev over `round_id<T_ref=20`, static, pre-attack-only). `p` — per-round cross-sectional (mean/stdev across that round's clients).
- **Persistence:** `P=3` consecutive rounds with `z>3`.
- **Combiner:** `FedTraceScore=max(z_m,z_p,z_t)`.
- **Development experiments:** `EXP_014` (Gaussian) — `z_m` near-perfect; `t` field not present in that trace schema (added later), so `t`-dependent detectors not evaluable. `EXP_015` (directional) — `z_t` (v1) correctly detects the attacker (onset 0, delay 2) but produces **3 honest false positives** (`client_001`/`003`/`004`, confirmed late — rounds 43/45/47).
- **Observed failure mode:** `t(k,r)` is not stationary — every client's value declines secularly across training — so a static pre-attack-only calibration window cannot anticipate the decline continuing past it (`analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`).
- **Why the next version was needed:** attribution survived only because the true attacker happened to confirm first; a less lucky ordering could have broken it. Decision: **B**, resolve before Stage B.
- **Superseded:** yes, entirely, by v3.
- **Lesson surviving into the final paper:** the "coverage" design pattern (multiple complementary signals, `max()` combiner) and the diagnosis method itself (exhaustive per-round/per-client inspection of a real failure, not guessing) both survive; the specific `z_t` calibration does not.

### FedTrace v2 (`research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`)

- **Problem it attempted to solve:** v1's non-stationary `z_t` calibration.
- **Fix:** same-round, cross-sectional, robust (median/MAD) reference: `center_t(r)=median_j t(j,r)`, `scale_t(r)=1.4826·MAD_t(r)`, one-sided `z_t_rel(k,r)=max(0,[center_t(r)-t(k,r)]/scale_t(r))`. No leave-one-out. `z_p` demoted to ablation-only (never once contributed a detection or false alarm at any development or validation stage — this holds through v3 as well).
- **Persistence/combiner:** unchanged (`P=3`; `FedTraceV2Score=max(z_m,z_t_rel)`).
- **Development experiments:** same `EXP_014`/`EXP_015`.
- **Observed failure mode:** v2 *did* eliminate v1's three late false positives, but introduced a **new, worse, attribution-breaking** one: honest `client_000` confirmed persistent at **round 11** — entirely *before* the attack starts at round 20 — because the same-round MAD scale estimate is unstable at `n=5` clients (on specific rounds, the 4 non-outlier clients happened to cluster tightly by chance, inflating an ordinary gap into `z>3`) (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`).
- **Why the next version was needed:** this flips attribution to the wrong client (`client_000` confirms before `client_002`). Decision: **B** again — a regression on the metric that matters most, despite improving the raw false-positive *count*.
- **Superseded:** yes, entirely, by v3.
- **Lesson surviving into the final paper:** the same-round median *reference* (which correctly cancels shared drift) is worth keeping; cross-sectional *scale* estimation at `n=5` is fundamentally unstable and must not be used for the primary detection scale.

### FedTrace v3 (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`) — **FROZEN**

- **Problem it attempted to solve:** keep v2's fix for non-stationarity while replacing only the part that caused v2's new failure (a categorically different problem: small-sample scale instability, not non-stationarity).
- **Fix:** keep v2's same-round median reference `c(r)=median_j t(j,r)` (no scale computed from it at all); compute the residual `d(k,r)=c(r)-t(k,r)`; calibrate **its** scale **historically/longitudinally per-client** (the same class of calibration already proven stable for `z_m`): `center_k=mean(D_k)`, `scale_k=stdev(D_k)` (sample stdev, `n−1`) over `D_k={d(k,r):r∈[1,T_ref)}`. `z_t_res(k,r)=max(0,[d(k,r)-center_k]/scale_k)`.
- **Persistence/combiner:** unchanged (`P=3`; `FedTraceV3Score=max(z_m,z_t_res)`).
- **Development experiments:** same `EXP_014`/`EXP_015`, re-checked **exhaustively** (all 49 rounds × 5 clients, not sampled).
- **Observed failure mode:** **none.** `client_000`'s previously-breaking rounds dropped from `z≈3.9/12.4/4.5` to `z≈0.37/0.28/0.52`; `client_002` remained overwhelmingly detected (`z` in the 20s–30s); zero honest persistent false positives anywhere, at any round, for any client (`analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`).
- **Decision:** **A** — passes retrospective gate, freeze for Stage B.
- **Superseded:** no — this is the version evaluated in every subsequent experiment (`EXP_016`–`EXP_019`, `EXP_026`, `EXP_027`) with **zero further modification**.

### Design vs. Validation Evidence — Explicit Separation

| | Development/design evidence | Independent validation evidence |
|---|---|---|
| Runs | `EXP_014`, `EXP_015` (seed 42) | `EXP_016`–`EXP_019` (seeds 123, 2026) |
| Role | Diagnose failures, motivate fixes | Test the *frozen* formula, unmodified |
| Permitted to change the formula? | Yes (this is what design means) | **No — never, under any outcome** |
| Can be cited as "FedTrace works"? | No — this would be circular | Yes — this is the actual evidence |

## 3. Final FedTrace v3 Specification (Exactly as Implemented)

Source of truth: `src/analysis/fedtrace_detector.py` (read in full; not modified in this or any prior task in this series). Notation restated exactly as coded — no equation here is invented beyond what the implementation computes.

Let client `k`'s round-`r` update be `Δ_{k,r} = w_{k,r}^{submitted} − w_r^{global}`.

```
m(k,r)  = ‖Δ_{k,r}‖₂                              ("update_l2_norm")
t(k,r)  = cos(Δ_{k,r}, Δ_{k,r−1})                 ("self_prev_round_cosine")

--- magnitude score (unchanged since v1) ---
m_cal_k = (mean, stdev) of {m(k,r): r<T_ref}, T_ref=20   [sample stdev, n−1]
z_m(k,r) = |m(k,r) − mean(m_cal_k)| / stdev(m_cal_k)

--- temporal residual score (v3) ---
c(r)      = median_j t(j,r)                        (same round, every client incl. itself; no leave-one-out)
d(k,r)    = c(r) − t(k,r)
D_k       = {d(k,r): 1 ≤ r < T_ref}                (round 0 excluded — t(k,0) undefined)
center_k  = mean(D_k);  scale_k = stdev(D_k)        [sample stdev, n−1]
z_t_res(k,r) = max(0, [d(k,r) − center_k] / scale_k)

--- combiner ---
FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))

--- decision rule ---
threshold  : score > 3.0                            (DEFAULT_THRESHOLD)
persistence: P = 3 consecutive anomalous rounds      (DEFAULT_PERSISTENCE)
confirmation round: the round at which the P-th consecutive anomalous round occurs
onset estimate    : the FIRST round of that P-round run (not the confirmation round)
attribution       : earliest-confirmed client; ties -> higher mean score during its run, then client_id
```

Both `m_cal_k` and `D_k`/`center_k`/`scale_k` are excluded (`None`, never fabricated as zero) if fewer than `MIN_CALIBRATION_OBSERVATIONS=2` values are available, or if the resulting scale is exactly zero. The peer signal `p(k,r)=peer_mean_cosine_similarity` is computed (v1's own cross-sectional mean/stdev calibration) but is **not part of the v3 primary score** — retained for ablation/monitoring only, per the empirical finding (every development and validation run) that it never once contributed a detection or a false alarm.

**Disagreement check between implementation and older analysis documents:** none found. Every formula restated above matches, term-for-term, both `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16 and `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §3. **One documentation-completeness discrepancy was found (not a formula disagreement) and is reported here rather than silently resolved:** `analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §10 states "No honest client's `z_m` or `z_t_res` ever exceeds 3 at any round, pre- or post-attack." Direct re-computation from `EXP_018`'s own trace (performed during the `EXP_026` generalization task) shows every one of its five clients' `z_m` briefly exceeds 3 at round 0 — a real, harmless, non-persistent, fully explained artifact (§7 below), not a threshold/formula issue. `EXP_018`'s own PASS verdict and its "0 honest **persistent** false positives" finding remain correct; only that one descriptive sentence is imprecise. `EXP_018` was not, and is not here, modified.

## 4. Claim ↔ Evidence Matrix

| # | Claim | Supporting experiments | Supporting analysis | Strength | Allowed wording | Overclaim to avoid | Known limitation |
|---|---|---|---|---|---|---|---|
| C1 | FedTrace provides client-level forensic attribution | `EXP_016`–`EXP_019`, `EXP_026`, `EXP_027` | `FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`, `EXP_026`/`EXP_027` analyses | **STRONG** (within tested scope) | "FedTrace correctly attributed the malicious client in 6/6 tested attack runs across two attack families, two seeds, one alternate identity, and one alternate onset" | "FedTrace attributes malicious clients in general FL settings" | One dataset/model/client-count/heterogeneity level; 2 of 5 possible attacker identities tested |
| C2 | FedTrace localizes attack onset | Same as C1 | Same as C1 | **STRONG** | "Onset error 0, delay 2 (the theoretical minimum under P=3) in all 6 tested attack runs" | "Onset localization is exact for any onset" | Both tested onsets (20, 30) are after the fixed `T_ref=20` calibration window closes; an onset before `T_ref` is untested and structurally harder (§6) |
| C3 | Longitudinal/per-client calibration is important under strong Non-IID heterogeneity | `EXP_009` (heterogeneity quantification); `EXP_014`/`EXP_015` (v1/v2 failure diagnosis); `EXP_016`–`EXP_019` (v3 success) | `EXP_009_Three_Seed_Baseline_Analysis.md`; `FEDTRACE_STAGE_A/V2/V3_RETROSPECTIVE_ANALYSIS.md` | **STRONG** | "Two successive cross-sectional/static calibration designs produced honest false positives (one attribution-breaking) at this heterogeneity level; per-client longitudinal calibration resolved both" | "Longitudinal calibration is necessary under any heterogeneity" | Demonstrated at exactly one heterogeneity level (`α=0.1`); not tested at other `α` values |
| C4 | Magnitude and temporal-residual signals provide complementary coverage | `EXP_016`–`EXP_019`, `EXP_026`, `EXP_027` | Stage B synthesis §6–§8; `EXP_026`/`EXP_027` §4 | **STRONG** | "`z_m` was decisive for the Gaussian family, `z_t_res` for the directional family, across every tested run, automatically, with no attack-type-specific branch" | "The two signals are complementary for all attack types" | Only two attack families tested |
| C5 | The final combiner is best characterized as coverage rather than synergistic fusion | All of the above | Stage B synthesis §8, §12; every individual `EXP_01X`/`EXP_026`/`EXP_027` analysis §"signal dominance" | **STRONG** | "The combined score never produced a better outcome than its best individual component in any of the 8 attack runs evaluated to date" | "Combining signals improves detection beyond the best single signal" (directly contradicted by the evidence) | This is a negative/honesty finding — must be stated as a constraint on the claim, not smoothed over |
| C6 | Robust aggregation and forensic attribution are different problems | `EXP_018`/`EXP_019` vs. `EXP_024`/`EXP_025` (paired) | `ROBUST_AGGREGATION_VS_TRACEABILITY_SYNTHESIS.md`, `MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` | **STRONG** | "Byzantine-robust aggregation and forensic attribution answer different questions; repurposing per-round exclusion as an attribution mechanism can fail, as demonstrated here" | "FedTrace beats Multi-Krum as a detector" (Multi-Krum was never designed to be one — see §5) | One external baseline; `f=1` known in advance for Multi-Krum but not for FedTrace (an advantage given to Multi-Krum, not taken from it) |
| C7 | Multi-Krum exclusion is strongly attack-geometry dependent | `EXP_022`/`EXP_023` (Gaussian) vs. `EXP_024`/`EXP_025` (directional) | `GAUSSIAN_VS_DIRECTIONAL_MULTIKRUM_SEED123_SYNTHESIS.md`, `MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md` §11 | **STRONG** | "100% (30/30) post-attack rejection for the Gaussian family vs. 0% (0/30) for the norm-preserving directional family, reproduced identically at both seeds — a fully-crossed 2×2 result" | "Robust aggregation is blind to all direction-only attacks" (only one specific rotation, `target_cosine=0.0`, was tested) | One `f` value, one client count, one rotation severity |
| C8 | FedTrace remains effective for both tested Gaussian and directional attacks | `EXP_014`/`016`/`017` (Gaussian), `EXP_015`/`018`/`019` (directional) | Stage B synthesis | **STRONG** (validation runs); development runs excluded from this claim | "4/4 independent validation PASS across both families" | Extending to unseen attack families (label-flip, backdoor, sign-flip) without new evidence | Only two attack families exist in this project |
| C9 | The result replicates across the tested seeds | `EXP_016` vs `017`; `EXP_018` vs `019`; `EXP_020` vs `021`; `EXP_022` vs `023`; `EXP_024` vs `025` | All Stage B and Multi-Krum cross-seed synthesis documents | **MODERATE** | "Replicated at 2 independent seeds per condition" | "Statistically significant across seeds" (n=2 is not a significance sample — stated explicitly and repeatedly throughout this project's own documents) | n=2 seeds is descriptive replication, not a population estimate |
| C10 | FedTrace survives the targeted attacker-identity generalization check | `EXP_026` | `EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md` | **MODERATE** (one new identity) | "Frozen v3, unmodified, correctly attributed a second, deliberately-chosen (largest-partition) identity with every pre-registered criterion matching `EXP_018` exactly" | "FedTrace is identity-invariant" or "works for any attacker" | Only 2 of 5 clients ever tested as attacker; one seed, one attack family, no joint identity+onset test |
| C11 | FedTrace survives the targeted onset-round generalization check | `EXP_027` | `EXP_027_ONSET_GENERALIZATION_ANALYSIS.md` | **MODERATE** (one new onset) | "Confirmation round tracked the shifted true onset exactly (onset+2), demonstrating the decision is onset-relative, not tied to round 20 specifically" | "FedTrace works for any onset" | Only 2 onsets ever tested, both after `T_ref=20`; an onset before `T_ref` is a structurally different, untested case |

## 5. Multi-Krum External Baseline — Consolidated

**Design:** `f=1` (matching this project's single-attacker threat model — a known-in-advance advantage given to Multi-Krum), `m=4` (neighbors averaged), unweighted 1/m averaging, seeds 123/2026, both attack families, plus clean controls. Six runs (`EXP_020`–`EXP_025`), each code-verified before any GPU run (`MULTIKRUM_IMPLEMENTATION_VERIFICATION.md`).

| Condition | Seed 123 | Seed 2026 | True-attacker rejection rate | Regime switch | Rejection-proxy attribution | Largest-partition association |
|---|---|---|---|---|---|---|
| Clean | `EXP_020` | `EXP_021` | N/A (no attacker) | N/A | N/A | Persistently rejects the largest-partition honest client (95–98%), both seeds |
| Gaussian | `EXP_022` | `EXP_023` | **100% (30/30)**, both seeds | **Instantaneous**, both seeds | Wrong first (honest FP), attacker *eventually* confirmed (round 22), both seeds | Same honest client misattributed first, both seeds |
| Norm-preserving directional | `EXP_024` | `EXP_025` | **0% (0/30)**, both seeds | **None**, either seed | Wrong (honest FP), true attacker **never** confirmed, either seed | Same honest client misattributed, never corrected |

**Attack-geometry dichotomy:** a fully-crossed 2×2 (seed × attack family) design with zero exceptions — Gaussian always 100%/instant, directional always 0%/never. Mechanistically explained (`ROBUST_AGGREGATION_VS_TRACEABILITY_SYNTHESIS.md`): Krum's raw squared-Euclidean-distance score decomposes as `‖Δ_i‖²+‖Δ_j‖²−2⟨Δ_i,Δ_j⟩`; a norm-preserving attack leaves the magnitude terms roughly unchanged, so exclusion depends on whether the rotation-induced distance increase exceeds the heterogeneity-driven distance gap already present among honest clients — which, at `α=0.1`, it does not.

**Utility effects:** clean Multi-Krum costs ≈7pp Top-1 vs. clean FedAvg. Under Gaussian attack, attacked Multi-Krum > clean Multi-Krum but < attacked FedAvg (both seeds). Under directional attack, attacked Multi-Krum is **substantially worse than both** clean Multi-Krum and attacked FedAvg (−8.74pp/−10.06pp Top-1 vs. attacked FedAvg, both seeds) — traced to Multi-Krum's unweighted 1/m≈25% averaging giving the undetected directional attacker *more* relative influence than FedAvg's sample-weighted ≈16–20% would.

**Comparison with FedTrace (same seeds, same directional attack, paired via the shared honest baseline):** FedTrace v3 (via `EXP_018`/`EXP_019`) — correct attribution, onset error 0, delay 2, both seeds. Multi-Krum rejection proxy (`EXP_024`/`EXP_025`) — wrong client, never corrects, both seeds.

**Required wording discipline (verbatim, to prevent overclaim):**
- **Never:** "FedTrace beats Multi-Krum as a detector."
- **Always:** "Byzantine-robust aggregation and forensic attribution answer different questions. Multi-Krum's per-round exclusion decision is a memoryless function of that round's own cross-section — it cannot distinguish a chronically-peripheral honest client from a newly-anomalous one, and its exclusion pattern is attack-geometry dependent rather than tracking any consistent notion of misbehavior. Repurposing that exclusion signal as an attribution mechanism, with no onset foreknowledge, fails for the directional attack family tested here."
- **Known-`f` asymmetry must be stated every time this comparison is made:** Multi-Krum requires an assumed bound on attacker count as an input; FedTrace does not. This works *against* the paper's own thesis (Multi-Krum is given an advantage FedTrace lacks), which makes Multi-Krum's observed failures more, not less, notable.

## 6. Generalization Gate — Consolidated

| Axis | Run | Change relative to anchor (`EXP_018`) | Result |
|---|---|---|---|
| Attacker identity | `EXP_026` | `client_002` → `client_004` (largest partition, seed 123), onset held at 20 | **PASS** — correct attribution, onset error 0, delay 2, 0 honest FP, `z_t_res` dominant 30/30 (magnitude-only never confirmed at all — a *stronger* demonstration than `EXP_018`) |
| Attack onset | `EXP_027` | round 20 → round 30, identity held at `client_002` | **PASS** — correct attribution, onset error 0, delay 2, 0 honest FP, `z_t_res` dominant 20/20; confirmation tracked the shifted onset exactly (`onset+2`) |

**Why these narrow, but do not close, the evidence gap — stated conservatively, per this project's own standing discipline:**
- One additional attacker identity tested (of 5 possible); one additional onset tested (2 total, both after `T_ref=20`).
- Each axis tested **separately** (never jointly) — whether the two effects compose without interaction is untested.
- Both runs are **seed-123-only** and **directional-attack-only** — neither axis has been tested against the Gaussian family or at seed 2026.
- **No onset earlier than `T_ref=20` has ever been tested** — this is a structurally different, harder regime (the magnitude calibration window itself would contain already-attacked rows), explicitly flagged as untested in both `EXP_027`'s own analysis and here.
- No second seed exists for either targeted generalization test.

**Why the final gate nevertheless became A:** the ICML/ICLR audit named this gap narrowly — *identity/onset generalization*, to be closed by a small, targeted, one-factor-at-a-time package, not a broad sweep (its own Part 6 explicitly weighed against over-recommending experiments). Both halves of that named package are complete and both passed every pre-registered criterion with zero retuning. No methodological issue was found. The remaining items above (third identity, before-`T_ref` onset, second seed, joint identity+onset, Gaussian-family generalization) are real but are **HIGH-VALUE-IF-BUDGET-ALLOWS**, not blocking — no comparably total confound (i.e., affecting *every* prior run) remains. Full reasoning: `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`.

## 7. Round-0 Self-Inclusive Calibration Artifact

**Exact mechanism, determined from the implementation, not assumed:** `compute_longitudinal_calibration`'s `z_m` reference window is `round_id < T_ref=20` — a **self-inclusive** 20-observation sample that includes round 0 itself. At the seed-123 partition (and, by the same construction, presumably at other seeds/partitions — not separately re-verified for every run in this document), **round 0 is the largest `update_l2_norm` value within every client's own 20-round window** — a structural consequence of round 0 being each client's very first local-training step from the shared random initialization. A self-inclusive z-score for the sample maximum of an `n=20` window is bounded by `(n−1)/√n≈4.25`; the observed values (`3.19`–`3.93` across `EXP_026`/`EXP_027`'s clients) sit within that bound.

**Direct answers, from the data:**
- **Can isolated `z_m>3` occur at round 0?** Yes — reproduced identically for every client, honest or not, in `EXP_018`, `EXP_026`, and `EXP_027` (bit-identical values across all three, since all three share bit-identical honest-phase data through at least round 19).
- **Does it persist?** No — confined to round 0 alone (occasionally round 1 for `z_t_res`, e.g. `client_002`'s own `z_t_res=3.29` at round 1 in multiple runs); never two consecutive anomalous rounds from this cause; `P=3` is never satisfied by it.
- **Is any client ever confirmed because of it?** No — checked exhaustively in every run this artifact was investigated in (`EXP_026` §5, `EXP_027` §6); `confirmation_round=None` for every honest client in every run.
- **Does it affect attribution?** No.
- **Does it affect onset localization?** No — confined to rounds 0–1, well separated (18–29 rounds) from every tested true onset (20, 30).

**Recommended paper wording, explicit:**
- **Prefer:** "zero honest persistent false positives" (this is what the evidence actually, unconditionally supports across every run checked).
- **Do not use:** "no honest client ever exceeds the threshold" — this specific phrasing appears in `analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §10 and does **not** hold under direct re-verification (§3 above). Any similar phrasing in a future manuscript must be checked against this finding before use.

## 8. Paper Evidence Inventory

| Category | Status | Notes |
|---|---|---|
| Main results (Stage B 4/4 PASS) | **READY** | `EXP_016`–`EXP_019` + synthesis; numbers verified in this task |
| Ablations (magnitude/peer/temporal-only, v1/v2/v3 comparison) | **READY** | Present for every development and validation run |
| Controls (clean honest baselines, paired) | **READY** | `EXP_001`, `EXP_002`–`EXP_009` |
| Negative results (v1/v2 failures; combiner-vs-best-component) | **READY** | This is unusually strong, rare material — an honest failure narrative |
| External baseline (Multi-Krum) | **READY** | Six runs, fully crossed, code-verified |
| Cross-seed evidence | **READY** (n=2, descriptive) | Every attack condition has a paired seed |
| Attack-family evidence | **READY** | Two structurally distinct families, deliberately chosen |
| Generalization evidence | **READY** (narrow scope) | `EXP_026`/`EXP_027`, explicitly scoped per §6 |
| Utility metrics (Top-1/Top-5/loss, every run) | **READY** | Verified in `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` |
| Detector metrics (onset/delay/FP, every attack run) | **READY** | Verified in this task and prior analysis documents |
| Trace diagnostics (`delta_attack_norm_ratio`, `realized_cosine_to_honest`) | **READY** | Verified to `~1e-10` precision in every attack run checked |
| Existing plots (per-experiment matplotlib: validation curves, confusion matrices) | **READY, but not publication-formatted** | Exist per-run under each `results/federated/<name>/*.png`; not cross-run comparison figures |
| Cross-run comparison figures (onset trajectory, complementarity, Multi-Krum heatmap, etc.) | **NEEDS DERIVATION FROM EXISTING RESULTS** | All underlying data already exists in trace CSVs/summary.json; no new experiment needed — see `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 11 for the exact figure plan |
| Publication tables (Tables 1–5) | **NEEDS DERIVATION FROM EXISTING RESULTS** | Column specs already exist (`ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 12); populate from `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` and per-run analyses |
| Theoretical/mathematical formalization | **NEEDS DERIVATION FROM EXISTING RESULTS** | Propositions/lemmas identified but not yet written as manuscript prose (`ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 8) |
| Related-work literature review | **NEEDS DERIVATION FROM EXISTING RESULTS (partially)** | Citations for Multi-Krum/coordinate-median/trimmed-mean/FoolsGold already verified (`FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §5); full lit review not assembled |
| Third identity, before-`T_ref` onset, second seed for generalization axes, second external baseline, Gaussian-family generalization | **OPTIONAL NEW EXPERIMENT** | None labeled mandatory — no genuine new blocker was found by this consolidation (§9, §10) |

## 9. Top-Tier (ICML/ICLR-Style) Readiness — Percentages

| Dimension | % | Why |
|---|---|---|
| Method completeness | **95%** | v1→v2→v3 fully iterated, each failure diagnosed and fixed from real data, frozen; only remaining gap is written theory (not new method work) |
| Experimental completeness | **85%** | Stage A/B and Multi-Krum matrices fully done; generalization gate closed on its named scope; broader breadth (dataset/model/client-count/heterogeneity-level) remains untested |
| Detector validation | **90%** | 8 attack runs total (`EXP_014`/`015` dev + `016`–`019` validation + `026`/`027` generalization), zero honest persistent FP in any of them, exhaustively checked in every one re-audited in this task |
| External baseline evidence | **90%** for the one baseline pursued (Multi-Krum, unusually thorough); **~30%** if judged against "external baselines" plural (coordinate-median/trimmed-mean remain documented-but-unimplemented) |
| Generalization evidence | **55%** | Both named-essential axes closed with one passing test each; explicitly not broad (2 of 5 identities, 2 onsets both post-`T_ref`, no joint test, no second seed) |
| Reproducibility | **60%** | Excellent per-experiment reproducibility (config/seed/git-commit/summary.json for all 26 runs); severely stale cross-experiment master database (§11) drags this down; no third-party reproduction attempted |
| Theory/mathematical presentation | **20%** | Formulas are precisely implemented and documented (§3) but not yet written as formal propositions/lemmas in manuscript form; scoped, achievable theory content identified but not drafted (`ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 8) |
| Visualization | **15%** | Only default per-run matplotlib plots exist; zero cross-run, publication-quality figures produced yet |
| Manuscript readiness | **5%** | No LaTeX, no Overleaf structure populated beyond an outline (`ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 13); this document and its predecessors are raw material, not manuscript text |
| Overall evidence-package readiness | **70%** | Core empirical/methodological evidence is strong and honestly scoped; what remains is derivation/writing work (figures, tables, theory prose, lit review), not new experiments — see §10 |

## 10. Next Phase Plan

| Phase | Input | Output | Files likely involved | Code changes? | GPU experiments? |
|---|---|---|---|---|---|
| **P1 — Evidence freeze** | This document + registry | A version-tagged snapshot of the evidence state (already effectively performed by this task) | `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md`, `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` | No | No |
| **P2 — Master results synchronization** | This document's registry + §11's findings | Backfilled `Research_Database` CSVs (schema extended for attack/attribution/Multi-Krum fields) | `Research_Database/FedTrace_Master_Results/*.csv`, `Research_Database/Experiment_Index.csv`; possibly a new `sync_master_results.py` if automation is explicitly authorized | Yes (a new sync script, if authorized) | No |
| **P3 — Mathematical/theoretical formalization** | `ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 8 + this document §3 | Written propositions/lemmas/failure-condition analyses (Theoretical Analysis manuscript section) | New manuscript-draft file (location TBD, likely under a new `paper/` or `manuscript/` directory) | No | No |
| **P4 — Final figure generation** | `ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 11 + existing `client_round_trace.csv`/`summary.json` files | Publication-quality cross-run figures (onset trajectory, complementarity, Multi-Krum heatmap, utility 2×2, etc.) | New plotting scripts (read-only against `results/`), new figure files | Yes (new, additive plotting scripts only — no existing code touched) | No |
| **P5 — Final table generation** | `ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 12 + `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` | Camera-ready Tables 1–5 | New table-generation scripts or manual authoring | Optional (helper scripts, read-only) | No |
| **P6 — Overleaf/LaTeX manuscript construction** | P3–P5 outputs + every existing analysis document | Full manuscript draft | New LaTeX files (outside this repository or under a new directory) | No (repository code) | No |
| **P7 — Internal reviewer-style audit** | Manuscript draft | Findings list, revision requests | Draft manuscript + a review-notes document | No | Only if the audit itself surfaces a genuine blocking gap (not assumed here) |
| **P8 — Venue-specific polishing** | Reviewed manuscript | Submission-ready manuscript | Manuscript + venue template | No | No |

**No phase above is begun by this task**, per its explicit "consolidate, do not begin the next phase" instruction.

## 11. Master Workbook / Synchronization Audit

**`Research_Database/FedTrace_Master_Results.xlsx`:** exists. Per its own spec (`Research_Database/FedTrace_Master_Results.md`), it is manually maintained and mirrors 8 CSV source tables under `Research_Database/FedTrace_Master_Results/`.

**`sync_master_results.py`:** re-verified in this task via a full repository search (excluding `.venv/`) — **does not exist anywhere.** `Research_Database/README.md` states this explicitly: no automation exists; rows are added to the CSVs by hand.

**Which experiments are represented:** `Research_Database/Experiment_Index.csv` — re-read directly in this task — contains exactly **4 rows: `EXP_001`–`EXP_004`.** The other seven `FedTrace_Master_Results/*.csv` source tables are, per the spec document's own "Current Row Counts" table, populated with at most 2 experiment rows for most sheets and 0 for the Attack/Attribution/Paper-Tables sheets (not independently re-verified row-by-row in this task beyond `Experiment_Index.csv`, but no evidence contradicts this and no experiment past `EXP_004` appears anywhere in the database).

**Which experiments are missing:** **`EXP_005` through `EXP_027`, all 23 of them** — the entire heterogeneity-characterization tail, all of FedProx, the entire attack/FedTrace development-and-validation program, the entire Multi-Krum external-baseline branch, and both generalization-gate experiments.

**Whether the workbook can represent the final FedTrace fields:** **no, not without schema extension.** The current 8-sheet schema (`01_Experiment_Summary` through `08_Notes`) predates FedTrace entirely; no column anticipates attack type/onset/malicious-client-id, FedTrace score/attribution/onset-estimate/delay fields, or any Multi-Krum-specific diagnostic (`krum_score`, `krum_rank`, `selected`/`rejected`, `f`/`m`). The `05_Attack_Results`/`06_Attribution_Results` sheets exist as named placeholders with an apparently reasonable column layout for attack results, but have never been populated and have not been checked in this task against the actual fields this project's `client_round_trace.csv` files produce.

**Recommended synchronization design (exact design only — not built or run in this task, per explicit restriction):**
1. Extend `Research_Database/FedTrace_Master_Results/05_Attack_Results.csv` and `06_Attribution_Results.csv` schemas to include: attack family, malicious client, onset round, severity/target-cosine, FedTrace version used, predicted client, onset estimate, onset error, detection delay, honest persistent FP count, dominant signal.
2. Add a new `09_MultiKrum_Diagnostics.csv` (or extend `04_Federated_Results.csv`) with: `f`, `m`, per-round selected/rejected status summary, true-attacker rejection rate, regime-switch round (if any), proxy-predicted client.
3. Backfill all 23 missing experiment rows from their already-existing, already-verified `summary.json`/`client_round_trace.csv`/analysis-document sources — a bookkeeping task with zero new experimental content, directly sourced from `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` (this task's own output).
4. Only after (1)–(3) are schema-designed and reviewed, consider whether a `sync_master_results.py` automation script is worth building (this document does not decide that; it is a P2-phase question, §10).

**Whether another synchronization mechanism exists:** no — `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` (this task's own output) and this master document are the closest things to a synchronized cross-experiment record that currently exist, and neither is the `Research_Database`'s replacement — they are analysis-layer documents, not the project's designated tabular database.

**Additional finding, beyond this Part's original scope, surfaced directly in this task (see the note at the top of this document):** the root-level `FEDTRACE_FM_MASTER.md`/`PROJECT_CONTEXT.md`/`PROJECT_ROADMAP.md`/`RESEARCH_DIARY.md` quartet — which self-describes as the project's single source of truth — is stale in the same way and to a similar degree as `Research_Database`: frozen at `EXP_004`, with Phases 4/5 (attack simulation, FedTrace attribution) marked "Not Started" despite being, in fact, essentially complete through the generalization gate. This is a second, independent instance of the same underlying pattern (rich, disciplined documentation infrastructure that was not kept current as the project accelerated past its original phase-by-phase IID/IID-round-budget scope) and should be addressed alongside, not instead of, the `Research_Database` synchronization work in P2.

## 12. Research Continuity — Can This Project Be Reconstructed From Files Alone?

**Yes, with this document as the entry point.** Minimum reading order for a new session: (1) this file; (2) `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv` for exact per-run numbers; (3) `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` for the frozen method's own design rationale; (4) `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` for the independent-validation narrative; (5) `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` for the external baseline; (6) `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md` for the generalization gate; (7) `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` for the full gap analysis this document's Parts 4/8/9/11 draw from. **Do not** rely on `FEDTRACE_FM_MASTER.md`/`PROJECT_CONTEXT.md`/`PROJECT_ROADMAP.md`/`RESEARCH_DIARY.md` or `Research_Database/` for anything past `EXP_004` — both are stale, per §11.

---

## Final Report

1. **Files created:** `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md`, `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv`.
2. **Number of experiments registered:** 27 rows (`EXP_001`–`EXP_027`), including `EXP_013` (recorded as a synthesis-only entry with no independent run, per its actual nature).
3. **Missing/unverifiable experiment IDs:** none missing — every ID `EXP_001`–`EXP_027` was located and verified against a real result artifact or (for `EXP_013`) confirmed as a synthesis-only entry. `EXP_002_SMOKE` and the two centralized smoke/dev artifacts exist but are not part of the numbered `EXP_001`–`EXP_027` sequence and were excluded from the registry per the task's own scope.
4. **Final frozen FedTrace version:** v3, exactly as specified in §3, unmodified since `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` was written and before `EXP_016` was ever run.
5. **Development/validation separation verdict:** clean — `EXP_014`/`EXP_015` (development-only) vs. `EXP_016`–`EXP_019` (frozen-formula independent validation) is explicitly, consistently maintained across every source document re-audited in this task; no development-stage result is cited anywhere as validation evidence.
6. **Strongest final contribution:** the empirical demonstration (§5, §6) that forensic attribution and Byzantine-robust aggregation are distinct problems, combined with FedTrace v3's own validated, multi-signal, per-client-calibrated design (§2, §3).
7. **Strongest empirical evidence:** the fully-crossed 2-seed × 2-attack-family Multi-Krum dichotomy (100% vs. 0% rejection, zero exceptions) paired with FedTrace's identical, unaffected success across the same conditions (§5).
8. **Strongest external-baseline finding:** Multi-Krum's rejection-proxy misattributes an honest client in all four attacked runs and never confirms the true directional attacker at either seed (§5).
9. **Attacker-identity generalization verdict:** PASS on one new, deliberately-chosen identity (`client_004`); not proof of arbitrary-identity invariance (§6).
10. **Onset-generalization verdict:** PASS on one new onset (round 30); not proof of arbitrary-onset invariance; onsets before `T_ref=20` remain untested (§6).
11. **Round-0 artifact status:** confirmed present, fully mechanistically explained, non-persistent, zero effect on any attribution/onset conclusion in any run checked; one prior document's descriptive sentence ("no honest client ever exceeds 3") does not hold under re-verification and should not be reused (§3, §7).
12. **Claim/evidence matrix status:** built for all 11 requested claims (C1–C11), each with exact supporting experiments/documents, strength rating, allowed wording, overclaim to avoid, and known limitation (§4).
13. **Unsupported/overclaim risks:** "FedTrace beats Multi-Krum as a detector" (§5); "combining signals improves detection beyond the best single signal" (§4/C5, directly contradicted by the evidence); "no honest client ever exceeds threshold" (§7); any arbitrary-identity/arbitrary-onset/arbitrary-heterogeneity/arbitrary-dataset generalization claim (§4, §6).
14. **Master workbook status:** exists but stale — only `EXP_001`–`EXP_004` represented; schema predates FedTrace/Multi-Krum entirely (§11).
15. **`sync_master_results.py` status:** does not exist anywhere in the repository, re-confirmed by full search in this task (§11).
16. **Experiments missing from the workbook:** `EXP_005` through `EXP_027` — 23 of 27 (§11).
17. **Overall evidence-package readiness:** **70%** (§9, with per-dimension breakdown).
18. **Manuscript readiness:** **5%** (§9) — raw evidence is strong and organized; no LaTeX/figures/tables/theory prose yet exists.
19. **Whether any essential GPU experiment remains:** **No** — consistent with `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`'s own conclusion, re-confirmed by this consolidation; no genuinely new blocker was discovered while building this document.
20. **Optional experiments, if any:** a third attacker identity, an onset earlier than `T_ref=20`, a second seed for either generalization axis, a joint identity+onset run, Gaussian-family generalization, a second external baseline (coordinate-median/trimmed-mean), a second heterogeneity level — all **HIGH-VALUE-IF-BUDGET-ALLOWS**, none mandatory (§4, §6, §8).
21. **Exact recommended next phase:** **P1 (evidence freeze) is effectively complete as of this document**; the next phase to begin is **P2 (master results synchronization)** or **P3 (theoretical formalization)** — both are pure derivation/documentation work requiring no new experiments and no source-code changes beyond an optional new sync script (§10). This task does not begin either.
22. **Files modified outside `analysis/`:** none.
23. **Confirmation no experiment was run:** confirmed — every number in this document and the registry was read from already-existing `summary.json`/`client_round_trace.csv`/analysis-document artifacts.
24. **Confirmation FedTrace unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was read (in a prior task in this series, its full content already verified against this document's §3) but not modified in this task; no formula, threshold, persistence rule, `T_ref`, or combiner was altered.
25. **Confirmation attack code unchanged:** confirmed — no file under `src/attacks/` was read or touched in this task.
26. **All commands executed:** this task used `Glob`, `Read`, and `Write` tool calls throughout, plus a small number of read-only `Bash` calls running Python one-liners to (a) enumerate every `summary.json` under `results/` and extract `experiment_id`/`aggregation_strategy`/`seed`/`partition`/`num_clients`/`final_test`/`attack` fields for the registry, (b) count and print `Research_Database/Experiment_Index.csv` rows, and (c) search the repository (excluding `.venv/`) for `sync_master_results*`. No training, no experiment runner, no test suite, and no file-modifying command of any kind was executed.
