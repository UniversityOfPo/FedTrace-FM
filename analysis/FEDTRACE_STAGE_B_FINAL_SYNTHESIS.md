# FedTrace Stage B: Final Independent-Validation Synthesis

**Status:** Synthesis document only. No code, config, test, README, Research Database, or research-plan file was created or modified in producing it. No experiment was run. This document combines already-completed, already-analyzed evidence into one scoped scientific picture; it does not introduce any new finding, formula, or run.

---

## 1. Executive Scientific Verdict

**Stage B independently validated all four of its pre-registered targets, within an explicitly bounded scope:**

- **Malicious-client attribution:** correctly identified `client_002` in all 4 of 4 independent validation runs.
- **Attack-onset localization:** onset error `0` and detection delay `2` (the minimum possible under the frozen `P=3` persistence rule) in all 4 of 4 runs.
- **Zero persistent honest-client false positives:** across all 4 runs, all 5 clients, every round with a defined signal — no honest client was ever confirmed, let alone confirmed before the true malicious client.
- **Cross-attack coverage:** the same frozen detector, with no attack-specific tuning, correctly and promptly attributed both a magnitude-revealing attack (Gaussian update noise) and a norm-preserving attack (directional rotation) that was specifically designed to defeat a magnitude shortcut.
- **Cross-seed reproducibility:** each attack family was independently confirmed on two seeds (123, 2026) never used during FedTrace v3's design, in addition to the seed-42 development runs that were.

**Exact scope — this verdict does not generalize beyond it:** two attack families (`gaussian_update_noise` at `ρ=1.0`; `norm_preserving_directional` at `target_cosine=0.0`), two unseen validation seeds per family (123, 2026), strong Non-IID Dirichlet `α=0.1`, FedAvg aggregation, 5 clients (all 5 selected every round), CIFAR-100, ResNet-18, 50 communication rounds, exactly one malicious client (`client_002`), attack onset fixed at round 20. Every claim in this document is qualified by this exact configuration; §17 makes explicit what is *not* claimed.

## 2. Development vs. Validation Boundary

**Development-only (used to design and debug FedTrace v1/v2/v3):** `EXP_014` (`gaussian_update_noise`, seed 42) and `EXP_015` (`norm_preserving_directional`, seed 42). These runs exposed the peer-cosine baseline's failure (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`), the magnitude signal's near-total defeat by a norm-preserving attack (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`), and — critically — two successive methodological flaws found only by directly inspecting these runs' own traces: v1's non-stationary temporal calibration (`analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`) and v2's small-sample same-round scale instability (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`).

**Independent validation (never used to change the frozen v3 detector):** `EXP_016` (Gaussian, seed 123), `EXP_017` (Gaussian, seed 2026), `EXP_018` (directional, seed 123), `EXP_019` (directional, seed 2026). FedTrace v3's formulas were frozen in `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16 *before* any of these four traces existed, and — per each of their own analysis documents' explicit discipline — no result from any of them was permitted to, or did, modify any formula, threshold, calibration rule, or missing-value convention.

**This boundary is the primary reason this evidence carries more weight than the development-stage findings alone.** A method that only works on the data used to design it demonstrates nothing beyond curve-fitting; a method that continues to work on data it never saw, using formulas fixed in advance, demonstrates that its calibration and combination rules generalize at least across independent draws of this project's own strong-heterogeneity partition process. That is what Stage B was built to test, and — within the scope of §1 — is what it found.

## 3. Frozen FedTrace v3 Method

Restated exactly, unaltered by anything in this synthesis:

```
t(k,r) = self_prev_round_cosine

c(r) = median_j t(j,r)                              (same-round, no leave-one-out)

d(k,r) = c(r) - t(k,r)

D_k = { d(k,r) : r in [1, T_ref) },  T_ref = 20

center_k = mean(D_k)
scale_k  = sample standard deviation(D_k)            (n-1 denominator)

z_t_res(k,r) = max(0, [d(k,r) - center_k] / scale_k)

z_m(k,r)                                             (unchanged from v1: per-client
                                                       longitudinal calibration over
                                                       round_id < T_ref)

FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))

threshold: score > 3
persistence: P = 3 consecutive anomalous rounds
attribution: earliest-confirmed client (ties -> higher mean score, then client_id)
onset estimate: first round of the confirmed P-round persistent run
```

**Peer cosine (`z_p`) is not part of the primary score** — computed only for ablation/monitoring, per the same decision carried unchanged from v2 through v3 on the evidence that it never contributed a single detection or false alarm in any development or validation run (§6, §12).

## 4. Why v1 and v2 Failed

This is reported as methodological failure analysis leading to a fresh frozen version — not as tuning until success. Each version was fully specified and evaluated before the next was designed, and each failure was diagnosed from data, not guessed.

- **v1 failed** because its temporal signal was calibrated against a static, pre-attack-only absolute reference (`mean_pre_t(k)`, fixed from rounds 1–19). `self_prev_round_cosine` is not stationary — every client's value declines secularly across training — so a calibration window that ends at round 20 cannot anticipate the decline continuing past it. This produced late honest false positives on 3 of 4 honest clients in `EXP_015` (confirmed at rounds 43, 45, 47), though attribution survived in that single instance because the true attack was still detected first (`analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`).
- **v2 failed** because, having correctly diagnosed v1's non-stationarity problem and fixed it with a same-round cross-sectional median/MAD reference (which cancels a shared secular trend by construction), it estimated that reference's *scale* from only the 5 clients present in a single round — too small a sample for a stable robust-spread estimate. On specific rounds where the 4 non-outlier clients happened to cluster tightly by chance, the resulting `scale_t(r)` was small enough to inflate an ordinary honest client's typical deviation past threshold. This produced a **pre-attack** confirmed false positive for `client_000` (rounds 9–11, `EXP_015`) — one that was earlier than the true attack's own confirmation, which flipped attribution to the wrong client entirely (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`).
- **v3's correction** kept v2's useful part (the same-round median reference, which still removes the shared secular drift) and replaced only the flawed part: the scale is now estimated **longitudinally**, from each client's own ~19 historical residuals, the same class of calibration already proven stable for `z_m`. This resolved both the v1 (non-stationarity) and v2 (small-sample instability) false-positive mechanisms on the same development data that exposed them (`analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`), and was then frozen — with no further formula change — before Stage B began.

## 5. Stage B Experiment Matrix

| Experiment | Attack family | Seed | Paired honest baseline | Malicious client | Attack onset | Attack severity | Predicted client | First anomaly | Confirmation | Onset est. | Onset error | Delay | Honest persistent FP | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `EXP_016` | Gaussian update-noise | 123 | `EXP_008` | `client_002` | 20 | `ρ=1.0` | `client_002` | 20 | 22 | 20 | 0 | 2 | 0 | **PASS** |
| `EXP_017` | Gaussian update-noise | 2026 | `EXP_009` | `client_002` | 20 | `ρ=1.0` | `client_002` | 20 | 22 | 20 | 0 | 2 | 0 | **PASS** |
| `EXP_018` | Norm-preserving directional | 123 | `EXP_008` | `client_002` | 20 | `target_cosine=0.0` | `client_002` | 20 | 22 | 20 | 0 | 2 | 0 | **PASS** |
| `EXP_019` | Norm-preserving directional | 2026 | `EXP_009` | `client_002` | 20 | `target_cosine=0.0` | `client_002` | 20 | 22 | 20 | 0 | 2 | 0 | **PASS** |

Every value is taken directly from the corresponding `analysis/EXP_01{6,7,8,9}_STAGE_B_*.md` document — none re-derived or adjusted here.

## 6. Gaussian Family Synthesis (`EXP_016` + `EXP_017`)

Both seeds PASS: correct attribution, onset estimate `20`, confirmation round `22`, delay `2`, zero honest persistent FP, in both runs.

**Component behavior differed between the two seeds, even within the same attack family:**
- `EXP_016` (seed 123): `z_m` dominated **all 30/30** malicious anomalous rounds.
- `EXP_017` (seed 2026): dominance was **mixed** — `z_m` won 10/30 rounds (round 20 itself, and rounds 40–49 except 43), `z_t_res` won 20/30 rounds (21–39, 43).

This is itself a useful, unforced finding: the Gaussian attack is magnitude-revealing in the sense that magnitude *is* a reliable signal for it (both seeds detect at the earliest possible round via at least one component), but *which* component is momentarily the larger of the two, round to round, is not fixed even within one attack family — it depends on the specific realized trajectory of `update_l2_norm` versus `self_prev_round_cosine` at that seed. **This is evidence for the value of unified multi-view coverage** (a detector relying on `z_m` alone would have been correct at both seeds here; a detector relying on `z_t_res` alone would also have been correct at both, since `EXP_017`'s temporal-residual-only result independently matched `z_m`'s onset/confirmation exactly) — **it is not evidence that the combined detector statistically outperforms magnitude-only on this attack family**, since the combined score never produced an earlier or more precise detection than magnitude-only did in either Gaussian run.

## 7. Directional Family Synthesis (`EXP_018` + `EXP_019`)

Both seeds PASS: correct `client_002` attribution, onset `20`, confirmation `22`, delay `2`, zero honest persistent FP, in both runs.

**Magnitude-only behavior:**
- `EXP_018`: first anomaly round 47, confirmation round 49 (the very last round of the run), onset estimate 47 — onset error 27, delay 29.
- `EXP_019`: first anomaly round 44, confirmation round 46, onset estimate 44 — onset error 24, delay 26.

Magnitude-only **eventually reacted in both runs** (driven by the honest-counterfactual global-model drift documented in both analysis documents' §3, not by the attack directly manipulating magnitude — `delta_attack_norm_ratio ≈ 1.0` to `~1e-10` precision at every active round in both runs), but far too late to satisfy the pre-registered onset-error (`≤2`) or delay (`≤4`) criteria on its own.

**Temporal-residual behavior:** in both directional validation runs, `z_t_res` produced immediate onset detection at round 20 and confirmation at round 22 — identical to the combined detector's own result.

**Signal dominance:** `EXP_018`: `z_t_res` 30/30. `EXP_019`: `z_t_res` 30/30.

**Why this is especially important:** this attack family preserves update norm by construction (§3 of both underlying analysis documents confirms this to floating-point precision, not merely by name). Successful early attribution here could not have come from a magnitude shortcut — it required a genuinely directional/temporal inconsistency signal, and `z_t_res` is what provided it, consistently, at both unseen seeds.

## 8. Cross-Attack Coverage

**The core Stage B question — can one frozen detector identify malicious-client behavior across both a magnitude-revealing Gaussian attack and a norm-preserving directional attack, without knowing the attack type beforehand — is answered yes, within the tested scope.** `FedTraceV3Score` used `z_m` where it was informative (Gaussian family) and `z_t_res` where it was informative (directional family), automatically, via the same fixed `max()` rule, with no per-attack branch or retuning anywhere in the implementation.

**This is precisely stated as coverage, not fusion.** In all four Stage B runs, `FedTraceV3Score`'s onset and delay exactly match whichever single component dominated that run — the combined detector has never, across any of the four runs, produced a detection outcome better than its best individual component for that specific run. The `max()` combiner's demonstrated value is **attack-agnostic coverage**: it inherits whichever complementary signal happens to be informative in a given run, sparing a deployer from having to know or guess the attack family in advance. It has not been shown to synergistically combine evidence to outperform the single best signal, and this document does not claim that it does.

## 9. Attribution Performance

| Run | Predicted client | Correct? | Honest client confirmed first? |
|---|---|---|---|
| `EXP_016` | `client_002` | Yes | No |
| `EXP_017` | `client_002` | Yes | No |
| `EXP_018` | `client_002` | Yes | No |
| `EXP_019` | `client_002` | Yes | No |

**4/4 runs correctly attributed. Zero wrong attributions. Zero honest clients ever confirmed at all, let alone before `client_002`, in any of the four runs.**

## 10. Onset Localization

| Run | True onset | Estimated onset | Onset error | Confirmation delay |
|---|---|---|---|---|
| `EXP_016` | 20 | 20 | 0 | 2 |
| `EXP_017` | 20 | 20 | 0 | 2 |
| `EXP_018` | 20 | 20 | 0 | 2 |
| `EXP_019` | 20 | 20 | 0 | 2 |

**All four runs satisfy `onset error ≤ 2` and `detection delay ≤ 4` — and do so at the tightest possible values achievable under the frozen `P=3` rule** (delay `2` is the theoretical minimum: three consecutive anomalous rounds starting at the true onset confirm at `onset + 2`).

## 11. False-Positive Control

| Run | Honest persistent FP |
|---|---|
| `EXP_016` | 0 |
| `EXP_017` | 0 |
| `EXP_018` | 0 |
| `EXP_019` | 0 |

**Total across all four independent validation runs: 0 honest persistent false positives, out of 16 honest-client-run opportunities (4 honest clients × 4 runs).** No honest client was persistently misclassified in any run.

**Isolated raw anomaly vs. confirmed persistent anomaly — distinguished explicitly, not conflated:** a single round where `FedTraceV3Score(k,r) > 3` is a raw anomaly; it only becomes a confirmed detection after `P=3` consecutive such rounds. Across all four validation runs' exhaustive per-client-round checks, no honest client produced even a single isolated raw anomaly post-attack in any run (`EXP_016`/`017`/`018`/`019` analysis documents §5/§6/§10/§6 respectively) — the false-positive control demonstrated here is stronger than merely "no run of 3," it is "no honest client-round ever exceeded threshold at all" in every one of the four runs. **`P=3` matters because it is the mechanism that would have caught an isolated fluctuation before it could be mistaken for a detection** — its value was directly demonstrated during development (v1's and v2's false positives were themselves *persistent*, multi-round events that a weaker single-round rule would have made even easier to trigger by chance), even though no validation run required it to intervene.

**This is an observed-zero result, not a claim of zero false-positive probability.** Sixteen honest-client-run opportunities across four runs is not a sample size from which a false-positive rate can be estimated with any statistical confidence; it is a report of what was actually observed, nothing more.

## 12. Baseline Comparison

- **Peer-cosine baseline:** failed to detect the malicious client in every one of the four validation runs (`naive_flag` never `True` in any of the 1,000 trace rows across `EXP_016`–`EXP_019`) — the same failure mode already established at the development stage, now confirmed at both unseen seeds and both attack families.
- **Magnitude-only:** strong and fast for the Gaussian family (onset `20`, delay `2` in both `EXP_016`/`EXP_017`); very late for the directional family (`EXP_018`: delay `29`; `EXP_019`: delay `26` — both far outside the `≤4` criterion).
- **Temporal-residual-only:** strong and fast for the directional family (onset `20`, delay `2` in both `EXP_018`/`EXP_019`); also useful in the Gaussian runs, but with seed-dependent strength (matched magnitude-only exactly in `EXP_017`; lagged it by one round in `EXP_016`, onset `21`/confirm `23`).
- **FedTrace v3:** one frozen detector, with no attack-specific branch, matched the best available single signal's onset/delay in all four runs.

**The main demonstrated advantage is unified attack-agnostic coverage — not necessarily beating the best attack-specific signal on each attack**, as already stated in §8 and reiterated here to keep the baseline comparison from being read as a superiority claim it does not support.

## 13. Global Model Impact

| Comparison | Test Top-1 Δ | Best validation Top-1 Δ |
|---|---|---|
| `EXP_016` vs. `EXP_008` (Gaussian, seed 123) | −0.33 pp | −0.78 pp |
| `EXP_017` vs. `EXP_009` (Gaussian, seed 2026) | −0.18 pp | +0.10 pp (noise-level) |
| `EXP_018` vs. `EXP_008` (directional, seed 123) | **−3.73 pp** | **−3.16 pp** |
| `EXP_019` vs. `EXP_009` (directional, seed 2026) | **−3.70 pp** | **−3.64 pp** |

The Gaussian attack family was **highly traceable even without catastrophic global damage** — sub-1-percentage-point differences in both seeds, essentially neutral. The directional attack family caused **visibly stronger, consistent degradation** at both seeds (3.2–3.7pp), reproducing (not merely matching in direction) the same effect already documented for this attack family at seed 42 (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`). **The traceability objective does not require catastrophic accuracy loss to succeed** — `EXP_016`/`EXP_017` demonstrate a stealthy-in-effect, still-fully-traceable attack, which is arguably the more operationally relevant case for a security narrative than a self-evidently damaging one. Global performance numbers are reported here for completeness and are not further interpreted or extrapolated.

## 14. Role of Honest Heterogeneity

`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` established, from three honest FedAvg seeds at this exact `α=0.1` configuration, that Dirichlet partition realization is itself highly seed-sensitive (client sample-count coefficient of variation ranging ≈10.9%–30.9% across seeds 42/123/2026) and that single-seed conclusions at this configuration were shown, directly, to be unreliable (a feature that appeared 2-of-2 at seeds 42/123 failed to replicate at seed 2026, §6 of that document). A 2.85-percentage-point test-Top-1 range was observed from honest FedAvg alone, with no attack and no method change at all.

**This is precisely why Stage B validated FedTrace v3 on unseen seeds rather than accepting the seed-42 development result as sufficient**: this project's own prior work had already shown that a single-seed conclusion, at this exact heterogeneity level, is not trustworthy on its own — the same partition-realization variance that makes honest-only single-seed claims unreliable is exactly the kind of natural variability a malicious-client detector must not confuse with an attack. Validating FedTrace v3 against two independently-drawn strong-Non-IID partitions, neither used during design, is the direct, disciplined response to that already-established risk. **This does not claim that Dirichlet `α=0.1` captures all forms of real-world heterogeneity** — it is one specific, well-characterized synthetic heterogeneity generator, used here because this project has already invested in characterizing it thoroughly; nothing in Stage B's results speaks to any other heterogeneity model.

## 15. What FedProx Contributed to the Story

FedProx (`μ=0.01`) was tested as an optimization-side response to the substantial FedAvg-vs-IID degradation `EXP_009` established, using the identical three paired seeds (42/123/2026). The three-seed result (`analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md`) was mixed/near-neutral: no metric improved in all three seeds, the mean test-Top-1 difference was slightly negative (−0.11pp), and the result formally satisfied the project's own pre-registered failure criteria for adopting FedProx. **FedAvg was retained as the primary baseline**, and FedProx remains documented only as a completed, tried comparison, not carried forward into the attack-simulation or FedTrace work.

**This kept optimization improvement and malicious attribution as genuinely separate questions**, which is scientifically disciplined: FedTrace v3's entire development and validation series (`EXP_014`–`EXP_019`) is built exclusively on FedAvg, the same aggregation strategy characterized by the three-seed honest baseline — no result in this synthesis is confounded by, or needs to account for, a change in aggregation strategy. FedProx's mixed result is not part of, and does not need to be reconciled with, the traceability findings above; it is mentioned here only as prior context for why FedAvg (not FedProx) is the aggregation strategy this entire Stage B evidence package is built on.

## 16. Claims Supported by Current Evidence

**A.** Under the evaluated strong Non-IID CIFAR-100 setting, FedTrace v3 correctly attributed the malicious client across two distinct attack families and two unseen seeds per family (4/4 independent validation runs).

**B.** FedTrace v3 localized attack onset with zero-round onset error in all four Stage B validation runs, with confirmation two rounds later as an unavoidable consequence of the frozen `P=3` persistence requirement.

**C.** No honest client produced a persistent confirmed false positive in any of the four independent validation runs (0 of 16 honest-client-run opportunities).

**D.** Magnitude and temporal-residual signals showed complementary usefulness across the two tested attack families: magnitude was decisive (and, in one of two seeds, exclusively decisive) for the Gaussian update-noise regime; temporal residual was decisive, exclusively, for the norm-preserving directional regime, in both of its seeds.

**E.** The same frozen detector, with identical thresholds, persistence, and calibration rules, was used across all four validation runs without any attack-specific tuning.

## 17. Claims Not Supported by Current Evidence

- Universal malicious-client detection, of any attack type whatsoever.
- Robustness to arbitrary, unstudied attack strategies (only two specific, pre-registered attack transforms were tested).
- Robustness to multiple simultaneous malicious clients — every run tested exactly one.
- Robustness to an adaptive, white-box adversary aware of FedTrace v3's exact design (e.g. one that deliberately keeps both `update_l2_norm` and `self_prev_round_cosine` within honest-looking bounds simultaneously).
- Robustness across arbitrary datasets, model architectures, client counts, or partial-participation regimes — only CIFAR-100/ResNet-18/5-clients-all-selected was tested.
- Statistical significance across a large population of seeds — two unseen seeds per family is independent-validation evidence, not a population-level significance claim.
- A production-ready security guarantee of any kind.
- Robustness to a different attacker identity (only `client_002` was ever the malicious client).
- Robustness to arbitrary attack onset times (only round 20 was ever tested).
- Robustness to arbitrary attack strengths/severities (only `ρ=1.0` and `target_cosine=0.0` were ever tested — no severity sweep exists).
- Any mitigation or recovery capability — FedTrace v3 is a detection/attribution method only; it does not modify aggregation or exclude flagged clients.
- Byzantine-resilient aggregation of any kind.
- Any privacy guarantee.
- Superiority over any existing external attribution or security method — no external baseline has been evaluated against this data.

## 18. Limitations

Stated without omission: single dataset (CIFAR-100); single architecture (ResNet-18); five clients, all five participating every round (no partial participation ever tested); exactly one malicious client per run; a fixed attacker identity (`client_002`) across every attack run in this entire series; a fixed attack onset round (20) in every attack run; fixed attack severities (`ρ=1.0`; `target_cosine=0.0`) with no severity sweep; exactly two attack families; exactly two unseen validation seeds per family (four validation runs total); a single, specific strong-heterogeneity generator (Dirichlet `α=0.1`); no adaptive attacker of any kind; no partial client participation; no multiple-attacker scenario; no mitigation or aggregation-side response evaluated; a genuinely small validation sample (n=2 per family, n=4 overall) that supports descriptive, per-seed reporting only, not a broad statistical-significance claim of any kind.

## 19. Flagship Paper Core Story

Under strong Non-IID heterogeneity (Dirichlet `α=0.1`), honest client behavior already varies enough — in both partition realization and training dynamics — that naive single-view anomaly detectors cannot reliably separate malicious signal from ordinary heterogeneity noise; this project's own three-seed honest baseline and its naive peer-cosine detector's complete failure against two different attack families both demonstrate this concretely, not hypothetically. Two qualitatively different malicious-client attacks were then constructed to stress this problem from opposite directions: a Gaussian update-noise attack that reveals itself through magnitude, and a norm-preserving directional attack specifically designed to remove that magnitude shortcut entirely, forcing any successful detector to rely on directional/temporal inconsistency instead.

Two successive calibration designs for a temporal-consistency signal each failed for a different, concretely diagnosed reason before a third succeeded: a static pre-attack absolute reference violated the signal's own non-stationarity and produced late honest false positives; a same-round cross-sectional design fixed that problem but estimated its scale from too few clients to be numerically stable, producing an earlier, attribution-breaking false positive instead. The third design — FedTrace v3 — kept the same-round reference that correctly removes shared training drift, but calibrates each client's own residual scale from its longitudinal history instead of a single round's small cross-section, resolving both prior failure modes on the same data that exposed them.

Once frozen, this single detector was evaluated on four runs it had never seen — two seeds per attack family — and correctly attributed the malicious client with exact onset localization and zero honest false positives in all four, automatically relying on whichever of its two component signals was informative for a given run without being told the attack type in advance. This establishes unified, attack-agnostic traceability across two structurally different attack strategies under realistic strong-heterogeneity conditions — a result explicitly scoped to the evaluated configuration (§1), not claimed to extend beyond it.

## 20. Publication Readiness Assessment

| Component | Status | Rationale |
|---|---|---|
| Problem motivation | **READY** | Concretely established from this project's own heterogeneity/naive-baseline evidence, not asserted abstractly. |
| Strong Non-IID baseline | **READY** | Three-seed FedAvg characterization at `α=0.1` complete, with partition-variance and degradation-magnitude both quantified. |
| Attack simulation | **READY** | Two qualitatively distinct, pre-registered, invariant-verified attack transforms exist and are exercised across two seeds each. |
| Naive baseline weaknesses | **READY** | Peer-cosine baseline's failure is documented across both development runs and all four validation runs. |
| Method design | **READY** | Three explicit design iterations (v1→v2→v3), each with a stated, diagnosed reason for the prior version's failure, culminating in a frozen, fully specified method. |
| Ablation logic | **READY** | Magnitude-only, temporal-only, peer-only, and full-combiner ablations are reported for every development and validation run. |
| Independent validation | **READY** | Four runs, two attack families, two unseen seeds each, formulas frozen before any were inspected. |
| Reproducibility | **PARTIALLY READY** | Every config, seed, and result artifact is documented and paired; no external re-implementation or third-party reproduction has occurred. |
| Statistical breadth | **NOT READY** | n=2 seeds per family; explicitly not a basis for a significance claim (§9 of `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`'s own discipline, carried forward here). |
| Comparison with external state-of-the-art security methods | **NOT READY** | No external Byzantine-robust-aggregation or attribution baseline has been implemented or evaluated against this data. |
| Multiple datasets/models | **NOT READY** | CIFAR-100/ResNet-18 only. |
| Adaptive-adversary evaluation | **NOT READY** | No attacker aware of FedTrace v3's design has been constructed or tested. |
| Mitigation evaluation | **NOT READY** | FedTrace v3 detects and attributes; no exclusion, reweighting, or recovery mechanism has been designed or tested. |
| Final paper figures/tables | **NOT READY** | Not yet produced (§23 proposes a minimal future structure only). |
| Manuscript writing | **NOT READY** | Not started. |

## 21. Next Scientific Decision

**The single biggest remaining gap between current evidence and a high-quality flagship paper is the absence of any external comparison point** — no existing Byzantine-robust-aggregation method (e.g. Krum, trimmed mean, median-based aggregation) or published attribution/anomaly-detection approach has been evaluated against this project's own data, under this project's own strong-heterogeneity conditions. Every comparison made so far (§12) is internal — FedTrace v3 against its own ablations and against the already-known-weak peer-cosine baseline this project itself built. A reviewer's most likely and most damaging objection to the current evidence package is not about seed count or dataset breadth (both are honestly scoped and defensible as a first validated result) but about whether FedTrace v3's reported performance is meaningful relative to *any* existing alternative, since none has yet been measured on the same ground. This is the gap most directly tied to publication risk: without it, the paper's contribution can be read as "we built a detector and it worked on our own benchmark," rather than "we built a detector that works where an appropriate existing approach does not, or does less precisely." Closing this gap — establishing what an external baseline (robust aggregation or an existing attribution method, adapted to log the same trace fields) actually achieves on the same `EXP_016`–`EXP_019` traces or an equivalent paired protocol — is the recommended single highest-priority next research direction. This document does not recommend implementing it in this task.

## 22. Experiment Budget Discipline

Explicitly **not** recommended as a next step, regardless of how easy any of these would be to run:

- **Unnecessary seed multiplication** (e.g. a third or fourth seed per attack family) — two unseen seeds per family already independently validated every pre-registered criterion; a third seed would add cost without addressing the actual gap identified in §21.
- **Arbitrary hyperparameter sweeps** (threshold, `P`, `T_ref`, `ρ`, `target_cosine`) with no specific claim attached — every Stage A/B document in this series has explicitly prohibited post-hoc tuning, and nothing in this synthesis changes that discipline.
- **Extra attack families** introduced without a clear, pre-registered claim they are meant to test — the two families already evaluated were each chosen to stress a specific, articulated hypothesis (magnitude-revealing vs. norm-preserving); a third family without an equally clear rationale would dilute rather than strengthen the paper.
- **Unrelated foundation-model or medical-domain extensions** — entirely outside the scope this evidence package supports; would not close the §21 gap and would introduce new, unvalidated confounds.
- **Optimization experiments unrelated to traceability** (e.g. reopening the FedProx/`local_epochs`/SCAFFOLD line, §15) — already concluded as a separate, closed track; re-opening it does not serve the traceability publication goal.

## 23. Paper Tables to Prepare Later (Structure Only, Not Produced Now)

- **Table 1 — Honest baseline / heterogeneity characterization:** three-seed FedAvg `α=0.1` summary statistics and partition coefficient-of-variation, drawn from `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`.
- **Table 2 — Stage B attribution/onset results:** the four-run matrix in §5 of this document (experiment, attack family, seed, predicted client, onset error, delay, honest FP, verdict).
- **Table 3 — Baseline/ablation comparison:** peer-cosine / magnitude-only / temporal-residual-only / full FedTrace v3, per run, per §12 of this document.
- **Table 4 — Paired model-performance impact:** the four attacked-vs-honest comparisons in §13 of this document.

No figures or camera-ready tables are generated in this task.

## 24. Final Stage B Verdict

**PASS.** Applying `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`'s pre-registered criteria exactly: all four runs individually satisfy all six per-run criteria (§5, §10, §11); both attack families independently satisfy the cross-seed requirement that both unseen seeds pass with no averaging away of a failure (§6, §7 — no failure occurred to average away). Stage B independently validates FedTrace v3 within the pre-registered scope stated in §1.

## 25. Next-Gate Decision

**A. Stage B passes; freeze this evidence package and proceed to the single highest-priority publication gap identified in §21** (an external baseline comparison). This phase is not started in this task.

---

## Final Report

1. **File created:** `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`.
2. **Files modified:** none — every source document was read-only.
3. **Final Stage B verdict:** PASS (§24).
4. **Gaussian-family verdict:** VALIDATED — both `EXP_016` and `EXP_017` PASS every criterion (§6).
5. **Directional-family verdict:** VALIDATED — both `EXP_018` and `EXP_019` PASS every criterion (§7).
6. **Attribution result across all four validation runs:** 4/4 correct (`client_002` in every case); 0 wrong attributions; 0 honest clients ever confirmed (§9).
7. **Onset-localization result across all four runs:** onset error `0` and detection delay `2` in every run — the minimum possible under the frozen `P=3` rule (§10).
8. **Honest persistent FP result:** 0 across all four runs (0 of 16 honest-client-run opportunities); no honest client ever raised even a single raw anomaly post-attack in any run (§11).
9. **Cross-attack signal-complementarity result:** magnitude dominant/decisive for the Gaussian family (100% in `EXP_016`, mixed 10/30 vs. 20/30 in `EXP_017`); temporal-residual dominant/decisive (30/30 in both seeds) for the directional family, which is norm-preserving by construction and verified as such to `~1e-10` precision (§7, §8).
10. **Global-model impact summary:** Gaussian family essentially neutral (Top-1 Δ: −0.33pp, −0.18pp); directional family substantially, consistently degraded (Top-1 Δ: −3.73pp, −3.70pp) (§13).
11. **Strongest supported scientific claim:** FedTrace v3, frozen and unmodified, correctly attributed the malicious client with exact (zero-round-error) onset localization and zero honest false positives across two structurally different attack families and two unseen seeds per family (§16-A/B/C).
12. **Most important unsupported claim:** universal or general-purpose malicious-client/Byzantine robustness — nothing in this evidence extends beyond the two specific, pre-registered attack transforms tested, one attacker, one onset round, one dataset/architecture, and no adaptive adversary (§17).
13. **Top three limitations:** (1) only two attack families, one attacker identity, one fixed onset round, and one severity value per family were ever tested — no sweep of any of these exists; (2) n=2 seeds per family is independent-validation evidence, not a statistical-significance sample; (3) no external baseline (robust aggregation or existing attribution method) has been evaluated against this data (§18, §21).
14. **Publication-readiness assessment:** problem motivation, heterogeneity baseline, attack simulation, naive-baseline weakness, method design, ablation logic, and independent validation are all READY; reproducibility is PARTIALLY READY; statistical breadth, external comparison, multi-dataset/model validation, adaptive-adversary evaluation, mitigation evaluation, final figures/tables, and manuscript writing are all NOT READY (§20).
15. **Single highest-priority next research gap:** establishing an external comparison point (a robust-aggregation method or an existing attribution approach) evaluated against the same paired protocol — the gap most directly tied to publication risk (§21).
16. **Experiments explicitly NOT recommended next:** unnecessary seed multiplication; arbitrary hyperparameter sweeps; new attack families without a clear attached claim; unrelated foundation-model/medical extensions; further FedProx/optimization-track experiments (§22).
17. **Next-gate decision:** **A** — freeze this evidence package and proceed toward the §21 gap; not started in this task (§25).
18. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was read but not modified; no formula, threshold, calibration rule, or missing-value convention was altered.
19. **Confirmation no experiment/config/code/test/database changed:** confirmed — only this one synthesis document was created; no `EXP_020` or any other config, code, test, README, or Research Database file was created or modified.
20. **Confirmation no commands executed:** confirmed — this task used only `Read` calls against already-existing documents; no shell, PowerShell, or Bash command was run, and no web research was performed.
