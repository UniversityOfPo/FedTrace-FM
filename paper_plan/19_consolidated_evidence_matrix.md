# 19 — Consolidated Evidence Matrix (archived NDSS + development reviewer-response runs)

> **Review note — 2026-10-26 (Prompt 33, P-49).** Added §8 (claim-support classification). Numbers unchanged.

Prepared 2026-10-26 (Prompt 32). Labels: **CLAUDE-REPORTED** (checked by Claude from artifacts; no row is
PI-VERIFIED yet), **DEVELOPMENT ONLY** (seeds 123 / 2026 re-used as development seeds; no independent seed),
**POST-HOC EXPLORATORY**, **PROPOSED**. Every number below maps to a ledger row (`04_claim_evidence_ledger.md`) and to
an artifact; source artifacts (local, git-ignored) with SHA-256 prefixes:
`results_rev/pilot/phase_s/phase_s.json` (9823d8a5…), `results_rev/pilot/benign_drift/benign_drift.json`
(b651b38e…), `results_rev/pilot/evidence_matrix/evidence_matrix.json` (7f7c9ff7…; code `src/rev/evidence_matrix.py`,
commit `2e1e0ad`). Archived results: doc 18.

## 1. Verification of the PI-listed findings (all checked against artifacts on 2026-10-26)

| PI statement | Artifact value | Verdict |
|---|---|---|
| BadNets seed 123 did not meet the preregistered 50 % ASR criterion | final ASR 43.54 % (≥ 50 % never reached in rounds 20–49) — L-27 | CONFIRMED |
| BadNets seed 2026 ≈ 64.7 % ASR | 64.67 % — L-42 | CONFIRMED |
| v3 falsely attributed an honest client in seed-2026 poisoning | BadNets: client_004 attributed (honest client_001, client_004 confirmed at r23; attacker at r24); relabel-only: client_004 attributed, attacker never confirmed — L-41, L-51 | CONFIRMED (both poisoning arms) |
| Benign drift also caused false accusations | v3 (seed 2026: client_001, client_004 at r23); norm z-score (123: drifted client_002 at r23; 2026: client_001 at r49); norm CUSUM (client_002 at r28 / r37) — L-64 | CONFIRMED (an alarm on the drifted client is an accusation of an honest client) |
| Norm detectors competitive on attacked runs but fail benign-drift specificity | z-score h 1.75 and CUSUM h 11.75: correct 4/4 attacked, 0 honest alarms; both falsified by drift — L-56, L-64 | CONFIRMED, with the caveat that their thresholds were set on the clean runs of the same seeds (L-57) |
| D1 source localization falsified | F1 FALSIFIED (seed-2026 BadNets, client_002 ranked 5/5) — L-72, L-76 | CONFIRMED |
| Label-shift-invariant correction falsified | H-LS FALSIFIED (seed 2026, relabel-only residual below drift; uncorrected baseline as good or better) — L-78, L-79 | CONFIRMED |

## 2. Development matrix (rev runs; intervened client client_002, onset 20; K = 5; CIFAR-100 / ResNet-18 / α 0.1)

Detection = any confirmation; attribution = first confirmed client is the attacker; honest FA = honest clients
confirmed (in benign drift every confirmation is an honest false accusation); onset error / delay for the attacker's
own confirmation. "Diag." columns use the matched clean counterfactual (never detector inputs).

| Run | Malicious | ASR | Δ top-1 vs clean (pp) | Norm ratio r25–35 (diag.) | Self-cos Δ r21–29 (diag.) | Max abs. ref. movement on others, z (diag.) | **v3**: detect / attrib / honest FA / onset err / delay | **ZP-norm (h 1.75)** | **CUSUM-norm (h 11.75)** |
|---|---|---|---|---|---|---|---|---|---|
| 123 clean | – | 0.73 % | 0 | – | – | – | none | none | none |
| 123 BadNets | yes | 43.54 % (criterion not met) | −1.94 | 0.819 | +0.120 | −0.72 | yes / **correct** / 0 / 8 / 10 | yes / correct / 0 / 1 / 3 | yes / correct / 0 / 0 / 6 |
| 123 relabel-only | yes | 0.72 % (no trigger by design) | −1.26 | 0.874 | +0.075 | −0.72 | **missed** | yes / correct / 0 / 1 / 3 | yes / correct / 0 / 0 / 7 |
| 123 drift | **no** | 0.63 % | −1.75 | 0.872 | +0.064 | −0.72 | none | **client_002 (honest) r23** | **client_002 (honest) r28** |
| 2026 clean | – | 0.27 % | 0 | – | – | – | none | none | none |
| 2026 BadNets | yes | 64.67 % | −0.94 | 0.791 | +0.115 | +2.31 | yes / **wrong (client_004)** / 2 / 2 / 4 | yes / correct / 0 / 0 / 2 | yes / correct / 0 / 0 / 5 |
| 2026 relabel-only | yes | 0.39 % | −1.60 | 0.860 | +0.084 | +2.55 | attacker **missed**; honest client_004 attributed / 2 | yes / correct / 0 / 1 / 3 | yes / correct / 0 / 0 / 6 |
| 2026 drift | **no** | 0.17 % | −1.79 | 0.917 | +0.067 | +2.06 | **client_001, client_004 (honest) r23** | **client_001 (honest) r49** | **client_002 (honest) r37** |

Accuracy: training-time final test evaluation (AMP); the pilot's re-evaluation differs by ≤ 0.03 pp (L-70).
Reference movement: L-70 (ii) definition; seed 123 values are identical across arms (round 20 only, L-54).

## 3. Archived perturbation attacks (doc 18; v3 recomputed 2026-10-26)

| Family | Runs | v3 outcome | Honest alarms |
|---|---|---|---|
| Gaussian ρ = 1, FedAvg K = 5 | EXP_016/017 (+ EXP_014 dev) | correct; onset 20; delay 2 | none |
| Directional c = 0, FedAvg K = 5 | EXP_018/019, 026 (client_004), 027 (onset 30) (+ EXP_015 dev) | correct; onset error 0; delay 2 | none |
| Severity / intermittent | E2 (ρ 0.5: onset 41, delay 23; c 0.5: delay 2), E3 | correct; ρ 0.5 late | none |
| K = 10 | E1 Gaussian, directional, clean | correct; clean: no alarm | none |
| Multi-Krum aggregation | EXP_020/021 (clean), 022/023 (Gaussian), 024/025 (directional) | attacked: correct | **clean: honest confirmation at r46 (both seeds); directional: late honest alarm (r47 / r46)** |

## 4. Five outcomes kept separate

| Outcome | What the evidence shows |
|---|---|
| 1. Malicious-client detection (any confirmation of the attacker) | Archived perturbations: v3 detects all. Data poisoning: v3 detects BadNets (both seeds), misses relabel-only (both); norm z-score / CUSUM detect all four |
| 2. Correct attacker attribution | v3: archived perturbations correct; poisoning 1/4 (123 BadNets); norm detectors 4/4 |
| 3. Benign-drift false accusation | All three frozen detectors accuse an honest client under benign class-composition drift in at least one seed (H-BEN falsified, L-64) |
| 4. Attack effectiveness | BadNets ASR 43.5 % (below criterion) / 64.7 %; relabel-only has no trigger effect and no lasting target bias (L-52); accuracy costs ≈ 1–2 pp in every intervened arm including drift |
| 5. Attack-onset localisation | Archived: v3 onset error 0 (except E2 ρ 0.5: 21); poisoning: v3 error 8 (123 BadNets) / 2 (2026 BadNets attacker); ZP 0–1; CUSUM 0. **Caveat:** the true onset equals T_ref = 20 in all but one run, which favours change-point estimators (L-34); only EXP_027 tests onset ≠ T_ref |

## 5. Mechanism findings (POST-HOC EXPLORATORY, DEVELOPMENT ONLY)

- **Shared-reference contamination** (L-43–L-45, L-59): at seed 2026 the attacker / drifted client was the median
  self-consistency client before onset; its post-onset rise moved the inclusive median up by +1.1 … +2.6 z for the
  other clients, and the two least self-consistent honest clients were confirmed first; at seed 123 the intervened
  client was already above the median, so no contamination after round 20. Leave-one-out and trimmed references are
  contaminated or not depending on the attacker's rank (L-59); a historical reference avoids contamination but
  produces late honest alarms (L-47).
- **Non-specificity** (L-65, L-71): norm decrease, self-cosine rise, signed residual and reference deviation all occur
  under benign drift; norm ratio 0.872 at seed 123 drift is within the poisoning range.
- **Temporal dependence** (L-61): lag-1 autocorrelation 0.82–0.91, cross-client correlation ≈ 0.95; effective n ≈
  1.5–2.9 calibration rounds per client; the z-score threshold's binding clean statistic has a 0.031 margin.
- **Identifiability** (doc 15, Amendment 1): detection, source attribution, malice and onset are distinct; under
  data poisoning, malice is not identifiable from updates without a restriction on admissible benign data
  (two-point bound with TV = 0 when the classes share a law); precedence (Lemma 2) makes the *source of a single first
  change* identifiable in principle under P1–P6, with a one-round margin.

## 6. Failed exploratory directions (must not be presented as contributions)

| Direction | Protocol | Verdict | Ledger |
|---|---|---|---|
| D1 explaining-away source localization | 14 (frozen `f66cc62`) | FALSIFIED; median-position artefacts; no gain over median influence | L-72–L-76 |
| Classifier-row concentration (D2) | none (observation) | covered by DeepSight NEUP / TE; not a contribution | L-71, checklist §A3 |
| Label-shift-invariant conditional shift | 16 (frozen `a7b6aa2`) | FALSIFIED; uncorrected baseline as good or better | L-77–L-81 |

## 7. Limitations (all apply to any paper use)

Two development seeds that were also the archived "validation" seeds; one intervened client (client_002), one onset
(20, = T_ref), one poisoning target class; BadNets at seed 123 below the effectiveness criterion; benign drift is one
deliberately selected type on the same client (stress test, protocol 13 Amendment 1); norm-detector thresholds
calibrated on clean runs that share seeds and partitions with the attacked runs (L-57); K = 5 with full
participation; one dataset / architecture; provenance caveats for EXP_020–025 and E1–E3 (L-17).

## 8. Claim-support classification (Prompt 33, P-49)

| Class | Claims |
|---|---|
| **Supported now** (artifact-backed, CLAUDE-REPORTED; development or archived scope as stated) | v3 attributes and localises Gaussian / norm-preserving directional perturbations in the archived FedAvg runs (onset error 0, delay 2; E2 ρ 0.5 late); v3 misattributes in both seed-2026 poisoning arms and misses relabel-only in both seeds; plain norm z-score / CUSUM attribute all four poisoning runs and accuse honest clients under drift; all three frozen detectors accuse an honest client under benign drift; v3 accuses an honest client in clean and directional Multi-Krum runs; matched prefixes are bit-identical; the inclusive-median contamination mechanism at seed 2026 (retrospective) |
| **Require CPU verification** (doc 25) | the order-statistic contamination characterisation and its deployable proxy across all 29 traces; whether it explains every reference-driven honest accusation; v3 ablations and reference variants on the development runs; the unified outcome table |
| **Require new confirmation runs** (doc 26, deferred, P-53) | any statement that the seed-2026 contamination / false-accusation pattern or the drift falsification *replicates*; any rate or frequency claim |
| **Exploratory only** | classifier-row concentration; BadNets residual rise under the label-shift correction (L-81); cross-seed threshold transfer (L-58) |
| **Not supportable** | FedTrace outperforms baselines; FedTrace attributes data poisoning reliably; "no honest false positives" unscoped; generalisation from two seeds; malicious intent identified; D1 or the label-shift correction as working methods; FLTracer / FLDetector superiority |
