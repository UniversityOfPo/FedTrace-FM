# 23 — Workshop Manuscript Blueprint (C4AI4C'27; 4 pages + 1 page references)

> **Review note — 2026-10-26 (Prompt 33, P-49).** Superseded in part by the Prompt 33 Task 9 revision (§7 below, added after the CPU validation).

Prepared 2026-10-26 (Prompt 32). Labels: **PROPOSED** (all text below), **CLAUDE-REPORTED** (cited evidence),
**PENDING** (needs an experiment in doc 22). No result is invented; PENDING cells stay empty until the run exists.

## 1. Identity

- **Title (proposed):** *Detection Is Not Attribution: Reference Contamination and Benign Drift in Client-Level
  Attack Forensics for Cross-Silo Federated Learning*
  (alternative: *When Do Federated Forensic Detectors Blame the Wrong Client?*)
- **Research question:** In few-client federated learning, when does server-side update evidence that detects an
  attack also identify the right client and onset — and when does it accuse honest clients?
- **Main technical claim:** Shared same-round references transfer one client's behaviour change to the residuals of
  honest clients in a predictable, order-statistic way, so detection, source attribution and onset localisation fail
  in different conditions; own-history evidence avoids this contamination but cannot separate poisoning from benign
  drift, and malice is not identifiable from updates without an explicit restriction on admissible benign data.
- **Contributions (≤ 3):**
  1. A formal separation of detection, source attribution, malice and onset, with an order-statistic
     characterisation of reference-induced false attribution (doc 21 Option A; doc 15 A3–A7) — *PENDING its
     preregistered test (CPU-1)*.
  2. A matched-counterfactual forensic evaluation protocol (bit-identical pre-onset prefixes; clean / BadNets /
     no-trigger relabel / benign-drift arms; separate endpoints) — CLAUDE-REPORTED, built and used.
  3. An empirical reliability study of FedTrace and standard detectors across perturbation and data-poisoning attacks
     and benign drift, reporting false accusations and failures — CLAUDE-REPORTED (development) + PENDING
     (confirmatory rows, GPU-1).

## 2. Section plan and page budget (≈ 4.0 content pages)

| Section | Content | Budget |
|---|---|---|
| Abstract (≤ 200 words in the portal; same in PDF) | Problem → contamination mechanism → protocol → findings (honest accusations, drift non-specificity, norm baselines competitive) → identifiability caveat | 0.2 |
| 1 Introduction | Forensics in cross-silo FL (few clients, plaintext updates); attribution ≠ detection; why benign drift matters; contributions | 0.55 |
| 2 Related work | FLTracer (closest: attribution + attack time; abrupt benign shifts sacrificed), FLDetector (temporal consistency, no onset), FLForensics (post hoc), FedREDefense (model poisoning only), DeepSight / FoolsGold / BackdoorIndicator / CrowdGuard (different threat models); SPC change detection | 0.35 |
| 3 Problem and observation model | Observation model, A–D definitions, two-point bound (scope!), precedence lemma (one paragraph each) | 0.5 |
| 4 Instruments | FedTrace v3 (magnitude + temporal residual, τ, P, calibration, onset rule) in ≈ 6 lines; baselines (ZP-norm, CUSUM-norm, reference variants); Proposition 1 (contamination) with proof sketch | 0.6 |
| 5 Evaluation protocol | Setting (CIFAR-100, ResNet-18, K = 5, α = 0.1, 50 rounds); matched arms; endpoints; seeds and their status; preregistration | 0.4 |
| 6 Results | Table 1, Fig. 1, Fig. 2; text: archive perturbations, poisoning, drift, contamination test | 1.1 |
| 7 Limitations and conclusion | two development seeds (+ confirmatory if run), one client / onset / drift type, BadNets 123 below criterion, K = 5 only, no FLTracer run, malice not identifiable | 0.3 |
| References | ≈ 15–20 entries | page 5 |

## 3. Figures and tables (≤ 2 + ≤ 2)

| Item | Content | Exact artifacts | Status |
|---|---|---|---|
| **Fig. 1** Contamination mechanism | Seed-2026 BadNets: attacker self-cosine vs inclusive median (attacked and matched-clean counterfactual), honest client_001 / client_004 z_t_res crossing τ = 3 at r21–23; inset: the order-statistic shift predicted by Prop. 1 | `results_rev/X2_badnets_d2026/client_round_trace.csv`, `results_rev/X1_clean_d2026/client_round_trace.csv` (counterfactual for illustration only), `src/rev/reference_audit.py` | data CLAUDE-REPORTED; prediction overlay PENDING (CPU-1) |
| **Fig. 2** Outcome grid | Rows: conditions (archive Gaussian, directional, onset-30; dev BadNets, relabel-only, drift, clean; confirmatory rows if run); columns: detectors (v3, magnitude-only, t_res-only, ZP-norm, CUSUM-norm, LOO-reference v3); cell = correct attribution / attacker detected but honest first / missed / honest accusation in clean or drift | archive traces (doc 18), `results_rev/pilot/phase_s/phase_s.json`, `benign_drift/benign_drift.json`, `evidence_matrix/evidence_matrix.json`; new runs (GPU-1/2/3) | development cells CLAUDE-REPORTED; ablation cells for dev runs PENDING (CPU-2); confirmatory PENDING |
| **Table 1** Main outcomes | Per condition × detector: detection, attribution, honest FA count, onset error, delay; plus ASR and Δ top-1 per condition | same as Fig. 2; ASR / accuracy from `evidence_matrix.json` and pair `effect_*.csv` | dev CLAUDE-REPORTED (doc 19 §2); rest PENDING |
| **Table 2** Positioning | Methods × {online, per-client attribution, onset, data poisoning, benign drift evaluated, K evaluated, inputs}: FedTrace v3, ZP / CUSUM, FLTracer, FLDetector, FLForensics, FedREDefense, DeepSight, BackdoorIndicator | checklist §A1–§A5, doc 20 | VERIFIED-SOURCE (papers read) |

Archived NDSS figures (fig02–fig05) are **not** reused as is (budget; they show only perturbation attacks);
fig02's onset trajectory can be cited in text.

## 4. Claims → evidence map (every result sentence must map here)

| Claim (draft) | Evidence | Status |
|---|---|---|
| v3 attributes and localises Gaussian / norm-preserving directional perturbations (onset error 0, delay 2) in 8 FedAvg K = 5 runs and 2 K = 10 runs | doc 18 (recomputed) | CLAUDE-REPORTED |
| Under BadNets and relabel-only, v3 attributes correctly in 1 of 4 development runs and accuses two honest clients in both seed-2026 runs | L-41, L-51, L-55 | CLAUDE-REPORTED, DEVELOPMENT ONLY |
| A plain norm z-score / CUSUM attributes all four poisoning runs but accuses honest drifting clients | L-56, L-64 | CLAUDE-REPORTED, DEVELOPMENT ONLY (thresholds from same-seed clean runs) |
| The false accusations follow an upward crossing of the inclusive median by the intervened client | L-43–L-45; Prop. 1 test | mechanism CLAUDE-REPORTED; test PENDING |
| Under data poisoning, malice is not identifiable from updates without restricting benign data | doc 15 A3–A4 (proof) | CLAUDE-REPORTED (analytic; PI to check) |
| Two exploratory corrections (source exclusion; label-shift invariance) did not resolve this | L-72–L-81 | CLAUDE-REPORTED — one sentence in Limitations at most |

## 5. Reviewer-objection matrix (Task 10)

| Objection | Current evidence | Remaining gap | Smallest remedy | Resolvable before 11 Nov? |
|---|---|---|---|---|
| What is new relative to FLTracer? | FLTracer: attribution + attack time, cross-round Kalman, sacrifices abrupt benign shifts; no analysis of reference-induced false attribution (parts read) | no run of FLTracer | Position the paper as a reliability analysis, not a better detector; Table 2; state FLTracer's K / regime mismatch | yes (framing) |
| Why does FedTrace falsely accuse honest clients? | contamination mechanism L-43–L-45 | general condition untested | Prop. 1 + CPU-1 + confirmatory runs | yes if CPU-1 / GPU-1 run |
| Why only two development seeds? | seeds 123 / 2026, effective n small | independence | GPU-1 confirmatory seeds (S1 or S2) | yes (≈ 8 GPU-h) |
| Why is one BadNets run below the effectiveness criterion? | ASR 43.5 % (criterion 50 %), preregistered | — | report as is; treat seed-123 BadNets as "partially effective"; do not re-tune | yes (disclosure) |
| Why are standard norm detectors competitive? | ZP / CUSUM 4/4 on poisoning | thresholds from same-seed clean runs | report honestly; that is a finding; confirmatory seeds remove the same-seed calibration bias | yes |
| Is the benign drift representative? | one class-composition drift, deliberately selected client (stress test) | other drift types | state as a stress test; GPU-4 optional | partially |
| Is onset estimation truly online? | online wrapper identical to offline on all 21 archived traces (L-12); calibration self-inclusive offline (L-13) | onset = T_ref confound | GPU-3 (onset 30) | yes if GPU-3 runs |
| Are the baselines fair? | same traces, frozen thresholds, no tuning on attacked outcomes | no FLTracer / FLDetector run | Table 2 positioning; optional CPU-4 FLDetector component | partially |
| Does the method distinguish anomaly from malicious intent? | No — doc 15 two-point bound | — | state explicitly as a contribution / limitation | yes |
| Is the paper relevant to cybersecurity for AI? | data-poisoning forensics; 2026 C4AI4C example areas include data poisoning | 2027 call not published | cite scope from the 2026 edition; re-check when the 2027 call appears | likely |

## 6. Integrity notes for drafting

Report negative results (Multi-Krum honest confirmations, E2 late detection, poisoning misattribution, drift false
accusations); never pool attack families; never call two seeds "generalisation"; state the probe-free, update-only
observation model; mark development vs confirmatory rows; disclose provenance caveats (L-17) and the archived
"validation" seeds being re-used as development seeds.
