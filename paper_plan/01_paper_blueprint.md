# 01 — Paper Blueprint (FedTrace, NDSS resubmission)

Status: living document. Created 2026-10-24 (Prompt 15). Update in every phase. No experimental result is
stated here unless it already exists in a hash-sealed archive; planned results are marked **PENDING**.
Evidence status labels follow `04_claim_evidence_ledger.md`: VERIFIED / CLAUDE-REPORTED / HYPOTHESIS / UNSUPPORTED.

## 0. CVPR 2027 Main Track framing (D-018, P-15; supersedes the NDSS framing where they conflict)

**Working research question.** Can heterogeneity-aware, causally evaluated client-update evidence distinguish
malicious poisoning from benign non-IID drift in federated vision learning?

Primary endpoints (both required; neither alone answers the question):
1. *Specificity under benign drift*: honest-client persistent alarms in clean non-IID runs (per run; per 1,000
   honest client-rounds as secondary).
2. *Sensitivity to realistic poisoning*: attribution, online alarm, delay and onset error for BadNets (and the
   detector-aware variant), with the attack's effect measured (attack success rate on triggered non-target test
   images; paired clean top-1).

Development evidence motivating the question (CLAUDE-REPORTED, development data only): frozen v3 raises a
persistent honest alarm late in both clean Multi-Krum runs (L-11); in those runs all clients' late-round update
norms drift upward by a shared ≈ 4 % (median late/calibration ratio 1.037, 1.041) and the falsely confirmed
client drifts most (1.091, 1.100) (L-24). Benign drift is therefore real, partly shared and partly client-specific.

**Venue risks (CVPR).** CVPR reviewers will expect: (i) more than one vision dataset and backbone (currently
CIFAR-100 + ResNet-18 only; adding datasets needs download/licence approval, P-16); (ii) a visual-recognition
threat (BadNets backdoor is the natural fit); (iii) comparison with published FL poisoning detection/forensics
methods (sources not yet obtained, P-07); (iv) a clear methodological contribution beyond a calibrated z-score.
The security framing (accountability, onset) stays, but the paper must read as a federated *vision learning*
methodology paper.

### Candidate improvements (evaluate separately; do NOT merge into a new method until each is justified)

Each candidate changes exactly one component of frozen v3, so its ablation is a single swap. Selection uses
development data only (seeds 123, 2026 + archived runs); the chosen configuration is frozen and pre-registered
before any validation seed is run. Literature status of every "closest technique" below: standard statistical
technique, citation not yet verified against the bibliography — **no novelty is claimed for any candidate in
isolation**.

| | **A. Client-specific robust calibration** | **B. Global training-drift adjustment** | **C. Causal onset localisation** |
|---|---|---|---|
| Component changed | calibration statistics (μ_k, σ_k) | input feature of the magnitude signal | onset estimator (alarm unchanged) |
| Formulation | μ_k = median_{r<T_ref} x_k^r, σ_k = 1.4826·MAD_{r<T_ref}(x_k^r); z^A = \|x_k^r − μ_k\|/σ_k (same for the temporal residual, one-sided) | x̃_k^r = log m_k^r − median_{j∈S_r} log m_j^r (S_r = clients submitting in round r); z_m computed on x̃ with v3's calibration | at alarm a_k: τ̂_k = 1 + last r ∈ [T_ref−1, a_k) with C_k^r = 0, where C_k^r = max(0, C_k^{r−1} + S_k^r − κ₀), C = 0 before T_ref, κ₀ fixed a priori (proposed κ₀ = 1, i.e. one calibration sd) |
| Difference from frozen v3 | v3 uses mean / sample sd over rounds < T_ref, including the round being scored when r < T_ref; non-robust, hence the masking bound (L-14) | v3's z_m uses the raw norm; only the temporal signal has a same-round reference. Shared training dynamics move every client's norm | v3's onset = start of the confirming P-streak; late when early post-onset scores stay below τ (archived E2 ρ = 0.5: onset 20, estimate 41) |
| Closest known technique | Hampel identifier / median–MAD robust z-score | cross-sectional normalisation; v3's own temporal residual c(r) − t; median-norm references in robust aggregation | Page's CUSUM change-time estimate; Hinkley's maximum-likelihood change point |
| Hypothesis | H-A: with k contaminated calibration rounds (onset < T_ref) robust calibration still detects where v3 masks; with clean calibration it does not increase honest alarms | H-B: removing the shared drift reduces honest late-round alarms in clean runs without delaying attribution of BadNets | H-C: τ̂ has lower onset error than the streak start when alarms are late; identical when detection is immediate |
| Potential failure | MAD with n = 20 is inefficient → σ underestimated → more honest alarms; MAD = 0 for near-constant series; heavy-tailed honest variation | client-specific drift (≈ half of the excess in L-24) remains; with K = 5 the median can be moved by ≥ 3 colluders; partial participation changes S_r; an attacker scaling its norm with the round median | pre-onset excursions above κ₀ pull τ̂ early; gradual/ramped attacks have no single change point; κ₀ choice |
| Minimal experiment | CPU on traces of X1/X2 (dev) + calibration-contamination run X4 (GPU, dev seed first) | CPU on archived clean MK traces (dev) + X1/X2 (dev); no extra GPU | CPU on archived E2 ρ = 0.5 trace + X2/X3 traces; no extra GPU |
| Exact ablation | v3 vs v3 with (μ, σ) → (median, 1.4826·MAD) for both signals; nothing else changed | v3 vs v3 with z_m on x̃ instead of m; t_res unchanged | v3 alarms; onset = streak start vs τ̂; alarms identical by construction |
| Target claim (only if supported) | "Robust client-specific calibration removes the calibration-masking blind spot without loss of clean specificity." | "Adjusting for shared training drift reduces benign-drift false alarms in non-IID federated vision training." | "A causal change-time estimate localises attack onset more accurately than the persistence-streak start without changing alarm time." |
| Target figure | F-A: z_max(n, k) bound + detection vs number of contaminated calibration rounds | F-B (extends F5): honest z_m trajectories raw vs drift-adjusted, clean FedAvg and clean MK | F-C: onset error per run, streak start vs τ̂ |

Decision rule (pre-specified): a candidate is adopted only if, on development data, it improves its own target
endpoint and does not worsen the other endpoint (honest alarms or attribution/delay) on any development run;
combinations are tested only among adopted candidates.

## 1. Central research problem

When a federated-learning deployment is poisoned, an operator needs **accountability**, not only robustness:
*which* client is responsible, *from when*, and *how early* this can be established online without
look-ahead. Robust aggregation (e.g. Multi-Krum) answers a different question — which updates to down-weight in
one round — and the archived NDSS evidence shows that per-round exclusion is not a reliable attribution signal
(Multi-Krum never excluded the norm-preserving directional attacker; its rejection proxy persistently
implicated an honest client).

Problem statement for the paper: *online, causal attribution and onset localisation of a poisoning client
from the sequence of submitted updates, under non-IID heterogeneity, with explicit false-alarm control and
against attackers that know the detector.*

## 2. Target threat model (to be written as its own section; reviewer #368B)

| Element | Definition used in the paper |
|---|---|
| System | Cross-silo FL, K clients, server runs FedAvg (main) or Multi-Krum (comparison); full participation in the main setting, partial participation as a robustness condition |
| Defender | Server-side auditor that observes every submitted update (or features of it) and the global model each round; no access to client data; decisions must be causal (use rounds ≤ r only) |
| Attacker capability | Controls one client's local data, training and submitted update (main); two colluding clients (robustness, if completed); cannot see or modify other clients' data or updates unless an *omniscient* variant is declared (not implemented) |
| Attacker objective | Class B/C: a measurable harmful objective — backdoor success (triggered non-target inputs classified as the target class) or targeted misclassification (label flip) — while keeping clean accuracy |
| Attacker knowledge | Class B: none of the detector. Class C: white-box knowledge of FedTrace (τ, P, T_ref, score formulas) and of its own update history; no peer updates |
| Attack timing | Onset at round t₀ ≥ T_ref (main); t₀ < T_ref is the calibration-contamination condition |
| Class A (geometry stress tests) | Gaussian update noise and norm-preserving directional perturbation — no harmful objective is claimed; they test which signal fires |
| Out of scope | Sybil / client-count manipulation, server compromise, attacks on the auditor's logs, privacy leakage from logged updates (handled operationally, §5 of the storage plan) |

## 3. Research questions (labels fixed — reviewer #368B noted inconsistent RQ labels)

- **RQ1 (Attribution & onset).** Does online FedTrace attribute a poisoning client with a measurable harmful
  objective, and how early (alarm, actionable round, onset error), with misses reported?
- **RQ2 (Specificity).** How often does it raise persistent alarms on honest clients in clean runs and in
  attacked runs, per run and per 1,000 honest client-rounds?
- **RQ3 (Adaptive attacker).** What does a detector-aware attacker have to give up (attack success) to stay
  below the detector, and does FedTrace still attribute it?
- **RQ4 (Assumptions).** How do calibration contamination, partial participation and (if completed)
  multiple attackers change RQ1–RQ2?
- **RQ5 (Baselines).** How does FedTrace compare with sequential change detectors (z-score+persistence, CUSUM,
  Page–Hinkley) at matched false-alarm level, and with published forensic/detection methods whose mechanics
  have been verified from the original papers or code?
- **RQ6 (Components).** Which evidence component (magnitude, temporal residual) carries which attack, and how
  sensitive are outcomes to τ, P and T_ref?

## 4. Candidate technical contributions (each must survive the novelty risks in §5)

1. **Forensic formulation with explicit online event semantics** — calibration period, earliest valid scoring
   round, alarm, actionable round, onset estimate, censored delay, retrospective onset kept separate; causal
   attribution tie-break. (Implemented: `src/rev/online.py`.)
2. **Client-calibrated two-signal evidence** — longitudinal magnitude z-score plus a temporal residual against
   a same-round cross-client median with per-client historical scale (frozen v3; NDSS).
3. **Calibration-masking analysis** — with self-inclusive calibration over n rounds, k equal-magnitude
   outliers reach at most z = √((n−k)(n−1)/(kn)); for n = 20, k ≥ 2 stays below τ = 3. This bounds what the
   detector can see during calibration and motivates the calibration-contamination experiment. (Equal-height
   case derived; general case HYPOTHESIS.)
4. **Evaluation methodology for FL forensics** — matched clean controls (bit-identical pre-onset prefix as an
   enforced check), runs as the statistical unit, misses kept in denominators, matched-FPR baselines,
   detector-aware attacks with measured harmful objectives, bit-exact replay of logged updates.
5. **Robust aggregation ≠ attribution** — archived Multi-Krum evidence (geometry dichotomy, honest-client
   rejection proxy), now extended by the finding that FedTrace v3 itself raises a late honest alarm in clean
   Multi-Krum runs (CLAUDE-REPORTED; must be investigated, §8).

## 5. Novelty risks (be explicit in the paper)

| Risk | Why it is real | Mitigation |
|---|---|---|
| "Just a z-score with persistence" | z_m + persistence is a textbook control chart | Contribution must be the forensic formulation, temporal residual, causal semantics and evaluation; include z-score/CUSUM/PH baselines and show where FedTrace differs (or honestly where it does not) |
| Prior FL forensics (FLForensics, NeurIPS 2025) | Same problem name "tracing back malicious clients" | Read the paper/code before claiming differences; mechanics are UNVERIFIED today |
| Prior temporal-consistency detectors (FLDetector) | Also uses update histories across rounds | Not in our bibliography; mechanics UNVERIFIED; must be verified and either compared or discussed |
| FedREDefense (ICML 2024) | Per-round reconstruction-error detection | Verify mechanism; compare only on per-round flags if comparable |
| Reference [10] (reviewer #368B) | Unidentified | Do not guess; requires the submitted PDF's bibliography |
| Narrow setting (one dataset, K=5) | Reviewers called it non-security | Security objective (backdoor) is added; second dataset is OPTIONAL and needs download approval |

## 6. Main scientific hypotheses (pre-register before the validation batch)

- **H1** (RQ1): under the unconstrained backdoor (effective attack, success ≥ pre-registered threshold), online
  FedTrace attributes the attacker in ≥ 4/5 validation seeds with alarm delay ≤ 5 rounds. *Threshold values are
  PROVISIONAL until sealed in the pre-registration.*
- **H2** (RQ2): in clean FedAvg K=5 runs, no honest client is persistently confirmed (0 runs with an honest
  alarm out of 5). The clean Multi-Krum finding makes H2 genuinely at risk.
- **H3** (RQ3): the norm-constrained attacker either loses attack success relative to the unconstrained attacker
  of the same seed, or is still attributed through the temporal-residual signal. (Two-sided: either outcome is
  publishable; the hypothesis is that it cannot have both full success and evasion.)
- **H4** (RQ4): calibration contamination (onset < T_ref) reduces detection; reported as a limitation if so.
- **H5** (RQ5): at matched clean false-alarm level, FedTrace's delay is no worse than CUSUM/PH on the same
  features. (May fail; then the contribution is the semantics/evaluation, not the decision rule.)

## 7. Required mathematical definitions (paper §3; equation library: `analysis/FEDTRACE_LATEX_EQUATION_LIBRARY.md`)

1. Update Δ_k^r = θ_k^r − θ^{r−1} over trainable parameters; magnitude m_k^r = ‖Δ_k^r‖₂.
2. z_m: (m_k^r − μ_k)/σ_k with μ_k, σ_k the sample mean/sd over rounds r < T_ref (absolute value).
3. Temporal cosine t_k^r = cos(Δ_k^r, Δ_k^{prev}); same-round reference c^r = median_k t_k^r; residual
   d_k^r = c^r − t_k^r; z_t_res = max(0, (d_k^r − μ_k^d)/σ_k^d) with calibration over rounds [1, T_ref).
4. Score S_k^r = max(z_m, z_t_res) over defined components.
5. Persistence rule: alarm a_k = first r ≥ T_ref ending P consecutive rounds with S > τ (online); onset
   estimate â_k = a_k − P + 1; actionable round a_k + 1 (post-aggregation monitoring).
6. Attribution: argmin over alarmed clients of a_k; ties by mean S over [â_k, a_k], then client id.
7. Delay δ = a − t₀ for detected runs; for missed runs δ is right-censored at R − 1 − t₀.
8. False-alarm units: (i) runs with ≥ 1 persistently confirmed honest client; (ii) honest client-rounds in
   alarm per 1,000 eligible honest client-rounds (secondary; not independent samples).
9. Attack success rate on triggered non-target-class test images; paired clean-accuracy impact per seed.
10. Constrained attacker: scale Δ so |m − μ| ≤ κσ on its own calibration statistics.
11. CUSUM, Page–Hinkley definitions (`src/rev/baselines.py`) and the matched-FPR threshold rule.
12. Clopper–Pearson interval for k/n; calibration-masking bound (contribution 3).

## 8. Proposed section structure (changes from the rejected draft in bold)

1. Introduction (problem, why robust aggregation is not accountability, contributions, **RQ list**)
2. Background and Related Work (**FL forensics: FLForensics, FLDetector, FedREDefense, [10] once identified**;
   change detection; robust aggregation)
3. **Threat Model** (own section, table from §2 above)
4. FedTrace (signals, calibration, persistence, **online event semantics**, **calibration-masking analysis**)
5. **Attacks** (Class A stress tests, Class B poisoning, Class C detector-aware; capabilities and metrics)
6. Experimental Setup (**moved here from former §III.H** — reviewer #368B asked to move III.H to IV.A)
7. Results: RQ1 attribution & onset; RQ2 specificity; RQ3 adaptive; RQ4 assumptions; RQ5 baselines; RQ6 ablation
8. Robust aggregation versus attribution (archived Multi-Krum + clean-MK honest alarm analysis)
9. Discussion, limitations (single dataset, K, calibration masking, partial participation semantics, logging
   privacy), ethics
10. Conclusion; Appendix: reproduction (Stage B), per-seed tables, sensitivity, online/offline equivalence

## 9. Claims already supported (archive; see ledger for IDs and exact status)

- On archived FedAvg K=5 runs (EXP_016–019, seeds 123/2026; plus seed 42 EXP_014/015), frozen v3 attributes
  Gaussian and directional geometry perturbations at onset 20 with confirmation round 22 (L-01..L-04).
- Generalisation to attacker client_004 (EXP_026) and onset 30 (EXP_027) in single runs (L-05).
- Multi-Krum rejects the Gaussian attacker every post-attack round but never the directional attacker, at two
  seeds; its rejection proxy implicates the largest-partition honest client (L-07, L-08).
- Online FedTrace (causal wrapper) gives the same alarms/attributions as offline v3 on all 21 archived traces
  (CLAUDE-REPORTED, L-12).

## 10. Claims that remain hypothetical (need the experiment matrix)

H1–H5; clean-run specificity under FedAvg K=5 (no archived clean FedAvg K=5 trace exists); any claim about
backdoor or label-flip detection; adaptive-attacker robustness; partial participation; multiple attackers;
any baseline comparison beyond Multi-Krum.

## 10a. Reproduction is not scientific validation

Stage B (B-R0, B-R1) answers only engineering questions: does the current code revision reproduce the archived
EXP_016 run, and does update logging leave training unchanged? A bit-identical reproduction re-establishes the
*provenance* of one archived Class A stress-test run; it adds **no** new scientific evidence (same seed, same
attack, same outcome by construction). Scientific validation comes only from the validation-seed batch
(X1–X3, sealed pre-registration). The paper reports Stage B in the appendix (A1) as a reproducibility check and
never counts R0/R1 as additional seeds.

## 10b. Direct forensic baselines are required

Change detectors (z-score, CUSUM, Page–Hinkley) and Multi-Krum are necessary but not sufficient: both reviews ask
for a comparison with published FL forensic/detection methods (FedREDefense, FLForensics; FLDetector; [10]).
Requirement: each must be either (a) run under matched conditions after its mechanics are verified from the
original paper or code, or (b) explicitly listed as "not compared" with the reason. Silent omission or a
from-memory re-implementation is not acceptable. Status: sources not yet obtained (decision P-07).

## 11. Claims that must NOT be made

- "No false positives" without scope — clean FedAvg K=5 specificity is untested, and v3 raised a persistent
  honest alarm in both clean Multi-Krum runs (CLAUDE-REPORTED).
- That Gaussian/directional perturbations are effective poisoning attacks (no harmful objective measured;
  the "ρ=0.5 is harmless" figure is withdrawn).
- That the constrained attacker is evasive or that FedTrace is robust to adaptive attackers in general.
- Any statement about FLDetector, FedREDefense, FLForensics or [10] beyond their bibliographic identity until
  verified from the original sources.
- "Zero onset error" as a general property — it is the streak-start convention at onset = T_ref in two seeds.
- Bit-identical GPU reproduction of EXP_016 (pending Stage B), or that CPU toy tests establish GPU
  equivalence.
- Statistical significance from two seeds; client-rounds as independent samples.
