# 10 — Prior-Work Comparison Checklist (Task 5)

Status: 2026-10-25. **Every cell marked TO VERIFY must be filled from the original paper or official code before any
comparative statement is written.** No capability is asserted to be absent without that verification. Bibliographic
identity verified so far (from the submission's `.bib`): FedREDefense (Xie, Fang, Gong; ICML 2024, PMLR 235,
54460–54474); FLForensics (Jia, Fang, Liu, Zhang, Gong; NeurIPS 2025, vol. 38, 116470–116497,
doi 10.52202/085713-3882). FLDetector: no bibliography entry — identity itself TO VERIFY. Reference [10] of the
NDSS reviews: unidentified.

## A0. Identity and abstract-level scope — verified 2026-10-25 from primary landing pages (full texts NOT yet read)

| Method | Verified identity (source) | Abstract-level scope (verified) | Category | Still TO VERIFY |
|---|---|---|---|---|
| FLDetector | Zhang, Cao, Jia, Gong. "FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients." KDD 2022 (Research Track); arXiv:2207.09209 (https://arxiv.org/abs/2207.09209v3) | detects malicious clients by checking the **consistency of each client's model updates across iterations**; the server predicts a client's update from its history (Cauchy mean value theorem, L-BFGS) and flags inconsistency; detected clients are removed before Byzantine-robust aggregation; abstract does not mention non-IID data or attack onset | training-time detection + removal (defence) | inputs, update-prediction details, decision rule, evaluation setup (K, participation, non-IID), adaptive attacks, code (a GitHub repository is reported by search results, not opened) |
| FLForensics | Jia, Fang, Liu, Zhang, Gong. "Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning." NeurIPS 2025; arXiv:2407.07221 (https://arxiv.org/abs/2407.07221v2) | "poison-forensics": **post hoc**, after a misclassified target input is identified in a deployed poisoned model, traces back the malicious clients; complements training-phase defences; existing and adaptive attacks on five datasets; abstract does not mention onset estimation | post-hoc forensics | required stored artefacts (client updates/models?), attack types (targeted framing implied, not stated), decision rule, K, code (repository reported by search results, not opened) |
| FedREDefense | Xie, Fang, Gong. "FedREDefense: …using Model Update Reconstruction Error." ICML 2024, PMLR 235:54460–54474 (https://proceedings.mlr.press/v235/xie24c.html) | judges each update **by its own characteristics** (not cross-client information): genuine updates can be approximately reconstructed from distilled local knowledge, poisoned handcrafted updates cannot; filters clients by reconstruction-error discrepancies | training-time filtering (defence) | distillation inputs, decision rule, attack types, K, code (PMLR page lists none) |

Implication (no capability is asserted absent): FLDetector's update-consistency principle is the closest prior work to FedTrace's temporal-consistency evidence; any claim about consistency-based detection must be positioned against it after reading the full paper. FLForensics is the closest forensic (attribution) work but is post hoc and query-driven; FedREDefense avoids cross-client references by design, which is directly relevant to the reference-contamination finding.

## A. Per-method checklist (fill one column per method)

| Question | FLDetector | FLForensics | FedREDefense |
|---|---|---|---|
| Exact citation, version, code URL and licence | TO VERIFY | TO VERIFY (bib verified) | TO VERIFY (bib verified) |
| Threat model: attack types, number of attackers, adaptive attacker considered? | TO VERIFY | TO VERIFY | TO VERIFY |
| Required inputs (per-round client updates? global models? server-held clean data? historical states?) | TO VERIFY | TO VERIFY | TO VERIFY |
| Output: per-round flags, malicious-client set, onset/round of first malicious behaviour, scores? | TO VERIFY | TO VERIFY | TO VERIFY |
| Online (causal) or post-hoc? Detection delay reported? | TO VERIFY | TO VERIFY | TO VERIFY |
| Uses cross-client references (median / mean / clustering)? Attacker inclusion in the reference? | TO VERIFY | TO VERIFY | TO VERIFY |
| Uses client-specific temporal consistency or historical prediction of updates? | TO VERIFY | TO VERIFY | TO VERIFY |
| Handling of non-IID heterogeneity / benign drift; false-positive analysis on clean runs | TO VERIFY | TO VERIFY | TO VERIFY |
| Number of clients and participation in the evaluation (small K?) | TO VERIFY | TO VERIFY | TO VERIFY |
| Datasets / models (vision?) | TO VERIFY | TO VERIFY | TO VERIFY |
| Can it run monitor-only on our logged updates without changing training? | TO VERIFY | TO VERIFY | TO VERIFY |
| Hyper-parameters and how they were set | TO VERIFY | TO VERIFY | TO VERIFY |
| Fair-comparison conditions (what inputs we must give it; which of our endpoints it can produce) | TO DERIVE after verification | TO DERIVE | TO DERIVE |

## B. Standard techniques that may already cover proposed elements

| Proposed element | Standard technique it may coincide with | Status |
|---|---|---|
| Leave-one-out reference | jackknife / leave-one-out statistics; cross-validation style references | standard — not a contribution by itself |
| Trimmed / robust cross-client reference | trimmed mean and coordinate-wise median (robust aggregation literature); Hodges–Lehmann | standard |
| Client-specific robust calibration (burn-in exclusion, median/MAD) | statistical process control Phase I / Phase II; Hampel identifier | standard |
| Two-sided residual monitoring | two-sided CUSUM / Shewhart charts | standard |
| Onset localisation at alarm | CUSUM change-time estimate; maximum-likelihood change point | standard |
| Historical client reference | per-client baselining (v1's temporal score) | already in FedTrace v1 |
| Drift-adjusted magnitude | cross-sectional normalisation by the round median | standard |

Citations for the standard techniques are to be added from verified sources; none is asserted here.

## C. What could be new (only if A and B do not already cover it)

- An empirical characterisation, in federated vision training with few clients, of how a single poisoning client
  contaminates cross-client forensic references and causes honest-client false accusation (H-MECH).
- A reference construction or decision rule that bounds a single client's influence on the reference while still
  removing shared drift, validated under the protocol of `09_method_development_protocol.md`. Not yet proposed.

## D. Strongest counterexamples to the direction

benign consistency increases (correctly labelled data shift, learning-rate changes, late training); adaptive
attackers that hold their pre-onset consistency; multiple colluding attackers; larger K or partial participation
changing the reference set; attacks that lower consistency; false alarms through the magnitude signal (clean
Multi-Krum).
