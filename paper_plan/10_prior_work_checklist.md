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

## A1. Full-text / code verification — 2026-10-25 (Prompt 26)

Sources read: FLDetector full text (ar5iv HTML of arXiv:2207.09209); FLForensics full text (arXiv HTML 2407.07221v2,
first ~100k characters; the remainder not read); FedREDefense — full text **not readable here** (PMLR PDF URL returned
404; the NSF PAR PDF could not be text-extracted without installing a PDF tool, which is not permitted); its
implementation was read instead (`github.com/xyq7/FedREDefense`, `codes/run_ours.py`). No third-party code was run.

| Item | FLDetector (KDD 2022) | FLForensics (NeurIPS 2025) | FedREDefense (ICML 2024) |
|---|---|---|---|
| 1 Threat model | m malicious (fake or compromised) clients, default 28 %; untargeted (Fang) and backdoor (Scaling, DBA, ALIE); adaptive attacks (Sec. 3, 5.1, 4.5) | targeted poisoning only (Scaling, ALIE backdoors; Edge triggerless); default 20 % malicious, 10–40 % varied; untargeted left to future work (Sec. 3, 5.1) | code implements label flip, Fang, MPAF, Min-Max, Min-Sum, Scaling, DBA (`run_ours.py`, attack selection); paper threat model UNKNOWN (text unread) |
| 2 Online / post-hoc | online during training; flagged clients removed, training restarted (Sec. 3, 4.2, Alg. 3) | post hoc, after a misclassified target input is identified in a deployed model (Sec. 1, 3) | online each round (code: detection inside the training loop) |
| 3 Evidence | per-round L2 distance between a client's received update and a prediction from history: ĝ = g^{t−1} + Ĥ(w_t − w_{t−1}), Ĥ by L-BFGS over the last N rounds; score = mean over N rounds (Sec. 4.1–4.2, Eq. 3) | 2-D influence score per client: loss gradient on the target input (and on a non-target input) times the client's stored updates at check-points (Sec. 4.2, Alg. 1) | per-client reconstruction loss from per-client synthetic (distilled) data (`synthesize_single`, trajectory-style inferred from variable names); paper details UNKNOWN |
| 4 Cross-client reference | yes: distances L1-normalised across clients each round; gap statistic + k-means (k = 2) on scores; higher-mean cluster flagged (Sec. 4.2, Alg. 2) | yes: HDBSCAN clustering of the 2-D scores; data-driven ratio threshold pooled across potentially malicious clients (Sec. 4.3) | partly: fixed threshold 0.6 on each client's loss, then a polarity check comparing mean losses of the two predicted groups (`threshold_detection`, `run_ours.py`) |
| 5 Non-IID | evaluated with Fang et al.'s non-IID degree 0.1–1.0, default 0.5; theory assumes IID (Sec. 5.1–5.2, Thm. 1) | non-IID via [16]; non-target score separates benign clients rich in the target label (Sec. 4.2, 5.4) | code: Dirichlet `alpha`; paper claims robustness to high non-IID (abstract); details UNKNOWN |
| 6 Individual clients | yes, per-client labels; FPR/FNR reported (Table 2); all benign if nothing detected (Sec. 5.1); no attacker-free detection experiment reported | yes, set of malicious clients; attacker-free case not reported in the read text | yes, per-client flags; flagged clients excluded from aggregation and later selection (`run_ours.py`) |
| 7 Onset | no: "ignorant of when the attack starts or ends"; detection starts at a preset round (50) (Sec. 5.1–5.2) | no (not addressed in the read text) | UNKNOWN (none in the code read) |
| 8 Calibration | N = 10, B = 20, K = 10, start round 50; no fixed score threshold (Sec. 5.1) | check-point frequency (10 rounds), non-target input, HDBSCAN settings, α = 0.2; thresholds data-driven (Sec. 4–5, 7) | detection threshold 0.6; synthesis hyper-parameters from config (`run_ours.py`) |
| 9 Cost | storage O(Np); time O(N³ + KBn² + (6N+2n)p + Nn) per round; no client overhead (Sec. 4.3) | storage linear in model size × check-points × clients; no runtime numbers in the read text | per-client data distillation each round (UNKNOWN magnitude) |
| 10 Fair direct comparator? | partially: online, per-client, consistency-based — the closest to FedTrace's evidence; but it removes clients (changes training), uses cross-client normalisation and clustering, needs many clients (100–300) and does not estimate onset; a monitor-only adaptation with K = 5 would be our modification, not the published method | no for online attribution/onset (post hoc, query-driven, targeted only); yes as a post-hoc forensic reference on BadNets if its stored-artefact requirements are met | partially: per-client, online, mostly reference-free; requires distillation per client; text unverified |

Setting mismatch to report: all three evaluate tens to hundreds of clients (FLDetector: 100 / 300; FLForensics: 100),
while our development setting has K = 5 with full participation.

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
