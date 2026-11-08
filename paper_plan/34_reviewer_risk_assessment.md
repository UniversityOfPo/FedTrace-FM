# 34 — Reviewer Risk Assessment (simulated critical IEEE workshop review)

Prepared 2026-10-27 (Prompt 35). Simulated review of the paper as planned in docs 32, 33 and the LaTeX skeleton.
No remedy below requires a GPU experiment.

## 1. Simulated review (summary)

*Relevance:* data-poisoning forensics in federated learning fits cybersecurity for AI (the 2026 edition of C4AI4C
names data poisoning; the 2027 topic list was not found — doc 17). *Strengths:* matched interventions with
bit-identical histories, preregistered confirmation seeds, explicit negative results, seed-level reporting.
*Weaknesses:* below.

## 2. Five most serious objections and non-GPU remedies

| # | Objection | Why serious | Remedy without new GPU runs |
|---|---|---|---|
| 1 | **Novelty is thin**: the outcome decomposition and the median identity are elementary; FLTracer already attributes clients and reports attack time | could be judged an evaluation note with no new method | Frame explicitly as a reliability evaluation; state what is not new (Relation to Prior Work, mandatory); make the contribution the cross-seed, matched, preregistered evidence that *detection, attribution and drift-silence fail in different ways* (C-01–C-04) and the rank-conditioned contamination route (C-06b); cite FLTracer's own statement that abrupt benign shifts are sacrificed |
| 2 | **Statistical validity**: four seeds, two preregistered; rounds are autocorrelated; no rates or intervals | conclusions could look anecdotal | Report seed-level outcomes only (Table 1); state that no rate is estimated and why (lag-1 autocorrelation 0.82–0.91, cross-client correlation ≈ 0.95, effective calibration n ≈ 2 — L-61); phrase every claim as an existence or seed-level statement (doc 32) |
| 3 | **Attack effectiveness**: two of four BadNets runs below the 50 % ASR criterion; v3's misses could be blamed on weak attacks | undermines the detection result | Report ASR per seed next to each outcome (Table 1); note that v3 missed W1 at 63.3 % ASR (effective) and that ZP / CUSUM detected all four including the weak ones — the miss is a calibration property (L-31), not only an attack-strength artefact; keep the preregistered criterion unchanged |
| 4 | **Baseline fairness**: norm thresholds were calibrated on clean runs of the development seeds; FLTracer / FLDetector not run | comparison could be called biased or incomplete | State the calibration source (L-57) and that the W-seed results are the out-of-sample test of those thresholds (C-04 shows transfer failure); explain why FLTracer's cross-client channels and FLDetector's decision stage are not applicable at K = 5 (doc 27, checklist §A5) and present them qualitatively; no numerical comparison |
| 5 | **Generalisation**: one dataset, model, K = 5, one intervened client, onset at the calibration end, one drift type; mechanism observed once | conclusions may exceed evidence | Limitations paragraph listing each restriction; title / abstract avoid general claims; contamination presented as a conditional explanation with its precondition (pre-onset median rank) observed in 1 of 4 seeds; the identifiability statement (C-13) bounds what any update-only method can claim |

Additional risks to address in text: honest false accusations under benign drift may be read as "drift is just
another attack" — state explicitly that the drift intervention keeps correct labels (protocol 13); reproducibility
— cite the frozen configs, seeds, commit hashes and evidence manifest (doc 31) in a data / code availability
sentence if the venue allows; AI-use disclosure (doc 36).

## 3. Whether conclusions exceed the evidence (checklist)

| Planned conclusion | Within evidence? |
|---|---|
| "detection, attribution and drift-silence fail in different, seed-dependent ways" | yes (C-01–C-04, Table 1) |
| "reference contamination causes false accusations" | only as "can cause … when the intervened client starts inside the band; observed once" (C-06b) |
| "standard norm detectors are better" | no — they attribute BadNets but accuse honest clients under drift and transferred thresholds |
| "FedTrace fails" | only seed-level: missed BadNets on W1 / W2, misattributed at 2026; succeeded on archived perturbations |
