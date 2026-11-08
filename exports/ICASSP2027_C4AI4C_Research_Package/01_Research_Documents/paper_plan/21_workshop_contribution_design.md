# 21 — Workshop Contribution Design (C4AI4C'27)

> **Review note — 2026-10-26 (Prompt 33, P-49).** PI decisions: P-51 analysis-paper framing APPROVED; P-52 Option A CONDITIONALLY APPROVED — its formal statement and frozen CPU test are doc 25 (the clip identity is elementary; the contribution is the forensic chain and its test, not the identity). Option B not pursued.

Prepared 2026-10-26 (Prompt 32). Labels: **CLAUDE-REPORTED** (evidence exists), **PROPOSED** (plan), **UNVERIFIED**.
Evidence references: ledger rows L-xx, docs 15 / 18 / 19 / 20.

## 1. Task 5 — Does the original FedTrace remain useful?

| Component | Technically meaningful? | Supported claim | Needs qualification | Overlap with established methods | Keep? |
|---|---|---|---|---|---|
| Problem formulation: online, monitor-only attribution **and onset localisation** in few-client cross-silo FL | yes | the task definition itself; onset is not addressed by FLDetector, FedREDefense, FLForensics (§A1/§A2) | FLTracer reports an attack time (first flagged round) — onset is not unique to FedTrace | FLTracer (attribution + time) | **yes** (framing) |
| Magnitude evidence z_m (two-sided norm z vs own calibration) | yes | detects Gaussian perturbation (archive), BadNets and relabel-only norm drops (via ZP baseline) | non-specific: benign drift produces the same drop (L-65) | standard SPC (Shewhart z with run rule) | as a component, not as novelty |
| Temporal residual z_t_res (self-cosine vs same-round inclusive median, one-sided) | yes — the archive shows it is what catches the norm-preserving directional attack (Table 3) | directional perturbations (6 runs, archive) | one-sided: misses the attacker whose self-consistency **rises** (all poisoning arms, L-32); inclusive median → honest false accusations under upward crossing (L-43, L-51, L-64) | self-consistency over time ≈ FLDetector's history consistency; median reference = robust statistics | keep as the **object of the reliability analysis** |
| Peer evidence z_p (v1) | no (never contributed, Table 3) | – | – | – | drop (mention in one line) |
| Calibration (rounds < 20) + τ = 3, P = 3 | yes | E4: P ≥ 3 preserves archive conclusions | first rounds inflate sd 1.8–6.3× (L-31, L-40); self-inclusive offline calibration (L-13/L-14); effective n ≈ 2 (L-61) | SPC Phase I / II | keep, disclose |
| Onset = streak start | yes | archive onset error 0 (12 of 13 runs at onset = T_ref) | favoured by onset = T_ref (L-34); errors 8 (123 BadNets) / 21 (E2 ρ 0.5) | change-time estimation is standard | keep with caveat |

**Strongest evidence:** archive perturbation results recomputed exactly (21 traces), the matched-counterfactual
reproduction and logging non-interference (L-20/L-21), the matched development arms with bit-identical prefixes, and
the contamination mechanism (L-43–L-45). **Materially constraining failures:** v3 attribution 1/4 on data poisoning;
relabel-only missed in both seeds; honest accusations under poisoning and benign drift at seed 2026; plain norm
detectors 4/4 on poisoning; late honest alarms under Multi-Krum. Old results stay valid for their own attack family;
they cannot be used to imply poisoning or drift performance.

## 2. Task 6 — Candidate three-contribution structure

| Candidate | Assessment |
|---|---|
| C1 "FedTrace forensic framework" (temporal + peer-relative evidence for attribution and onset) | **Not defensible as a novel method.** Its components are standard (SPC z-scores, persistence, median reference); FLTracer already does online per-client cross-round attribution with an attack time; FLDetector already uses per-client temporal consistency; on data poisoning v3 is outperformed by a plain norm z-score. Usable as the **case-study framework** under analysis, not as the contribution |
| C2 Controlled matched evaluation (clean / BadNets / relabel-only / benign drift; separate endpoints) | **Defensible.** Bit-identical pre-onset prefixes (L-26, L-36, L-49, L-63) make every post-onset difference attributable to the intervention; endpoints separate detection, attribution, false accusation, effectiveness and onset. Novelty modest (evaluation methodology), value high for a workshop |
| C3 Forensic reliability under benign drift (false accusations, contamination, temporal dependence, detection ≠ attribution) | **Defensible and the core finding**, with the caveat of two development seeds unless confirmatory runs are added (doc 22) |

**Coherence:** C2 + C3 form one paper; C1 does not fit as stated. **Recommended narrower framing (PROPOSED):**
*an analysis paper on when client-level attack attribution in few-client FL is reliable* — FedTrace and standard
detectors are the instruments; the contributions are (i) a formal separation of detection, source attribution,
malice and onset with an order-statistic characterisation of reference-induced false attribution (Option A below),
(ii) the matched-counterfactual evaluation protocol, (iii) the empirical reliability findings, including the negative
ones. FedTrace's original results enter as the perturbation-attack evidence (where its temporal residual is what
works), not as a superiority claim.

## 3. Task 7 — Feasible technical strengthening (at most three)

### Option A — Order-statistic characterisation of reference-induced false attribution (RECOMMENDED)

1. **Contribution.** A proposition, with proof, stating when a same-round order-statistic reference (inclusive
   median of K self-consistency values; also leave-one-out and trimmed variants) transfers one client's change into
   other clients' residuals: for the inclusive median the reference changes only if the source **crosses the median
   rank**; the shift is bounded by the adjacent order-statistic gaps (e.g. for an upward crossing from rank m to above,
   Δ = a_(m+1) − a_(m) evaluated in that round); an honest client k gains Δ / σ_k in z units, so a one-sided residual
   detector confirms k iff the shifted score stays ≥ τ for P rounds. Corollaries: (a) upward crossing + small σ_k ⇒
   honest false accusation; (b) downward crossing or no crossing ⇒ no contamination-driven honest accusation;
   (c) leave-one-out / trimmed references are contaminated or not depending on the source's rank (L-59); (d) combined
   with Lemma 2 (precedence), contamination-driven accusations can never precede the source's own change.
2. **Nontrivial because** it turns an observed failure (one seed) into a deterministic, checkable condition that
   predicts *which* honest clients will be accused and *when*, and explains why the same detector is correct at seed
   123 and wrong at seed 2026, and why the archived perturbation runs had no honest accusations.
3. **Relationship to FedTrace.** Explains v3's z_t_res behaviour; no change to v3.
4. **Difference from prior work.** Robust statistics studies breakdown of estimators, not forensic false attribution;
   FLTracer / FLDetector / DeepSight use median / MAD or clustering without analysing reference-induced accusations
   (in the parts read; checklist §A1–§A5). Novelty claim limited to the characterisation + its forensic use.
5. **Assumptions.** Synchronous rounds, full participation, one source, inclusive same-round order-statistic
   reference, per-client calibration, P1–P6 of doc 15 A6.
6. **Existing evidence.** Mechanism observed at seed 2026 (L-43–L-45), its absence at seed 123 (L-54), rank
   dependence of alternatives (L-59). These motivated the proposition and **cannot test it**.
7. **Smallest falsification test (CPU, preregistered before computing).** Compute, on traces only, the crossing
   indicators and predicted shifts for (i) all 21 archived traces (not yet examined for crossings) and (ii) any new
   confirmatory runs (doc 22); falsified if an inclusive-median v3 honest t_res confirmation occurs with no upward
   crossing in the preceding P rounds, or if the predicted shifted score ≥ τ for P rounds without a confirmation.
8. **Complexity.** Low: a read-only CPU script (~1 day with tests) + a half-page proof.
9. **GPU.** 0 (benefits from the confirmatory runs of doc 22, which are needed anyway).
10. **Risk.** Moderate-low: the algebra is deterministic; the risk is that the prediction is trivial for archived
    runs (no crossings ⇒ vacuous) — then only new runs test it.
11. **Fit.** Yes: one proposition, one figure, ≈ 0.6 page.

### Option B — Precedence-consistency flag for attributions

1. Flag an attribution as unreliable when the attributed client's own-history evidence changed **after** another
   client's (Lemma 2). 2. Nontrivial: uses an exact property of synchronous FedAvg. 3. Post-processing of any
   detector's alarms. 4. Not found in prior work read. 5. P1–P6; per-round detectability. 6. Partially: seed-2026
   honest accusations (r23) preceded the attacker's own confirmation (r24) — the flag would fire, but this was seen
   before design. 7. Freeze the rule, test on new runs only. 8. Low-medium. 9. Needs new runs (doc 22). 10. **High**:
   change-time estimates are off by 0–8 rounds while the precedence margin is one round (doc 15 A7) — resolution likely
   insufficient. 11. Fits, but it is a new decision rule designed after seeing two seeds. **Not recommended now.**

### Option C — Matched-counterfactual evaluation protocol as a stated contribution

1. Formalise the protocol (bit-identical prefix check; clean / poisoning / no-trigger relabel / benign-drift arms
   from the same partition and initialisation; separate endpoints; preregistration). 2. Nontrivial in rigour, modest
   in novelty. 3. Already built (`src/rev/pilot.py`, `update_log.py`, protocols 08–16). 4. Prior FL defences report
   aggregate TPR / FPR without matched counterfactuals (in the papers read). 5. Deterministic training; logging.
   6. Yes (all prefix checks VALID). 7. n/a (methodology). 8. none. 9. 0. 10. low. 11. Yes — but it is already part
   of C2, not an additional algorithmic contribution.

**Recommendation:** Option A only, as an analytical (not algorithmic) strengthening, conditional on PI approval and
on preregistering its falsification test before computing any crossing statistic. No new detector is recommended.
D1 and the label-shift correction are not recycled.
