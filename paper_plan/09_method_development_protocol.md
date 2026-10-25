# 09 — Method-Development Protocol (P-24) and Relabel-Only Control Specification

Status: written 2026-10-25 (Prompt 23); **no run in this document is approved for GPU execution**. Rules marked
FIXED may change only by a dated amendment (§8), never after the corresponding results exist. Seeds 123 and 2026 are
DEVELOPMENT seeds and are permanently excluded from independent validation.

## 1. Evidence so far (development only; ledger L-26–L-48)

Descriptive facts on two development seeds: the BadNets attacker's update norm falls (≈ 0.80× counterfactual); its
temporal self-consistency rises (raw self-cosine +0.12; Observation D rule SUPPORTED on seed 2026); calibration
spread is inflated by the first two rounds; with K = 5 the inclusive same-round median moved by the attacker produced
two honest false accusations on seed 2026 (all of the honest residual increase attributable to the attacker's
movement of the median); no standard reference alternative avoided false alarms on both seeds.

## 2. Revised hypothesis (replaces the rank "if and only if" rule; Task 2)

**Descriptive mechanism (H-MECH).** For honest client k with calibration scale σ_k, the post-onset change of its
signed residual is the sum of an attacker-induced reference movement M_k^r = (ref_attacked − ref with the attacker's
self-cosine replaced by its counterfactual value)/σ_k and its own behavioural change O_k^r. In K = 5 data-poisoning
runs the hypothesis states:

- (P1) honest own-behaviour terms are small: |mean O_k| < 0.5 z over rounds onset … onset+9 for every honest client;
- (P2) an honest client is falsely confirmed by frozen v3 (via z_t_res) in a run **only if** its attacker-induced term
  exceeds its pre-onset margin, M_k^r > τ − z̄_k^pre, in ≥ P consecutive rounds, where z̄_k^pre is its mean signed
  residual over rounds onset−5 … onset−1;
- (P3) the size of M is explained jointly — not deterministically — by the attacker's pre-onset position relative to
  the median (influence), the gap between the median and the neighbouring order statistic (client heterogeneity), the
  size of the attacker's self-cosine change, and σ_k (calibration uncertainty).

Falsification: P1 fails if any honest |mean O_k| ≥ 0.5 z; P2 fails if an honest client is confirmed via z_t_res
without the margin condition holding, or the condition holds for P rounds without confirmation; P3 is evaluated
descriptively across all attacked development and validation runs (no single variable is required to be decisive).

H-MECH is a **descriptive failure mechanism** (partly an accounting identity plus the empirical claims P1–P2). It is
not an algorithmic contribution. An algorithmic contribution would be a reference construction or decision rule
that bounds attacker influence with few clients while preserving correction for shared drift, shown on independent
validation — not established; the standard alternatives traded one failure for another (L-47).

Counterexamples to watch: honest accusations through z_m (benign drift; clean Multi-Krum, L-11); an attacker whose
consistency falls (moves the median down); honest clients whose own behaviour changes because the global model
shifts; several attackers moving the reference jointly; larger K (smaller order-statistic gaps per client); partial
participation (round-dependent reference set).

## 3. Relabel-only control (Task 3)

**Question.** Is the attacker's consistency rise / norm drop caused by trigger-specific backdoor optimisation (A) or by
relabelling to a target class (B)?

| Item | Specification (FIXED) |
|---|---|
| Runs | `X3_relabel_d123` and `X3_relabel_d2026` (`configs/rev/X3_relabel_d{123,2026}.rev.yaml`) |
| Transformation | from round 20, the selected samples of `client_002` are returned with label 0 and the **unchanged image** — the trigger is entirely absent (`BackdoorPoisonedDataset(apply_trigger=False)`) |
| Samples relabelled | exactly the BadNets set: 50 % of the attacker's non-target-class samples, selected with the same key (`derive_attack_seed(seed, client_002, "backdoor")`), hence identical indices to `X2_badnets_d{seed}` (tested); 3,601 (seed 123) / ≈ 4,586 (seed 2026; 50 % of 9,171) |
| Target class, fraction, attacker, onset | 0; 0.5; `client_002`; round 20 — unchanged |
| Model, optimiser, schedule | ResNet-18 (CIFAR), SGD lr 0.01, momentum 0.9, wd 5e-4, batch 32, 1 local epoch, 50 rounds, AMP — unchanged |
| Partition, participation | seed-specific Dirichlet α = 0.1, K = 5, full participation — unchanged base configs |
| RNG isolation | as BadNets: selection by a private generator; base sample read once per access; no global RNG use (tests) |
| Matched controls | the existing `X1_clean_d{seed}` runs. Comparability: since their commits (`bb995c1`, `a7290fc`) the only change on the training path is the opt-in `apply_trigger` switch (clean arm: no wrapper; BadNets arm unchanged); NDSS, hook, logger and runner code unchanged. **Required after the run:** the matched-prefix check (`src.rev.pilot`) — rounds < 20 bit-identical to the clean control; otherwise the comparison is INVALID and the clean control must be rerun at the same commit |
| Outputs | trace, sealed update log (301 records), checkpoints, `rev_attack_records.json`; pair analysis vs clean (per-round clean top-1, untriggered target rate, triggered ASR); diagnostics |
| Disk / GPU | ≈ 13.1 GiB per run (26.2 GiB); ≈ 1.2–1.4 GPU-h per run (G1 runs: 1 h 11–12 min) + ≈ 10 min analysis per pair; the existing BadNets/clean pairs must also be re-analysed once to add the untriggered-target-rate column (≈ 10 min each). Total ≈ 3.0–3.5 GPU-h |

**Why relabel-only is a malicious control, not a benign drift control.** It deliberately falsifies labels toward an
attacker-chosen class (dirty-label poisoning with a targeted objective: biasing predictions toward class 0); benign
non-IID drift involves correctly labelled data. Benign drift controls are the clean runs (and, if approved later, an
honest client whose correctly labelled data distribution changes).

**Primary endpoints (FIXED)**, per seed, each arm X ∈ {BadNets B, relabel R} against the matched clean run C:
1. raw attacker self-cosine change Δt_X = mean over rounds 21–29 of (t_X − t_C) for `client_002`;
2. attacker signed residual change (mean over rounds 21–29, inclusive median, v3 calibration);
3. attacker norm ratio X/C (mean over rounds 25–35);
4. shared reference movement: attacker-induced term M_k (mean rounds 20–29) for each honest client;
5. honest false alarms (frozen v3, online);
6. correct attribution (frozen v3, online);
7. clean top-1 (final and per round);
8. target-class effects: untriggered target rate (non-target test images predicted as 0) and triggered ASR.

**Decision rule (FIXED; B-arm values already known: Δt_B = +0.120 / +0.115):** with ρ = Δt_R / Δt_B per seed,
- *relabelling sufficient* if ρ ≥ 0.75 in both seeds;
- *trigger-specific* if ρ ≤ 0.25 in both seeds;
- *mixed* if 0.25 < ρ < 0.75 in both seeds; otherwise *inconsistent across seeds*.
The same rule is applied to the norm effect with (1 − ratio_R)/(1 − ratio_B).
Manipulation checks (reported, not used to change parameters): R's final triggered ASR − clean ASR < 0.10 (trigger
absent); R's untriggered target rate > clean's (relabelling effective). If a manipulation check fails, the run is
reported as failed control, not re-parameterised.

## 4. Method-development phases (Task 4)

| Phase | Content | Data | Exit gate |
|---|---|---|---|
| M — mechanism discovery | relabel-only control; H-MECH predictions P1–P3 on all attacked development runs | development seeds only | descriptive report; no method choice |
| S — standard baselines | inclusive median (v3), leave-one-out median, trimmed mean, historical client reference; client-wise z-score + persistence; two-sided CUSUM and Page–Hinkley on norm and on each residual; Multi-Krum rejection proxy (archived) | development | all implemented causally; thresholds per §5 |
| C — candidate development | a candidate is written down (definition, inputs, causality argument) **before** it is evaluated; it must follow from Phase M evidence | development, then development-confirmation seeds | adoption rule §6 |
| V — independent validation | frozen method, thresholds, analysis code; ≥ 5 fresh sealed seeds | never used before | report every pre-registered endpoint, including failures |

Causality requirement (all phases): a detector decision at round r uses only submissions, global states and the
client's own history up to r. The matched clean counterfactual, the true onset and the malicious identity are
evaluation-only and may never be detector inputs (enforced by code review and tests of every method module).

## 5. Threshold calibration (FIXED)

Frozen v3 keeps τ = 3, P = 3, T_ref = 20. Any method with a free threshold uses: the smallest value on a grid fixed in
advance with **zero client-level alarms over all clean development runs** (currently `X1_clean_d123`,
`X1_clean_d2026`; future clean development runs are added before calibration, never after seeing attacked results).
Reference values (CUSUM k, Page–Hinkley δ) are fixed a priori (0.5). A threshold calibrated on fewer than three clean
runs is flagged as unreliable (L-35 showed a degenerate threshold from one run).

## 6. Selection safeguards and adoption rule (FIXED)

- No method is selected because it performs best on seeds 123 and 2026: candidates are compared on development
  seeds, then **confirmed on ≥ 2 additional development-confirmation seeds** (new, never used for design) before
  validation; a candidate that fails confirmation is dropped, not tuned.
- Adoption requires, on all development and confirmation runs: no increase in honest false accusations versus v3, no
  loss of correct attribution, and no increase in median delay; and an improvement on at least one of these.
- All compared methods and their results are reported, including those not adopted.

## 7. Robustness tiers (planned; each requires separate approval)

benign non-IID drift (clean runs at each K; clean Multi-Krum archived; optional benign data-shift client); attack
types (BadNets, relabel-only, targeted label flip; onset 20 and 30 — onset ≠ T_ref); larger K (K = 10 full
participation); partial participation (K = 10, 5 per round); multiple attackers (2 colluding BadNets clients). Each
tier ≥ 3 seeds before any claim about that factor.

**Minimum evidence for a methodological novelty claim:** (i) the prior-work checklist (`10_prior_work_checklist.md`)
completed from the original papers/code, showing the capability is not already provided; (ii) on independent
validation the candidate improves the primary endpoint (honest false accusations) over the strongest standard
alternative of §4 without loss of attribution or delay, with Clopper–Pearson intervals; (iii) the effect holds for ≥ 2
attack types and both onsets; (iv) benign drift controls show no increase in false alarms; (v) H-MECH predictions hold
on validation. Otherwise the contribution is reported as an empirical characterisation, not a new method.

## 8. Amendments

### Amendment 1 — 2026-10-25 (Prompt 24; written before any relabel-only run). The original text above is unchanged.

**A. Triggered ASR in relabel-only runs** is a manipulation / behavioural check, not a validity criterion. The
§3 sentence "If a manipulation check fails, the run is reported as failed control" is superseded: an unexpected ASR
or untriggered-target-rate result is reported as an observation; the run is neither retuned nor discarded.

**B. ρ is descriptive, not causal proof.** The original thresholds (0.25, 0.75) and classifications (relabelling
sufficient / trigger-specific / mixed / inconsistent) are kept and reported, each with a cautious reading
("consistent with …"). Handling, per seed and per effect (implemented and tested in `src/rev/relabel_compare.py`
before the runs; ε = 0.02, i.e. five times the honest clients' self-cosine change of ≈ 0.004 observed on both seeds):
- missing or undefined measurement (incomplete window) → UNDEFINED;
- near-zero denominator |BadNets effect| < ε → UNDEFINED;
- opposite sign (|relabel effect| ≥ ε, sign differs) → counted as ≤ 0.25 by the original rule, flagged OPPOSITE_SIGN
  and read as "relabel-only moves the measure the other way", not as evidence of a trigger effect;
- |relabel effect| < ε → counted as ≤ 0.25, flagged NO_DETECTABLE_RELABEL_EFFECT;
- if any seed is UNDEFINED, the cross-seed classification is UNDEFINED.

**C. H-MECH: algebra versus causation.** The decomposition Δz = M + O (reference term plus own-behaviour term, both
relative to the matched clean run) is an exact identity given the calibration scale; it does not by itself show that
the attacker caused the honest score change, because the clean run differs from the attacked run in every later
global model. The attacker-induced term M (median recomputed with only the attacker's self-cosine replaced by its
counterfactual value) is a counterfactual attribution within the attacked run, valid under the assumption that the
other clients' self-cosines are those observed. Under the complete frozen v3 rule, reference movement produces a
**confirmed honest false alarm** only if, for that client, max(z_m, max(0, z_pre + M + O)) > τ in P = 3 consecutive
scored rounds (rounds ≥ T_ref); it produces a **false attribution** only if, in addition, that confirmation occurs no
later than every other confirmation, with ties broken by the higher mean score over the confirming streak (then by
client id). Because z_t_res is one-sided, downward reference movement can never create a z_t_res alarm.

**D. Metrics for any future method evaluation.** Primary: correct attribution of the malicious client; honest-client
false accusations (per run). Secondary: alarm delay; onset error; robustness across attack types, onsets, K and
participation. Safety: confirmed alarms in clean runs and in benign-drift controls. Frozen v3 and its thresholds are
unchanged by this amendment.

**E. Confirmation seeds** sealed before the relabel-only runs: see `11_seed_registry.md`.
