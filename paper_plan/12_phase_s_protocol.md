# 12 — Phase S Protocol: Standard Baseline Evaluation (CPU-only, exploratory)

Status: **FROZEN 2026-10-25, before any Phase S comparison was computed.** Changes only by dated amendment (§9).
All results are CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. No method is adopted on the basis of Phase S.
Frozen FedTrace v3 is evaluated exactly as implemented (`src/analysis/fedtrace_detector.py`, online wrapper
`src/rev/online.py`). Implementation: `src/rev/phase_s.py` (read-only harness over existing baseline code).

## 1. Data

Six development runs (seeds 123 and 2026; development only): `X1_clean_d{s}`, `X2_badnets_d{s}`, `X3_relabel_d{s}`.
Inputs are the sealed traces (`client_round_trace.csv`: per client and round, update norm and self-cosine).
True attacker `client_002`, true onset 20 — evaluation only. Sealed confirmation seeds are not used.

## 2. Features (per client k, round r; all computable online at round r)

| ID | Feature | Definition |
|---|---|---|
| m | update norm | ‖Δ_k^r‖₂ (trace `update_l2_norm`) |
| t | raw self-cosine | cos(Δ_k^r, Δ_k^{prev}) (trace `self_prev_round_cosine`; undefined at round 0) |
| dA | residual, inclusive median | median over all clients of round r of t, minus t_k^r (frozen v3's reference) |
| dB | residual, leave-one-out median | median over the other clients of round r, minus t_k^r |
| dC | residual, trimmed mean | mean of round r after removing the minimum and maximum (all clients), minus t_k^r |
| dD | residual, historical reference | k's own mean of t over rounds 1–19, minus t_k^r |

Standardisation (within run, causal): each client's feature is standardised by its own sample mean and sample sd over
rounds < 20 (rounds 1–19 where round 0 is undefined). Scores are produced for rounds ≥ 20 only.

## 3. Methods

**Group V (fixed thresholds, frozen decision rule).** v3's score max(z_m, max(0, z_d)) with τ = 3, P = 3, where z_m is
v3's own two-sided norm score and z_d is the one-sided standardised residual with reference A (= frozen v3), B, C or D
(`src/rev/reference_audit.py`). V-A is frozen v3.

**Group Q (standard sequential detectors, calibrated thresholds).** For each feature f ∈ {m, t, dA, dB, dC, dD}:
- ZP — two-sided z-score with persistence: alarm at the 3rd consecutive round with |z| > h; onset = streak start;
- CUSUM — two-sided (upper on z and on −z), reference k = 0.5, alarm when a side exceeds h; onset = start of the
  current positive excursion;
- PH — two-sided Page–Hinkley on the standardised series from round 20 on, δ = 0.5, alarm when a side exceeds λ;
  onset = round after the minimum of the cumulative statistic.
All use `src/rev/baselines.py` unchanged. A Q-method uses one feature only; no hybrids.

## 4. Threshold calibration (Group Q only)

Category 1 — retrospective exploratory calibration: for each (detector, feature), the smallest value on the fixed
grid {0.25, 0.50, …, 60.0} with **zero client-level alarms over both clean development runs** (5 clients × rounds
20–49 each). Uses only clean development runs; never attacker identity, onset, attacked-run scores, counterfactuals,
ASR or test data. If no grid value gives zero alarms, the method is NOT CALIBRATABLE.
Category 2 — causal online calibration: the per-client standardisation over rounds < 20 within each run (§2).
Category 3 — fixed before the experiment: τ = 3, P = 3 (Group V and ZP persistence), k = 0.5, δ = 0.5, the grid.
Limitations (stated before results): with two clean runs (10 client-series), the calibrated threshold is the maximum
clean statistic, so clean-run alarms of Group Q are zero by construction and say nothing about false-positive rates;
zero observed false alarms is never a false-positive guarantee.

## 5. Decision, attribution and onset rules

Alarm = first confirmation per client (rounds ≥ 20). First alarmed client = earliest alarm round; ties broken by the
larger detection statistic at the alarm round (Group V: mean score over the confirming streak up to the alarm, as in
the online wrapper), then client id. Final attributed client = first alarmed client. Actionable round = alarm + 1
(post-aggregation monitoring). Delay = attacker's alarm − 20 (missed → censored at 29). Onset error = |estimated onset
− 20| for the attacker's alarm.

## 6. Metrics

Primary: (a) correct attribution — the first alarmed client is the attacker; (b) honest false accusations — number and
identity of honest clients alarmed in the run. Secondary: attacker detected at all (alarm ≥ 20, whether or not first);
first alarm round; actionable round; delay; onset estimate and error. Safety: clean runs — number of alarmed clients,
earliest alarm, per-client confirmation (Group V only informative; Group Q zero by construction). Detection and
attribution are reported separately: detecting the attacker after an honest client is blamed is **not** a correct
attribution.

## 7. Undefined statistics

A client-feature with fewer than two calibration values or zero calibration sd produces no score and cannot alarm; it
is reported as UNDEFINED with the client and feature. Residual references that are undefined in a round (fewer than
2 values; fewer than 3 for C) give no value for that round.

## 8. Reference-contamination audit and not-evaluable items

For references A–D on the four attacked runs: attacker-induced movement (median/reference recomputed with only the
attacker's self-cosine replaced by its clean-counterfactual value — DIAGNOSIS ONLY), honest standardised residual
change, attacker evidence (raw t change; standardised residual change), and the resulting decisions of §3.
NOT EVALUABLE: FLDetector, FLForensics, FedREDefense (definitions not verified from original sources); any detector
needing data not in the traces/logs (e.g. server-side clean validation data).

## 9. Amendments

(none)
