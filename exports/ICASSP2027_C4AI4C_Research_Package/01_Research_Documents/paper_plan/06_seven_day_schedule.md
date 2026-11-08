# 06 — Seven-Day Schedule

Status: created 2026-10-24 (Day 0 = this planning phase). Days are working days counted from PI approval of the
source checkpoint. Resources: one RTX 5060 (one run at a time), C: drive with ~635 GB free (100 GiB reserve
kept), CPU for analysis, PI time for decisions and writing. Runtimes: measured K=5 FedAvg 1.33 h/run; all
other durations are estimates and must be re-planned after S0/R1 measure logging overhead.

Rule: no stage starts on results that its gate has not accepted; validity beats the deadline.

## Revision 2026-10-24 (CVPR 2027 framing; Stage B done). Supersedes the day table below where they conflict.

Measured basis: 50-round K = 5 run with update logging ≈ 1 h 23 min wall (R1); Stage B consumed day 0.

| Day | GPU (sequential) | GPU-h | CPU (Claude) | PI | Gate |
|---|---|---|---|---|---|
| 1 | P1 pair: X1_clean_d123 → X2_badnets_d123 → pair analysis (`src.rev.pilot`) | ≈ 3.0 | C6 (B) and C7 (C) on archived traces; CUSUM/PH on the pair traces (two-sided where needed); development table D1 | approve P-17 (GPU pair), P-16 (second dataset yes/no), send baseline sources (P-07) | pair valid (prefix bit-identical) and BadNets effective (final ASR ≥ provisional 0.5) |
| 1–2 | P2 pair on seed 2026 (same configs, seed changed) | ≈ 2.8 | candidate decisions under the §0 rule | — | candidates adopted / dropped |
| 2 | seal pre-registration; validation: clean + BadNets × 5 fresh seeds | ≈ 14 | baselines at matched clean FPR (development clean runs only) | seal pre-registration | all validation pairs valid |
| 3 | detector-aware X3 × 5 (if adopted design needs it) or second dataset (if P-16 approved) | ≈ 7 | analysis T3–T6, F2, F3, F-B/F-C | Method + Threat Model text | results frozen |
| 4–6 | reruns only | ≤ 3 | figures, appendix, claim audit | writing (CVPR format) | full draft |
| 7 | — | 0 | final checks | submission decision | — |

If BadNets is ineffective in P1, the next GPU slot goes to one development re-parameterisation (poison fraction),
not to validation seeds.

| Day | GPU queue (sequential) | GPU-h | CPU / engineering (Claude) | PI | Gate at end of day |
|---|---|---|---|---|---|
| 1 | S0 smoke (R = 2, logging) → B-R0 → B-R1 → dev batch: X1-D ×2, X2-D ×2 (overnight) | ≈ 8 | Create S0 configs; fix machine-specific path in `tests/test_rev_preservation.py`; Stage B comparison script (R0 vs archive, R1 vs R0: trace, history w/o timing, partition, final tensors, per-round hashes); run-analysis pipeline (run dir → online v3 + baselines → per-run outcome JSON) | Approve checkpoint commit; send submitted PDF (for [10]) and PDFs/code links for FLForensics, FedREDefense, FLDetector | **Stage B gate** (A1): GO / Level-B-with-cause / BLOCKED |
| 2 | X3-D ×2 → (after sealing) validation batch starts: X1-V ×5, X2-V ×5, X3-V ×5 | ≈ 2.7 + start of 20 | Analyse dev runs (backdoor success, constraint records); C2 ablation and C3 clean-MK analysis on archive; matched-FPR baseline selection on D clean runs | Decide backdoor parameters from D only; **seal pre-registration** (seeds, H1–H5 thresholds, κ, success threshold, Level B tolerance) | Attack effective on D? If not: one more D iteration (+2.7 h), then seal |
| 3 | validation batch continues (ends ≈ evening) → X4 ×5 overnight | ≈ 20 + 6.7 | Verify published baselines from sources (if received) and implement only verified ones (C4); figure scripts F2–F5 on dev data | Write Threat Model + Attacks sections | All V runs COMPLETED + verified (logs sealed, prefixes matched) |
| 4 | X5 (K = 10, 5/round) ×6 → X6 (if engineered) or X7 / X8 | ≈ 8–12 | Full analysis of V: T3–T6, F2–F5, A1–A5; ledger update | Decide X6 go/no-go by morning; review T3/T4 drafts | Results frozen for main tables |
| 5 | optional runs only (X7/X8); GPU otherwise idle | 0–13 | T7 and appendix tables; reproducibility package (configs, run registry, hashes) | Rewrite Introduction, Method (online semantics, masking bound), Results | Draft complete except Discussion |
| 6 | none (reserve for a failed-run rerun) | 0–2.7 | Claim–evidence audit: every result sentence ↔ ledger row; regenerate all assets from sources | Discussion, Limitations, Related Work ([10] if identified); fix B4–B6 | Full draft + appendix |
| 7 | none | 0 | Final consistency checks (numbers in text = tables = sources), anonymisation, page limit, build | Independent read-through; submission checks | Submit / hold decision |

Total planned GPU: ≈ 31 h MUST + ≈ 15–21 h SHOULD; leaves ≈ 1.5 days of GPU buffer for one failed batch.

## Disk plan
Full logs for S0, B-R1, D seeds and one V seed per condition (≈ 10 logged runs ≈ 150 GiB). Other runs keep
traces, checkpoints and per-round state hashes. Re-plan if C4 (update-based baselines) needs more logged runs;
an encrypted external drive is needed before exceeding ≈ 400 GiB of new data.

## Minimal defensible paper (if the full matrix cannot finish)
Stage B reproduction (A1) + clean controls X1 + unconstrained backdoor X2 + constrained backdoor X3 on 5
validation seeds + CPU baselines (z-score/CUSUM/Page–Hinkley at matched FPR) + ablation C2 + archived Class A
stress tests + Multi-Krum analysis including the clean-MK honest alarm (L-11) + analytic calibration-masking
bound. Partial participation, multiple attackers and published forensic baselines are then stated as
limitations / not compared — never implied.

## Stop conditions (do not submit as-is)
- Stage B BLOCKED (unexplained divergence or logging interference) and no time to diagnose — evidence chain broken.
- X2 backdoor ineffective after one development iteration — no realistic-poisoning claim possible.
- X1 shows persistent honest alarms in clean FedAvg and the cause is not understood — specificity claim
  impossible; the paper's framing must change before submission.
