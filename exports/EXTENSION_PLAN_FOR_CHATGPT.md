# FedTrace C4AI4C'27: extension plan, 3–10 November 2026 (share with ChatGPT)

## Status
- **Version 9 is locked and is the fallback submission.** New results go only into a separate Version 10.
- Version 10 replaces Version 9 only if every integrity check passes and the audit is finished by 9 November.
- Upload deadline: 10 November 2026 (official deadline 11 November; time zone unverified).

## Preregistered extension (paper_plan/37_extension_protocol.md, frozen 3 Nov 2026 before any run)

| Block | Runs | Purpose | Cost |
|---|---|---|---|
| A: more seeds (K = 5) | W3 = 1966286922, W4 = 1465380568, W5 = 431235112 × {clean, BadNets, honest drift} | Does the detection/false-accusation trade-off recur on new seeds? (4 → 7 seeds) | 9 runs, ≈ 11 GPU-h |
| B: K = 10 | seeds 123, 2026 × {clean, BadNets, honest drift} | Answers "K = 5 is too small"; tests transfer of the frozen K = 5 thresholds | 6 runs, ≈ 18 GPU-h |
| Later, only if time allows | onset at round 30; CIFAR-10 | removes the "onset = T_ref" limitation; second dataset | not yet approved |

How the new seeds were chosen:
- They continue the same SHA-256 seed rule that produced W1 and W2 (positions 3–5 of that list).
- The sealed seeds C1 and C2 stay sealed.

What stays frozen:
- every detector and every threshold: v3 (τ = 3, P = 3), ZP (h = 1.75), CUSUM (h = 11.75), Page–Hinkley (h = 6.5);
- the arms: client 002 is intervened on from round 20; BadNets uses 50 % poisoning and target class 0; honest drift
  uses a retained fraction of 0.5;
- nothing is recalibrated for K = 10.

## Frozen reporting rule (from the protocol)
- Report seed by seed: no pooled rates and no confidence intervals.
- For each seed, report:
  - F1: did v3 confirm the BadNets attacker?
  - F2: did ZP, CUSUM or Page–Hinkley confirm it?
  - F3: which honest clients were accused in the drift arm?
  - F4: which honest clients were accused in the clean arm?
  - F5: ASR and accuracy.
- The **"trade-off pattern" is present on a seed** only if all three hold:
  1. ZP or CUSUM attributes the attacker first;
  2. ZP or CUSUM accuses an honest client in the drift or clean arm;
  3. v3 has no clean alarm.
- The paper may say "present in n of m seeds" as a count, never as a rate.
- K = 10 is a separate block and is never pooled with K = 5.
- A seed where the pattern is absent is reported with equal prominence.

## How Version 10 would change (only after the results are in)
- Table 1: add rows for W3–W5, or add a compact "extension" table if space is tight.
- New compact table for K = 10 (two seeds × three arms × four detectors).
- Abstract and Results:
  - "four seeds" → "seven seeds";
  - one sentence on K = 10;
  - the Limitations paragraph is updated accordingly.
- No other text changes; Figures 1–3 stay unless a new seed changes what they show.

## Roles
- Claude runs the experiments on the PI's laptop GPU, writes and reviews the code, verifies every number against
  the artifacts, and prepares Version 10.
- ChatGPT reviews the wording only. It must not change numbers or add claims beyond the frozen rule above.
