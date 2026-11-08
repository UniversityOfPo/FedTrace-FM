# Prompt for ChatGPT: revision v2 of the C4AI4C'27 paper

You are revising our ICASSP 2027 C4AI4C satellite-workshop paper on server-side forensic monitoring in federated
learning. You already have the original research package (ICASSP2027_C4AI4C_Research_Package.zip) and your previous
draft. I am now attaching **ICASSP2027_revision_v2.zip**, which contains ready-made, artifact-verified figures,
LaTeX tables and an algorithm. Use them instead of the old ones. **Do not change any number.** Every number in these
files was regenerated from the frozen experiment artifacts. If something in your earlier draft disagrees with them,
these files win.

## 0. Hard constraints (do not violate)

- Template: official ICASSP 2027 `spconf.sty` + `IEEEbib.bst`, two columns, letter size.
- Length: at most 4 pages of content. Page 5 may contain only references, the AI-use disclosure / acknowledgements
  and Compliance with Ethical Standards. Review is not blind, so keep author names. Every author needs an ORCID.
- The paper is an **evaluation (reliability audit)** of a frozen monitor. FedTrace v3 is **evaluated, not
  proposed**. No new method, no tuning, no new experiments.
- Forbidden claims:
  - any "outperforms / state of the art / superior" statement;
  - pooled rates or percentages over seeds (report seed by seed only);
  - saying the contamination mechanism is "confirmed" or "general";
  - saying that separating detection / attribution / onset / false accusation is new (FLDetector and FLTracer already
    report per-client errors and timing);
  - any numerical comparison with FLTracer, FLDetector, DeepSight, FLForensics or FedREDefense (none were run);
  - generalising beyond CIFAR-100 / ResNet-18 / K=5 / Dirichlet α=0.1 / 50 rounds.
- Wording: say "honest client confirmed" or "false accusation", never "false positive rate". Say "seeds W1, W2
  (preregistered confirmation seeds)", never "test set".
- Remove every TODO / TODO-VERIFY. Use complete bibliographic entries, with page numbers where they exist. Order the
  references by first citation (IEEEbib).

## 1. New figures (all in `figures/`, PDF for LaTeX)

| File | Placement | What it shows / caption guidance |
|---|---|---|
| `fig_overview.pdf` | `figure*` at the top of page 1 or 2, `width=\textwidth` | **Replaces the old Fig. 1 (A/B split).** One integrated pipeline: five clients (client 002 intervened from round 20 in every arm) → server-side v3 monitor (magnitude channel z^m, temporal/peer channel z^t, fusion S = max, persistence S ≥ 3 for 3 rounds) → forensic verdict (client, onset, confirmation round) → reliability audit (ASR, detected?, correct attribution?, onset error and delay, honest clients accused?). Baselines use the same norm score. Caption: 2–3 sentences explaining the flow and that thresholds were frozen before the confirmation seeds. |
| `fig_reliability_matrix.pdf` | `figure*`, `width=\textwidth` | **Replaces the old tiny outcome grid.** 9 detectors (full v3; 5 ablations of v3; ZP, CUSUM and Page-Hinkley) × 14 runs (BadNets 4 seeds, relabel 2 dev seeds, honest drift 4 seeds, clean 4 seeds). This is the main **baseline + ablation** figure. Caption must define ✓, ✓+FA, W, M, FA and –, and state that `*` = preregistered seed. |
| `fig_calibration.pdf` | `figure*` (or column width if it stays readable) | **New evidence.** (a) Attacker's norm score for rounds 20–49 on the four seeds, against v3 τ = 3 and ZP h = 1.75. On W1/W2 the score stays between 1.75 and 3, so ZP fires and v3 does not. (b) W1 clean run: honest client 004 drifts upward late and crosses 1.75, which is the honest alarm of ZP / CUSUM / PH. |
| `fig_contamination.pdf` | `figure*` | Reference-contamination mechanism. (a) Seed 2026: client 002's self-cosine crosses up through the inclusive median and drags c_r. (b) The honest clients' z^t jumps above 3 while the attacker's z^t is 0. (c) Seed W1: client 002 is already above the band, so the precondition does not hold. Observed only at seed 2026 (1 of 4 seeds). |

All figures use fonts of 6–8 pt at final size. Do not shrink them below `\textwidth` / `\columnwidth`.

## 2. New tables (all in `tables/`, booktabs; add `\usepackage{booktabs,amssymb}`)

| File | Use | Notes |
|---|---|---|
| `tab_main_outcomes.tex` | **Mandatory.** `table*`; **replaces the badly formatted old Table 1 ("Frozen forensic decision procedure")** | Seed-level outcomes: top-1, ASR, and the v3 / ZP / CUSUM / PH verdicts for BadNets, relabel, honest drift and clean. |
| `tab_calibration.tex` | **Mandatory** (small, single column) | Key new insight. Attacker peak norm score 3.26 / 3.57 / 3.04 / 2.94; rounds ≥ 3: 10 / 21 / 2 / 0; rounds > 1.75: 29 / 30 / 29 / 29; calibration σ inflation 6.28 / 7.05 / 1.92 / 6.95. |
| `tab_ablation.tex` | Optional | Duplicates rows of the reliability matrix. Use it only if you drop the matrix figure. |
| `tab_archived.tex` | Optional, keep if space allows | Archived NDSS perturbation results: model-update perturbations, **not** data poisoning. v3 onset error 0, delay 2, except reduced-severity Gaussian (21/23). Multi-Krum clean runs: honest client confirmed at r46. |
| `tab_sensitivity.tex` | Optional | Archived τ / P grid: P = 2 produces honest alarms; P ≥ 3 has none on the archived runs. |

Keep the previous Table 2 (it was good) only if it is not duplicated by `tab_main_outcomes`.

## 3. Algorithm

Insert `algorithm/alg_fedtrace_v3.tex` as **Algorithm 1** (single column) in the Method section. Add
`\usepackage{algorithm,algpseudocode}`. It was verified line by line against the frozen code. Do not alter it. If space
is tight, use `\small`.

## 4. Key scientific content to state clearly (all verified)

1. **Setting.** CIFAR-100, ResNet-18, K = 5 clients, Dirichlet α = 0.1, 50 rounds, 1 local epoch, SGD lr 0.01,
   batch 32, FedAvg. Calibration window rounds 0–19 (T_ref = 20). client_002 is intervened from round 20 in every arm
   (clean / BadNets / relabel-only / honest drift). Matched arms are identical before round 20.
   - Seeds 123 and 2026 are development seeds.
   - Seeds W1 and W2 are preregistered confirmation seeds; thresholds were frozen before they ran.
2. **Baselines.**
   - ZP: two-sided z-score + persistence, h = 1.75, P = 3.
   - CUSUM: h = 11.75, k = 0.5.
   - Page-Hinkley: h = 6.5, δ = 0.5.
   - All three run on the per-client update norm, with thresholds set to the smallest grid value giving zero alarms on
     the two development clean runs.
   - **Important verified fact:** the ZP statistic is *identical* to v3's magnitude score z^m (max difference 0.0). The
     whole v3-vs-ZP difference on BadNets is therefore the threshold (3 vs 1.75) and the fused temporal channel, not a
     different statistic. State this; it makes the baseline comparison fair and explains the results.
3. **BadNets** (ASR 43.5 / 64.7 / 63.3 / 44.2 %; two seeds are below the 50 % success criterion, so say so):
   - v3: correct at 123 (onset error 8, delay 10); wrong at 2026 (honest clients 1 and 4 confirmed at r23, attacker at
     r24); missed at W1 and W2.
   - ZP: correct on all four seeds (onset error ≤ 1, delay 2–3), plus honest client 004 at W1.
   - CUSUM: correct on all four seeds (onset error 0, delay 5–6), plus honest client 004 at W1.
   - Page-Hinkley: misses on every seed, with honest alarms on three.
4. **Relabel-only** (development seeds only; ASR 0.7 / 0.4 %, so the backdoor did not succeed):
   - ZP and CUSUM confirm client 002.
   - v3 misses (and accuses clients 1 and 4 at 2026).
5. **Honest drift** (client 002 benignly changes data; no attacker exists):
   - ZP and CUSUM accuse an honest client on **all four seeds**.
   - v3 accuses at 2026 and W1.
   - PH accuses at 2026 and W1.
6. **Clean:** v3 raises no alarm on any seed. ZP, CUSUM and PH all accuse honest client 004 at W1 (late drift; see
   Fig. calibration (b)), so thresholds transferred from development seeds did not hold.
7. **Core message: a detection versus false-accusation trade-off.**
   - The lower threshold (ZP / CUSUM) catches every BadNets attacker but also accuses honest drifting clients.
   - The stricter v3 threshold avoids clean alarms but misses the attacker on both preregistered seeds.
   - On the norm channel, no single frozen threshold separated "attacker" from "honest client whose data changed" in
     our runs.
   - Present this as an observation in this setting, not a theorem. You may cite the conditional argument that malice
     is not identifiable from update statistics without restricting benign data (doc 15 in the package) as a
     conclusion-level remark.
8. **Why v3 misses (calibration).**
   - Including rounds 0–1 in the calibration window inflates σ^m by about 1.7–7.1× across clients and seeds (attacker:
     6.28 / 7.05 / 1.92 / 6.95).
   - This pushes the attacker's score below 3 (peaks 3.04 and 2.94 on W1 / W2).
   - Present this as an observed contributing factor, not a proven cause.
9. **Reference contamination (Fig. contamination).**
   - The clip identity c(x) = min{max{x, s_{m−1}}, s_m} explains the six honest accusations at seed 2026 exactly.
   - The precondition (client 002 at median rank before onset) held only at 2026 (rank 3; rank 4 on the other seeds).
   - Mechanism shown in 1 of 4 seeds; not exercised on W1 / W2.
10. **Detection ≠ attribution:** at 2026 BadNets the attacker was detected but only after two honest clients.
11. **Ablations** (same thresholds):
    - Magnitude-only is correct on the development seeds but misses W1 / W2.
    - Temporal-only misses BadNets everywhere.
    - Leave-one-out and trimmed-mean references fix 2026 but are wrong at 123 and miss W1 / W2.
    - The client-historical reference accuses honest clients in **every** clean run.
12. **Accuracy:**
    - Clean top-1 56.88 / 57.94 / 58.04 / 60.22 %.
    - BadNets Δ −1.94 / −0.94 / −0.96 / −0.86 pp.
    - Drift Δ −1.75 / −1.79 / −2.29 / −1.47 pp.

## 5. Structure (fit into 4 pages)

1. **Title.** Make it evaluative and specific, for example: *"Accusing the Honest? A Seed-Level Reliability Audit of
   Server-Side Forensic Attribution in Federated Learning"* or *"Detection Is Not Attribution: Auditing a Frozen
   Forensic Monitor for Federated Learning"*. No hype words.
2. **Abstract** (100–150 words): setting → what was audited (detection, attribution, onset, honest accusations) →
   three findings:
   - the trade-off;
   - v3 missed on the preregistered seeds while same-statistic baselines detected but accused honest drift;
   - reference contamination at one seed.
   Then the takeaway. Add 5 keywords.
3. **Introduction:** motivation (server-side forensics must name the *right* client; a false accusation removes an
   honest participant). Why K = 5: the cross-silo regime, where one false accusation is 20 % of participants. Bullet
   the contributions as audit contributions.
4. **Related work** (required by the template): FLTracer, FLDetector, DeepSight, FLForensics, FedREDefense, CUSUM /
   Page-Hinkley. Use the verified one-line descriptions in `paper_plan/36_publication_policy_audit.md` §6. Claim no
   novelty of the evaluation decomposition.
5. **Monitor and protocol:** Eq. for z^m, z^t (with c_r the *inclusive* median), S, persistence; Algorithm 1; the
   outcome definitions; seeds and freezing; Fig. 1.
6. **Results:** main table; reliability matrix (baselines + ablations); calibration evidence (Fig. + Table); reference
   contamination (Fig.); one sentence on archived perturbations.
7. **Limitations** (explicit, short):
   - one dataset / model;
   - K = 5;
   - one intervened client;
   - onset at T_ref;
   - four seeds (two preregistered);
   - BadNets ASR < 50 % on two seeds;
   - the deployable reference proxy was unreliable on the confirmation seeds (MAE 0.6–2.7 z);
   - no executed comparison with published FL forensics methods.
8. **Conclusion:** the trade-off and recommendations (report detection, attribution, onset and honest accusations
   separately; report seed by seed; test benign drift arms).
9. **Page 5:** references + acknowledgements with the **AI-use disclosure** (the IEEE policy requires naming the AI
   systems and the sections they touched: analysis code, figures, tables, algorithm transcription, editing).

## 6. Presentation quality checklist

- Every figure and table is referenced in the text, in order, and its caption can be understood stand-alone.
- No float sits before its first mention. Prefer `[t]` placement.
- Equation numbering is consistent. Notation is the same in equations, the algorithm and the figures (z^m, z^t, S, τ,
  P, c_r, T_ref).
- Fix the earlier issues: the Eq. 2 definition (inclusive median, max{0,·}), reference ordering, the TODO in ref [10],
  the e-mail / affiliation format.
- Run through the claim checklist (`manuscript_c4ai4c27/CLAIM_EVIDENCE_CHECKLIST.md`). Every result sentence must map to
  C-01 … C-13.
- Finally, give me: the full `main.tex`, the updated `refs.bib`, a list of every number in the paper with its source
  file, and the page count estimate.
