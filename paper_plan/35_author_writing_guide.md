# 35 — Author-Writing Guide for the C4AI4C'27 Paper

Prepared 2026-10-27 (Prompt 35) for the authors. The LaTeX package (`manuscript_c4ai4c27/`) contains structure,
verified equations, generated tables / figures and notes — **not** scientific prose. The authors write and own every
sentence; every result sentence must map to a claim ID (doc 32) and pass the checklist
(`manuscript_c4ai4c27/CLAIM_EVIDENCE_CHECKLIST.md`).

## 1. Research question and message (approved P-61)

- Question: *When do federated-learning anomaly detectors fail to identify the correct attacker or to distinguish
  malicious updates from benign distribution drift?*
- Message (doc 23 §8): detection, attribution and silence under legitimate drift fail in different, seed-dependent
  ways; reference contamination is one predictable, rank-conditioned route to false attribution, observed in one
  development seed.
- FedTrace v3 is the **evaluated framework** (P-63) — never "our proposed detector" or "superior".

## 2. Page budget (4 pages content; page 5 = references, funding acknowledgements, ethics statement only)

| Section | Budget | Must contain |
|---|---|---|
| Title, authors, abstract (100–150 words), keywords (≤ 5) | 0.3 | five outcomes; setting; seed-level findings; no "novel detector" |
| 1 Introduction | 0.55 | motivation (few-client cross-silo FL), the five outcomes, question, ≤ 2–3 contributions (doc 33 §1) |
| 2 Relation to prior work (mandatory per template) | 0.35 | doc 36 §6 only; what is not new |
| 3 Setting and evaluated framework | 0.6 | observation model; Eqs. 1–2; baselines and threshold provenance |
| 4 Reference contamination | 0.45 | Eq. 3 labelled elementary; Eq. 4 necessity; precondition; observed once |
| 5 Protocol | 0.35 | data, model, K, interventions, matched histories, seeds (development vs preregistered), frozen thresholds |
| 6 Results | 1.2 | Table 1, Fig. 1 (Fig. 2 if space); one paragraph per outcome |
| 7 Limitations and conclusion | 0.3 | doc 34 §2 items 2, 3, 5; C-13 |

## 3. Writing rules

1. Report seed-level outcomes; never a rate, mean or confidence interval over seeds or rounds.
2. Keep the five outcomes separate in every sentence ("detected" ≠ "attributed").
3. Label each finding: confirmed on the preregistered seeds vs development-only (doc 32 class).
4. Archived Gaussian / directional results are *model-update perturbations*, not data poisoning.
5. The median identity is elementary — say so.
6. Name the drift intervention as correctly labelled class-composition drift of an honest client.
7. Do not cite or describe any method beyond the verified statements (doc 36 §6); no numbers for unexecuted methods.
8. Complete every `TODO-VERIFY` field in `refs.bib`; add Krum / ResNet / CIFAR-100 / Page–Hinkley only after
   verifying them.
9. Disclose AI use in the acknowledgments (doc 36 §4).
10. Remove all `\authornote` before submission; keep captions only after reviewing them against the figures.

## 4. Verified elements already in `main.tex`

- Eqs. 1–2 match `src/analysis/fedtrace_detector.py::compute_all_scores_v3` (frozen `7f6ca8b`): z_m = |n − μ| / σ
  over rounds 0–19; residual d = c − t with c the inclusive same-round median; z_t = max(0, (d − μ)/σ) over rounds
  1–19; sample standard deviations; S = max(z_m, z_t); τ = 3, P = 3; attribution = first confirmation; onset = start
  of the streak.
- Eq. 3 (odd-K clip identity) and Eq. 4 (necessity) match doc 25 and `src/rev/contamination.py` (identity error 0.0
  on all traces).
- Table 1 and Figs. 1–2 are generated from frozen artifacts; regenerate with `make_assets.py`; hashes in
  `ASSET_PROVENANCE.json`.

## 5. Remaining author tasks

Title confirmation; author list with ORCID iDs; abstract; keywords; all section prose; caption review; Relation to
Prior Work; Limitations; Compliance with Ethical Standards; acknowledgments with AI disclosure; bibliography
completion; compile (no TeX engine here) and check page limits / PDF requirements; portal fields.
