# Claim-to-Evidence Checklist (complete before submission)

Every result sentence in `main.tex` must cite one claim ID; tick only after checking the number against the listed
artifact. Claim definitions and classes: `paper_plan/32_final_claim_matrix.md`. Artifact hashes:
`paper_plan/31_evidence_manifest.sha256`.

| ☐ | Claim ID | Permitted statement (do not strengthen) | Ledger | Artifact | Where in paper |
|---|---|---|---|---|---|
| ☐ | C-01 | v3 did not confirm the BadNets attacker on W1 or W2; late at 123; after honest clients at 2026 | L-89, L-28, L-41 | `results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json`, `unified_outcomes.json` | Results, Table 1 |
| ☐ | C-02 | ZP / CUSUM attributed BadNets on all four seeds (onset error ≤ 1) and also confirmed honest client_004 in W1 BadNets | L-56, L-92 | same | Results, Table 1 |
| ☐ | C-03 | benign drift → honest accusations (ZP, CUSUM all four seeds; v3 at 2026 and W1) | L-64, L-92, L-89 | same | Results, Table 1, Fig. 2 |
| ☐ | C-04 | transferred thresholds → honest alarm in the W1 clean run | L-92 | same | Results |
| ☐ | C-05 | reference-driven misattribution only at seed 2026 | L-41, L-51, L-83, L-90 | `results_rev/pilot/contamination/contamination*.json` | Section 4, Fig. 1 |
| ☐ | C-06a | Eq. 3 clip identity, equal transfer, one-sidedness, Eq. 4 necessity — elementary / exact | doc 25 | `contamination*.json` (identity error 0.0) | Section 4 |
| ☐ | C-06b | the six seed-2026 accusations are explained; precondition (median rank) held in 1 of 4 seeds; not exercised on W1 / W2 | L-83, L-84, L-90 | same; traces | Section 4, Fig. 1 |
| ☐ | C-07 | ASR 43.5 / 64.7 / 63.3 / 44.2 %; two below the 50 % criterion | L-27, L-42, L-88 | pair `summary.json` files | Protocol / Results, Table 1 |
| ☐ | C-08 | deployable proxy unreliable on confirmation seeds (MAE 0.6–2.7 z) | L-91 | `contamination_confirmation.json` | Limitations |
| ☐ | C-09 | detection ≠ attribution at 2026 BadNets (attacker r24 after honest r23) | L-41 | `unified_outcomes.json` | Results |
| ☐ | C-10 | magnitude-only vs t_res-only ablation of v3 | L-86 | docs 28, 30 | Results (optional) |
| ☐ | C-11 | archived perturbations: v3 onset error 0, delay 2 (except reduced-severity Gaussian) — perturbations, not poisoning | docs 18, 28 | archived traces | Introduction / Results (one sentence) |
| ☐ | C-12 | historical-reference variant accuses honest clients in every clean run | L-93 | unified tables | optional |
| ☐ | C-13 | malice not identifiable without restricting benign data (conditional) | doc 15 A3–A4 | — | Conclusion |
| ☐ | U-01 … U-07 | **must not appear** (superiority, rates, mechanism confirmed, novelty of the decomposition, comparisons with unexecuted methods, generalisation, D1 / label-shift as methods) | doc 32 | — | — |

Also check: Table 1 regenerated from current artifacts (`ASSET_PROVENANCE.json` hashes match); captions reviewed;
`refs.bib` TODO-VERIFY resolved; AI disclosure present; ORCID for every author; ≤ 4 content pages.
