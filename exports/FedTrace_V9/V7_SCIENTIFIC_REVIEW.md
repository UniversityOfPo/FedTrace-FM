# Fate Trace Version 7 — scientific and layout review

## Source choices
- Figure 1: kept the original **Version 6** vector PDF, as expressly requested; Claude Version 7 TikZ overview is retained only as an unused alternative.
- Calibration time-series figure and contamination figure: selected from the uploaded Claude Version 7 package.
- Main seed-level table, calibration table, ablation table, archived perturbation table, and Algorithm 1: selected from the supplied Version 7 assets.
- Preserved the distinction between local-data poisoning and archived model-update perturbations.

## Independent editorial corrections
- Removed the misleading suggestion that CUSUM is a threshold-only version of the FedTrace magnitude score: the numerical identity is specifically between ZP and the v3 magnitude statistic.
- Qualified the calibration plot description: W1 briefly reaches 3 but fails three-round persistence; it does not stay below 3 at every round.
- Changed an overstrong causal reading of matched arms to controlled within-seed comparisons.
- Qualified the W1 clean plot interpretation as coincident with the three detector alarms, not a proof of their identical mechanics.
- Shortened the AI-use statement to avoid asserting unverified detailed authorship/workflow responsibilities; author confirmation is required.
- Unnumbered acknowledgments and ethics headings; kept references and these statements on page 5.

## Verification
- Compiled with pdfLaTeX, total five pages, pages 1–4 technical content and page 5 acknowledgments, ethics and references.
- No overfull box or unresolved citation warnings in the final LaTeX log.
- No new experiments or raw numerical evidence were produced. Uploaded numerical tables and figures were adopted from supplied evidence assets; this is not an independent re-run of frozen code.

## Mandatory author checks before submission
1. Confirm all three institutional affiliations, exact email addresses, and ORCID IDs. Coauthor emails are not yet supplied.
2. Confirm AI-use disclosure, any funding, and prior FedTrace NDSS submission status against the actual workshop/IEEE policies.
3. Verify bibliography metadata against official publisher pages and all reported numbers against frozen artifacts.
4. Review print-size legibility of the main table and Algorithm 1, and ensure the required workshop template and page-count rules remain current.
5. Check the source algorithm tie-breaking rule against the frozen implementation, rather than treating the supplied text as proof of correctness.
