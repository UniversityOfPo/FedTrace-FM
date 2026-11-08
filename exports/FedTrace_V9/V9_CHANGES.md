# Version 9: changes from Version 8

All results, numbers, tables, Figures 2–3 and Algorithm 1 are unchanged.

## Layout
- Tighter spacing between floats and text. The preamble sets `\textfloatsep`, `\floatsep`, `\dbltextfloatsep`,
  `\dblfloatsep`, `\intextsep` and the caption skips, which removes the large gaps below tables, figures and the
  algorithm.
- Fig. 1: the vector PDF replaces the PNG. The figure itself is fixed: the "Clients (K = 5)" title no longer
  overlaps the Δ label.

## Text
- Contributions are now a compact three-item list with the same wording, followed by a one-sentence
  paper-organization paragraph.
- §5: the two consecutive "Table 1 …" sentences are merged into one.

## Accuracy fixes
- §6: "a retrospective historical-offset proxy … not deployable" is now "a deployable proxy for this displacement …
  not reliable enough to deploy". It is the deployable proxy (claim C-08), not a retrospective one.
- §7: added the Multi-Krum provenance caveat. The archived runs executed code not contained in their recorded
  commit (ledger L-17).

## Back matter (page 5)
- Added Compliance with Ethical Standards and the Acknowledgments with the AI-use disclosure (text from Version 7).

## Checked, not changed
- "temporal scores of 3.4–4.6 in rounds 21–23" is correct: the data give a minimum of 3.436 and a maximum of 4.551.
- Every cell of Table 3 was verified against `unified_outcomes*.json`.

## Before upload
- Compile on Overleaf (pdfLaTeX) and confirm that the Conclusion ends on page 4. Page 5 may contain only the ethics
  statement, the acknowledgments and the references.
- If it overflows, move the paper-organization sentence (end of the Introduction) to a comment first.
- Add co-author e-mails and the funding line if any.
