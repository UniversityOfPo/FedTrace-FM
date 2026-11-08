# C4AI4C'27 manuscript: final_v1 (Overleaf-ready)

## Compile

Upload `C4AI4C27_final_v1_overleaf.zip` to Overleaf (New Project → Upload Project). Set the compiler to pdfLaTeX
and the main file to `main.tex`.

Locally, run: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Contents

| File | What it is |
|---|---|
| `main.tex` | Full manuscript |
| `refs.bib` | Bibliography (21 entries) |
| `spconf.sty`, `IEEEbib.bst` | Official ICASSP 2027 template files (unmodified) |
| `figures/fig_overview.pdf` | Fig. 1, the pipeline overview |
| `figures/fig_calibration_col.pdf` | Fig. 3, attacker vs. honest norm scores |
| `figures/fig_contamination.pdf` | Fig. 2, the reference-contamination mechanism |

All figures and numbers come from the frozen artifacts (`exports/ICASSP2027_revision_v2`, `PROVENANCE.json`).

## Before submission (PI)

1. Replace `EMAIL-TO-CONFIRM@htu.edu` with your address and confirm the co-authors' addresses.
2. Make sure every author has an ORCID iD in the submission portal.
3. Review the AI-use disclosure in the Acknowledgments, add the funding statement, and resolve the two `% PI:`
   comments.
4. Confirm the references marked `% CHECK` in `refs.bib` against the publisher pages.
5. Check the page budget. Content (through the Conclusion) must end on page 4; page 5 may contain only the ethics
   statement, the acknowledgments and the references. If it overflows, use these knobs in order:
   - `width=0.94\textwidth` → `0.88` for Fig. 1;
   - `0.92` → `0.86` for Fig. 2;
   - shorten Table 3 (archived) to its first seven rows;
   - change `\vspace{-3mm}` to `-4mm` under the floats.
6. If FedTrace v3 was submitted to NDSS, disclose it according to the IEEE prior-submission policy.
