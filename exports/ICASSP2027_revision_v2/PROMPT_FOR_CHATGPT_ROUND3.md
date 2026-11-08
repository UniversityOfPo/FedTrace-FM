# Round-3 revision instructions (audit of the "FedTrace_V5" draft)

The attached ICASSP2027_revision_v2.zip has been updated. New items:
- figures/fig_calibration_col.pdf (single-column version)
- backmatter/acknowledgments_ethics.tex
- backmatter/refs_fixes.bib

All numbers in the current draft were re-checked against the artifacts and are correct. **Do not change any
number.** Apply ONLY the fixes below, then return the full main.tex, refs.bib and a change list.

## A. Items from the previous round that were NOT applied (mandatory)

### A1. Fig. 1: replace it
Replace the current Fig. 1 (A/B split) with `figures/fig_overview.pdf`, as a `figure*` at `width=\textwidth`.
The current figure has a rendering bug: the arrow and the dashed line run through the text "clean / BadNets" and
"Reliability audit". Use this caption:

> **Fig. 1. Audited pipeline and audit outcomes.** Five clients send identity-labelled updates; client 002 is
> intervened on from round 20 (clean, BadNets, relabel-only or honest drift; arms identical before round 20).
> The frozen FedTrace v3 monitor combines a magnitude and a temporal/peer score, fuses them by maximum and
> confirms a client after three consecutive rounds with S ≥ 3. Norm baselines (ZP, CUSUM, Page–Hinkley) use
> the same norm score. Each run is audited for attack success, detection, first attribution, onset error and
> delay, and honest-client accusations.

### A2. Table 1 "Frozen v3 decision procedure": replace it with Algorithm 1
A procedure must not be presented as a table. Delete Table 1 and insert `algorithm/alg_fedtrace_v3.tex` as
**Algorithm 1**, in a single column, after Eq. (1). Add `\usepackage{algorithm,algpseudocode}`. Do not alter it.
- Remove the sentence "the archived code is the authority for tie-breaking". The tie-break is specified: the
  earliest confirmed client; ties go to the larger mean streak score (the last line of Algorithm 1).
- Renumber the remaining tables (Table 2 → Table 1, and so on) and update every reference to them.

### A3. Add the calibration figure (the core evidence for the threshold contrast)
Insert `figures/fig_calibration_col.pdf` as a single-column `figure` at `width=\columnwidth`, placed next to the
paragraph "A controlled threshold contrast". Use this caption:

> **Fig. 3. Norm score of the attacker and of a clean honest client.** (a) BadNets attacker (client 002), rounds
> 20–49, four seeds: on W1* and W2* the score stays between the ZP threshold (h = 1.75, dotted) and the v3
> threshold (τ = 3, solid), so ZP confirms the attacker and v3 does not. (b) W1* clean run: honest client 004
> drifts upward late in training and crosses h = 1.75, which produces the ZP/CUSUM/Page–Hinkley clean alarm;
> v3 raises none.

Cite it in Sec. 5 ("Effective attacks can be missed": "…neither meets three-round persistence (Fig. 3a)") and in
the specificity paragraph ("…raise a W1 clean alarm (Fig. 3b)").

### A4. Fig. 2: replace it
Replace Fig. 2 with `figures/fig_contamination.pdf` (`figure*`). The current panel (a) has its legend overlapping
the curves, and the legends say "client_002"; the text says "client 002".

### A5. Page-5 back matter (required by IEEE / ICASSP)
Insert `backmatter/acknowledgments_ethics.tex` after the conclusion, before the references. The template allows it
on page 5. Keep the bracketed PI placeholders as LaTeX comments.

### A6. References
Replace [2] FedProx, [7] FLDetector and [9] Krum with the entries in `backmatter/refs_fixes.bib` (adds the missing
page numbers).

## B. New problems introduced in this draft

### B1. Too many overlapping sections (11 sections in 4 pages)
Sections 7, 8 and 9 restate earlier material. Make these cuts, which also free the space for A2 and A3:
- **Delete §8 "What Table 2 actually measures"** entirely. It restates the table caption and §4.
- **Delete §8 "How to read the forensic outcomes"**, keeping only one sentence: "A checkmark means the *first*
  confirmed source is the intervened client; it does not exclude later honest confirmations (e.g., ZP on W1:
  ✓(1,3)+FA(4))." Move that sentence into the Table 1 caption.
- **Merge §8 "Why the ablation matters" into the §5 paragraph "Ablations and reference sensitivity".** Do not say
  the same thing twice. Keep the key sentence "No reference choice is uniformly safe…" and the archived-table
  warning.
- **§9 "A controlled threshold contrast" duplicates §3 and §5.** Merge it into §5 "Effective attacks can be
  missed", where it should cite Fig. 3. Keep the caveat that the comparison does not isolate the temporal channel.
- **§9 "What Fig. 2 explains" duplicates §6.** Keep at most two sentences of it inside §6. In particular keep:
  "It is neither evidence that the same mechanism caused the W1/W2 misses nor a validated real-time correction."
- **Move the TV-bound paragraph ("An update-only monitor cannot always infer intent")** out of the "Archived
  perturbations" section into the final section, as a discussion point.
- Target structure:

  1. Introduction
  2. Prior work and scope
  3. Frozen monitor: Eq. 1 + Algorithm 1 + baselines
  4. Matched interventions and outcomes
  5. Findings: 5.1 attacks missed and the threshold contrast (Fig. 3); 5.2 specificity: clean and drift; 5.3 ablations
  6. Conditional reference contamination (Fig. 2)
  7. Archived perturbations (3–4 lines + table)
  8. Discussion, limitations and conclusion: keep "Consequences for a forensic workflow" (3 sentences), "Evidence
     still needed" (2 sentences), the TV bound and the limitations

### B2. Overclaim
§8: "The first four rows establish that the interventions change the trained model". This is unsupported: there is
one run per seed and no variance estimate, and the changes are 0.9–2.3 pp. Write instead: "The accuracy rows
report the top-1 cost of each intervention relative to the matched clean run (single runs; no variance estimate)."

### B3. Inconsistencies
- §8 says "the more sensitive ZP and CUSUM produce a W1 clean alarm". Page–Hinkley also does, so write "ZP, CUSUM
  and Page–Hinkley".
- The Fig. 1 caption says "Authenticated client updates" while the text says "identity-labelled". Use
  "identity-labelled" everywhere.
- Table caption: "–" is defined as "not run", but "none" (no alarm) is never defined. Add: "none: no client
  confirmed."
- §7 says "Multi-Krum … (round 46)". Write "(first confirmation at round 46)".
- Title of §5: keep it. Title of §7: rename to "Archived model-update perturbations".

### B4. Author block
The affiliations are now correct, but every e-mail address was removed. ICASSP papers list contact e-mails. Restore
them under each affiliation (PI to confirm the correct HTU address).

## C. Final checks before returning
- [ ] Content fits on 4 pages; page 5 holds only the references, the ethics statement and the acknowledgments.
- [ ] Every float is cited in the text before it appears, in order: Fig. 1, Algorithm 1, Table 1, Fig. 3 placement
      near §5, Fig. 2 in §6.
- [ ] No remaining "client_002", "Table 2" or "Table 1" references left over from the old numbering.
- [ ] Notation is the same in Eq. 1, Algorithm 1 and the figures (z^m, z^t, S, τ, P, c_r, T_ref, r⋆).
- [ ] Return a list of every number in the paper together with its source (claim ID C-01…C-13 or ledger L-xx).
