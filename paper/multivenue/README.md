# Multi-venue LaTeX source for "Differentiable Hierarchical Risk Parity"

This directory contains a single venue-agnostic body (`body.tex`,
`bibliography.tex`, `appendix.tex`) plus five thin per-venue wrappers
that build the same content against the official style files of
NeurIPS 2025, ICML 2025, ICLR 2026, AAAI 2026, and KDD 2025
(ACM SIGKDD).

The body was extracted deterministically from `results/hrp_paper_fixed.tex`
by `scripts/extract_body.py`. **Do not edit `body.tex` by hand** — instead,
edit `results/hrp_paper_fixed.tex` (the canonical source) and re-run the
extraction script:

```bash
cd /path/to/HRP_AI
python scripts/extract_body.py
```

## File layout

```
paper/multivenue/
├── README.md                  ← this file
├── body.tex                   ← venue-agnostic content (Section 1 → Limitations)
├── bibliography.tex           ← \begin{thebibliography}…\end{thebibliography}
├── appendix.tex               ← Appendices A–F
├── main_neurips.tex           ← root: NeurIPS 2025 (single-col, 9pp)
├── main_icml.tex              ← root: ICML 2025      (two-col, 8pp)
├── main_iclr.tex              ← root: ICLR 2026      (single-col, 9pp)
├── main_aaai.tex              ← root: AAAI 2026      (two-col, 7pp)  ← BINDING
└── main_kdd.tex               ← root: KDD 2025/ACM   (two-col, 8pp)
```

## Style files (NOT included; download per venue)

| Venue | Style file | Download |
|---|---|---|
| NeurIPS 2025 | `neurips_2025.sty` | <https://neurips.cc/Conferences/2025/CallForPapers> |
| ICML 2025 | `icml2025.sty`, `icml2025.bst` | <https://media.icml.cc/Conferences/ICML2025/Styles/icml2025.zip> |
| ICLR 2026 | `iclr2026_conference.sty`, `iclr2026_conference.bst`, `natbib.sty` | <https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip> |
| AAAI 2026 | `aaai2026.sty`, `aaai2026.bst` | <https://aaai.org/authorkit26/> |
| KDD 2025 | `acmart` (TeX Live) | <https://www.acm.org/publications/proceedings-template> |

Place each style file alongside the matching `main_*.tex` and run the
build command below.

## Build commands

Each wrapper is self-contained. From `paper/multivenue/`:

```bash
# NeurIPS 2025
pdflatex main_neurips ; bibtex main_neurips ; pdflatex main_neurips ; pdflatex main_neurips

# ICML 2025
pdflatex main_icml    ; bibtex main_icml    ; pdflatex main_icml    ; pdflatex main_icml

# ICLR 2026
pdflatex main_iclr    ; bibtex main_iclr    ; pdflatex main_iclr    ; pdflatex main_iclr

# AAAI 2026  (the binding-constraint build — fail here = need to trim body.tex)
pdflatex main_aaai    ; bibtex main_aaai    ; pdflatex main_aaai    ; pdflatex main_aaai

# KDD 2025 (ACM acmart)
pdflatex main_kdd     ; bibtex main_kdd     ; pdflatex main_kdd     ; pdflatex main_kdd
```

## Page-budget summary (verified from official CFPs)

| Venue | Submission limit (main text) | Camera-ready | Refs / appendix | Column | Source |
|---|---:|---:|---|---|---|
| NeurIPS 2025 | 9 pp | 10 pp | unlimited | single | <https://neurips.cc/Conferences/2025/CallForPapers> |
| ICML 2025 | 8 pp | 9 pp | unlimited | two | <https://icml.cc/Conferences/2025/CallForPapers> |
| ICLR 2026 | 9 pp | 10 pp | unlimited (after bib) | single | <https://iclr.cc/Conferences/2026/CallForPapers> |
| AAAI 2026 | 7 pp technical | 7+1 pp (typical) | unlimited | two | <https://aaai.org/conference/aaai/aaai-26/submission-instructions/> |
| KDD 2025 | 8 pp | 9 pp (typical) | unlimited | two (acmart sigconf) | <https://kdd2025.kdd.org/research-track-call-for-papers/> |

## Submission policy: serialize, do not parallelize

All five venues explicitly forbid parallel submission to other peer-reviewed
conferences during their review window:

- **NeurIPS 2025**: <https://neurips.cc/Conferences/2025/CallForPapers> — "Submissions … submitted in parallel to other peer-reviewed venues … may not be submitted to NeurIPS."
- **ICML 2025**: <https://icml.cc/Conferences/2025/CallForPapers> — "Authors may not submit papers that are identical, or substantially similar … submitted in parallel to other conferences or journals."
- **ICLR 2026**: <https://iclr.cc/Conferences/2026/CallForPapers> — same restriction; arXiv preprints and non-archival workshops are explicitly OK.
- **AAAI 2026**: <https://aaai.org/conference/aaai/aaai-26/submission-instructions/> — "Once authors have made a submission to AAAI-26, they may not submit the same paper to another archival conference or journal until they receive an accept/reject decision … or they withdraw."
- **KDD 2025**: ACM-wide dual-submission policy.

**Recommended order** (highest acceptance prestige first, with arXiv between rounds):

1. NeurIPS 2025 → if rejected, post arXiv.
2. ICLR 2026 → if rejected, post arXiv.
3. ICML 2025 → if rejected, post arXiv.
4. AAAI 2026 → fallback (binding 7-page constraint already satisfied).
5. KDD 2025 → fallback (data-mining venue, acmart format).

Withdraw from each before submitting to the next.

## Common edits to keep body.tex venue-portable

When editing `results/hrp_paper_fixed.tex`, observe these rules so the
re-extracted `body.tex` compiles against all five venues:

1. **Use only `\citep{}` and `\citet{}`** — never raw `\cite{}`. natbib is
   loaded by every wrapper, but the `acmart` (KDD) bibliography style
   prefers numeric citations and works best with `\citep`.
2. **Use `\linewidth` for figures and tables**, never `\textwidth` or
   `\columnwidth`. This lets the same figure float correctly under both
   single-column (NeurIPS, ICLR) and two-column (ICML, AAAI, KDD) layouts.
3. **Avoid `\paragraph{}` width tricks** — AAAI's style file rejects them
   silently. Use `\subsubsection*{}` or a bold inline lead-in.
4. **No `\usepackage{}` inside body.tex** — all packages live in the
   wrapper preambles. Adding a package to the body breaks AAAI's package
   blacklist.
5. **No `\geometry`, `\fullpage`, or `\setlength` of margins** — every
   venue style fixes its own margins; overriding triggers desk-reject.
6. **Section labels** must use the same `\label{sec:foo}` pattern across
   builds. The body already does this for `sec:method`, `sec:llmablation`,
   `tab:main`, `tab:factors`, `app:proof`, `app:hyperparams`.

## AAAI is the binding constraint

If `pdflatex main_aaai` overflows 7 pages of technical content, the body
is too long for the universal target. Cut order (in priority):

1. Move appendix-eligible content (extra ablations, hyperparameter tables,
   long proofs) to `appendix.tex`.
2. Compress the related-work section to one tight paragraph per theme.
3. Merge subfigures, drop the weakest baseline panel.
4. Last resort: convert a body table to a figure and move it to the appendix.

Do **not** apply micro-compression (`\frenchspacing`, smaller fonts, tighter
margins). AAAI desk-rejects "obviously squeezed" papers per the AAAI
Author Guide (<https://arxiv.org/html/2405.17488v1>).

## Provenance

- Body extracted from `results/hrp_paper_fixed.tex` commit `2898f8f`.
- Per-venue formatting verified by web-search agent against the official
  CFPs (see commit message of the multivenue scaffold for full URLs).
- Submission policies verified against each venue's published CFP.
