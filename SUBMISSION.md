# NeurIPS 2026 E&D Submission Checklist

**Target**: NeurIPS 2026 Evaluations & Datasets track
**Portal**: https://openreview.net/group?id=NeurIPS.cc/2026/Evaluations_and_Datasets_Track
**Abstract deadline**: May 4, 2026 (11:59 PM AoE)
**Full paper deadline**: May 6, 2026 (11:59 PM AoE)
**Decision**: September 24, 2026
**Conference**: December 6-12, 2026 (Sydney + Atlanta + Paris satellites)

The abstract and full paper are submitted in the **same OpenReview workflow**.
There is no separate "abstract registration" step — by May 4 you just lock
in the title/abstract; by May 6 you upload the final PDF + supplementary.

---

## File Limits

| Item | Limit |
|------|-------|
| Main PDF | 50 MB |
| Supplementary ZIP | 100 MB |
| Main text page count | **9 pages** (figures + tables included) |
| References | unlimited (do not count) |
| Appendix + checklist | unlimited (do not count) |
| PDF font types | Type 1 or embedded TrueType only — no Type 3 |
| Filename | `submission.pdf` (no author names in filename) |

---

## T-5 Days (May 1) — One-Time Setup

- [ ] Pull latest: `git pull`
- [ ] Verify Colab CSVs are pulled into `results/`:
  - [ ] `results/sharpe_pivot_multiseed_mean.csv`
  - [ ] `results/sharpe_pivot_multiseed_std.csv`
  - [ ] `results/psr_pivot_multiseed.csv`
  - [ ] `results/all_universes_multiseed_summary.csv`
  - [ ] `results/Commodities_results.csv` (for FF3+AQR alphas)
- [ ] Download `neurips_2026.sty` and `checklist.tex` from
      [official Overleaf template](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc)
      and place both in `paper/`
- [ ] Verify your OpenReview account exists and is logged in
- [ ] Create an anonymous fork on
      [anonymous.4open.science](https://anonymous.4open.science/)
      pointing to the latest `main` of this repo

---

## T-4 Days (May 2) — Generate + Build

```bash
# 1. Generate all tables and figures from CSVs
python scripts/generate_paper_tables.py    # writes paper/tables/*.tex
python scripts/generate_paper_figures.py   # writes paper/figures/*.pdf

# 2. Build the paper PDF
cd paper/
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
cd ..

# 3. Validate
python scripts/validate_paper.py           # 0 errors required
python scripts/anonymize_check.py --paper-only

# 4. Page-count check (manual)
texcount -inc paper/main.tex                  # should be <= 9 main pages
pdffonts paper/main.pdf | grep -i "type 3"    # should be EMPTY
```

If `texcount` reports > 9 main-text pages: tighten Section 5 (Results) and
Section 2 (Related Work). Move detailed tables to appendix.

---

## T-3 Days (May 3) — Internal Review

- [ ] Read `paper/main.pdf` end-to-end
- [ ] Verify abstract numbers match `paper/sections/abstract.tex`
- [ ] Verify Table 1 numbers match `paper/tables/headline_sharpe.tex`
- [ ] Verify Figure 1 architecture diagram is present
      (currently `figures/architecture.pdf` is TODO — needs hand-drawing)
- [ ] Confirm all `\citep{...}` resolve in the bibliography
- [ ] Spot-check 5 random citations against arXiv/OpenReview URLs
- [ ] Check section flow: Introduction → Method → Benchmark → Results →
      LLM Ablation → Limitations → Conclusion

---

## T-2 Days (May 4 AoE) — Abstract Lock

**Deadline**: May 4, 2026, 11:59 PM AoE.

- [ ] Open https://openreview.net/group?id=NeurIPS.cc/2026/Evaluations_and_Datasets_Track
- [ ] Click "Submit"
- [ ] Track: **Evaluations & Datasets Track**
- [ ] Title: "DHRP: A Differentiable Hierarchical Risk Parity Architecture and Multi-Universe Portfolio Benchmark"
- [ ] Abstract: copy from `paper/sections/abstract.tex` (strip LaTeX)
- [ ] Authors: anonymized (default for E&D)
- [ ] Keywords: portfolio optimization; differentiable optimization;
      hierarchical risk parity; multi-seed evaluation; financial benchmark;
      Probabilistic Sharpe Ratio; LLM ablation
- [ ] Subject area / TC area: select most relevant
- [ ] **Submit abstract** (you can still update the PDF until May 6)

---

## T-0 (May 5-6) — Final Polish + Submit

### May 5 (T-1)

- [ ] Final pdflatex build
- [ ] Confirm `paper/main.pdf` ≤ 9 main-text pages
- [ ] Confirm no Type 3 fonts: `pdffonts paper/main.pdf`
- [ ] Confirm filename is `submission.pdf` (rename if needed)
- [ ] Build supplementary ZIP:
  ```bash
  zip -r supplementary.zip \
      data/croissant/dhrp-8universe.json \
      REPRODUCIBILITY.md \
      requirements-frozen.txt \
      scripts/reproduce_all.sh \
      scripts/generate_paper_tables.py \
      scripts/generate_paper_figures.py
  ```
- [ ] Validate Croissant metadata at:
      https://huggingface.co/spaces/JoaquinVanscholen/croissant-checker
      (upload `data/croissant/dhrp-8universe.json` and confirm "PASSED")
- [ ] Run final validators:
  ```bash
  python scripts/validate_paper.py     # exit 0
  python scripts/anonymize_check.py    # exit 0 (full repo scan)
  ```

### May 6 AoE — Submit Paper (DEADLINE)

- [ ] Open OpenReview submission edit page
- [ ] Upload `submission.pdf` (the main paper)
- [ ] Upload `supplementary.zip` (Croissant + reproducibility bundle)
- [ ] Add anonymized GitHub URL: `https://anonymous.4open.science/r/<your-fork-id>`
- [ ] Confirm all required fields filled:
  - [ ] Track: Evaluations & Datasets
  - [ ] Authors hidden
  - [ ] Conflicts of interest declared (institutional + collaborator within last 4 years)
  - [ ] Code of ethics agreement
  - [ ] Reproducibility checklist included in PDF
- [ ] Click **Submit**
- [ ] Take a screenshot of the confirmation page

---

## After Submission

| Date | Event |
|------|-------|
| May 7 - July 7 | Initial review period |
| July 8 - July 31 | Author rebuttal window |
| August - September | Discussion + meta-review |
| **September 24, 2026** | Decision notification |
| October-November | Camera-ready (if accepted) |
| December 6-12, 2026 | Conference (Sydney + Atlanta + Paris) |

---

## Backup Venues (If Rejected)

| Venue | Expected deadline | Notes |
|-------|-------------------|-------|
| ICLR 2027 | ~Sep 24, 2026 | 9 pages + appendix; double-blind |
| AAAI 2027 | ~Aug 1, 2026 | 8-9 pages; double-blind |
| ICML 2027 | ~Jan 27, 2027 | 9 pages + appendix; double-blind |
| Journal of Financial Econometrics | rolling | longer-form, theory-friendly |
| Journal of ML Research | rolling | longest review cycle |
| ICAIF 2027 | ~Aug 2, 2027 | 8 pages strict; finance-focused |

---

## Common Last-Week Failures (Avoid)

1. **Page count > 9** — single most common desk-reject. Run `texcount` early.
2. **Type 3 fonts** in figures — fix at figure-generation time
   (`scripts/generate_paper_figures.py` already sets Type 42).
3. **Croissant validation failure** — test at
   https://huggingface.co/spaces/JoaquinVanscholen/croissant-checker.
4. **Inaccessible code/data URLs** — test in incognito mode.
5. **De-anonymizing strings** — `python scripts/anonymize_check.py` catches.
6. **Author name in PDF metadata** — strip with
   `exiftool -all= paper/main.pdf`.
7. **Missing checklist** — `paper/checklist.tex` is complete but verify it's
   `\input` from main.tex.
8. **Last-minute git commits with author identity** — the anonymized fork
   should be a separate repo, not a branch.

---

## Quick One-Liner Commands

```bash
# Full validation + build pipeline
make all                      # if Makefile is present (see paper/Makefile)
# OR manually:
python scripts/validate_paper.py && \
python scripts/anonymize_check.py --paper-only && \
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd .. && \
texcount -inc paper/main.tex && \
pdffonts paper/main.pdf
```

If all checks pass and texcount shows ≤ 9 pages → **safe to submit**.
