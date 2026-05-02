# Makefile for the NeurIPS 2026 E&D paper build pipeline
#
# Usage:
#   make pdf            build paper/main.pdf
#   make tables         generate paper/tables/*.tex from results/*.csv
#   make figures        generate paper/figures/*.pdf from results/*.csv
#   make validate       run pre-submission integrity checks
#   make anon-check     run double-blind anonymization scan
#   make submit-check   full pipeline: tables + figures + build + validate + anon-check
#   make supplementary  build supplementary.zip for OpenReview upload
#   make clean          remove build artifacts (keeps tables and figures)
#   make distclean      full clean (also removes generated tables and figures)
#
# Run from repo root:  make -C paper <target>

# --- Configuration ---
ROOT := $(realpath ..)
PYTHON := python
LATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX := bibtex

PAPER := main
TEX_SOURCES := main.tex \
               sections/abstract.tex \
               sections/01_introduction.tex \
               sections/02_related_work.tex \
               sections/03_method.tex \
               sections/04_benchmark_design.tex \
               sections/05_results.tex \
               sections/06_llm_negative_ablation.tex \
               sections/07_limitations_discussion.tex \
               sections/08_conclusion.tex \
               sections/appendix.tex \
               checklist.tex

# --- Phony targets ---
.PHONY: all pdf tables figures validate anon-check submit-check supplementary clean distclean help

all: submit-check

help:
	@echo "Targets:"
	@echo "  make pdf            build paper/main.pdf"
	@echo "  make tables         regenerate booktabs tables from CSVs"
	@echo "  make figures        regenerate 300-DPI figures from CSVs"
	@echo "  make validate       pre-submission integrity check"
	@echo "  make anon-check     double-blind anonymization scan"
	@echo "  make submit-check   tables + figures + pdf + validate + anon-check"
	@echo "  make supplementary  build supplementary.zip for OpenReview"
	@echo "  make clean          remove .aux/.log/.out/.toc/.bbl/.blg"
	@echo "  make distclean      also remove generated tables/figures and main.pdf"

# --- Build the paper PDF ---
$(PAPER).pdf: $(TEX_SOURCES) references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

pdf: $(PAPER).pdf

# --- Regenerate tables and figures ---
tables:
	cd $(ROOT) && $(PYTHON) scripts/generate_paper_tables.py

figures:
	cd $(ROOT) && $(PYTHON) scripts/generate_paper_figures.py

# --- Validation ---
validate:
	cd $(ROOT) && $(PYTHON) scripts/validate_paper.py

anon-check:
	cd $(ROOT) && $(PYTHON) scripts/anonymize_check.py --paper-only

# --- Full submit-check pipeline ---
submit-check: tables figures pdf validate anon-check
	@echo ""
	@echo "===================================================================="
	@echo "  All checks passed. Ready to upload to OpenReview."
	@echo "  Portal: https://openreview.net/group?id=NeurIPS.cc/2026/Evaluations_and_Datasets_Track"
	@echo "  Validate Croissant: https://huggingface.co/spaces/JoaquinVanscholen/croissant-checker"
	@echo "===================================================================="

# --- Supplementary materials zip ---
supplementary:
	cd $(ROOT) && rm -f supplementary.zip && \
	zip -r supplementary.zip \
		README.md \
		REPRODUCIBILITY.md \
		SUBMISSION.md \
		requirements.txt \
		data/croissant/dhrp-8universe.json \
		notebooks/llm_dhrp_experiments.ipynb \
		src \
		scripts \
		-x 'src/**/__pycache__/*' '**/.ipynb_checkpoints/*' 'scripts/__pycache__/*'
	@echo "supplementary.zip created in repo root"
	@ls -lh $(ROOT)/supplementary.zip

# --- Cleanup ---
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf
	rm -f tables/*.tex
	rm -f figures/*.pdf
