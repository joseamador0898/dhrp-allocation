# Build instructions

Primary ICAIF/ACM source:

```bash
pdflatex main_icaif
bibtex main_icaif
pdflatex main_icaif
pdflatex main_icaif
```

`main.tex` and `main.pdf` are aliases of the ICAIF version for upload convenience.

Backup AAAI-style source:

```bash
pdflatex main_aaai_submission
bibtex main_aaai_submission
pdflatex main_aaai_submission
pdflatex main_aaai_submission
```

The AAAI draft uses the AAAI 2026 style as a proxy until the AAAI 2027 author kit is released.
