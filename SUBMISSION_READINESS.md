# Submission readiness memo

## Recommended venue

Primary near-term target: **ICAIF 2026 (ACM International Conference on AI in Finance)**.

Rationale: the call is currently open, the paper deadline is August 2, 2026, and the scope explicitly includes AI/ML methods for finance, portfolio allocation, reinforcement/sequential decision-making, benchmarks, robustness, and financial NLP/LLM workflows. The paper is more naturally framed as an AI-in-finance benchmark and allocation architecture than as a pure general-AI algorithm paper.

Backup general-AI target: **AAAI 2027**. The OpenReview page lists submission start on June 24, 2026, abstract registration on July 21, 2026, and full paper deadline on July 28, 2026. For AAAI, strengthen the paper as a general decision-focused learning and differentiable routing contribution.

NeurIPS 2026 E&D is no longer available for a new submission because the full-paper deadline has passed. Do not dual-submit to ICAIF while a substantially similar NeurIPS archival submission is under active review.

## What is included

- `paper/main_icaif.tex`: primary ACM/ICAIF anonymous submission source.
- `paper/main_icaif.pdf`: compiled primary ICAIF PDF, 5 pages including references.
- `paper/main_aaai_submission.tex`: AAAI-style backup draft using the available AAAI 2026 style proxy until the 2027 kit is released.
- `paper/main_aaai_submission.pdf`: compiled anonymous AAAI-style backup PDF, 6 pages.
- `paper/references.bib`: bibliography.
- `paper/paper_text_only_chunks.md`: text-only paper draft in copyable chunks under 1,500 words each.
- `src/`, `scripts/`, `notebooks/`, `results/`, `data/croissant/`: sanitized reproducibility package.

## Sanitization performed

The package excludes `.git`, `.venv`, `.env`, local IDE/tool config directories, notebook drive tokens, local caches, Python bytecode, raw local logs, and notebook outputs. API-key literals and private environment values from the raw uploaded archive were not copied into the final package. The anonymization scanner reports zero flagged files.

## Verified in this environment

- ICAIF PDF: 5 pages, no unresolved citations/references detected in the final LaTeX log, Type 1 embedded fonts, rendered page images inspected.
- AAAI backup PDF: 6 pages, no unresolved citations/references detected in the final LaTeX log, Type 1 embedded fonts.
- `scripts/anonymize_check.py`: passed with zero flagged files.
- `scripts/validate_claims.py`: passed.
- `scripts/validate_paper.py`: passed for the project-level validator.

## Scientific readiness assessment

The package is submission-shaped and sanitized, but the science is still borderline for a top venue unless the authors address the remaining empirical concerns before final upload.

High-priority improvements before submission:

1. Add paired stationary-bootstrap confidence intervals for DHRP versus HRP, risk parity, and mean-variance in every universe. PSR only supports positive sample Sharpe; it is not a paired baseline test.
2. Make the ICAIF paper fuller if time allows. The current primary version fits well under the 8-page total limit, but reviewers may expect more detail. Add one more figure or a paired-test table if allowed.
3. Confirm that all reported HRP numbers use the intended classical HRP implementation. If the HRP baseline is changed to single-linkage HRP, regenerate all DHRP-vs-HRP claims.
4. Validate the Croissant metadata with the current MLCommons/Croissant validator and ensure the public anonymous URL is accessible without credentials.
5. Decide whether to advertise 8 headline methods or 12 implemented allocators. The current text uses both deliberately: 8 headline methods are reported across the full benchmark, while 12 allocators are implemented.
6. Do not overstate the factor alpha. Commodity DHRP FF3 alpha is 6.3%, t=1.58, not significant; the strongest DHRP claim is sample Sharpe/PSR, not alpha.
7. Keep the LLM result narrow: global FinBERT summary does not improve DHRP under this protocol; the paper does not rule out per-asset text attention or retrieval-augmented event models.

## Suggested submission framing

Use this one-sentence positioning:

> DHRP-8 is a reproducible AI-in-finance benchmark and DHRP is a differentiable, HRP-inspired allocation layer that performs strongly in commodities and crypto while exposing where hierarchical allocation and global LLM text summaries fail.

Avoid this framing:

> DHRP is universally superior to prior allocation methods or is an exact implementation of every classical HRP step.
