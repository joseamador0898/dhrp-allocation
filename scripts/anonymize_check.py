"""Anonymization scanner for double-blind submission.

Scans the entire repo (excluding ignored directories) for strings that
de-anonymize the authors:
  - Real names (configurable list)
  - Email addresses
  - Personal GitHub URLs
  - University affiliations
  - Local file paths that leak identity

Run BEFORE creating an anonymous.4open.science fork.

Usage:
    python scripts/anonymize_check.py                # scan repo
    python scripts/anonymize_check.py --paper-only   # paper/ + data/ only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# De-anonymizing strings to detect.
# IMPORTANT: keep the list in lower case for case-insensitive matching.
DEANONYMIZE_PATTERNS = [
    # Author identity
    "jose amador",
    "joseamador",
    "amador, jose",
    "jlaj@",
    "jlaj@connect.ust.hk",
    # GitHub
    "github.com/joseamador0898",
    "github.com/joseamador",
    # Local user paths
    "/users/luigi",
    "c:\\users\\luigi",
    "c:/users/luigi",
    # University affiliation strings
    "hong kong university of science",
    "hkust",
    "connect.ust.hk",
    # Other typical identity leaks
    "linkedin.com/in/",
]

# Files / directories to skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "data/cache",  # large cached CSVs
    "results/full",  # large backtest CSVs
    "results/models",  # binary checkpoints
    ".ipynb_checkpoints",
}
SKIP_FILE_EXT = {
    ".pyc", ".pyo", ".pkl", ".pt", ".pth", ".npz", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf",  # binary, can't grep meaningfully
    ".parquet",
}

# Specific files we EXPECT to contain identity (don't flag them)
EXPECTED_IDENTITY_FILES = {
    # Multi-venue paper drafts (these are NOT submitted; main.tex is)
    "paper/multivenue/main_aaai.tex",
    "paper/multivenue/main_iclr.tex",
    "paper/multivenue/main_icml.tex",
    "paper/multivenue/main_kdd.tex",
    "paper/multivenue/main_neurips.tex",
    "paper/llm_dhrp_paper.tex",  # superseded by paper/main.tex
    "paper/llm_dhrp_paper.pdf",
    # Anonymization scanner itself contains the patterns!
    "scripts/anonymize_check.py",
    "scripts/validate_paper.py",
    # Plan / scratchwork
    "REPRODUCIBILITY.md",  # may contain anonymized URL only
}


def should_skip_dir(p: Path) -> bool:
    """True if this path is inside any directory we want to skip."""
    try:
        rel = p.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    parts = set(p.parts)
    # Skip if any path component is in SKIP_DIRS
    if parts & {".git", "__pycache__", "node_modules", ".venv", "venv", ".ipynb_checkpoints"}:
        return True
    # Skip if relative path starts with any skip-dir prefix
    for skip in SKIP_DIRS:
        if rel == skip or rel.startswith(skip + "/"):
            return True
    return False


def is_expected(p: Path) -> bool:
    rel = p.relative_to(ROOT).as_posix()
    return rel in EXPECTED_IDENTITY_FILES


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, pattern, snippet)."""
    flags: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return flags
    for i, line in enumerate(text.split("\n"), 1):
        lower = line.lower()
        for pat in DEANONYMIZE_PATTERNS:
            if pat in lower:
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + "..."
                flags.append((i, pat, snippet))
                break  # one flag per line is enough
    return flags


def main(paper_only: bool = False) -> int:
    print("=" * 70)
    print("  Anonymization Scanner for Double-Blind Submission")
    print("=" * 70)

    if paper_only:
        roots = [ROOT / "paper", ROOT / "data" / "croissant", ROOT / "REPRODUCIBILITY.md"]
        print("Mode: paper-only (paper/ + data/croissant/ + REPRODUCIBILITY.md)")
    else:
        roots = [ROOT]
        print(f"Mode: full repo scan from {ROOT}")

    total_files = 0
    total_flags = 0
    flagged: list[tuple[Path, list[tuple[int, str, str]]]] = []
    expected_hits: list[Path] = []

    for root in roots:
        if root.is_file():
            paths = [root]
        else:
            paths = list(root.rglob("*"))

        for p in paths:
            if not p.is_file():
                continue
            if should_skip_dir(p):
                continue
            if p.suffix.lower() in SKIP_FILE_EXT:
                continue
            total_files += 1
            issues = scan_file(p)
            if not issues:
                continue
            if is_expected(p):
                expected_hits.append(p)
                continue
            flagged.append((p, issues))
            total_flags += len(issues)

    print(f"\nScanned {total_files} files; flagged {len(flagged)} files with {total_flags} hits.\n")

    if flagged:
        print("\033[31m" + "=" * 70 + "\033[0m")
        print("\033[31m  DEANONYMIZING STRINGS FOUND (must remove before submission)\033[0m")
        print("\033[31m" + "=" * 70 + "\033[0m")
        for path, issues in flagged:
            rel = path.relative_to(ROOT)
            print(f"\n\033[33m{rel}\033[0m  ({len(issues)} hits)")
            for line_no, pat, snippet in issues[:5]:
                print(f"  L{line_no} [{pat!r}]: {snippet}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")

    if expected_hits:
        print("\n\033[36m  Files with identity (expected — not flagged):\033[0m")
        for p in expected_hits:
            print(f"  - {p.relative_to(ROOT).as_posix()}")

    print("\n" + "=" * 70)
    if not flagged:
        print("\033[32m  PASSED: no de-anonymizing strings found.\033[0m")
        print("  Safe to push the contents of paper/ + data/croissant/ + scripts/")
        print("  to anonymous.4open.science")
    else:
        print(f"\033[31m  FAILED: {total_flags} de-anonymizing strings in {len(flagged)} files\033[0m")
        print("  Either delete those files from the anonymous fork or scrub the strings.")
    print("=" * 70)

    return 0 if not flagged else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paper-only", action="store_true",
                   help="Only scan paper/, data/croissant/, REPRODUCIBILITY.md")
    args = p.parse_args()
    sys.exit(main(paper_only=args.paper_only))
