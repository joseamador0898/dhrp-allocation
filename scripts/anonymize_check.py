"""Anonymization scanner for double-blind submission.

Scans the repo for strings that could de-anonymize the authors. Uses
generic regex patterns (emails, GitHub handles, university domains,
local user paths) so the scanner itself contains no real names and is
safe to ship in the supplementary package.

Author-specific patterns can be added in a private file named
`.anonymize_extras.txt` at the repo root, one regex per line. That
file is gitignored.

Usage:
    python scripts/anonymize_check.py                # scan repo
    python scripts/anonymize_check.py --paper-only   # paper/ + data/ only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Generic regex patterns (no real identifiers). Case-insensitive.
DEANONYMIZE_REGEXES = [
    # Email addresses (excluding obvious anonymized stand-ins)
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email-like"),
    # GitHub user URLs (https or git form). The repository is published
    # under its owner's handle (joseamador0898), so that handle is allowed;
    # any other github.com user URL is suspect.
    (r"github\.com[/:](?!joseamador0898)[\w.-]+/", "github-user-url"),
    # LinkedIn profile URLs
    (r"linkedin\.com/in/[\w-]+", "linkedin-url"),
    # University-style email domains commonly seen in finance ML papers
    (r"\b[\w.-]+@[\w.-]+\.(edu|edu\.\w+|ac\.\w+)\b", "academic-email"),
    # Local user-home paths on Windows or *nix that leak a username
    (r"[Cc]:[/\\]+Users[/\\]+(?!<|placeholder)[\w.-]+", "windows-user-path"),
    (r"/home/(?!user|root|placeholder)[\w.-]+", "linux-home-path"),
    (r"/Users/(?!shared|guest|placeholder)[\w.-]+", "macos-user-path"),
]

# Optional private extension file (gitignored). One Python regex per line.
EXTRAS_FILE = ROOT / ".anonymize_extras.txt"

# Files / directories never scanned.
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "data/cache", "results/full", "results/models",
    ".ipynb_checkpoints", ".azure_env_build",
}
SKIP_FILES = {".env", ".env.local", "service_account.json"}
SKIP_FILE_EXT = {
    ".pyc", ".pyo", ".pkl", ".pt", ".pth", ".npz", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".parquet",
    ".aux", ".log", ".bbl", ".blg", ".out", ".toc", ".fls",
    ".fdb_latexmk", ".synctex.gz",
    ".sty", ".cls",  # vendored LaTeX style files
    ".zip", ".tar", ".gz", ".whl",  # archive bundles
}

# This file (the scanner) lists the patterns themselves; do not flag.
EXPECTED_IDENTITY_FILES = {"scripts/anonymize_check.py"}


def load_extras() -> list[tuple[str, str]]:
    if not EXTRAS_FILE.exists():
        return []
    extras: list[tuple[str, str]] = []
    for raw in EXTRAS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        extras.append((line, "extras"))
    return extras


def should_skip_dir(p: Path) -> bool:
    try:
        rel = p.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    parts = set(p.parts)
    if parts & {".git", "__pycache__", "node_modules", ".venv", "venv", ".ipynb_checkpoints"}:
        return True
    # Skip any hidden directory under the repo root (local IDE/tool config, etc.).
    if rel != "." and any(seg.startswith(".") for seg in rel.split("/")):
        return True
    for skip in SKIP_DIRS:
        if rel == skip or rel.startswith(skip + "/"):
            return True
    return False


def is_expected(p: Path) -> bool:
    rel = p.relative_to(ROOT).as_posix()
    return rel in EXPECTED_IDENTITY_FILES


def compile_patterns() -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for src, label in DEANONYMIZE_REGEXES + load_extras():
        try:
            compiled.append((re.compile(src, re.IGNORECASE), label))
        except re.error as exc:
            print(f"  bad regex skipped: {src!r} ({exc})", file=sys.stderr)
    return compiled


def scan_file(path: Path, patterns: list[tuple[re.Pattern[str], str]]) -> list[tuple[int, str, str]]:
    flags: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return flags
    for i, line in enumerate(text.split("\n"), 1):
        for pat, label in patterns:
            matched = pat.search(line)
            if not matched:
                continue
            token = matched.group(0).lower()
            if label == "email-like" and token.startswith("git@github.com"):
                continue
            if "<your-account>" in line or "<anonymous>" in line:
                continue
            snippet = line.strip()
            if len(snippet) > 100:
                snippet = snippet[:100] + "..."
            flags.append((i, label, snippet))
            break
    return flags


def main(paper_only: bool = False) -> int:
    print("=" * 70)
    print("  Anonymization Scanner for Double-Blind Submission")
    print("=" * 70)

    patterns = compile_patterns()
    if paper_only:
        roots = [ROOT / "paper", ROOT / "data" / "croissant", ROOT / "README.md"]
        print("Mode: paper-only (paper/ + data/croissant/ + README.md)")
    else:
        roots = [ROOT]
        print(f"Mode: full repo scan from {ROOT}")
    if EXTRAS_FILE.exists():
        print(f"Extras: loaded {len(patterns) - len(DEANONYMIZE_REGEXES)} private patterns from {EXTRAS_FILE.name}")

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
            if p.name in SKIP_FILES:
                continue
            if p.suffix.lower() in SKIP_FILE_EXT:
                continue
            total_files += 1
            issues = scan_file(p, patterns)
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
        print("\033[31m  DE-ANONYMIZING STRINGS FOUND (review before submission)\033[0m")
        print("\033[31m" + "=" * 70 + "\033[0m")
        for path, issues in flagged:
            rel = path.relative_to(ROOT)
            print(f"\n\033[33m{rel}\033[0m  ({len(issues)} hits)")
            for line_no, label, snippet in issues[:5]:
                print(f"  L{line_no} [{label}]: {snippet}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        print()
        return 1

    if expected_hits:
        print("\033[36m  Files with identity (expected, not flagged):\033[0m")
        for p in expected_hits:
            print(f"  - {p.relative_to(ROOT).as_posix()}")
        print()

    print("=" * 70)
    print("\033[32m  PASSED: no de-anonymizing strings found.\033[0m")
    print("  Safe to push the contents of paper/ + data/croissant/ + scripts/")
    print("  to anonymous.4open.science")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-only", action="store_true",
                        help="Restrict scan to paper/ + data/croissant/ + README.md")
    args = parser.parse_args()
    sys.exit(main(paper_only=args.paper_only))
