"""Fail fast on stale submission claims and de-anonymizing patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLAIM_FILES = [
    ROOT / "README.md",
    ROOT / "paper" / "main.tex",
    ROOT / "data" / "croissant" / "dhrp-8universe.json",
    ROOT / "notebooks" / "llm_dhrp_experiments.ipynb",
    ROOT / "src" / "data.py",
    ROOT / "src" / "models.py",
    ROOT / "src" / "training.py",
    ROOT / "src" / "evaluation.py",
]

FORBIDDEN_PATTERNS = [
    (r"\b7\s*/\s*8\b", "stale 7/8 HRP-win claim"),
    (r"\b7\s+of\s+8\b", "stale 7 of 8 HRP-win claim"),
    (r"t-statistic\s+above\s+2\.0", "stale commodity alpha checklist"),
    (r"FF3\s*\+\s*AQR", "stale FF3+AQR headline claim"),
    (r"\bAQR\s+(?:commodity\s+)?(?:factor\s+)?alphas?\b", "stale AQR-alpha claim"),
    (r"\bQwen3\b", "stale Qwen3 notebook/text claim"),
    (r"\b5\s+seeds?\b", "stale 5-seed claim"),
    (r"\bFF6\b", "stale FF6 claim"),
    (r"recovers\s+classical\s+HRP", "overstrong HRP-recovery claim"),
    (r"HRP\s+recovery\s+in\s+the\s+\$?\\tau", "overstrong HRP-recovery theorem wording"),
]

DEANONYMIZING_REGEXES = [
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email-like"),
    (r"github\.com[/:](?!anonymous|orgs/anonymous)[\w.-]+", "github-user-url"),
    (r"linkedin\.com/in/[\w-]+", "linkedin-url"),
    (r"[Cc]:[/\\]+Users[/\\]+(?!<|placeholder)[\w.-]+", "windows-user-path"),
    (r"/Users/(?!shared|guest|placeholder)[\w.-]+", "macos-user-path"),
    (r"/home/(?!user|root|placeholder)[\w.-]+", "linux-home-path"),
]


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    if not path.exists():
        return hits
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), 1):
        for pattern, label in FORBIDDEN_PATTERNS + DEANONYMIZING_REGEXES:
            if re.search(pattern, line, flags=re.IGNORECASE):
                snippet = line.strip()
                if len(snippet) > 140:
                    snippet = snippet[:140] + "..."
                hits.append((line_no, label, snippet))
    return hits


def main() -> int:
    failures = []
    for path in CLAIM_FILES:
        hits = _scan_file(path)
        if hits:
            failures.append((path, hits))

    if failures:
        print("Stale or unsafe submission claims found:")
        for path, hits in failures:
            rel = path.relative_to(ROOT)
            print(f"\n{rel}")
            for line_no, label, snippet in hits[:20]:
                print(f"  L{line_no} [{label}]: {snippet}")
            if len(hits) > 20:
                print(f"  ... and {len(hits) - 20} more")
        return 1

    print("Claim validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
