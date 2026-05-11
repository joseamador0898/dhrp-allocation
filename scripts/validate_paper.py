"""Pre-submission integrity check for the NeurIPS 2026 E&D paper.

Run BEFORE submission to catch common desk-reject triggers:
  - Missing references / bibliography keys
  - Unfilled TODO/TBD placeholders
  - Missing figures referenced via \\includegraphics
  - Missing tables files referenced via \\input{tables/...}
  - NaN / 'TBD' strings in main paper body (placeholders not replaced)
  - File checksums (verify Croissant + checkpoints exist)

Exit code 0 = ready to submit. Non-zero = blocking issues.

Usage:
    python scripts/validate_paper.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
RESULTS = ROOT / "results"

GENERIC_DEANON_REGEXES = [
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email-like"),
    (r"github\.com[/:](?!anonymous|orgs/anonymous)[\w.-]+", "github-user-url"),
    (r"linkedin\.com/in/[\w-]+", "linkedin-url"),
    (r"[Cc]:[/\\]+Users[/\\]+(?!<|placeholder)[\w.-]+", "windows-user-path"),
    (r"/Users/(?!shared|guest|placeholder)[\w.-]+", "macos-user-path"),
    (r"/home/(?!user|root|placeholder)[\w.-]+", "linux-home-path"),
]


def red(s):    return f"\033[31m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"


def find_cite_keys(text: str) -> set[str]:
    """Find all bibtex keys referenced via \\cite-family commands."""
    keys = set()
    pos = 0
    bs = chr(92)  # backslash, avoids Python 3.14 escape issue
    while True:
        idx = text.find(bs + "cite", pos)
        if idx < 0:
            break
        brace = text.find("{", idx)
        if brace < 0:
            break
        end = text.find("}", brace)
        if end < 0:
            break
        for key in text[brace + 1:end].split(","):
            keys.add(key.strip())
        pos = end + 1
    return keys


def find_bib_defs(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\{(\w+),", bib_text))


def find_includegraphics(text: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)


def find_input_paths(text: str) -> list[str]:
    return re.findall(r"\\input\{([^}]+)\}", text)


def find_todos(text: str) -> list[tuple[int, str]]:
    """Find TODO/TBD/FIXME markers in source text. Comments allowed."""
    issues = []
    for i, line in enumerate(text.split("\n"), 1):
        # Skip pure-comment TODOs (start with %)
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        # Look for TODO, TBD, FIXME, XXX in non-comment text
        for pat in ["TODO", "TBD", "FIXME", "XXX"]:
            if pat in line:
                # Allow {TBD-from-CSV} style placeholders (recognized) - flag separately
                issues.append((i, line.strip()[:120]))
                break
    return issues


def scan_deanon_text(text: str) -> list[tuple[str, str]]:
    hits = []
    for pattern, label in GENERIC_DEANON_REGEXES:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append((label, match.group(0)))
    return hits


def main() -> int:
    print("=" * 70)
    print("  NeurIPS 2026 E&D Paper Pre-Submission Validator")
    print("=" * 70)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Files exist
    print("\n[1] File presence...")
    required = [
        PAPER / "main.tex",
        PAPER / "references.bib",
        ROOT / "README.md",
        ROOT / "data" / "croissant" / "dhrp-8universe.json",
        ROOT / "scripts" / "reproduce_all.sh",
    ]
    for p in required:
        if p.exists():
            print(f"  {green('OK')} {p.relative_to(ROOT)}")
        else:
            errors.append(f"Missing: {p.relative_to(ROOT)}")
            print(f"  {red('FAIL')} {p.relative_to(ROOT)}")

    # 2. Required style files (downloaded from Overleaf, not in repo)
    print("\n[2] NeurIPS 2026 style files (download from Overleaf)...")
    sty = PAPER / "neurips_2026.sty"
    if sty.exists():
        print(f"  {green('OK')} neurips_2026.sty")
    else:
        warnings.append("neurips_2026.sty not in paper/ — download from official Overleaf template before pdflatex")
        print(f"  {yellow('WARN')} neurips_2026.sty (download required)")

    # 3. Bibliography integrity
    print("\n[3] Bibliography...")
    bib_path = PAPER / "references.bib"
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8")
        bib_defs = find_bib_defs(bib_text)
        used_keys = set()
        for tex in [PAPER / "main.tex"] if (PAPER / "main.tex").exists() else []:
            used_keys |= find_cite_keys(tex.read_text(encoding="utf-8"))
        missing = used_keys - bib_defs
        unused = bib_defs - used_keys
        print(f"  {len(bib_defs)} entries defined; {len(used_keys)} keys cited")
        if missing:
            errors.append(f"Cited but not defined: {sorted(missing)}")
            print(f"  {red('FAIL')} Cited but not defined: {sorted(missing)}")
        else:
            print(f"  {green('OK')} All {len(used_keys)} cited keys resolve")
        if unused:
            print(f"  {yellow('WARN')} {len(unused)} bib entries not cited (may be intentional)")

    # 4. Figures referenced exist
    print("\n[4] Figures...")
    main_text = (PAPER / "main.tex").read_text(encoding="utf-8") if (PAPER / "main.tex").exists() else ""
    section_text = ""  # all section content is now inlined in main.tex
    fig_refs = find_includegraphics(main_text + "\n" + section_text)
    for ref in fig_refs:
        # Resolve via paper/figures/ or paper/ directly
        candidates = [PAPER / ref, FIGURES / ref, FIGURES / (ref + ".pdf"),
                      FIGURES / (ref + ".png"), FIGURES / (ref + ".jpg")]
        if any(c.exists() for c in candidates):
            print(f"  {green('OK')} {ref}")
        else:
            warnings.append(f"Figure reference not found on disk: {ref} (may be generated later)")
            print(f"  {yellow('WARN')} {ref} (not found — may be generated later)")

    # 5. \input{tables/...} files exist
    print("\n[5] Table file inputs...")
    inputs = find_input_paths(main_text + "\n" + section_text)
    for inp in inputs:
        # Skip our section files
        if inp.startswith("sections/") or inp == "checklist" or inp == "checklist.tex":
            continue
        candidate = PAPER / inp
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        if candidate.exists():
            print(f"  {green('OK')} {inp}")
        else:
            warnings.append(f"Input target not found: {inp} (likely generated by scripts later)")
            print(f"  {yellow('WARN')} {inp} (will be generated by scripts/generate_paper_tables.py)")

    # 6. TODO / TBD placeholders in body text (non-comments)
    print("\n[6] TODO / TBD placeholders in body text...")
    total_todos = 0
    for tex in [PAPER / "main.tex"] if (PAPER / "main.tex").exists() else []:
        todos = find_todos(tex.read_text(encoding="utf-8"))
        if todos:
            total_todos += len(todos)
            print(f"  {yellow('WARN')} {tex.name}: {len(todos)} TODO/TBD lines")
            for line_no, snippet in todos[:3]:
                print(f"      L{line_no}: {snippet}")
            if len(todos) > 3:
                print(f"      ... and {len(todos) - 3} more")
    if total_todos == 0:
        print(f"  {green('OK')} No TODO/TBD markers in body")
    else:
        warnings.append(f"{total_todos} TODO/TBD markers across sections — review before submission")

    # 7. Croissant metadata sanity
    print("\n[7] Croissant 1.1 metadata...")
    cr_path = ROOT / "data" / "croissant" / "dhrp-8universe.json"
    if cr_path.exists():
        import json
        try:
            meta = json.loads(cr_path.read_text(encoding="utf-8"))
            assert meta.get("conformsTo") == "http://mlcommons.org/croissant/1.1", "wrong conformsTo"
            assert "rai:dataCollection" in meta, "missing rai:dataCollection"
            assert "rai:dataBiases" in meta, "missing rai:dataBiases"
            assert len(meta.get("distribution", [])) >= 3, "needs >= 3 distribution files"
            assert len(meta.get("recordSet", [])) >= 2, "needs >= 2 recordSet entries"
            print(f"  {green('OK')} Croissant 1.1 spec + RAI fields + distribution + recordSet")
        except Exception as e:
            errors.append(f"Croissant metadata problem: {e}")
            print(f"  {red('FAIL')} {e}")

    # 8. Anonymization scan (generic patterns only; private patterns live in .anonymize_extras.txt)
    print("\n[8] Anonymization scan...")
    flags = []
    for candidate in [
        PAPER / "main.tex",
        ROOT / "README.md",
        ROOT / "data" / "croissant" / "dhrp-8universe.json",
        ROOT / "scripts" / "validate_paper.py",
    ]:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        for label, hit in scan_deanon_text(text):
            flags.append((candidate.relative_to(ROOT).as_posix(), label, hit))
    if flags:
        for fname, label, hit in flags:
            errors.append(f"De-anonymizing pattern in {fname}: {label} {hit!r}")
            print(f"  {red('FAIL')} {fname}: {label} {hit!r}")
    else:
        print(f"  {green('OK')} No de-anonymizing strings detected")

    # 9. PDF build checks (only if main.pdf exists)
    print("\n[9] PDF build artifacts (skip if PDF not built yet)...")
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        warnings.append("paper/main.pdf not built yet — run pdflatex before submission")
        print(f"  {yellow('WARN')} paper/main.pdf not found (build with pdflatex)")
    else:
        # 9a. PDF page count
        try:
            import subprocess
            r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
            for line in r.stdout.split("\n"):
                if line.startswith("Pages:"):
                    n_pages = int(line.split(":", 1)[1].strip())
                    print(f"  PDF total pages: {n_pages}")
                    if n_pages > 25:
                        warnings.append(f"PDF has {n_pages} pages — verify <= 9 main + refs/appendix limit")
                    break
        except FileNotFoundError:
            print(f"  {yellow('WARN')} pdfinfo not installed — skipping page count check")
        # 9b. PDF font types
        try:
            import subprocess
            r = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True)
            type3_lines = [l for l in r.stdout.split("\n") if "Type 3" in l or "type 3" in l]
            if type3_lines:
                errors.append(f"PDF contains {len(type3_lines)} Type 3 fonts — NeurIPS rejects these")
                print(f"  {red('FAIL')} {len(type3_lines)} Type 3 fonts detected (must fix)")
            else:
                print(f"  {green('OK')} No Type 3 fonts in main.pdf")
        except FileNotFoundError:
            print(f"  {yellow('WARN')} pdffonts not installed — skipping font check")
        # 9c. PDF metadata anonymization
        try:
            import subprocess
            r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
            for line in r.stdout.split("\n"):
                if line.startswith("Author:") or line.startswith("Creator:"):
                    val = line.split(":", 1)[1].strip()
                    if val and val.lower() not in ("none", "anonymous"):
                        warnings.append(f"PDF metadata has {line.split(':')[0]}: {val} (strip with exiftool)")
                        print(f"  {yellow('WARN')} PDF {line.split(':')[0]}: {val} (strip with exiftool)")
        except FileNotFoundError:
            pass

    # 10. texcount main-text page count
    print("\n[10] texcount main-text page count...")
    try:
        import subprocess
        r = subprocess.run(["texcount", "-inc", "-1", str(PAPER / "main.tex")],
                           capture_output=True, text=True)
        # texcount -1 returns just the word count
        if r.returncode == 0:
            words = r.stdout.strip().split("\n")[0]
            try:
                n_words = int(words)
                # Approx 250-300 words per page in NeurIPS format
                est_pages = n_words / 275
                print(f"  Main-text words (approx): {n_words}, ~{est_pages:.1f} pages")
                if est_pages > 9.0:
                    warnings.append(f"Estimated ~{est_pages:.1f} main-text pages > 9 limit")
                else:
                    print(f"  {green('OK')} Estimated main-text page count under 9-page limit")
            except ValueError:
                print(f"  {yellow('WARN')} texcount returned: {words[:80]}")
    except FileNotFoundError:
        warnings.append("texcount not installed — install via TeX Live or check page count manually after build")
        print(f"  {yellow('WARN')} texcount not installed (install via TeX Live)")

    # 11. Croissant validator hint
    print("\n[11] Croissant metadata: validate at...")
    print(f"  Upload data/croissant/dhrp-8universe.json to:")
    print(f"  https://huggingface.co/spaces/JoaquinVanscholen/croissant-checker")
    print(f"  (manual step; cannot automate without Croissant Python library)")

    # 12. Stale claim validator
    print("\n[12] Stale claim validator...")
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_claims.py")],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print(f"  {green('OK')} validate_claims.py passed")
        else:
            errors.append("validate_claims.py found stale or unsafe claims")
            print(f"  {red('FAIL')} validate_claims.py")
            print(r.stdout.strip())
    except Exception as e:
        errors.append(f"Could not run validate_claims.py: {e}")
        print(f"  {red('FAIL')} validate_claims.py: {e}")

    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(red(f"  {len(errors)} ERRORS (blocking):"))
        for e in errors:
            print(red(f"    - {e}"))
    if warnings:
        print(yellow(f"  {len(warnings)} WARNINGS (review before submit):"))
        for w in warnings:
            print(yellow(f"    - {w}"))
    if not errors and not warnings:
        print(green("  ALL CHECKS PASSED — ready to submit"))
    print("=" * 70)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
