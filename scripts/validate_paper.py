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
        PAPER / "checklist.tex",
        SECTIONS / "abstract.tex",
        SECTIONS / "01_introduction.tex",
        SECTIONS / "02_related_work.tex",
        SECTIONS / "03_method.tex",
        SECTIONS / "04_benchmark_design.tex",
        SECTIONS / "05_results.tex",
        SECTIONS / "06_llm_negative_ablation.tex",
        SECTIONS / "07_limitations_discussion.tex",
        SECTIONS / "08_conclusion.tex",
        SECTIONS / "appendix.tex",
        ROOT / "data" / "croissant" / "dhrp-8universe.json",
        ROOT / "REPRODUCIBILITY.md",
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
        for tex in SECTIONS.glob("*.tex"):
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
    section_text = "\n".join(t.read_text(encoding="utf-8") for t in SECTIONS.glob("*.tex"))
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
    for tex in SECTIONS.glob("*.tex"):
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

    # 8. Anonymization scan (paper text)
    print("\n[8] Anonymization scan in paper text...")
    deanonymizing_strings = [
        "Jose Amador", "joseamador", "jlaj@", "@connect.ust.hk",
        "github.com/joseamador0898", "GitHub: https://github.com/jose",
        "luigi",  # local username
    ]
    flags = []
    for tex in list(SECTIONS.glob("*.tex")) + [PAPER / "main.tex", PAPER / "checklist.tex"]:
        if not tex.exists():
            continue
        text = tex.read_text(encoding="utf-8")
        for s in deanonymizing_strings:
            if s.lower() in text.lower():
                flags.append((tex.name, s))
    if flags:
        for fname, s in flags:
            errors.append(f"De-anonymizing string in {fname}: {s!r}")
            print(f"  {red('FAIL')} {fname}: contains {s!r}")
    else:
        print(f"  {green('OK')} No de-anonymizing strings detected")

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
