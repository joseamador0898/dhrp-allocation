"""One-shot: extract multi-seed CSVs from notebook Cell 23 text output.

The Colab run produced the multi-seed pivot tables but the actual CSVs
weren't synced back to this local repo. The pivot tables are however
visible in the notebook cell outputs as plain text. Parse them here.

Usage:
    python scripts/_extract_csvs_from_notebook.py

Writes to results/:
    sharpe_pivot_multiseed_mean.csv
    sharpe_pivot_multiseed_std.csv
    psr_pivot_multiseed.csv
    all_universes_multiseed_summary.csv
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "llm_dhrp_experiments.ipynb"
RESULTS = ROOT / "results"


def get_cell23_text() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    parts = []
    for o in nb["cells"][23].get("outputs", []):
        if "text" in o:
            t = o["text"]
            parts.append("".join(t) if isinstance(t, list) else t)
    return "".join(parts)


def parse_pivot(block: str) -> pd.DataFrame:
    """Parse a fixed-width pivot table printed by pandas to_string()."""
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if not lines:
        return pd.DataFrame()
    header = lines[0]
    # Tokenize header: first token is "Universe", rest are universe names
    cols = header.split()
    # The first column ("Universe") is actually the row-index header; column
    # names are in cols[1:]. The next line "Method" indicates the row index.
    if cols[0] == "Universe":
        col_names = cols[1:]
    else:
        col_names = cols
    # Skip the second line if it's just "Method" alone
    body_start = 1
    if body_start < len(lines) and lines[body_start].strip() == "Method":
        body_start = 2
    rows = []
    for ln in lines[body_start:]:
        toks = ln.split()
        if len(toks) < len(col_names) + 1:
            continue
        method = toks[0]
        # Remaining tokens are the values, with NaN as "NaN"
        vals = toks[-len(col_names):]
        parsed = []
        for v in vals:
            if v.lower() in ("nan", "n/a"):
                parsed.append(float("nan"))
            else:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(float("nan"))
        rows.append([method] + parsed)
    df = pd.DataFrame(rows, columns=["Method"] + col_names)
    df = df.set_index("Method")
    return df


def main() -> None:
    text = get_cell23_text()

    # Find the three blocks
    def slice_block(start_marker: str, end_marker: str | None) -> str:
        s = text.find(start_marker)
        if s < 0:
            return ""
        s = text.find("\n", s) + 1  # skip the marker line itself
        if end_marker:
            e = text.find(end_marker, s)
            if e < 0:
                return text[s:]
            return text[s:e]
        return text[s:]

    mean_block = slice_block("Mean Sharpe by Method x Universe", "Std Sharpe by Method")
    std_block = slice_block("Std Sharpe by Method x Universe", "Probabilistic Sharpe Ratio")
    psr_block = slice_block("Probabilistic Sharpe Ratio (PSR > 0.95",
                             "Multi-seed multi-universe expansion complete")

    mean_df = parse_pivot(mean_block)
    std_df = parse_pivot(std_block)
    psr_df = parse_pivot(psr_block)

    print(f"Mean pivot:  {mean_df.shape}")
    print(f"Std pivot:   {std_df.shape}")
    print(f"PSR pivot:   {psr_df.shape}")
    print()
    print("Mean preview:")
    print(mean_df.round(3).to_string())

    if mean_df.empty or std_df.empty or psr_df.empty:
        print("ERROR: failed to parse one or more pivot tables")
        return

    # Write CSVs
    RESULTS.mkdir(exist_ok=True)
    mean_df.to_csv(RESULTS / "sharpe_pivot_multiseed_mean.csv")
    std_df.to_csv(RESULTS / "sharpe_pivot_multiseed_std.csv")
    psr_df.to_csv(RESULTS / "psr_pivot_multiseed.csv")
    print(f"\nWrote: {RESULTS / 'sharpe_pivot_multiseed_mean.csv'}")
    print(f"Wrote: {RESULTS / 'sharpe_pivot_multiseed_std.csv'}")
    print(f"Wrote: {RESULTS / 'psr_pivot_multiseed.csv'}")

    # Build long-format summary for the LLM ablation table
    rows = []
    for u in mean_df.columns:
        for m in mean_df.index:
            mv = mean_df.loc[m, u]
            sv = std_df.loc[m, u] if (m in std_df.index and u in std_df.columns) else float("nan")
            pv = psr_df.loc[m, u] if (m in psr_df.index and u in psr_df.columns) else float("nan")
            if pd.isna(mv):
                continue
            rows.append({"Method": m, "Universe": u,
                         "Sharpe_mean": mv, "Sharpe_std": sv, "PSR_mean": pv})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(RESULTS / "all_universes_multiseed_summary.csv", index=False)
    print(f"Wrote: {RESULTS / 'all_universes_multiseed_summary.csv'} ({len(long_df)} rows)")


if __name__ == "__main__":
    main()
