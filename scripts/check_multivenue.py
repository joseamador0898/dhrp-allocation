"""Validate paper/multivenue/body.tex + bibliography.tex + appendix.tex
for cross-venue portability."""
import re
from pathlib import Path

base = Path('paper/multivenue')
body = (base / 'body.tex').read_text(encoding='utf-8')
bib = (base / 'bibliography.tex').read_text(encoding='utf-8')
app = (base / 'appendix.tex').read_text(encoding='utf-8')

full = body + '\n' + app + '\n' + bib

# 1. Citations vs bibitems
cite_pattern = re.compile(r'\\cite[pt]?\{([^}]+)\}')
refs = set()
for m in cite_pattern.finditer(full):
    for k in m.group(1).split(','):
        refs.add(k.strip())
bib_pattern = re.compile(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}')
defined = set(m.group(1).strip() for m in bib_pattern.finditer(full))
print(f'Citations used: {len(refs)}')
print(f'Bibitems defined: {len(defined)}')
print(f'Undefined refs (BAD): {sorted(refs - defined)}')
print(f'Unused bibitems: {sorted(defined - refs)}')

# 2. Brace balance
opens = full.count('{')
closes = full.count('}')
print(f'Braces: {opens} open / {closes} close, diff={opens - closes}')

# 3. Labels vs refs
labels = set(re.findall(r'\\label\{([^}]+)\}', full))
rused = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', full))
print(f'Labels: {len(labels)}, refs_used: {len(rused)}')
print(f'Refs missing labels (BAD): {sorted(rused - labels)}')

# 4. Portability checks on BODY only (wrappers handle preamble)
print()
print('--- Portability checks on body.tex ---')

# raw \cite{} (should be 0 — use \citep / \citet)
raw_cites = re.findall(r'\\cite\{[^}]+\}', body)
print(f'Raw \\cite{{}} in body (should be 0 for KDD/acmart compat): {len(raw_cites)}')

# \textwidth or \columnwidth in body (should be 0; use \linewidth)
tw = re.findall(r'\\textwidth|\\columnwidth', body)
print(f'\\textwidth / \\columnwidth in body (should be 0): {len(tw)}')

# \usepackage{} in body (should be 0 — packages live in wrappers)
upkg = re.findall(r'\\usepackage', body)
print(f'\\usepackage in body (should be 0): {len(upkg)}')

# \geometry / \fullpage / margin overrides (should be 0)
margin = re.findall(r'\\geometry|\\fullpage|\\setlength\{\\textwidth', body)
print(f'Margin overrides in body (should be 0): {len(margin)}')

# \maketitle / \title / \author / \begin{document} (should be 0 — wrappers own these)
meta = re.findall(r'\\maketitle|\\title\{|\\author\{|\\begin\{document\}|\\end\{document\}', body)
print(f'Document-level meta in body (should be 0): {len(meta)}')

# 5. Resizebox (uses \textwidth — flag for review)
rbox = re.findall(r'\\resizebox\{[^}]+\}', body)
print(f'\\resizebox calls in body (should use \\linewidth): {len(rbox)}')
for r in rbox:
    print(f'  {r}')
