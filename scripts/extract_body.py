"""Extract the venue-agnostic body of results/hrp_paper_fixed.tex
into paper/multivenue/body.tex.

The body excludes the preamble (everything up to and including \\maketitle),
the \\end{document}, and the \\bibliography section, which is reattached by each
per-venue wrapper.
"""
import re
from pathlib import Path

src_path = Path('results/hrp_paper_fixed.tex')
dst_path = Path('paper/multivenue/body.tex')
dst_path.parent.mkdir(parents=True, exist_ok=True)

text = src_path.read_text(encoding='utf-8')

# 1. Find the start of the body: line after \maketitle
m_start = re.search(r'\\maketitle\s*\n', text)
if not m_start:
    raise SystemExit('FATAL: \\maketitle not found in source')
body_start = m_start.end()

# 2. Find the end of the body: line before \end{document}
m_end = re.search(r'\\end\{document\}', text)
if not m_end:
    raise SystemExit('FATAL: \\end{document} not found in source')
body_end = m_end.start()

body = text[body_start:body_end].strip()

# 3. The bibliography is recreated per-venue. We split the body BEFORE the
# bibliography block so each wrapper can re-attach it with its own .bst.
m_bib = re.search(r'\\bibliographystyle\{[^}]*\}', body)
if m_bib:
    body_before_bib = body[:m_bib.start()].rstrip()
else:
    body_before_bib = body

# 4. Capture the thebibliography environment as a separate file
m_thebib = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                     body, re.DOTALL)
if not m_thebib:
    raise SystemExit('FATAL: \\begin{thebibliography} not found')

# Capture appendices (everything after \appendix)
m_appendix = re.search(r'\\appendix', body)
if not m_appendix:
    raise SystemExit('FATAL: \\appendix not found')
appendix_block = body[m_appendix.start():].strip()

# Body excluding bibliography AND appendix
main_body = body[:m_thebib.start()].rstrip()
# But we want main_body to STOP before \clearpage \bibliographystyle, not before
# \appendix. Find the \clearpage that precedes the bibliography.
m_clearpage = re.search(r'\\clearpage\s*\n\s*\\bibliographystyle', body)
if m_clearpage:
    main_body = body[:m_clearpage.start()].rstrip()
else:
    # fall back: cut at first \bibliographystyle
    main_body = body[:m_bib.start() if m_bib else m_thebib.start()].rstrip()

# Bibliography block (just the \begin{thebibliography}...\end{thebibliography})
bib_block = m_thebib.group(0)

# Write outputs
(dst_path.parent / 'body.tex').write_text(main_body + '\n', encoding='utf-8')
(dst_path.parent / 'bibliography.tex').write_text(bib_block + '\n', encoding='utf-8')
(dst_path.parent / 'appendix.tex').write_text(appendix_block + '\n', encoding='utf-8')

print(f'Wrote body.tex ({len(main_body)} chars)')
print(f'Wrote bibliography.tex ({len(bib_block)} chars)')
print(f'Wrote appendix.tex ({len(appendix_block)} chars)')
