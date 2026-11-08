"""Static checks for the LaTeX skeleton (no TeX engine needed). Run from the repository root."""

import re
import sys
from pathlib import Path

P = Path("manuscript_c4ai4c27")
tex = (P / "main.tex").read_text(encoding="utf-8")
body = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else "" for line in tex.splitlines())
issues = []
for f in re.findall(r"\\(?:input|includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}", body):
    if not (P / f).exists() and not (P / (f + ".tex")).exists():
        issues.append(f"missing file: {f}")
bib = (P / "refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
for k in {c.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for c in group.split(",")}:
    if k not in keys:
        issues.append(f"undefined citation: {k}")
for name in ("figure*", "table*", "align", "equation", "abstract", "keywords", "document"):
    b, e = body.count(f"\\begin{{{name}}}"), body.count(f"\\end{{{name}}}")
    if b != e:
        issues.append(f"unbalanced environment {name}: {b} begin / {e} end")
for name in ("table1_seed_level.tex",):
    t = (P / "tables" / name).read_text(encoding="utf-8")
    if t.count("\\begin{") != t.count("\\end{"):
        issues.append(f"unbalanced environments in {name}")
depth = 0
for ch in body.replace("\\{", "").replace("\\}", ""):
    depth += (ch == "{") - (ch == "}")
    if depth < 0:
        break
if depth != 0:
    issues.append(f"brace imbalance in main.tex: {depth}")
for s in ("spconf.sty", "IEEEbib.bst"):
    if not (P / s).exists():
        issues.append(f"missing {s}")
notes = body.count("\\authornote{")
print("static check:", "PASS" if not issues else "FAIL")
for i in issues:
    print(" -", i)
print(f"author notes remaining: {notes} (must be 0 or hidden before submission); citation keys used: "
      f"{len(set(re.findall(r'cite{([^}]+)}', body)))}; compiled: NO (no TeX engine in this environment)")
sys.exit(1 if issues else 0)
