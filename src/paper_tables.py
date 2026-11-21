"""Build paper tables directly from the processed CSVs.

python -m src.paper_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "experiments" / "processed"
PAPER = ROOT / "paper"
LABEL = {"A": "A (baseline)", "B": "B (neutral)", "C": "C (procedural)", "X": "X (proximate)",
         "D": "D (interference)", "E5": "E5 (reinforced/5)", "E10": "E10 (reinforced/10)",
         "E20": "E20 (reinforced/20)", "F": "F (injection)"}


def fmt_p(p: float) -> str:
    if p != p:
        return "n/a"
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.3f}"


def main():
    PAPER.mkdir(exist_ok=True)
    table = pd.read_csv(PROC / "group_checkpoint.csv")
    cps = sorted(table.checkpoint.unique())
    lines = [r"\begin{tabular}{l" + "c" * len(cps) + r"}", r"\toprule",
             "Arm & " + " & ".join(str(c) for c in cps) + r" \\", r"\midrule"]
    for g in table["group"].drop_duplicates():
        sub = table[table.group == g].set_index("checkpoint")
        cells = [f"{sub.loc[c,'spvr']*100:.0f}" if c in sub.index else "--" for c in cps]
        lines.append(f"{LABEL.get(g, g)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_spvr.tex").write_text("\n".join(lines) + "\n")

    gee = PROC / "gee.json"
    if gee.exists():
        d = json.loads(gee.read_text())
        rows = [r"\begin{tabular}{lr r}", r"\toprule",
                r"Term & $\hat\beta$ & $p$ \\", r"\midrule"]
        for k, v in d["contrasts"].items():
            rows.append(f"{k.replace('_', r'\_')} & {v['coef']:+.2f} & {fmt_p(v['p'])} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        (PAPER / "tab_gee.tex").write_text("\n".join(rows) + "\n")
    print("paper tables written to", PAPER)


if __name__ == "__main__":
    main()
