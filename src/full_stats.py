"""Score the full-probe blind batches (all arms x all probes x models).

Probe names are decoded only here, after collection.

python -m src.full_stats
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from .probes import PROBES, score_probe
from .simulated_tools import SimulatedEnvironment, ToolCallRecord

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "experiments" / "processed"
FIG = ROOT / "figures"
ARM_COLOR = {"neutral": "#1f77b4", "procedural": "#d62728", "reinforced": "#2ca02c"}


def score_records(paths: list[str]) -> pd.DataFrame:
    by_id = {p.id: p for p in PROBES}
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "probe_map" not in rec:
                continue
            for cloak, t in (rec.get("transcripts") or {}).items():
                pid = rec["probe_map"][cloak]
                probe = by_id.get(pid)
                if probe is None:
                    continue
                calls = [ToolCallRecord(seq=i, tool=c["tool"], args=c["args"],
                                        executed=bool(c.get("executed")),
                                        result=str(c.get("result", "")), error=None)
                         for i, c in enumerate(t.get("calls") or [])]
                out = score_probe(probe, calls, t.get("answer", ""), SimulatedEnvironment())
                rows.append({
                    "model": rec["model"], "arm": rec["arm"], "depth": rec["depth"],
                    "seed": rec["seed"], "measured_tokens": rec.get("measured_tokens"),
                    "gate_ok": rec.get("token_gate_ok"), "probe": pid,
                    "clause": out.clause, "kind": out.kind,
                    "violated": out.violated, "executed": out.executed,
                    "text_violation": out.text_violation,
                    "false_refusal": out.false_refusal})
    return pd.DataFrame(rows)


def tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adv = df[df.kind == "adversarial"]
    per = (adv.groupby(["model", "probe", "clause"], as_index=False)
           .agg(trials=("violated", "size"), violations=("violated", "sum"),
                text_violations=("text_violation", "sum")))
    per["action_rate"] = (per.violations / per.trials).round(3)
    per["text_rate"] = (per.text_violations / per.trials).round(3)
    by_arm = (adv.groupby(["model", "arm", "depth"], as_index=False)
              .agg(trials=("violated", "nunique"), probes=("violated", "size"),
                   violations=("violated", "sum"), text_violations=("text_violation", "sum")))
    by_arm["spvr"] = (by_arm.violations / by_arm.probes).round(3)
    by_arm["text_spvr"] = (by_arm.text_violations / by_arm.probes).round(3)
    return per, by_arm


def figure(per: pd.DataFrame, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if per.empty:
        return False
    piv = per.pivot_table(index="probe", columns="model", values="action_rate")
    piv = piv.reindex([p for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P10", "P11"]
                       if p in piv.index])
    if piv.empty:
        return False
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = range(len(piv))
    w = 0.8 / max(1, len(piv.columns))
    for k, col in enumerate(piv.columns):
        ax.bar([i + k * w for i in x], piv[col].fillna(0), width=w, label=col)
    ax.set_xticks([i + w * (len(piv.columns) - 1) / 2 for i in x])
    ax.set_xticklabels(piv.index, fontsize=7)
    ax.set_ylabel("violation rate (action view)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    paths = args.input or sorted(glob.glob(str(ROOT / "experiments/raw/full_*.jsonl")))
    if not paths:
        print("no full-batch records yet")
        return
    df = score_records(paths)
    if df.empty:
        print("no scored records yet")
        return
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "full_probe_level.csv", index=False)
    per, by_arm = tables(df)
    per.to_csv(PROC / "full_per_probe.csv", index=False)
    by_arm.to_csv(PROC / "full_by_arm.csv", index=False)
    print("== per probe (action vs text) ==")
    print(per.to_string(index=False))
    print("\n== by arm ==")
    print(by_arm.to_string(index=False))
    print("\nfailing token gate:", int((~df.gate_ok.fillna(False)).sum()))
    ben = df[df.kind == "benign"]
    if not ben.empty:
        print("\nfalse-refusal rate by model/arm:")
        print(ben.groupby(["model", "arm"])["false_refusal"].mean().round(2).to_string())
    if args.figures:
        made = figure(per, FIG / "fig8_full_per_probe.png")
        print("figure:", "written" if made else "skipped")


if __name__ == "__main__":
    main()
