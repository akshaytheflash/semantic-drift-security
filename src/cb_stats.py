"""Score the compaction battery: rule retention in summaries vs probe outcomes.

python -m src.cb_stats
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from .compaction_battery import rule_profile
from .probes import PROBES, score_probe
from .simulated_tools import SimulatedEnvironment, ToolCallRecord

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "experiments" / "processed"
COND = {"C1": "neutral content", "C2": "two-stage", "C3": "summary-prompt sweep",
        "C4": "task pressure", "C5": "multi-task"}


def load():
    rows, summaries = [], []
    for path in sorted(glob.glob(str(ROOT / "experiments/raw/compactbattery_*.jsonl"))):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p1 = {p.id: p for p in PROBES}
            for pid, t in r["transcripts"].items():
                probe = p1.get(pid)
                if probe is None:
                    continue
                calls = [ToolCallRecord(seq=i, tool=c["tool"], args=c["args"],
                                        executed=bool(c.get("executed")),
                                        result=str(c.get("result", "")), error=None)
                         for i, c in enumerate(t.get("calls") or [])]
                out = score_probe(probe, calls, t.get("answer", ""), SimulatedEnvironment())
                empty = len((t.get("answer") or "").strip()) <= 2
                rows.append({"answer_empty": empty,"condition": r["condition"], "seed": r["seed"],
                             "cell": r["cell"], "depth": r["session_depth"],
                             "compact_tokens": r["prompt_tokens"], "probe": pid,
                             "clause": out.clause, "kind": out.kind,
                             "violated": out.violated, "text_violation": out.text_violation,
                             "false_refusal": out.false_refusal})
            for st in r.get("stages", []):
                prof = rule_profile(st["note"])   # re-scan: stored counts used an older detector
                summaries.append({"condition": r["condition"], "seed": r["seed"],
                                  "cell": r["cell"], "stage": st["stage"],
                                  "kept": sum(v for k, v in prof.items()
                                              if not k.startswith("_")),
                                  "name_only": prof.get("_name_only_mentions", 0),
                                  "rules": prof, "chars": len(st["note"])})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def main():
    df, sm = load()
    if df.empty:
        print("no records yet")
        return
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "cb_probe_level.csv", index=False)
    sm.to_csv(PROC / "cb_summaries.csv", index=False)
    adv = df[(df.kind == "adversarial") & (~df.answer_empty)]
    print("dropped empty/incomplete answers:", int(((df.kind == "adversarial") & df.answer_empty).sum()))
    t = (adv.groupby("condition", as_index=False)
         .agg(cells=("cell", "nunique"), probe_runs=("violated", "size"),
              violations=("violated", "sum"), text_flags=("text_violation", "sum"),
              compact_tokens=("compact_tokens", "median")))
    t["spvr"] = (t.violations / t.probe_runs).round(3)
    t["condition"] = t.condition.map(lambda c: COND.get(c, c))
    print("== compaction battery: action-level violations ==")
    print(t.to_string(index=False))
    print("\n== rule retention in the final summary (of 4) ==")
    print(sm.groupby("condition")["kept"].agg(["mean", "min", "max", "count"]).round(2).to_string())
    print("\n== per probe (rate on valid answers, with n) ==")
    piv = adv.pivot_table(index="probe", columns="condition", values="violated",
                          aggfunc=["mean", "size"])
    print(piv.round(2).to_string())
    print("\n== does a summary carrying DOCUMENT NAMES but not the RULES predict P11? ==")
    p11 = df[(df.kind == "adversarial") & (df.probe == "P11") & (~df.answer_empty)]
    merged = p11.merge(sm.groupby(["condition", "seed", "cell"], as_index=False)
                       .agg(kept=("kept", "last"), names=("name_only", "last")),
                       on=["condition", "seed", "cell"], how="left")
    if not merged.empty:
        merged["names_only_no_rules"] = (merged.names > 0) & (merged.kept == 0)
        print(merged.groupby("names_only_no_rules")["violated"]
              .agg(["mean", "size"]).round(2).to_string())
    ben = df[df.kind == "benign"]
    print("\nfalse-refusal rate by condition:")
    print(ben.groupby("condition")["false_refusal"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
