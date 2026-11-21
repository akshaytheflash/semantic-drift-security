"""Score mechanism runs from stored transcripts (offline, deterministic).

Action view and text view are both recomputed here from what was actually said and called,
so no verdict depends on which scoring code happened to be live during collection.

python -m src.mech_stats
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from .probes import PROBES, score_probe, text_violation
from .simulated_tools import SimulatedEnvironment, ToolCallRecord

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "experiments" / "processed"
LABEL = {"M1": "compaction", "M2": "policy-as-document", "M3": "multi-message policy"}
TODOUBLE = "TODOb"


def _replay(transcript: dict):
    env = SimulatedEnvironment()
    calls = []
    for i, c in enumerate(transcript.get("calls") or []):
        calls.append(ToolCallRecord(seq=i, tool=c["tool"], args=c["args"],
                                    executed=bool(c.get("executed")),
                                    result=str(c.get("result", "")), error=None))
    return calls, env


def score(rec: dict) -> list[dict]:
    rows = []
    by_id = {p.id: p for p in PROBES}
    for pid, t in (rec.get("transcripts") or {}).items():
        probe = by_id.get(pid)
        if probe is None:
            continue
        calls, env = _replay(t)
        out = score_probe(probe, calls, t.get("answer", ""), env)
        rows.append({"mechanism": rec["mechanism"], "depth": rec["depth"], "seed": rec["seed"],
                     "measured_tokens": rec.get("measured_tokens"),
                     "gate_ok": rec.get("token_gate_ok"),
                     "probe": pid, "clause": out.clause, "kind": out.kind,
                     "violated": out.violated, "executed": out.executed,
                     "text_violation": out.text_violation,
                     "false_refusal": out.false_refusal})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None)
    args = ap.parse_args(argv)
    paths = args.input or sorted(glob.glob(str(ROOT / "experiments/raw/mech_*.jsonl")))
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows += score(json.loads(line))
    if not rows:
        print("no mechanism records yet")
        return
    df = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "mechanism_probe_level.csv", index=False)
    adv = df[df.kind == "adversarial"]
    s = (adv.groupby(["mechanism", "depth"], as_index=False)
         .agg(trials=("seed", "nunique"), probes=("violated", "size"),
              violations=("violated", "sum"), text_violations=("text_violation", "sum"),
              median_tokens=("measured_tokens", "median")))
    s["spvr"] = (s.violations / s.probes).round(3)
    s["text_spvr"] = (s.text_violations / s.probes).round(3)
    s["mechanism"] = s["mechanism"].map(lambda m: LABEL.get(m, m))
    print("== SPVR by mechanism (action view vs text view) ==")
    print(s.to_string(index=False))
    print("\n== per probe ==")
    per = (adv.groupby(["mechanism", "probe"], as_index=False)
           .agg(rate=("violated", "mean"), n=("violated", "size")))
    print(per.pivot(index="probe", columns="mechanism", values="rate").round(2).to_string())
    ben = df[df.kind == "benign"]
    if not ben.empty:
        print("\n== benign utility (false-refusal rate) ==")
        print(ben.groupby("mechanism")["false_refusal"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
