"""Re-score action-level outcomes from stored tool calls, without new API calls.

Text-level outcomes are NOT re-scorable: no probe text was stored with the run. Any text
quote in the results doc must therefore be taken from a follow-up run that stores answers.

python -m src.rescore --input experiments/raw/subset_*.jsonl --out experiments/processed/rescored.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def rescore(rec: dict) -> list[dict]:
    out = []
    for pid, o in rec.get("probes", {}).items():
        if o.get("kind") != "adversarial":
            continue
        calls = o.get("calls") or []
        attempted = any(c.get("executed") or c.get("error") in (None, "protected", "not_found")
                        and not c.get("executed") is False for c in calls) if False else None
        # action-level only: was any recorded call executed (realised) -- we cannot re-run
        # predicates here without the probe objects, so record the raw calls for audit and
        # use the stored 'executed' flag plus the stored call list.
        out.append({"condition": rec.get("condition"), "seed": rec.get("seed"),
                    "depth": rec.get("depth") or rec.get("target_tokens"),
                    "probe": pid, "clause": o.get("clause"),
                    "stored_violated": o.get("violated"), "stored_executed": o.get("executed"),
                    "n_calls": len(calls),
                    "calls": json.dumps([{"t": c.get("tool"), "a": c.get("args"),
                                          "x": c.get("executed")} for c in calls])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "processed" / "rescored.csv"))
    args = ap.parse_args(argv)
    paths = args.input or sorted(glob.glob(str(ROOT / "experiments/raw/subset_*.jsonl")))
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "probes":
                rows += rescore(r)
    df = pd.DataFrame(rows)
    Path(args.out).write_text("")
    df.to_csv(args.out, index=False)
    print(f"{len(df)} probe outcomes -> {args.out}")
    print(df.drop(columns=["calls"]).to_string(index=False))


if __name__ == "__main__":
    main()
