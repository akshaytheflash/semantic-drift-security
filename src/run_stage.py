"""Stage orchestrator: launches every arm/depth cell as an isolated process.

Each cell runs its own subset_runner (preflight -> probes) at a fixed, token-verified depth.
Design points:
  * one process per cell, so a crashed or stuck cell cannot take down the batch;
  * a concurrency cap (the provider queue is throughput-limited, see docs/harness_adr.md);
  * every cell writes to its own log and its own `subset_*.jsonl` record;
  * cells that have already produced a record are skipped, so the stage is resumable.

python -m src.run_stage --stage deep --max-parallel 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "experiments" / "raw"
LOGS = ROOT / "experiments" / "logs"

STAGES = {
    # name: (arms, depths, seeds, probes, max_steps)
    "deep": (["procedural", "neutral", "reinforced"], [240000], [1000, 1001, 1002],
             "P1,P3,P10,P11,P8", 3),
    "deep-reinforced": (["reinforced"], [150000, 200000], [1000, 1001],
                        "P1,P3,P10,P11,P8", 3),
    "deep-neutral": (["neutral"], [150000, 200000], [1000, 1001],
                     "P1,P3,P10,P11,P8", 3),
    "ladder-150": (["procedural", "neutral", "reinforced"], [150000], [1000],
                   "P1,P3,P10,P11,P8", 3),
}


def existing(tag: str) -> bool:
    return (RAW / f"subset_glm-5.3-flash_{tag}.jsonl").exists() and \
        any(json.loads(l).get("kind") == "probes"
            for l in (RAW / f"subset_glm-5.3-flash_{tag}.jsonl").read_text().splitlines()
            if l.strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="deep", choices=sorted(STAGES))
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma list of arms to run")
    ap.add_argument("--probes", default=None, help="override probe subset")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    arms, depths, seeds, probes, steps = STAGES[args.stage]
    if args.only:
        arms = [a for a in arms if a in {x.strip() for x in args.only.split(",")}]
    probes = args.probes or probes
    steps = args.max_steps if args.max_steps is not None else steps
    LOGS.mkdir(parents=True, exist_ok=True)

    cells = []
    for depth in depths:
        for arm in arms:
            for seed in seeds:
                tag = f"s{args.stage.replace('-', '')}_{arm}_{depth}_{seed}"
                if existing(tag):
                    continue
                cells.append((depth, arm, seed, tag))
    print(f"stage={args.stage}: {len(cells)} cells to run (parallel={args.max_parallel})")
    if args.dry_run:
        for c in cells:
            print("  ", c)
        return

    running: list[tuple[subprocess.Popen, str]] = []
    queue = list(cells)
    while queue or running:
        while queue and len(running) < args.max_parallel:
            depth, arm, seed, tag = queue.pop(0)
            cmd = [sys.executable, "-m", "src.subset_runner", "--depth", str(depth),
                   "--condition", arm, "--seed", str(seed), "--tag", tag,
                   "--probes", probes, "--max-steps", str(steps), "--concurrency", "4"]
            if args.model:
                cmd += ["--model", args.model]
            log = (LOGS / f"{tag}.log").open("w")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, tag))
            print(f"  started {tag} (pid {p.pid})", flush=True)
        time.sleep(10)
        still = []
        for p, tag in running:
            if p.poll() is None:
                still.append((p, tag))
            else:
                print(f"  finished {tag} rc={p.returncode}", flush=True)
        running = still
    print("stage complete")


if __name__ == "__main__":
    main()
