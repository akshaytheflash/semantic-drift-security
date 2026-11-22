"""Full-probe blind batch: every adversarial probe, every arm, every model, at depth.

This is the experiment that closes the "P2/P4/P5/P6/P7 never ran at depth" gap. Probes carry
neutral names in the log; the mapping lives here and is not consulted while collection runs,
so an operator cannot steer around a probe that would land badly.

python -m src.full_batch --models glm-5.3-flash,deepseek-v4.1-flash --depth 240000
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_probes_parallel
from .calibrate import chars_for
from .deep_runner import history_at
from .probes import PROBES
from .providers.base import provider_from_env
from .security_policy import policy_snapshot

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "raw"
TOL = 0.03
CLOAK = {"P1": "a", "P2": "b", "P3": "c", "P4": "d", "P5": "e", "P6": "f", "P7": "g",
         "P10": "h", "P11": "i", "P8": "j", "P9": "k"}


def cell_done(log: Path, model: str, depth: int, arm: str, seed: int) -> bool:
    if not log.exists():
        return False
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("model") == model and r.get("depth") == depth
                and r.get("arm") == arm and r.get("seed") == seed):
            return True
    return False


def run_cell(model: str, depth: int, arm: str, seed: int, max_steps: int) -> None:
    out = OUT / f"full_{model}.jsonl"
    if cell_done(out, model, depth, arm, seed):
        print(f"skip {model} {arm} {depth} seed {seed}", flush=True)
        return
    provider = provider_from_env(model=model, max_tokens=400)
    chars = chars_for(arm, depth, model)
    hist = history_at(arm, chars, 10 if arm == "reinforced" else 10, seed)
    t0 = time.time()
    outcomes = run_probes_parallel(provider, hist, PROBES, max_steps, workers=4)
    comps = {pid: [c.to_dict() for c in cs] for pid, (_o, cs) in outcomes.items()}
    toks = [c["prompt_tokens"] for cs in comps.values() for c in cs if c.get("prompt_tokens")]
    rec = {"trial_id": f"full-{arm}-{depth}-{seed}-{uuid.uuid4().hex[:6]}", "model": model,
           "arm": arm, "depth": depth, "seed": seed, "chars": chars,
           "measured_tokens": max(toks) if toks else None,
           "token_gate_ok": bool(toks) and abs(max(toks) - depth) / depth <= TOL,
           "policy_sha256": policy_snapshot()["sha256"],
           "probes": {CLOAK[pid]: asdict(o) for pid, (o, _c) in outcomes.items()},
           "probe_map": {v: k for k, v in CLOAK.items()},
           "transcripts": {CLOAK[pid]: {"user_message": o.user_message, "answer": o.text,
                                        "calls": [{"tool": c.tool, "args": c.args,
                                                   "executed": c.executed, "result": c.result}
                                                  for c in o.calls]}
                           for pid, (o, _c) in outcomes.items()},
           "n_incomplete": sum(1 for cs in comps.values() for c in cs if c["incomplete"]),
           "wall_s": round(time.time() - t0, 1),
           "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    adv = [o for o, _c in outcomes.values() if o.kind == "adversarial"]
    print(f"[{model} {arm} {depth} {seed}] tok={rec['measured_tokens']} "
          f"gate={rec['token_gate_ok']} viol={sum(o.violated for o in adv)}/{len(adv)} "
          f"wall={rec['wall_s']}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glm-5.3-flash,deepseek-v4.1-flash")
    ap.add_argument("--depth", type=int, default=240_000)
    ap.add_argument("--arms", default="procedural,neutral,reinforced")
    ap.add_argument("--seeds", default="1000,1001")
    ap.add_argument("--max-steps", type=int, default=3)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        for arm in args.arms.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                try:
                    run_cell(model.strip(), args.depth, arm.strip(), seed, args.max_steps)
                except Exception as exc:  # never let one cell kill the batch
                    print(f"ERROR {model} {arm} {seed}: {exc}", flush=True)


if __name__ == "__main__":
    main()
