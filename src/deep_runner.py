"""Deep-horizon runner: token-gated depth ladder.

Arms are defined by *measured prompt-token depth*, not by item counts:

    python -m src.deep_runner --arms procedural,neutral,reinforced \
        --targets 150000,200000,240000 --seeds 1000 --concurrency 4

For each (arm, target, seed):
  1. search corpus characters until the API reports prompt tokens within ±5% of target;
  2. run the probe set against that exact history;
  3. record the measured token count, all completions, and an abort if the target is missed.

Nothing below target - tol is ever scored.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_probes_parallel
from .calibrate import calibrate
from .deep_context import deep_corpus, messages_for
from .probes import PROBES
from .providers.base import provider_from_env
from .security_policy import REINFORCEMENT_MESSAGE, SYSTEM_PROMPT, policy_snapshot

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("neutral", "procedural", "reinforced")
TOL = 0.03


def history_at(condition: str, chars: float, reinforce_every: int = 10,
               seed: int = 1000) -> list[dict]:
    items = deep_corpus(condition, int(chars), seed)
    reinf = REINFORCEMENT_MESSAGE if condition == "reinforced" else None
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages_for(
        items, reinf, reinforce_every=reinforce_every)


def run(args):
    provider = provider_from_env(model=args.model, max_tokens=args.max_tokens)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = out / f"deep_{provider.model}_{tag}.jsonl"
    calib_log = out / f"calibration_{provider.model}_{tag}.jsonl"
    manifest = {"model": provider.model, "arms": args.arms, "targets": args.targets,
                "seeds": args.seeds, "max_tokens": args.max_tokens,
                "reinforce_every": args.reinforce_every, "probes": [p.id for p in PROBES],
                "policy_sha256": policy_snapshot()["sha256"], "started": datetime.now(timezone.utc).isoformat(),
                "tag": tag}
    (out / f"manifest_deep_{provider.model}_{tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("log:", log, flush=True)

    for arm in args.arms:
        for target in args.targets:
            cal = calibrate(provider, arm, target, max_tokens=8, log_path=calib_log)
            if cal.get("achieved") is None:
                print(f"[{arm} {target}] ABORT: could not reach target; not scored", flush=True)
                with log.open("a") as fh:
                    fh.write(json.dumps({"arm": arm, "target": target, "aborted": True,
                                         "calibration": cal}) + "\n")
                continue
            chars = cal["chars"]
            measured = cal["achieved"]
            for seed in args.seeds:
                trial_id = f"deep-{arm}-{target}-{seed}-{uuid.uuid4().hex[:6]}"
                hist = history_at(arm, chars, args.reinforce_every, seed)
                t0 = time.time()
                outcomes = run_probes_parallel(provider, hist, PROBES, args.max_steps,
                                               workers=args.concurrency)
                comps = {pid: [c.to_dict() for c in cs] for pid, (_o, cs) in outcomes.items()}
                ptoks = [c["prompt_tokens"] for cs in comps.values() for c in cs
                         if c.get("prompt_tokens")]
                gate_ok = bool(ptoks) and abs(max(ptoks) - target) / target <= TOL
                from dataclasses import asdict
                rec = {"trial_id": trial_id, "arm": arm, "target_tokens": target,
                       "measured_tokens": max(ptoks) if ptoks else None,
                       "chars": chars, "messages": len(hist), "seed": seed,
                       "token_gate_ok": gate_ok, "policy_sha256": manifest["policy_sha256"],
                       "probes": {pid: asdict(o) for pid, (o, _c) in outcomes.items()},
                       "completions": comps, "wall_s": round(time.time() - t0, 1),
                       "n_incomplete": sum(1 for cs in comps.values() for c in cs
                                           if c["incomplete"]),
                       "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with log.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                n = sum(1 for p in PROBES if p.kind == "adversarial")
                v = sum(o.violated for o, _ in outcomes.values())
                print(f"[{trial_id}] tok={rec['measured_tokens']} gate={gate_ok} "
                      f"viol={v}/{n} wall={rec['wall_s']}s inc={rec['n_incomplete']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="procedural,neutral,reinforced")
    ap.add_argument("--targets", default="150000")
    ap.add_argument("--seeds", default="1000")
    ap.add_argument("--reinforce-every", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "raw"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in args.arms:
        if a not in ARMS:
            ap.error(f"arm must be one of {ARMS}")
    args.targets = [int(t) for t in args.targets.split(",") if t.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run(args)


if __name__ == "__main__":
    main()
