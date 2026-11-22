"""Walk DeepSeek's context window upward until it actually refuses.

Each step: a tiny calibration call at the target depth (records the provider's token count and
whether the request succeeds), then - only if that succeeds - a probe cell. The ladder stops
at the first depth where the provider rejects or truncates, and that ceiling is reported as a
measured platform limit rather than assumed.

python -m src.million_ladder --targets 500000,650000,800000,950000 --seeds 1000
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
PROBE_TOKENS_MAX = 1_000_000


def probe_target(provider, hist: list[dict], max_tokens: int = 12) -> dict:
    """One cheap call: does the provider accept this depth, and what does it count?"""
    provider.max_tokens = max_tokens
    comp = provider.complete(hist + [{"role": "user", "content": "Reply with one word: OK"}])
    return {"prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens,
            "finish_reason": comp.finish_reason, "incomplete": comp.incomplete,
            "error": comp.error, "latency_s": round(comp.latency_s, 1),
            "request_id": comp.request_id,
            "accepted": comp.prompt_tokens is not None and comp.finish_reason is not None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--arms", default="procedural")
    ap.add_argument("--targets", default="500000,650000,800000,950000")
    ap.add_argument("--seeds", default="1000")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-output", type=int, default=400)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"ladder_{args.model}.jsonl"
    cal_log = OUT / f"ladder_cal_{args.model}.jsonl"

    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r.get("arm"), r.get("depth"), r.get("seed")))
    ceiling = None
    for arm in args.arms.split(","):
        for target in [int(t) for t in args.targets.split(",")]:
            if ceiling is not None and target > ceiling:
                print(f"skip {arm} {target}: above measured ceiling {ceiling}", flush=True)
                continue
            for seed in [int(s) for s in args.seeds.split(",")]:
                if (arm, target, seed) in done:
                    print(f"skip {arm} {target} {seed} (recorded)", flush=True)
                    continue
                provider = provider_from_env(model=args.model, max_tokens=args.max_output,
                                             max_wait_s=7200)
                chars = chars_for(arm, target, args.model)
                hist = history_at(arm, chars, 10, seed)
                pf = probe_target(provider, hist)
                with cal_log.open("a") as fh:
                    fh.write(json.dumps({"arm": arm, "depth": target, "seed": seed,
                                         "chars": chars, **pf}) + "\n")
                print(f"[cal {arm} {target}] tok={pf['prompt_tokens']} "
                      f"finish={pf['finish_reason']} accepted={pf['accepted']} "
                      f"err={pf['error']}", flush=True)
                if not pf["accepted"]:
                    ceiling = target
                    print(f"CEILING reached at {target}: provider did not accept the request",
                          flush=True)
                    break
                if abs((pf["prompt_tokens"] or 0) - target) / target > TOL:
                    # resize once using the measured ratio and retry the calibration
                    mult = target / max(1, pf["prompt_tokens"])
                    chars = int(chars * mult)
                    hist = history_at(arm, chars, 10, seed)
                    pf2 = probe_target(provider, hist)
                    with cal_log.open("a") as fh:
                        fh.write(json.dumps({"arm": arm, "depth": target, "seed": seed,
                                             "chars": chars, "resized": True, **pf2}) + "\n")
                    print(f"[cal {arm} {target}] resized -> tok={pf2['prompt_tokens']} "
                          f"accepted={pf2['accepted']}", flush=True)
                    if not pf2["accepted"]:
                        ceiling = target
                        break
                    pf = pf2

                t0 = time.time()
                outcomes = run_probes_parallel(provider, hist, PROBES, args.max_steps,
                                               workers=4)
                comps = {pid: [c.to_dict() for c in cs] for pid, (_o, cs) in outcomes.items()}
                toks = [c["prompt_tokens"] for cs in comps.values() for c in cs
                        if c.get("prompt_tokens")]
                rec = {"trial_id": f"ladder-{arm}-{target}-{seed}-{uuid.uuid4().hex[:6]}",
                       "model": args.model, "arm": arm, "depth": target, "seed": seed,
                       "chars": chars, "measured_tokens": max(toks) if toks else None,
                       "token_gate_ok": bool(toks) and abs(max(toks) - target) / target <= TOL,
                       "policy_sha256": policy_snapshot()["sha256"],
                       "probes": {CLOAK[pid]: asdict(o) for pid, (o, _c) in outcomes.items()},
                       "probe_map": {v: k for k, v in CLOAK.items()},
                       "transcripts": {CLOAK[pid]: {"user_message": o.user_message,
                                                    "answer": o.text,
                                                    "calls": [{"tool": c.tool, "args": c.args,
                                                               "executed": c.executed,
                                                               "result": c.result}
                                                              for c in o.calls]}
                                       for pid, (o, _c) in outcomes.items()},
                       "n_incomplete": sum(1 for cs in comps.values() for c in cs
                                           if c["incomplete"]),
                       "wall_s": round(time.time() - t0, 1),
                       "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with log.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                adv = [o for o, _c in outcomes.values() if o.kind == "adversarial"]
                print(f"[{args.model} {arm} {target} {seed}] tok={rec['measured_tokens']} "
                      f"gate={rec['token_gate_ok']} "
                      f"viol={sum(o.violated for o in adv)}/{len(adv)} "
                      f"wall={rec['wall_s']}s", flush=True)
                provider.max_tokens = args.max_output


if __name__ == "__main__":
    main()
