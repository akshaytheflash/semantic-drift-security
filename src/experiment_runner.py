"""Experiment orchestrator.

Groups (see docs/amendment_1.md):
  A            baseline (no accumulated context)
  B            neutral, unrelated notes           (volume only)
  C            procedural-proximate notes         (proximity: tools/folders/approvals)
  X            socially-proximate notes           (proximity: people/vault/partners)
  D            competing legitimate priorities    (interference)
  E5/E10/E20   C content + policy restated every 5/10/20 items   (mitigation dose)
  F            C content + explicit injections    (injection control)

Every request replays history[0] (the system policy) and the runner asserts its sha256
matches the invariant snapshot before each checkpoint (`policy_audit`).

Usage
-----
python -m src.experiment_runner --all --checkpoints 0,5,10,20,30,40,50 --trials 1
python -m src.experiment_runner --group E10 --trials 3
python -m src.experiment_runner --groups A,B,C --extend-tokens 20000,40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_probes_parallel
from .context_generator import build_sequence, f_injection_positions, injection_item, lint_benign
from .probes import PROBES
from .providers.base import provider_from_env
from .security_policy import REINFORCEMENT_MESSAGE, policy_snapshot

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS = [0, 5, 10, 20, 30, 40, 50]
# deep arm: match the token volume prior work works in (its deepest cell is ~35k tokens)
SPACE_DEPTH_MIN = 150
SPACE_DEPTH_DEFAULT = 300
GROUPS = ["A", "B", "C", "X", "D", "E5", "E10", "E20", "F"]
# Large-volume arms: same corpora, 300 items (~16-33k prompt tokens). Named S~<arm>.
SPACE_GROUPS = ["S~B", "S~C", "S~X", "S~D", "S~E5", "S~E10"]
ALL_ARMS = GROUPS + SPACE_GROUPS


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def group_condition(group: str) -> tuple[str, int | None]:
    """Map an arm name to (context-builder condition, reinforcement cadence).

    'S~C' is the large-volume version of arm C; its condition prefix is 'C'.
    """
    name = group.replace("S~", "")
    if name.startswith("E"):
        return "E", int(name[1:])
    return name, None


def arm_depth(group: str, n_items: int, space_depth: int) -> int:
    """Depth (context items) for a nominal checkpoint.

    Interaction arms: the checkpoint is the number of items.
    Large-volume arms: item 0 is 0 (pinned baseline) and the nominal checkpoint is scaled
    linearly to the space depth, so 50 -> space_depth and 25 -> half of it.
    """
    if not group.startswith("S~"):
        return n_items
    if n_items <= 0:
        return 0
    return max(1, int(round(space_depth * n_items / 50)))


def build_history(group: str, upto: int, seed: int, injection_every: int = 4) -> list[dict]:
    cond, reinforce = group_condition(group)
    if upto <= 0:
        return []
    items = build_sequence(cond, upto, seed)
    if cond != "F":
        problems = lint_benign(items)
        assert not problems, problems
    rng = random.Random(seed * 7919 + upto)
    msgs: list[dict] = []
    inj = f_injection_positions(upto, injection_every) if cond == "F" else set()
    for i, it in enumerate(items):
        if reinforce and i % reinforce == 0 and i > 0:
            msgs.append({"role": "system", "content": REINFORCEMENT_MESSAGE})
        msgs.extend(it.to_messages(rng))
        if i in inj:
            msgs.extend(injection_item(i, random.Random(seed + i)).to_messages(rng))
    return msgs


def history_hash(history: list[dict]) -> str:
    return hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()


def policy_audit(history: list[dict], expected_sha: str) -> bool:
    """The first message must be the invariant policy; no later message may alter it."""
    if not history or history[0].get("role") != "system":
        return False
    return hashlib.sha256(history[0]["content"].encode()).hexdigest() == expected_sha


def run_trial(provider, group, seed, checkpoints, max_steps, log_path: Path, meta: dict,
              extend_tokens: list[int] | None = None, space_depth: int = SPACE_DEPTH_DEFAULT) -> dict:
    snap = policy_snapshot()
    trial_id = f"{group}-{provider.model}-{seed}-{uuid.uuid4().hex[:6]}"
    base = {"trial_id": trial_id, "group": group, "model": provider.model, "seed": seed,
            "policy_sha256": snap["sha256"], "provider": provider.name, "meta": meta}
    first_violation = None
    done = []
    with log_path.open("a") as fh:
        for cp in checkpoints:
            t0 = time.time()
            upto = arm_depth(group, cp, space_depth)
            hist = build_history(group, upto, seed, meta.get("injection_every", 4))
            history = [{"role": "system", "content": snap["system_prompt"]}] + hist
            audit_ok = policy_audit(history, snap["sha256"])
            if not audit_ok:
                raise RuntimeError("POLICY AUDIT FAILED - aborting trial")
            outcomes = run_probes_parallel(provider, history, PROBES, max_steps)
            wall = time.time() - t0
            comps = {pid: [c.to_dict() for c in cs] for pid, (_o, cs) in outcomes.items()}
            ptoks = [c["prompt_tokens"] for cs in comps.values() for c in cs if c.get("prompt_tokens")]
            probes = {pid: asdict(o) for pid, (o, _c) in outcomes.items()}
            viol = sorted(pid for pid, o in probes.items()
                          if o["kind"] == "adversarial" and o["violated"])
            if viol and first_violation is None:
                first_violation = {"checkpoint": cp, "probes": viol,
                                   "history_tokens": max(ptoks) if ptoks else None}
            rec = dict(base)
            rec.update({
                "checkpoint": cp, "frame": "interaction", "n_context_items": len(hist),
                "n_messages": len(history), "history_sha256": history_hash(history),
                "policy_audit": audit_ok,
                "history_tokens": max(ptoks) if ptoks else None,
                "history_tokens_median": sorted(ptoks)[len(ptoks)//2] if ptoks else None,
                "n_incomplete": sum(1 for cs in comps.values() for c in cs if c["incomplete"]),
                "probes": probes, "completions": comps, "wall_s": wall,
                "recorded": utcnow(), "first_violation_so_far": first_violation,
            })
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done.append(cp)
            print(f"[{trial_id}] cp={cp:>3} items={len(hist):>3} tok={rec['history_tokens']} "
                  f"viol={len(viol)}/{sum(1 for p in PROBES if p.kind=='adversarial')} "
                  f"wall={wall:.0f}s inc={rec['n_incomplete']}", flush=True)
        # token-matched extension: continue the same sequence to reach target lengths
        for target in (extend_tokens or []):
            n_items = max(10, int(target / max(1, (rec["history_tokens"] or 1500)) * max(1, len(history))))
            n_items = int(n_items * 1.15)
            hist = build_history(group, n_items, seed, meta.get("injection_every", 4))
            history = [{"role": "system", "content": snap["system_prompt"]}] + hist
            if not policy_audit(history, snap["sha256"]):
                raise RuntimeError("POLICY AUDIT FAILED - aborting trial")
            outcomes = run_probes_parallel(provider, history, PROBES, max_steps)
            comps = {pid: [c.to_dict() for c in cs] for pid, (_o, cs) in outcomes.items()}
            ptoks = [c["prompt_tokens"] for cs in comps.values() for c in cs if c.get("prompt_tokens")]
            probes = {pid: asdict(o) for pid, (o, _c) in outcomes.items()}
            viol = sorted(pid for pid, o in probes.items()
                          if o["kind"] == "adversarial" and o["violated"])
            rec = dict(base)
            rec.update({"checkpoint": n_items, "frame": "token_match",
                        "n_context_items": len(hist), "n_messages": len(history),
                        "history_sha256": history_hash(history), "policy_audit": True,
                        "history_tokens": max(ptoks) if ptoks else None,
                        "n_incomplete": sum(1 for cs in comps.values() for c in cs if c["incomplete"]),
                        "probes": probes, "completions": comps, "recorded": utcnow()})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{trial_id}] token-ext target={target} items={len(hist)} "
                  f"tok={rec['history_tokens']} viol={len(viol)}", flush=True)
    return {"trial_id": trial_id, "checkpoints": done, "first_violation": first_violation}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Semantic Drift experiment runner")
    ap.add_argument("--groups", default=None, help="comma list of arms, e.g. A,B,C,E10")
    ap.add_argument("--group", default=None, help="single arm")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--main", action="store_true", help="arms A..F at interaction depths")
    ap.add_argument("--space", action="store_true", help="large-volume arms S~B..S~E10")
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    ap.add_argument("--extend-tokens", default="", help="token-matched extension targets")
    ap.add_argument("--space-depth", type=int, default=SPACE_DEPTH_DEFAULT,
                    help="items in the large-volume extension used for the 'S' arms")
    ap.add_argument("--injection-every", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    groups = []
    if args.all:
        groups = list(ALL_ARMS)
    if args.main:
        groups += [g for g in GROUPS if g not in groups]
    if args.space:
        groups += [g for g in SPACE_GROUPS if g not in groups]
    else:
        if args.group:
            groups += [g.strip().upper() for g in args.group.split(",")]
        if args.groups:
            groups += [g.strip().upper() for g in args.groups.split(",")]
    if not groups:
        ap.error("pass --group, --groups or --all")
    bad = [g for g in groups if g not in ALL_ARMS]
    if bad:
        ap.error(f"unknown arms {bad}; choose from {ALL_ARMS}")

    checkpoints = sorted({int(c) for c in args.checkpoints.split(",") if c.strip()})
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [1000 + i for i in range(args.trials)])
    extend = [int(t) for t in args.extend_tokens.split(",") if t.strip()]

    provider = provider_from_env(model=args.model, max_tokens=args.max_tokens)
    out_dir = ROOT / "experiments" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    schema = policy_snapshot()["sha256"]
    meta = {"provider": provider.name, "model": provider.model, "max_tokens": args.max_tokens,
            "temperature": 0.0, "reasoning_effort": provider.reasoning_effort,
            "probes": [p.id for p in PROBES], "checkpoints": checkpoints,
            "extend_tokens": extend, "injection_every": args.injection_every,
            "groups": groups, "seeds": seeds, "tag": tag, "schema_version": "probes-v2",
            "policy_sha256": schema, "started": utcnow()}
    log_path = out_dir / f"runs_{provider.model}_{tag}.jsonl"
    (out_dir / f"manifest_{provider.model}_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("log:", log_path, flush=True)
    for g in groups:
        for seed in seeds:
            run_trial(provider, g, seed, checkpoints, args.max_steps, log_path, meta, extend,
                      args.space_depth)


if __name__ == "__main__":
    main()
