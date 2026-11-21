"""Token-gated depth calibration.

Measures the API's own prompt-token count for a candidate history, then searches for the
corpus size that hits a token target. Nothing is scored until the measured count is inside
tolerance of the target.

python -m src.calibrate --condition procedural --target 150000 --max-tokens 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .deep_context import deep_corpus, messages_for
from .tokens import count_messages, estimate_tokens, measure_ratio, save_ratio
from .providers.base import provider_from_env
from .security_policy import REINFORCEMENT_MESSAGE, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.03  # accept within 3% of target (a probe miss fails the gate, see deep_runner)


def build(condition: str, chars: float, seed: int = 1000,
          reinforce_every: int = 10) -> list[dict]:
    items = deep_corpus(condition, int(chars), seed)
    reinforce = REINFORCEMENT_MESSAGE if condition == "reinforced" else None
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages_for(
        items, reinforce, reinforce_every=reinforce_every)


def chars_for(condition: str, target_tokens: int, model: str, seed: int = 1000,
              reinforce_every: int = 10) -> int:
    """Corpus characters that should hit `target_tokens`, from tiktoken + calibrated ratio."""
    # density is not constant in corpus size (overhead amortises, and the reinforcement
    # channel is a fixed share), so solve the linear model  tokens = a*chars + b  exactly
    (c1, t1), (c2, t2) = (100_000, 0), (600_000, 0)
    t1 = count_messages(build(condition, c1, seed, reinforce_every))
    t2 = count_messages(build(condition, c2, seed, reinforce_every))
    slope = (t2 - t1) / (c2 - c1)
    intercept = t1 - slope * c1
    r = measure_ratio(model)
    ratio = r["api_over_cl100k"] if r else 1.0          # api_tokens / cl100k_tokens
    cl100k_needed = target_tokens / ratio
    return max(1_000, int((cl100k_needed - intercept) / slope))


def check_depth(provider, condition: str, target: int, seed: int = 1000,
                reinforce_every: int = 10, max_tokens: int = 8) -> dict:
    """One API call at the intended size: records the API's token count and whether the
    token gate would pass. This is the cheap pre-flight before probes are spent."""
    chars = chars_for(condition, target, provider.model, seed, reinforce_every)
    msgs = build(condition, chars, seed, reinforce_every)
    local = count_messages(msgs)
    provider.max_tokens = max_tokens
    comp = provider.complete(msgs)
    if comp.prompt_tokens and local:
        save_ratio(provider.model, comp.prompt_tokens / local, local,
                   f"check:{condition}:{target}")
    ok = bool(comp.prompt_tokens) and abs(comp.prompt_tokens - target) / target <= 0.03
    return {"condition": condition, "target": target, "chars": chars,
            "cl100k_estimate": local, "api_prompt_tokens": comp.prompt_tokens,
            "gate_ok": ok, "latency_s": round(comp.latency_s, 1),
            "request_id": comp.request_id, "error": comp.error}


def measure(provider, condition: str, chars: float, max_tokens: int = 8) -> dict:
    msgs = build(condition, chars)
    local = count_messages(msgs)
    provider.max_tokens = max_tokens
    comp = provider.complete(msgs)
    if comp.prompt_tokens and local:
        save_ratio(provider.model, comp.prompt_tokens / local, local,
                   f"calibration:{condition}:{int(chars)}")
    return {"chars": int(chars), "messages": len(msgs), "cl100k_tokens": local,
            "api_prompt_tokens": comp.prompt_tokens, "incomplete": comp.incomplete,
            "finish_reason": comp.finish_reason, "latency_s": round(comp.latency_s, 1),
            "error": comp.error, "request_id": comp.request_id}


def calibrate(provider, condition: str, target: int, max_tokens: int = 8,
              log_path: Path | None = None) -> dict:
    """Secant search on corpus characters until measured tokens are within TOL of target."""
    # size the first attempt with tiktoken (+ calibrated correction) instead of a guess
    probe = build(condition, 40_000)
    per_char = count_messages(probe) / 40_000
    ratio = 1.0
    r = measure_ratio(provider.model) if hasattr(provider, "model") else None
    if r:
        ratio = r["api_over_cl100k"]
    chars = target / (per_char * ratio)
    hist = []
    for it in range(6):
        m = measure(provider, condition, chars, max_tokens)
        if log_path:
            with log_path.open("a") as fh:
                fh.write(json.dumps({"condition": condition, "target": target, **m}) + "\n")
        hist.append(m)
        tok = m["api_prompt_tokens"]
        print(f"[cal] {condition} target={target} chars={int(chars)} "
              f"tokens={tok} lat={m['latency_s']}s", flush=True)
        if tok is None:
            chars *= 0.8  # request failed or too large: back off
            continue
        if abs(tok - target) / target <= TOL:
            return {"condition": condition, "target": target, "achieved": tok,
                    "chars": int(chars), "messages": m["messages"], "history": hist}
        if m["finish_reason"] == "length" and tok < max_tokens:
            # truncation is relative to max_tokens, not to context: shrink and retry
            chars *= 0.9
            continue
        # secant step toward the target, damped
        ratio = target / tok
        chars = chars * (1 + 0.7 * (ratio - 1))
    return {"condition": condition, "target": target, "achieved": None,
            "chars": int(chars), "history": hist, "note": "not converged"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="procedural",
                    choices=["neutral", "procedural", "reinforced"])
    ap.add_argument("--target", type=int, default=150_000)
    ap.add_argument("--mode", default="search", choices=["search", "check"],
                    help="check: one pre-flight call at the tiktoken-sized depth")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "raw"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provider = provider_from_env(model=args.model, max_tokens=args.max_tokens)
    log = out / f"calibration_{provider.model}.jsonl"
    if args.mode == "check":
        res = check_depth(provider, args.condition, args.target, max_tokens=args.max_tokens)
        with log.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(json.dumps(res, indent=2))
        (out / f"check_{provider.model}_{args.condition}_{args.target}.json").write_text(
            json.dumps(res, indent=2) + "\n")
        return
    res = calibrate(provider, args.condition, args.target, args.max_tokens, log)
    print(json.dumps(res, indent=2))
    (out / f"calibration_{provider.model}_{args.condition}_{args.target}.json").write_text(
        json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
