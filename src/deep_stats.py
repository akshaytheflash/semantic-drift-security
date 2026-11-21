"""Deep-horizon analysis: measured prompt tokens on the x-axis, per-arm violation rates.

Handles both `deep_runner` records (full probe set, token gate) and `subset_runner` records
(preflight + a probe subset at one depth).

python -m src.deep_stats
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "experiments" / "processed"
FIG = ROOT / "figures"
ARM_COLOR = {"neutral": "#1f77b4", "procedural": "#d62728", "reinforced": "#2ca02c"}


def _rows_from_probes(rec: dict, kind: str) -> list[dict]:
    out = []
    for pid, o in rec.get("probes", {}).items():
        out.append({
            "source": kind, "trial_id": rec.get("trial_id", rec.get("kind", "")),
            "arm": rec.get("arm") or rec.get("condition"),
            "depth_requested": rec.get("target_tokens") or rec.get("depth"),
            "measured_tokens": rec.get("measured_tokens") or rec.get("preflight", {}).get("prompt_tokens"),
            "token_gate_ok": rec.get("token_gate_ok", True),
            "seed": rec.get("seed"), "probe": pid, "kind": o["kind"], "clause": o["clause"],
            "visible": o.get("visible", True), "violated": o["violated"],
            "attempted": o["attempted"], "executed": o.get("executed", False),
            "text_violation": o.get("text_violation", False),
            "false_refusal": o.get("false_refusal"),
        })
    return out


def load(paths=None):
    paths = paths or (sorted(glob.glob(str(ROOT / "experiments/raw/deep_*.jsonl")))
                      + sorted(glob.glob(str(ROOT / "experiments/raw/subset_*.jsonl"))))
    rows, pf = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "preflight":
                pf.append({"file": Path(p).name, "depth": r["depth"],
                           "condition": r["condition"], **r["preflight"]})
                continue
            if r.get("aborted"):
                continue
            kind = "deep" if "target_tokens" in r else "subset"
            rows += _rows_from_probes(r, kind)
    return pd.DataFrame(rows), pd.DataFrame(pf)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.token_gate_ok & (df.kind == "adversarial")]
    if d.empty:
        return pd.DataFrame()
    s = (d.groupby(["source", "arm", "depth_requested", "probe"], as_index=False)
         .agg(probes=("violated", "size"), violations=("violated", "sum"),
              text_violations=("text_violation", "sum"),
              measured_tokens=("measured_tokens", "median")))
    s["spvr"] = s.violations / s.probes
    s["text_spvr"] = s.text_violations / s.probes
    return s


def figure(df, pf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.token_gate_ok & (df.kind == "adversarial")]
    s = summary(df)
    if s.empty and pf.empty:
        return False
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    if not s.empty:
        for arm, sub in s.groupby("arm"):
            for src, ss in sub.groupby("source"):
                ss = ss.sort_values("measured_tokens")
                ax.plot(ss.measured_tokens / 1000, ss.spvr, marker="o", ms=3, lw=1.2,
                        ls="-" if src == "deep" else "--",
                        color=ARM_COLOR.get(arm, "#777"),
                        label=f"{arm} ({src})")
    if not pf.empty:
        ok = pf[pf.preflight_finish >= 0] if "preflight_finish" in pf else pf
    ax.set_xlabel("measured prompt tokens (thousands)")
    ax.set_ylabel("SPVR")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None)
    args = ap.parse_args(argv)
    df, pf = load(args.input)
    PROC.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(PROC / "deep_probe_level.csv", index=False)
    if not pf.empty:
        pf.to_csv(PROC / "deep_preflight.csv", index=False)
        print("== depth preflights ==")
        cols = [c for c in ("file", "condition", "depth", "prompt_tokens", "completion_tokens",
                            "finish_reason", "latency_s", "informative") if c in pf]
        print(pf[cols].to_string(index=False))
    if df.empty:
        print("no scored probe records yet")
        return
    s = summary(df)
    print("\n== deep SPVR ==")
    print(s.to_string(index=False))
    print("\nfailing token gate:", int((~df.token_gate_ok).sum()))
    made = figure(df, pf, FIG / "fig7_deep_spvr.png")
    print("figure:", "written" if made else "skipped")


if __name__ == "__main__":
    main()
