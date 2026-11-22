"""Draw the depth ladder and compaction figures from processed data.

python -m src.depth_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "experiments" / "raw"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def ladder_rows() -> pd.DataFrame:
    rows = []
    for name, model in (("ladder_deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),):
        f = RAW / f"{name}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            adv = [o for k, o in r["probes"].items()
                   if o["kind"] == "adversarial"]
            rows.append({"model": model, "tokens": r["measured_tokens"],
                         "viol": sum(o["violated"] for o in adv), "n": len(adv)})
    for name, model in (("full_deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
                        ("full_glm-5.3-flash", "GLM-5.3 Flash")):
        f = RAW / f"{name}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            pm = r["probe_map"]
            adv = [o for k, o in r["probes"].items()
                   if o["kind"] == "adversarial"]
            rows.append({"model": model, "tokens": r["measured_tokens"],
                         "viol": sum(o["violated"] for o in adv), "n": len(adv)})
    return pd.DataFrame(rows)


def fig_ladder(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for model, sub in df.groupby("model"):
        agg = (sub.groupby("tokens", as_index=False)
               .agg(v=("viol", "sum"), n=("n", "sum")))
        agg["rate"] = agg.v / agg.n
        ax.plot(agg.tokens / 1000, agg.rate, marker="o", ms=3.5, lw=1.3,
                label=f"{model}")
    ax.axvline(1000, ls="--", lw=0.8, color="#999")
    ax.text(1000, 0.04, " advertised\n 1M window", fontsize=5.5, color="#666")
    ax.set_xlabel("measured prompt tokens (thousands)")
    ax.set_ylabel("violation rate (action view)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_depth_ladder.png")
    plt.close(fig)
    return True


def fig_ceiling() -> bool:
    f = list(RAW.glob("ceiling_*.jsonl")) + list(RAW.glob("ladder_cal_*.jsonl"))
    rows = []
    for p in f:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "prompt_tokens" in r:
                rows.append({"tok": r.get("prompt_tokens") or r.get("target"),
                             "ok": bool(r.get("accepted"))})
    if not rows:
        return False
    d = pd.DataFrame(rows).sort_values("tok")
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ok = d[d.ok]
    bad = d[~d.ok]
    ax.scatter(ok.tok / 1000, [1] * len(ok), s=14, color="#2ca02c", label="accepted")
    ax.scatter(bad.tok / 1000, [1] * len(bad), s=14, color="#d62728", marker="x",
               label="rejected")
    ax.axvline(1048.576, ls="--", lw=0.9, color="#444")
    ax.text(1048.576, 1.05, " enforced\n 1,048,576", fontsize=5.5, color="#444")
    ax.set_yticks([])
    ax.set_ylim(0.9, 1.15)
    ax.set_xlabel("prompt tokens (thousands)")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_ceiling.png")
    plt.close(fig)
    return True


def fig_compaction() -> bool:
    f = RAW / "compact_deepseek-v4.1-flash.jsonl"
    if not f.exists():
        return False
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, r in enumerate(rows):
        s = r["summary"].lower()
        keep = sum(k in s for k in ("/vault", "approval", "external", "explicit"))
        ax.bar(i, r["session_tokens"] / 1000, color="#bbb", width=0.5)
        ax.text(i, 8, f"{keep}/4 rules\nkept", ha="center", fontsize=6)
        ax.text(i, r["session_tokens"] / 1000 + 8,
                f"→ {r['compact_prompt_tokens']} tok", ha="center", fontsize=6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"compaction @ {r['depth']//1000}k" for r in rows], fontsize=6.5)
    ax.set_ylabel("session tokens (thousands)")
    ax.set_ylim(0, max(r["session_tokens"] for r in rows) / 1000 * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_compaction.png")
    plt.close(fig)
    return True


if __name__ == "__main__":
    lad = fig_ladder(ladder_rows())
    print("ladder figure:", lad)
    print("ceiling figure:", fig_ceiling())
    print("compaction figure:", fig_compaction())
