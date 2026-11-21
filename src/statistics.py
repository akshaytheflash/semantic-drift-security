"""Analysis: raw JSONL -> processed tables, metrics, statistics, figures.

python -m src.statistics --input experiments/raw/*.jsonl       # write processed tables
python -m src.statistics --figures                             # regenerate all figures

Every number printed here is computed from recorded trials; nothing is hard-coded.
Trial-clustered inference uses GEE (exchangeable working correlation) so that the
multiple checkpoints of one trial are not treated as independent samples.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "experiments" / "raw"
PROC = ROOT / "experiments" / "processed"
FIG = ROOT / "figures"
ADV = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P10", "P11"]
GROUP_ORDER = ["A", "B", "C", "X", "D", "E5", "E10", "E20", "F",
               "S~B", "S~C", "S~X", "S~D", "S~E5", "S~E10"]
LABEL = {"A": "Baseline", "B": "Neutral (volume)", "C": "Procedural", "X": "Proximate",
         "D": "Interference", "E5": "Reinforced /5", "E10": "Reinforced /10",
         "E20": "Reinforced /20", "F": "Injection control",
         "S~B": "Large: neutral", "S~C": "Large: procedural", "S~X": "Large: proximate",
         "S~D": "Large: interference", "S~E5": "Large: reinforced /5",
         "S~E10": "Large: reinforced /10"}


# --------------------------------------------------------------------------- load
CURRENT_SCHEMA = "probes-v2"


def _record_schema(r: dict) -> str:
    return (r.get("meta") or {}).get("schema_version", "probes-v1")


def load(paths: list[str], schema: str = CURRENT_SCHEMA) -> pd.DataFrame:
    """Load records of ONE probe schema. Mixing schemas would silently change the probe
    denominator, so records from other schemas are skipped with a warning."""
    rows, skipped = [], 0
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if _record_schema(r) != schema:
                skipped += 1
                continue
            for pid, o in r["probes"].items():
                rows.append({
                    "trial_id": r["trial_id"], "group": r["group"],
                    "model": r["model"], "seed": r["seed"],
                    "checkpoint": r["checkpoint"], "frame": r.get("frame", "interaction"),
                    "n_context_items": r["n_context_items"],
                    "history_tokens": r.get("history_tokens"),
                    "probe": pid, "kind": o["kind"], "clause": o["clause"],
                    "channel": o["channel"], "visible": o.get("visible", True),
                    "violated": o["violated"], "attempted": o["attempted"],
                    "executed": o.get("executed", False),
                    "text_violation": o.get("text_violation"),
                    "false_refusal": o.get("false_refusal"),
                    "benign_success": o.get("benign_success"),
                    "n_incomplete": r.get("n_incomplete"),
                    "policy_audit": r.get("policy_audit"),
                })
    if skipped:
        print(f"[load] skipped {skipped} record(s) from a different probe schema "
              f"(wanted {schema}); re-run those arms to include them")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no records with schema {schema}")
    df["checkpoint"] = df["checkpoint"].astype(int)
    return df


# --------------------------------------------------------------------------- metrics
def trial_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial x checkpoint: rates for the adversarial and benign probe sets."""
    a = df[df["kind"] == "adversarial"]
    b = df[df["kind"] == "benign"]
    out = (a.groupby(["trial_id", "group", "model", "seed", "checkpoint", "frame",
                      "n_context_items", "history_tokens"])["violated"]
           .agg(n_probes="size", n_viol="sum").reset_index())
    out["spvr"] = out["n_viol"] / out["n_probes"]
    tv = (a.groupby(["trial_id", "checkpoint"])["text_violation"].sum().rename("n_text_viol"))
    fr = (b.groupby(["trial_id", "checkpoint"])["false_refusal"].mean().rename("false_refusal_rate"))
    out = out.merge(tv, on=["trial_id", "checkpoint"], how="left")
    out = out.merge(fr, on=["trial_id", "checkpoint"], how="left")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def group_checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, cp), sub in df[df["kind"] == "adversarial"].groupby(["group", "checkpoint"]):
        k, n = int(sub["violated"].sum()), len(sub)
        lo, hi = wilson(k, n)
        rows.append({"group": g, "checkpoint": cp, "violations": k, "probes": n,
                     "spvr": k / n, "ci_lo": lo, "ci_hi": hi,
                     "text_spvr": float(sub["text_violation"].mean()),
                     "mean_tokens": float(sub["history_tokens"].dropna().mean())
                     if sub["history_tokens"].notna().any() else float("nan")})
    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], GROUP_ORDER, ordered=True)
    return out.sort_values(["group", "checkpoint"]).reset_index(drop=True)


def per_probe_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["kind"] == "adversarial"]
    piv = (a.groupby(["probe", "clause", "group", "checkpoint"])["violated"].mean()
           .unstack("checkpoint"))
    return piv.reset_index()


def onset_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["kind"] == "adversarial"]
    tr = (a.groupby(["trial_id", "group", "model", "seed", "checkpoint"])["violated"].sum()
          .reset_index())
    first = (tr[tr["violated"] > 0].groupby(["trial_id", "group", "model", "seed"])["checkpoint"]
             .min().reset_index().rename(columns={"checkpoint": "first_violation_cp"}))
    return first


def recoveries(df: pd.DataFrame) -> pd.DataFrame:
    """Recovery after a restatement: share of checkpoints following a restatement where the
    trial is violation-free, split by reinforcement cadence vs the no-reinforcement arm."""
    a = df[df["kind"] == "adversarial"]
    tr = (a.groupby(["trial_id", "group", "checkpoint"])["violated"].mean()
          .reset_index().rename(columns={"violated": "spvr"}))
    tr["clean"] = tr["spvr"] == 0
    return tr.groupby(["group", "checkpoint"])["clean"].mean().reset_index()


# --------------------------------------------------------------------------- statistics
def fit_gee(tl: pd.DataFrame):
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    d = tl.dropna(subset=["spvr"]).copy()
    if d["group"].nunique() < 2 or len(d) < 10:
        return None
    d["condition"] = d["group"].astype(str)
    ref = "A" if "A" in set(d["condition"]) else sorted(d["condition"])[0]
    design = pd.get_dummies(d["condition"], prefix="g", drop_first=False).astype(float)
    drop = f"g_{ref}"
    if drop in design:
        design = design.drop(columns=[drop])
    d["cp_c"] = d["checkpoint"] - d["checkpoint"].mean()
    X = pd.concat([design, d[["cp_c"]]], axis=1)
    for c in [c for c in X.columns if c.startswith("g_")]:
        X[f"{c}:cp"] = X[c] * d["cp_c"]
    X = sm.add_constant(X)
    model = sm.GEE(d["spvr"].clip(0, 1), X, groups=d["trial_id"], family=Binomial(),
                   cov_struct=Exchangeable())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(maxiter=200)
    return res


def mcnemar(a: np.ndarray, b: np.ndarray):
    """Exact McNemar over paired trial-checkpoint outcomes."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    if n10 + n01 == 0:
        return {"n10": n10, "n01": n01, "p": 1.0}
    return {"n10": n10, "n01": n01, "p": binomtest(n10, n10 + n01, 0.5).pvalue}


def wilcoxon_paired(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import wilcoxon
    if len(a) < 3 or (a == b).all():
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
                         "legend.frameon": False, "axes.titlesize": 9,
                         "axes.labelsize": 8})
    return plt


_COLORS = {"A": "#444444", "B": "#1f77b4", "C": "#d62728", "X": "#9467bd", "D": "#ff7f0e",
           "E5": "#2ca02c", "E10": "#98df8a", "E20": "#006d2c", "F": "#8c564b"}


def fig_architecture(path):
    """Schematic of the harness (drawn, not data)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    def box(x, y, w, h, t, fc="#eef3fb"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#3b4a63", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7)
    box(0.1, 2.6, 2.0, 1.1, "Invariant system\npolicy (sha256\nchecked per call)")
    box(2.4, 2.6, 2.2, 1.1, "Accumulated\nbenign context\n(B/C/X/D/E/F)")
    box(0.1, 0.6, 2.0, 1.1, "Actor model\n(BharatCode API)")
    box(2.4, 0.6, 2.2, 1.1, "Tool-call loop\n+ simulated tools")
    box(4.9, 0.6, 2.1, 1.1, "Deterministic\nscorer\n(action + text)")
    box(4.9, 2.6, 2.1, 1.1, "9 adversarial\n2 benign probes\n(branched)")
    box(7.3, 1.6, 2.5, 1.1, "Records: probe outcomes,\ncalls, tokens, request ids")
    ax.annotate("", xy=(2.4, 3.15), xytext=(2.1, 3.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(1.1, 1.7), xytext=(1.1, 2.6), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(2.4, 1.15), xytext=(2.1, 1.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(2.4, 1.5), xytext=(1.1, 1.7), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(4.9, 1.15), xytext=(4.6, 1.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(5.95, 1.7), xytext=(5.95, 2.6), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(7.3, 2.15), xytext=(7.0, 2.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_adherence(table: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for g, sub in table.groupby("group", observed=True):
        ax.plot(sub["checkpoint"], 1 - sub["spvr"], marker="o", ms=2.5, lw=1.1,
                color=_COLORS.get(str(g), "#777"), label=LABEL.get(str(g), str(g)))
        ax.fill_between(sub["checkpoint"], 1 - sub["ci_hi"], 1 - sub["ci_lo"],
                        color=_COLORS.get(str(g), "#777"), alpha=0.12, lw=0)
    ax.set_xlabel("interactions accumulated")
    ax.set_ylabel("policy adherence (1 − SPVR)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5.5, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_violation_prob(table: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for g, sub in table.groupby("group", observed=True):
        ax.plot(sub["checkpoint"], sub["spvr"], marker="o", ms=2.5, lw=1.1,
                color=_COLORS.get(str(g), "#777"), label=LABEL.get(str(g), str(g)))
    ax.set_xlabel("interactions accumulated")
    ax.set_ylabel("SPVR (violations / probes)")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_composition(table: pd.DataFrame, path):
    """B vs C vs X at matched checkpoints: volume against proximity."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, frame, title in zip(axs, ("interaction", "token_match"),
                                ("interaction frame", "token-matched frame")):
        sub_all = table[table["frame"] == frame] if "frame" in table else table
        for g in ("B", "C", "X", "D"):
            sub = sub_all[sub_all["group"].astype(str) == g]
            if sub.empty:
                continue
            ax.plot(sub["checkpoint"], sub["spvr"], marker="o", ms=2.5, lw=1.1,
                    color=_COLORS[g], label=LABEL[g])
        ax.set_title(title)
        ax.set_xlabel("context size")
    axs[0].set_ylabel("SPVR")
    axs[1].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mitigation(table: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for g, sub in table.groupby("group", observed=True):
        if str(g) not in ("C", "E5", "E10", "E20"):
            continue
        ax.plot(sub["checkpoint"], sub["spvr"], marker="o", ms=2.5, lw=1.2,
                color=_COLORS[str(g)], label=LABEL[str(g)])
    ax.set_xlabel("interactions accumulated")
    ax.set_ylabel("SPVR")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_models(table: pd.DataFrame, path):
    plt = _plt()
    models = sorted(table["model"].unique()) if "model" in table else []
    if len(models) < 2:
        return False
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m, sub in table.groupby("model"):
        ax.plot(sub["checkpoint"], sub["spvr"], marker="o", ms=2.5, lw=1.1, label=m)
    ax.set_xlabel("interactions accumulated")
    ax.set_ylabel("SPVR")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None)
    ap.add_argument("--schema", default=CURRENT_SCHEMA)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    paths = args.input or sorted(glob.glob(str(RAW / "*.jsonl")))
    if not paths:
        print("no raw data")
        return
    df = load(paths, args.schema)
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "probe_level.csv", index=False)
    tl = trial_level(df)
    tl.to_csv(PROC / "trial_level.csv", index=False)
    table = group_checkpoint_table(df)
    table.to_csv(PROC / "group_checkpoint.csv", index=False)
    per_probe_table(df).to_csv(PROC / "per_probe.csv", index=False)
    onset_table(df).to_csv(PROC / "onset.csv", index=False)
    print(table.to_string(index=False))
    audit = df["policy_audit"]
    print("\npolicy audit: checked", int(audit.notna().sum()),
          "records, failures", int((audit == False).sum()),   # noqa: E712
          "| unaudited (older schema)", int(audit.isna().sum()))
    res = fit_gee(tl)
    if res is not None:
        out = {"contrasts": {k: {"coef": float(v), "p": float(res.pvalues[k])}
                             for k, v in res.params.items()},
               "n_obs": int(res.model.endog.shape[0])}
        (PROC / "gee.json").write_text(json.dumps(out, indent=2))
        print("\nGEE (trial-clustered) coefficients:")
        for k, v in out["contrasts"].items():
            print(f"  {k:24s} {v['coef']:+.3f}  p={v['p']:.3g}")
    if args.figures:
        fig_adherence(table, FIG / "fig2_adherence_vs_context.png")
        fig_violation_prob(table, FIG / "fig3_spvr_vs_context.png")
        fig_composition(table, FIG / "fig4_composition.png")
        fig_mitigation(table, FIG / "fig5_mitigation.png")
        fig_architecture(FIG / "fig1_architecture.png")
        made = fig_models(table, FIG / "fig6_models.png")
        print("figures written", "" if made else "(fig6 needs >=2 models)")


if __name__ == "__main__":
    main()
