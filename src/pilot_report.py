"""Reproducible pilot report: runs the deterministic pipeline and prints the headline
numbers from the supplied raw files. No numbers are typed by hand.

python -m src.pilot_report --input experiments/raw/runs_*.jsonl
"""

from __future__ import annotations

import argparse

import pandas as pd

from .statistics import ADV, fit_gee, group_checkpoint_table, load, trial_level


def amenities(table: pd.DataFrame) -> pd.DataFrame:
    """Action-level vs text-level verdicts, and silent vs visible probes."""
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--schema", default="probes-v2")
    args = ap.parse_args(argv)
    df = load(args.input, args.schema)
    tl = trial_level(df)
    table = group_checkpoint_table(df)

    print("== checkpoint table ==")
    print(table[["group", "checkpoint", "violations", "probes", "spvr", "ci_lo", "ci_hi",
                 "text_spvr", "mean_tokens"]].to_string(index=False))

    adv = df[(df.kind == "adversarial")]
    print("\n== action vs text view (violations by arm) ==")
    view = adv.groupby("group").agg(action=("violated", "mean"),
                                   text=("text_violation", "mean"),
                                   attempts=("attempted", "mean")).reset_index()
    view["action_only"] = view["action"] - view["text"]
    print(view.round(3).to_string(index=False))

    print("\n== per-probe violation rate ==")
    pp = (adv.groupby(["probe", "clause", "visible"])
          .agg(rate=("violated", "mean"), n=("violated", "size")).reset_index())
    print(pp.round(3).to_string(index=False))

    print("\n== silent probes (P10, P11) ==")
    sil = adv[~adv.visible].groupby(["group", "probe"]).agg(
        rate=("violated", "mean"), text=("text_violation", "mean")).reset_index()
    print(sil.round(3).to_string(index=False))

    print("\n== benign utility (false refusal) ==")
    ben = df[df.kind == "benign"].groupby("group")["false_refusal"].mean()
    print(ben.round(3).to_string())

    res = fit_gee(tl)
    if res is not None:
        print(f"\n== GEE (n={int(res.model.endog.shape[0])} trial-checkpoints) ==")
        for k, v in res.params.items():
            print(f"  {k:22s} {v:+.3f}   p={res.pvalues[k]:.3g}")


if __name__ == "__main__":
    main()
