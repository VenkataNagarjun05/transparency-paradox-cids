"""Turn the JSONL results into tables, paired tests and figures."""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd


def load(pattern: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            if "error" in r:
                continue
            fin = r.pop("final")
            r.pop("curve", None)
            r.update({f"final_{k}": v for k, v in fin.items()})
            rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates("run_id")
    return df


def with_drop(df: pd.DataFrame) -> pd.DataFrame:
    """Attack effect = clean-run score minus attacked score, paired by defense and seed."""
    base = df[df.attack == "none"].groupby(["defense", "seed"])[["final_macro_f1", "final_rec_target"]].mean()
    base.columns = ["clean_f1", "clean_rec_target"]
    d = df.merge(base.reset_index(), on=["defense", "seed"], how="left")
    d["f1_drop"] = d.clean_f1 - d.final_macro_f1
    d["target_recall_drop"] = d.clean_rec_target - d.final_rec_target
    return d


def summary(df: pd.DataFrame, mal_frac=0.3) -> pd.DataFrame:
    d = with_drop(df)
    d = d[(d.mal_frac == mal_frac) | (d.attack == "none")]
    g = d.groupby(["attack", "defense"]).agg(
        f1=("final_macro_f1", "mean"), f1_sd=("final_macro_f1", "std"),
        rec_target=("final_rec_target", "mean"), rec_target_sd=("final_rec_target", "std"),
        target_drop=("target_recall_drop", "mean"),
        honest_rejected=("honest_rejected", "mean"), mal_rejected=("mal_rejected", "mean"),
        mass_on_mal=("mass_on_mal", "mean"), n=("seed", "count")).round(3)
    return g


def paired_test(df: pd.DataFrame, attack: str, a: str, b: str, metric="final_rec_target", mal_frac=0.3):
    """Wilcoxon signed-rank across seeds: is defense `a` better than `b` under `attack`?"""
    from scipy.stats import wilcoxon

    d = df[(df.attack == attack) & (df.mal_frac == mal_frac)]
    pa = d[d.defense == a].set_index("seed")[metric]
    pb = d[d.defense == b].set_index("seed")[metric]
    idx = pa.index.intersection(pb.index)
    if len(idx) < 3:
        return dict(n=len(idx), note="too few paired seeds")
    diff = (pa[idx] - pb[idx]).values
    try:
        stat = wilcoxon(diff, alternative="greater")
        p = float(stat.pvalue)
    except ValueError:
        p = float("nan")
    return dict(n=len(idx), mean_diff=float(diff.mean()), p_value=p)


def select_best(df: pd.DataFrame, max_honest_rejected: float = 0.10, mal_frac: float = 0.3):
    """Pick one setting per defense from the TUNING runs.

    Score = the worst DoS-family recall over {no attack, rule-aware, rule-aware with reputation farming}, so a setting
    cannot win by being strong under attack but poor on clean traffic. Settings that reject more than
    `max_honest_rejected` of honest tenants without an attack are not eligible (if none qualifies, the least
    harmful one is taken). Ties go to the higher clean macro-F1."""
    d = df.copy()
    if "dparams" not in d:
        d["dparams"] = "{}"
    rows = []
    for (defense, dp), g in d.groupby(["defense", "dparams"]):
        per = g.groupby("attack")["final_rec_target"].mean()
        if not {"none", "rule_aware", "rule_aware_farm"} <= set(per.index):
            continue
        clean = g[g.attack == "none"]
        hr = clean["honest_rejected"].fillna(0).mean()
        rows.append(dict(defense=defense, dparams=dp, worst_recall=per[["none", "rule_aware", "rule_aware_farm"]].min(),
                         clean_recall=per["none"], rule_aware=per["rule_aware"], farm=per["rule_aware_farm"],
                         clean_f1=clean["final_macro_f1"].mean(), honest_rejected=hr, n=len(g)))
    T = pd.DataFrame(rows)
    best = {}
    for defense, g in T.groupby("defense"):
        ok = g[g.honest_rejected <= max_honest_rejected]
        pool = ok if len(ok) else g.sort_values("honest_rejected").head(1)
        pick = pool.sort_values(["worst_recall", "clean_f1"], ascending=False).iloc[0]
        best[defense] = json.loads(pick.dparams)
    return best, T.sort_values(["defense", "worst_recall"], ascending=[True, False])
