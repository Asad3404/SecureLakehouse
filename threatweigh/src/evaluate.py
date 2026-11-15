"""
Stage 6 - EVALUATION: turn saved test predictions into the result tables + charts.

Run:  python -m src.evaluate --profile budget

Reads results/preds/<run>/<metric>_seed<seed>.csv (written by train.py / baselines.py)
and writes to results/:
    main_comparison.{csv,md}    accuracy per metric (mean ± std over seeds), mean accuracy,
                                macro-F1, and Base-Score MSE / MAE
    per_metric_details.csv      accuracy, macro-precision, macro-recall, macro-F1 per metric/model/seed
    ablation.{csv,md}           (a) alpha only, (b) +expert, (c) +KeyBERT, (d) full
    sensitivity.{csv,md}        the "both" weight: 2, 5, 10, 20
    temporal.{csv,md}           time-aware split
    significance.{csv,md}       ThreatWeigh vs plain SBERT: paired bootstrap CI + McNemar test
    figures/*.png               bar charts

Base Score: the 8 predicted labels of a CVE are fed into the official CVSS v3.1
formula (src/cvss.py) and compared with the true NVD score.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.cvss import base_score
from src.data import load_split
from src.utils import load_config, metric_names, path

PRETTY = {
    "tfidf_lr": "TF-IDF + LogReg",
    "tfidf_lr_fulltrain": "TF-IDF + LogReg (all 110k train rows)",
    "sbert_mean": "SBERT, mean pooling",
    "distilbert": "DistilBERT, mean pooling",
    "tw_alpha": "(a) learned alpha only",
    "tw_expert": "(b) alpha + expert lexicon",
    "tw_keybert": "(c) alpha + KeyBERT",
    "threatweigh": "ThreatWeigh (alpha + expert + KeyBERT)",
    "threatweigh_w2": "ThreatWeigh, both-weight 2",
    "threatweigh_w5": "ThreatWeigh, both-weight 5",
    "threatweigh_w20": "ThreatWeigh, both-weight 20",
}


# --------------------------------------------------------------------------- #
# Loading predictions
# --------------------------------------------------------------------------- #
def load_run(run_dir) -> dict[int, dict[str, pd.DataFrame]]:
    """{seed: {metric: predictions DataFrame}} for one run folder."""
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for f in sorted(run_dir.glob("*_seed*.csv")):
        m = re.match(r"(\w+?)_seed(\d+)\.csv", f.name)
        if m:
            out.setdefault(int(m.group(2)), {})[m.group(1)] = pd.read_csv(f, keep_default_na=False)
    return out


def metric_scores(df: pd.DataFrame, labels: list[str]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(df["true"], df["pred"], labels=labels,
                                                  average="macro", zero_division=0)
    return {"accuracy": accuracy_score(df["true"], df["pred"]), "macro_precision": p,
            "macro_recall": r, "macro_f1": f1}


def score_errors(preds: dict[str, pd.DataFrame], true_scores: pd.Series, metrics) -> dict:
    """MSE / MAE between the Base Score from predicted labels and the true score."""
    if not all(m in preds for m in metrics):
        return {"score_mse": np.nan, "score_mae": np.nan}
    table = pd.DataFrame({m: preds[m].set_index("ID")["pred"] for m in metrics})
    predicted = table.apply(lambda row: base_score(row.to_dict()), axis=1)
    true = true_scores.loc[predicted.index].astype(float)
    err = predicted - true
    return {"score_mse": float((err ** 2).mean()), "score_mae": float(err.abs().mean())}


def summarise(cfg, run_name: str, temporal=False) -> pd.DataFrame | None:
    """One row per (seed, metric) with all scores, plus score MSE/MAE per seed."""
    run_dir = path(cfg["paths"]["results_dir"]) / "preds" / run_name
    if not run_dir.exists():
        return None
    true_scores = load_split(cfg, "test", temporal=temporal).set_index("ID")["score"]
    rows = []
    for seed, preds in load_run(run_dir).items():
        errs = score_errors(preds, true_scores, metric_names(cfg))
        for m, df in preds.items():
            rows.append({"seed": seed, "metric": m, **metric_scores(df, cfg["metrics"][m]), **errs})
    return pd.DataFrame(rows) if rows else None


def pm(values) -> str:
    """'mean ± std' over seeds (just the mean when there is one seed)."""
    v = np.asarray(values, dtype=float) * 100
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def comparison_table(cfg, runs: dict[str, str], temporal=False) -> pd.DataFrame:
    """Rows = models; columns = accuracy per metric, mean acc, mean macro-F1, MSE, MAE."""
    metrics = metric_names(cfg)
    rows = []
    for label, run_name in runs.items():
        s = summarise(cfg, run_name, temporal)
        if s is None or set(s["metric"]) != set(metrics):
            continue
        seeds = sorted(s["seed"].unique())
        row = {"model": label, "seeds": len(seeds)}
        for m in metrics:
            row[m.upper()] = pm(s[s.metric == m].sort_values("seed")["accuracy"])
        per_seed = s.groupby("seed")
        row["mean_acc"] = pm(per_seed["accuracy"].mean())
        row["mean_macro_f1"] = pm(per_seed["macro_f1"].mean())
        mse, mae = per_seed["score_mse"].first(), per_seed["score_mae"].first()
        row["score_MSE"] = f"{mse.mean():.3f}" + (f" ± {mse.std(ddof=1):.3f}" if len(seeds) > 1 else "")
        row["score_MAE"] = f"{mae.mean():.3f}" + (f" ± {mae.std(ddof=1):.3f}" if len(seeds) > 1 else "")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Significance: ThreatWeigh vs plain SBERT on the same test rows
# --------------------------------------------------------------------------- #
def significance(cfg, run_a: str, run_b: str, n_boot: int, ci: float, seed: int) -> pd.DataFrame:
    """
    For each metric: difference in accuracy (A - B), averaged over the seeds both
    runs share, with a PAIRED bootstrap confidence interval (resampling test CVEs),
    plus an exact McNemar test per seed on the discordant pairs.
    """
    res = path(cfg["paths"]["results_dir"]) / "preds"
    A, B = load_run(res / run_a), load_run(res / run_b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for m in metric_names(cfg) + ["MEAN"]:
        ms = metric_names(cfg) if m == "MEAN" else [m]
        # correctness matrices [seeds, metrics, N]
        ca = np.stack([[(A[s][k]["true"] == A[s][k]["pred"]).to_numpy() for k in ms] for s in seeds])
        cb = np.stack([[(B[s][k]["true"] == B[s][k]["pred"]).to_numpy() for k in ms] for s in seeds])
        n = ca.shape[-1]
        diff = (ca.mean() - cb.mean()) * 100
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[b] = (ca[..., idx].mean() - cb[..., idx].mean()) * 100
        lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        row = {"metric": m.upper(), "acc_diff_pp": round(diff, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
               "CI_excludes_0": bool(lo > 0 or hi < 0)}
        if m != "MEAN":
            pvals = []
            for k in range(len(seeds)):
                a_only = int((ca[k, 0] & ~cb[k, 0]).sum())
                b_only = int((~ca[k, 0] & cb[k, 0]).sum())
                pvals.append(binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0)
            row["mcnemar_p_per_seed"] = ", ".join(f"{p:.3f}" for p in pvals)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def bar_chart(df: pd.DataFrame, title: str, out_file, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    means = df[[m.upper() for m in metrics]].map(lambda s: float(str(s).split(" ")[0]))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(df)
    x = np.arange(len(metrics))
    for k, (_, row) in enumerate(df.iterrows()):
        ax.bar(x + k * width - 0.4 + width / 2, means.iloc[k], width, label=row["model"])
    ax.set_xticks(x, [m.upper() for m in metrics])
    ax.set_ylabel("Test accuracy (%)")
    lo = max(0, np.nanmin(means.to_numpy()) - 5)
    ax.set_ylim(lo, 100)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def write(df: pd.DataFrame, name: str, cfg, title: str):
    if df is None or df.empty:
        print(f"(no data yet for {name})")
        return
    out = path(cfg["paths"]["results_dir"])
    df.to_csv(out / f"{name}.csv", index=False)
    with open(out / f"{name}.md", "w") as f:
        f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n")
    print(f"\n### {title}\n{df.to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="budget")
    args = ap.parse_args()
    cfg = load_config(args.profile)
    p = args.profile
    metrics = metric_names(cfg)
    fig_dir = path(cfg["paths"]["results_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def runs(names, temporal=False):
        return {PRETTY[n]: f"{n}{'_temporal' if temporal else ''}_{p}" for n in names}

    main_runs = runs(["tfidf_lr", "tfidf_lr_fulltrain", "sbert_mean", "distilbert", "threatweigh"])
    main_tab = comparison_table(cfg, main_runs)
    write(main_tab, "main_comparison", cfg, f"Main comparison (test accuracy %, profile={p})")
    bar_chart(main_tab, "Test accuracy per metric", fig_dir / "main_comparison.png", metrics)

    abl = comparison_table(cfg, runs(["tw_alpha", "tw_expert", "tw_keybert", "threatweigh"]))
    write(abl, "ablation", cfg, "Ablation (test accuracy %)")
    bar_chart(abl, "Ablation", fig_dir / "ablation.png", metrics)

    sens = comparison_table(cfg, runs(["threatweigh_w2", "threatweigh_w5", "threatweigh", "threatweigh_w20"]))
    if not sens.empty:
        sens["model"] = sens["model"].replace({PRETTY["threatweigh"]: "ThreatWeigh, both-weight 10 (paper)"})
    write(sens, "sensitivity", cfg, "Sensitivity of the 'both' weight (test accuracy %)")

    temp = comparison_table(cfg, runs(["tfidf_lr", "sbert_mean", "threatweigh"], temporal=True), temporal=True)
    write(temp, "temporal", cfg, "Time-aware split: train <= 2021, test >= 2023 (test accuracy %)")

    sig = significance(cfg, f"threatweigh_{p}", f"sbert_mean_{p}", cfg["bootstrap"]["n_resamples"],
                       cfg["bootstrap"]["ci"], cfg["seed"])
    write(sig, "significance", cfg,
          "ThreatWeigh minus plain SBERT: accuracy difference (pp), 95% paired bootstrap CI, McNemar p")

    # Full per-metric details (accuracy, macro P/R/F1) for every run.
    details = []
    for run_dir in sorted((path(cfg["paths"]["results_dir"]) / "preds").glob(f"*_{p}")):
        s = summarise(cfg, run_dir.name, temporal="_temporal" in run_dir.name)
        if s is not None:
            details.append(s.assign(run=run_dir.name))
    if details:
        d = pd.concat(details)[["run", "seed", "metric", "accuracy", "macro_precision", "macro_recall",
                                "macro_f1", "score_mse", "score_mae"]].round(4)
        d.to_csv(path(cfg["paths"]["results_dir"]) / "per_metric_details.csv", index=False)
        agg = d.groupby(["run", "metric"])[["accuracy", "macro_precision", "macro_recall", "macro_f1"]] \
            .mean().mul(100).round(2).reset_index()
        agg.to_csv(path(cfg["paths"]["results_dir"]) / "per_metric_summary.csv", index=False)


if __name__ == "__main__":
    main()
