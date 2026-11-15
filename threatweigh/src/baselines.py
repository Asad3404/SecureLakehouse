"""
Stage 4 - BASELINES.

Run:
    python -m src.baselines --profile budget            # TF-IDF + logistic regression (seconds)
    python -m src.baselines --profile budget --neural   # + plain SBERT (mean pooling) and DistilBERT

1. TF-IDF + logistic regression, one classifier per metric.
   * "tfidf_lr"           trained on the same rows as the neural models (fair comparison)
   * "tfidf_lr_fulltrain" trained on the WHOLE training split (cheap for a linear model),
                          evaluated on the same test rows, to show how much the
                          budget subsample costs.
   The regularisation strength C is picked on the validation split.
2. Plain SBERT fine-tuned with mean pooling (no symbolic guidance)  -> src/train.py, exp "sbert_mean"
3. DistilBERT fine-tuned (distilbert-base-uncased, mean pooling)   -> src/train.py, exp "distilbert"

All predictions are written to results/preds/<run>/<metric>_seed<seed>.csv,
exactly like the neural models, so src/evaluate.py treats every model the same way.
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split
from src.utils import load_config, metric_names, path, set_seed


def run_tfidf(cfg: dict, full_train: bool = False, temporal: bool = False) -> str:
    set_seed(cfg["seed"])
    test = load_split(cfg, "test", temporal=temporal)
    if full_train:
        # Same validation/test rows, but the classifier sees every training row.
        full_cfg = load_config()
        train = load_split(full_cfg, "train", temporal=temporal)
    else:
        train = load_split(cfg, "train", temporal=temporal)
    val = load_split(cfg, "val", temporal=temporal)

    name = "tfidf_lr_fulltrain" if full_train else "tfidf_lr"
    run_name = f"{name}{'_temporal' if temporal else ''}_{cfg['run']['profile']}"
    out_dir = path(cfg["paths"]["results_dir"]) / "preds" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if all((out_dir / f"{m}_seed{cfg['seed']}.csv").exists() for m in metric_names(cfg)):
        print(f"skip {run_name} (already done)")
        return run_name

    # Word uni+bi-grams, sub-linear term frequency. Fitted on the training rows only.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr, Xva, Xte = vec.fit_transform(train["text"]), vec.transform(val["text"]), vec.transform(test["text"])
    print(f"=== {run_name}: train={len(train):,} val={len(val):,} test={len(test):,} vocab={len(vec.vocabulary_):,}")

    for m in metric_names(cfg):
        best = None
        for C in (0.1, 1.0, 10.0, 100.0):          # pick C on the validation split
            clf = LogisticRegression(C=C, max_iter=3000)
            clf.fit(Xtr, train[m])
            acc = (clf.predict(Xva) == val[m]).mean()
            if best is None or acc > best[0]:
                best = (acc, C, clf)
        val_acc, C, clf = best
        pred = clf.predict(Xte)
        conf = clf.predict_proba(Xte).max(1)
        pd.DataFrame({"ID": test["ID"], "true": test[m], "pred": pred, "confidence": conf}) \
            .to_csv(out_dir / f"{m}_seed{cfg['seed']}.csv", index=False)
        print(f"  [{m.upper()}] C={C:<5} val acc={val_acc:.4f}  TEST acc={(pred == test[m]).mean():.4f}")
    return run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="budget", choices=["full", "budget", "quick"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--neural", action="store_true", help="also fine-tune plain SBERT and DistilBERT")
    ap.add_argument("--seeds", type=int, nargs="*")
    args = ap.parse_args()
    cfg = load_config("quick" if args.quick else args.profile)

    run_tfidf(cfg, temporal=args.temporal)
    run_tfidf(cfg, full_train=True, temporal=args.temporal)

    if args.neural:
        from src.train import run

        seeds = args.seeds or [cfg["seed"]]
        run("sbert_mean", cfg, seeds, metric_names(cfg), temporal=args.temporal, save_model=True)
        run("distilbert", cfg, seeds[:1], metric_names(cfg), temporal=args.temporal, save_model=True)


if __name__ == "__main__":
    main()
