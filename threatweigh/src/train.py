"""
Stage 5 - TRAINING: one independent model per CVSS metric.

Examples
    python -m src.train --exp threatweigh --quick                  # smoke test, CPU, minutes
    python -m src.train --exp threatweigh --profile budget         # reported setting (CPU)
    python -m src.train --exp sbert_mean  --profile budget --seeds 42 43 44
    python -m src.train --exp threatweigh --profile full           # paper setting (GPU)
    python -m src.train --exp threatweigh --profile budget --temporal   # time-aware split

For every metric it:
  1. tokenises the train / val / test rows and pre-computes the symbolic weights,
  2. fine-tunes encoder + pooling + classifier with AdamW and cross-entropy
     (no class weighting), evaluating validation accuracy after each epoch,
  3. keeps the checkpoint with the best validation accuracy (early stopping),
  4. predicts the test rows ONCE with that best checkpoint and saves the predictions to
       results/preds/<run>/<metric>_seed<seed>.csv
     (and, with --save-model, the weights to models/<run>/<metric>.pt).
Tables are produced afterwards by src/evaluate.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import load_split
from src.keybert_cache import keywords_for, load_cache
from src.lexicon import expert_cues
from src.model import EXPERIMENTS, ThreatWeighModel, resolve_experiment, symbolic_weights
from src.utils import get_device, load_config, metric_names, path, set_seed


# --------------------------------------------------------------------------- #
# Turning a table of descriptions into model inputs
# --------------------------------------------------------------------------- #
def encode(df: pd.DataFrame, metric: str, exp: dict, cfg: dict, tokenizer, kw_cache: dict) -> list[dict]:
    """Tokenise each description and attach its symbolic weights and label id."""
    labels = cfg["metrics"][metric]
    cues = expert_cues(metric, cfg) if exp["use_exp"] else None
    keywords = keywords_for(df, cfg, kw_cache) if exp["use_kw"] else [None] * len(df)
    enc = tokenizer(df["text"].tolist(), truncation=True, max_length=cfg["model"]["max_length"],
                    return_offsets_mapping=True)
    items = []
    for k, text in enumerate(df["text"]):
        offsets = enc["offset_mapping"][k]
        w = symbolic_weights(offsets, text, cues, keywords[k], exp["both_weight"], exp["one_weight"])
        items.append({
            "input_ids": enc["input_ids"][k],
            "weights": w,
            "label": labels.index(df[metric].iloc[k]) if metric in df else -1,
        })
    return items


def collate(batch: list[dict], pad_id: int) -> dict:
    """Pad a batch to its own longest sequence (dynamic padding = faster on CPU)."""
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    w = torch.ones((len(batch), T))
    for r, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[r, :n] = torch.tensor(b["input_ids"])
        mask[r, :n] = 1
        w[r, :n] = torch.tensor(b["weights"])
    return {"input_ids": ids, "attention_mask": mask, "sym_weights": w,
            "labels": torch.tensor([b["label"] for b in batch])}


def length_grouped_batches(items: list[dict], batch_size: int, rng: random.Random) -> list[list[int]]:
    """
    Shuffle, then sort inside chunks of 50 batches by length, so each batch holds
    sequences of similar length (less padding), then shuffle the batches.
    """
    idx = list(range(len(items)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(items[i]["input_ids"]))
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, items, pad_id, device, batch_size=64):
    """Return class probabilities [N, n_labels] for a list of encoded items."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))  # fast: similar lengths
    probs = np.zeros((len(items), model.classifier.out_features), dtype=np.float32)
    for b in range(0, len(order), batch_size):
        ids = order[b:b + batch_size]
        batch = {k: v.to(device) for k, v in collate([items[i] for i in ids], pad_id).items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["sym_weights"])
        probs[ids] = torch.softmax(logits, -1).cpu().numpy()
    return probs


# --------------------------------------------------------------------------- #
# Training one metric
# --------------------------------------------------------------------------- #
def train_metric(cfg, metric, exp, seed, splits, tokenizer, kw_cache, device, log):
    set_seed(seed)
    labels = cfg["metrics"][metric]
    items = {k: encode(df, metric, exp, cfg, tokenizer, kw_cache) for k, df in splits.items()}
    pad_id = tokenizer.pad_token_id

    model = ThreatWeighModel(exp["checkpoint"], len(labels), exp["pooling"], cfg["model"]["eps"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()                     # no class weighting (as in the paper)
    rng = random.Random(seed)

    best_acc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, cfg["run"]["epochs"] + 1):
        model.train()
        t0, total = time.time(), 0.0
        batches = length_grouped_batches(items["train"], cfg["train"]["batch_size"], rng)
        for bidx in batches:
            batch = {k: v.to(device) for k, v in collate([items["train"][i] for i in bidx], pad_id).items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch["sym_weights"])
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        val_probs = predict(model, items["val"], pad_id, device)
        val_acc = float((val_probs.argmax(1) == np.array([i["label"] for i in items["val"]])).mean())
        history.append({"epoch": epoch, "train_loss": total / len(batches), "val_acc": val_acc,
                        "seconds": round(time.time() - t0, 1)})
        log(f"  [{metric.upper()}] epoch {epoch}: loss={total / len(batches):.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.0f}s)")

        if val_acc > best_acc:                       # keep the best checkpoint (by validation accuracy)
            best_acc, best_state, bad_epochs = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["run"]["patience"]:
                log(f"  [{metric.upper()}] early stopping")
                break

    model.load_state_dict(best_state)
    test_probs = predict(model, items["test"], pad_id, device)   # the ONE look at the test set
    return model, test_probs, history, best_acc


def run(exp_name: str, cfg: dict, seeds: list[int], metrics: list[str], temporal=False, save_model=False):
    exp = resolve_experiment(exp_name, cfg)
    torch.set_num_threads(cfg["train"]["num_threads"])
    device = get_device()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(exp["checkpoint"])
    run_name = f"{exp_name}{'_temporal' if temporal else ''}_{cfg['run']['profile']}"
    pred_dir = path(cfg["paths"]["results_dir"]) / "preds" / run_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(pred_dir / "train_log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    splits = {k: load_split(cfg, k, temporal=temporal) for k in ("train", "val", "test")}
    kw_cache = load_cache(cfg)
    log(f"=== {run_name} on {device} | {exp} | train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} | epochs<={cfg['run']['epochs']}")

    for seed in seeds:
        for metric in metrics:
            out = pred_dir / f"{metric}_seed{seed}.csv"
            if out.exists():
                log(f"  skip {out.name} (already done)")
                continue
            t0 = time.time()
            model, probs, history, best_val = train_metric(cfg, metric, exp, seed, splits, tokenizer,
                                                           kw_cache, device, log)
            labels = cfg["metrics"][metric]
            test = splits["test"]
            pd.DataFrame({"ID": test["ID"], "true": test[metric], "pred": [labels[i] for i in probs.argmax(1)],
                          "confidence": probs.max(1)}).to_csv(out, index=False)
            json.dump({"history": history, "best_val_acc": best_val, "minutes": (time.time() - t0) / 60},
                      open(pred_dir / f"{metric}_seed{seed}_history.json", "w"), indent=1)
            acc = (pd.read_csv(out, keep_default_na=False).eval("true == pred")).mean()
            log(f"  [{metric.upper()}] seed {seed}: best val acc={best_val:.4f}  TEST acc={acc:.4f}  "
                f"({(time.time() - t0) / 60:.1f} min)")
            if save_model and seed == seeds[0]:
                mdir = path(cfg["paths"]["models_dir"]) / run_name
                mdir.mkdir(parents=True, exist_ok=True)
                torch.save({"experiment": exp, "labels": labels, "metric": metric,
                            "state_dict": model.state_dict()}, mdir / f"{metric}.pt")
            del model
    log_f.close()
    return run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="threatweigh", choices=list(EXPERIMENTS))
    ap.add_argument("--profile", default="budget", choices=["full", "budget", "quick"])
    ap.add_argument("--quick", action="store_true", help="shortcut for --profile quick")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--metrics", nargs="*")
    ap.add_argument("--temporal", action="store_true", help="use the time-aware split")
    ap.add_argument("--save-model", action="store_true", help="save weights of the first seed to models/")
    ap.add_argument("--epochs", type=int, help="override the profile's epoch count")
    args = ap.parse_args()

    cfg = load_config("quick" if args.quick else args.profile)
    if args.epochs:
        cfg["run"]["epochs"] = args.epochs
    seeds = args.seeds or [cfg["seed"]]
    metrics = args.metrics or metric_names(cfg)
    run(args.exp, cfg, seeds, metrics, temporal=args.temporal, save_model=args.save_model)


if __name__ == "__main__":
    main()
