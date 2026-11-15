"""
Stage 7 - EXPLAINABILITY with Captum Integrated Gradients (IG).

Run:
    python -m src.explain --profile budget                       # all metrics, ThreatWeigh vs SBERT (+DistilBERT)
    python -m src.explain --profile budget --metrics av --n 50   # small run

What IG does, in one sentence: it walks in small steps from a "neutral" input
(the reference baseline) to the real input and adds up how much each token's
embedding pushed the predicted class probability up or down along the way.

Two reference baselines are compared:
    "pad"  - every token embedding replaced by the [PAD] token embedding
    "ones" - every token embedding replaced by a vector of ones

Outputs (results/explain/):
    examples_<metric>.md        top-10 attributed words for sample CVEs, model vs model
    quantitative.{csv,md}       per model/metric/baseline:
        noise_rate   % of the top-10 words that are stop-words, pure numbers / version
                     strings, or <= 2 characters long (lower = cleaner explanation)
        lexicon_hit  % of the top-10 words that are in the metric's expert lexicon
        infidelity   Captum infidelity (lower = attributions predict the effect of
                     small input perturbations better)
        sensitivity  Captum sensitivity_max (lower = explanation changes less under tiny noise)
        deletion_auc area under "probability of the predicted class" while the
                     highest-attributed tokens are deleted first (lower = the
                     attribution found tokens the model really relies on);
                     deletion_auc_random is the same with a random order, as a reference.

IMPORTANT: attributions are DESCRIPTIVE. They show which inputs the trained
network's output is sensitive to; they are not proof that the model "reasons"
like a security analyst, and they do not establish causality.
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.data import load_split
from src.keybert_cache import keywords_for, load_cache
from src.lexicon import expert_cues, find_spans
from src.model import ThreatWeighModel, symbolic_weights
from src.utils import load_config, metric_names, path, set_seed

NUMBER_OR_VERSION = re.compile(r"^(v|version)?[\d][\d.\-_x]*$|^[\d.\-_x]+$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Loading a trained model
# --------------------------------------------------------------------------- #
def load_trained(run_name: str, metric: str, cfg: dict):
    from transformers import AutoTokenizer

    f = path(cfg["paths"]["models_dir"]) / run_name / f"{metric}.pt"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - train with --save-model first.")
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    exp = ckpt["experiment"]
    model = ThreatWeighModel(exp["checkpoint"], len(ckpt["labels"]), exp["pooling"], cfg["model"]["eps"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, AutoTokenizer.from_pretrained(exp["checkpoint"]), exp, ckpt["labels"]


def prepare(text: str, metric: str, exp: dict, tokenizer, cfg: dict, keywords=None):
    """Tokenise one description and build its symbolic weights (exactly as in training)."""
    enc = tokenizer(text, truncation=True, max_length=cfg["model"]["max_length"], return_offsets_mapping=True)
    cues = expert_cues(metric, cfg) if exp["use_exp"] else None
    kws = keywords if exp["use_kw"] else None
    w = symbolic_weights(enc["offset_mapping"], text, cues, kws, exp["both_weight"], exp["one_weight"])
    ids = torch.tensor([enc["input_ids"]])
    return ids, torch.ones_like(ids), torch.tensor([w]), enc["offset_mapping"]


def make_forward(model):
    """Captum calls this with token EMBEDDINGS; it returns class probabilities."""
    def forward(embeds, mask, weights):
        # Captum may expand the batch (IG steps, perturbations): repeat mask/weights to match.
        reps = embeds.shape[0] // mask.shape[0]
        if reps > 1:
            mask, weights = mask.repeat(reps, 1), weights.repeat(reps, 1)
        return torch.softmax(model(attention_mask=mask, sym_weights=weights, inputs_embeds=embeds), -1)
    return forward


def reference(kind: str, embeds: torch.Tensor, model, tokenizer) -> torch.Tensor:
    if kind == "pad":
        pad = model.embeddings()(torch.tensor([tokenizer.pad_token_id]))
        return pad.expand_as(embeds).detach().clone()
    if kind == "ones":
        return torch.ones_like(embeds)
    raise ValueError(kind)


def integrated_gradients(model, tokenizer, ids, mask, w, target, kind, steps):
    from captum.attr import IntegratedGradients

    embeds = model.embeddings()(ids).detach()
    ig = IntegratedGradients(make_forward(model))
    attr = ig.attribute(embeds, baselines=reference(kind, embeds, model, tokenizer),
                        additional_forward_args=(mask, w), target=target, n_steps=steps,
                        internal_batch_size=steps)
    return attr, embeds


# --------------------------------------------------------------------------- #
# From sub-word attributions to word attributions
# --------------------------------------------------------------------------- #
def word_attributions(text: str, offsets, token_attr: np.ndarray):
    """
    Sum sub-word attributions into whole words (split on whitespace/punctuation),
    skipping special tokens. Returns a list of (word, start, end, attribution).
    """
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[\w\-]+|[^\w\s]", text)]
    scores = np.zeros(len(words))
    for (s, e), a in zip(offsets, token_attr):
        if s == e:
            continue  # [CLS] / [SEP]
        for k, (_, ws, we) in enumerate(words):
            if s < we and ws < e:
                scores[k] += a
                break
    return [(w, s, e, float(a)) for (w, s, e), a in zip(words, scores)]


def top_words(word_attr, k=10):
    """The k words with the largest |attribution| (words cut off by truncation have 0 and are skipped)."""
    ranked = sorted((x for x in word_attr if x[3] != 0), key=lambda x: -abs(x[3]))
    return ranked[:k]


def is_noise(word: str) -> bool:
    w = word.lower()
    return w in ENGLISH_STOP_WORDS or bool(NUMBER_OR_VERSION.match(w)) or len(w) <= 2


def in_lexicon(word_tuple, lexicon_spans) -> bool:
    _, s, e, _ = word_tuple
    return any(s < le and ls < e for ls, le, _ in lexicon_spans)


# --------------------------------------------------------------------------- #
# Quantitative checks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deletion_auc(model, ids, w, token_attr, target, order="attr", rng=None, steps=10):
    """
    Delete the most-attributed tokens first (or a random order) and track the
    probability of the originally predicted class. Returns the normalised AUC.
    """
    content = [i for i in range(1, ids.shape[1] - 1)]          # never delete [CLS]/[SEP]
    if order == "attr":
        content.sort(key=lambda i: -token_attr[i])              # most positive evidence first
    else:
        rng.shuffle(content)
    probs = []
    for k in range(steps + 1):
        n_del = round(len(content) * k / steps)
        drop = set(content[:n_del])
        keep = [i for i in range(ids.shape[1]) if i not in drop]
        p = torch.softmax(model(ids[:, keep], torch.ones(1, len(keep), dtype=torch.long), w[:, keep]), -1)
        probs.append(p[0, target].item())
    return float(np.trapezoid(probs, dx=1 / steps))


def explain_model(run_name: str, metric: str, cfg: dict, sample: pd.DataFrame, kw_lists, kinds, steps,
                  n_sens: int, n_infid: int):
    """Run IG + all quantitative checks for one model and one metric."""
    from captum.metrics import infidelity, sensitivity_max

    model, tok, exp, labels = load_trained(run_name, metric, cfg)
    cues = expert_cues(metric, cfg)
    fwd = make_forward(model)
    rng = np.random.default_rng(cfg["seed"])
    examples, per_kind = [], {k: {"noise": [], "lex": [], "infid": [], "sens": [], "del": [], "del_rand": []}
                              for k in kinds}

    for n, (row, kws) in enumerate(zip(sample.itertuples(), kw_lists)):
        ids, mask, w, offsets = prepare(row.text, metric, exp, tok, cfg, kws)
        with torch.no_grad():
            probs = torch.softmax(model(ids, mask, w), -1)[0]
        target = int(probs.argmax())
        lex_spans = find_spans(row.text, cues)
        for kind in kinds:
            attr, embeds = integrated_gradients(model, tok, ids, mask, w, target, kind, steps)
            tok_attr = attr.sum(-1)[0].detach().numpy()
            words = top_words(word_attributions(row.text, offsets, tok_attr), cfg["explain"]["top_k"])
            st = per_kind[kind]
            if words:
                st["noise"].append(np.mean([is_noise(x[0]) for x in words]))
                st["lex"].append(np.mean([in_lexicon(x, lex_spans) for x in words]))
            st["del"].append(deletion_auc(model, ids, w, tok_attr, target, "attr"))
            st["del_rand"].append(deletion_auc(model, ids, w, tok_attr, target, "random", rng))

            if n < n_infid:
                def perturb(inp):
                    noise = torch.randn_like(inp) * 0.01
                    return noise, inp - noise
                st["infid"].append(infidelity(fwd, perturb, embeds, attr.detach(), additional_forward_args=(mask, w),
                                              target=target, n_perturb_samples=cfg["explain"]["infidelity_perturbations"]
                                              ).item())
            if n < n_sens:
                def explain_fn(inp, mask_, w_):
                    from captum.attr import IntegratedGradients
                    return IntegratedGradients(fwd).attribute(
                        inp, baselines=reference(kind, inp, model, tok), additional_forward_args=(mask_, w_),
                        target=target, n_steps=max(4, steps // 2), internal_batch_size=steps)
                st["sens"].append(sensitivity_max(explain_fn, embeds, perturb_radius=0.01, n_perturb_samples=3,
                                                  additional_forward_args=(mask, w)).item())
            if kind == kinds[0] and n < 8:
                examples.append({"ID": row.ID, "text": row.text, "true": getattr(row, metric), "pred": labels[target],
                                 "conf": float(probs[target]),
                                 "top": [(x[0], round(x[3], 3), in_lexicon(x, lex_spans)) for x in words]})
    rows = []
    for kind, st in per_kind.items():
        rows.append({"run": run_name, "metric": metric.upper(), "baseline": kind, "n": len(sample),
                     "noise_rate_%": 100 * np.mean(st["noise"]), "lexicon_hit_%": 100 * np.mean(st["lex"]),
                     "infidelity": np.mean(st["infid"]) if st["infid"] else np.nan,
                     "sensitivity": np.mean(st["sens"]) if st["sens"] else np.nan,
                     "deletion_auc": np.mean(st["del"]), "deletion_auc_random": np.mean(st["del_rand"])})
    return rows, examples


def examples_markdown(metric: str, per_model: dict[str, list[dict]]) -> str:
    """Side-by-side top-10 words for the same CVEs. Lexicon words are marked with *."""
    names = list(per_model)
    out = [f"# Top-10 attributed words, metric {metric.upper()} (IG, [PAD] baseline)\n",
           "Words marked `*` are in the expert lexicon. Attribution is descriptive, not causal.\n"]
    for k in range(min(len(v) for v in per_model.values())):
        first = per_model[names[0]][k]
        out.append(f"\n## {first['ID']} (true {metric.upper()} = {first['true']})\n\n> {first['text'][:400]}\n")
        for name in names:
            ex = per_model[name][k]
            words = ", ".join(f"{'*' if lex else ''}{w}" for w, _, lex in ex["top"])
            out.append(f"- **{name}** -> predicted {ex['pred']} ({ex['conf']:.2f}): {words}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="budget")
    ap.add_argument("--metrics", nargs="*")
    ap.add_argument("--n", type=int, help="number of test CVEs (default: config explain.n_samples)")
    ap.add_argument("--models", nargs="*", default=["threatweigh", "sbert_mean", "distilbert"])
    ap.add_argument("--distilbert-n", type=int, default=150, help="DistilBERT is ~3x slower: fewer CVEs")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["train"]["num_threads"])
    ex_cfg = cfg["explain"]
    n = args.n or ex_cfg["n_samples"]
    test = load_split(cfg, "test")
    sample = test.sample(n=min(n, len(test)), random_state=cfg["seed"]).reset_index(drop=True)
    kw_lists = keywords_for(sample, cfg, load_cache(cfg))
    out_dir = path(cfg["paths"]["results_dir"]) / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    qfile = out_dir / "quantitative.csv"
    all_rows = pd.read_csv(qfile).to_dict("records") if qfile.exists() else []
    done = {(r["run"], r["metric"]) for r in all_rows}

    for metric in args.metrics or metric_names(cfg):
        per_model = {}
        for model_name in args.models:
            run_name = f"{model_name}_{args.profile}"
            if (run_name, metric.upper()) in done:
                print(f"skip {run_name} {metric} (done)")
                continue
            m_n = min(len(sample), args.distilbert_n) if model_name == "distilbert" else len(sample)
            t0 = time.time()
            try:
                rows, examples = explain_model(run_name, metric, cfg, sample.iloc[:m_n], kw_lists[:m_n],
                                               ["pad", "ones"], ex_cfg["ig_steps"],
                                               n_sens=min(ex_cfg["n_sensitivity"], m_n), n_infid=min(200, m_n))
            except FileNotFoundError as e:
                print(e)
                continue
            all_rows += rows
            per_model[model_name] = examples
            pd.DataFrame(all_rows).round(4).to_csv(qfile, index=False)
            for r in rows:
                print(f"{run_name:>22} {metric.upper():>2} {r['baseline']:>4}: noise={r['noise_rate_%']:.1f}% "
                      f"lex={r['lexicon_hit_%']:.1f}% infid={r['infidelity']:.4g} sens={r['sensitivity']:.4g} "
                      f"delAUC={r['deletion_auc']:.3f} (random {r['deletion_auc_random']:.3f})  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if per_model:
            (out_dir / f"examples_{metric}.md").write_text(examples_markdown(metric, per_model))

    q = pd.DataFrame(all_rows)
    if not q.empty:
        summary = q.groupby(["run", "baseline"])[["noise_rate_%", "lexicon_hit_%", "infidelity", "sensitivity",
                                                  "deletion_auc", "deletion_auc_random"]].mean().round(4).reset_index()
        summary.to_csv(out_dir / "quantitative_summary.csv", index=False)
        (out_dir / "quantitative_summary.md").write_text(
            "### Explanation quality, averaged over metrics\n\n" + summary.to_markdown(index=False) + "\n")
        print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
