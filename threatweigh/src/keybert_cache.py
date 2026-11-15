"""
Stage 3b - MODEL KEYWORDS (K_mdl): KeyBERT keywords, computed once and cached.

Run:  python -m src.keybert_cache                 (rows used by the budget + quick profiles)
      python -m src.keybert_cache --profile full  (every row; slow on CPU)

KeyBERT embeds the description and each candidate word with the SAME Sentence-BERT
model ThreatWeigh fine-tunes (here: the pre-trained checkpoint, before any
fine-tuning), then keeps the words whose embedding is closest (cosine) to
the description embedding. Settings follow the paper:
    top_n=5, keyphrase_ngram_range=(1,1), stop_words=None, no MMR.

The cache is a JSON file  data/cache/keybert_<encoder>.json  mapping
CVE ID -> list of 5 keywords. It is built BEFORE any training.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import pandas as pd

from src.utils import load_config, path, set_seed


def cache_file(cfg: dict):
    slug = cfg["model"]["encoder"].split("/")[-1]
    return path(cfg["paths"]["cache_dir"]) / f"keybert_{slug}.json"


@lru_cache(maxsize=2)
def _keybert(encoder: str):
    """Load KeyBERT on top of the Sentence-BERT checkpoint (cached in memory)."""
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    return KeyBERT(model=SentenceTransformer(encoder, device="cpu"))


def extract_keywords(texts: list[str], cfg: dict) -> list[list[str]]:
    """Top-n unigram KeyBERT keywords for each text (lowercase words)."""
    kb, k = _keybert(cfg["model"]["encoder"]), cfg["keybert"]
    if not texts:
        return []
    out = kb.extract_keywords(
        texts,
        keyphrase_ngram_range=tuple(k["ngram_range"]),
        stop_words=k["stop_words"],
        top_n=k["top_n"],
        use_mmr=k["use_mmr"],
    )
    # KeyBERT returns a flat list for a single document, a list of lists otherwise.
    if len(texts) == 1:
        out = [out]
    return [[word for word, _score in doc] for doc in out]


def load_cache(cfg: dict) -> dict[str, list[str]]:
    f = cache_file(cfg)
    return json.load(open(f)) if f.exists() else {}


def keywords_for(df: pd.DataFrame, cfg: dict, cache: dict | None = None) -> list[list[str]]:
    """Keywords for every row of df (by CVE ID), computing any that are missing."""
    cache = load_cache(cfg) if cache is None else cache
    missing = df[~df["ID"].isin(cache)]
    if len(missing):
        print(f"KeyBERT: computing {len(missing)} missing descriptions...")
        for i in range(0, len(missing), 2000):
            chunk = missing.iloc[i:i + 2000]
            for cve, kws in zip(chunk["ID"], extract_keywords(chunk["text"].tolist(), cfg)):
                cache[cve] = kws
        f = cache_file(cfg)
        f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(f, "w"))
    return [cache[c] for c in df["ID"]]


def main() -> None:
    from src.data import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", nargs="*", default=["budget", "quick"])
    args = ap.parse_args()

    base = load_config()
    set_seed(base["seed"])
    frames = []
    for prof in args.profile:
        cfg = load_config(prof)
        for temporal in (False, True):
            for split in ("train", "val", "test"):
                frames.append(load_split(cfg, split, temporal=temporal))
    rows = pd.concat(frames).drop_duplicates("ID")
    print(f"{len(rows):,} unique descriptions need keywords.")

    import time
    t = time.time()
    kws = keywords_for(rows, base)
    print(f"Done in {time.time() - t:.0f}s. Cache: {cache_file(base)} ({len(load_cache(base)):,} entries)")
    for (_, r), k in list(zip(rows.iterrows(), kws))[:4]:
        print(f"\n{r['ID']}: {r['text'][:140]}...\n  -> {k}")


if __name__ == "__main__":
    main()
