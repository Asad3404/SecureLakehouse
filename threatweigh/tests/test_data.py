"""Tests for the data splits: no leakage, correct proportions, reproducible subsamples."""
import pandas as pd
import pytest

from src.data import clean, load_split, random_split
from src.utils import ROOT, load_config

SPLITS = ROOT / "data/splits"
needs_data = pytest.mark.skipif(not (SPLITS / "train.csv").exists(), reason="run `python -m src.data` first")


def _toy_frame(n=400):
    """A small synthetic table with the real column layout."""
    av = (["N"] * 6 + ["L"] * 3 + ["A"])  # 60/30/10 AV mix
    rows = []
    for k in range(n):
        rows.append({"ID": f"CVE-20{16 + k % 8}-{k}", "text": f"description number {k}",
                     "vectorString": "", "av": av[k % 10], "ac": "L", "pr": "N", "ui": "N", "s": "U",
                     "c": "H", "i": "H", "a": "H", "score": 9.8, "baseSeverity": "CRITICAL"})
    rows.append(dict(rows[0]))                  # an exact duplicate text
    rows.append(dict(rows[1], text=None))       # a missing text
    return pd.DataFrame(rows)


def test_clean_drops_missing_and_duplicates():
    cfg = load_config()
    df = clean(_toy_frame(), cfg)
    assert df["text"].notna().all()
    assert df["text"].is_unique
    assert len(df) == 400


def test_random_split_is_disjoint_and_80_10_10():
    cfg = load_config()
    df = clean(_toy_frame(), cfg)
    s = random_split(df, cfg)
    assert len(s["train"]) == 320 and len(s["val"]) == 40 and len(s["test"]) == 40
    ids = [set(s[k]["ID"]) for k in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    # Stratification keeps the AV mix in every split.
    for part in s.values():
        assert abs((part["av"] == "N").mean() - 0.6) < 0.05


@needs_data
def test_saved_splits_have_no_leakage():
    parts = {k: pd.read_csv(SPLITS / f"{k}.csv", keep_default_na=False) for k in ("train", "val", "test")}
    texts = {k: set(v["text"]) for k, v in parts.items()}
    ids = {k: set(v["ID"]) for k, v in parts.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (texts[a] & texts[b]), f"text leakage between {a} and {b}"
        assert not (ids[a] & ids[b]), f"ID leakage between {a} and {b}"
    total = sum(len(v) for v in parts.values())
    assert abs(len(parts["train"]) / total - 0.8) < 0.01
    # "N" (None/Network) must survive CSV reading as the letter N, not NaN.
    assert "N" in set(parts["train"]["av"])


@needs_data
def test_profile_subsample_is_fixed_and_shared():
    cfg = load_config("budget")
    a = load_split(cfg, "train")
    b = load_split(cfg, "train")
    assert len(a) == cfg["run"]["train_n"]
    assert list(a["ID"]) == list(b["ID"])        # identical rows every time
    test_ids = set(load_split(cfg, "test")["ID"])
    assert not (set(a["ID"]) & test_ids)
