"""
Stage 1 - DATA: download, clean, split and describe the NVD dataset.

Run:  python -m src.data

What it does, step by step:
  1. Makes sure the WSymBERT repository is cloned under data/raw/ (it holds the
     dataset zip). If the clone fails, run `python fetch_nvd.py` instead.
  2. Unzips Nvd/Dataset/final.zip -> data/raw/nvd/final.csv (~137k rows).
  3. Cleans the table: drops rows with no text, removes exact duplicate texts
     ("strict" deduplication: the same description is kept only once), and
     checks that every label is one of the allowed letters.
  4. Splits 80/10/10 into train/val/test, stratified on Attack Vector (AV),
     with a fixed seed, and saves the splits as CSV so every model uses them.
  5. Builds a second, time-aware split (old CVEs -> train, new CVEs -> test).
  6. Prints label distributions per metric (a small EDA) and writes them to
     results/eda_label_distribution.csv.

Other modules call load_split() to read a split (optionally subsampled for
the CPU "budget"/"quick" profiles; the subsample is fixed and shared by all models).
"""
from __future__ import annotations

import subprocess
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, metric_names, path

REPO_URL = "https://github.com/ShahbazianR/WSymBert"


# --------------------------------------------------------------------------- #
# 1-2. Download + unzip
# --------------------------------------------------------------------------- #
def ensure_raw_csv(cfg: dict) -> pd.DataFrame:
    """Return the raw NVD table, cloning/unzipping it first if needed."""
    csv_path = path(cfg["paths"]["raw_csv"])
    zip_path = path(cfg["paths"]["raw_zip"])

    if not csv_path.exists():
        if not zip_path.exists():
            repo_dir = zip_path.parents[2]  # .../WSymBert
            print(f"Cloning {REPO_URL} -> {repo_dir}")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
            if result.returncode != 0 or not zip_path.exists():
                raise SystemExit("Clone failed. Run `python fetch_nvd.py` to build the CSV from the NVD API.")
        print(f"Unzipping {zip_path}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            # The zip holds a single file, final.csv.
            zf.extractall(csv_path.parent)

    return pd.read_csv(csv_path)


# --------------------------------------------------------------------------- #
# 3. Clean
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop missing text, strictly deduplicate on text, validate labels."""
    n0 = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    n1 = len(df)

    # Strict deduplication: identical description text -> keep the first row only.
    # (Otherwise the same text could land in both train and test = data leakage.)
    df = df.drop_duplicates(subset=["text"], keep="first")
    n2 = len(df)

    # Keep only rows whose 8 labels are all valid single letters.
    for m, allowed in cfg["metrics"].items():
        df[m] = df[m].astype(str).str.strip().str.upper()
        df = df[df[m].isin(allowed)]
    n3 = len(df)

    # The year in the CVE ID ("CVE-2019-1234" -> 2019). It is the year the ID
    # was reserved, a close proxy for the publication year; used for the
    # time-aware split.
    df["year"] = df["ID"].str.extract(r"CVE-(\d{4})-")[0].astype(int)

    print(f"Rows: raw={n0:,}  after dropping empty text={n1:,}  "
          f"after dedup={n2:,} (removed {n1 - n2:,} duplicates)  after label check={n3:,}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4-5. Split
# --------------------------------------------------------------------------- #
def random_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """80/10/10 split stratified on AV, fixed seed."""
    seed, strat = cfg["seed"], cfg["data"]["stratify_on"]
    val_frac, test_frac = cfg["data"]["val_frac"], cfg["data"]["test_frac"]

    # First cut off the test set, then split the rest into train and val.
    rest, test = train_test_split(df, test_size=test_frac, stratify=df[strat], random_state=seed)
    val_share_of_rest = val_frac / (1 - test_frac)
    train, val = train_test_split(rest, test_size=val_share_of_rest, stratify=rest[strat], random_state=seed)
    return {"train": train, "val": val, "test": test}


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Train on older CVEs, validate on the next year, test on the newest ones."""
    t = cfg["data"]["temporal"]
    return {
        "train": df[df["year"] <= t["train_max_year"]],
        "val": df[df["year"] == t["val_year"]],
        "test": df[df["year"] >= t["test_min_year"]],
    }


def save_splits(splits: dict[str, pd.DataFrame], out_dir) -> None:
    out_dir = path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)


# --------------------------------------------------------------------------- #
# Reading splits back (used by every other stage)
# --------------------------------------------------------------------------- #
def load_split(cfg: dict, name: str, temporal: bool = False) -> pd.DataFrame:
    """
    Read a saved split ("train", "val" or "test").
    If the active profile sets e.g. train_n=2000, a FIXED stratified (on AV)
    subsample of that size is returned. It depends only on cfg["seed"], so
    every model in every experiment sees exactly the same rows.
    """
    folder = cfg["paths"]["temporal_dir"] if temporal else cfg["paths"]["splits_dir"]
    f = path(folder) / f"{name}.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found - run `python -m src.data` first.")
    df = pd.read_csv(f, keep_default_na=False)  # keep_default_na: label "N" must not become NaN

    n = cfg["run"].get(f"{name}_n")
    if n is not None and n < len(df):
        strat = df[cfg["data"]["stratify_on"]]
        # Classes with a single row cannot be stratified; fall back to plain sampling.
        if strat.value_counts().min() >= 2:
            df, _ = train_test_split(df, train_size=n, stratify=strat, random_state=cfg["seed"])
        else:
            df = df.sample(n=n, random_state=cfg["seed"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. EDA
# --------------------------------------------------------------------------- #
def label_distribution(splits: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Percentage of each label per metric and per split, as one tidy table."""
    rows = []
    for m, labels in cfg["metrics"].items():
        for lab in labels:
            row = {"metric": m.upper(), "label": lab}
            for name, part in splits.items():
                row[f"{name}_%"] = round(100 * (part[m] == lab).mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = clean(ensure_raw_csv(cfg), cfg)

    splits = random_split(df, cfg)
    save_splits(splits, cfg["paths"]["splits_dir"])
    print("Random split sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    tsplits = temporal_split(df, cfg)
    save_splits(tsplits, cfg["paths"]["temporal_dir"])
    print("Temporal split sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in tsplits.items()))

    # ---- EDA -------------------------------------------------------------- #
    print("\nCVEs per year (from the CVE ID):")
    print(df["year"].value_counts().sort_index().to_string())

    lengths = df["text"].str.split().str.len()
    print(f"\nDescription length in words: median={lengths.median():.0f}, "
          f"mean={lengths.mean():.1f}, 95th pct={lengths.quantile(.95):.0f}, max={lengths.max()}")
    print("Vector string versions:", df["vectorString"].str.slice(0, 8).value_counts().to_dict())

    dist = label_distribution(splits, cfg)
    print("\nLabel distribution per metric (% of rows):")
    print(dist.to_string(index=False))
    out = path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    dist.to_csv(out / "eda_label_distribution.csv", index=False)

    # Leakage check (also covered by tests/test_data.py).
    tr, va, te = (set(splits[k]["text"]) for k in ("train", "val", "test"))
    print(f"\nText overlap  train∩val={len(tr & va)}  train∩test={len(tr & te)}  val∩test={len(va & te)}")


if __name__ == "__main__":
    main()
