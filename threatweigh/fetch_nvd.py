"""
Fallback data source: build final.csv straight from the NVD API 2.0.

Only needed if cloning https://github.com/ShahbazianR/WSymBert fails.

    python fetch_nvd.py                          # all CVEs with a CVSS v3.1 primary score
    NVD_API_KEY=... python fetch_nvd.py          # faster (higher rate limit)
    python fetch_nvd.py --max 5000               # just a sample

Writes data/raw/nvd/final.csv with the same columns as the WSymBERT dataset:
ID, text, vectorString, av, ac, pr, ui, s, c, i, a, score, baseSeverity

Rate limits (NVD): 5 requests / 30 s without a key, 50 / 30 s with one. We sleep
politely between requests and retry on errors.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

from src.cvss import parse_vector
from src.utils import load_config, path

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE = 2000  # the API maximum


def fetch(max_rows: int | None = None) -> pd.DataFrame:
    key = os.environ.get("NVD_API_KEY")
    headers = {"apiKey": key} if key else {}
    pause = 0.7 if key else 6.5                      # stay under the published rate limits
    rows, start, total = [], 0, None
    while total is None or start < total:
        for attempt in range(5):
            r = requests.get(API, params={"startIndex": start, "resultsPerPage": PAGE,
                                          "noRejected": ""}, headers=headers, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(10 * (attempt + 1))         # back off on 403/429/503
        r.raise_for_status()
        data = r.json()
        total = data["totalResults"]
        for item in data["vulnerabilities"]:
            cve = item["cve"]
            text = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), None)
            metrics = cve.get("metrics", {}).get("cvssMetricV31", [])
            primary = next((m for m in metrics if m.get("type") == "Primary"), None)
            if not text or not primary:
                continue
            cv = primary["cvssData"]
            lab = parse_vector(cv["vectorString"])
            rows.append({"ID": cve["id"], "text": text, "vectorString": cv["vectorString"],
                         **{k: lab[k] for k in ("av", "ac", "pr", "ui", "s", "c", "i", "a")},
                         "score": cv["baseScore"], "baseSeverity": cv["baseSeverity"]})
        start += PAGE
        print(f"{min(start, total):,}/{total:,} CVEs scanned, {len(rows):,} with CVSS v3.1", flush=True)
        if max_rows and len(rows) >= max_rows:
            break
        time.sleep(pause)
    return pd.DataFrame(rows[:max_rows] if max_rows else rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, help="stop after this many usable CVEs")
    args = ap.parse_args()
    cfg = load_config()
    out = path(cfg["paths"]["raw_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch(args.max)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}. Next: python -m src.data")


if __name__ == "__main__":
    main()
