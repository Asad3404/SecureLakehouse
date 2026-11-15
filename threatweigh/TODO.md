# ThreatWeigh build TODO

Legend: [x] done · [~] in progress · [ ] not started

1. [x] Setup, data download, EDA (`src/data.py`)
2. [x] `src/cvss.py` + unit tests (brief's 4.6 example is really 4.3; verified against all 110k dataset scores)
3. [x] Expert lexicons (`src/lexicon.py`, `lexicons/*.json`, coverage 85.4%) + KeyBERT cache (`src/keybert_cache.py`)
4. [~] Baselines: TF-IDF+LR [x], plain SBERT [ ], DistilBERT [ ] (`src/baselines.py`)
5. [~] ThreatWeigh model + training (`src/model.py`, `src/train.py`) — code + tests done, budget run in progress
6. [ ] Ablations (a–d), 3 seeds, bootstrap CI, weight sensitivity, time-aware split (queued in run_all.sh)
7. [~] Explainability (`src/explain.py`) — smoke-tested on quick models; full run queued
8. [x] Streamlit UI (`app.py`) + CLI (`predict.py`) — tested with Playwright on budget models (3 tabs, no errors)
9. [~] README (draft) + final results summary
