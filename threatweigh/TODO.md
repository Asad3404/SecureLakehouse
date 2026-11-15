# ThreatWeigh build TODO

Legend: [x] done · [~] in progress · [ ] not started

1. [x] Setup, data download, EDA (`src/data.py`)
2. [x] `src/cvss.py` + unit tests (note: brief's 4.6 example is really 4.3)
3. [~] Expert lexicons (`src/lexicon.py`, `lexicons/*.json`) + KeyBERT cache (`src/keybert_cache.py`)
4. [ ] Baselines: TF-IDF+LR, plain SBERT (mean pooling), DistilBERT (`src/baselines.py`)
5. [ ] ThreatWeigh model + training (`src/model.py`, `src/train.py`)
6. [ ] Ablations (a–d), 3 seeds, bootstrap CI, weight sensitivity, time-aware split
7. [ ] Explainability: Captum IG, top-10 tokens, noise rate, infidelity/sensitivity/deletion-AUC (`src/explain.py`)
8. [ ] Streamlit UI (`app.py`) + CLI (`predict.py`)
9. [ ] README + final results summary
