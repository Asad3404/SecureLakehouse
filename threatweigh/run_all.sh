#!/usr/bin/env bash
# =============================================================================
# Run the whole ThreatWeigh pipeline, stage by stage.
#   bash run_all.sh            -> "budget" profile (CPU, the setting reported in results/)
#   bash run_all.sh quick      -> smoke test, a few minutes
#   bash run_all.sh full       -> the paper's setting (whole dataset, 15 epochs; needs a GPU)
# It is resumable: finished (metric, seed) runs are skipped.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
P=${1:-budget}
SEEDS="42 43 44"

python -m src.data                                   # 1. data + EDA
python -m pytest -q                                  # 2. unit tests (CVSS, weighting, splits)
[ -f lexicons/av.json ] || python -m src.lexicon     # 3a. expert lexicons (training split only)
python -m src.keybert_cache                          # 3b. KeyBERT cache (before training)
python -m src.baselines --profile "$P"               # 4. TF-IDF baselines

T="python -m src.train --profile $P"
$T --exp threatweigh --seeds 42 --save-model         # 5. ThreatWeigh (weights saved for the app)
$T --exp sbert_mean  --seeds 42 --save-model         # 4. plain SBERT baseline
$T --exp distilbert  --seeds 42 --save-model         # 4. DistilBERT baseline
$T --exp threatweigh --seeds $SEEDS                  # 6. seed variance
$T --exp sbert_mean  --seeds $SEEDS
for e in tw_alpha tw_expert tw_keybert; do $T --exp $e; done                         # 6. ablation
for e in threatweigh_w2 threatweigh_w5 threatweigh_w20; do $T --exp $e; done        # 6. sensitivity
python -m src.baselines --profile "$P" --temporal                                    # 6. time-aware split
$T --exp sbert_mean --temporal
$T --exp threatweigh --temporal

python -m src.evaluate --profile "$P"                # tables + charts
python -m src.explain  --profile "$P"                # 7. explainability
echo "ALL DONE"
