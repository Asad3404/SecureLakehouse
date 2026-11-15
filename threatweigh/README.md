# ⚖️ ThreatWeigh — interpretable CVSS v3.1 prediction from vulnerability text

ThreatWeigh reads the English description of a software vulnerability (a CVE) and predicts the
eight **CVSS v3.1 Base metrics** — Attack Vector (AV), Attack Complexity (AC), Privileges Required
(PR), User Interaction (UI), Scope (S), and the Confidentiality / Integrity / Availability impacts
(C, I, A). The eight predicted labels go into the **official CVSS v3.1 formula** to give the 0–10
Base Score, and **Integrated Gradients** shows which words drove each prediction.

ThreatWeigh is a **simplified, student-friendly re-implementation** of WSymBERT:

> S. L. Mirtaheri et al., *"WSymBERT: a weighted neuro-symbolic model for interpretable CVSS
> prediction"*, International Journal of Information Security, 2026.
> Code and data: <https://github.com/ShahbazianR/WSymBert>

It is **not** the authors' code, and it does **not** reproduce the paper's numbers. The paper
reports a mean accuracy of about **96.2 %** on NVD; that figure is given here for reference only.
See [Results](#results) and [Limitations](#limitations).

---

## Contents
1. [The problem](#the-problem)
2. [The method](#the-method)
3. [Setup](#setup)
4. [Running each stage](#running-each-stage)
5. [Results](#results)
6. [Explainability](#explainability)
7. [Limitations](#limitations)
8. [Project structure](#project-structure)
9. [Credits and licences](#credits-and-licences)

---

## The problem

New CVEs are published every day, and analysts score each one by hand from its text. The task here
is a set of **eight independent text-classification problems**, one per metric:

| Metric | Labels |
|---|---|
| AV Attack Vector | N Network · A Adjacent · L Local · P Physical |
| AC Attack Complexity | L Low · H High |
| PR Privileges Required | N None · L Low · H High |
| UI User Interaction | N None · R Required |
| S Scope | U Unchanged · C Changed |
| C / I / A impact | N None · L Low · H High |

Several classes are rare (AV=P is 1 % of CVEs, A=L is 2 %, AC=H is 6 %), so we report **macro**
precision, recall and F1 next to accuracy.

## The method

There is **one independent model per metric**: 8 models with no shared parameters and no multi-task loss.

```
CVE text ─► Sentence-BERT ─► h_1..h_n ─► learned weights α ─┐
   │                                                           ├─► fused pooling β ─► z ─► linear+softmax ─► label
   ├─► expert lexicon K_exp ──┐                                │                                (×8 metrics)
   └─► KeyBERT keywords K_mdl ┴─► symbolic weight w (11/2/1) ──┘          8 labels ─► CVSS v3.1 formula ─► score
```

1. **Encoder.** A Sentence-BERT checkpoint, fine-tuned separately for each metric. The default is
   `sentence-transformers/all-MiniLM-L6-v2` (fast enough for a CPU). `all-mpnet-base-v2` is the
   stronger option: set `model.encoder` in `config.yaml`. The paper does not name its checkpoint,
   so this choice is ours. Maximum sequence length is 128 tokens (11 % of descriptions are longer and get truncated).
2. **Learned pooling weights.** For token vectors $h_i$:
   $s_i = W_{pool}h_i + b$, and $\alpha_i = \mathrm{softmax}_i(s_i)$, with padding masked out.
3. **Expert lexicon $K_{exp}$** (`src/lexicon.py` → `lexicons/*.json`). For every metric there is a
   *question-style decision tree* written from the FIRST CVSS v3.1 specification. For example:
   *"Does the attacker need to physically touch the device?"* → AV=P → `physical access`, `usb`, `lock screen`, …
   Each candidate cue is then checked **on the training split only**. It is kept if it occurs in at least 5
   descriptions and it is either right at least 50 % of the time or at least 1.5× the label's base rate. The
   lexicons are then **frozen**. Only generic CVSS wording is used: no product names and no CVE-specific phrases.
4. **KeyBERT keywords $K_{mdl}$** (`src/keybert_cache.py`). These are the top 5 unigram keywords per description,
   found with the same SBERT checkpoint, with `keyphrase_ngram_range=(1,1)`, no stop-word filtering and no MMR.
   They are cached to disk before training.
5. **Symbolic weight per token.**
   $w_i = 10$ if the token is in $K_{exp}$ **and** $K_{mdl}$, $1$ if it is in exactly one of them, and $0$
   otherwise; then $w_i \leftarrow w_i + 1$, giving **11 / 2 / 1**. Matching uses character spans, so a
   multi-word phrase such as *"requires local access"* marks every sub-word token inside it.
6. **Fused pooling.**
   $\beta_i = \dfrac{\alpha_i w_i}{\sum_j \alpha_j w_j + \epsilon}$, and $z = \sum_i \beta_i h_i$.
7. **Classifier.** A linear layer and softmax on $z$, trained with cross-entropy (no class weighting).
8. **Training.** AdamW, learning rate 2e-5, batch size 16, up to 15 epochs with early stopping on validation
   accuracy. The best checkpoint is chosen by validation accuracy, and the test split is used **once**, at the end.
9. **Score.** `src/cvss.py` implements the official v3.1 Base Score equations, including the integer-based
   `Roundup` and the scope-changed impact $7.52(ISS-0.029) - 3.25(ISS-0.02)^{15}$. It reproduces the
   `score` column of all 109,991 training CVEs exactly.

**Baselines** (same splits, same hyper-parameters where applicable): TF-IDF + logistic regression;
SBERT fine-tuned with plain **mean pooling**; DistilBERT (`distilbert-base-uncased`) fine-tuned with mean pooling.

### Differences from the paper and its code (so you know what you are comparing)
- The authors' training scripts clip the token weight with `min(1 + w, 5)` and use lexicons that include product
  names and CVE-specific phrases (e.g. vendor product strings), which may come from the evaluation data. ThreatWeigh
  follows the paper's equations (11/2/1) and builds its lexicons only from generic CVSS wording, filtered on the
  training split.
- The dataset zip under `combine/Dataset/` in the authors' repository is **byte-identical** (same SHA-256) to the NVD
  zip, so it is not a separate dataset. **ThreatWeigh uses only the NVD dataset and makes no claim about a "Combine" experiment.**
- The reported numbers come from a reduced CPU compute budget (see below), not the paper's full 15-epoch training.

## Setup

```bash
cd threatweigh
python -m venv .venv && source .venv/bin/activate
# CPU-only PyTorch first (~200 MB; the default PyPI wheel pulls several GB of CUDA libraries)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

A GPU is detected and used automatically (`src/utils.get_device`). Everything also runs on a CPU.

## Running each stage

Everything is driven by a single **`config.yaml`**. It has three compute profiles:

| profile | train / val / test rows | epochs | use |
|---|---|---|---|
| `full` | 109,991 / 13,750 / 13,750 | 15 (early stopping) | the paper's setting; needs a GPU |
| `budget` | 2,000 / 500 / 2,000 (fixed, stratified, same rows for every model) | 3 | **the numbers in this README** (4-core CPU) |
| `quick` | 300 / 100 / 200 | 2 | smoke test (`--quick`), a few minutes |

Run everything: `bash run_all.sh budget`. The script is resumable, and it accepts `quick` or `full` too. Or run stage by stage:

```bash
python -m src.data                      # 1. clone + unzip + clean + dedup + 80/10/10 split + EDA
python -m pytest                        # 2. unit tests (CVSS, 11/2/1 weights, pooling, split leakage)
python -m src.lexicon                   # 3a. expert lexicons from the training split + coverage
python -m src.keybert_cache             # 3b. KeyBERT keyword cache
python -m src.baselines --profile budget             # 4. TF-IDF + LogReg
python -m src.train --exp sbert_mean  --profile budget --save-model   # 4. plain SBERT
python -m src.train --exp distilbert  --profile budget --save-model   # 4. DistilBERT
python -m src.train --exp threatweigh --profile budget --save-model   # 5. ThreatWeigh
python -m src.train --exp threatweigh --profile budget --seeds 42 43 44   # 6. seeds
python -m src.train --exp tw_alpha    --profile budget                    # 6. ablation (a); also tw_expert, tw_keybert
python -m src.train --exp threatweigh_w5 --profile budget                 # 6. weight sensitivity (w2, w5, w20)
python -m src.train --exp threatweigh --profile budget --temporal         # 6. time-aware split
python -m src.evaluate --profile budget # tables + charts -> results/
python -m src.explain  --profile budget # 7. Integrated Gradients + quantitative checks
python -m src.train --exp threatweigh --quick --save-model   # any stage: CPU smoke test
```

Use the model:

```bash
python predict.py --explain "A remote unauthenticated attacker can execute arbitrary code via a crafted HTTP request."
streamlit run app.py
```

If cloning the authors' repository fails, `python fetch_nvd.py` rebuilds `final.csv` from the NVD API 2.0
(set `NVD_API_KEY` for a higher rate limit).

## Results

<!-- RESULTS -->
*(filled in after the budget runs finish)*
<!-- /RESULTS -->

## Explainability

<!-- EXPLAIN -->
*(filled in after `src/explain.py` finishes)*
<!-- /EXPLAIN -->

**Attribution is descriptive, not proof of causal reasoning.** Integrated Gradients measures how sensitive the trained
network's output is to each input token, relative to a chosen reference input. It does not show that the
model "understands" the vulnerability, and different reference baselines give different explanations.

## Limitations

- **Compute budget.** All reported numbers come from 2,000 training CVEs and at most 3 epochs on a CPU; the paper trains
  on about 110k CVEs for 15 epochs on GPUs. Absolute accuracies are therefore much lower than the paper's, and
  differences between neural variants at this size are small compared with seed noise. The `full` profile runs the
  paper's setting if you have the hardware.
- **Test subsample.** Neural models are evaluated on a fixed stratified 2,000-row subsample of the 13,750-row test split.
- **Single dataset.** Only NVD descriptions with NVD-assigned labels. Those labels are themselves noisy and change over time.
- **Lexicons are hand-written.** They reflect one reading of the CVSS specification. Cues are filtered on training data,
  so they also absorb NVD labelling habits.
- **Truncation.** Descriptions longer than 128 tokens (about 11 %) are cut off.
- **Temporal split.** The year comes from the CVE ID (the year the ID was reserved), which is only a proxy for the publication date.
- **No causal claims.** Observed differences between models are associations under this setup, not proof that symbolic
  guidance causes better predictions.
- **Not a production scorer.** Use it to triage or as a second opinion, never as a replacement for analyst review.

## Project structure

```
threatweigh/
  README.md  requirements.txt  config.yaml  run_all.sh  TODO.md
  app.py              Streamlit UI (Predict / Model results / How it works)
  predict.py          CLI + ThreatWeighPredictor used by the app
  fetch_nvd.py        fallback downloader (NVD API 2.0)
  src/
    utils.py          config, seeds, device, paths
    data.py           clone/unzip, clean, dedup, stratified + temporal splits, EDA
    cvss.py           official CVSS v3.1 calculator
    lexicon.py        decision trees -> filtered, frozen expert lexicons
    keybert_cache.py  KeyBERT keywords (cached)
    model.py          symbolic weights, fused pooling, ThreatWeighModel, experiment variants
    train.py          per-metric training with early stopping
    baselines.py      TF-IDF + LogReg; runs plain SBERT / DistilBERT through train.py
    evaluate.py       tables, bootstrap CIs, McNemar, charts
    explain.py        Captum Integrated Gradients + noise rate / infidelity / sensitivity / deletion AUC
  tests/              pytest: CVSS, weighting & pooling, data splits
  lexicons/           frozen expert lexicons (one JSON per metric, with kept/dropped cues and their statistics)
  data/  models/      created by the pipeline (git-ignored)
  results/            tables (.csv/.md), figures, predictions, explanations
```

## Credits and licences

- **Paper:** S. L. Mirtaheri et al., "WSymBERT: a weighted neuro-symbolic model for interpretable CVSS prediction",
  *International Journal of Information Security*, 2026. ThreatWeigh re-implements its method in simplified form.
- **Code:** the authors' repository <https://github.com/ShahbazianR/WSymBert> is MIT-licensed. ThreatWeigh does not
  copy its code. It is used here to obtain the dataset.
- **Data:** the NVD dataset (`Nvd/Dataset/final.zip`, 137,521 CVEs) is published by the WSymBERT authors under
  **CC-BY-4.0**. The descriptions and CVSS vectors come from the U.S. National Vulnerability Database (NIST).
  *This product uses data from the NVD API but is not endorsed or certified by the NVD.*
- **CVSS:** FIRST, *Common Vulnerability Scoring System v3.1: Specification Document*.
- **Libraries:** Hugging Face Transformers, Sentence-Transformers, KeyBERT (M. Grootendorst), Captum, scikit-learn, Streamlit.
