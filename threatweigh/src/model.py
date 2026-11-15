"""
The ThreatWeigh model (and the plain neural baselines, which reuse it).

For ONE metric (e.g. Attack Vector) the model is:

    tokens  --Sentence-BERT encoder-->  h_1 ... h_n          (one vector per sub-word token)

    learned pooling scores   s_i     = W_pool h_i + b
    learned pooling weights  alpha_i = softmax_i(s_i)        (padding tokens masked out)

    symbolic weight          w_i = 10 if token i is in K_exp AND K_mdl
                                   1  if it is in exactly one of them
                                   0  otherwise
                             w_i = w_i + 1                    -> 11, 2 or 1

    fused pooling            beta_i = alpha_i * w_i / (sum_j alpha_j * w_j + eps)
                             z      = sum_i beta_i * h_i

    classifier               p(label) = softmax(W_cls z + b_cls)   trained with cross-entropy

K_exp = expert lexicon cues for this metric (src/lexicon.py)
K_mdl = KeyBERT keywords of this description (src/keybert_cache.py)

Pooling options (used by the baselines / ablations):
    "mean"      plain mean of the token vectors (plain SBERT / DistilBERT baselines)
    "weighted"  the fused pooling above; with all w_i = 1 it is just learned attention (ablation a)
"""
from __future__ import annotations

import re
from functools import lru_cache

import torch
from torch import nn

from src.lexicon import find_spans

# --------------------------------------------------------------------------- #
# Experiment variants. Every experiment in the report is one of these.
# --------------------------------------------------------------------------- #
EXPERIMENTS: dict[str, dict] = {
    # ---- baselines ----
    "sbert_mean":  {"encoder": "sbert",      "pooling": "mean",     "use_exp": False, "use_kw": False},
    "distilbert":  {"encoder": "distilbert", "pooling": "mean",     "use_exp": False, "use_kw": False},
    # ---- ablation: what each knowledge source adds ----
    "tw_alpha":    {"encoder": "sbert", "pooling": "weighted", "use_exp": False, "use_kw": False},  # (a)
    "tw_expert":   {"encoder": "sbert", "pooling": "weighted", "use_exp": True,  "use_kw": False},  # (b)
    "tw_keybert":  {"encoder": "sbert", "pooling": "weighted", "use_exp": False, "use_kw": True},   # (c)
    "threatweigh": {"encoder": "sbert", "pooling": "weighted", "use_exp": True,  "use_kw": True},   # (d) full
    # ---- sensitivity of the "both" weight (10 in the paper) ----
    "threatweigh_w2":  {"encoder": "sbert", "pooling": "weighted", "use_exp": True, "use_kw": True, "both_weight": 2},
    "threatweigh_w5":  {"encoder": "sbert", "pooling": "weighted", "use_exp": True, "use_kw": True, "both_weight": 5},
    "threatweigh_w20": {"encoder": "sbert", "pooling": "weighted", "use_exp": True, "use_kw": True, "both_weight": 20},
}


def resolve_experiment(name: str, cfg: dict) -> dict:
    """Fill in defaults (checkpoint name, weights) for an experiment variant."""
    exp = dict(EXPERIMENTS[name])
    exp["name"] = name
    exp["checkpoint"] = cfg["model"]["distilbert"] if exp["encoder"] == "distilbert" else cfg["model"]["encoder"]
    exp.setdefault("both_weight", cfg["model"]["both_weight"])
    exp.setdefault("one_weight", cfg["model"]["one_weight"])
    return exp


# --------------------------------------------------------------------------- #
# Symbolic weights (one number per token)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8192)
def _word_regex(word: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)


def keyword_spans(text: str, keywords: list[str]) -> list[tuple[int, int]]:
    """Character spans of every occurrence of every KeyBERT keyword."""
    return [(m.start(), m.end()) for kw in keywords for m in _word_regex(kw).finditer(text)]


def _overlaps(tok_start: int, tok_end: int, spans) -> bool:
    return any(tok_start < e and s < tok_end for s, e, *_ in spans)


def symbolic_weights(
    offsets: list[tuple[int, int]],
    text: str,
    expert_cues: list[str] | None,
    keywords: list[str] | None,
    both_weight: float = 10.0,
    one_weight: float = 1.0,
) -> list[float]:
    """
    Compute w_i for every token.

    `offsets` are the (start, end) character positions of each token, as
    returned by a Hugging Face "fast" tokenizer (return_offsets_mapping=True).
    Special tokens ([CLS], [SEP], padding) have offset (0, 0) -> weight 1.

    A multi-word lexicon phrase such as "requires local access" marks EVERY
    sub-word token that lies inside it, because we match by character spans.

    Pass expert_cues=None (or keywords=None) to switch that source off (ablations).
    """
    exp_spans = find_spans(text, expert_cues) if expert_cues else []
    kw_spans = keyword_spans(text, keywords) if keywords else []
    weights = []
    for start, end in offsets:
        if start == end:              # special / padding token
            weights.append(1.0)
            continue
        in_exp = _overlaps(start, end, exp_spans)
        in_kw = _overlaps(start, end, kw_spans)
        if in_exp and in_kw:
            w = both_weight
        elif in_exp or in_kw:
            w = one_weight
        else:
            w = 0.0
        weights.append(w + 1.0)        # final values: both+1 (11), 2 or 1
    return weights


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over tokens where mask==1; padding (mask==0) gets probability 0."""
    scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


def fused_pooling(h: torch.Tensor, alpha: torch.Tensor, w: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    beta_i = alpha_i * w_i / (sum_j alpha_j * w_j + eps),   z = sum_i beta_i h_i
    Shapes: h [B, T, H], alpha/w/mask [B, T]. Returns (z [B, H], beta [B, T]).
    """
    aw = alpha * w * mask
    beta = aw / (aw.sum(dim=-1, keepdim=True) + eps)
    z = torch.einsum("bt,bth->bh", beta, h)
    return z, beta


def mean_pooling(h: torch.Tensor, mask: torch.Tensor):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


class ThreatWeighModel(nn.Module):
    """Encoder + (mean | fused weighted) pooling + linear classifier, for ONE metric."""

    def __init__(self, checkpoint: str, n_labels: int, pooling: str = "weighted", eps: float = 1e-8):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.eps = eps
        self.pool_scorer = nn.Linear(hidden, 1)          # s_i = W_pool h_i + b
        self.classifier = nn.Linear(hidden, n_labels)   # softmax is inside the cross-entropy loss

    def embeddings(self):
        """The word-embedding layer (used by Captum Integrated Gradients)."""
        return self.encoder.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, sym_weights=None, inputs_embeds=None,
                return_pooling: bool = False):
        out = self.encoder(input_ids=input_ids if inputs_embeds is None else None,
                           inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state                        # [B, T, H]
        mask = attention_mask.to(h.dtype)

        if self.pooling == "mean":
            z = mean_pooling(h, mask)
            beta = mask / mask.sum(-1, keepdim=True)
            alpha = beta
        else:
            alpha = masked_softmax(self.pool_scorer(h).squeeze(-1), mask)
            w = torch.ones_like(alpha) if sym_weights is None else sym_weights.to(h.dtype)
            z, beta = fused_pooling(h, alpha, w, mask, self.eps)

        logits = self.classifier(z)
        if return_pooling:
            return logits, {"alpha": alpha, "beta": beta}
        return logits
