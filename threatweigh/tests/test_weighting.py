"""Tests for the symbolic weights (11 / 2 / 1 logic) and the fused pooling."""
import torch

from src.model import fused_pooling, masked_softmax, symbolic_weights


def _offsets(text):
    """Whitespace 'tokenizer' with character offsets, plus a [CLS]-like (0, 0) token."""
    offs, pos = [(0, 0)], 0
    for word in text.split():
        start = text.index(word, pos)
        offs.append((start, start + len(word)))
        pos = start + len(word)
    return offs


TEXT = "requires local access and the victim must click"


def test_11_2_1_logic():
    cues = ["requires local access", "click"]       # K_exp (multi-word phrase + a word)
    kws = ["local", "victim"]                        # K_mdl
    w = symbolic_weights(_offsets(TEXT), TEXT, cues, kws, both_weight=10, one_weight=1)
    words = ["[CLS]"] + TEXT.split()
    got = dict(zip(words, w))
    assert got["[CLS]"] == 1          # special token
    assert got["local"] == 11         # in K_exp (inside the phrase) AND K_mdl
    assert got["requires"] == 2       # only in K_exp (part of the phrase)
    assert got["access"] == 2         # only in K_exp (part of the phrase)
    assert got["victim"] == 2         # only in K_mdl
    assert got["click"] == 2          # only in K_exp
    assert got["the"] == 1            # in neither
    assert set(w) <= {1, 2, 11}


def test_phrase_maps_to_subword_tokens():
    # "unauthenticated" split into sub-words: both pieces belong to the cue.
    text = "unauthenticated attackers"
    offsets = [(0, 0), (0, 5), (5, 15), (16, 25)]   # "unaut" "henticated" "attackers"
    w = symbolic_weights(offsets, text, ["unauthenticated"], None)
    assert w == [1, 2, 2, 1]


def test_sources_can_be_switched_off():
    w = symbolic_weights(_offsets(TEXT), TEXT, None, None)
    assert all(x == 1 for x in w)                    # ablation (a): plain attention


def test_sensitivity_weight():
    w = symbolic_weights(_offsets(TEXT), TEXT, ["local"], ["local"], both_weight=20)
    assert max(w) == 21


def test_pooling_weights_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    B, T, H = 3, 7, 5
    h = torch.randn(B, T, H)
    mask = torch.ones(B, T)
    mask[1, 4:] = 0                                  # second example has padding
    alpha = masked_softmax(torch.randn(B, T), mask)
    w = torch.tensor([[1, 2, 11, 1, 1, 2, 1]] * B, dtype=torch.float)
    z, beta = fused_pooling(h, alpha, w, mask)
    assert torch.allclose(alpha.sum(-1), torch.ones(B), atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.ones(B), atol=1e-6)
    assert torch.all(beta[1, 4:] == 0)               # padding gets no weight
    assert torch.all(alpha[1, 4:] == 0)
    # z is a convex combination of the token vectors
    assert torch.allclose(z, torch.einsum("bt,bth->bh", beta, h))


def test_higher_symbolic_weight_raises_share():
    alpha = torch.full((1, 4), 0.25)
    mask = torch.ones(1, 4)
    _, beta = fused_pooling(torch.zeros(1, 4, 2), alpha, torch.tensor([[1., 11., 1., 1.]]), mask)
    assert abs(beta[0, 1].item() - 11 / 14) < 1e-6
