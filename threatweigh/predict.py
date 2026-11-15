"""
ThreatWeigh command-line predictor.

    python predict.py "A remote unauthenticated attacker can execute arbitrary code via a crafted HTTP request."
    python predict.py --explain "..."        # also show the top attributed words per metric
    python predict.py --json "..."           # machine-readable output

It loads the 8 metric-specific ThreatWeigh models (models/threatweigh_<profile>/),
predicts AV, AC, PR, UI, S, C, I, A, builds the CVSS v3.1 vector string, and
computes the Base Score with the official formula (src/cvss.py).

The ThreatWeighPredictor class below is also used by the Streamlit app (app.py).
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from src.cvss import base_score, severity, vector_string
from src.explain import integrated_gradients, load_trained, prepare, top_words, word_attributions
from src.keybert_cache import extract_keywords
from src.lexicon import expert_cues, find_spans
from src.utils import load_config, metric_names, path

METRIC_NAMES = {
    "av": "Attack Vector", "ac": "Attack Complexity", "pr": "Privileges Required", "ui": "User Interaction",
    "s": "Scope", "c": "Confidentiality", "i": "Integrity", "a": "Availability",
}
LABEL_NAMES = {
    "av": {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
    "ac": {"L": "Low", "H": "High"},
    "pr": {"N": "None", "L": "Low", "H": "High"},
    "ui": {"N": "None", "R": "Required"},
    "s": {"U": "Unchanged", "C": "Changed"},
    "c": {"N": "None", "L": "Low", "H": "High"},
    "i": {"N": "None", "L": "Low", "H": "High"},
    "a": {"N": "None", "L": "Low", "H": "High"},
}


def find_model_run(cfg: dict, model: str = "threatweigh") -> str:
    """Use the best-trained models available: full > budget > quick."""
    for profile in ("full", "budget", "quick"):
        run = f"{model}_{profile}"
        if all((path(cfg["paths"]["models_dir"]) / run / f"{m}.pt").exists() for m in metric_names(cfg)):
            return run
    raise SystemExit(f"No trained {model} models found in models/. "
                     f"Train them first, e.g.  python -m src.train --exp {model} --quick --save-model")


class ThreatWeighPredictor:
    def __init__(self, run_name: str | None = None):
        self.cfg = load_config()
        torch.set_num_threads(self.cfg["train"]["num_threads"])
        self.run_name = run_name or find_model_run(self.cfg)
        self.models = {m: load_trained(self.run_name, m, self.cfg) for m in metric_names(self.cfg)}
        self.cues = {m: expert_cues(m, self.cfg) for m in metric_names(self.cfg)}

    def predict(self, text: str, explain: bool = False, top_k: int = 10) -> dict:
        text = " ".join(text.split())
        keywords = extract_keywords([text], self.cfg)[0]       # K_mdl for this description
        result = {"text": text, "keywords": keywords, "model_run": self.run_name, "metrics": {}}
        for m, (model, tok, exp, labels) in self.models.items():
            ids, mask, w, offsets = prepare(text, m, exp, tok, self.cfg, keywords)
            with torch.no_grad():
                probs = torch.softmax(model(ids, mask, w), -1)[0]
            k = int(probs.argmax())
            entry = {"label": labels[k], "label_name": LABEL_NAMES[m][labels[k]],
                     "confidence": float(probs[k]),
                     "probabilities": {lab: float(p) for lab, p in zip(labels, probs)}}
            if explain:
                attr, _ = integrated_gradients(model, tok, ids, mask, w, k, "pad", self.cfg["explain"]["ig_steps"])
                words = top_words(word_attributions(text, offsets, attr.sum(-1)[0].detach().numpy()), top_k)
                lex_spans = find_spans(text, self.cues[m])
                entry["top_words"] = [
                    {"word": wd, "start": s, "end": e, "attribution": a,
                     "in_lexicon": any(s < le and ls < e for ls, le, _ in lex_spans)}
                    for wd, s, e, a in words]
            result["metrics"][m] = entry
        labels = {m: v["label"] for m, v in result["metrics"].items()}
        result["vector"] = vector_string(labels)
        result["base_score"] = base_score(labels)
        result["severity"] = severity(result["base_score"])
        return result


def main():
    ap = argparse.ArgumentParser(description="Predict CVSS v3.1 Base metrics from a vulnerability description.")
    ap.add_argument("description", nargs="*", help="the vulnerability text (or pipe it on stdin)")
    ap.add_argument("--explain", action="store_true", help="show the top attributed words per metric")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--run", help="model folder under models/, e.g. threatweigh_budget")
    args = ap.parse_args()
    text = " ".join(args.description) or sys.stdin.read()
    if not text.strip():
        ap.error("please give a description")

    res = ThreatWeighPredictor(args.run).predict(text, explain=args.explain)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print(f"\nModels: {res['model_run']}   KeyBERT keywords: {', '.join(res['keywords'])}\n")
    for m, v in res["metrics"].items():
        line = f"  {METRIC_NAMES[m]:<20} {m.upper():>2} = {v['label']} ({v['label_name']:<9}) confidence {v['confidence']:.2f}"
        if args.explain:
            line += "   top words: " + ", ".join(f"{'*' if t['in_lexicon'] else ''}{t['word']}"
                                                   for t in v["top_words"][:6])
        print(line)
    print(f"\n  Vector:     {res['vector']}")
    print(f"  Base Score: {res['base_score']}  ({res['severity']})")
    if args.explain:
        print("\n  (* = word is in the expert lexicon. Attributions are descriptive, not causal.)")


if __name__ == "__main__":
    main()
