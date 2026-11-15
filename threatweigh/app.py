"""
ThreatWeigh - Streamlit app.

    streamlit run app.py

Tabs
  1. Predict       - paste a CVE description, get the 8 CVSS metrics, the vector string,
                     the Base Score + severity, and the words that mattered most.
  2. Model results - comparison tables and charts, ablation, seed variance, explainability.
  3. How it works  - plain-language explanation and the pipeline diagram.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="ThreatWeigh", page_icon="⚖️", layout="wide")

EXAMPLES = {
    "Remote code execution": "A remote unauthenticated attacker can execute arbitrary code on the server by "
                             "sending a crafted HTTP request to the /api/upload endpoint.",
    "Stored XSS": "A stored cross-site scripting (XSS) vulnerability in the comment form allows remote attackers "
                  "to inject arbitrary web script or HTML, which is executed when an administrator views the page.",
    "Local privilege escalation": "A race condition in the kernel driver allows local users to gain root "
                                  "privileges via a crafted application.",
    "Physical access": "An attacker with physical access to the device can bypass the lock screen via the USB "
                       "debug port and read stored user data.",
    "Bluetooth DoS": "A NULL pointer dereference in the Bluetooth stack allows an attacker within radio range to "
                     "cause a denial of service (crash) by sending malformed packets.",
}
SEVERITY_COLOURS = {"None": "#6b7280", "Low": "#16a34a", "Medium": "#ca8a04", "High": "#ea580c",
                    "Critical": "#dc2626"}


@st.cache_resource(show_spinner="Loading the 8 ThreatWeigh models...")
def get_predictor():
    from predict import ThreatWeighPredictor

    return ThreatWeighPredictor()


def highlight(text: str, top_words: list[dict]) -> str:
    """HTML for the description with the top attributed words highlighted.
    Orange = pushed towards the predicted label, blue = pushed against it;
    bold + underline + ★ = the word is in the expert lexicon."""
    if not top_words:
        return html.escape(text)
    peak = max(abs(t["attribution"]) for t in top_words) or 1.0
    out, pos = [], 0
    for t in sorted(top_words, key=lambda t: t["start"]):
        if t["start"] < pos:
            continue
        out.append(html.escape(text[pos:t["start"]]))
        strength = 0.15 + 0.6 * abs(t["attribution"]) / peak
        colour = f"rgba(234,88,12,{strength:.2f})" if t["attribution"] > 0 else f"rgba(37,99,235,{strength:.2f})"
        word = html.escape(text[t["start"]:t["end"]])
        style = f"background:{colour};border-radius:4px;padding:1px 3px;"
        if t["in_lexicon"]:
            style += "font-weight:700;text-decoration:underline;"
            word = "★" + word
        out.append(f'<span style="{style}" title="attribution {t["attribution"]:+.3f}">{word}</span>')
        pos = t["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- #
st.title("⚖️ ThreatWeigh")
st.caption("Interpretable CVSS v3.1 prediction from vulnerability text — a student-friendly re-implementation "
           "of WSymBERT (Mirtaheri et al., 2026). Research prototype: not a substitute for expert scoring.")

tab_predict, tab_results, tab_how = st.tabs(["🔮 Predict", "📊 Model results", "💡 How it works"])

# ---------------------------------------------------------------- Predict --- #
with tab_predict:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["text"] = text
    text = st.text_area("Vulnerability description", key="text", height=120,
                        placeholder="Paste a CVE description here...")
    explain = st.checkbox("Explain the prediction (Integrated Gradients, a few seconds)", value=True)

    if st.button("Predict", type="primary") and text.strip():
        try:
            pred = get_predictor()
        except SystemExit as e:
            st.error(str(e))
            st.stop()
        with st.spinner("Predicting..."):
            res = pred.predict(text, explain=explain)
        from predict import LABEL_NAMES, METRIC_NAMES

        c1, c2, c3 = st.columns([1, 1, 2])
        colour = SEVERITY_COLOURS[res["severity"]]
        c1.markdown(f"<div style='font-size:0.9rem;opacity:.7'>Base Score</div>"
                    f"<div style='font-size:3rem;font-weight:700;color:{colour}'>{res['base_score']}</div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div style='font-size:0.9rem;opacity:.7'>Severity</div>"
                    f"<div style='font-size:2rem;font-weight:700;color:{colour};margin-top:.6rem'>"
                    f"{res['severity']}</div>", unsafe_allow_html=True)
        c3.markdown("<div style='font-size:0.9rem;opacity:.7'>CVSS vector</div>", unsafe_allow_html=True)
        c3.code(res["vector"], language=None)
        c3.caption("KeyBERT keywords: " + ", ".join(res["keywords"]))

        table = pd.DataFrame([{"Metric": f"{METRIC_NAMES[m]} ({m.upper()})",
                               "Prediction": f"{v['label']} — {v['label_name']}",
                               "Confidence": v["confidence"]} for m, v in res["metrics"].items()])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="%.2f")})
        if explain:
            st.subheader("Which words mattered?")
            st.caption("Orange = pushed towards the predicted label, blue = pushed against it, darker = stronger. "
                       "**★ bold underlined** = in ThreatWeigh's expert lexicon for that metric. "
                       "Attributions describe the model's sensitivity; they are not proof of causal reasoning.")
            pick = st.radio("Metric", list(res["metrics"]), horizontal=True,
                            format_func=lambda m: f"{m.upper()} = {res['metrics'][m]['label']}")
            v = res["metrics"][pick]
            st.markdown(f"<div style='line-height:2;font-size:1.05rem'>{highlight(res['text'], v['top_words'])}</div>",
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"word": t["word"], "attribution": round(t["attribution"], 4),
                                        "expert lexicon": "★" if t["in_lexicon"] else ""}
                                       for t in v["top_words"]]), hide_index=True)
        st.caption(f"Models: `models/{res['model_run']}`")

# ---------------------------------------------------------- Model results --- #
with tab_results:
    def show(name: str, title: str, note: str = ""):
        f = RESULTS / f"{name}.csv"
        if f.exists():
            st.subheader(title)
            if note:
                st.caption(note)
            st.dataframe(pd.read_csv(f), hide_index=True, use_container_width=True)
        else:
            st.info(f"{title}: not computed yet (results/{name}.csv).")

    st.info("All numbers come from the CPU **budget** profile: every model is trained on the same fixed 2,000 "
            "training CVEs for at most 3 epochs and tested on the same 2,000 test CVEs. They are NOT the "
            "paper's numbers (the paper reports ≈96.2% mean accuracy on the full NVD data, for reference).")
    show("main_comparison", "Main comparison — test accuracy (%)",
         "mean ± std over seeds where several seeds were run. Score MSE/MAE: Base Score computed from the 8 "
         "predicted labels vs the true NVD score.")
    for fig, cap in [("main_comparison.png", "Accuracy per metric"), ("ablation.png", "Ablation")]:
        if (RESULTS / "figures" / fig).exists():
            st.image(str(RESULTS / "figures" / fig), caption=cap)
    show("ablation", "Ablation — what each knowledge source adds")
    show("significance", "ThreatWeigh vs plain SBERT — paired bootstrap 95% CI and McNemar test",
         "acc_diff_pp > 0 means ThreatWeigh was more accurate. A CI that includes 0 means no reliable difference.")
    show("sensitivity", "Sensitivity of the 'both' weight (paper: 10)")
    show("temporal", "Time-aware split (train ≤ 2021, test ≥ 2023)")
    show("lexicon_coverage", "Expert lexicon coverage")
    f = RESULTS / "explain" / "quantitative_summary.csv"
    if f.exists():
        st.subheader("Explanation quality (Integrated Gradients)")
        st.caption("noise_rate: % of top-10 words that are stop-words, numbers/versions or ≤2 characters (lower "
                   "is cleaner). deletion_auc: lower = deleting the top-attributed tokens first hurts the "
                   "prediction faster (compare with the random order).")
        st.dataframe(pd.read_csv(f), hide_index=True, use_container_width=True)
    show("per_metric_summary", "Per-metric accuracy / macro precision / recall / F1 (%)")

# ------------------------------------------------------------ How it works --- #
with tab_how:
    st.markdown("""
**The problem.** Every new software vulnerability (CVE) gets a CVSS score that tells defenders how bad it is.
Scoring is done by hand from the text description, which is slow. ThreatWeigh reads the description and
predicts the 8 CVSS v3.1 *Base metrics*, then computes the 0–10 score with the official formula.

**The idea.** A security analyst does not weigh all words equally: phrases like *"remote attackers"*,
*"physical access"* or *"administrator"* decide the answer. ThreatWeigh gives the model that hint:

1. **Read** – a small Sentence-BERT model turns every word piece into a vector.
2. **Learn what to look at** – a learned score per word piece (α) says how much it should count.
3. **Add expert knowledge** – two lists of important words boost α:
   - **K_exp**, an *expert lexicon* written from the CVSS specification, as yes/no questions per metric
     (e.g. *"Does the attacker need physical access?"* → `physical access`, `usb`, …), filtered on training data only;
   - **K_mdl**, the 5 *KeyBERT* keywords of the description.
   A word in both lists counts ×11, in one list ×2, otherwise ×1.
4. **Decide** – the weighted average vector goes to a small classifier. There is one separate model per metric.
5. **Score** – the 8 labels go into the official CVSS v3.1 formula.
6. **Explain** – Integrated Gradients shows which words pushed the prediction.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
  text [label="CVE description"]; enc [label="Sentence-BERT\\nencoder"]; alpha [label="learned pooling\\nweights α"];
  lex [label="expert lexicon\\nK_exp", fillcolor="#fef3c7"]; kb [label="KeyBERT keywords\\nK_mdl", fillcolor="#fef3c7"];
  w [label="symbolic weight\\n11 / 2 / 1", fillcolor="#fef3c7"]; pool [label="fused pooling\\nβ ∝ α·w"];
  clf [label="classifier\\n(×8 metrics)"]; cvss [label="CVSS v3.1\\nformula"]; out [label="vector + score\\n+ severity", fillcolor="#dcfce7"];
  ig [label="Integrated\\nGradients", fillcolor="#fce7f3"];
  text -> enc -> alpha -> pool; text -> lex -> w; text -> kb -> w; w -> pool; pool -> clf -> cvss -> out; clf -> ig;
}
""")
    st.markdown("**The equations**")
    st.latex(r"s_i = W_{pool} h_i + b,\qquad \alpha_i = \mathrm{softmax}_i(s_i)")
    st.latex(r"w_i = 1 + \begin{cases}10 & t_i \in K_{exp}\cap K_{mdl}\\ 1 & t_i \text{ in exactly one}\\ 0 & \text{otherwise}\end{cases}")
    st.latex(r"\beta_i = \frac{\alpha_i w_i}{\sum_j \alpha_j w_j + \epsilon},\qquad z=\sum_i \beta_i h_i,\qquad p = \mathrm{softmax}(W z + b)")
    st.markdown("""
**Honest limits.** Results come from a small CPU budget, not the paper's full training. Attributions show what
the model is *sensitive* to, not *why* it is right. Descriptions are often vague, and NVD labels themselves contain
noise. ThreatWeigh is a simplified re-implementation of WSymBERT, not the authors' code.
""")
