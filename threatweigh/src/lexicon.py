"""
Stage 3a - EXPERT LEXICONS (K_exp): security knowledge written down as cue words.

Run:  python -m src.lexicon

The idea (from the paper): a security analyst scores a CVE by asking a short
series of questions taken from the FIRST CVSS v3.1 specification, e.g.
    "Can the attacker exploit it remotely over a network?"   -> AV = Network
    "Does the attacker need physical access to the device?"  -> AV = Physical
Each question has typical wordings in vulnerability descriptions ("remote
attackers", "physically proximate", ...). Below, every metric has a small
decision tree: an ordered list of questions, each pointing to one label and a
list of CANDIDATE cue phrases.

How the final lexicon is built (only the TRAINING split is ever used):
  1. For every candidate cue, count in how many training descriptions it occurs
     (support) and how often the label it points to is the true label (precision).
  2. Keep the cue if support >= min_support AND
        (precision >= min_precision  OR  precision >= min_lift * base rate of the label).
     The "lift" rule lets rare labels (e.g. AV=Physical, 1% of CVEs) keep useful cues.
  3. Write the kept cues to lexicons/<metric>.json and FREEZE them (never
     re-built from validation or test data).

Then the coverage (% of descriptions with at least one lexicon match) is
reported on the test split and on 2,000 random CVEs (the paper reports 83.6%).
No product names or CVE-specific strings are used - only generic CVSS wording.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from src.utils import load_config, metric_names, path

# --------------------------------------------------------------------------- #
# Question-style decision trees, one per metric. Order matters only for reading.
# --------------------------------------------------------------------------- #
DECISION_TREES: dict[str, list[dict]] = {
    "av": [
        {"question": "Does the attacker need to physically touch or be next to the device?",
         "label": "P",
         "cues": ["physical access", "physically proximate", "physical proximity", "physical attacker",
                  "physically present", "usb", "jtag", "uart", "serial console", "unlocked device",
                  "lock screen", "lockscreen", "evil maid", "boot loader", "bootloader", "debug port",
                  "physical presence", "attached to the device"]},
        {"question": "Is the attack limited to the same local/shared network segment (Bluetooth, Wi-Fi, LAN)?",
         "label": "A",
         "cues": ["adjacent network", "adjacent attacker", "adjacent", "same network segment", "local network",
                  "same subnet", "bluetooth", "wi-fi", "wifi", "wlan", "802.11", "wireless", "nfc", "zigbee",
                  "same local area network", "layer 2", "broadcast domain", "within range"]},
        {"question": "Does the attacker need a local account, a local app, or to get a user to open a local file?",
         "label": "L",
         "cues": ["local users", "local user", "local attacker", "locally", "local access", "local authenticated",
                  "local privilege escalation", "escalate privileges", "elevation of privilege",
                  "malicious application", "crafted application", "malicious app", "kernel", "driver",
                  "symlink attack", "setuid", "open a crafted", "crafted file", "logged-in user",
                  "local account", "the local system", "via a local"]},
        {"question": "Can it be exploited remotely, across the network or the Internet?",
         "label": "N",
         "cues": ["remote attackers", "remote attacker", "remotely", "unauthenticated remote", "over the network",
                  "via the network", "network access", "crafted http request", "http request", "crafted request",
                  "crafted url", "web page", "web site", "website", "sql injection", "cross-site scripting",
                  "cross-site request forgery", "csrf", "xss", "server-side request forgery", "ssrf",
                  "parameter", "api", "endpoint", "crafted packet", "packets", "network packet", "web interface"]},
    ],
    "ac": [
        {"question": "Does success depend on conditions outside the attacker's control (race, timing, MITM, config)?",
         "label": "H",
         "cues": ["race condition", "time-of-check", "toctou", "man-in-the-middle", "mitm", "timing attack",
                  "side-channel", "side channel", "padding oracle", "brute force", "non-default configuration",
                  "specific configuration", "certain configurations", "under certain conditions", "intercept",
                  "downgrade", "cache poisoning", "dns rebinding", "predict", "spoof", "cryptographic",
                  "privileged network position", "high complexity", "difficult to exploit"]},
        {"question": "Can the attacker succeed repeatedly at will, with no special conditions?",
         "label": "L",
         "cues": ["unauthenticated", "without authentication", "crafted request", "crafted http request",
                  "sql injection", "cross-site scripting", "default configuration", "easily", "trivially"]},
    ],
    "pr": [
        {"question": "Does the attacker need administrator / root / high privileges?",
         "label": "H",
         "cues": ["administrator", "administrators", "admin", "administrative", "high privileges",
                  "high-privileged", "highly privileged", "privileged user", "root privileges", "root access",
                  "admin privileges", "admin access", "superuser", "system privileges", "privileged account",
                  "admin panel", "administrative privileges", "admin users"]},
        {"question": "Does the attacker need an ordinary logged-in account (basic user rights)?",
         "label": "L",
         "cues": ["authenticated user", "authenticated users", "remote authenticated", "authenticated attacker",
                  "low-privileged", "low privileged", "low-privilege", "logged-in", "logged in", "local users",
                  "local user", "standard user", "registered user", "normal user", "regular user",
                  "subscriber", "contributor", "authenticated", "user account", "valid account"]},
        {"question": "Can the attacker do it without any account or login?",
         "label": "N",
         "cues": ["unauthenticated", "without authentication", "anonymous", "unauthorized", "no authentication",
                  "pre-auth", "missing authentication", "authentication bypass", "bypass authentication",
                  "without login", "without logging in", "remote attackers", "not authenticated"]},
    ],
    "ui": [
        {"question": "Must a victim do something (click, visit, open a file) for the attack to work?",
         "label": "R",
         "cues": ["user interaction", "convince a user", "trick a user", "tricking a user", "victim", "visit",
                  "visiting", "click", "clicking", "open a crafted", "opens a crafted", "malicious link",
                  "crafted link", "crafted web page", "crafted web site", "crafted document", "crafted file",
                  "cross-site scripting", "xss", "csrf", "cross-site request forgery", "social engineering",
                  "phishing", "persuade", "entice", "inject arbitrary web script", "user-assisted"]},
        {"question": "Can the attack succeed with no action from any user?",
         "label": "N",
         "cues": ["without user interaction", "no user interaction", "user interaction is not needed",
                  "unauthenticated", "sql injection", "crafted request", "crafted http request", "crafted packet",
                  "automatically", "denial of service", "remote code execution", "command injection"]},
    ],
    "s": [
        {"question": "Does the attack reach beyond the vulnerable component (browser of another user, host, other tenants)?",
         "label": "C",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "stored xss", "sandbox escape",
                  "escape the sandbox", "sandbox", "guest", "hypervisor", "virtual machine", "container escape",
                  "host operating system", "server-side request forgery", "ssrf", "trustzone", "secure world",
                  "other users", "other tenants", "html injection", "javascript"]},
        {"question": "Is the impact confined to the vulnerable component itself?",
         "label": "U",
         "cues": ["denial of service", "sql injection", "buffer overflow", "out-of-bounds", "null pointer",
                  "use-after-free", "execute arbitrary code", "command injection", "obtain sensitive information",
                  "crash", "memory corruption", "integer overflow"]},
    ],
    "c": [
        {"question": "Can the attacker read all / any sensitive data (total loss of confidentiality)?",
         "label": "H",
         "cues": ["execute arbitrary code", "arbitrary code execution", "remote code execution", "code execution",
                  "command injection", "execute arbitrary commands", "sql injection", "read arbitrary files",
                  "arbitrary file read", "gain privileges", "gain root", "privilege escalation", "take over",
                  "complete control", "full control", "credentials", "password", "private key",
                  "obtain sensitive information", "sensitive information", "buffer overflow", "use-after-free"]},
        {"question": "Can the attacker read only some limited information?",
         "label": "L",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "information disclosure",
                  "potentially sensitive information", "version information", "path disclosure",
                  "full path", "enumerate", "username enumeration", "limited information", "some information",
                  "html injection", "open redirect", "csrf"]},
        {"question": "Is no information disclosed at all?",
         "label": "N",
         "cues": ["denial of service", "crash", "consume", "cpu consumption", "memory consumption",
                  "resource consumption", "reboot", "hang", "infinite loop", "null pointer dereference",
                  "assertion failure", "deadlock", "spoof", "spoofing", "open redirect"]},
    ],
    "i": [
        {"question": "Can the attacker modify any data / files (total loss of integrity)?",
         "label": "H",
         "cues": ["execute arbitrary code", "arbitrary code execution", "remote code execution", "code execution",
                  "command injection", "execute arbitrary commands", "write arbitrary files", "arbitrary file write",
                  "overwrite", "upload arbitrary files", "arbitrary file upload", "unrestricted upload",
                  "gain privileges", "privilege escalation", "sql injection", "delete arbitrary", "modify",
                  "buffer overflow", "use-after-free", "memory corruption"]},
        {"question": "Can the attacker modify only some data, with limited consequences?",
         "label": "L",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "html injection", "spoof",
                  "spoofing", "open redirect", "redirect users", "csrf", "cross-site request forgery",
                  "content spoofing", "log injection", "arbitrary web script"]},
        {"question": "Is no data modified at all?",
         "label": "N",
         "cues": ["denial of service", "obtain sensitive information", "information disclosure",
                  "read arbitrary files", "crash", "cleartext", "enumerate", "disclose", "out-of-bounds read",
                  "information exposure", "consume", "infinite loop", "null pointer dereference"]},
    ],
    "a": [
        {"question": "Can the attacker fully deny access to the component (crash, hang, exhaust resources)?",
         "label": "H",
         "cues": ["denial of service", "crash", "dos", "consume", "resource consumption", "cpu consumption",
                  "memory consumption", "infinite loop", "null pointer dereference", "reboot", "hang",
                  "kernel panic", "assertion failure", "deadlock", "execute arbitrary code",
                  "remote code execution", "buffer overflow", "use-after-free", "memory corruption",
                  "out-of-bounds write", "stack overflow"]},
        {"question": "Is availability only reduced (slower, partial, temporary)?",
         "label": "L",
         "cues": ["degrade", "degradation", "partial denial of service", "slow", "temporarily",
                  "limited denial", "performance", "intermittent"]},
        {"question": "Is availability not affected at all?",
         "label": "N",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "obtain sensitive information",
                  "information disclosure", "csrf", "cross-site request forgery", "open redirect", "spoof",
                  "html injection", "information exposure", "sensitive information", "cleartext"]},
    ],
}


# --------------------------------------------------------------------------- #
# Phrase matching helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def _cue_regex(cue: str) -> re.Pattern:
    """Case-insensitive whole-word(s) match: 'xss' matches 'XSS,' but not 'xssfoo'."""
    return re.compile(r"(?<![\w-])" + re.escape(cue) + r"(?![\w-])", re.IGNORECASE)


def find_spans(text: str, cues: list[str]) -> list[tuple[int, int, str]]:
    """Character spans (start, end, cue) of every occurrence of every cue in `text`."""
    spans = []
    for cue in cues:
        for m in _cue_regex(cue).finditer(text):
            spans.append((m.start(), m.end(), cue))
    return sorted(spans)


# --------------------------------------------------------------------------- #
# Building, saving and loading the frozen lexicons
# --------------------------------------------------------------------------- #
def build_lexicon(metric: str, train: pd.DataFrame, cfg: dict) -> dict:
    """Filter the candidate cues of one metric using ONLY the training split."""
    lc = cfg["lexicon"]
    texts = train["text"]
    gold = train[metric]
    base_rate = gold.value_counts(normalize=True).to_dict()

    tree_out = []
    for node in DECISION_TREES[metric]:
        label, kept, dropped = node["label"], [], []
        for cue in node["cues"]:
            hit = texts.str.contains(_cue_regex(cue), regex=True)
            support = int(hit.sum())
            precision = float((gold[hit] == label).mean()) if support else 0.0
            lift = precision / base_rate.get(label, 1e-9)
            info = {"cue": cue, "support": support, "precision": round(precision, 3), "lift": round(lift, 2)}
            if support < lc["min_support"]:
                dropped.append({**info, "reason": "too rare"})
            elif precision >= lc["min_precision"] or lift >= lc["min_lift"]:
                kept.append(info)
            else:
                dropped.append({**info, "reason": "unreliable"})
        tree_out.append({"question": node["question"], "label": label, "kept": kept, "dropped": dropped})

    cues_by_label: dict[str, list[str]] = {}
    for node in tree_out:
        cues_by_label.setdefault(node["label"], []).extend(c["cue"] for c in node["kept"])
    all_cues = sorted({c for cues in cues_by_label.values() for c in cues})
    return {
        "metric": metric,
        "built_from": f"training split only ({len(train):,} descriptions)",
        "frozen": True,
        "rule": {k: lc[k] for k in ("min_support", "min_precision", "min_lift")},
        "cues_by_label": cues_by_label,
        "all_cues": all_cues,
        "decision_tree": tree_out,
    }


def load_lexicon(metric: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    f = path(cfg["paths"]["lexicon_dir"]) / f"{metric}.json"
    if not f.exists():
        raise SystemExit(f"{f} missing - run `python -m src.lexicon` first.")
    with open(f) as fh:
        return json.load(fh)


def expert_cues(metric: str, cfg: dict | None = None) -> list[str]:
    """K_exp for one metric: the union of the kept cues of all its labels."""
    return load_lexicon(metric, cfg)["all_cues"]


def coverage(texts: pd.Series, cues: list[str]) -> float:
    """% of descriptions with at least one cue match."""
    pattern = re.compile("|".join(_cue_regex(c).pattern for c in cues), re.IGNORECASE)
    return 100 * texts.str.contains(pattern, regex=True).mean()


def main() -> None:
    from src.data import load_split

    cfg = load_config()  # full splits: the lexicon is built on the WHOLE training split
    train = load_split(cfg, "train")
    test = load_split(cfg, "test")
    sample2k = test.sample(n=2000, random_state=cfg["seed"])
    out_dir = path(cfg["paths"]["lexicon_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    union = set()
    for m in metric_names(cfg):
        lex = build_lexicon(m, train, cfg)
        with open(out_dir / f"{m}.json", "w") as fh:
            json.dump(lex, fh, indent=2)
        union.update(lex["all_cues"])
        n_cand = sum(len(n["cues"]) for n in DECISION_TREES[m])
        rows.append({"metric": m.upper(), "candidate_cues": n_cand, "kept_cues": len(lex["all_cues"]),
                     "coverage_test_%": round(coverage(test["text"], lex["all_cues"]), 1),
                     "coverage_2000_%": round(coverage(sample2k["text"], lex["all_cues"]), 1)})
        print(f"{m.upper():>2}: kept {len(lex['all_cues'])}/{n_cand} cues  " +
              "  ".join(f"{lab}={len(c)}" for lab, c in lex["cues_by_label"].items()))

    rows.append({"metric": "ANY", "candidate_cues": None, "kept_cues": len(union),
                 "coverage_test_%": round(coverage(test["text"], sorted(union)), 1),
                 "coverage_2000_%": round(coverage(sample2k["text"], sorted(union)), 1)})
    table = pd.DataFrame(rows)
    print("\nLexicon coverage (% of descriptions with >= 1 match). Paper: 83.6% on 2,000 CVEs.")
    print(table.to_string(index=False))
    res = path(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    table.to_csv(res / "lexicon_coverage.csv", index=False)


if __name__ == "__main__":
    main()
