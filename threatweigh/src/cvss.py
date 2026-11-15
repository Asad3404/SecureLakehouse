"""
The official CVSS v3.1 Base Score calculator.

Reference: FIRST, "CVSS v3.1 Specification Document", Section 7 (equations)
and Appendix A (the Roundup function).  https://www.first.org/cvss/v3.1/specification-document

Usage:
    >>> from src.cvss import base_score, vector_string
    >>> base_score({"av": "N", "ac": "L", "pr": "N", "ui": "N", "s": "U", "c": "H", "i": "H", "a": "H"})
    9.8
"""
from __future__ import annotations

import math

# ---- Numeric weights for each metric value (CVSS v3.1 spec, Table 16) ------ #
AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}   # Attack Vector
AC = {"L": 0.77, "H": 0.44}                          # Attack Complexity
PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}     # Privileges Required, Scope Unchanged
PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}       # Privileges Required, Scope Changed
UI = {"N": 0.85, "R": 0.62}                          # User Interaction
CIA = {"H": 0.56, "L": 0.22, "N": 0.00}              # Confidentiality / Integrity / Availability

METRIC_ORDER = ["av", "ac", "pr", "ui", "s", "c", "i", "a"]


def roundup(value: float) -> float:
    """
    The official CVSS v3.1 Roundup: the smallest number, to 1 decimal place,
    that is >= the input. Implemented with integers (as in Appendix A) to avoid
    floating-point surprises such as roundup(4.000000000001) giving 4.1.
    """
    int_input = round(value * 100_000)
    if int_input % 10_000 == 0:
        return int_input / 100_000.0
    return (math.floor(int_input / 10_000) + 1) / 10.0


def base_score(labels: dict[str, str]) -> float:
    """
    Compute the 0-10 Base Score from the 8 single-letter labels.
    `labels` maps lowercase metric names (av, ac, pr, ui, s, c, i, a) to letters.
    """
    lab = {k.lower(): str(v).upper() for k, v in labels.items()}
    scope_changed = lab["s"] == "C"

    # Impact Sub-Score (ISS): how much C, I and A are affected together.
    iss = 1 - (1 - CIA[lab["c"]]) * (1 - CIA[lab["i"]]) * (1 - CIA[lab["a"]])

    if scope_changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss

    pr = (PR_CHANGED if scope_changed else PR_UNCHANGED)[lab["pr"]]
    exploitability = 8.22 * AV[lab["av"]] * AC[lab["ac"]] * pr * UI[lab["ui"]]

    if impact <= 0:
        return 0.0
    if scope_changed:
        return roundup(min(1.08 * (impact + exploitability), 10))
    return roundup(min(impact + exploitability, 10))


def severity(score: float) -> str:
    """Qualitative severity rating scale (CVSS v3.1 spec, Table 14)."""
    if score == 0:
        return "None"
    if score < 4.0:
        return "Low"
    if score < 7.0:
        return "Medium"
    if score < 9.0:
        return "High"
    return "Critical"


def vector_string(labels: dict[str, str]) -> str:
    """Build the official vector string, e.g. CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H."""
    lab = {k.lower(): str(v).upper() for k, v in labels.items()}
    return "CVSS:3.1/" + "/".join(f"{m.upper()}:{lab[m]}" for m in METRIC_ORDER)


def parse_vector(vector: str) -> dict[str, str]:
    """Inverse of vector_string(): 'CVSS:3.1/AV:N/...' -> {'av': 'N', ...}."""
    parts = [p for p in vector.split("/") if ":" in p and not p.startswith("CVSS")]
    return {k.lower(): v for k, v in (p.split(":") for p in parts)}


if __name__ == "__main__":
    # Tiny demo: python -m src.cvss
    for v in ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
              "CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
              "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"]:
        s = base_score(parse_vector(v))
        print(f"{v}  ->  {s}  ({severity(s)})")
