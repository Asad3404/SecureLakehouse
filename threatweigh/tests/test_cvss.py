"""Unit tests for the CVSS v3.1 calculator (src/cvss.py)."""
import pytest

from src.cvss import base_score, parse_vector, roundup, severity, vector_string


def score(v: str) -> float:
    return base_score(parse_vector(v))


def test_required_examples():
    # The two examples required by the project specification.
    assert score("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") == 9.8
    # NOTE: the project brief listed 4.6 for this vector, but the official
    # formula gives 4.3 (impact 3.3734 + exploitability 0.9146 = 4.288 -> 4.3),
    # and NVD scores CVE-2017-2727 (this exact vector) as 4.3.
    assert score("CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L") == 4.3
    # The physical-access vector that really does score 4.6:
    assert score("CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N") == 4.6


def test_matches_every_score_in_the_dataset():
    """Recompute the score from the true labels for every training CVE (if the data exists)."""
    from src.utils import ROOT
    f = ROOT / "data/splits/train.csv"
    if not f.exists():
        pytest.skip("run `python -m src.data` first")
    import pandas as pd
    df = pd.read_csv(f, keep_default_na=False)
    calc = [base_score(r) for r in df[["av", "ac", "pr", "ui", "s", "c", "i", "a"]].to_dict("records")]
    mismatches = sum(abs(c - float(s)) > 1e-9 for c, s in zip(calc, df["score"]))
    assert mismatches == 0


@pytest.mark.parametrize("vector,expected", [
    # Values checked against the FIRST CVSS v3.1 calculator.
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),   # typical reflected XSS
    ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),   # local privilege escalation
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),   # remote DoS
    ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N", 5.9),
    ("CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:H", 7.2),
    ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H", 9.9),   # PR:L uses 0.68 when scope changes
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),   # no impact -> 0
])
def test_known_vectors(vector, expected):
    assert score(vector) == expected


def test_roundup_spec_examples():
    assert roundup(4.02) == 4.1
    assert roundup(4.00) == 4.0
    # The floating-point trap the official integer-based Roundup avoids.
    assert roundup(4.000000000000001) == 4.0


def test_severity_bands():
    assert [severity(s) for s in (0, 0.1, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0, 10)] == \
        ["None", "Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


def test_vector_roundtrip():
    labels = {"av": "N", "ac": "L", "pr": "N", "ui": "R", "s": "C", "c": "L", "i": "L", "a": "N"}
    assert vector_string(labels) == "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"
    assert parse_vector(vector_string(labels)) == labels
