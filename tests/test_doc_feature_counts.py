"""The docs quote how many inputs each second stage takes. Those numbers
were wrong three times in a row (soccer 46 for 44, NFL 41 for 49, CFB 33
for 34) because nothing checked them. This does.

If you add or drop a feature, update the sentence the test points at.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _counts() -> dict[str, int]:
    from CFB.model import advanced as cfb
    from NFL.model import advanced as nfl
    from soccer.clubs.model import train as soccer

    return {"soccer": len(soccer.FEATURES),
            "nfl": len(nfl.PRODUCTION_FEATURES),
            "cfb": len(cfb.PRODUCTION_FEATURES)}


# Each claim is (file, regex with one capture group holding the number).
# The regex has to be specific enough that it matches one sport's
# sentence and nothing else in the file.
CLAIMS = [
    ("soccer", "docs/ADVANCED_METRICS.md", r"context \((\d+) inputs\)"),
    ("nfl", "docs/ADVANCED_METRICS.md", r"\((\d+) inputs\)\. Walk-forward as before"),
    ("cfb", "docs/ADVANCED_METRICS.md", r"\((\d+) inputs\)\. Same fixed split"),
    ("nfl", "docs/MODEL_FEATURES.md", r"ties excluded\)\. (\d+) inputs"),
    ("nfl", "docs/MODEL_SOURCES.md", r"EWMA form column \((\d+) inputs\)"),
    ("cfb", "docs/MODEL_FEATURES.md", r"last completed week\)\. (\d+) inputs"),
    ("cfb", "docs/MODEL_SOURCES.md", r"efficiency column \((\d+) inputs\)"),
    ("cfb", "web/app/lib/modelFeatures.ts", r"the full (\d+)-input forest"),
]


@pytest.mark.parametrize("sport,rel,pattern", CLAIMS)
def test_documented_input_count_matches_the_model(sport, rel, pattern):
    text = (ROOT / rel).read_text(encoding="utf-8")
    found = re.findall(pattern, text)
    assert found, f"{rel}: no input-count sentence matched {pattern!r}"
    expected = _counts()[sport]
    for n in found:
        assert int(n) == expected, (
            f"{rel} says {n} inputs for {sport}; the model has {expected}")
