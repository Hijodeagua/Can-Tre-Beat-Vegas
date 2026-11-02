"""The NFL and CFB update emails show two probabilities that can disagree:
the pick model's (`p_home`, Elo + efficiency second stage) and pure Elo's
(`p_home_elo`). The old footnote called the pick probability "straight
from Elo", which was false, and the Elo-derived line and expected score
sat unlabelled next to a pick they could contradict (Detroit picked at 53%
with a Minnesota-favoured Elo line, 2026-11-01). Each column must carry the
model it comes from and the numbers must match the slate's own columns."""

from __future__ import annotations

import html as html_lib
import re

import pandas as pd

from CFB.daily import emails as cfb_emails
from NFL.daily import emails as nfl_emails


def _cells(html: str) -> list[list[str]]:
    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.S)
    out = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)
        out.append([html_lib.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
    return out


NFL_SLATE = pd.DataFrame([
    # the real 2026-11-01 contradiction: model picks DET, Elo favours MIN
    {"game_id": "2026_08_MIN_DET", "date": "2026-11-01", "weekday": "Sunday", "gametime": "13:00",
     "home_team": "DET", "away_team": "MIN", "neutral": False,
     "p_home": 0.531, "p_home_elo": 0.4647, "model": "elo+efficiency:random_forest",
     "pick": "DET", "pick_prob": 0.531, "elo_spread": -1.3,
     "pred_home_score": 21.6, "pred_away_score": 22.9, "pred_total": 44.6},
    {"game_id": "2026_08_BAL_BUF", "date": "2026-11-01", "weekday": "Sunday", "gametime": "13:00",
     "home_team": "BUF", "away_team": "BAL", "neutral": False,
     "p_home": 0.6846, "p_home_elo": 0.7513, "model": "elo+efficiency:random_forest",
     "pick": "BUF", "pick_prob": 0.6846, "elo_spread": 7.5,
     "pred_home_score": 29.8, "pred_away_score": 22.3, "pred_total": 52.1},
])


class TestNfl:
    def test_columns_are_labelled_by_model_and_match_the_slate(self):
        html = nfl_emails._fixtures_section(NFL_SLATE)
        assert "straight from Elo" not in html
        table = _cells(html)
        header = table[0]
        assert header == ["ET", "Matchup (home in bold)", "Pick", "Pick model win prob",
                          "Elo-only win prob", "Elo-only line (home)",
                          "Elo-only exp. score (pick first)", "Elo-only exp. total"]
        det = table[1]
        assert det[2].startswith("DET") and "≠ Elo" in det[2]
        assert det[3] == "53%"            # pick model, picked side (p_home)
        assert det[4] == "46%"            # pure Elo for the same side (p_home_elo)
        assert det[5] == "+1.3"           # Elo-only line from the home side: MIN favoured
        assert det[6] == "21.6–22.9"      # Elo-only expected score, pick first
        buf = table[2]
        assert buf[3] == "68%" and buf[4] == "75%" and "≠" not in buf[2]
        assert "disagree on 1 game" in html
        assert "elo + efficiency (random forest)" in html

    def test_elo_prob_is_for_the_picked_side(self):
        away = NFL_SLATE.iloc[[0]].copy()
        away["pick"], away["pick_prob"] = "MIN", 0.6
        away["p_home"] = 0.4
        html = nfl_emails._fixtures_section(away)
        row = _cells(html)[1]
        assert row[3] == "60%" and row[4] == "54%"   # 1 - 0.4647 for MIN
        assert "≠" not in row[2]

    def test_slate_without_elo_column_degrades(self):
        old = NFL_SLATE.drop(columns=["p_home_elo", "model"])
        html = nfl_emails._fixtures_section(old)
        assert _cells(html)[1][4] == "—"
        assert "pure Elo model" in html

    def test_past_week_labels(self):
        recent = pd.DataFrame([{
            "date": "2026-10-26", "home_team": "SEA", "away_team": "LAR", "neutral": False,
            "pick": "SEA", "pick_prob": 0.54, "elo_spread": 3.0, "home_score": 20,
            "away_score": 17, "log_loss": 0.62, "pick_correct": True, "tie": False}])
        html = nfl_emails._performance_section(recent, {"graded": 1, "correct": 1,
                                                        "accuracy": 1.0, "log_loss": 0.62})
        assert "Pick (pick model prob)" in html and "Elo-only line" in html


CFB_SLATE = pd.DataFrame([
    {"game_id": 1, "date": "2026-11-01", "home_team": "Air Force", "away_team": "UConn",
     "neutral": False, "home_fcs": False, "away_fcs": False,
     "p_home": 0.44, "p_home_elo": 0.58, "model": "elo+efficiency:random_forest",
     "pick": "UConn", "pick_prob": 0.56,
     "pred_home_score": 27.0, "pred_away_score": 24.0, "pred_total": 51.0},
])


class TestCfb:
    def test_columns_are_labelled_by_model_and_match_the_slate(self):
        html = cfb_emails._fixtures_section(CFB_SLATE)
        assert "straight from Elo" not in html
        table = _cells(html)
        assert table[0] == ["Matchup (home in bold)", "Pick", "Pick model win prob",
                            "Elo-only win prob", "Elo-only exp. score (pick first)",
                            "Elo-only exp. total"]
        row = table[1]
        assert row[1].startswith("UConn") and "≠ Elo" in row[1]
        assert row[2] == "56%"       # pick model for UConn
        assert row[3] == "42%"       # pure Elo for UConn = 1 - 0.58
        assert row[4] == "24.0–27.0"  # Elo-only score, pick first, still favours Air Force
        assert "disagree on 1 game" in html

    def test_past_week_labels(self):
        recent = pd.DataFrame([{
            "date": "2026-10-25", "home_team": "A", "away_team": "B", "neutral": False,
            "pick": "A", "pick_prob": 0.7, "home_points": 30, "away_points": 10,
            "log_loss": 0.36, "pick_correct": True}])
        html = cfb_emails._performance_section(recent, {"graded": 1, "correct": 1,
                                                        "accuracy": 1.0, "log_loss": 0.36})
        assert "Pick (pick model prob)" in html
