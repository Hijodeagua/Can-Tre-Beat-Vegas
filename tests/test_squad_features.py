"""Tests for squad-quality features and the awards scraper parsers.

The scraper's network hosts (PFR, Wikipedia) are blocked from CI, so the
parsers are exercised against fixture HTML shaped like the real pages. That is
also why the parsers are written to work off saved HTML — see
``scrape_awards.--reparse``.
"""

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2.squad import (
    ALLPRO_BANDS,
    _honor_weight,
    _norm_team,
    add_squad_features,
    draft_features,
    honor_features,
    interim_coach_flags,
    qb_features,
)
from data_jobs.rosters.scrape_awards import (
    parse_allpro,
    parse_pfr_players,
    parse_wikipedia_top100,
)


@pytest.fixture
def players():
    return pd.DataFrame({
        "gsis_id": ["p1", "p2", "p3", "p4", "p5"],
        "pfr_id": ["AaaA00", "BbbB00", "CccC00", "DddD00", "EeeE00"],
        "display_name": ["Star Guy", "Second Guy", "Late Guy", "Udfa Guy", "Qb Guy"],
        "position": ["WR", "OT", "LB", "CB", "QB"],
        "draft_year": [2018, 2019, 2020, np.nan, 2017],
        "draft_round": [1.0, 2.0, 6.0, np.nan, 1.0],
        "draft_pick": [4.0, 40.0, 190.0, np.nan, 10.0],
    })


@pytest.fixture
def rosters():
    rows = []
    for week in (1, 2):
        for pid in ["p1", "p2", "p3", "p4", "p5"]:
            rows.append({"season": 2024, "team": "KC", "week": week,
                         "status": "ACT", "gsis_id": pid, "position": "X"})
    # A second team with only the undrafted player active.
    for week in (1, 2):
        rows.append({"season": 2024, "team": "DEN", "week": week,
                     "status": "ACT", "gsis_id": "p4", "position": "X"})
    return pd.DataFrame(rows)


class TestDraftFeatures:
    def test_counts_first_and_top2_rounders(self, rosters, players):
        d = draft_features(rosters, players)
        kc = d[(d["team"] == "KC") & (d["week"] == 1)].iloc[0]
        assert kc["n_first_rounders"] == 2      # p1, p5
        assert kc["n_top2_rounders"] == 3       # + p2
        assert kc["roster_size"] == 5

    def test_undrafted_players_lower_pct_drafted(self, rosters, players):
        d = draft_features(rosters, players)
        kc = d[(d["team"] == "KC") & (d["week"] == 1)].iloc[0]
        den = d[(d["team"] == "DEN") & (d["week"] == 1)].iloc[0]
        assert kc["pct_drafted"] == pytest.approx(0.8)   # 4 of 5 drafted
        assert den["pct_drafted"] == 0.0

    def test_is_per_week_not_per_season(self, rosters, players):
        """A mid-season roster change must show up in that week only."""
        r = rosters[~((rosters["gsis_id"] == "p1") & (rosters["week"] == 2))]
        d = draft_features(r, players)
        wk1 = d[(d["team"] == "KC") & (d["week"] == 1)].iloc[0]
        wk2 = d[(d["team"] == "KC") & (d["week"] == 2)].iloc[0]
        assert wk1["n_first_rounders"] == 2
        assert wk2["n_first_rounders"] == 1


class TestHonorWeights:
    def test_bands_take_the_max_not_the_sum(self):
        # A selection 2 seasons ago is in every band; it must score 1.0.
        assert _honor_weight(2) == 1.0
        assert _honor_weight(4) == 0.75
        assert _honor_weight(9) == 0.5

    def test_bands_are_ordered_and_decreasing(self):
        weights = [w for _, w in ALLPRO_BANDS]
        assert weights == sorted(weights, reverse=True)

    def test_honors_absent_yields_nan_columns_not_errors(self, rosters, players):
        empty = {"allpro": pd.DataFrame(), "probowl": pd.DataFrame(),
                 "top100": pd.DataFrame()}
        h = honor_features(rosters, players, empty)
        assert h["allpro_score"].isna().all()
        assert len(h) == 4  # 2 teams x 2 weeks

    def test_only_prior_seasons_count(self, rosters, players):
        """An All-Pro selection in the game's own season must not be counted."""
        awards = {
            "allpro": pd.DataFrame({"season": [2024, 2022], "pfr_id": ["AaaA00", "BbbB00"]}),
            "probowl": pd.DataFrame(),
            "top100": pd.DataFrame(),
        }
        h = honor_features(rosters, players, awards)
        kc = h[(h["team"] == "KC") & (h["week"] == 1)].iloc[0]
        # 2024 selection is ignored (same season); 2022 is 2 seasons back -> 1.0
        assert kc["allpro_score"] == pytest.approx(1.0)

    def test_probowl_lookback_window(self, rosters, players):
        awards = {
            "allpro": pd.DataFrame(),
            "probowl": pd.DataFrame({"season": [2021, 2010], "pfr_id": ["AaaA00", "BbbB00"]}),
            "top100": pd.DataFrame(),
        }
        h = honor_features(rosters, players, awards)
        kc = h[(h["team"] == "KC") & (h["week"] == 1)].iloc[0]
        assert kc["n_probowlers"] == 1  # 2010 is outside the 5-season window


class TestQbFeatures:
    def _games(self):
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "gameday": pd.to_datetime(["2024-09-08", "2024-09-15", "2024-09-22"]),
            "home_team": ["KC", "KC", "KC"],
            "away_team": ["DEN", "DEN", "DEN"],
            "home_qb_name": ["Qb Guy", "Star Guy", "Qb Guy"],
            "away_qb_name": ["Second Guy"] * 3,
            "home_coach": ["A", "A", "A"], "away_coach": ["B", "B", "B"],
        })

    def test_prior_season_rating_is_used(self, players):
        epa = pd.DataFrame({"gsis_id": ["p5", "p1", "p2"],
                            "season": [2023, 2023, 2023],
                            "qb_epa": [0.20, 0.05, 0.10]})
        q = qb_features(self._games(), players, epa)
        g1 = q[(q["game_id"] == "g1") & (q["side"] == "home")].iloc[0]
        assert g1["qb_epa_prior"] == pytest.approx(0.20)

    def test_quality_drop_is_positive_on_a_downgrade(self, players):
        epa = pd.DataFrame({"gsis_id": ["p5", "p1", "p2"],
                            "season": [2023, 2023, 2023],
                            "qb_epa": [0.20, 0.05, 0.10]})
        q = qb_features(self._games(), players, epa)
        home = q[q["side"] == "home"].set_index("game_id")
        # g2 swaps the 0.20 starter for the 0.05 backup -> drop of +0.15
        assert home.loc["g2", "qb_quality_drop"] == pytest.approx(0.15)
        # g3 restores the starter -> negative drop (an upgrade)
        assert home.loc["g3", "qb_quality_drop"] == pytest.approx(-0.15)

    def test_same_season_rating_is_not_used(self, players):
        """A 2024 rating must never inform a 2024 game."""
        epa = pd.DataFrame({"gsis_id": ["p5"], "season": [2024], "qb_epa": [0.9]})
        q = qb_features(self._games(), players, epa)
        assert q["qb_epa_prior"].isna().all()


class TestInterimCoach:
    def test_flag_set_only_after_a_change(self):
        games = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "season": [2024] * 3,
            "gameday": pd.to_datetime(["2024-09-08", "2024-10-08", "2024-11-08"]),
            "home_team": ["KC"] * 3, "away_team": ["DEN"] * 3,
            "home_coach": ["Andy", "Andy", "Interim Guy"],
            "away_coach": ["Sean"] * 3,
        })
        f = interim_coach_flags(games).set_index(["game_id", "side"])
        assert f.loc[("g1", "home"), "is_interim_coach"] == 0
        assert f.loc[("g2", "home"), "is_interim_coach"] == 0
        assert f.loc[("g3", "home"), "is_interim_coach"] == 1
        assert f.loc[("g3", "away"), "is_interim_coach"] == 0


def test_team_code_normalisation():
    s = pd.Series(["ARZ", "BLT", "CLV", "HST", "SL", "KC"])
    assert list(_norm_team(s)) == ["ARI", "BAL", "CLE", "HOU", "STL", "KC"]


def test_add_squad_features_degrades_without_data(tmp_path, monkeypatch):
    """No roster cache -> NaN columns, never an exception."""
    import NFL.model.v2.squad as sq
    monkeypatch.setattr(sq, "NFLVERSE_DIR", tmp_path / "nope")
    monkeypatch.setattr(sq, "OUT_PATH", tmp_path / "missing.csv")
    games = pd.DataFrame({
        "game_id": ["g1"], "season": [2024], "week": [1],
        "gameday": pd.to_datetime(["2024-09-08"]),
        "home_team": ["KC"], "away_team": ["DEN"],
        "home_qb_name": ["X"], "away_qb_name": ["Y"],
        "home_coach": ["A"], "away_coach": ["B"],
    })
    out = sq.add_squad_features(games)
    assert len(out) == 1
    assert out["home_n_first_rounders"].isna().all()
    assert out["home_is_interim_coach"].iloc[0] == 0  # still computable


# --------------------------------------------------------------------------
# awards scraper parsers (fixture HTML — the live hosts are egress-blocked)
# --------------------------------------------------------------------------

ALLPRO_HTML = """
<html><body><table id="allpro"><tbody>
<tr><td>1st Team</td><td><a href="/players/M/MahoPa00.htm">Patrick Mahomes</a></td></tr>
<tr><td>2nd Team</td><td><a href="/players/A/AlleJo02.htm">Josh Allen</a></td></tr>
<tr><td>1st Team</td><td><a href="/players/A/AlleJo02.htm">Josh Allen</a></td></tr>
</tbody></table></body></html>
"""

COMMENTED_HTML = """
<html><body><!--<table><tbody>
<tr><td><a href="/players/K/KelcTr00.htm">Travis Kelce</a></td></tr>
</tbody></table>--></body></html>
"""

WIKI_HTML = """
<html><body><table class="wikitable">
<tr><th>Rank</th><th>Player</th><th>Team</th></tr>
""" + "".join(
    f"<tr><td>{i}</td><td>Player {i}[1]</td><td>KC</td></tr>" for i in range(1, 61)
) + """
</table></body></html>
"""


class TestAwardParsers:
    def test_allpro_extracts_ids_and_prefers_first_team(self):
        d = parse_allpro(ALLPRO_HTML)
        assert set(d["pfr_id"]) == {"MahoPa00", "AlleJo02"}
        # Josh Allen appears on both teams; first team must win.
        assert d.set_index("pfr_id").loc["AlleJo02", "team_level"] == 1

    def test_comment_wrapped_tables_are_read(self):
        """PFR hides tables in HTML comments; the parser must unwrap them."""
        d = parse_pfr_players(COMMENTED_HTML)
        assert list(d["pfr_id"]) == ["KelcTr00"]
        assert list(d["player"]) == ["Travis Kelce"]

    def test_pfr_player_dedupe(self):
        d = parse_pfr_players(ALLPRO_HTML)
        assert len(d) == 2

    def test_wikipedia_top100_ranks_and_cleans_footnotes(self):
        d = parse_wikipedia_top100(WIKI_HTML)
        assert len(d) == 60
        assert d.iloc[0]["rank"] == 1
        assert "[" not in d.iloc[0]["player"]

    def test_short_tables_are_rejected(self):
        small = WIKI_HTML.replace("</table>", "")[:400] + "</table>"
        assert parse_wikipedia_top100(small).empty
