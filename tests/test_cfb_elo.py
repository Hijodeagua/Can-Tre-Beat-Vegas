"""FCS pooling (plan §3.4) — the game-count clause is what keeps
name-mismatched FBS schools (stats pages say ``LSU``/``USC``, schedules say
``Louisiana State``/``Southern California``) out of the FCS pool."""

import pandas as pd

import CFB.elo as elo


def _game(season, week, home, away, hs=30, as_=10):
    return {
        "season": season,
        "week": week,
        "gameday": f"{season}-09-{week:02d}",
        "game_type": "REG",
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "location": "Home",
    }


def test_full_schedule_team_kept_small_sample_pooled(tmp_path, monkeypatch):
    # No stats universe on disk -> membership rides on game count alone.
    monkeypatch.setattr(elo, "AGG_DIR", tmp_path)
    rows = [_game(2019, w, "Big U", f"Opp {w}") for w in range(1, 7)]
    rows.append(_game(2019, 7, "Opp 1", "Tiny U"))
    pooled = elo.pool_fcs(pd.DataFrame(rows))

    teams = set(pooled["home_team"]) | set(pooled["away_team"])
    # Big U played 6 games -> FBS despite missing from stats.
    assert "Big U" in teams
    # One-game opponents pool into the synthetic team.
    assert "Tiny U" not in teams
    assert elo.FCS_TEAM in teams
    # The Opp-vs-Tiny pairing became FCS-vs-FCS and was dropped.
    assert len(pooled) == 6


def test_stats_universe_membership_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(elo, "AGG_DIR", tmp_path)
    pd.DataFrame({"season": [2019], "team": ["Listed U"]}).to_csv(
        tmp_path / "cfb_offense_team_season.csv", index=False
    )
    # Listed U plays only 2 games but is in the stats universe -> FBS.
    rows = [
        _game(2019, 1, "Listed U", "Nobody A"),
        _game(2019, 2, "Listed U", "Nobody B"),
    ]
    pooled = elo.pool_fcs(pd.DataFrame(rows))
    assert set(pooled["home_team"]) == {"Listed U"}
    assert set(pooled["away_team"]) == {elo.FCS_TEAM}
