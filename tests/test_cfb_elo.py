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


def test_cluster_regression_targets_cluster_mean():
    clusters = {(2001, "A1"): 0, (2001, "A2"): 0, (2001, "B1"): 1, (2001, "B2"): 1}
    eng = elo.ClusterRegressEngine(clusters, regression=0.5)
    eng.ratings.update({"A1": 1700.0, "A2": 1500.0, "B1": 1300.0, "B2": 1400.0})
    eng._last_season = 2000
    eng._roll_season(2001)
    # Cluster A mean 1600: A1 regresses 1700 -> 1650, A2 1500 -> 1550.
    assert eng.ratings["A1"] == 1650.0
    assert eng.ratings["A2"] == 1550.0
    # Cluster B mean 1350: B1 -> 1325, B2 -> 1375.
    assert eng.ratings["B1"] == 1325.0
    assert eng.ratings["B2"] == 1375.0


def test_cluster_regression_unclustered_falls_back_to_base():
    eng = elo.ClusterRegressEngine({}, regression=0.5)
    eng.ratings.update({"Lone U": 1700.0})
    eng._last_season = 2000
    eng._roll_season(2001)
    assert eng.ratings["Lone U"] == 1600.0  # halfway to 1500


def test_conference_clustering_separates_cliques():
    from CFB.conferences import _cluster_graph

    edges = []
    conf_a = ["A1", "A2", "A3", "A4"]
    conf_b = ["B1", "B2", "B3", "B4"]
    for conf in (conf_a, conf_b):
        edges += [(x, y) for i, x in enumerate(conf) for y in conf[i + 1:]]
    edges.append(("A1", "B1"))  # one cross-conference game
    labels = _cluster_graph(edges)
    assert len({labels[t] for t in conf_a}) == 1
    assert len({labels[t] for t in conf_b}) == 1
    assert labels["A1"] != labels["B1"]


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
