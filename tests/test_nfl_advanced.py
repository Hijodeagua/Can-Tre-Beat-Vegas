"""Tests for the NFL play-by-play aggregates (NFL/data/pbp.py) and the
pregame efficiency layer (NFL/model/advanced.py): metric definitions on a
hand-built game, one row per (game, team), ratings that never see the
week they predict, shifted form, franchise-key joins, determinism."""

import numpy as np
import pandas as pd
import pytest

from NFL.data import pbp
from NFL.model import advanced as adv


# --------------------------------------------------------------------------
# play-by-play fixture: one game, HOME vs AWAY, a handful of plays
# --------------------------------------------------------------------------
def _play(**kw):
    base = dict(game_id="2024_01_AWAY_HOME", season=2024, week=1, season_type="REG",
                game_date="2024-09-08", home_team="HOME", away_team="AWAY",
                posteam="HOME", defteam="AWAY", play_type="pass", qb_dropback=1, **{"pass": 1},
                rush=0, sack=0, complete_pass=1, epa=0.5, success=1, xpass=0.6, pass_oe=40.0,
                yards_gained=8, down=1, ydstogo=10, yardline_100=50, qtr=1,
                game_seconds_remaining=3000, score_differential=0, fixed_drive=1,
                fixed_drive_result="Touchdown", series=1, series_success=1,
                special_teams_play=0, first_down=0, posteam_score=0, posteam_score_post=0,
                home_score=7, away_score=3)
    base.update(kw)
    return base


def _game():
    rows = [
        # HOME drive 1: pass +0.5, run +0.3 (explosive 12), pass sack -1.0, 3rd-and-2 pass converts
        _play(epa=0.5, yards_gained=8, game_seconds_remaining=3000),
        _play(play_type="run", qb_dropback=0, rush=1, **{"pass": 0}, pass_oe=-30.0, epa=0.3,
              yards_gained=12, down=2, ydstogo=2, game_seconds_remaining=2970),
        _play(sack=1, epa=-1.0, success=0, yards_gained=-6, down=1, ydstogo=10,
              game_seconds_remaining=2940, yardline_100=15),
        _play(epa=1.2, yards_gained=15, down=3, ydstogo=2, first_down=1, yardline_100=15,
              game_seconds_remaining=2900, posteam_score_post=7),
        # AWAY drive 1: two runs, punt (fixed_drive_result Punt), 3rd-and-9 fails
        _play(posteam="AWAY", defteam="HOME", play_type="run", qb_dropback=0, rush=1,
              **{"pass": 0}, pass_oe=-20.0, epa=-0.2, success=0, yards_gained=1,
              fixed_drive=2, fixed_drive_result="Punt", series=2, series_success=0,
              game_seconds_remaining=2800, posteam_score=3, posteam_score_post=3),
        _play(posteam="AWAY", defteam="HOME", play_type="run", qb_dropback=0, rush=1,
              **{"pass": 0}, pass_oe=-20.0, epa=-0.4, success=0, yards_gained=0, down=2,
              ydstogo=9, fixed_drive=2, fixed_drive_result="Punt", series=2, series_success=0,
              game_seconds_remaining=2760, posteam_score=3, posteam_score_post=3),
        _play(posteam="AWAY", defteam="HOME", epa=-0.6, success=0, yards_gained=3, down=3,
              ydstogo=9, fixed_drive=2, fixed_drive_result="Punt", series=2, series_success=0,
              game_seconds_remaining=2720, posteam_score=3, posteam_score_post=3),
        # punt itself: special teams, AWAY kicking
        _play(posteam="AWAY", defteam="HOME", play_type="punt", qb_dropback=0, **{"pass": 0},
              epa=-0.3, special_teams_play=1, fixed_drive=2, fixed_drive_result="Punt",
              series=2, series_success=0, game_seconds_remaining=2700, posteam_score=3,
              posteam_score_post=3),
    ]
    return pd.DataFrame(rows)


class TestAggregates:
    def test_one_row_per_game_team_and_definitions(self):
        out = pbp.team_game_aggregates(_game())
        assert len(out) == 2
        assert not out.duplicated(["game_id", "team"]).any()
        home = out[out.team == "HOME"].iloc[0]
        away = out[out.team == "AWAY"].iloc[0]
        # offence: 4 pass/run plays, EPA mean of (0.5, 0.3, -1.0, 1.2)
        assert home.off_plays == 4
        assert home.off_epa == pytest.approx(np.mean([0.5, 0.3, -1.0, 1.2]))
        # dropbacks = qb_dropback == 1 (3), sack rate 1/3; rushes exclude dropbacks
        assert home.off_dropbacks == 3
        assert home.off_sack_rate == pytest.approx(1 / 3)
        assert home.off_rushes == 1 and home.off_rush_epa == pytest.approx(0.3)
        # early downs: three plays on downs 1-2
        assert home.off_early_epa == pytest.approx(np.mean([0.5, 0.3, -1.0]))
        # PROE is the mean pass_oe over pass+run plays
        assert home.off_proe == pytest.approx(np.mean([40, -30, 40, 40]))
        # explosive: the 12-yard run and the 15-yard pass? no — pass needs 20
        assert home.off_explosive_rate == pytest.approx(1 / 4)
        assert home.off_explosive_rush_rate == pytest.approx(1.0)
        assert home.off_explosive_pass_rate == pytest.approx(0.0)
        # third down: one attempt, converted, distance 2, short conv 1.0, long NaN
        assert home.off_third_att == 1 and home.off_third_conv == 1.0
        assert home.off_third_short_conv == 1.0 and np.isnan(home.off_third_long_conv)
        # drive: one drive, 7 points, 29 yards, 4 plays, red-zone trip with a TD
        assert home.off_drives == 1 and home.off_points_per_drive == 7
        assert home.off_yards_per_drive == 29 and home.off_plays_per_drive == 4
        assert home.off_rz_trips == 1 and home.off_rz_td_per_trip == 1.0
        # defence view mirrors the opponent's offence
        assert home.def_epa == pytest.approx(away.off_epa)
        assert home.def_drives == 1 and home.def_points_per_drive == 0
        assert away.def_epa == pytest.approx(home.off_epa)
        # ST net: AWAY's punt EPA -0.3 counts for AWAY and against HOME
        assert away.st_epa_net == pytest.approx(-0.3)
        assert home.st_epa_net == pytest.approx(0.3)
        assert home.is_home == 1 and home.points_for == 7 and home.points_against == 3

    def test_special_teams_plays_are_not_offence(self):
        out = pbp.team_game_aggregates(_game())
        away = out[out.team == "AWAY"].iloc[0]
        assert away.off_plays == 3          # the punt is excluded


# --------------------------------------------------------------------------
# team-game fixture for the ratings and form
# --------------------------------------------------------------------------
def _team_games(seasons=(2023, 2024), weeks=6):
    """Four teams, round robin, GOOD's offence produces +0.3 EPA against
    everyone; the rest are league-average. Metrics the ratings need are
    filled in; the rest are constants."""
    teams = ["GOOD", "B", "C", "D"]
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            pairs = [(teams[0], teams[(w) % 3 + 1]), tuple(t for t in teams[1:] if t != teams[(w) % 3 + 1])]
            for home, away in pairs:
                gid = f"{s}_{w:02d}_{away}_{home}"
                for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                    off = 0.3 if team == "GOOD" else 0.0
                    row = {"game_id": gid, "team": team, "opponent": opp, "is_home": is_home,
                           "season": s, "week": w, "season_type": "REG",
                           "date": f"{s}-09-{w + 3:02d}"}
                    for m in adv.FORM_METRICS:
                        row[m] = float(w) if team == "GOOD" else 1.0
                    for m in adv.ADJ_METRICS:      # off_epa is in both lists; this wins
                        row[f"off_{m}"] = off
                        row[f"def_{m}"] = 0.3 if opp == "GOOD" else 0.0
                    rows.append(row)
    return pd.DataFrame(rows)


class TestAdjustedRatings:
    def test_good_offence_rated_above_and_its_opponents_defences_not_punished(self):
        r = adv.adjusted_ratings(_team_games())
        last = r[(r.season == 2024) & (r.week == 6)].set_index("team")
        assert last.loc["GOOD", "off_epa_adj"] > 0.1
        assert all(last.loc[t, "off_epa_adj"] < 0.0 for t in ["B", "C", "D"])
        # Defences that faced GOOD gave up 0.3, but the fit attributes that to
        # GOOD's offence, so their defence ratings stay near zero.
        assert last.loc[["B", "C", "D"], "def_epa_adj"].abs().max() < 0.03

    def test_ratings_for_week_w_ignore_week_w_games(self):
        tg = _team_games()
        r1 = adv.adjusted_ratings(tg)
        # Blow up GOOD's week 4 (2024) result; weeks <= 4 must not move.
        tg2 = tg.copy()
        m = (tg2.season == 2024) & (tg2.week == 4) & (tg2.team == "GOOD")
        tg2.loc[m, "off_epa"] = 5.0
        r2 = adv.adjusted_ratings(tg2)
        for w in (1, 2, 3, 4):
            a = r1[(r1.season == 2024) & (r1.week == w)].sort_values("team")["off_epa_adj"].to_numpy()
            b = r2[(r2.season == 2024) & (r2.week == w)].sort_values("team")["off_epa_adj"].to_numpy()
            np.testing.assert_allclose(a, b)
        a = r1[(r1.season == 2024) & (r1.week == 5)].set_index("team").loc["GOOD", "off_epa_adj"]
        b = r2[(r2.season == 2024) & (r2.week == 5)].set_index("team").loc["GOOD", "off_epa_adj"]
        assert b > a + 0.1

    def test_week_one_uses_only_the_discounted_prior_season(self):
        r = adv.adjusted_ratings(_team_games())
        w1 = r[(r.season == 2024) & (r.week == 1)].set_index("team")
        w6_prev = r[(r.season == 2023) & (r.week == 6)].set_index("team")
        assert 0 < w1.loc["GOOD", "off_epa_adj"] < w6_prev.loc["GOOD", "off_epa_adj"]
        assert w1.loc["GOOD", "n_eff"] < w6_prev.loc["GOOD", "n_eff"]
        # No previous season at all -> no row for 2023 week 1.
        assert r[(r.season == 2023) & (r.week == 1)].empty

    def test_one_row_per_season_week_team(self):
        r = adv.adjusted_ratings(_team_games())
        assert not r.duplicated(["season", "week", "team"]).any()

    def test_deterministic(self):
        tg = _team_games()
        pd.testing.assert_frame_equal(adv.adjusted_ratings(tg), adv.adjusted_ratings(tg))


class TestForm:
    def test_ewm_is_shifted_and_shrunk(self):
        tg = _team_games(seasons=(2024,))
        f = adv.ewm_form(tg).merge(tg[["game_id", "team", "week"]], on=["game_id", "team"])
        good = f[f.team == "GOOD"].sort_values("week")
        # First game of the only season: no history -> the league mean.
        league_mean = tg["off_proe"].mean()
        assert good["off_proe_ewm"].iloc[0] == pytest.approx(league_mean)
        # Week w's feature never includes week w's value (w), which is the
        # largest so far: it must be below w after shrinkage toward the mean.
        for _, row in good.iloc[1:].iterrows():
            assert row["off_proe_ewm"] < row["week"]

    def test_one_row_per_game_team(self):
        f = adv.ewm_form(_team_games())
        assert not f.duplicated(["game_id", "team"]).any()


class TestGameTable:
    def _history(self, tg):
        games = tg[tg.is_home == 1]
        return pd.DataFrame({
            "game_id": games.game_id.to_numpy(), "season": games.season.to_numpy(),
            "week": games.week.to_numpy(), "home_team": games.team.to_numpy(),
            "away_team": games.opponent.to_numpy(), "p_home": 0.6, "home_win": 1.0,
        })

    def test_matchup_and_join(self):
        tg = _team_games()
        table = adv.build_game_table(self._history(tg), tg)
        assert len(table) == len(tg) // 2
        assert not table.duplicated("game_id").any()
        row = table[(table.season == 2024) & (table.week == 6) & (table.home_team == "GOOD")].iloc[0]
        assert row.epa_home_vs_away == pytest.approx(row.home_off_epa_adj + row.away_def_epa_adj)
        assert row.epa_matchup_net > 0.15
        assert row.elo_logit == pytest.approx(np.log(0.6 / 0.4))
        for f in adv.all_features():
            assert f in table.columns

    def test_historical_abbreviations_join_through_the_franchise_map(self):
        tg = _team_games()
        tg = tg.replace({"team": {"GOOD": "LV"}, "opponent": {"GOOD": "LV"}})
        tg["game_id"] = tg["game_id"].str.replace("GOOD", "LV")
        hist = self._history(tg).replace({"home_team": {"LV": "OAK"}, "away_team": {"LV": "OAK"}})
        table = adv.build_game_table(hist, tg)
        oak = table[table.home_team == "OAK"]
        assert oak["home_off_epa_adj"].notna().any()

    def test_missing_efficiency_rows_stay_nan(self):
        tg = _team_games()
        hist = self._history(tg)
        extra = hist.iloc[:1].copy()
        extra["game_id"], extra["season"], extra["week"] = "1999_01_B_GOOD", 1999, 1
        table = adv.build_game_table(pd.concat([hist, extra]), tg)
        old = table[table.season == 1999].iloc[0]
        assert np.isnan(old.home_off_epa_adj) and np.isnan(old.epa_matchup_net)
        assert not np.isnan(old.elo_logit)


class TestSecondStage:
    def _history(self, tg):
        games = tg[tg.is_home == 1]
        # GOOD's games are home wins, the others coin flips by week parity.
        return pd.DataFrame({
            "game_id": games.game_id.to_numpy(), "season": games.season.to_numpy(),
            "week": games.week.to_numpy(), "home_team": games.team.to_numpy(),
            "away_team": games.opponent.to_numpy(), "p_home": 0.55,
            "elo_home_pre": np.where(games.team.to_numpy() == "GOOD", 1600.0, 1500.0),
            "elo_away_pre": np.where(games.opponent.to_numpy() == "GOOD", 1600.0, 1500.0),
            "home_win": np.where(games.team.to_numpy() == "GOOD", 1.0,
                                 (games.week.to_numpy() % 2).astype(float)),
        })

    def test_snapshot_for_an_unplayed_week_matches_the_replay_row(self):
        tg = _team_games()
        full = adv.adjusted_ratings(tg)
        # Week 4 recomputed on its own, from a table that stops at week 3.
        snap = adv.rating_snapshot(tg[(tg.season < 2024) | (tg.week < 4)], 2024, 4)
        a = full[(full.season == 2024) & (full.week == 4)].set_index("team")["off_epa_adj"]
        b = snap.set_index("team")["off_epa_adj"]
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index())

    def test_stage_predicts_and_falls_back(self):
        tg = _team_games(seasons=(2022, 2023, 2024), weeks=8)
        stage = adv.SecondStage(tg, self._history(tg))
        assert stage.n_train > 0
        p = stage.p_home("GOOD", "B", 0.55, 2024, 9, elo_home=1600.0, elo_away=1500.0)
        assert 0 < p < 1
        # The stage sees every feature: current form for a week not yet played.
        row = stage.feature_row("GOOD", "B", 0.55, 2024, 9, 1600.0, 1500.0)
        assert all(f in row for f in adv.PRODUCTION_FEATURES)
        assert row["elo_home_pre"] == 1600.0 and not np.isnan(row["home_off_proe_ewm"])
        # A team with no rating -> None, caller falls back to Elo.
        assert stage.p_home("GOOD", "NOBODY", 0.55, 2024, 9, 1600.0, 1500.0) is None
        assert stage.p_home("GOOD", "B", 0.55, 2030, 1, 1600.0, 1500.0) is None
        # No raw Elo supplied -> None too: the Elo inputs are required.
        assert stage.p_home("GOOD", "B", 0.55, 2024, 9) is None

    def test_freshness_reads_stale_when_the_spine_is_ahead(self):
        tg = _team_games(seasons=(2024,), weeks=4)
        games = pd.DataFrame({"date": ["2024-09-07", "2024-10-20"], "completed": [True, True]})
        assert not adv.aggregates_freshness(tg, games).fresh
        games = pd.DataFrame({"date": ["2024-09-07", "2024-09-12"], "completed": [True, True]})
        assert adv.aggregates_freshness(tg, games).fresh
        # Off-season: nothing newer completed -> fresh; empty aggregates -> stale.
        assert adv.aggregates_freshness(tg, games.iloc[0:0]).fresh
        assert not adv.aggregates_freshness(tg.iloc[0:0], games).fresh


class TestCurrentForm:
    def test_next_game_form_matches_a_shifted_value(self):
        tg = _team_games(seasons=(2024,), weeks=6)
        form = adv.current_form(tg, 2024)
        # GOOD's off_proe was 1..6; the next game's EWMA is the shifted value
        # a 7th game would carry: below 6, above the league mean.
        v = form.at["GOOD", "off_proe_ewm"]
        assert tg["off_proe"].mean() < v < 6.0
        # A new season: nothing played -> shrunk fully to the league mean.
        assert adv.current_form(tg, 2025).at["GOOD", "off_proe_ewm"] == pytest.approx(tg["off_proe"].mean())
