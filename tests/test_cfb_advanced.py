"""Tests for the CFB weekly-summary layer: the through_week == W rule
(a week-W game sees the W-1 snapshot), postseason on the final snapshot,
the regressed prior-season blend for early weeks, the id crosswalk, the
frozen-copy detection behind freshness, and the spine's new id columns."""

import numpy as np
import pandas as pd
import pytest

from CFB.data import fetch_schedule, fetch_weekly
from CFB.model import advanced as adv


def _weeks(seasons=(2023, 2024), weeks=6, teams=((1, "Alpha"), (2, "Beta"), (3, "Gamma"))):
    """Weekly rows whose adj_off_epa equals the through_week (so a join
    on the wrong week is visible) and whose plays grow 65 per week."""
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            for tid, name in teams:
                row = {"season": s, "team_id": tid, "team": name, "through_week": w,
                       "division": "fbs", "conference": "X"}
                for m in fetch_weekly.METRICS:
                    row[m] = float(w) if m != "plays_off" else 65.0 * w
                row["adj_def_epa"] = -float(w)
                rows.append(row)
    return pd.DataFrame(rows)[fetch_weekly.COLUMNS]


class TestSnapshots:
    def test_week_w_game_sees_week_w_minus_one(self):
        tw = _weeks()
        snap = adv.snapshot_before(tw, 2024, 5)
        assert set(snap["through_week"]) == {4}
        # Week 1: nothing this season yet.
        assert adv.snapshot_before(tw, 2024, 1).empty

    def test_postseason_uses_the_final_snapshot(self):
        tw = _weeks()
        snap = adv.snapshot_before(tw, 2024, 1, postseason=True)
        assert set(snap["through_week"]) == {6}

    def test_missing_week_falls_back_to_latest_earlier(self):
        tw = _weeks()
        tw = tw[~((tw.season == 2024) & (tw.through_week == 4))]
        assert set(adv.snapshot_before(tw, 2024, 5)["through_week"]) == {3}

    def test_strength_blends_current_with_regressed_prior(self):
        tw = _weeks()
        # Prior final (2023 week 6): every team 6.0 -> league mean 6.0 -> regressed = 6.0.
        # Week 1 of 2024: prior only.
        s1 = adv.team_strength(tw, 2024, 1)
        assert s1.loc[1, "adj_off_epa"] == pytest.approx(6.0)
        assert s1.loc[1, "games_seen"] == 0.0
        # Week 5: current snapshot is week 4 (value 4, 4 games) blended with prior 6
        s5 = adv.team_strength(tw, 2024, 5)
        assert s5.loc[1, "games_seen"] == pytest.approx(4.0)
        assert s5.loc[1, "adj_off_epa"] == pytest.approx((4 * 4.0 + adv.PRIOR_GAMES * 6.0) / (4 + adv.PRIOR_GAMES))

    def test_prior_is_regressed_toward_the_mean(self):
        tw = _weeks()
        tw.loc[(tw.season == 2023) & (tw.team_id == 1), "adj_off_epa"] = 10.0
        prior = adv.prior_final(tw, 2024).set_index("team_id")
        mean = (10.0 + 6.0 + 6.0) / 3
        assert prior.loc[1, "adj_off_epa"] == pytest.approx(10.0 * 0.5 + mean * 0.5)

    def test_no_prior_and_no_snapshot_is_nan(self):
        tw = _weeks(seasons=(2024,))
        s = adv.team_strength(tw, 2024, 1)
        assert s.empty or s["adj_off_epa"].isna().all()


class TestGameTable:
    def _history(self):
        return pd.DataFrame({
            "game_id": [1, 2, 3], "season": [2024, 2024, 2024], "week": [5, 1, 1],
            "season_type": ["regular", "regular", "postseason"],
            "home_team": ["Alpha", "Alpha", "Beta"], "away_team": ["Beta", "Gamma", "Gamma"],
            "home_id": [1, pd.NA, 2], "away_id": [2, pd.NA, 3], "p_home": [0.6, 0.6, 0.5],
        })

    def test_features_come_from_the_right_snapshot(self):
        tw = _weeks()
        table = adv.build_game_table(self._history(), tw)
        w5 = table[table.game_id == 1].iloc[0]
        expected = (4 * 4.0 + adv.PRIOR_GAMES * 6.0) / (4 + adv.PRIOR_GAMES)
        assert w5.home_adj_epa_off == pytest.approx(expected)
        assert w5.home_adj_epa_def == pytest.approx(-expected)
        # net = (off − def)_home − (off − def)_away; both sides identical here.
        assert w5.net_adj_epa_diff == pytest.approx(0.0)
        post = table[table.game_id == 3].iloc[0]
        final = (6 * 6.0 + adv.PRIOR_GAMES * 6.0) / (6 + adv.PRIOR_GAMES)
        assert post.home_adj_epa_off == pytest.approx(final)
        assert w5.elo_logit == pytest.approx(np.log(0.6 / 0.4))

    def test_missing_ids_use_the_name_crosswalk(self):
        tw = _weeks()
        table = adv.build_game_table(self._history(), tw)
        w1 = table[table.game_id == 2].iloc[0]
        assert w1.home_adj_epa_off == pytest.approx(6.0)     # prior only, via name -> id 1
        assert w1.home_games_seen == 0.0
        for f in adv.all_features():
            assert f in table.columns

    def test_unknown_team_is_nan_not_an_error(self):
        tw = _weeks()
        h = self._history()
        h.loc[0, ["away_team", "away_id"]] = ["Nobody", pd.NA]
        table = adv.build_game_table(h, tw)
        row = table[table.game_id == 1].iloc[0]
        assert np.isnan(row.away_adj_epa_off) and np.isnan(row.adj_epa_matchup_net)
        assert not np.isnan(row.elo_logit)


class TestFreshness:
    def test_frozen_copies_do_not_count_as_coverage(self):
        tw = _weeks(seasons=(2026,), weeks=3)
        # Weeks 4-6 exist upstream but carry week 3's totals.
        frozen = pd.concat([tw[tw.through_week == 3].assign(through_week=w) for w in (4, 5, 6)])
        tw = pd.concat([tw, frozen], ignore_index=True)
        assert adv.real_weeks(tw)[2026] == 3
        games = pd.DataFrame({"season": [2026] * 6, "week": [1, 2, 3, 4, 5, 6],
                              "season_type": ["regular"] * 6, "completed": [True] * 6})
        report = adv.coverage_report(tw, games)
        assert report.age_days == 3 and not report.fresh
        games["completed"] = [True, True, True, True, False, False]
        assert adv.coverage_report(tw, games).fresh

    def test_empty_table_is_stale(self):
        games = pd.DataFrame({"season": [2026], "week": [1], "season_type": ["regular"], "completed": [True]})
        assert not adv.coverage_report(pd.DataFrame(columns=fetch_weekly.COLUMNS), games).fresh


class TestSchema:
    def test_normalize_weekly_keeps_schema_and_dedupes(self):
        raw = pd.DataFrame({"season": [2024, 2024], "team_id": ["194", "194"], "pos_team": ["Ohio State"] * 2,
                            "through_week": [3, 3], "adj_off_epa": [0.1, 0.2], "plays_off": [100, 100]})
        out = fetch_weekly.normalize(raw)
        assert list(out.columns) == fetch_weekly.COLUMNS
        assert len(out) == 1 and out.iloc[0]["adj_off_epa"] == 0.2
        assert out["success_off"].isna().all()          # absent upstream -> NaN, not missing

    def test_spine_ids_are_additive(self):
        raw = pd.DataFrame({
            "game_id": [1], "season": [2024], "week": [1], "season_type": ["regular"],
            "start_date": ["2024-08-31T16:00:00Z"], "home_team": ["Ohio State"], "away_team": ["Akron"],
            "home_division": ["fbs"], "away_division": ["fbs"], "home_points": [52], "away_points": [6],
            "neutral_site": [False], "conference_game": [False], "completed": [True],
            "home_id": [194], "away_id": [2006],
        })
        out = fetch_schedule.normalize(raw)
        assert out.loc[0, "home_id"] == 194 and out.loc[0, "away_id"] == 2006
        without = fetch_schedule.normalize(raw.drop(columns=["home_id", "away_id"]))
        assert pd.isna(without.loc[0, "home_id"])
        assert fetch_schedule.COLUMNS[-2:] == ["home_id", "away_id"]

    def test_backfill_ids_fills_only_missing(self, tmp_path):
        games = pd.DataFrame({"game_id": [1, 2], "home_id": pd.array([7, pd.NA], dtype="Int64"),
                              "away_id": pd.array([pd.NA, pd.NA], dtype="Int64")})
        (tmp_path / "cfb_schedules_2024.csv").write_text("game_id,home_id,away_id\n1,99,8\n2,5,6\n")
        out = fetch_schedule.backfill_ids(games, tmp_path)
        assert out["home_id"].tolist() == [7, 5] and out["away_id"].tolist() == [8, 6]


def test_the_production_forest_uses_the_tuned_hyperparameters():
    """College is the one sport where the hyperparameter search paid: a
    leaf of 100 and 0.6 of the features per split beat the shared
    defaults on all 12 seeds (+0.00168 mean, t = 7.97). If this drifts
    back to the shared defaults the model silently gets worse and
    noisier, which no other test would notice."""
    from CFB.model import advanced as adv

    clf = adv.make_model().named_steps["clf"]
    assert clf.min_samples_leaf == 100
    assert clf.max_features == 0.6


def test_the_tuned_parameters_do_not_leak_into_other_learners():
    """A head-to-head has to stay a head-to-head: asking for the
    logistic or boosting must give that family at the shared defaults,
    not the forest's tuned leaf."""
    from common import learners
    from CFB.model import advanced as adv

    gbm = adv.make_model("gbm").named_steps["clf"]
    assert gbm.min_samples_leaf == learners.GBM_MIN_LEAF
    adv.make_model("logistic")  # would raise if forest params were passed on
