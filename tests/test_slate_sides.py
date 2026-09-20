"""Tests for the per-side slate stats and the week-against-trend report.

The load-bearing property here is a negative one: surfacing each side's
own numbers must not change what the model is trained on or what the
graded CSV contains. Both are asserted directly.
"""

import numpy as np
import pandas as pd
import pytest

from soccer.clubs.daily import predict, trends
from soccer.clubs.model import advanced as adv
from soccer.clubs.model import features as feat


def _history(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def two_matches():
    return _history([
        {"league": "epl", "season": "2025-26", "date": "2025-09-01",
         "home_team": "Alpha FC", "away_team": "Beta FC"},
        {"league": "epl", "season": "2025-26", "date": "2025-09-08",
         "home_team": "Beta FC", "away_team": "Alpha FC"},
    ])


class TestMetricCatalogue:
    def test_every_metric_names_two_distinct_columns(self):
        for key, home_col, away_col, label, group, higher in predict.SIDE_METRICS:
            assert home_col != away_col, key
            assert label and group
            assert higher in (True, False, None), key

    def test_keys_are_unique(self):
        keys = [m[0] for m in predict.SIDE_METRICS]
        assert len(keys) == len(set(keys))

    def test_side_columns_cover_every_metric(self):
        assert len(predict.SIDE_COLUMNS) == 2 * len(predict.SIDE_METRICS)

    def test_wage_is_withheld_while_no_wage_source_exists(self):
        """`wage_z` is 0.0 on every row because nothing fills it. Published
        beside a club that reads as "average wage bill", so the card drops
        it until a real wage upload lands."""
        published = {m[0] for m in predict.published_metrics()}
        if feat.wages_available():
            assert "wage_z" in published
        else:
            assert "wage_z" not in published
        # Nothing else is ever withheld.
        withheld = {m[0] for m in predict.SIDE_METRICS} - published
        assert withheld <= {"wage_z"}


class TestSideStats:
    def test_reads_both_sides(self):
        row = {"elo_home_pre": 1600.0, "elo_away_pre": 1500.0,
               "home_xg_for_ewm": 1.8, "away_xg_for_ewm": 1.2}
        out = predict.side_stats(row)
        assert out["home"]["elo"] == 1600.0
        assert out["away"]["elo"] == 1500.0
        assert out["home"]["xg_for_ewm"] == 1.8
        assert out["away"]["xg_for_ewm"] == 1.2

    def test_missing_reading_is_absent_not_zero(self):
        """A club short of the warm-up minimum has no number. Publishing 0
        would claim it is exactly average at something nobody measured."""
        row = {"elo_home_pre": 1600.0, "elo_away_pre": 1500.0,
               "home_xg_for_ewm": float("nan")}
        out = predict.side_stats(row)
        assert "xg_for_ewm" not in out["home"]
        assert "xg_for_ewm" not in out["away"]
        assert out["home"]["elo"] == 1600.0

    def test_unknown_columns_are_skipped(self):
        out = predict.side_stats({"elo_home_pre": 1500.0, "elo_away_pre": 1500.0})
        assert set(out) == {"home", "away"}
        assert out["home"] == {"elo": 1500.0}


class TestKeepSides:
    def test_advanced_adds_side_columns_only_when_asked(self, two_matches):
        metrics = adv.match_metrics()
        plain = adv.attach_advanced(two_matches, metrics)
        wide = adv.attach_advanced(two_matches, metrics, keep_sides=True)
        # The differentials are identical either way ...
        for col in adv.ALL_ADVANCED:
            assert col in plain and col in wide
        # ... and keep_sides only ever adds columns.
        assert set(plain.columns) < set(wide.columns)
        added = set(wide.columns) - set(plain.columns)
        assert added
        assert all(c.startswith(("home_", "away_")) for c in added)

    def test_features_keep_sides_only_adds_columns(self, two_matches):
        plain = feat.attach_features(two_matches)
        wide = feat.attach_features(two_matches, keep_sides=True)
        for col in feat.ALL_FEATURES:
            assert col in plain and col in wide
            # The differential itself is unchanged by keeping the sides.
            assert plain[col].fillna(0).tolist() == wide[col].fillna(0).tolist()
        assert set(plain.columns) <= set(wide.columns)

    def test_keep_sides_never_admits_an_outcome_column(self, two_matches):
        """The columns `keep_sides=True` adds are enumerated, not matched
        on a `home_`/`away_` prefix, and this is why.

        The replay history carries `home_score` / `away_score` beside the
        features. A prefix scan sweeps them into the model and the holdout
        comes back at log loss 0.39 with 95% accuracy — which is not a
        good model, it is the final score being handed to the classifier.
        This asserts the enumerated lists cannot drift into that.
        """
        metrics = adv.match_metrics()
        wide = adv.attach_advanced(
            feat.attach_features(two_matches, keep_sides=True),
            metrics, keep_sides=True)
        added = set(adv.ALL_ADVANCED_SIDES) | set(feat.SIDE_FEATURES) \
            | set(feat.SIDE_RAW_FEATURES)
        for col in added:
            assert "score" not in col, col
        assert not (added & {"home_score", "away_score", "outcome"})
        for col in ("home_xg_for_ewm", "away_xpts_for_ewm", "home_value_z",
                    "home_squad_value_eur_m"):
            assert col in wide.columns, col

    def test_the_plain_path_still_has_no_side_columns(self, two_matches):
        """`keep_sides` defaults to off, and off must stay clean — the
        raw euro columns joined for the per-side path must not leak in as
        `*_x` / `*_y` suffixes from the differential's own merge."""
        plain = adv.attach_advanced(feat.attach_features(two_matches),
                                    adv.match_metrics())
        for col in plain.columns:
            assert not col.endswith(("_x", "_y")), col
            assert "eur_m" not in col, col
        for col in ("home_xg_for_ewm", "home_value_z", "home_squad_value_eur_m"):
            assert col not in plain.columns, col


class TestJoinSafety:
    def test_attaching_features_never_multiplies_rows(self):
        """Both economics joins are left merges on (league, season, club).
        A duplicate key in the source table would silently fan one match
        out into several training rows — the kind of corruption that
        shows up as a suspiciously good holdout rather than an error."""
        h = pd.DataFrame([{
            "league": "epl", "season": "2025-26", "date": "2025-09-01",
            "home_team": "Manchester City FC", "away_team": "Luton Town FC",
        }] * 5)
        assert len(feat.attach_features(h)) == 5
        assert len(feat.attach_features(h, keep_sides=True)) == 5

    def test_the_economics_tables_have_unique_keys(self):
        for table in (feat._load_value_z(), feat._load_transfer_z()):
            assert not table.duplicated(["league", "season", "club"]).any()


class TestFrameParity:
    def test_the_feature_set_is_buildable_from_the_attach_chain(self, two_matches):
        """Every column in FEATURES has to come out of the same attach
        chain both `model/train.py` and `daily/state.py` run.

        These two build the training frame separately — one offline, one
        refit in-run every day — and they drifted apart the moment
        FEATURES grew per-side columns: the offline builder was updated,
        the daily one was not, and the pipeline died on a 58-column
        KeyError at `model.fit`. Non-feature columns are exempt; the point
        is that nothing in FEATURES can be missing.
        """
        from soccer.clubs.model.train import FEATURES, attach_context
        frame = attach_context(adv.attach_advanced(
            feat.attach_features(two_matches, keep_sides=True),
            adv.match_metrics(), keep_sides=True))
        # Exempt the columns that enter from elsewhere: the Elo ratings
        # come off the replay itself, upstream of this chain, and the
        # xg/shots form columns from their own attach step. Everything
        # else has to be here.
        from soccer.clubs.model.shots import SHOT_FEATURES
        from soccer.clubs.model.train import RAW_ELO
        from soccer.clubs.model.xg import XG_FEATURES
        elsewhere = set(XG_FEATURES) | set(SHOT_FEATURES) | set(RAW_ELO) | {"elo_gap"}
        expected = [c for c in FEATURES if c not in elsewhere]
        missing = [c for c in expected if c not in frame.columns]
        assert missing == [], missing


class TestPersistedSchema:
    def test_the_graded_csv_keeps_its_columns(self, tmp_path, monkeypatch):
        """`grade.py` reads this file back and every past prediction lives
        in it, so widening the frame for the site must not widen the CSV."""
        monkeypatch.setattr(predict, "PREDICTIONS_DIR", tmp_path)
        wide = pd.DataFrame([{
            **{c: 0 for c in predict.SLATE_COLUMNS},
            "home_xg_for_ewm": 1.8, "away_xg_for_ewm": 1.2,
        }])
        predict.persist_slate(wide, "2026-09-20")
        written = pd.read_csv(tmp_path / "slate_2026-09-20.csv")
        assert list(written.columns) == predict.SLATE_COLUMNS


class TestWeekTrends:
    @staticmethod
    def _slate():
        return pd.DataFrame([{
            "date": "2026-09-20", "league": "epl", "season": "2026-27",
            "home_team": "Alpha FC", "away_team": "Beta FC",
            "home_xg_for_ewm": 2.0, "away_xg_for_ewm": 1.0,
            "home_xg_against_ewm": 1.0, "away_xg_against_ewm": 2.0,
        }])

    @staticmethod
    def _matches():
        """Two clubs with a season of history each, at known levels."""
        rows = []
        for i in range(30):
            rows.append({
                "league": "epl", "season": "2025-26",
                "date": f"2025-{(i % 9) + 1:02d}-{(i % 27) + 1:02d}",
                "match_id": f"m{i}",
                "home_team": "Alpha FC", "away_team": "Beta FC",
            })
        m = pd.DataFrame(rows)
        for metric in adv.METRICS:
            m[f"{metric}_home"] = 1.0
            m[f"{metric}_away"] = 1.0
        return m

    def test_baseline_uses_only_completed_earlier_seasons(self):
        base = trends.club_baselines("2026-27", matches=self._matches())
        assert set(base["team"]) == {"Alpha FC", "Beta FC"}
        # Nothing from 2026-27 can enter a baseline for 2026-27.
        later = trends.club_baselines("2025-26", matches=self._matches())
        assert later.empty

    def test_clubs_short_of_history_are_skipped(self):
        thin = self._matches().head(3)
        out = trends.week_trends(self._slate(), "2026-27", matches=thin)
        assert out["clubs"] == 0
        assert out["aggregate"] == []

    def test_direction_is_about_quality_not_magnitude(self):
        """Conceding more xG is a bigger number and a worse side. The
        report has to separate the two or it reads backwards."""
        out = trends.week_trends(self._slate(), "2026-27",
                                 matches=self._matches())
        by_metric = {a["metric"]: a for a in out["aggregate"]}
        created = by_metric["xG created"]
        conceded = by_metric["xG conceded"]
        # Both sides average 1.5 against a baseline of 1.0, so both moved up.
        assert created["delta"] > 0 and conceded["delta"] > 0
        assert created["better"] is True
        assert conceded["better"] is False

    def test_a_move_too_small_to_print_is_reported_level(self):
        rows = trends.week_trends(self._slate(), "2026-27",
                                  matches=self._matches())["aggregate"]
        for a in rows:
            if a["delta"] == 0:
                assert a["level"] is True
            else:
                assert a["level"] is False

    def test_each_metric_publishes_at_its_own_precision(self):
        for _form, _raw, label, _better, decimals in trends.TREND_METRICS:
            assert decimals in (2, 3), label
        by_key = {m[0]: m[4] for m in trends.TREND_METRICS}
        # Metrics living between 0 and 1 would round to "+0.00" at two.
        assert by_key["xg_per_shot_ewm"] == 3
        assert by_key["deep_share_ewm"] == 3

    def test_movers_are_ranked_by_scaled_gap(self):
        out = trends.week_trends(self._slate(), "2026-27",
                                 matches=self._matches())
        zs = [abs(m["z"]) for m in out["movers"] if m["z"] is not None]
        assert zs == sorted(zs, reverse=True)
        assert len(out["movers"]) <= trends.TOP_MOVERS

    def test_empty_slate_is_handled(self):
        out = trends.week_trends(pd.DataFrame(), "2026-27")
        assert out["fixtures"] == 0
        assert out["aggregate"] == [] and out["movers"] == []
