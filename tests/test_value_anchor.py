"""Tests for the squad-value Elo anchor (soccer/clubs/model/value_anchor.py)."""

import math

import pandas as pd
import pytest

from soccer.clubs.model import value_anchor


def _ratings(rows: list[tuple[str, str, float]]) -> dict:
    """[(league, team, elo), ...] -> the ratings_payload() shape."""
    out: dict = {}
    for league, team, elo in rows:
        out.setdefault(league, {"clubs": []})["clubs"].append({"team": team, "elo": elo})
    return out


def _values(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """[(league, club, season, value), ...] -> load_market_values_raw()'s shape."""
    return pd.DataFrame(rows, columns=["league", "club", "season", "squad_value_eur_m"])


def _synthetic(n: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """n clubs with an exact log-linear elo/value relationship, split
    across two glued leagues — enough to fit a clean line with no noise."""
    import random

    rng = random.Random(seed)
    ratings_rows, value_rows = [], []
    for i in range(n):
        league = "epl" if i % 2 == 0 else "la_liga"
        value = 5.0 + i * 2.0
        elo = 1200 + 50 * math.log(value)  # exact relationship, no noise
        ratings_rows.append((league, f"Club{i}", elo))
        value_rows.append((league, f"Club{i}", "2026", value))
    return _ratings(ratings_rows), _values(value_rows)


class TestFit:
    def test_too_few_clubs_returns_none(self):
        ratings, values = _synthetic(value_anchor.MIN_FIT_CLUBS - 1)
        assert value_anchor.fit_glued_value_elo(ratings, values, ["epl", "la_liga"]) is None

    def test_empty_values_returns_none(self):
        ratings, _ = _synthetic(50)
        empty = pd.DataFrame(columns=["league", "club", "season", "squad_value_eur_m"])
        assert value_anchor.fit_glued_value_elo(ratings, empty, ["epl", "la_liga"]) is None

    def test_recovers_exact_line_with_no_noise(self):
        ratings, values = _synthetic(60)
        fit = value_anchor.fit_glued_value_elo(ratings, values, ["epl", "la_liga"])
        assert fit is not None
        assert fit.n_clubs == 60
        assert fit.intercept == pytest.approx(1200, abs=1)
        assert fit.slope == pytest.approx(50, abs=1)
        assert fit.r2 == pytest.approx(1.0, abs=1e-6)
        assert fit.residual_std_elo == pytest.approx(0.0, abs=1e-6)

    def test_ignores_leagues_outside_the_glued_set(self):
        # Extra clubs in a league not passed as "glued" must not enter the fit.
        ratings, values = _synthetic(60)
        noise_ratings, noise_values = _synthetic(60, seed=1)
        for league, table in noise_ratings.items():
            for c in table["clubs"]:
                c["team"] = "Other_" + c["team"]
        noise_values["club"] = "Other_" + noise_values["club"]
        noise_values["league"] = "serie_a"  # not in the glued set passed below
        for league, table in noise_ratings.items():
            ratings.setdefault(league, {"clubs": []})["clubs"] += table["clubs"]
        values = pd.concat([values, noise_values], ignore_index=True)

        fit = value_anchor.fit_glued_value_elo(ratings, values, ["epl", "la_liga"])
        assert fit.n_clubs == 60  # only the real glued-league clubs

    def test_takes_each_clubs_most_recent_season_only(self):
        ratings, values = _synthetic(60)
        # Duplicate every row into a stale earlier season with a wildly
        # different value; if the fit didn't dedupe by latest season this
        # would corrupt the line.
        stale = values.copy()
        stale["season"] = "2020"
        stale["squad_value_eur_m"] = 999.0
        fit = value_anchor.fit_glued_value_elo(ratings, pd.concat([stale, values]), ["epl", "la_liga"])
        assert fit.n_clubs == 60
        assert fit.slope == pytest.approx(50, abs=1)


class TestAnchor:
    def test_anchor_elo_applies_the_line(self):
        fit = value_anchor.ValueEloFit(intercept=1200, slope=50, n_clubs=60, r2=1.0, residual_std_elo=0.0)
        assert value_anchor.anchor_elo(fit, math.e) == pytest.approx(1250)

    def test_anchor_elo_none_for_missing_or_nonpositive_value(self):
        fit = value_anchor.ValueEloFit(intercept=1200, slope=50, n_clubs=60, r2=1.0, residual_std_elo=0.0)
        assert value_anchor.anchor_elo(fit, None) is None
        assert value_anchor.anchor_elo(fit, 0) is None
        assert value_anchor.anchor_elo(fit, -5) is None

    def test_anchor_clubs_drops_unusable_values(self):
        fit = value_anchor.ValueEloFit(intercept=1200, slope=50, n_clubs=60, r2=1.0, residual_std_elo=0.0)
        out = value_anchor.anchor_clubs(fit, [math.e, None, 0, math.e ** 2])
        assert len(out) == 2
        assert out[0] == pytest.approx(1250)
        assert out[1] == pytest.approx(1300)
