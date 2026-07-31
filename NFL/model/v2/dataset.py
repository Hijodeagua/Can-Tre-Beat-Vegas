"""Game-level feature builder for the v2 NFL model.

Everything here is derived from the committed nflverse schedule
(``data/schedules/nflverse_games.csv``), which carries every game from 1999
through the current season with scores, closing spread, closing total,
moneylines, rest days, roof/surface, weather and starting QBs. That makes it a
far better spine than the old per-team box-score CSV, which stopped at 2025
week 3 and never saw the season we are trying to learn from.

One row per game, from the **home team's perspective**. Three targets:

- ``home_win``   — home team wins outright
- ``home_cover`` — home team beats the closing spread (pushes dropped)
- ``over``       — combined score clears the closing total (pushes dropped)

Every rolling/expanding feature is shifted so it only sees games that had
already been played when the modelled game kicked off.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .elo import compute_elo

REPO_ROOT = Path(__file__).resolve().parents[3]
GAMES_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"

ROLL_WINDOW = 5
MIN_SEASON = 2002  # first season of the current 32-team, 8-division alignment

# nflverse abbrev -> stadium (lat, lon). Relocated franchises keep their
# historical city so travel distance stays honest for old seasons.
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7868),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2578), "NE": (42.0909, -71.2643),
    "NO": (29.9509, -90.0815), "NYG": (40.8136, -74.0740),
    "NYJ": (40.8136, -74.0740), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9694), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
    # defunct / pre-relocation
    "OAK": (37.7516, -122.2005), "SD": (32.7831, -117.1196),
    "STL": (38.6328, -90.1885),
}

INDOOR_ROOFS = {"dome", "closed"}
DOME_TEMP_F = 68.0

FEATURE_COLS = [
    # market
    "spread_line", "total_line", "market_home_prob", "market_vig",
    # ratings
    "elo_diff", "elo_home_prob", "elo_spread", "elo_vs_spread",
    # recent form (home minus away, plus the raw sides)
    "home_roll_margin", "away_roll_margin", "roll_margin_diff",
    "home_roll_pf", "away_roll_pf", "home_roll_pa", "away_roll_pa",
    "home_roll_ats", "away_roll_ats", "roll_ats_diff",
    "home_roll_total_vs_line", "away_roll_total_vs_line",
    "home_std_margin", "away_std_margin",
    # season to date
    "home_winpct_std", "away_winpct_std", "winpct_diff",
    "home_ppg_diff_std", "away_ppg_diff_std",
    # schedule / context
    "week", "home_rest", "away_rest", "rest_diff",
    "home_short_week", "away_short_week", "home_off_bye", "away_off_bye",
    "div_game", "is_neutral", "is_primetime", "is_playoff",
    "travel_miles", "indoor", "temp", "wind",
    "home_qb_change", "away_qb_change",
]


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * asin(sqrt(h))


def load_games(path: Path | str = GAMES_PATH, min_season: int = MIN_SEASON) -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g[g["season"] >= min_season].copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["gameday", "game_type", "home_team"]).reset_index(drop=True)


def _american_to_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def add_market_features(g: pd.DataFrame) -> pd.DataFrame:
    """No-vig home win probability from the closing moneyline.

    ~23% of games have no moneyline in the schedule file; those fall back to a
    logistic map of the closing spread, which is a very good stand-in.
    """
    g = g.copy()
    hp = pd.Series(_american_to_prob(g["home_moneyline"]), index=g.index)
    ap = pd.Series(_american_to_prob(g["away_moneyline"]), index=g.index)
    book = hp + ap
    g["market_vig"] = book - 1.0
    g["market_home_prob"] = hp / book

    # Spread -> probability fallback. 0.145 logit-per-point is the standard
    # NFL conversion and lands within a point of the moneyline where both exist.
    spread_prob = 1.0 / (1.0 + np.exp(-0.145 * g["spread_line"]))
    g["market_home_prob"] = g["market_home_prob"].fillna(pd.Series(spread_prob, index=g.index))
    g["market_vig"] = g["market_vig"].fillna(g["market_vig"].median())
    return g


def to_team_rows(g: pd.DataFrame) -> pd.DataFrame:
    """Long form: two rows per game, one per team, with results and lines."""
    base = ["game_id", "season", "week", "gameday", "game_type"]
    home = g[base + ["home_team", "away_team", "home_score", "away_score",
                     "spread_line", "total_line", "home_qb_name"]].copy()
    home.columns = base + ["team", "opp", "pf", "pa", "spread_line", "total_line", "qb"]
    home["is_home"] = 1
    home["team_spread"] = home["spread_line"]

    away = g[base + ["away_team", "home_team", "away_score", "home_score",
                     "spread_line", "total_line", "away_qb_name"]].copy()
    away.columns = base + ["team", "opp", "pf", "pa", "spread_line", "total_line", "qb"]
    away["is_home"] = 0
    away["team_spread"] = -away["spread_line"]

    t = pd.concat([home, away], ignore_index=True)
    t["margin"] = t["pf"] - t["pa"]
    # Positive = beat the number by that many points.
    t["ats_margin"] = t["margin"] - t["team_spread"]
    t["total_vs_line"] = (t["pf"] + t["pa"]) - t["total_line"]
    t["won"] = np.where(t["margin"] > 0, 1.0, np.where(t["margin"] < 0, 0.0, 0.5))
    return t.sort_values(["team", "gameday", "game_type"]).reset_index(drop=True)


def add_rolling_form(t: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Trailing form (crosses season boundaries) + season-to-date (resets)."""
    t = t.sort_values(["team", "gameday", "game_type"]).copy()
    grp = t.groupby("team", sort=False)

    for src, dst in [("margin", "roll_margin"), ("pf", "roll_pf"), ("pa", "roll_pa"),
                     ("ats_margin", "roll_ats"), ("total_vs_line", "roll_total_vs_line")]:
        t[dst] = grp[src].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).mean()
        )
    t["std_margin"] = grp["margin"].transform(
        lambda s: s.shift(1).rolling(window * 2, min_periods=4).std()
    )

    season_grp = t.groupby(["team", "season"], sort=False)
    t["winpct_std"] = season_grp["won"].transform(lambda s: s.shift(1).expanding().mean())
    t["ppg_diff_std"] = season_grp["margin"].transform(lambda s: s.shift(1).expanding().mean())

    # QB change: this game's starter differs from the previous game's starter.
    prev_qb = grp["qb"].shift(1)
    t["qb_change"] = ((t["qb"] != prev_qb) & prev_qb.notna() & t["qb"].notna()).astype(int)
    return t


ROLL_OUT_COLS = [
    "roll_margin", "roll_pf", "roll_pa", "roll_ats", "roll_total_vs_line",
    "std_margin", "winpct_std", "ppg_diff_std", "qb_change",
]


def merge_form_back(g: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Pivot the per-team form columns back onto the one-row-per-game frame."""
    keys = ["game_id", "team"]
    form = t[keys + ROLL_OUT_COLS]

    home = form.rename(columns={"team": "home_team",
                                **{c: f"home_{c}" for c in ROLL_OUT_COLS}})
    away = form.rename(columns={"team": "away_team",
                                **{c: f"away_{c}" for c in ROLL_OUT_COLS}})
    out = g.merge(home, on=["game_id", "home_team"], how="left")
    out = out.merge(away, on=["game_id", "away_team"], how="left")
    return out.rename(columns={"home_qb_change": "home_qb_change",
                               "away_qb_change": "away_qb_change"})


def add_context_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["is_neutral"] = (g["location"].fillna("Home") != "Home").astype(int)
    g["is_playoff"] = (g["game_type"] != "REG").astype(int)

    hour = pd.to_datetime(g["gametime"], format="%H:%M", errors="coerce").dt.hour
    g["is_primetime"] = ((hour >= 19) | g["weekday"].isin(["Thursday", "Monday"])).astype(int)

    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["home_short_week"] = (g["home_rest"] <= 4).astype(int)
    g["away_short_week"] = (g["away_rest"] <= 4).astype(int)
    g["home_off_bye"] = (g["home_rest"] >= 13).astype(int)
    g["away_off_bye"] = (g["away_rest"] >= 13).astype(int)

    coords = TEAM_COORDS
    g["travel_miles"] = [
        haversine_miles(coords[h], coords[a])
        if (h in coords and a in coords and not neutral) else np.nan
        for h, a, neutral in zip(g["home_team"], g["away_team"], g["is_neutral"].astype(bool))
    ]
    g["travel_miles"] = g["travel_miles"].fillna(g["travel_miles"].median())

    g["indoor"] = g["roof"].isin(INDOOR_ROOFS).astype(int)
    g["temp"] = pd.to_numeric(g["temp"], errors="coerce")
    g["wind"] = pd.to_numeric(g["wind"], errors="coerce")
    g.loc[g["indoor"] == 1, "temp"] = g.loc[g["indoor"] == 1, "temp"].fillna(DOME_TEMP_F)
    g.loc[g["indoor"] == 1, "wind"] = g.loc[g["indoor"] == 1, "wind"].fillna(0.0)
    # Outdoor games with no reading (mostly future games) stay NaN — LightGBM
    # handles missing natively and the logistic baselines impute at fit time.

    g["roll_margin_diff"] = g["home_roll_margin"] - g["away_roll_margin"]
    g["roll_ats_diff"] = g["home_roll_ats"] - g["away_roll_ats"]
    g["winpct_diff"] = g["home_winpct_std"] - g["away_winpct_std"]
    g["elo_vs_spread"] = g["elo_spread"] - g["spread_line"]
    return g


def add_targets(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    margin = g["home_score"] - g["away_score"]
    total = g["home_score"] + g["away_score"]

    g["margin"] = margin
    g["home_win"] = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, np.nan))

    ats = margin - g["spread_line"]
    g["ats_margin_home"] = ats
    g["home_cover"] = np.where(ats > 0, 1.0, np.where(ats < 0, 0.0, np.nan))

    tot = total - g["total_line"]
    g["total_margin"] = tot
    g["over"] = np.where(tot > 0, 1.0, np.where(tot < 0, 0.0, np.nan))
    return g


def build_dataset(path: Path | str = GAMES_PATH, min_season: int = MIN_SEASON,
                  with_squad: bool = False) -> pd.DataFrame:
    """Full feature frame, one row per game, chronologically sorted.

    ``with_squad`` merges the roster-quality columns from ``squad.py`` (draft
    pedigree, honors, QB quality, interim coach). It is opt-in so the models
    trained on the original 45 features keep loading unchanged.
    """
    g = load_games(path, min_season)
    g = compute_elo(g)
    g = add_market_features(g)

    t = add_rolling_form(to_team_rows(g))
    g = merge_form_back(g, t)
    g = add_context_features(g)
    g = add_targets(g)
    if with_squad:
        from .squad import add_squad_features
        g = add_squad_features(g)
    return g.sort_values(["gameday", "game_type", "home_team"]).reset_index(drop=True)


def available_squad_cols(df: pd.DataFrame) -> list[str]:
    """Squad columns present and not entirely missing (honors need the scrape)."""
    from .squad import SQUAD_FEATURE_COLS
    return [c for c in SQUAD_FEATURE_COLS if c in df.columns and df[c].notna().any()]


def feature_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    cols = features if features is not None else FEATURE_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature frame is missing columns: {missing}")
    return df[cols].astype(float)


if __name__ == "__main__":
    d = build_dataset()
    print(f"{len(d)} games, {d['season'].min()}-{d['season'].max()}, "
          f"{len(FEATURE_COLS)} features")
    played = d.dropna(subset=["home_score"])
    print(f"played: {len(played)}  |  unplayed: {len(d) - len(played)}")
    print(played.groupby("season")[["home_win", "home_cover", "over"]].mean().tail(8).to_string())
