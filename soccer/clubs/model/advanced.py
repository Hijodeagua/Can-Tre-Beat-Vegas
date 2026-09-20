"""
Pregame advanced-metric features for the club outcome model: leakage-safe
rolling and exponentially weighted form built from the Understat match
table (xG, npxG, xPts, PPDA, deep completions), the shots table, and the
fixture calendar (rest, congestion, European matches).

Three layers, in order:

1. `match_metrics()` — one row per completed match with both sides'
   metrics. Prefers the processed Understat table; any match it does not
   cover is filled from the legacy `xg_matches.csv` (xG only), so the
   feature layer works on the data that exists today and upgrades in
   place when the fuller feed lands.
2. `team_rows()` — the same matches in team-match long form (two rows a
   match): what the team did (`*_for`) and what it allowed (`*_against`).
3. `attach_advanced(history)` — the pregame feature columns for every
   row of a replay-history table (or a slate of unplayed fixtures). Every
   form column is a shifted statistic over that team's *earlier* rows in
   the same league, so the match being predicted never contributes to its
   own feature. Unplayed fixtures are appended as metric-less rows, which
   is exactly what makes the same code serve the live slate: a row with
   no metrics of its own still sees everything before it.

Two versions of every form statistic ship, because the brief asked for
both and because they disagree in interesting places: a conventional
rolling mean over the last WINDOW league matches, and an exponentially
weighted mean with a HALF_LIFE-match half-life. Both need MIN_MATCHES
earlier matches to be non-null; below that the feature is NaN and the
model's imputer (fit on training rows) fills it — the honest way to say
"not enough data", as opposed to a 0 that pretends to be a neutral
differential.

Home/away attack splits are the team's form in *its own home matches*
(for the home side) or *its own away matches* (for the away side), so
`home_att_vs_away_def` is the home side's home attack against the away
side's away defence — the matchup a fixture actually is.

`deep_share` is deep completions for / (for + allowed): a field-tilt
proxy. Understat does not publish true field tilt, and this is labelled
as the proxy it is.

Rest and congestion come from the whole fixture calendar — league matches
in `results.csv` and European ties in `uefa_results.csv` — so a club that
played in Milan on Wednesday shows three days' rest and a European flag
on Saturday even though the league table never saw Wednesday.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import pregame
from soccer.clubs.data import understat

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHOTS_CSV = DATA_DIR / "shots_matches.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
UEFA_CSV = DATA_DIR / "uefa_results.csv"

WINDOW = 10          # rolling window, league matches (matches xg.py / shots.py)
HALF_LIFE = 5.0      # EWMA half-life, matches
MIN_MATCHES = 5      # both versions are NaN until a club has this many
SPLIT_MIN = 3        # home-only / away-only splits warm up faster
CONGESTION_DAYS = 14
UEFA_DAYS = 7
# A feed that has not covered a club's last match within this many days
# is treated as stale for that club: its form is NaN, not a year-old number.
MAX_AGE_DAYS = 130

METRICS = ["xg", "npxg", "xpts", "ppda", "deep", "shots", "sot", "goals"]

# --- feature groups, for the ablation and for production selection -------
XG_FEATURES = [
    "xg_for_ewm_diff", "xg_against_ewm_diff",
    "npxg_for_ewm_diff", "npxg_against_ewm_diff",
]
XG_ROLLING_FEATURES = [
    "xg_for_r10_diff", "xg_against_r10_diff",
    "npxg_for_r10_diff", "npxg_against_r10_diff",
]
MATCHUP_FEATURES = [
    "home_att_vs_away_def", "away_att_vs_home_def",
    "npxg_home_att_vs_away_def", "npxg_away_att_vs_home_def",
    "xg_per_shot_diff",
]
TERRITORY_FEATURES = [
    "deep_diff", "deep_share_diff", "ppda_diff", "xpts_ewm_diff",
]
REST_FEATURES = [
    "rest_diff", "congestion14_home", "congestion14_away",
    "uefa7_home", "uefa7_away",
]
ALL_ADVANCED = XG_FEATURES + XG_ROLLING_FEATURES + MATCHUP_FEATURES + TERRITORY_FEATURES + REST_FEATURES


# --------------------------------------------------------------------------
# 1. match metrics
# --------------------------------------------------------------------------
def load_shots(path: Path = SHOTS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["league", "date", "home_team", "away_team",
                                     "shots_home", "shots_away", "sot_home", "sot_away"])
    return pd.read_csv(path)


def match_metrics(processed: pd.DataFrame | None = None,
                  legacy_xg: pd.DataFrame | None = None,
                  shots: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per completed match with both sides' metrics.

    Processed Understat rows win; legacy xG rows fill matches the
    processed table lacks (xG only, everything else NaN); shots on target
    join by (league, date, home, away).
    """
    key = ["league", "date", "home_team", "away_team"]
    proc = understat.load_processed() if processed is None else processed
    proc = proc.copy()
    if legacy_xg is None:
        legacy_xg = (pd.read_csv(understat.LEGACY_XG_CSV)
                     if understat.LEGACY_XG_CSV.exists() else pd.DataFrame(columns=key + ["xg_home", "xg_away"]))
    have = set(map(tuple, proc[key].to_numpy())) if len(proc) else set()
    fill = legacy_xg[~legacy_xg[key].apply(tuple, axis=1).isin(have)].copy() if len(legacy_xg) else legacy_xg
    for c in understat.COLUMNS:
        if c not in fill.columns:
            fill[c] = np.nan
    fill["match_id"] = ""
    matches = pd.concat([proc[understat.COLUMNS], fill[understat.COLUMNS]], ignore_index=True)
    matches["source"] = ["understat"] * len(proc) + ["legacy_xg"] * len(fill)

    shots = load_shots() if shots is None else shots
    if len(shots):
        matches = matches.merge(shots[key + ["shots_home", "shots_away", "sot_home", "sot_away"]],
                                on=key, how="left")
    else:
        for c in ("shots_home", "shots_away", "sot_home", "sot_away"):
            matches[c] = np.nan
    matches["ppda_home"] = matches["ppda_att_home"] / matches["ppda_def_home"].replace(0, np.nan)
    matches["ppda_away"] = matches["ppda_att_away"] / matches["ppda_def_away"].replace(0, np.nan)
    return matches.sort_values(["date", "league", "home_team"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. team-match long form
# --------------------------------------------------------------------------
def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: (team, opponent, is_home, *_for, *_against)."""
    sides = []
    for side, other, is_home in (("home", "away", 1), ("away", "home", 0)):
        r = pd.DataFrame({
            "league": matches["league"], "season": matches["season"],
            "date": matches["date"], "match_id": matches["match_id"],
            "team": matches[f"{side}_team"], "opponent": matches[f"{other}_team"],
            "is_home": is_home,
        })
        for m in METRICS:
            r[f"{m}_for"] = matches[f"{m}_{side}"]
            r[f"{m}_against"] = matches[f"{m}_{other}"]
        sides.append(r)
    long = pd.concat(sides, ignore_index=True)
    long["xg_per_shot"] = long["xg_for"] / long["shots_for"].replace(0, np.nan)
    long["deep_share"] = long["deep_for"] / (long["deep_for"] + long["deep_against"]).replace(0, np.nan)
    return long.sort_values(["league", "team", "date"]).reset_index(drop=True)


FORM_COLS = ["xg_for", "xg_against", "npxg_for", "npxg_against", "xpts_for",
             "ppda_for", "deep_for", "deep_against", "deep_share", "xg_per_shot"]
SPLIT_COLS = ["xg_for", "xg_against", "npxg_for", "npxg_against"]


def team_form(long: pd.DataFrame) -> pd.DataFrame:
    """Shifted rolling (r10) and EWMA (ewm) form per (league, team), plus
    home-only / away-only splits per (league, team, is_home), plus the
    date of the team's previous *covered* match for the staleness rule."""
    df = long.sort_values(["league", "team", "date"]).reset_index(drop=True)
    df["_grp"] = df["league"] + "|" + df["team"]
    df = pregame.shifted_rolling(df, "_grp", FORM_COLS, WINDOW, min_periods=MIN_MATCHES, suffix="r10")
    df = pregame.shifted_ewm(df, "_grp", FORM_COLS, HALF_LIFE, min_periods=MIN_MATCHES, suffix="ewm")
    df["_split"] = df["_grp"] + "|" + df["is_home"].astype(str)
    df = df.sort_values(["_split", "date"]).reset_index(drop=True)
    df = pregame.shifted_rolling(df, "_split", SPLIT_COLS, WINDOW, min_periods=SPLIT_MIN, suffix="split")
    df = df.sort_values(["league", "team", "date"]).reset_index(drop=True)
    # Staleness: the date of the previous row that actually carried xG.
    covered_date = df["date"].where(df["xg_for"].notna())
    df["prev_covered"] = covered_date.groupby(df["_grp"], sort=False).transform(
        lambda s: s.shift(1).ffill())
    return df.drop(columns=["_grp", "_split"])


# --------------------------------------------------------------------------
# 3. pregame features for a history table or a slate
# --------------------------------------------------------------------------
def _calendar() -> pd.DataFrame:
    """Every dated appearance of every club — league results (played or
    scheduled) and European ties — for rest and congestion."""
    frames = []
    if RESULTS_CSV.exists():
        r = pd.read_csv(RESULTS_CSV, usecols=["date", "league", "home_team", "away_team",
                                              "home_score", "away_score"])
        played = r["home_score"].notna()
        for side in ("home_team", "away_team"):
            frames.append(pd.DataFrame({"team": r[side], "league": r["league"],
                                        "date": r["date"], "played": played, "uefa": False}))
    if UEFA_CSV.exists():
        u = pd.read_csv(UEFA_CSV, usecols=["date", "home_team", "home_league",
                                           "away_team", "away_league", "home_score"])
        played = u["home_score"].notna()
        for side in ("home", "away"):
            frames.append(pd.DataFrame({"team": u[f"{side}_team"], "league": u[f"{side}_league"],
                                        "date": u["date"], "played": played, "uefa": True}))
    cal = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["team", "league", "date", "played", "uefa"])
    cal = cal[cal["league"].notna() & (cal["league"] != "")]
    return cal.drop_duplicates(["team", "date"]).sort_values(["team", "date"]).reset_index(drop=True)


def rest_and_congestion(rows: pd.DataFrame, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each (team, date) in `rows`: days since the club's previous
    played match, matches in the previous CONGESTION_DAYS, and whether it
    played in Europe in the previous UEFA_DAYS. All strictly before the
    date, from the whole calendar (league + UEFA)."""
    cal = _calendar() if calendar is None else calendar
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    day = np.timedelta64(1, "D")
    # Per team: sorted date arrays, so every lookup is a searchsorted.
    idx = {}
    for team, g in cal.groupby("team", sort=False):
        g = g.sort_values("date")
        all_d = g["date"].to_numpy()
        idx[team] = (all_d,
                     g.loc[g["played"].astype(bool), "date"].to_numpy(),
                     g.loc[g["uefa"].astype(bool), "date"].to_numpy())
    out = np.full((len(rows), 3), np.nan)
    dates = pd.to_datetime(rows["date"]).to_numpy()
    for i, (team, d) in enumerate(zip(rows["team"], dates)):
        hit = idx.get(team)
        if hit is None:
            out[i] = (np.nan, 0, 0); continue
        all_d, played_d, uefa_d = hit
        n_before = np.searchsorted(all_d, d, side="left")
        n_played = np.searchsorted(played_d, d, side="left")
        rest = (d - played_d[n_played - 1]) / day if n_played else np.nan
        n_window = n_before - np.searchsorted(all_d, d - CONGESTION_DAYS * day, side="left")
        n_uefa = (np.searchsorted(uefa_d, d, side="left")
                  - np.searchsorted(uefa_d, d - UEFA_DAYS * day, side="left"))
        out[i] = (rest, n_window, int(n_uefa > 0))
    return pd.DataFrame(out, columns=["rest", "congestion14", "uefa7"], index=rows.index)


# The per-side form columns `keep_sides=True` leaves on the frame. Listed
# explicitly rather than derived from a prefix scan: the replay history
# also carries `home_score` / `away_score`, and a prefix match would sweep
# the match result itself into the feature set.
SIDE_FORM_COLS = [
    "xg_for_r10", "xg_against_r10", "npxg_for_r10", "npxg_against_r10",
    "xpts_for_r10", "ppda_for_r10", "deep_for_r10", "deep_against_r10",
    "deep_share_r10", "xg_per_shot_r10",
    "xg_for_ewm", "xg_against_ewm", "npxg_for_ewm", "npxg_against_ewm",
    "xpts_for_ewm", "ppda_for_ewm", "deep_for_ewm", "deep_against_ewm",
    "deep_share_ewm", "xg_per_shot_ewm",
    "xg_for_split", "xg_against_split", "npxg_for_split", "npxg_against_split",
]
ALL_ADVANCED_SIDES = [f"{side}_{c}" for c in SIDE_FORM_COLS
                      for side in ("home", "away")]


def attach_advanced(history: pd.DataFrame, matches: pd.DataFrame | None = None,
                    calendar: pd.DataFrame | None = None,
                    keep_sides: bool = False) -> pd.DataFrame:
    """Add every advanced pregame column to a table of matches to predict
    (league, date, home_team, away_team). Rows not in the match-metrics
    table (unplayed fixtures, uncovered leagues) still get features from
    the sides' earlier matches; a side with too little or too stale
    coverage gets NaN.

    The model trains on differentials, so by default the per-side values
    are subtracted and thrown away. `keep_sides=True` also keeps them as
    `home_<col>` / `away_<col>` — what a published match card needs, since
    "xG for form +0.4" says nothing about whether that is two good attacks
    or two bad ones. It is off by default so the training frame keeps
    exactly the columns it always had.
    """
    history = history.copy()
    matches = match_metrics() if matches is None else matches

    # Virtual rows for anything the metrics table doesn't cover, so the
    # shifted form has a row to land on.
    key = ["league", "date", "home_team", "away_team"]
    covered = set(map(tuple, matches[key].to_numpy())) if len(matches) else set()
    # A club plays at most once a day, so a history row whose home or
    # away side already has a metrics row that day is the same match under
    # another spelling (an alias the fetcher missed), not an uncovered
    # match: adding a virtual row for it would give that club two rows on
    # one date and break the join. The side that matches keeps its form;
    # the misspelt side reads NaN until the alias is added.
    if len(matches):
        seen = (set(map(tuple, matches[["league", "date", "home_team"]].to_numpy()))
                | set(map(tuple, matches[["league", "date", "away_team"]].to_numpy())))
    else:
        seen = set()
    uncovered = ~history[key].apply(tuple, axis=1).isin(covered)
    side_seen = (history[["league", "date", "home_team"]].apply(tuple, axis=1).isin(seen)
                 | history[["league", "date", "away_team"]].apply(tuple, axis=1).isin(seen))
    extra = history[uncovered & ~side_seen][key].drop_duplicates()
    if len(extra):
        extra = extra.assign(season=history.get("season", pd.Series(index=extra.index)), match_id="")
        for c in understat.COLUMNS:
            if c not in extra.columns:
                extra[c] = np.nan
        for c in ("shots_home", "shots_away", "sot_home", "sot_away", "ppda_home", "ppda_away", "source"):
            extra[c] = np.nan
        matches = pd.concat([matches, extra[matches.columns]], ignore_index=True)

    form = team_form(team_rows(matches))
    form["date_dt"] = pd.to_datetime(form["date"])
    form["prev_dt"] = pd.to_datetime(form["prev_covered"])
    stale = (form["date_dt"] - form["prev_dt"]).dt.days > MAX_AGE_DAYS
    formcols = [c for c in form.columns if c.endswith(("_r10", "_ewm", "_split"))]
    form.loc[stale | form["prev_dt"].isna(), formcols] = np.nan

    def side(prefix: str, team_col: str, is_home: int) -> pd.DataFrame:
        f = form[form["is_home"] == is_home][["league", "date", "team"] + formcols]
        # One row per club-date, whatever the source tables held.
        f = f.drop_duplicates(["league", "date", "team"], keep="first")
        f = f.rename(columns={c: f"{prefix}_{c}" for c in formcols})
        return history[["league", "date", team_col]].merge(
            f, left_on=["league", "date", team_col], right_on=["league", "date", "team"],
            how="left").drop(columns=["team"])

    h = side("h", "home_team", 1)
    a = side("a", "away_team", 0)
    out = history.copy()
    for m in ("xg_for", "xg_against", "npxg_for", "npxg_against"):
        for kind in ("ewm", "r10"):
            out[f"{m}_{kind}_diff"] = h[f"h_{m}_{kind}"].to_numpy() - a[f"a_{m}_{kind}"].to_numpy()
    out["home_att_vs_away_def"] = h["h_xg_for_split"].to_numpy() - a["a_xg_against_split"].to_numpy()
    out["away_att_vs_home_def"] = a["a_xg_for_split"].to_numpy() - h["h_xg_against_split"].to_numpy()
    out["npxg_home_att_vs_away_def"] = h["h_npxg_for_split"].to_numpy() - a["a_npxg_against_split"].to_numpy()
    out["npxg_away_att_vs_home_def"] = a["a_npxg_for_split"].to_numpy() - h["h_npxg_against_split"].to_numpy()
    out["xg_per_shot_diff"] = h["h_xg_per_shot_ewm"].to_numpy() - a["a_xg_per_shot_ewm"].to_numpy()
    out["deep_diff"] = h["h_deep_for_ewm"].to_numpy() - a["a_deep_for_ewm"].to_numpy()
    out["deep_share_diff"] = h["h_deep_share_ewm"].to_numpy() - a["a_deep_share_ewm"].to_numpy()
    out["ppda_diff"] = h["h_ppda_for_ewm"].to_numpy() - a["a_ppda_for_ewm"].to_numpy()
    out["xpts_ewm_diff"] = h["h_xpts_for_ewm"].to_numpy() - a["a_xpts_for_ewm"].to_numpy()

    hr = rest_and_congestion(history.rename(columns={"home_team": "team"})[["team", "date"]], calendar)
    ar = rest_and_congestion(history.rename(columns={"away_team": "team"})[["team", "date"]], calendar)
    out["rest_home"] = hr["rest"].clip(upper=21).to_numpy()
    out["rest_away"] = ar["rest"].clip(upper=21).to_numpy()
    out["rest_diff"] = out["rest_home"] - out["rest_away"]
    out["congestion14_home"] = hr["congestion14"].to_numpy()
    out["congestion14_away"] = ar["congestion14"].to_numpy()
    out["uefa7_home"] = hr["uefa7"].to_numpy()
    out["uefa7_away"] = ar["uefa7"].to_numpy()

    if keep_sides:
        for c in formcols:
            out[f"home_{c}"] = h[f"h_{c}"].to_numpy()
            out[f"away_{c}"] = a[f"a_{c}"].to_numpy()
    return out


def coverage_by_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Per league-season: matches with xG, with npxG, and the source mix —
    the table that says what the model could actually see."""
    m = matches.copy()
    m["season"] = m["season"].fillna(m["date"].str[:4])
    g = m.groupby(["league", "season"])
    return pd.DataFrame({
        "matches": g.size(),
        "with_xg": g["xg_home"].apply(lambda s: int(s.notna().sum())),
        "with_npxg": g["npxg_home"].apply(lambda s: int(s.notna().sum())),
        "with_ppda": g["ppda_home"].apply(lambda s: int(s.notna().sum())),
        "with_shots": g["sot_home"].apply(lambda s: int(s.notna().sum())),
        "last_date": g["date"].max(),
    }).reset_index()
