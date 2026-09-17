"""
Pregame efficiency features for the college football second stage, from
the SportsDataverse weekly team summaries (`data/college_football/team_weeks.csv`,
built by `CFB.data.fetch_weekly`).

The one rule that matters: the snapshot with `through_week == W`
*includes* week W's games, so a week-W game is featured from the W-1
snapshot (`snapshot_before`). Postseason games (the spine restarts `week`
at 1 with `season_type == "postseason"`) use the season's final snapshot,
which predates every bowl. `tests/test_cfb_advanced.py` asserts both.

Early season: the opponent-adjusted columns are NaN for most teams until
week 3 (the publisher's ridge needs games), and week 1 has no snapshot
at all. Every feature is therefore a shrunk blend of the current
season's snapshot and the previous season's final snapshot regressed
halfway to the league mean:

    value = (n · current + PRIOR_GAMES · prior) / (n + PRIOR_GAMES)

with `n` the number of games in the current snapshot (`plays_off / 65`,
capped) and the prior standing in alone before the first snapshot. A
team with neither (a first FBS season, or no team id) reads NaN and the
imputer supplies the training median, which is the Elo-only fallback in
practice: the logistic then leans on `elo_logit`.

Team identity is the ESPN `team_id` the spine carries (`home_id`,
`away_id`) — the same id the weekly file uses — with a name crosswalk
(`team_ids`) only for a spine row that has none.

Feature groups are the ones the ablation compares. Production reads
`PRODUCTION_FEATURES`, set from the evaluation's verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CFB.data.fetch_weekly import load_team_weeks
from CFB.data.teams import canonical

PRIOR_GAMES = 4.0         # pseudo-games of prior-season strength
PRIOR_REGRESSION = 0.5    # prior-season final regressed this far toward league mean
PLAYS_PER_GAME = 65.0
MAX_GAMES = 15.0
TOLERANCE_WEEKS = 2       # aggregates may lag the spine's completed weeks by this much

# (feature stem, offence column, defence column). A defence column is
# what the team allowed; lower is better for every efficiency metric
# except havoc_def, where higher is better.
METRICS = {
    "adj_epa": ("adj_off_epa", "adj_def_epa"),
    "epa": ("EPAplay_off", "EPAplay_def"),
    "success": ("success_off", "success_def"),
    "early_epa": ("early_down_EPA_off", "early_down_EPA_def"),
    "explosive": ("explosive_off", "explosive_def"),
    "havoc": ("havoc_off", "havoc_def"),
    "epa_drive": ("EPAdrive_off", "EPAdrive_def"),
    "drives_game": ("drivesgame_off", "drivesgame_def"),
    "plays_drive": ("playsdrive_off", "playsdrive_def"),
    "yards_drive": ("yardsdrive_off", "yardsdrive_def"),
    "rz_success": ("red_zone_success_off", "red_zone_success_def"),
    "third_success": ("third_down_success_off", "third_down_success_def"),
    "third_dist": ("third_down_distance_off", "third_down_distance_def"),
    "pass_epa": ("EPAplay_off_pass", "EPAplay_def_pass"),
    "rush_epa": ("EPAplay_off_rush", "EPAplay_def_rush"),
}
TENDENCY = ["passrate_off", "rushrate_off"]

ELO = ["elo_logit"]
# net_adj_epa_diff is left out: it is algebraically adj_epa_matchup_net.
CORE = ["home_adj_epa_off", "away_adj_epa_off", "home_adj_epa_def", "away_adj_epa_def",
        "adj_epa_matchup_net"]
SUCCESS = ["home_success_off", "away_success_off", "home_success_def", "away_success_def",
           "success_matchup_net"]
EARLY_EXPLOSIVE = ["early_epa_matchup_net", "explosive_matchup_net",
                   "home_havoc_def", "away_havoc_def", "home_havoc_off", "away_havoc_off"]
DRIVE = ["epa_drive_matchup_net", "home_drives_game_off", "away_drives_game_off",
         "home_plays_drive_off", "away_plays_drive_off",
         "home_yards_drive_off", "away_yards_drive_off"]
SITUATIONAL = ["rz_success_matchup_net", "third_success_matchup_net",
               "home_third_dist_off", "away_third_dist_off",
               "home_passrate_off", "away_passrate_off"]
SPLITS = ["pass_epa_matchup_net", "rush_epa_matchup_net"]

GROUPS = {"elo": ELO, "core": CORE, "success": SUCCESS, "early_explosive": EARLY_EXPLOSIVE,
          "drive": DRIVE, "situational": SITUATIONAL, "splits": SPLITS}

# What the daily pipeline ships (verdict of `eval_advanced`, see
# docs/ADVANCED_METRICS.md): Elo plus the publisher's opponent-adjusted
# EPA per play, offence and defence, both sides, and the matchup net.
# Clean 2024-2025 test: log loss 0.49768 -> 0.49285 overall (+1.87 SE),
# 0.55358 -> 0.54598 on FBS-vs-FBS games (+2.60 SE). Adding success,
# situational or split groups on top gains under 0.001 and the greedy
# combined set chosen on 2023 is worse than core alone on the test
# window, so the compact set ships.
PRODUCTION_FEATURES: list[str] = ELO + CORE
PRODUCTION_C = 1.0


def all_features() -> list[str]:
    return [f for g in GROUPS.values() for f in g]


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------
def real_weeks(tw: pd.DataFrame) -> pd.Series:
    """Per season, the last `through_week` whose snapshot carries new
    games. The publisher's live-season file has rows for every week with
    the totals frozen at the last played one; those copies are harmless
    for the join (still pre-game) but must not count as coverage."""
    plays = tw.groupby(["season", "through_week"])["plays_off"].sum().sort_index()
    out = {}
    for season, s in plays.groupby(level=0):
        s = s.droplevel(0)
        grew = s.diff().fillna(s.iloc[0]) > 0
        out[season] = int(s.index[grew].max()) if grew.any() else 0
    return pd.Series(out, name="last_real_week")


def snapshot_before(tw: pd.DataFrame, season: int, week: int,
                    postseason: bool = False) -> pd.DataFrame:
    """The latest snapshot a game in (season, week) may see: through_week
    == week − 1 for the regular season (falling back to the latest earlier
    week the table has), the season's final snapshot for the postseason.
    Empty before week 2."""
    s = tw[tw["season"] == season]
    if s.empty:
        return s
    if postseason:
        cut = int(s["through_week"].max())
    else:
        earlier = s.loc[s["through_week"] < week, "through_week"]
        if earlier.empty:
            return s.iloc[0:0]
        cut = int(earlier.max())
    return s[s["through_week"] == cut]


def prior_final(tw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Previous season's final snapshot, every metric regressed
    PRIOR_REGRESSION toward that season's league mean."""
    s = tw[tw["season"] == season - 1]
    if s.empty:
        return s
    final = s[s["through_week"] == s["through_week"].max()].copy()
    cols = [c for pair in METRICS.values() for c in pair] + TENDENCY
    mean = final[cols].mean()
    final[cols] = final[cols] * (1 - PRIOR_REGRESSION) + mean * PRIOR_REGRESSION
    return final


def team_strength(tw: pd.DataFrame, season: int, week: int,
                  postseason: bool = False) -> pd.DataFrame:
    """One row per team_id with every metric shrunk between the current
    snapshot and the regressed prior-season final, for games in
    (season, week). Index: team_id."""
    cols = [c for pair in METRICS.values() for c in pair] + TENDENCY
    cur = snapshot_before(tw, season, week, postseason).set_index("team_id")
    prior = prior_final(tw, season).set_index("team_id")
    ids = cur.index.union(prior.index)
    out = pd.DataFrame(index=ids)
    n = (cur["plays_off"] / PLAYS_PER_GAME).clip(upper=MAX_GAMES).reindex(ids).fillna(0.0) \
        if len(cur) else pd.Series(0.0, index=ids)
    for c in cols:
        c_cur = cur[c].reindex(ids) if len(cur) else pd.Series(np.nan, index=ids)
        c_pri = prior[c].reindex(ids) if len(prior) else pd.Series(np.nan, index=ids)
        w_cur = n.where(c_cur.notna(), 0.0)
        w_pri = pd.Series(PRIOR_GAMES, index=ids).where(c_pri.notna(), 0.0)
        num = c_cur.fillna(0.0) * w_cur + c_pri.fillna(0.0) * w_pri
        den = w_cur + w_pri
        out[c] = (num / den.replace(0.0, np.nan))
    out["games_seen"] = n
    return out


# --------------------------------------------------------------------------
# game table
# --------------------------------------------------------------------------
def team_ids(tw: pd.DataFrame) -> dict[str, int]:
    """Name -> team_id crosswalk from the weekly table (latest season
    wins), for spine rows that carry no id. Names pass through the
    spine's alias map first."""
    latest = tw.sort_values("season").drop_duplicates("team", keep="last")
    return {canonical(t): int(i) for t, i in zip(latest["team"], latest["team_id"])}


def _features_from(strength: pd.DataFrame, home_id, away_id) -> dict:
    row = {}
    ok_h = home_id in strength.index
    ok_a = away_id in strength.index
    for stem, (off, dfn) in METRICS.items():
        h_off = float(strength.at[home_id, off]) if ok_h else np.nan
        h_def = float(strength.at[home_id, dfn]) if ok_h else np.nan
        a_off = float(strength.at[away_id, off]) if ok_a else np.nan
        a_def = float(strength.at[away_id, dfn]) if ok_a else np.nan
        row[f"home_{stem}_off"], row[f"home_{stem}_def"] = h_off, h_def
        row[f"away_{stem}_off"], row[f"away_{stem}_def"] = a_off, a_def
        # Offence vs the defence it faces: for havoc the defence column is
        # the team's own havoc *caused*, so the matchup is off − opp havoc.
        if stem == "havoc":
            row[f"{stem}_matchup_net"] = (h_off - a_def) - (a_off - h_def)
        else:
            row[f"{stem}_matchup_net"] = (h_off + a_def) - (a_off + h_def)
    row["net_adj_epa_diff"] = (row["home_adj_epa_off"] - row["home_adj_epa_def"]) \
        - (row["away_adj_epa_off"] - row["away_adj_epa_def"])
    for t in TENDENCY:
        row[f"home_{t}"] = float(strength.at[home_id, t]) if ok_h else np.nan
        row[f"away_{t}"] = float(strength.at[away_id, t]) if ok_a else np.nan
    row["home_games_seen"] = float(strength.at[home_id, "games_seen"]) if ok_h else 0.0
    row["away_games_seen"] = float(strength.at[away_id, "games_seen"]) if ok_a else 0.0
    return row


def build_game_table(history: pd.DataFrame, tw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elo replay history (home_team, away_team, season, week, season_type,
    p_home, home_id, away_id, …) + every pregame feature. Strength is
    computed once per (season, week, postseason) and reused."""
    tw = load_team_weeks() if tw is None else tw
    h = history.copy()
    ids = team_ids(tw)
    for side in ("home", "away"):
        col = f"{side}_id"
        if col not in h.columns:
            h[col] = pd.NA
        fallback = h[f"{side}_team"].map(ids)
        h[col] = pd.to_numeric(h[col], errors="coerce").fillna(fallback)
    p = h["p_home"].clip(1e-4, 1 - 1e-4)
    h["elo_logit"] = np.log(p / (1 - p))
    h["is_post"] = h["season_type"].eq("postseason") if "season_type" in h.columns else False

    cache: dict[tuple, pd.DataFrame] = {}
    rows = []
    for r in h.itertuples(index=False):
        key = (int(r.season), int(r.week), bool(r.is_post))
        if key not in cache:
            cache[key] = team_strength(tw, key[0], key[1], key[2])
        hid = int(r.home_id) if r.home_id == r.home_id and r.home_id is not pd.NA else None
        aid = int(r.away_id) if r.away_id == r.away_id and r.away_id is not pd.NA else None
        rows.append(_features_from(cache[key], hid, aid))
    feats = pd.DataFrame(rows, index=h.index)
    return pd.concat([h.drop(columns=["is_post"]), feats], axis=1)


# --------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------
def coverage_report(tw: pd.DataFrame, games: pd.DataFrame):
    """How far behind the spine the weekly file is, in weeks, for the
    current season: (season, last real snapshot week, last completed spine
    week). Fresh when the gap is at most TOLERANCE_WEEKS."""
    from common.freshness import FreshnessReport
    played = games[games["completed"].astype(bool) & games["season_type"].eq("regular")]
    if played.empty or tw.empty:
        return FreshnessReport("sportsdataverse weekly summaries (team_weeks.csv)",
                               None, None, TOLERANCE_WEEKS, int(len(tw)))
    season = int(played["season"].max())
    spine_week = int(played.loc[played["season"] == season, "week"].max())
    real = real_weeks(tw[tw["season"] == season])
    have = int(real.get(season, 0))
    return FreshnessReport("sportsdataverse weekly summaries (team_weeks.csv)",
                           f"{season} week {have}", max(spine_week - have, 0),
                           TOLERANCE_WEEKS, int(len(tw)))


# --------------------------------------------------------------------------
# production second stage
# --------------------------------------------------------------------------
class SecondStage:
    """Elo + the production feature set -> home win probability, fit
    in-run on the replay history joined to the weekly table (2005 onward,
    ties excluded). `p_home` returns None when either side has no
    strength row for the game's (season, week), and the caller falls
    back to Elo."""

    def __init__(self, tw: pd.DataFrame, history: pd.DataFrame,
                 features: list[str] | None = None, C: float = PRODUCTION_C):
        from common import evaluate
        self.tw = tw
        self.features = list(features or PRODUCTION_FEATURES)
        self._strength: dict[tuple, pd.DataFrame] = {}
        table = build_game_table(history, tw)
        train = table[(table["season"] >= 2005) & (table["home_win"] != 0.5)]
        train = train[train[self.features].notna().all(axis=1)]
        self.n_train = int(len(train))
        self.seasons = (int(train["season"].min()), int(train["season"].max())) if self.n_train else None
        self.model = evaluate.make_logistic(C).fit(train[self.features],
                                                   (train["home_win"] == 1.0).astype(int))

    def strength(self, season: int, week: int, postseason: bool) -> pd.DataFrame:
        key = (int(season), int(week), bool(postseason))
        if key not in self._strength:
            self._strength[key] = team_strength(self.tw, key[0], key[1], key[2])
        return self._strength[key]

    def p_home(self, home_id, away_id, p_elo: float, season: int, week: int,
               postseason: bool = False) -> float | None:
        if home_id is None or away_id is None or pd.isna(home_id) or pd.isna(away_id):
            return None
        strength = self.strength(season, week, postseason)
        row = _features_from(strength, int(home_id), int(away_id))
        p = min(max(p_elo, 1e-4), 1 - 1e-4)
        row["elo_logit"] = float(np.log(p / (1 - p)))
        X = pd.DataFrame([row])[self.features]
        if X.isna().any(axis=None):
            return None
        return float(self.model.predict_proba(X)[0, 1])
