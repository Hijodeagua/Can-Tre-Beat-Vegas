"""Feature engineering for NFL game prediction.

Source data is per-team per-game with post-game box-score stats. Those raw
stats leak the outcome, so we transform them into rolling averages of the
team's last N games (shifted so the current game is excluded). Opponent
rolling stats are then merged in to give a matchup view.

Schedule context (home/away, rest, weather, roof, QB) is merged from the
nflverse games.csv via schedule.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from schedule import load_schedule, to_team_perspective


ROLLING_WINDOW = 4

ROLLING_STAT_COLS = [
    "TD_p", "Cmp_p", "Att_p", "Cmp%_p", "Yds_p", "Int_p", "Rate_p",
    "Sk_p", "Y/A_p", "NY/A_p", "ANY/A_p",
    "TD_r", "Att_r", "Yds_r", "Y/A_r",
    "Bltz_ad", "Hrry_ad", "QBKD_ad", "Prss_ad",
    "1stD_dn", "3DAtt_dn", "3DConv_dn", "3D%_dn",
    "1stDOpp_od", "Opp3DAtt_od", "Opp3DConv_od", "Opp3D%_od",
    "TeamScore", "OppScore",
]

ROOF_CATEGORIES = ["outdoors", "dome", "closed", "open"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Team"]).reset_index(drop=True)
    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Team", "Date"])
    df["rest_days"] = df.groupby("Team")["Date"].diff().dt.days
    df["rest_days"] = df["rest_days"].fillna(7).clip(upper=21)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)
    grouped = df.groupby("Team", group_keys=False)
    rolled = (
        grouped[ROLLING_STAT_COLS]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"roll_{c}" for c in rolled.columns]
    df = pd.concat([df, rolled], axis=1)
    df["games_played"] = grouped.cumcount()
    return df


def merge_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    roll_cols = [c for c in df.columns if c.startswith("roll_")]
    opp_view = df[["Date", "Team"] + roll_cols + ["games_played", "rest_days"]].copy()
    opp_view = opp_view.rename(
        columns={
            "Team": "Opp",
            "games_played": "opp_games_played",
            "rest_days": "opp_rest_days",
            **{c: f"opp_{c}" for c in roll_cols},
        }
    )
    return df.merge(opp_view, on=["Date", "Opp"], how="left")


def merge_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Join nflverse schedule for home/away, weather, roof, QB, etc."""
    sched = to_team_perspective(load_schedule())
    sched = sched.rename(columns={"gameday": "Date"})
    sched["Date"] = pd.to_datetime(sched["Date"])

    merged = df.merge(
        sched[["Date", "Team", "Opp", "is_home", "team_rest_sched",
               "opp_rest_sched", "team_qb", "opp_qb", "team_coach", "opp_coach",
               "closing_spread_team", "total_line", "roof", "surface",
               "temp", "wind", "div_game"]],
        on=["Date", "Team", "Opp"], how="left",
    )

    miss = merged["is_home"].isna().sum()
    if miss:
        print(f"  [warn] {miss}/{len(merged)} rows missing schedule merge")
    return merged


def add_qb_change(df: pd.DataFrame) -> pd.DataFrame:
    """Flag when the starting QB differs from the team's previous game."""
    df = df.sort_values(["Team", "Date"])
    df["prev_qb"] = df.groupby("Team")["team_qb"].shift(1)
    df["qb_change"] = (
        df["team_qb"].notna() & df["prev_qb"].notna()
        & (df["team_qb"] != df["prev_qb"])
    ).astype(int)
    df["opp_prev_qb"] = df.groupby("Opp")["opp_qb"].shift(1)
    df["opp_qb_change"] = (
        df["opp_qb"].notna() & df["opp_prev_qb"].notna()
        & (df["opp_qb"] != df["opp_prev_qb"])
    ).astype(int)
    return df.drop(columns=["prev_qb", "opp_prev_qb"])


def build_feature_frame(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = load_raw(path)
    df = add_rest_days(df)
    df = add_rolling_features(df)
    df = merge_opponent_features(df)
    df = merge_schedule(df)
    df = add_qb_change(df)

    df["day_num"] = df["Day_o_week"].map(
        {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    )

    for cat in ROOF_CATEGORIES:
        df[f"roof_{cat}"] = (df["roof"] == cat).astype(int)
    df["is_dome_or_closed"] = ((df["roof"] == "dome") | (df["roof"] == "closed")).astype(int)

    df["temp"] = df["temp"].astype(float)
    df["wind"] = df["wind"].astype(float)
    df.loc[df["is_dome_or_closed"] == 1, "wind"] = df.loc[df["is_dome_or_closed"] == 1, "wind"].fillna(0)
    df.loc[df["is_dome_or_closed"] == 1, "temp"] = df.loc[df["is_dome_or_closed"] == 1, "temp"].fillna(70)

    df["ats_cover"] = (df["vs._Line_vg"] == "covered").astype(int)
    df.loc[df["vs._Line_vg"] == "push", "ats_cover"] = np.nan

    df["rest_diff"] = df["rest_days"] - df["opp_rest_days"]

    roll_cols = [c for c in df.columns if c.startswith("roll_") or c.startswith("opp_roll_")]
    schedule_cols = [
        "is_home", "rest_days", "opp_rest_days", "rest_diff",
        "total_line", "temp", "wind", "div_game",
        "qb_change", "opp_qb_change", "is_dome_or_closed",
    ] + [f"roof_{c}" for c in ROOF_CATEGORIES]
    pregame_cols = ["Spread_vg", "Over/Under_vg", "Week", "day_num",
                    "games_played", "opp_games_played"]
    feature_cols = pregame_cols + schedule_cols + roll_cols

    df = df[df["games_played"] >= 2].reset_index(drop=True)
    return df, feature_cols
