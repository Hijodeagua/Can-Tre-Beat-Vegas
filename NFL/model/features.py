"""Feature engineering for NFL game prediction.

Source data is per-team per-game with post-game box-score stats. Those raw
stats leak the outcome, so we transform them into rolling averages of the
team's last N games (shifted so the current game is excluded). Opponent
rolling stats are then merged in to give a matchup view.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


ROLLING_WINDOW = 4

PREGAME_PASSTHROUGH = [
    "Spread_vg",
    "Over/Under_vg",
    "Week",
    "Day_o_week",
]

ROLLING_STAT_COLS = [
    "TD_p", "Cmp_p", "Att_p", "Cmp%_p", "Yds_p", "Int_p", "Rate_p",
    "Sk_p", "Y/A_p", "NY/A_p", "ANY/A_p",
    "TD_r", "Att_r", "Yds_r", "Y/A_r",
    "Bltz_ad", "Hrry_ad", "QBKD_ad", "Prss_ad",
    "1stD_dn", "3DAtt_dn", "3DConv_dn", "3D%_dn",
    "1stDOpp_od", "Opp3DAtt_od", "Opp3DConv_od", "Opp3D%_od",
    "TeamScore", "OppScore",
]


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
    merged = df.merge(opp_view, on=["Date", "Opp"], how="left")
    return merged


def build_feature_frame(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = load_raw(path)
    df = add_rest_days(df)
    df = add_rolling_features(df)
    df = merge_opponent_features(df)

    df["day_num"] = df["Day_o_week"].map(
        {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    )

    df["ats_cover"] = (df["vs._Line_vg"] == "covered").astype(int)
    df.loc[df["vs._Line_vg"] == "push", "ats_cover"] = np.nan

    roll_cols = [c for c in df.columns if c.startswith("roll_") or c.startswith("opp_roll_")]
    feature_cols = (
        ["Spread_vg", "Over/Under_vg", "Week", "day_num", "rest_days",
         "opp_rest_days", "games_played", "opp_games_played"]
        + roll_cols
    )

    df = df[df["games_played"] >= 2].reset_index(drop=True)
    return df, feature_cols
