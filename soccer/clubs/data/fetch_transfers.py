"""
Refresh the committed club-season transfer aggregates from ewenme/transfers
(Transfermarkt-sourced player transfer fees, EUR millions).

The upstream is one CSV per league with a row per player movement. We keep
only clubs/seasons the Elo data models and aggregate to one row per
club-season: gross spend, gross sales, net spend, and movement counts.
Fees marked NA upstream (loans, free transfers, undisclosed) contribute to
counts but not to the money columns.

Upstream currency note: fees are EUR across the board (the upstream
converted historical GBP in 2022). Upstream coverage currently ends with
the 2022-23 winter window — the committed aggregate simply extends whenever
the upstream's summer/winter scrape actions resume.

Usage:
    python -m soccer.clubs.data.fetch_transfers
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests

from soccer.clubs.data.leagues import LEAGUES, canonical

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "club_season_transfers.csv"
RAW_BASE = "https://raw.githubusercontent.com/ewenme/transfers/master/data"

FILES = {
    "epl": "premier-league",
    "bundesliga": "1-bundesliga",
    "la_liga": "primera-division",
    "serie_a": "serie-a",
    "ligue_1": "ligue-1",
}

# Transfermarkt spellings -> our canonical names. Only clubs inside each
# league's modeled window appear; upstream rows for clubs that predate the
# window (Hércules, AC Siena, Valenciennes, …) are dropped by the
# known-clubs filter below.
TRANSFER_ALIASES: dict[str, dict[str, str]] = {
    "epl": {
        "Leeds United": "Leeds United FC",
        "Nottingham Forest": "Nottingham Forest FC",
    },
    "bundesliga": {
        "1.FC Kaiserslautern": "1. FC Kaiserslautern",
        "1.FC Nuremberg": "1. FC Nürnberg",
        "1.FC Union Berlin": "1. FC Union Berlin",
        "1.FSV Mainz 05": "1. FSV Mainz 05",
        "Bayern Munich": "FC Bayern München",
        "VfL Bochum": "VfL Bochum 1848",
    },
    "la_liga": {
        "Athletic Bilbao": "Athletic Club",
        "Atlético de Madrid": "Club Atlético de Madrid",
        "Celta de Vigo": "RC Celta de Vigo",
        "Deportivo de La Coruña": "RC Deportivo La Coruña",
        "RCD Espanyol Barcelona": "RCD Espanyol de Barcelona",
        "Racing Santander": "Real Racing Club de Santander",
    },
    "serie_a": {
        "Carpi FC 1909": "Carpi FC",
        "Catania SSD": "Calcio Catania",
        "Delfino Pescara 1936": "Delfino Pescara",
        "FC Empoli": "Empoli FC",
        "FC Internazionale": "FC Internazionale Milano",
        "Inter Milan": "FC Internazionale Milano",
        "SPAL": "SPAL 2013 Ferrara",
        "SPAL 2013": "SPAL 2013 Ferrara",
        "US Sassuolo": "US Sassuolo Calcio",
    },
    "ligue_1": {
        "AS Nancy-Lorraine": "AS Nancy Lorraine",
        "FC Girondins Bordeaux": "Girondins Bordeaux",
        "FC Toulouse": "Toulouse FC",
        "FC Évian Thonon Gaillard": "Évian Thonon Gaillard",
        "GFC Ajaccio": "Gazélec FC Ajaccio",
        "LOSC Lille": "Lille OSC",
        "Olympique Lyon": "Olympique Lyonnais",
        "Stade Reims": "Stade de Reims",
        "Stade Rennais FC": "Stade Rennais FC 1901",
    },
}


def known_clubs() -> dict[str, set]:
    r = pd.read_csv(DATA_DIR / "results.csv")
    return {
        lg: set(r.loc[r["league"] == lg, "home_team"])
        | set(r.loc[r["league"] == lg, "away_team"])
        for lg in r["league"].unique()
    }


def fetch_league(league: str, timeout: int = 60) -> pd.DataFrame | None:
    url = f"{RAW_BASE}/{FILES[league]}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! {league}: fetch failed ({exc}); keeping committed copy")
        return None
    df = pd.read_csv(io.BytesIO(resp.content))
    df["season"] = df["season"].str.replace(
        r"^(\d{4})/\d{2}(\d{2})$", r"\1-\2", regex=True
    )
    first = LEAGUES[league].first_season
    df = df[df["season"] >= first].copy()
    df["club"] = df["club_name"].map(
        lambda n: canonical(league, TRANSFER_ALIASES.get(league, {}).get(n, n))
    )
    return df


def aggregate(df: pd.DataFrame, league: str, clubs: set) -> pd.DataFrame:
    df = df[df["club"].isin(clubs)].copy()
    fee = pd.to_numeric(df["fee_cleaned"], errors="coerce")
    df["spend"] = fee.where(df["transfer_movement"] == "in", 0.0).fillna(0.0)
    df["sales"] = fee.where(df["transfer_movement"] == "out", 0.0).fillna(0.0)
    out = (
        df.groupby(["season", "club"])
        .agg(
            spend_eur_m=("spend", "sum"),
            sales_eur_m=("sales", "sum"),
            arrivals=("transfer_movement", lambda s: int((s == "in").sum())),
            departures=("transfer_movement", lambda s: int((s == "out").sum())),
        )
        .reset_index()
    )
    out["net_eur_m"] = out["spend_eur_m"] - out["sales_eur_m"]
    out.insert(0, "league", league)
    return out


def main() -> None:
    print("Fetching Transfermarkt transfer data (ewenme/transfers)…")
    clubs = known_clubs()
    frames = []
    for league in FILES:
        df = fetch_league(league)
        if df is None:
            continue
        agg = aggregate(df, league, clubs[league])
        dropped = sorted(set(df.loc[~df["club"].isin(clubs[league]), "club"]))
        print(
            f"  + {league}: {len(agg)} club-seasons "
            f"({agg['season'].min()} -> {agg['season'].max()})"
            + (f"; dropped pre-window clubs: {dropped}" if dropped else "")
        )
        frames.append(agg)
    if not frames:
        print("Nothing fetched; keeping the committed CSV untouched.")
        sys.exit(1)
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["league", "season", "club"]
    )
    cols = ["league", "season", "club", "spend_eur_m", "sales_eur_m",
            "net_eur_m", "arrivals", "departures"]
    result[cols].round(2).to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(result)} club-season rows to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
