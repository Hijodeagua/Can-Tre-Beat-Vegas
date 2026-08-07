"""
Build data/mlb/games_2009_2026.csv from Retrosheet game logs (2009-2025)
plus the MLB statsapi schedule for the in-progress 2026 season.

Output columns:
    date, season, game_num, away, home, away_score, home_score
Team codes are Baseball-Reference conventions per season (FLA through 2011,
MIA after; OAK through 2024, ATH after), plus canonical `away_fr`/`home_fr`
franchise codes (current-day codes) for Elo continuity.
"""

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "mlb" / "raw_gamelogs"
OUT = REPO / "data" / "mlb" / "games_2009_2026.csv"

# Retrosheet code -> BRef code (season-invariant part)
RS_TO_BREF = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "FLO": "FLA", "HOU": "HOU", "KCA": "KCR", "LAN": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM",
    "OAK": "OAK", "ATH": "ATH", "PHI": "PHI", "PIT": "PIT", "SDN": "SDP",
    "SEA": "SEA", "SFN": "SFG", "SLN": "STL", "TBA": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WAS": "WSN",
}

# canonical franchise code (current-day) for Elo continuity
TO_FRANCHISE = {"FLA": "MIA", "OAK": "ATH"}

# statsapi full name -> BRef 2026 code
NAME_TO_BREF = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def franchise(code: str) -> str:
    return TO_FRANCHISE.get(code, code)


def parse_retrosheet(years) -> list[dict]:
    rows = []
    for y in years:
        zp = RAW / f"gl{y}.zip"
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
            text = z.read(name).decode("latin-1")
        for rec in csv.reader(io.StringIO(text)):
            date, gnum = rec[0], rec[1]
            away, home = RS_TO_BREF[rec[3]], RS_TO_BREF[rec[6]]
            a_sc, h_sc = int(rec[9]), int(rec[10])
            rows.append({
                "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "season": int(date[:4]),
                "game_num": int(gnum),
                "away": away, "home": home,
                "away_fr": franchise(away), "home_fr": franchise(home),
                "away_score": a_sc, "home_score": h_sc,
            })
    return rows


def fetch_2026(end_date: str = "2026-08-06") -> list[dict]:
    url = (
        "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
        f"&startDate=2026-03-01&endDate={end_date}"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            teams = g["teams"]
            away_name = teams["away"]["team"]["name"]
            home_name = teams["home"]["team"]["name"]
            if away_name not in NAME_TO_BREF or home_name not in NAME_TO_BREF:
                continue  # exhibition vs non-MLB
            if "score" not in teams["away"] or "score" not in teams["home"]:
                continue
            away, home = NAME_TO_BREF[away_name], NAME_TO_BREF[home_name]
            rows.append({
                "date": d["date"],
                "season": 2026,
                "game_num": int(g.get("gameNumber", 1)),
                "away": away, "home": home,
                "away_fr": franchise(away), "home_fr": franchise(home),
                "away_score": int(teams["away"]["score"]),
                "home_score": int(teams["home"]["score"]),
            })
    return rows


def main():
    rows = parse_retrosheet(range(2009, 2026))
    rows += fetch_2026()
    rows.sort(key=lambda r: (r["date"], r["game_num"], r["home"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_season = {}
    for r in rows:
        by_season[r["season"]] = by_season.get(r["season"], 0) + 1
    print(f"wrote {len(rows)} games -> {OUT}")
    for s in sorted(by_season):
        print(f"  {s}: {by_season[s]}")


if __name__ == "__main__":
    main()
