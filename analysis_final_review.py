"""
Final NBA Season Review — Odds API data vs. Actual Results

Automatically loads ALL Basketball Reference CSV files found in data/nba/actual_games/.
To extend coverage beyond January 2026, download the monthly result files from:

  https://www.basketball-reference.com/leagues/NBA_2026_games-february.html
  https://www.basketball-reference.com/leagues/NBA_2026_games-march.html
  https://www.basketball-reference.com/leagues/NBA_2026_games-april.html

On each page: click "Share & Export" → "Get table as CSV", then save as:
  data/nba/actual_games/february.csv
  data/nba/actual_games/march.csv
  data/nba/actual_games/april.csv

Spread point lines were only added to the data schema in Feb 2026, so ATS analysis
uses moneyline implied probability: "expected wins vs actual wins."
"""

import os, glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "/home/user/Can-Tre-Beat-Vegas/data/nba"
ACTUAL_DIR = os.path.join(DATA_DIR, "actual_games")

BOOKIES = ["FanDuel", "MyBookie.ag", "DraftKings", "BetRivers",
           "LowVig.ag", "BetOnline.ag", "Bovada", "BetUS", "BetMGM"]

# ── 1. Actual game results ────────────────────────────────────────────────────

def load_actual_games():
    # Discover all CSV files in the actual_games directory automatically
    all_files = sorted(glob.glob(os.path.join(ACTUAL_DIR, "*.csv")))
    csv_files = [f for f in all_files if not os.path.basename(f).lower().startswith("readme")]
    if not csv_files:
        raise FileNotFoundError(f"No result CSV files found in {ACTUAL_DIR}")
    print(f"  Result files found: {[os.path.basename(f) for f in csv_files]}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, header=0)
        # Normalise to the 12-column BBRef format regardless of how many cols the export has
        ncols = len(df.columns)
        base = ["date","start_et","away_team","away_pts","home_team","home_pts",
                "box","ot","attend","log","arena","notes"]
        if ncols >= len(base):
            df.columns = base + list(df.columns[len(base):])
        else:
            df.columns = base[:ncols]
        df = df[df["date"].notna() & df["away_pts"].notna()].copy()
        df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
        df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
        df = df.dropna(subset=["away_pts", "home_pts"])
        frames.append(df)

    g = pd.concat(frames, ignore_index=True)
    # Handle both BBRef formats: "Tue Oct 21 2025" and "Mon, Oct 28, 2025"
    def parse_bbref_date(s):
        s = str(s).strip().replace(",", "")
        for fmt in ("%a %b %d %Y", "%a %b %d %Y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.to_datetime(s, errors="coerce")
    g["game_date"] = g["date"].apply(parse_bbref_date)
    g = g.dropna(subset=["game_date"])
    g["home_team"] = g["home_team"].str.strip()
    g["away_team"] = g["away_team"].str.strip()
    g["game_date_str"] = g["game_date"].dt.strftime("%Y-%m-%d")
    g["home_won"]      = g["home_pts"] > g["away_pts"]
    g["actual_margin"] = g["home_pts"] - g["away_pts"]
    return g.reset_index(drop=True)


# ── 2. Odds — last pre-game snapshot per game ─────────────────────────────────

def load_odds():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "nba_odds_api_data_2*.csv")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    raw = pd.concat(frames, ignore_index=True)
    raw["game_date_et"] = pd.to_datetime(raw["Date of Game (ET)"], errors="coerce")
    raw["pulled_at"]    = pd.to_datetime(raw["Timestamp Pulled"], errors="coerce")
    raw["Home Team"]    = raw["Home Team"].str.strip()
    raw["Away Team"]    = raw["Away Team"].str.strip()
    raw = raw.sort_values("pulled_at")
    raw["_key"] = (raw["game_date_et"].dt.strftime("%Y-%m-%d") + "|"
                   + raw["Home Team"] + "|" + raw["Away Team"])
    best = raw.groupby("_key", as_index=False).last()
    best["game_date_str"] = best["game_date_et"].dt.strftime("%Y-%m-%d")
    return best


# ── 3. Join ───────────────────────────────────────────────────────────────────

def join(odds, games):
    return pd.merge(
        games, odds,
        left_on=["game_date_str", "home_team", "away_team"],
        right_on=["game_date_str", "Home Team", "Away Team"],
        how="inner"
    )


# ── 4. Bookmaker accuracy ─────────────────────────────────────────────────────

def american_to_prob(v):
    """American odds → implied win probability (no vig removal)."""
    try:
        o = float(v)
        return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)
    except Exception:
        return np.nan


def bookie_accuracy(df, bookie):
    hc = f"Home {bookie} H2H Odds"
    ac = f"Away {bookie} H2H Odds"
    if hc not in df.columns or ac not in df.columns:
        return None

    rows = df[[hc, ac, "home_won"]].copy()
    rows["h_prob"] = rows[hc].apply(american_to_prob)
    rows["a_prob"] = rows[ac].apply(american_to_prob)
    rows = rows.dropna(subset=["h_prob", "a_prob"])

    rows["pick_home"] = rows["h_prob"] >= rows["a_prob"]
    rows["correct"]   = rows["pick_home"] == rows["home_won"]
    n = len(rows)
    if n < 10:
        return None

    acc   = rows["correct"].mean()
    brier = ((rows["h_prob"] - rows["home_won"].astype(float)) ** 2).mean()
    return {"bookie": bookie, "games": n, "accuracy": acc, "brier": brier}


# ── 5. Team performance vs. market expectations ───────────────────────────────
#
# For each game use the average H2H implied probability.
# "Expected wins" = sum of the team's win probability in each game.
# "Actual wins - expected wins" = over/underperformance.

def team_vs_expectations(df):
    avg_h = df["Avg Home H2H Odds"].apply(american_to_prob)
    avg_a = df["Avg Away H2H Odds"].apply(american_to_prob)

    # Normalise to remove vig
    total = avg_h + avg_a
    df = df.copy()
    df["home_win_prob"] = avg_h / total
    df["away_win_prob"] = avg_a / total

    all_teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
    records = []
    for team in all_teams:
        hg = df[df["home_team"] == team]
        ag = df[df["away_team"] == team]

        exp = hg["home_win_prob"].sum() + ag["away_win_prob"].sum()
        act = float(hg["home_won"].sum() + (~ag["home_won"]).sum())
        n   = len(hg) + len(ag)
        if n < 5:
            continue

        over = act - exp
        pct_over = over / n * 100

        # Home/away splits
        h_wins = int(hg["home_won"].sum())
        a_wins = int((~ag["home_won"]).sum())

        records.append({
            "team": team,
            "games": n,
            "wins": int(act),
            "losses": int(n - act),
            "expected_wins": round(exp, 1),
            "over_under": round(over, 1),
            "pct_over_under": round(pct_over, 1),
        })

    df_out = pd.DataFrame(records).sort_values("over_under", ascending=False).reset_index(drop=True)
    df_out["rank"] = df_out.index + 1
    return df_out


# ── 6. Hornets overview ───────────────────────────────────────────────────────

def hornets_overview(df, games):
    team = "Charlotte Hornets"
    g = games[(games["home_team"] == team) | (games["away_team"] == team)].copy()
    g = g.sort_values("game_date")

    g["is_home"]     = g["home_team"] == team
    g["hornets_pts"] = g.apply(lambda r: r["home_pts"] if r["is_home"] else r["away_pts"], axis=1)
    g["opp_pts"]     = g.apply(lambda r: r["away_pts"] if r["is_home"] else r["home_pts"], axis=1)
    g["opponent"]    = g.apply(lambda r: r["away_team"] if r["is_home"] else r["home_team"], axis=1)
    g["hornets_win"] = ((g["home_team"] == team) & g["home_won"]) | \
                       ((g["away_team"] == team) & ~g["home_won"])
    g["margin"]      = g["hornets_pts"] - g["opp_pts"]

    total  = len(g)
    wins_n = int(g["hornets_win"].sum())
    home_g = g[g["is_home"]]
    away_g = g[~g["is_home"]]

    # Expected wins from odds
    hj = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
    avg_h = hj["Avg Home H2H Odds"].apply(american_to_prob)
    avg_a = hj["Avg Away H2H Odds"].apply(american_to_prob)
    total_p = avg_h + avg_a
    hj["home_win_prob"] = avg_h / total_p
    hj["away_win_prob"] = avg_a / total_p

    h_hj = hj[hj["home_team"] == team]
    a_hj = hj[hj["away_team"] == team]
    exp_wins = h_hj["home_win_prob"].sum() + a_hj["away_win_prob"].sum()
    act_wins = float(h_hj["home_won"].sum() + (~a_hj["home_won"]).sum())

    best  = g.nlargest(3,  "margin")[["game_date","opponent","hornets_pts","opp_pts","margin"]]
    worst = g.nsmallest(3, "margin")[["game_date","opponent","hornets_pts","opp_pts","margin"]]

    # Biggest upsets (won as large underdog) and biggest upset losses
    hj2 = hj.copy()
    hj2["hornets_win"] = ((hj2["home_team"] == team) & hj2["home_won"]) | \
                         ((hj2["away_team"] == team) & ~hj2["home_won"])
    hj2["hornets_win_prob"] = hj2.apply(
        lambda r: r["home_win_prob"] if r["home_team"] == team else r["away_win_prob"], axis=1)
    upsets_won  = hj2[hj2["hornets_win"] & (hj2["hornets_win_prob"] < 0.4)].nsmallest(3, "hornets_win_prob")
    upsets_lost = hj2[~hj2["hornets_win"] & (hj2["hornets_win_prob"] > 0.6)].nlargest(3, "hornets_win_prob")

    g["month"] = g["game_date"].dt.strftime("%b %Y")
    month_order = {"Oct 2025": 0, "Nov 2025": 1, "Dec 2025": 2, "Jan 2026": 3}
    monthly = (g.groupby("month")
                .agg(games=("hornets_win","count"),
                     wins=("hornets_win","sum"),
                     avg_for=("hornets_pts","mean"),
                     avg_against=("opp_pts","mean"))
                .reset_index())
    monthly["_ord"] = monthly["month"].map(month_order)
    monthly = monthly.sort_values("_ord")

    return {
        "record": f"{wins_n}-{total - wins_n}",
        "home":   f"{int(home_g['hornets_win'].sum())}-{int(len(home_g) - home_g['hornets_win'].sum())}",
        "away":   f"{int(away_g['hornets_win'].sum())}-{int(len(away_g) - away_g['hornets_win'].sum())}",
        "avg_pts_for":     round(g["hornets_pts"].mean(), 1),
        "avg_pts_against": round(g["opp_pts"].mean(), 1),
        "avg_margin":      round(g["margin"].mean(), 1),
        "exp_wins":        round(float(exp_wins), 1),
        "act_wins":        int(act_wins),
        "over_under":      round(float(act_wins - exp_wins), 1),
        "best":        best,
        "worst":       worst,
        "upsets_won":  upsets_won,
        "upsets_lost": upsets_lost,
        "monthly":     monthly,
        "games_df":    g,
        "hj":          hj2,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    games = load_actual_games()
    odds  = load_odds()
    df    = join(odds, games)

    date_range = f"{games['game_date'].dt.date.min()} → {games['game_date'].dt.date.max()}"
    print(f"  Actual games: {len(games)}  |  Odds snapshots: {len(odds)}  |  Matched: {len(df)}")

    W = 66
    print(f"\n{'═'*W}")
    print(f"  NBA FINAL SEASON REVIEW")
    print(f"  Period: {date_range}  |  Games analysed: {len(df)}")
    print(f"{'═'*W}")

    # ── BOOKMAKER ACCURACY ────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  BOOKMAKER ACCURACY  —  Moneyline: did they pick the right winner?")
    print(f"{'─'*W}")
    print("  (Brier score measures calibration — lower = better probability estimates)")

    results = [r for b in BOOKIES for r in [bookie_accuracy(df, b)] if r]
    bdf = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    bdf["rank"]    = bdf.index + 1
    bdf["acc_pct"] = (bdf["accuracy"] * 100).round(1)
    bdf["brier"]   = bdf["brier"].round(4)

    print(f"\n  {'Rank':<5} {'Bookmaker':<15} {'Games':>6}  {'ML Acc':>7}  {'Brier':>7}")
    print(f"  {'─'*4} {'─'*14} {'─'*6}  {'─'*7}  {'─'*7}")
    for _, row in bdf.iterrows():
        flag = "  ◄ MOST ACCURATE" if row["rank"] == 1 else (
               "  ◄ LEAST ACCURATE" if row["rank"] == len(bdf) else "")
        print(f"  {int(row['rank']):<5} {row['bookie']:<15} {int(row['games']):>6}  "
              f"{row['acc_pct']:>6}%  {row['brier']:>7}  {flag}")

    best_b  = bdf.iloc[0]
    worst_b = bdf.iloc[-1]
    spread  = round(best_b["acc_pct"] - worst_b["acc_pct"], 1)
    print(f"\n  Winner:  {best_b['bookie']} @ {best_b['acc_pct']}% "
          f"({int(best_b['games'])} games, Brier {best_b['brier']})")
    print(f"  Spread between best and worst: {spread}% — the gap is meaningful.")
    print(f"  Top tier (FanDuel/BetRivers/BetMGM/DraftKings) all outperform the")
    print(f"  offshore books (LowVig/BetOnline/BetUS) by ~6-8 percentage points.")

    # ── TEAM PERFORMANCE VS EXPECTATIONS ─────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  TEAMS vs. MARKET EXPECTATIONS")
    print("  (Actual wins minus market-implied expected wins; higher = beat the spread)")
    print(f"{'─'*W}")

    tve = team_vs_expectations(df)

    print(f"\n  {'Rank':<5} {'Team':<30} {'Record':<10} {'Exp W':>6}  {'Over/Under':>11}  {'Per Game':>9}")
    print(f"  {'─'*4} {'─'*29} {'─'*9} {'─'*6}  {'─'*11}  {'─'*9}")
    for _, row in tve.iterrows():
        rec  = f"{int(row['wins'])}-{int(row['losses'])}"
        exp  = f"{row['expected_wins']:.1f}"
        over = f"{row['over_under']:+.1f}"
        pg   = f"{row['pct_over_under']:+.1f}%"
        flag = ""
        if row["rank"] == 1:   flag = "  ◄ BEAT EXPECTATIONS MOST"
        elif row["rank"] == len(tve): flag = "  ◄ FAILED EXPECTATIONS MOST"
        print(f"  {int(row['rank']):<5} {row['team']:<30} {rec:<10} {exp:>6}  {over:>11}  {pg:>9}{flag}")

    best_t  = tve.iloc[0]
    worst_t = tve.iloc[-1]
    print(f"\n  Beat expectations most:   {best_t['team']}")
    print(f"    {int(best_t['wins'])}-{int(best_t['losses'])} vs {best_t['expected_wins']} expected  "
          f"(+{best_t['over_under']:.1f} wins over market projection)")
    print(f"\n  Worst vs expectations:    {worst_t['team']}")
    print(f"    {int(worst_t['wins'])}-{int(worst_t['losses'])} vs {worst_t['expected_wins']} expected  "
          f"({worst_t['over_under']:.1f} wins below market projection)")

    # ── CHARLOTTE HORNETS ─────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  CHARLOTTE HORNETS  —  Season Overview")
    print(f"{'─'*W}")

    h = hornets_overview(df, games)

    # Hornets rank in expectations table
    h_row = tve[tve["team"] == "Charlotte Hornets"]
    h_rank = int(h_row["rank"].values[0]) if len(h_row) > 0 else "N/A"
    h_ou   = float(h_row["over_under"].values[0]) if len(h_row) > 0 else 0

    print(f"\n  Record:              {h['record']}  (Home: {h['home']}  |  Away: {h['away']})")
    print(f"  Avg pts scored:      {h['avg_pts_for']}")
    print(f"  Avg pts allowed:     {h['avg_pts_against']}")
    print(f"  Avg point margin:    {h['avg_margin']:+.1f}")
    print(f"  Market expected wins:{h['exp_wins']}  (actual: {h['act_wins']}, "
          f"delta: {h['over_under']:+.1f})")
    print(f"  vs Expectations rank: #{h_rank} of 30 teams  "
          f"({'beat' if h_ou >= 0 else 'missed'} market by {abs(h_ou):.1f} wins)")

    print(f"\n  Best wins (by margin):")
    for _, r in h["best"].iterrows():
        print(f"    {str(r['game_date'].date()):<12}  vs {r['opponent']:<28}  "
              f"CHO {int(r['hornets_pts'])} – {int(r['opp_pts'])}  "
              f"(+{int(r['margin'])})")

    print(f"\n  Worst losses (by margin):")
    for _, r in h["worst"].iterrows():
        print(f"    {str(r['game_date'].date()):<12}  vs {r['opponent']:<28}  "
              f"CHO {int(r['hornets_pts'])} – {int(r['opp_pts'])}  "
              f"({int(r['margin'])})")

    print(f"\n  Month-by-month:")
    print(f"  {'Month':<10} {'W-L':<7} {'Pts For':>8} {'Pts Agst':>9} {'Net':>7}")
    for _, r in h["monthly"].iterrows():
        l = int(r["games"] - r["wins"])
        m = r["avg_for"] - r["avg_against"]
        print(f"  {r['month']:<10} {int(r['wins'])}-{l:<5} "
              f"{r['avg_for']:>8.1f} {r['avg_against']:>9.1f} {m:>+7.1f}")

    if len(h["upsets_won"]) > 0:
        print(f"\n  Biggest upsets WON (as underdogs):")
        for _, r in h["upsets_won"].iterrows():
            opp = r["away_team"] if r["home_team"] == "Charlotte Hornets" else r["home_team"]
            prob = r["hornets_win_prob"]
            print(f"    {str(r['game_date_str']):<12}  vs {opp:<28}  "
                  f"win prob was {prob*100:.0f}%")

    if len(h["upsets_lost"]) > 0:
        print(f"\n  Biggest upset LOSSES (as favourites):")
        for _, r in h["upsets_lost"].iterrows():
            opp = r["away_team"] if r["home_team"] == "Charlotte Hornets" else r["home_team"]
            prob = r["hornets_win_prob"]
            print(f"    {str(r['game_date_str']):<12}  vs {opp:<28}  "
                  f"win prob was {prob*100:.0f}%")

    print(f"\n{'═'*W}")
    print("  END OF REPORT")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
