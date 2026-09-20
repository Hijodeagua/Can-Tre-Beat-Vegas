"""
Walk-forward backtest of the MLS forecast (`mls_forecast.py`).

Rerun the forecast as it would have run at a set of past cutoff dates,
with every MLS match after the cutoff removed from the Elo replay, the
score calibration and the table alike, then score what it said against
what actually happened.

What is measured, and why each one:

- **Expected points** — mean absolute error and bias against each club's
  real final total. MAE says how tight the run-in projection is; bias is
  the one that would be damning, because a points model that drifts high
  or low moves every threshold in the table at once.
- **Playoff qualification Brier** — the forecast's sharpest binary call
  (18 of 30 clubs qualify), scored against the 0.24 you get by quoting
  the base rate at everybody.
- **Shield / MLS Cup** — log score and the probability the model put on
  the club that actually won. With one full 30-club season these are
  anecdotes, not measurements, and they are reported as such: a single
  Cup winner cannot distinguish a good bracket model from a lucky one.

The honest scope, stated once here so nobody reads more into the numbers
than they carry: MLS has been a 30-club, 15-per-conference league for
exactly one completed season (2025 — San Diego's arrival made it 30, and
`data/mls.py`'s structure check correctly refuses 2023 and 2024 as
29-club seasons with a different shape). So the walk forward is several
cutoffs *within one season*, whose errors are correlated — three cutoffs
in the same season are not three independent seasons. It is enough to
catch a broken model, not enough to claim a calibrated one.

One acknowledged leak, small but real: the MLS pool's Elo hyperparameters
in `artifacts/tuned_params.json` were tuned over a window that includes
the seasons being backtested. Retuning per cutoff would remove it; the
effect on a K of 10 is far below the noise from a single season, so it
is documented rather than fixed.

    python -m soccer.clubs.model.backtest_mls [--season 2025] [--sims N]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from soccer.clubs.data import mls
from soccer.clubs.model.elo import DATA_DIR
from soccer.clubs.model.mls_forecast import ARTIFACTS, forecast

BACKTEST_FILE = ARTIFACTS / "mls_backtest.json"

# Fractions of the regular season played at each cutoff. The last one is
# where the 2026 season sits today, so it is the cutoff whose error bars
# actually describe the published forecast.
CUTOFF_FRACTIONS = (0.35, 0.50, 0.70)
DEFAULT_SIMS = 4000
EPS = 1e-6  # log-score floor, so a missed call is finite


def as_of(results: pd.DataFrame, season: str, cutoff: str) -> pd.DataFrame:
    """`results` with every MLS match after `cutoff` removed — later
    seasons included, so the replay cannot see its own future."""
    mls_rows = results["league"] == mls.LEAGUE_KEY
    future = mls_rows & (
        (results["season"] > season)
        | ((results["season"] == season) & (results["date"] > cutoff))
    )
    return results[~future].copy()


def actual_outcomes(results: pd.DataFrame, season: str) -> dict:
    """What really happened: final table, who qualified, who won the
    Shield, and who won MLS Cup.

    The Cup winner is read off the last playoff match of the season —
    the playoff rows are whatever `mls.season_results` excluded from the
    regular season, and the final one of those is MLS Cup.
    """
    final = mls.standings(results, season)
    table = sorted(final.values(), key=lambda s: s.sort_key())
    qualified = set()
    for conf in ("East", "West"):
        in_conf = [s.team for s in table if mls.conference(s.team) == conf]
        qualified |= set(in_conf[:mls.PLAYOFF_SPOTS])

    reg = mls.season_results(results, season)
    season_rows = results[(results["league"] == mls.LEAGUE_KEY)
                          & (results["season"] == season)]
    playoffs = season_rows.loc[~season_rows.index.isin(reg.index)]
    playoffs = playoffs.dropna(subset=["home_score", "away_score"])
    cup_winner = None
    if not playoffs.empty:
        last = playoffs.sort_values("date", kind="stable").iloc[-1]
        # MLS Cup cannot end level; the log carries the decisive score.
        cup_winner = (last["home_team"]
                      if last["home_score"] > last["away_score"]
                      else last["away_team"])
    return {
        "points": {s.team: s.points for s in table},
        "qualified": qualified,
        "shield": table[0].team,
        "cup": cup_winner,
        "playoff_matches": int(len(playoffs)),
    }


def score_cutoff(out: dict, truth: dict) -> dict:
    """Score one forecast against the season's real outcomes."""
    rows = {c["team"]: c for c in out["clubs"]}
    errs, briers = [], []
    for team, c in rows.items():
        if team not in truth["points"]:
            continue
        errs.append(c["exp_points"] - truth["points"][team])
        briers.append((c["p_playoffs"] - (team in truth["qualified"])) ** 2)
    errs = np.array(errs, dtype=float)

    def p_of(key, winner):
        return rows[winner][key] if winner in rows else None

    p_shield = p_of("p_shield", truth["shield"])
    p_cup = p_of("p_cup", truth["cup"]) if truth["cup"] else None
    return {
        "clubs_scored": int(len(errs)),
        "exp_points_mae": round(float(np.abs(errs).mean()), 3),
        "exp_points_bias": round(float(errs.mean()), 3),
        "exp_points_rmse": round(float(np.sqrt((errs ** 2).mean())), 3),
        "playoff_brier": round(float(np.mean(briers)), 4),
        "playoff_brier_baseline": round(
            float(np.mean([(len(truth["qualified"]) / len(errs)
                            - (t in truth["qualified"])) ** 2
                           for t in rows if t in truth["points"]])), 4),
        "shield_winner": truth["shield"],
        "p_shield_winner": p_shield,
        "shield_log_score": round(-log(max(p_shield, EPS)), 3),
        "shield_was_favorite": bool(
            p_shield is not None
            and p_shield >= max(c["p_shield"] for c in rows.values())),
        "cup_winner": truth["cup"],
        "p_cup_winner": p_cup,
        "cup_log_score": (round(-log(max(p_cup, EPS)), 3)
                          if p_cup is not None else None),
    }


def run(season: str = "2025", n_sims: int = DEFAULT_SIMS,
        seed: int = 7, results: pd.DataFrame | None = None) -> dict:
    if results is None:
        results = pd.read_csv(DATA_DIR / "results.csv")
    problems = mls.verify_structure(results, season)
    if problems:
        raise ValueError(
            f"{season} does not match the season structure in data/mls.py, so "
            "a backtest of it would be scoring the wrong thing:\n  - "
            + "\n  - ".join(problems))

    reg = mls.season_results(results, season)
    dates = sorted(reg["date"].unique())
    truth = actual_outcomes(results, season)

    cutoffs = []
    for frac in CUTOFF_FRACTIONS:
        target = frac * len(reg)
        cutoff = next(d for d in dates if (reg["date"] <= d).sum() >= target)
        frame = as_of(results, season, cutoff)
        played = int((mls.season_results(frame, season)).shape[0])
        out = forecast(frame, season, n_sims=n_sims, seed=seed)
        cutoffs.append({
            "cutoff": cutoff,
            "fraction_played": round(played / len(reg), 3),
            "matches_played": played,
            "matches_remaining": out["remaining_matches"],
            **score_cutoff(out, truth),
        })

    return {
        "season": season,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sims": n_sims,
        "seed": seed,
        "regular_season_matches": int(len(reg)),
        "playoff_matches": truth["playoff_matches"],
        "scope": (
            f"{season} is the only completed 30-club MLS season, so these "
            "cutoffs are correlated views of one season, not independent "
            "seasons. Enough to catch a broken model, not enough to call it "
            "calibrated."
        ),
        "cutoffs": cutoffs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(BACKTEST_FILE),
                    help="artifact path; '-' to skip writing")
    args = ap.parse_args()

    report = run(args.season, n_sims=args.sims, seed=args.seed)
    print(f"MLS {report['season']} backtest — {report['sims']} sims per cutoff\n")
    hdr = (f"{'cutoff':12s} {'played':>7s} {'left':>5s} {'xPts MAE':>9s} "
           f"{'bias':>6s} {'PO Brier':>9s} {'base':>6s} {'P(shield)':>10s} {'P(cup)':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for c in report["cutoffs"]:
        pc = "  n/a" if c["p_cup_winner"] is None else f"{c['p_cup_winner']:6.1%}"
        print(f"{c['cutoff']:12s} {c['matches_played']:7d} "
              f"{c['matches_remaining']:5d} {c['exp_points_mae']:9.2f} "
              f"{c['exp_points_bias']:+6.2f} {c['playoff_brier']:9.4f} "
              f"{c['playoff_brier_baseline']:6.3f} "
              f"{c['p_shield_winner']:10.1%} {pc}")
    last = report["cutoffs"][-1]
    print(f"\nShield winner: {last['shield_winner']} "
          f"(model's favorite at the final cutoff: {last['shield_was_favorite']})")
    print(f"MLS Cup winner: {last['cup_winner']}")
    print(f"\n{report['scope']}")

    if args.out != "-":
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
