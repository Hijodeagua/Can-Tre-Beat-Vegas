"""
MLS rest-of-season forecast: Supporters' Shield, conference seeding, and
the MLS Cup Playoffs bracket, from a Monte Carlo over the remaining
schedule.

Same engine as every other league's rest-of-season sim
(`soccer/clubs/daily/simulate.py`): each simulated season carries its own
copy of the MLS Elo pool's ratings and updates them live with the pool's
tuned K and margin-of-victory rules as sampled results come in, so a club
that starts hot in a sim keeps mattering in that sim. Scorelines come from
the same independent-Poisson grid parameterized by the live Elo home
expectancy, so home advantage enters the forecast exactly the way it
enters the ratings — one rating per club plus the pool's tuned
`home_advantage` on whichever side is at home.

Three things MLS needs that the European sim does not:

**The schedule has to be reconstructed.** MLS's upstream is a log of
played matches with no fixture list, so `data/mls.py` rebuilds what is
left of the regular season from the format: intra-conference fixtures
exactly, cross-conference ones as per-club home/away quotas. Those ~25
cross-conference opponents are genuinely unknown, so each simulation
draws its own valid pairing of them (`_sample_inter`) rather than every
simulation running the same invented schedule — the uncertainty lands in
the spread of the odds, where it belongs.

**Standings are not just points.** MLS separates equal-points clubs by
wins first, then goal difference, then goals for, and seeding decides who
hosts every playoff round — so the sim tracks wins and goals per club and
orders with `Standing.sort_key`, with a random jitter only for clubs that
tie on all four.

**The playoffs are most of the question.** Winning the Shield and winning
MLS Cup are close to unrelated: the Shield is 34 matches, the Cup is at
most six, and the format hands the top seed a best-of-3 against a Wild
Card survivor rather than a bye. The bracket is simulated match by match
with the same Elo and scoreline model — a drawn playoff match goes to a
shootout, which the sim treats as a coin flip (`SHOOTOUT_HOME_EDGE`),
because that is what the evidence supports and pretending otherwise
would quietly hand home sides a second edge on top of the one already in
the ratings.

Deliberately absent: the Elo projection block the European sim exports.
That block plots each club's rating at checkpoint *dates*, and the
reconstructed schedule has no dates — MLS publishes them, the results log
does not carry them, and inventing a fixture calendar to draw a smooth
line through would be presenting a guess as data. Match order within the
run-in barely moves final standings, so the sim shuffles the remaining
fixtures per simulation instead.

    python -m soccer.clubs.model.mls_forecast [--season 2026] [--sims N]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from soccer.clubs.daily import scoring
from soccer.clubs.daily.config import MAX_GOALS
from soccer.clubs.data import mls
from soccer.clubs.data.mls import EAST, WEST, Standing
from soccer.clubs.model.elo import expected_score, mov_multiplier, run_pool

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FORECAST_FILE = ARTIFACTS / "mls_forecast.json"

DEFAULT_SIMS = 20000
# Resolution of the scoreline lookup table: the Elo home expectancy is
# rounded into this many bins over (0, 1) and each bin's full scoreline
# grid is precomputed once. At 500 bins two expectancies that share a bin
# differ by at most 0.002, which moves a goal rate by ~0.005 goals — far
# below the model's own error, and it turns every match draw into one
# uniform and a binary search.
EXP_BINS = 500
# P(the home side wins a drawn playoff match's shootout). Half, on
# purpose: a shootout is close to a coin flip, and the home side's real
# advantage is already priced into the 90 minutes through
# `home_advantage`. Exposed as a constant so it can be revisited if a
# study says otherwise rather than being buried in the loop.
SHOOTOUT_HOME_EDGE = 0.5
# How many random pairings of the remaining cross-conference fixtures to
# try before accepting one that repeats a matchup. Collisions are rare
# (each club has at most 3 slots against 15 candidates), so this almost
# never has to give up; the cap only guarantees the sim terminates.
INTER_DRAW_ATTEMPTS = 50


# --------------------------------------------------------------------------
# Scoreline sampling
# --------------------------------------------------------------------------

def _score_table(params: scoring.ScoreParams, league: str = mls.LEAGUE_KEY,
                 bins: int = EXP_BINS) -> np.ndarray:
    """(bins, (MAX_GOALS+1)^2) cumulative scoreline distributions, indexed
    by binned Elo home expectancy.

    Each row is `scoring.score_grid` for that bin's expectancy, flattened
    and cumulated, so sampling a scoreline is `searchsorted(row, u)` and
    `divmod(idx, MAX_GOALS + 1)`. Same distribution the daily slate
    publishes from, just precomputed — the truncated grid is renormalized
    rather than clipped, so no probability mass piles up on an 8-goal
    scoreline.
    """
    g = np.arange(MAX_GOALS + 1)
    fact = np.array([factorial(int(k)) for k in g], dtype=float)
    centers = (np.arange(bins) + 0.5) / bins
    rows = np.empty((bins, (MAX_GOALS + 1) ** 2), dtype=np.float64)
    for i, exp_home in enumerate(centers):
        lam_h, lam_a = params.lambdas(league, float(exp_home))
        ph = np.exp(-lam_h) * lam_h ** g / fact
        pa = np.exp(-lam_a) * lam_a ** g / fact
        grid = np.outer(ph, pa)
        rows[i] = np.cumsum((grid / grid.sum()).ravel())
    rows[:, -1] = 1.0
    return rows


# --------------------------------------------------------------------------
# Cross-conference fixture sampling
# --------------------------------------------------------------------------

def _sample_inter(rng: np.random.Generator,
                  rem: mls.RemainingSchedule) -> list[tuple[str, str]]:
    """One valid pairing of the remaining cross-conference fixtures.

    The quotas (how many cross-conference matches each club still has at
    home and away) are known exactly; the opponents are not. This draws a
    random pairing that respects the quotas and, where it can, avoids
    giving a club a rematch it has already played this season or drawing
    the same matchup twice in one season.
    """
    fixtures: list[tuple[str, str]] = []
    for hosts, guests in ((EAST, WEST), (WEST, EAST)):
        home_slots = [t for t in hosts for _ in range(rem.inter_home[t])]
        away_slots = [t for t in guests for _ in range(rem.inter_away[t])]
        if len(home_slots) != len(away_slots):
            raise ValueError(
                "cross-conference quotas do not balance; run "
                "mls.verify_structure before forecasting")
        best = None
        for _ in range(INTER_DRAW_ATTEMPTS):
            order = rng.permutation(len(away_slots))
            pairs = [(h, away_slots[j]) for h, j in zip(home_slots, order)]
            drawn = {frozenset(p) for p in pairs}
            if (len(drawn) == len(pairs)
                    and not (drawn & rem.played_inter)):
                best = pairs
                break
            best = best or pairs
        fixtures.extend(best)
    return fixtures


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

def _seed_order(idx: list[int], pts, wins, gf, ga,
                rng: np.random.Generator) -> list[int]:
    """Club indices ordered by the MLS tiebreakers, ties broken at random.

    Points, then wins, then goal difference, then goals for — the first
    four of the league's published tiebreakers. Below those come
    disciplinary points and away/home goal splits, which this model does
    not carry, so a club still tied on all four is separated by a jitter
    far smaller than a goal instead of by a pretend rule.
    """
    return sorted(
        idx,
        key=lambda i: (-pts[i], -wins[i], -(gf[i] - ga[i]), -gf[i],
                       rng.random()),
    )


def forecast(results: pd.DataFrame, season: str, *, engine=None,
             score_params: scoring.ScoreParams | None = None,
             n_sims: int = DEFAULT_SIMS, seed: int | None = 0) -> dict:
    """Rest-of-season + playoff Monte Carlo for one MLS season.

    `engine` and `score_params` default to a standalone replay of the MLS
    Elo pool and a score calibration fit on that pool's own history — the
    daily pipeline passes its already-built `DailyState` pieces instead so
    the published odds come from the same ratings as the published slate.
    """
    problems = mls.verify_structure(results, season)
    if problems:
        raise ValueError(
            "MLS season structure does not match the format in "
            "soccer/clubs/data/mls.py:\n  - " + "\n  - ".join(problems))

    if engine is None or score_params is None:
        built_engine, history = run_pool(mls.LEAGUE_KEY, df=results)
        engine = engine or built_engine
        score_params = score_params or scoring.fit(history)

    table = mls.standings(results, season)
    rem = mls.remaining_fixtures(results, season)
    clubs = sorted(mls.CONFERENCE)
    n = len(clubs)
    index = {c: i for i, c in enumerate(clubs)}
    east_idx = [index[c] for c in EAST]
    west_idx = [index[c] for c in WEST]

    home_adv = engine.home_advantage
    k = engine.k
    start = np.array([engine.rating_for(c, mls.LEAGUE_KEY) for c in clubs],
                     dtype=np.float64)
    base_pts = np.array([table[c].points for c in clubs], dtype=np.int32)
    base_wins = np.array([table[c].wins for c in clubs], dtype=np.int32)
    base_gf = np.array([table[c].goals_for for c in clubs], dtype=np.int32)
    base_ga = np.array([table[c].goals_against for c in clubs], dtype=np.int32)

    cum = _score_table(score_params)
    width = MAX_GOALS + 1
    intra = [(index[h], index[a]) for h, a in rem.intra]

    rng = np.random.default_rng(seed)
    counters = {
        name: np.zeros(n, dtype=np.int64)
        for name in ("shield", "playoffs", "top_seed", "conf_semi",
                     "conf_final", "conf_title", "cup")
    }
    pts_sum = np.zeros(n, dtype=np.float64)
    seed_sum = np.zeros(n, dtype=np.float64)
    seed_hist = np.zeros((n, mls.PLAYOFF_SPOTS + 1), dtype=np.int64)

    def play(h: int, a: int, ratings, u: float) -> tuple[int, int, float]:
        """One match: sampled scoreline plus the Elo delta it implies."""
        exp = expected_score(ratings[h] + home_adv, ratings[a])
        b = int(exp * EXP_BINS)
        if b >= EXP_BINS:
            b = EXP_BINS - 1
        hs, as_ = divmod(int(np.searchsorted(cum[b], u)), width)
        actual = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        delta = k * mov_multiplier(hs - as_) * (actual - exp)
        ratings[h] += delta
        ratings[a] -= delta
        return hs, as_, delta

    for _ in range(n_sims):
        fixtures = intra + [(index[h], index[a])
                            for h, a in _sample_inter(rng, rem)]
        # Match order barely moves final standings, but it does change
        # each club's live-Elo path through the run-in, so every sim gets
        # its own order rather than all of them sharing one.
        rng.shuffle(fixtures)

        ratings = start.copy()
        pts = base_pts.copy()
        wins = base_wins.copy()
        gf = base_gf.copy()
        ga = base_ga.copy()

        draws = rng.random(len(fixtures))
        for (h, a), u in zip(fixtures, draws):
            hs, as_, _ = play(h, a, ratings, float(u))
            gf[h] += hs
            ga[h] += as_
            gf[a] += as_
            ga[a] += hs
            if hs > as_:
                pts[h] += mls.WIN_POINTS
                wins[h] += 1
            elif hs < as_:
                pts[a] += mls.WIN_POINTS
                wins[a] += 1
            else:
                pts[h] += mls.DRAW_POINTS
                pts[a] += mls.DRAW_POINTS

        pts_sum += pts
        overall = _seed_order(list(range(n)), pts, wins, gf, ga, rng)
        counters["shield"][overall[0]] += 1
        # Regular-season rank across the whole league, used to decide who
        # hosts MLS Cup between two conference champions.
        overall_rank = {c: r for r, c in enumerate(overall)}

        champions = []
        for conf in (east_idx, west_idx):
            order = _seed_order(conf, pts, wins, gf, ga, rng)
            for pos, c in enumerate(order):
                seed_sum[c] += pos + 1
                if pos < mls.PLAYOFF_SPOTS:
                    seed_hist[c, pos] += 1
                else:
                    seed_hist[c, mls.PLAYOFF_SPOTS] += 1
            qualified = order[:mls.PLAYOFF_SPOTS]
            for c in qualified:
                counters["playoffs"][c] += 1
            counters["top_seed"][order[0]] += 1
            champions.append(_run_bracket(qualified, ratings, rng, play,
                                          counters))

        home, away = (champions if overall_rank[champions[0]]
                      < overall_rank[champions[1]] else champions[::-1])
        hs, as_, _ = play(home, away, ratings, float(rng.random()))
        if hs > as_:
            winner = home
        elif hs < as_:
            winner = away
        else:
            winner = home if rng.random() < SHOOTOUT_HOME_EDGE else away
        counters["cup"][winner] += 1

    return _payload(clubs, table, rem, start, counters, pts_sum, seed_sum,
                    seed_hist, n_sims, season, engine)


def _knockout(high: int, low: int, ratings, rng, play) -> int:
    """A single-elimination match hosted by the higher seed; a draw after
    90 (and, in the real thing, extra time) goes to a shootout."""
    hs, as_, _ = play(high, low, ratings, float(rng.random()))
    if hs > as_:
        return high
    if hs < as_:
        return low
    return high if rng.random() < SHOOTOUT_HOME_EDGE else low


def _series(high: int, low: int, ratings, rng, play) -> int:
    """Round One's best-of-3: higher seed hosts games 1 and 3, a drawn
    game is decided on penalties and counts as a win in the series, and
    game 3 is only played if the first two split."""
    high_wins = low_wins = 0
    for home_is_high in mls.ROUND_ONE_HOSTS:
        h, a = (high, low) if home_is_high else (low, high)
        hs, as_, _ = play(h, a, ratings, float(rng.random()))
        if hs > as_:
            winner = h
        elif hs < as_:
            winner = a
        else:
            winner = h if rng.random() < SHOOTOUT_HOME_EDGE else a
        if winner == high:
            high_wins += 1
        else:
            low_wins += 1
        if high_wins == mls.ROUND_ONE_WINS or low_wins == mls.ROUND_ONE_WINS:
            break
    return high if high_wins > low_wins else low


def _run_bracket(qualified: list[int], ratings, rng, play, counters) -> int:
    """One conference's playoffs; returns its MLS Cup finalist.

    `qualified` is seeds 1-9 in order. The 8/9 Wild Card match is played
    first and its winner takes the 8 seed into Round One, so the bracket
    below it is always 1v8, 4v5, 2v7, 3v6.
    """
    lo, hi = mls.WILD_CARD_SEEDS
    eight = _knockout(qualified[lo - 1], qualified[hi - 1], ratings, rng, play)
    seeds = qualified[:lo - 1] + [eight]

    # (1v8, 4v5) meet in one semifinal, (2v7, 3v6) in the other.
    halves = (((0, 7), (3, 4)), ((1, 6), (2, 5)))
    finalists = []
    for half in halves:
        winners = [_series(seeds[h], seeds[l], ratings, rng, play)
                   for h, l in half]
        for w in winners:
            counters["conf_semi"][w] += 1
        # The higher seed of the two hosts the semifinal; `seeds` is in
        # seed order, so the earlier index is the better seed.
        rank = {seeds[i]: i for i in range(len(seeds))}
        a, b = sorted(winners, key=lambda c: rank[c])
        finalists.append(_knockout(a, b, ratings, rng, play))
    for f in finalists:
        counters["conf_final"][f] += 1
    rank = {seeds[i]: i for i in range(len(seeds))}
    a, b = sorted(finalists, key=lambda c: rank[c])
    champ = _knockout(a, b, ratings, rng, play)
    counters["conf_title"][champ] += 1
    return champ


def _payload(clubs, table, rem, start, counters, pts_sum, seed_sum,
             seed_hist, n_sims, season, engine) -> dict:
    rows = []
    for i, c in enumerate(clubs):
        s: Standing = table[c]
        rows.append({
            "team": c,
            "conference": mls.conference(c),
            "elo": round(float(start[i]), 1),
            "played": s.played,
            "points": s.points,
            "wins": s.wins,
            "goal_diff": s.goal_diff,
            "remaining": mls.MATCHES_PER_CLUB - s.played,
            "exp_points": round(float(pts_sum[i]) / n_sims, 1),
            "exp_conf_seed": round(float(seed_sum[i]) / n_sims, 2),
            "p_shield": round(float(counters["shield"][i]) / n_sims, 4),
            "p_playoffs": round(float(counters["playoffs"][i]) / n_sims, 4),
            # Finishing 8th or 9th is exactly "has to win a Wild Card
            # match to reach Round One", so it is read off the seed
            # distribution rather than counted separately.
            "p_wild_card": round(
                float(seed_hist[i, mls.WILD_CARD_SEEDS[0] - 1]
                      + seed_hist[i, mls.WILD_CARD_SEEDS[1] - 1]) / n_sims, 4),
            "p_top_seed": round(float(counters["top_seed"][i]) / n_sims, 4),
            "p_conf_semi": round(float(counters["conf_semi"][i]) / n_sims, 4),
            "p_conf_final": round(float(counters["conf_final"][i]) / n_sims, 4),
            "p_conf_title": round(float(counters["conf_title"][i]) / n_sims, 4),
            "p_cup": round(float(counters["cup"][i]) / n_sims, 4),
            "seed_distribution": [int(x) / n_sims for x in seed_hist[i]],
        })
    rows.sort(key=lambda r: (-r["p_cup"], -r["p_shield"], r["exp_conf_seed"]))
    return {
        "season": season,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sims": n_sims,
        "remaining_matches": len(rem),
        "schedule": {
            "intra_conference": len(rem.intra),
            "cross_conference": rem.inter_count,
            "note": (
                "Intra-conference fixtures are reconstructed exactly from the "
                "round robin; cross-conference fixtures are known only as "
                "per-club home/away counts, and each simulation draws its own "
                "valid pairing of them."
            ),
        },
        "elo": {
            "pool": mls.LEAGUE_KEY,
            "k": engine.k,
            "home_advantage": engine.home_advantage,
            "season_regression": engine.season_regression,
        },
        "format": {
            "playoff_spots_per_conference": mls.PLAYOFF_SPOTS,
            "wild_card": "8 hosts 9, single match",
            "round_one": "best-of-3, higher seed hosts games 1 and 3",
            "later_rounds": "single match, higher seed hosts",
            "mls_cup": "single match, hosted by the finalist with the better regular-season record",
        },
        "clubs": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=None,
                    help="MLS season (calendar year); defaults to the latest in results.csv")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(FORECAST_FILE),
                    help="artifact path; '-' to skip writing")
    args = ap.parse_args()

    from soccer.clubs.model.elo import DATA_DIR

    results = pd.read_csv(DATA_DIR / "results.csv")
    season = args.season or str(
        results[results["league"] == mls.LEAGUE_KEY]["season"].max())

    out = forecast(results, season, n_sims=args.sims, seed=args.seed)
    print(f"MLS {season}: {out['remaining_matches']} matches left, "
          f"{out['sims']} sims\n")
    hdr = f"{'club':26s} {'conf':5s} {'elo':>6s} {'pts':>4s} {'xPts':>6s} " \
          f"{'Shield':>7s} {'Playoff':>8s} {'Conf':>7s} {'Cup':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in out["clubs"]:
        print(f"{r['team']:26s} {r['conference']:5s} {r['elo']:6.0f} "
              f"{r['points']:4d} {r['exp_points']:6.1f} "
              f"{r['p_shield']:6.1%} {r['p_playoffs']:7.1%} "
              f"{r['p_conf_title']:6.1%} {r['p_cup']:6.1%}")

    if args.out != "-":
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
