"""Per-game score simulation and rest-of-season Monte Carlo.

Score model
-----------
Win probability comes straight from Elo (logistic, +24 home advantage).
Scores are layered on top:

- Expected margin: linear map from Elo home-win probability, fit on the
  engine's own historical per-game record (E[home margin | p_home]). This is
  refit on every run from the replayed history, so it tracks the scoring
  environment.
- Expected total: league mean runs/game over a recent window (current season
  to date, padded with the prior season early in the year).
- Each team's runs are drawn from a negative binomial with that mean split
  (total +/- margin)/2, with the dispersion fit from recent per-team run
  variance (MLB team runs are overdispersed vs Poisson, var/mean ~ 1.8).
- Ties are broken by awarding one run to a winner drawn with the Elo home
  probability - a cheap stand-in for extra innings (MLB games cannot tie).

Reported score: the *rounded conditional mean* - the average simulated score
across only the sims where the predicted (majority) winner actually won.
Chosen over the modal exact scoreline because score modes in baseball are
degenerate (always 3-2/4-3 type lines regardless of matchup strength); the
conditional mean is the MMSE estimate given the pick, preserves the predicted
margin's sign, and varies sensibly with team strength. The unconditional mean
would routinely produce scores that contradict the predicted winner.

Season Monte Carlo
------------------
Each sim replays the remaining schedule game by game with live Elo updates
(K=3, no margin-of-victory term since scores are not simulated here), win
probabilities from the current per-sim ratings. Vectorized across sims.

Playoff format assumed: current 12-team field - 3 division winners seeded
1-3 by record, 3 wild cards seeded 4-6. All within-sim ties (division,
seeding, wild card) are broken uniformly at random; MLB's real head-to-head
tiebreakers are not modeled, which washes out over thousands of sims but is
an explicit approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb.elo import HOME_ADVANTAGE, K, expected_score
from mlb.daily.config import (
    ALL_TEAMS, CURRENT_SEASON, DIVISIONS, TEAM_LEAGUE,
)

TEAM_INDEX = {t: i for i, t in enumerate(ALL_TEAMS)}


@dataclass
class ScoreParams:
    margin_intercept: float
    margin_slope: float
    total_mean: float
    dispersion: float  # negative binomial r (size); larger = closer to Poisson

    def expected_margin(self, p_home: float) -> float:
        return self.margin_intercept + self.margin_slope * (p_home - 0.5)


def calibrate(history: pd.DataFrame, games: pd.DataFrame,
              since_season: int = 2015) -> ScoreParams:
    """Fit score-model parameters from the Elo engine's own history."""
    h = history[history.season >= since_season]
    x = h.p_home.to_numpy() - 0.5
    y = h.run_diff.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    recent = games[games.season >= CURRENT_SEASON - 1]
    totals = (recent.home_score + recent.away_score).to_numpy(dtype=float)
    total_mean = float(totals.mean())

    per_team = np.concatenate([
        recent.home_score.to_numpy(dtype=float),
        recent.away_score.to_numpy(dtype=float),
    ])
    mean, var = per_team.mean(), per_team.var()
    # Method-of-moments NB fit; guard against (impossible) underdispersion.
    dispersion = mean**2 / max(var - mean, 1e-6)
    return ScoreParams(float(intercept), float(slope), total_mean, float(dispersion))


def _nb_draw(rng: np.random.Generator, mean: np.ndarray | float,
             r: float, size) -> np.ndarray:
    """Negative binomial with given mean and size parameter r."""
    p = r / (r + np.asarray(mean, dtype=float))
    return rng.negative_binomial(r, p, size=size)


def simulate_game(p_home: float, params: ScoreParams,
                  n: int = 10000, seed: int | None = None,
                  total: float | None = None) -> dict:
    """Simulate one matchup n times; return pick, probability and the
    rounded conditional-mean score.

    `total` overrides the league-average expected total with a
    matchup-specific one (from scoring.TeamRates); the Elo-implied margin
    is carved out of whichever total is used, so win probability and
    expected margin are unchanged by the override."""
    rng = np.random.default_rng(seed)
    margin = params.expected_margin(p_home)
    t = params.total_mean if total is None else total
    home_mean = max((t + margin) / 2.0, 0.25)
    away_mean = max((t - margin) / 2.0, 0.25)

    home_runs = _nb_draw(rng, home_mean, params.dispersion, n).astype(float)
    away_runs = _nb_draw(rng, away_mean, params.dispersion, n).astype(float)

    tied = home_runs == away_runs
    extra_home_win = rng.random(tied.sum()) < p_home
    home_runs[np.where(tied)[0][extra_home_win]] += 1
    away_runs[np.where(tied)[0][~extra_home_win]] += 1

    home_wins = home_runs > away_runs
    pick_home = p_home >= 0.5

    # Report expected runs at one decimal rather than a rounded integer
    # line: integer rounding collapses nearly every MLB matchup onto the
    # same few scorelines (6-3, 7-3) because game-to-game differences in
    # expected margin and total are smaller than a run. One decimal keeps
    # the estimate honest (it is just E[runs]) and lets matchups differ.
    home_score = round(home_mean, 1)
    away_score = round(away_mean, 1)
    # Keep the line consistent with the pick: near p_home = 0.5 the margin
    # fit's intercept can put the expected-runs edge on the other side.
    if pick_home and home_score <= away_score:
        home_score = round(away_score + 0.1, 1)
    elif not pick_home and away_score <= home_score:
        away_score = round(home_score + 0.1, 1)

    return {
        "pick_home": pick_home,
        "p_home": p_home,
        "sim_home_win_rate": float(home_wins.mean()),
        "home_score": home_score,
        "away_score": away_score,
    }


def slate_predictions(ratings: dict[str, float], slate: list[dict],
                      params: ScoreParams, n: int = 10000,
                      seed: int | None = 7,
                      rates=None, adjustments: dict | None = None,
                      model_version: str | None = None) -> pd.DataFrame:
    """Predictions for one day's games. `slate` rows need date, away, home,
    away_fr, home_fr, game_num. `rates` (scoring.TeamRates) makes the
    expected total matchup-specific; without it every game shares the
    league-average total.

    `adjustments` (mlb.daily.sp_state.slate_adjustments) maps
    (date, game_num, home_fr) -> per-side Elo adjustments; when present they
    enter the win probability and are carried as audit columns."""
    rows = []
    for g in slate:
        adj = (adjustments or {}).get(
            (g["date"], int(g["game_num"]), g["home_fr"]))
        home_extra = adj["home_adj"] if adj else 0.0
        away_extra = adj["away_adj"] if adj else 0.0
        p_home = expected_score(
            ratings[g["home_fr"]] + HOME_ADVANTAGE + home_extra,
            ratings[g["away_fr"]] + away_extra,
        )
        total = (rates.matchup_total(g["home_fr"], g["away_fr"])
                 if rates is not None else None)
        sim = simulate_game(p_home, params, n=n, seed=seed, total=total)
        pick = g["home_fr"] if sim["pick_home"] else g["away_fr"]
        row = {
            "date": g["date"],
            "away": g["away_fr"],
            "home": g["home_fr"],
            "game_num": int(g["game_num"]),
            # Probable starters. Display-only for v1; a model input (via
            # `adjustments`) for the SP-adjusted model.
            "away_sp": g.get("away_sp", ""),
            "home_sp": g.get("home_sp", ""),
            "away_sp_id": g.get("away_sp_id", ""),
            "home_sp_id": g.get("home_sp_id", ""),
            "pred_total": round(total if total is not None
                                else params.total_mean, 1),
            "p_home": round(p_home, 4),
            "pick": pick,
            "pick_prob": round(p_home if sim["pick_home"] else 1 - p_home, 4),
            "pred_home_score": sim["home_score"],
            "pred_away_score": sim["away_score"],
            "elo_home": round(ratings[g["home_fr"]], 1),
            "elo_away": round(ratings[g["away_fr"]], 1),
        }
        if model_version:
            row["model_version"] = model_version
        if adj:
            row.update({k: adj[k] for k in
                        ("home_sp_adj", "away_sp_adj", "home_sp_mode",
                         "away_sp_mode", "home_rt_adj", "away_rt_adj")})
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_season(ratings: dict[str, float], remaining: list[dict],
                    standings: pd.DataFrame, n_sims: int = 2000,
                    seed: int | None = 11) -> pd.DataFrame:
    """Monte Carlo the rest of the season. Returns per-team probabilities:
    division title, playoff berth, top seed, plus mean final wins."""
    rng = np.random.default_rng(seed)
    n_teams = len(ALL_TEAMS)

    base = np.array([ratings[t] for t in ALL_TEAMS])
    R = np.tile(base, (n_sims, 1))
    wins = np.tile(
        standings.set_index("team").loc[ALL_TEAMS, "wins"].to_numpy(dtype=float),
        (n_sims, 1),
    )
    losses0 = standings.set_index("team").loc[ALL_TEAMS, "losses"].to_numpy(dtype=float)

    for g in remaining:
        h, a = TEAM_INDEX[g["home_fr"]], TEAM_INDEX[g["away_fr"]]
        p = 1.0 / (1.0 + 10 ** (-((R[:, h] + HOME_ADVANTAGE) - R[:, a]) / 400.0))
        home_won = rng.random(n_sims) < p
        wins[:, h] += home_won
        wins[:, a] += ~home_won
        delta = K * (home_won - p)
        R[:, h] += delta
        R[:, a] -= delta

    # Random within-sim tiebreak: add U(0,1) noise; integer win gaps dominate.
    noisy = wins + rng.random(wins.shape)

    div_champ = np.zeros((n_sims, n_teams), dtype=bool)
    for teams in DIVISIONS.values():
        idx = np.array([TEAM_INDEX[t] for t in teams])
        winners = idx[np.argmax(noisy[:, idx], axis=1)]
        div_champ[np.arange(n_sims), winners] = True

    playoff = div_champ.copy()
    top_seed = np.zeros((n_sims, n_teams), dtype=bool)
    for lg in ("AL", "NL"):
        idx = np.array([TEAM_INDEX[t] for t in ALL_TEAMS if TEAM_LEAGUE[t] == lg])
        lg_champ = div_champ[:, idx]
        # Top seed: best record among the league's division winners.
        champ_noisy = np.where(lg_champ, noisy[:, idx], -np.inf)
        top = idx[np.argmax(champ_noisy, axis=1)]
        top_seed[np.arange(n_sims), top] = True
        # Wild cards: best three non-champion records.
        wc_noisy = np.where(lg_champ, -np.inf, noisy[:, idx])
        order = np.argsort(-wc_noisy, axis=1)[:, :3]
        for k in range(3):
            playoff[np.arange(n_sims), idx[order[:, k]]] = True

    games_left = np.zeros(n_teams)
    for g in remaining:
        games_left[TEAM_INDEX[g["home_fr"]]] += 1
        games_left[TEAM_INDEX[g["away_fr"]]] += 1

    wins0 = standings.set_index("team").loc[ALL_TEAMS, "wins"].to_numpy(dtype=float)
    mean_wins = wins.mean(axis=0)
    mean_losses = losses0 + games_left - (mean_wins - wins0)

    return pd.DataFrame({
        "team": ALL_TEAMS,
        "elo": [round(ratings[t], 1) for t in ALL_TEAMS],
        "mean_wins": mean_wins.round(1),
        "mean_losses": mean_losses.round(1),
        "division_pct": div_champ.mean(axis=0).round(4),
        "playoff_pct": playoff.mean(axis=0).round(4),
        "top_seed_pct": top_seed.mean(axis=0).round(4),
    })
