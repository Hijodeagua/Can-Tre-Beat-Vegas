# Model features — quick sheet

What each **live** model actually looks at, per sport. This is the forecast
on the site, not the research models in the repo: where a bigger model
exists but doesn't feed the board, it's listed under *Not in the live
forecast* so the two never get confused.

Every rating here is betting-blind — no closing line, no market price ever
reaches a rating or a pick.

Last checked against the code and the tuned artifacts on 2026-09-17.

- [Soccer](#soccer-model-features) · [Football — NFL](#football-model-features--nfl) · [Football — CFB](#football-model-features--cfb) · [Baseball — MLB](#baseball-model-features--mlb)
- [Why a projected Elo line looks flat](#why-a-projected-elo-line-looks-flat)

---

## Soccer model features

Ten league pools (five top flights + their second divisions) plus MLS.
Pick and forecast: `soccer/clubs/daily/{predict,simulate}.py`.

**Match outcome — multinomial logistic over {home win, draw, away win}**
(`soccer/clubs/model/train.py`, fitted artifact
`soccer/clubs/model/artifacts/outcome_model.pkl`). These seven features,
in the artifact's own order:

* **Home Elo** — club Elo, home side, plus the pool's home advantage
* **Away Elo** — club Elo, away side
* `elo_gap` — the venue-adjusted difference of the two above; the model
  sees the gap, not the two ratings separately
* `spend_diff_z` — gross transfer spend this season, home − away,
  z-scored within league-season
* `net_diff_z` — net spend (spend − sales), same normalisation
* `value_diff_z` — squad market value differential
* `wage_diff_z` — wage bill differential
* `xg_net_diff` — rolling xG net (for − against, last 10 league matches),
  home − away
* `sot_net_diff` — rolling shots-on-target net, same shape

**Live weight, from the fitted artifact** (coefficient on the home-win
class): `value_diff_z` +0.078 · `sot_net_diff` +0.061 · `spend_diff_z`
+0.024 · `xg_net_diff` +0.020 · `net_diff_z` +0.007 · `elo_gap` +0.003
(per Elo point, so ~+0.3 per 100) · `wage_diff_z` 0.000.

**Dormant right now, and why it matters:** `wage_diff_z` has a zero
coefficient (no wage uploads in `soccer/clubs/data/market_values/`), and
`xg_net_diff` is fed as 0 on every current slate because the Understat
backfill stops at 2025-01-04 and the staleness guard voids form older than
130 days. `sot_net_diff` (football-data.co.uk) is the chance-creation feed
that is actually updating. Both degrade to 0 by design rather than
poisoning the fit — see `soccer/clubs/model/form.py`.

**Scoreline** (`soccer/clubs/daily/scoring.py`): independent Poisson per
side, λ from the league's own goal rates over the last two completed
seasons, scaled by the Elo expectation. Grid capped at 8 goals a side.

**Elo mechanics** (`soccer/clubs/model/elo.py`, `europe.py`): one glued
scale across the UEFA leagues (cross-league matches carry 0.75 weight),
tuned per pool — EPL runs K 14, home +60, 10% season regression, entry
1380 (1150 from the second tier). Promoted and relegated clubs carry
their rating across divisions.

**Forecast** (`simulate.py`): 30,000 replays of the remaining fixtures,
each with its own copy of the ratings updated live by the tuned K/MOV
rules as sampled results land, scorelines drawn from the Poisson grid.
Title / UCL / UEL / relegation odds and expected points come out of that.

---

## Football model features — NFL

Pick and forecast: `NFL/daily/{predict,simulate}.py`.

**Win probability — Elo alone.** No regression on top; the rating carries
every situational edge:

* **Home Elo** — team Elo, plus home advantage (+48, and **zero** at a
  neutral site)
* **Away Elo** — team Elo
* **Rest** — a side coming off a bye (10+ days) gets +20 Elo at
  prediction time
* **Postseason weight** — playoff results update ratings at `K ×
  playoff_k_mult` (currently 1.0)
* **Margin of victory** — ln-damped and capped at 45 points, shrunk when
  the favourite wins
* **Season carryover** — 40% regression toward 1500 at each boundary;
  franchise continuity (STL→LA, SD→LAC, OAK→LV) carries a rating through
  a relocation
* **Ties** — scored as 0.5, plain K

Tuned on 2005–2023 by coordinate descent, holdout 2024–25: log loss
**0.624** vs 0.691 always-pick-home (`NFL/elo/artifacts/tuned_params.json`).

**Expected score** (`NFL/daily/scoring.py`), refit from the engine's own
replay every run: margin = a + b·(Elo difference); total from each team's
exponentially weighted points-scored and points-allowed rates (half-life
8 games) shrunk toward the league mean. Elo stays the only authority on
the win probability.

**Forecast** (`simulate.py`): 10,000 vectorized replays of the remaining
regular season with live in-sim Elo and the rest bonus, then the full
seven-team bracket per conference.

**Not in the live forecast:** `NFL/model/v2/` is a 45-feature LightGBM
research model (rolling box-score rates, weather, roof, QB, and the
closing line). It informs nothing on the site — the board is the Elo
engine above.

---

## Football model features — CFB

Pick and forecast: `CFB/daily/{predict,simulate}.py`. Same skeleton as the
NFL engine with the four things college needs.

* **Home Elo** — program Elo, plus home advantage (+50, zero at a neutral
  site)
* **Away Elo** — program Elo
* **Conference-aware season regression** — 30% regression at each
  boundary, toward a 0.75/0.25 blend of the program's **new** conference's
  mean and the 1500 base, so realignment is handled by construction;
  independents regress to the base
* **Pooled FCS opponent** — every non-FBS side is one synthetic team at
  950, and such a game updates only the FBS side (keeps ~13% of the
  schedule in the data)
* **FBS entry rating** — a program's first FBS game starts it at 1250,
  not at average
* **Margin of victory** — ln-damped, capped at 80 points

Tuned 2005–2023, holdout 2024–25: log loss **0.499**
(`CFB/model/artifacts/tuned_params.json`).

**Expected score** (`CFB/daily/scoring.py`): margin linear in the Elo
difference; total from exponentially weighted points-scored/allowed rates
(half-life ~10 games) shrunk toward the FBS mean.

**Forecast** (`simulate.py`): 10,000 replays with live in-sim Elo →
expected wins, bowl eligibility, an undefeated season, a conference-title-
game berth and the title. **The 12-team playoff field is deliberately not
modelled** — selection is a committee ranking, and inventing one would put
a made-up number next to the honest ones.

---

## Baseball model features — MLB

Pick and forecast: `mlb/daily/`. Active model version `v2-sp`.

* **Home Elo** — team Elo, plus home advantage (+24)
* **Away Elo** — team Elo
* **Starting pitcher** — `adj = 3.0 × (pitcher rGS − staff rGS)` in Elo
  points, where rGS is an exponentially weighted Bill James game score
  (half-life 20 starts) with a fallback ladder: rated starter → thin
  history shrunk toward league → TBD gets the staff rate (no adjustment).
  Tuned on 2012–2021 (`research/SP-BACKTEST.md`)
* **Rest** — `+2.3 × rest_days`, rest capped at 3 days
* **Travel** — `−0.31 × miles^(1/3)`, capped at −4 Elo, from a static
  per-era home-park coordinate table
* **Elo mechanics** — K 3.0 over a 2009-present replay rebuilt every run
  (no incremental state on disk to drift)

**Expected score** (`mlb/daily/scoring.py`): Dixon-Coles-style attack /
defense run rates, exponentially weighted (half-life 20 team-games) and
shrunk toward the league mean with a 60-game prior; matchup total clipped
to 5.5–13.5 runs.

**Forecast** (`mlb/daily/simulate.py`): 2,000 season replays; 10,000 sims
for a single game.

---

## Why a projected Elo line looks flat

An Elo update is `K × (actual − expected)`, and a simulation draws results
at exactly its own expected rate. So every team's **expected** rating
change is about zero, and the *mean* projected rating across thousands of
simulated seasons is nearly flat by construction — for the leader and the
bottom club alike. That is arithmetic, not an assumption that the table
stops moving.

The movement is in the spread, and it is large. From the 2026-09-17 EPL
run: Arsenal's mean goes 1767.8 → 1774.2 over the season, while the
10th–90th percentile band at the final matchday spans 94.5 Elo points —
more than twice the 39-point gap between Arsenal and Manchester City
today. Three individual simulated NFL seasons had Seattle finishing on
1704, 1620 and 1741 against a mean of 1660.8, and today's top five
survived intact in none of them.

So the site's projection draws **whole simulated seasons** (three per
labelled side, each one a coherent season shared across teams) with the
mean behind them, and the percentile band on hover. Anything that reports
only the mean — a chart, a table column, a summary line — is reporting the
one season in which nothing happens, and should be treated as a bug.
Exported as `elo_projection.samples` next to `elo_projection.clubs` /
`elo_projection.teams` in each sport's `latest.json`; produced in
`soccer/clubs/daily/simulate.py`, `NFL/daily/simulate.py` and
`CFB/daily/simulate.py`.
