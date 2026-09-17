# Model features — quick sheet

What each **live** model actually looks at, per sport. This is the forecast
on the site, not the research models in the repo: where a bigger model
exists but doesn't feed the board, it's listed under *Not in the live
forecast* so the two never get confused.

Every rating here is betting-blind — no closing line, no market price ever
reaches a rating or a pick.

Last checked against the code and the tuned artifacts on 2026-09-17.

- [Soccer](#soccer-model-features) · [Football — NFL](#football-model-features--nfl) · [Football — CFB](#football-model-features--cfb) · [Baseball — MLB](#baseball-model-features--mlb)
- [What the inputs are measured to be worth](#what-the-inputs-are-measured-to-be-worth)
- [What's missing, and what it would cost](#whats-missing-and-what-it-would-cost)
- [Why a projected Elo line looks flat](#why-a-projected-elo-line-looks-flat)

Every "worth" figure below is measured by
`python -m data_jobs.build_importance`, on seasons the model was never
fitted or tuned on, and is committed next to each model's own parameters.
Nothing in this file is an opinion about which feature matters.

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

**Win probability — Elo plus adjusted success rate.** A regularised
logistic on top of the Elo (`NFL/model/advanced.py`, fit in-run on
2002 → last completed week, ties excluded), with six inputs:

* **Elo logit** — the Elo win probability below, as a logit
* **Home offence success** — the home team's opponent-adjusted success
  rate above league average (weighted ridge over every offence-vs-defence
  game before this week; half-life 5 weeks, prior season at half weight)
* **Away offence success**, **home defence success** (allowed, so lower is
  better), **away defence success** — same fit
* **Success matchup net** — (home offence + away defence) − (away offence
  + home defence)

Success rate is nflverse's `success` flag (EPA > 0) over pass and run
plays. Walk-forward 2015–2025: log loss 0.63468 → 0.63007, +2.96 SE
paired; the clean 2024–25 window 0.62411 → 0.61853. Falls back to Elo
alone when `data/nfl/team_games.csv` is missing or more than 10 days
behind the spine's last completed game; the slate's `model` column says
which one each pick used. Full ablation and the metrics collected but not
promoted: [ADVANCED_METRICS.md](ADVANCED_METRICS.md).

**Elo** — the rating carries every situational edge:

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
closing line). It informs nothing on the site. The rest-of-season
simulation and the expected score still run on Elo alone; only the
pick's win probability carries the success-rate stage.

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

## What the inputs are measured to be worth

Increase in log loss when the input is taken away, on held-out seasons.
Two methods, because the models are two different kinds of thing: the
soccer outcome model is a fitted logistic, so its features can be
shuffled (permutation importance). An Elo engine has no feature matrix —
"home advantage" is a constant, not a column — so each component is
neutralised and the whole history replayed (ablation). An ablation folds
in everything downstream of it, so the two are not interchangeable and
the numbers are not comparable across the two tables.

Run `python -m data_jobs.build_importance` to refresh; the site shows the
same numbers under *What its inputs are worth* on `/models`.

**Club soccer outcome model** — permutation, 7,827 holdout matches
(2024-25 onward), baseline log loss 1.01755:

| feature | worth | note |
|---|---|---|
| `elo_gap` | **+0.048** | ±0.002 |
| `value_diff_z` | +0.011 | ±0.001 — the only economics feature doing any work |
| `sot_net_diff` | +0.006 | ±0.001 |
| `xg_net_diff` | +0.0001 | the Understat feed stops at 2025-01-04 |
| `spend_diff_z` | 0.000 | **constant over the window** — no data |
| `net_diff_z` | 0.000 | **constant over the window** — no data |
| `wage_diff_z` | 0.000 | **constant over the window** — no data |

**NFL Elo** — ablation, 586 holdout games (2024 onward), baseline 0.62424:

| component | worth | note |
|---|---|---|
| Margin of victory | **+0.025** | margin ignored ⇒ every win counts the same |
| Season regression | +0.019 | ratings carried over untouched |
| Home advantage | +0.002 | the +48 is worth very little out of sample |
| Postseason K weight | 0.000 | tuned to 1.0, so it is already switched off |
| Rest (bye week) | **−0.0006** | the model scores *better* without the +20 |

**College football Elo** — ablation, 2,038 holdout games, baseline 0.48462:

| component | worth | note |
|---|---|---|
| FCS opponent rating | **+0.038** | all of FCS pooled as one 950-rated team |
| Margin of victory | +0.037 | |
| Season regression | +0.028 | |
| Conference regression | +0.011 | realignment handling |
| Home advantage | +0.011 | |
| FBS entry rating | +0.001 | few programs arrive |

**MLB Elo** — ablation, 34,665 games (2012 onward), baseline 0.67965:

| component | worth | note |
|---|---|---|
| Season carryover | +0.003 | |
| Home advantage | +0.002 | |
| Margin of victory | +0.001 | |

A coin flip is 0.693 and always-picking-home is ~0.691, so the whole MLB
rating engine buys about 0.013 nats. Baseball at the game level is close
to a coin flip and the model says so. The starting-pitcher, rest and
travel adjustments are applied by the daily pipeline rather than the
replay, so they are **not** in that table yet — a gap, not a claim that
they do nothing.

---

## What's missing, and what it would cost

The measured table above is also the critique: three of soccer's seven
features contribute nothing, the NFL's rest bonus is worth less than
nothing, and college football's crudest component — one synthetic rating
standing in for all of FCS — is its most load-bearing. Ranked by what I
would do first, with the cost of each.

**Cheap, and already blocked on nothing.**

| # | Model | Change | Why | Cost |
|---|---|---|---|---|
| 1 | Soccer | Refresh the transfer aggregates past 2022-23 | `spend_diff_z` and `net_diff_z` are constant on every holdout row *and* every live slate. Two of seven features are decoration until this lands | a fetcher run, no modelling |
| 2 | Soccer | Add the wage column to the market-value uploads | `wage_diff_z` has a fitted weight of 0.000 for want of data | an upload |
| 3 | NFL | Drop or re-tune the bye-week bonus | Ablation says the model is *better* without it (−0.0006 on holdout). It was tuned to +20 on 2005-23; the effect has decayed | one tuner run |
| 4 | Soccer | Corners, cards and fouls | The football-data.co.uk files the shots fetcher already downloads carry HC/AC, HY/AY, HF/AF — they are being parsed and thrown away | one fetcher column change + the usual holdout test |
| 5 | CFB | Rate FCS opponents individually, or in tiers | The pooled 950 is the single biggest component in the ablation, which means ~13% of the schedule is handled by the crudest thing in the model | schedule already carries the opponent; needs an FCS rating pool |
| 6 | Soccer | Rest days and fixture congestion | Computable from results.csv alone (days since last match, midweek European tie). A real effect the model cannot currently see | no new data |

**The DVOA question.** DVOA itself is proprietary (FTN/Football
Outsiders, paywalled, no API), so it cannot go in. The honest open
analogue is **EPA per play** with offensive and defensive splits, plus
success rate — and both football models already read their spine from the
same publishers that ship it: nflverse for the NFL, cfbfastR-data for
college, both plain CSVs over raw.githubusercontent.com with no API key.
So the data is a fetcher away for both.

What is *not* a fetcher away is the decision it forces. The live NFL and
CFB boards are deliberately Elo alone: one rating, betting-blind, no
fitted layer. Adding EPA means either

- **(a)** folding it into the rating (an EPA-adjusted K or a preseason
  prior, the way the MLB engine prices its starting pitcher) — keeps the
  board a rating, modest gain, cheap; or
- **(b)** a fitted outcome model on top of Elo + EPA, the way club soccer
  already works — the bigger win and the bigger change, because it makes
  the football board a model-of-a-rating rather than a rating.

`NFL/model/v2/` is already a 45-feature LightGBM that does (b) and beats
Elo on holdout, but it also sees the closing line, so it cannot feed a
betting-blind board as-is. Stripping the market features out of it and
walking it forward honestly is the real project here, and it is a project,
not an afternoon.

**Aggregate margin.** Worth separating from the per-game MOV multiplier:
season point differential (and its Pythagorean win expectation) is a
well-established predictor that Elo only sees one game at a time, damped
and capped. "Rating vs point differential" residual is a cheap feature for
either football model and needs no new data.

**Other sports, briefly.** MLB's starting-pitcher, rest and travel layers
are applied by the daily pipeline rather than the replay, so they are not
in the ablation at all — closing that is the first job there, ahead of any
new feature (park factors, bullpen, lineup handedness, weather). Soccer
possession and passing need a new publisher (FBref), which is a scraper
and a terms question rather than a CSV, so it sits below everything above.

**Two results-derived candidates were tested rather than argued about**,
since both come free of new data. Neither ships. Run
`python -m soccer.clubs.model.eval_goals_form` to reproduce:

| candidate | coverage | holdout splits improved | weakest split | verdict |
|---|---|---|---|---|
| `goals_net_diff` (rolling goal difference) | 92.3% of rows | 3/3 | +0.4 SE | under the bar |
| `surprise_net_diff` (result minus Elo expectation) | 96.3% | 3/3 | +0.3 SE | under the bar |
| both together | — | 2/3 | **−0.6 SE** | worse than either |

The bar is the one the shots layer cleared: every split improves *and*
the paired test reaches about +2 SE. Both candidates move every split in
the right direction with a stable positive coefficient, so there is a
little signal there — just not enough to distinguish from noise on
58,908 matches, which is the expected answer. Elo is *built* from these
results, so the rating already contains most of what they say.

The third row is the interesting one. Put in together, the coefficients
flip — goal form +0.068, surprise −0.126 — and the pair scores worse than
either alone on the earliest split. They are near-substitutes fighting
each other, which is direct evidence that "recent goal difference" and
"recent overperformance against the rating" are the same fact twice.
A third variant along those lines should be expected to do the same.

What this says about the "what about goals?" question: the goals are
already in there, via the rating. The form features that earn their place
here are the ones carrying information the result log does *not* have —
which is why shots-on-target works (+2.9 SE when it was added) and why
the remaining candidates worth chasing are chance quality, possession and
efficiency rather than another summary of the score line.

Both modules stay in the repo next to `eval_goals_form.py`, which prints
the paired test and a SHIP / does-not-clear verdict, so the next
candidate is one command rather than an argument.

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

So the site's projection draws **simulated seasons and never an average
of them**: each side's median run (the one simulated season that finished
mid-distribution) in bold, three whole seasons shared across sides behind
it, and the percentile band on hover. Anything that reports the mean
instead — a chart, a table column, a summary line — is reporting the one
season in which nothing happens, and should be treated as a bug.
Exported as `elo_projection.samples` next to `elo_projection.clubs` /
`elo_projection.teams` in each sport's `latest.json`; produced in
`soccer/clubs/daily/simulate.py`, `NFL/daily/simulate.py` and
`CFB/daily/simulate.py`.
