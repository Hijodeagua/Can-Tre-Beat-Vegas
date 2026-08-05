# Starter prompt — MLB model

Paste everything below the line into a fresh window.

---

I want to build an MLB model, starting from an Elo rating engine. I've done this
for NFL in `Hijodeagua/Can-Tre-Beat-Vegas` — read `NFL/model/v2/README.md` there
if you can reach it for the evaluation conventions I like to work with, but
don't assume any of the conclusions carry over. Baseball is a different sport
with 9x the sample size and a different market structure.

## Stage 1 — get the data

Scrape and cache **complete game results from 2005 to present**. Build a proper
local cache with a refresh path, the way `data_jobs/` works in the NFL repo —
I want to re-run this weekly without re-downloading history.

Check what's actually reachable before planning around it. Likely candidates:

- **Retrosheet game logs** — complete game-level history, free, goes back
  further than I need. Probably the spine.
- **MLB Stats API** — official, covers 2005+, has probable pitchers and lineups.
- **pybaseball** — wraps Baseball Savant, FanGraphs, Baseball Reference.
- **Baseball Savant / Statcast** — only 2015+, so plan for a two-era dataset:
  traditional metrics 2005-2014, Statcast layered on from 2015.

Per game I want at minimum: date, teams, home/away, final score, innings (extras
matter), park, and **both starting pitchers**. Beyond that, capture whatever is
cheap to grab now and expensive to backfill later.

### Metrics worth pulling while you're in there

Team level: runs scored/allowed, run differential, Pythagorean win expectation,
team wOBA / wRC+ / OPS, team ERA / FIP.

Pitching — this is where baseball concentrates its signal: starting pitcher
identity and their rolling ERA / FIP / xFIP / K-BB%, days of rest, times through
the order, and **bullpen usage in the prior 2-3 days** (a gassed pen is a real
effect and it's knowable in advance).

Context: park factors, weather (wind direction matters enormously at some
parks), home plate umpire and their strike-zone tendencies, day/night, travel
distance and time-zone changes, day-after-night games, getaway days.

Statcast era (2015+): xwOBA, barrel rate, exit velocity, xERA — the
expected-vs-actual gap is its own feature class and worth having separated from
results.

Also pull **betting odds** — moneyline, runline, totals. This repo already has
an Odds API job on a schedule; reuse the pattern. Capture **opening lines as
well as closing**, from the start. Historical opening lines are hard to backfill
and easy to record going forward.

## Stage 2 — Elo

Build a MOV-adjusted Elo engine on the 2005+ results. Baseball needs some
specific handling that the NFL version doesn't:

- **Home advantage is much smaller** than in football — fit it from the data
  rather than porting a constant.
- **The starting pitcher is a first-class input**, not an adjustment bolted on
  afterwards. FiveThirtyEight's MLB Elo did this well: a team rating plus a
  pitcher rating, with the pitcher's contribution rolling over their recent
  starts. Worth building the pitcher-aware version alongside the plain one and
  comparing.
- **Margin of victory is noisy** — a 10-run blowout says less about team quality
  than a 10-point NFL win. Consider damping MOV harder, and test it.
- Season-to-season regression toward the mean, and a sensible K.
- Games are near-daily, so ratings update constantly. Watch that a hot streak
  doesn't run the rating away from true talent.

Show me the Elo's calibration and accuracy on its own before anything is layered
on top. I want to see a few variants compared, the way the NFL build compared
four.

## Stage 3 — features and models

Once Elo exists, build the feature matrix and we'll talk about model types. I'm
interested in what the features say as much as what any model scores — feature
importance and ablation, not just a leaderboard.

## House rules for evaluation

These are conventions, not predictions about outcomes. They exist because
without them you can't tell a result from an artifact.

1. **Walk-forward validation.** One retrain per season: fit before S−1,
   calibrate on S−1, score S. Never a single random split.
2. **Leakage discipline.** Every rolling stat shifted, every "who started"
   determined from strictly-prior games. Probable pitchers are announced in
   advance so they're legal — verify the timestamp, don't assume.
3. **Report against a baseline, always.** Model accuracy next to the closing
   line's, next to a naive "bet the favourite" rule, and next to the
   **agreement rate** with that rule. Two models can score identically and be
   doing entirely different things.
4. **Price everything at real odds.** Baseball moneylines run from −350 to +300,
   so break-even moves with every bet — there's no fixed threshold. A hit rate
   without a price attached doesn't mean anything.
5. **Cluster standard errors to the right unit.** Games on the same day share
   weather, umpires and news. Bootstrap days or weeks. For anything you'd
   actually deploy, also compute a season-level statistic — that's the unit you
   bet at.
6. **Correct for multiple testing** when sweeping hypotheses, and report what
   didn't survive alongside what did.
7. **Assert join coverage after every merge, with a test.** Team codes and
   player IDs disagree across sources; a silent 5% drop is very easy to ship and
   very hard to notice later.
8. **Say what the result is.** If something's a coin flip, show me the interval
   and say so. A clean null is a useful result and I'd rather have it than a
   flattering number.

## Working style

Show me outputs at each stage. Propose before you build, and don't commit until
I've signed off — that's how the NFL build ran and it worked well.

Start with Stage 1 and tell me what you can and can't reach.
