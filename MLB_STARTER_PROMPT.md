# Starter prompt — MLB model

Paste everything below the line into a fresh window. It carries the methodology
this repo arrived at over the NFL build, plus the places baseball differs enough
that copying the NFL approach would be wrong.

---

I want to build an MLB betting model. I've already done this for NFL in
`Hijodeagua/Can-Tre-Beat-Vegas` and the conclusions there were hard-won — I want
to start from those lessons rather than rediscover them. Read
`NFL/model/v2/README.md` in that repo first if you can reach it; it is the
methodology record.

## What I actually want to know

Not "can you predict baseball games." The market already does that well. The
question is **where, if anywhere, the market is mispriced** — and whether any of
it survives contact with the vig.

Start with an inventory and a baseline, not a model. I want to know what the
data supports before anything gets fit.

## Non-negotiable methodology

These are all things the NFL build got wrong at least once before getting right.

1. **The benchmark is the market, never 50% and never "did we predict the
   winner."** Every accuracy number must be reported next to what the closing
   line achieved on the same games, plus a naive control (bet the favourite).
   Also report **agreement rate** with that control — the NFL model hit 66.5%
   and agreed with "pick the favourite" 98% of the time, so its own contribution
   was 13 games in 3,018. Accuracy alone hid that completely.

2. **Hit rate is not ROI.** Always price bets at real odds. The NFL model's most
   confident picks hit 72.5% and returned **−10.8%**, because they were heavy
   favourites averaging 1.31 decimal. This matters far more in baseball, where
   moneyline prices range from −350 to +300 and break-even moves with every bet.
   There is no fixed 52.38% here — compute break-even per bet.

3. **Walk-forward only.** One retrain per season: fit on everything before S−1,
   calibrate on S−1, score S. Never a single random split.

4. **Cluster your standard errors, and use the deployment unit.** Games in the
   same day/week share weather, umpires, news. Bootstrap whole days or weeks,
   not games. Then also compute a **season-level t**, because you deploy a
   strategy once a year. The NFL early-season effect had a game-level p of 0.009
   and a season-level p of 0.058 — the gap between those *was* the finding.

5. **Correct for multiple testing and say what didn't survive.** BH-FDR and
   Bonferroni across the whole family of hypotheses tested. Report the ones that
   fail, not just the winner.

6. **Leakage discipline.** Every rolling stat shifted, every "who started"
   determined from strictly-prior games only. The NFL build shipped a QB feature
   that inferred the starter from whole-season counts — 13.6% of team-games
   carried a flag that used future information.

7. **Check join coverage explicitly, with a test.** A team-code mismatch
   silently dropped 5% of team-games in the NFL work — and not a random 5%,
   entire franchise-eras. Assert coverage after every merge.

8. **Never diagnose your forecast by conditioning on the market's.** Bucketing
   residuals by *Vegas's* favourite size produced a large, significant, entirely
   fake bias. Bucket by your own prediction, or use a test that conditions on
   neither.

9. **Report what the result is, not what I want it to be.** If it's a coin flip,
   say so plainly and show the interval. I would much rather have a clean null
   than a flattering number.

## Things the NFL work established that probably transfer

- **Model family barely matters.** Six regressors landed within 0.13 points of
  each other; calibrated logistic/ridge matched or beat every tree ensemble.
  Don't spend time on architecture until features are exhausted. Baseball has
  ~2,430 games a season vs the NFL's 272, so trees have a real chance here —
  but prove it against a calibrated linear baseline, don't assume it.
- **Calibration buys unbiasedness, not accuracy.** Worth doing, but don't expect
  it to improve error.
- **Line shopping was the only confirmed edge**: hold 4.44% → 2.08%, break-even
  52.22% → 51.04%, worth ~+2.85 pts of ROI per bet with no model skill required.
  Baseball's dime lines mean the hold is lower but the shopping edge is still
  likely the most reliable thing available. **Do this first — it pays before any
  model exists.**
- **Real effects can be too small to sell.** Market overreaction to team form
  was real, stable across eras, and worth 51.4% against a 52.4% break-even.

## Where baseball is genuinely different — don't copy the NFL blindly

- **The market structure is different.** No spread. Moneyline is primary, plus
  the runline (−1.5/+1.5) and totals. The runline is *not* a spread — it's a
  fixed handicap with a moving price, so it behaves like a different bet.
- **Per-game predictability is far lower.** A great MLB team wins ~60% of games;
  a great NFL team wins ~80%. Expect small edges, and much bigger samples to
  find them.
- **The starting pitcher is the single dominant input** and is announced in
  advance — this is the closest thing to a tradable information window. Bullpen
  usage over the prior 2–3 days is the second.
- **Statcast (2015+) separates results from process** — xwOBA, barrel rate,
  expected ERA vs actual. This is the most promising feature class, because the
  gap between what happened and what should have happened is exactly where a
  market that anchors on results would misprice. The NFL analogue (elite QBs
  fully reverting after bad games while ordinary QBs don't) was the strongest
  signal found, season t = 3.23. **I'd start here.**
- **Park factors and weather are real and large**, unlike NFL where weather
  barely mattered.
- **Umpires** have measurable, persistent strike-zone tendencies and are
  announced pre-game.
- **Lineup cards** matter — rest days, platoon splits, September call-ups.
- **Schedule fatigue** is heavier: day-after-night, getaway days, long road
  trips, travel across time zones.

## Data

Please check what's actually reachable before planning around anything.
Candidates: `pybaseball` (wraps Baseball Savant / FanGraphs / Baseball
Reference), Retrosheet game logs, the MLB Stats API. For odds, this repo already
pulls The Odds API on a schedule — reuse that pattern, and capture **opening
lines**, not just closing. Every NFL result was against closing lines, which is
the hardest version of the question, and I never got the opening-line comparison
built. Do it from the start here.

## What I want from you first

Before writing a model:

1. Confirm what data you can actually reach, and say plainly what you can't.
2. Build the game spine + a market baseline: how good is the closing moneyline,
   what's the hold, what does "bet the favourite" return.
3. Show me a leakage-safe feature inventory and where each feature comes from.
4. Then propose the model plan — and **wait for my sign-off before fitting
   anything.**

Show me outputs at each stage. Don't commit until I've signed off — that's how
the NFL build ran and it worked well.
