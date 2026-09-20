# MLS 2026 — Supporters' Shield, Playoffs and MLS Cup

*Elo forecast as of 2026-09-20 · 357 of 510 regular-season matches played · 20,000 simulated seasons*

Generated from `soccer/clubs/model/artifacts/mls_forecast.json`
(`python -m soccer.clubs.model.mls_forecast`) and
`artifacts/mls_backtest.json` (`python -m soccer.clubs.model.backtest_mls`).

---

## TL;DR

1. **Nashville SC is a heavy Shield favourite (72.5%) and only a
   moderate MLS Cup favourite (17.3%).** Nine points clear in the East with ten
   to play, it takes the top seed in 87% of simulations — and still
   loses the Cup in more than four seasons out of five, because the top
   seed's reward is a best-of-3 against a Wild Card survivor, not a bye.
   The Shield is 34 matches of signal; the Cup is at most six matches of
   noise.
2. **The Cup race is a three-way tie between clubs that are not close in
   the table.** Nashville 17.3%, Vancouver 17.0%, Inter Miami 16.2% —
   Miami is 9 points behind Nashville and 2 behind Vancouver, and has the
   highest Elo in the league (1602). Over a six-match bracket, rating
   matters and banked points barely do.
3. **Elo and the table disagree, and the disagreement is the forecast's
   main content.** Inter Miami (1602 Elo, 3rd in the East) is given a
   better Cup chance than Houston (1513, 2nd in the West) by a factor of
   2.7 despite a worse record. LAFC is the clearest case: 4th in the West
   on points, but its 1572 rating makes it the West's second most likely
   Cup winner at 8.7%.
4. **The Eastern playoff race is effectively settled at the top and open
   at the bottom.** Five Eastern clubs are above 99% to qualify; the
   last spot or two is a genuine scrap between Toronto (28.5%), D.C.
   United (25.9%), New York Red Bulls (24.0%) and Columbus (13.4%), with
   New York City FC (61.9%) not yet clear of them. The West's bubble is
   wider and lower: seven clubs sit between 17.8% and 59.7%.
5. **Sporting Kansas City is eliminated** — 15 points from 23 matches,
   0.0% in 20,000 simulations. It is the only club in either conference
   the model has already ruled out.
6. **Validated walk-forward on 2025**, the only completed 30-club
   season. At the same stage the 2026 season is at now, expected-points
   MAE was **3.13** with **+0.09** bias, and playoff-qualification Brier
   was **0.044** against a 0.240 base-rate baseline. The model's top
   Shield pick at that cutoff (Philadelphia Union, 32.1%) won it. One
   season is an anecdote for the Shield and Cup lines specifically — see
   §5.

---

## 1. Method

Same machinery as the five European leagues' rest-of-season forecasts
(`soccer/clubs/daily/simulate.py`), pointed at MLS:

| Piece | Setting |
|---|---|
| Ratings | The MLS Elo pool (`soccer/clubs/model/elo.py`), replayed from 2013 over every MLS match in `results.csv` |
| Tuned parameters | K = 10, home advantage = 75, season regression = 0.20 (`artifacts/tuned_params.json`) |
| Match model | Independent Poisson per side; expected margin linear in the Elo home expectancy, MLS goal total from the last two seasons |
| Live ratings | Every simulated season carries its own copy of the ratings and updates them with the same K and margin-of-victory rules as results come in |
| Replays | 20,000 seasons, each carried through the full playoff bracket |

Home advantage enters the forecast exactly the way it enters the ratings:
one number per club, plus the pool's tuned +75 on whichever side is at
home. That +75 ties La Liga for the largest of the six pools the repo
rates (the Premier League and Bundesliga are +60, Serie A and Ligue 1
+45) — travel in a continent-wide league is a real effect, and the
tuner finds it without being told to look.

### 1.1 The schedule had to be rebuilt

MLS is the one league here whose remaining fixtures cannot simply be
read. The upstream (philo92/mls-elo) is an Elo-history log of *played*
matches — there is no fixture list in it, which is why MLS has had
ratings but no forecast until now.

The format pins most of the run-in down anyway. MLS is 30 clubs, 15 per
conference, 34 matches each: 28 against conference rivals (each one once
home and once away) and 6 cross-conference, 3 home and 3 away against six
different opponents.

- **128 remaining conference fixtures are exact.** A fixture is still
  owed if and only if that ordered (home, away) pair has not been played.
- **25 remaining cross-conference fixtures are not.** Which opponents a
  club draws is a scheduling decision the log never reveals; only each
  club's remaining count of them, split home and away, is recoverable.

So the two are handled differently, and that difference is deliberate:
the conference fixtures go into every simulation unchanged, while the
cross-conference ones are drawn fresh in each simulation from the
pairings consistent with the quotas. The genuine uncertainty about those
25 matches ends up in the spread of the odds instead of in one invented
schedule that every run shares.

The reconstruction is checked, not assumed. `mls.verify_structure()`
re-runs every pipeline run and requires every club to land on exactly 34
matches with the two conferences' cross-conference quotas clearing
against each other; it currently passes on 2025 and 2026 and correctly
refuses 2023 and 2024, which were 29-club seasons with a different shape.
If MLS changes format, the pipeline publishes nothing rather than
publishing odds built on a schedule that cannot happen.

### 1.2 The bracket

2026 format, simulated match by match with the same Elo and scoreline
model:

| Round | Format |
|---|---|
| Qualification | Top 9 per conference |
| Wild Card | 8 hosts 9, single match |
| Round One | Best-of-3; higher seed hosts games 1 and 3 |
| Conference Semifinal / Final | Single match at the higher seed |
| MLS Cup | Single match, hosted by the finalist with the better regular-season record |

A drawn playoff match goes to a shootout, which the model treats as a
coin flip. The home side's real edge is already priced into the 90
minutes through `home_advantage`; giving it a second edge in the shootout
would be counting it twice.

---

## 2. Eastern Conference

*Seed = expected finishing position in the conference. WC = probability
of finishing 8th or 9th, i.e. having to win a Wild Card match first.*

| Seed | Club | Elo | GP | Pts | xPts | Playoffs | WC | 1 seed | Conf | Cup | Shield |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.14 | **Nashville SC** | 1572 | 24 | 53 | 69.6 | 100.0% | 0.0% | **87.3%** | 27.4% | **17.3%** | **72.5%** |
| 2.20 | **Inter Miami CF** | **1602** | 24 | 44 | 62.4 | 100.0% | 0.0% | 11.6% | 27.2% | 16.2% | 7.9% |
| 3.96 | Chicago Fire FC | 1518 | 23 | 39 | 55.1 | 99.7% | 2.3% | 0.9% | 10.1% | 4.6% | 0.4% |
| 4.58 | Charlotte FC | 1526 | 24 | 39 | 53.1 | 99.5% | 4.3% | 0.1% | 9.6% | 4.3% | 0.1% |
| 4.67 | New England Revolution | 1481 | 24 | 40 | 52.5 | 99.5% | 5.1% | 0.1% | 6.0% | 2.3% | 0.1% |
| 6.78 | FC Cincinnati | 1522 | 23 | 31 | 48.0 | 88.5% | 22.1% | 0.0% | 6.3% | 2.8% | 0.0% |
| 7.67 | Orlando City SC | 1509 | 24 | 31 | 45.7 | 79.3% | 29.7% | 0.0% | 3.9% | 1.5% | 0.0% |
| 8.01 | Philadelphia Union | 1537 | 24 | 30 | 44.9 | 75.5% | 32.9% | 0.0% | 4.7% | 1.8% | 0.0% |
| 8.91 | New York City FC | 1513 | 24 | 29 | 43.5 | 61.9% | 34.4% | 0.0% | 2.9% | 1.2% | 0.0% |
| 10.74 | Toronto FC | 1444 | 24 | 29 | 40.3 | 28.5% | 20.1% | 0.0% | 0.5% | 0.2% | 0.0% |
| 10.90 | D.C. United | 1418 | 23 | 28 | 39.9 | 25.9% | 17.5% | 0.0% | 0.4% | 0.1% | 0.0% |
| 11.03 | New York Red Bulls | 1457 | 24 | 27 | 39.4 | 24.0% | 17.6% | 0.0% | 0.5% | 0.2% | 0.0% |
| 11.92 | Columbus Crew | 1500 | 24 | 23 | 37.1 | 13.4% | 10.4% | 0.0% | 0.5% | 0.2% | 0.0% |
| 13.46 | Atlanta United FC | 1421 | 24 | 22 | 33.4 | 3.1% | 2.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| 14.03 | CF Montréal | 1415 | 24 | 21 | 31.9 | 1.2% | 1.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Nashville's two seasons.** It takes the top seed 87% of the time and
wins the Shield 72.5% of the time, and then wins MLS Cup 17.3% — which
is 63% of the times it wins the East, and 20% of the times it is the top
seed. Those are not inconsistent numbers, they are the format: seven
points over ten matches is close to decisive, and four knockout rounds
are not.

**Inter Miami is the model's disagreement with the table.** It sits 3rd
on points and 1st on rating, and the forecast splits the difference —
2.20 expected seed, but a Cup probability (16.2%) within a point of
Nashville's. Every extra round the bracket runs is a round in which the
rating matters more than the nine points it gave away.

**Columbus Crew is the East's biggest rating-to-points mismatch after
Philadelphia.** A 1500 Elo — 16th in the league by rating — attached to
23 points from 24 matches, which is 27th. The model gives it 13.4% to
qualify: far more than its points suggest, far less than its rating
alone would, because ten matches is not much time to make up four points
on three clubs at once.

---

## 3. Western Conference

| Seed | Club | Elo | GP | Pts | xPts | Playoffs | WC | 1 seed | Conf | Cup | Shield |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.40 | **Vancouver Whitecaps FC** | 1587 | 23 | 46 | 63.9 | 100.0% | 0.0% | **74.5%** | **30.2%** | **17.0%** | 17.5% |
| 3.04 | Houston Dynamo FC | 1513 | 24 | 43 | 57.7 | 99.9% | 1.1% | 12.2% | 13.4% | 6.0% | 1.0% |
| 3.61 | FC Dallas | 1521 | 24 | 40 | 56.5 | 99.4% | 2.4% | 7.3% | 13.2% | 5.8% | 0.3% |
| 4.11 | Los Angeles FC | 1572 | 25 | 40 | 54.9 | 99.1% | 3.4% | 3.5% | 17.3% | 8.7% | 0.2% |
| 5.58 | St. Louis City SC | 1495 | 24 | 38 | 52.2 | 94.3% | 12.0% | 1.3% | 7.0% | 2.8% | 0.0% |
| 5.63 | San Jose Earthquakes | 1471 | 24 | 38 | 51.6 | 93.9% | 13.4% | 1.1% | 5.4% | 2.1% | 0.0% |
| 8.84 | Portland Timbers | 1485 | 24 | 32 | 45.6 | 59.7% | 28.4% | 0.0% | 2.5% | 1.0% | 0.0% |
| 8.93 | Colorado Rapids | 1464 | 24 | 32 | 45.2 | 58.1% | 27.4% | 0.0% | 2.0% | 0.7% | 0.0% |
| 9.47 | Real Salt Lake | 1478 | 23 | 29 | 44.4 | 49.2% | 25.0% | 0.0% | 1.9% | 0.7% | 0.0% |
| 9.87 | Seattle Sounders FC | 1509 | 22 | 27 | 44.0 | 43.2% | 23.0% | 0.0% | 2.5% | 1.0% | 0.0% |
| 9.87 | San Diego FC | 1511 | 24 | 30 | 43.8 | 42.9% | 25.1% | 0.0% | 2.2% | 0.8% | 0.0% |
| 10.23 | Minnesota United FC | 1504 | 24 | 29 | 43.5 | 36.5% | 21.8% | 0.0% | 1.7% | 0.6% | 0.0% |
| 11.58 | LA Galaxy | 1466 | 25 | 29 | 40.9 | 17.8% | 12.5% | 0.0% | 0.5% | 0.2% | 0.0% |
| 12.88 | Austin FC | 1449 | 24 | 26 | 37.9 | 5.8% | 4.6% | 0.0% | 0.2% | 0.1% | 0.0% |
| 14.95 | Sporting Kansas City | 1373 | 23 | 15 | 25.7 | **0.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**LAFC is the West's rating story.** Fourth on points and fourth on
expected seed, but the second-highest rating in the conference — so the
model has it reaching MLS Cup more often than Houston or Dallas, both of
whom it currently trails, and gives it the West's second-best Cup number
at 8.7%. Seeding decides who hosts; rating decides who wins.

**Seattle has the most upside left.** Twenty-two matches played, the
fewest in the league, and a 1509 rating attached to 27 points. Twelve to
play is the most of anyone, which is why a club four points off the line
still gets 43.2%.

**The West bubble is seven clubs for three spots.** Portland (59.7%),
Colorado (58.1%), Real Salt Lake (49.2%), Seattle (43.2%), San Diego
(42.9%), Minnesota (36.5%) and LA Galaxy (17.8%) are competing for
seeds 7–9. Note how flat that is: six of the seven sit between 36% and
60%, which is the model saying it genuinely cannot separate them.

---

## 4. What the format does to the Shield favourite

The single most useful thing this forecast says is how weakly the
regular season translates into the Cup:

| Club | Shield | Wins conference | MLS Cup | P(Cup \| wins conference) |
|---|---:|---:|---:|---:|
| Nashville SC | 72.5% | 27.4% | 17.3% | 63.1% |
| Vancouver Whitecaps FC | 17.5% | 30.2% | 17.0% | 56.5% |
| Inter Miami CF | 7.9% | 27.2% | 16.2% | 59.8% |
| Los Angeles FC | 0.2% | 17.3% | 8.7% | 49.9% |
| Houston Dynamo FC | 1.0% | 13.4% | 6.0% | 44.5% |

Nashville is 4× more likely than Vancouver to win the Shield and
slightly *less* likely to win the Cup. Winning the East is worth more to
Nashville than to anyone else (63% to convert, because it almost always
hosts the final), but reaching that point requires surviving a best-of-3
and two single-elimination matches — and the top seed's Round One
opponent is a Wild Card winner that has already proved it can win a
knockout match.

The conference-title probabilities also show the East is the more
top-heavy half of the bracket: its top two combine for 54.5%, the
West's for 47.5%, even though the West's best club (Vancouver, 30.2%)
is individually the likeliest finalist in the league.

---

## 5. Does this model work? (walk-forward on 2025)

`python -m soccer.clubs.model.backtest_mls` reruns the whole forecast at
past cutoffs with every match after the cutoff hidden from the Elo
replay, the score calibration and the table alike, then scores it against
what actually happened.

| Cutoff | Played | Left | xPts MAE | Bias | Playoff Brier | Baseline | P(actual Shield winner) | P(actual Cup winner) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-05-11 | 179 | 331 | 5.96 | +0.32 | 0.0926 | 0.240 | 8.9% | 8.3% |
| 2025-06-13 | 255 | 255 | 4.81 | +0.02 | 0.1065 | 0.240 | 22.4% | 9.5% |
| 2025-07-26 | 364 | 146 | **3.13** | **+0.09** | **0.0437** | 0.240 | **32.1%** | 11.5% |

Read this as three things:

- **Expected points are unbiased and tighten as they should.** MAE falls
  5.96 → 4.81 → 3.13 as the remaining schedule shrinks, and the bias
  never exceeds +0.32 points. A points model that drifted would move
  every threshold in the table at once, so this is the number that most
  needed checking.
- **Playoff qualification carries real skill.** Brier 0.044 against a
  0.240 base-rate baseline at the last cutoff — an 82% reduction.
- **The Shield and Cup lines are anecdotes, not measurements.** The
  eventual Shield winner (Philadelphia Union) was the model's outright
  favourite at the last cutoff, and the eventual Cup winner (Inter
  Miami) was given 11.5%, near the top of a flat field. Both are single
  observations. One Cup winner cannot distinguish a good bracket model
  from a lucky one, and nothing here should be read as claiming
  otherwise.

**The scope limit, stated plainly.** MLS has been a 30-club,
15-per-conference league for exactly one completed season. The three
cutoffs above are correlated views of that one season, not three
independent seasons — 2023 and 2024 were 29-club seasons that the
structure check correctly refuses. This is enough to catch a broken
model; it is not enough to call the model calibrated.

There is also one acknowledged leak: the MLS pool's Elo hyperparameters
were tuned over a window that includes the backtested season. At K = 10
the effect is far below the noise from a single season, so it is
documented rather than fixed.

---

## 6. Known model limits

- **Draws are under-predicted by about 3 points.** On the 2026 matches
  played so far the independent-Poisson model implies 22.6% draws
  against 25.5% observed (and 49.5% home wins against 46.5%). This is
  the same deficit the repo documents for the European leagues. It
  slightly inflates the variance of simulated points totals, since a
  missing draw becomes a 3-point swing instead of a 1-point one — so
  the bubble probabilities here are, if anything, a touch flatter than
  the truth.
- **Tiebreakers stop at goals for.** MLS separates level clubs by
  points, wins, goal difference, goals for, and then disciplinary
  points and away/home goal splits. The first four are modeled; clubs
  still level after them are separated at random rather than by a
  pretend rule.
- **No injuries, no transfers, no congestion.** The rating is the whole
  team model. A club whose form changes for a reason the results have
  not shown yet is not handled.
- **The Elo chart is drawn against matches played, not dates.** The
  remaining fixtures have no dates, and estimating them from the
  season's own cadence is not accurate enough to plot against: run on
  2025 at the three-quarter mark, that estimate puts the final matchday
  on 12 September against an actual 18 October. Matches played is also
  the better axis here — MLS clubs are up to three games apart, and a
  date axis would show Seattle level with clubs that have already spent
  those games.
- **The cross-conference draw is sampled, not known.** Twenty-five of
  the 153 remaining matches have opponents the model assigns itself,
  consistent with each club's true remaining home/away counts. This is
  handled honestly (the uncertainty is in the spread) but it is not the
  same as knowing the schedule.

---

## 7. The bracket, and why it is seeded the way it is

The site draws the playoff bracket from these same simulations, and the
one judgement call in it is worth stating. Each seed slot has a most
likely occupant, but those are *marginals*: taken independently they put
Portland Timbers top of both the 8 and the 9 slot in the West and leave
another club out of the bracket altogether. Every one of those numbers is
correct and the bracket they compose is nonsense.

So the slots are filled by expected finishing position, which gives every
club exactly one, and each slot carries how often that club really
finishes on that exact seed. Those numbers are low on purpose — 87% for
Nashville at the 1 seed, but 13-27% through most of the middle of both
conferences — because a seed is a fine distinction and the model is
saying so. The runners-up for each slot sit underneath.

Read across rounds instead of down seeds and the format does the talking:

| Club | Qualify | Conf Semi | Conf Final | MLS Cup | Champion |
|---|---:|---:|---:|---:|---:|
| Nashville SC | 100% | 67% | 45% | 27% | 17% |
| Vancouver Whitecaps | 100% | 70% | 46% | 30% | 17% |
| Inter Miami CF | 100% | 70% | 47% | 27% | 16% |
| Los Angeles FC | 99% | 61% | 33% | 17% | 9% |
| Houston Dynamo FC | 100% | 55% | 28% | 13% | 6% |

A club that is certain to make the playoffs is a coin flip to survive
Round One, and the single likeliest MLS Cup pairing in the league
(Nashville v Vancouver) comes up in 8% of seasons.

---

## 8. Reproducing this

```bash
python -m soccer.clubs.data.fetch_mls          # refresh MLS results
python -m soccer.clubs.model.mls_forecast      # -> artifacts/mls_forecast.json
python -m soccer.clubs.model.backtest_mls      # -> artifacts/mls_backtest.json
python -m soccer.clubs.daily.run               # the whole pipeline, incl. this forecast
```

The daily pipeline publishes the same block to
`web/public/data/soccer/latest.json` under `mls_forecast`, where the
`/soccer` page's Forecasts tab renders it as a per-conference table, an
Elo chart per conference (season to date, and the projection out to 34
matches) and the playoff bracket. The tables also go into the
twice-weekly update email.
