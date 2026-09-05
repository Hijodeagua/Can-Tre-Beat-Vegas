# NFL Elo model

The NFL sibling of `CFB/`, `mlb/daily/` and `soccer/clubs/`: a
betting-blind Elo over every game since 1999, a daily pipeline that grades
what played, predicts the next week and Monte Carlos the rest of the
season through the playoff bracket, the model card at
`whosyurgoat.app/vegas/nfl`, and a Tue/Thu update email.

It is deliberately a different animal from `NFL/model/v2/`. That model
answers "can we beat the closing line?" (no — see its README) and is
allowed to see the market. This one never does: its only inputs are
scores, dates and rest days, so its picks and its own spread are the
thing to hold up *against* the market rather than something built on top
of it. The two share nothing but the schedule file.

## Layout

```
NFL/
├── elo/
│   ├── engine.py           # NflEloEngine + replay(); grown from model/v2/elo.py
│   ├── teams.py            # franchise continuity, divisions, names, week labels
│   ├── tune.py             # coordinate-descent grid search -> artifacts/tuned_params.json
│   └── artifacts/tuned_params.json
└── daily/                  # the daily pipeline — see daily/README.md

data/schedules/nflverse_games.csv   # the spine (shared with the LightGBM model)
data/nfl/predictions/               # slate_{D}.csv per run + grades.csv, the running ledger
web/public/data/nfl/                # latest.json + per-day history for the site
reports/nfl/                        # update.html per run, manifest, send ledger
```

```bash
python3 NFL/model/schedule.py --refresh    # pull the latest nflverse schedule
python3 -m NFL.elo.tune                    # refit the parameters (~1 s)
python3 -m NFL.elo.engine                  # print the current ratings
python3 -m NFL.daily.run --skip-fetch      # one full daily run against committed data
python3 -m pytest tests/test_nfl_elo.py tests/test_nfl_daily.py
```

## Data

`data/schedules/nflverse_games.csv` is the
[nflverse](https://github.com/nflverse/nfldata) games file the v2 model
already reads: one row per game from 1999, home perspective, with scores,
kickoff date and time, `location` (neutral flag), `game_type`
(REG / WC / DIV / CON / SB), each side's rest days, and the closing line.
The Elo reads scores, dates, the neutral flag, the game type and rest;
the line never reaches it.

nflverse labels each game with the abbreviation the franchise used at the
time (STL, SD, OAK). `teams.FRANCHISE` maps those onto LA / LAC / LV at
load, so a relocation carries its rating — a move does not reset a roster.

## The engine (`elo/engine.py`)

Shared skeleton with the other three engines (logistic expectation, home
edge in Elo points, K scaled by a ln-damped margin that shrinks when the
favourite wins, fractional regression to 1500 each off-season), plus what
the NFL specifically wants:

| Rule | What it does |
|---|---|
| Rest edge | A side with `REST_BONUS_DAYS`+ (10) days of rest — a bye — carries `rest_bonus` Elo into the game. |
| Playoff K | Postseason updates are scaled by `playoff_k_mult`. |
| Margin cap | \|margin\| is clipped at `margin_cap` before the multiplier. |
| Ties | Real in the NFL (about one a season): the outcome is 0.5, and the log loss scores both halves. |

Tuned by `elo/tune.py` on one-step-ahead log loss over every game,
regular season and playoffs, seasons 2005-2023 (1999-2004 are burn-in
from the flat 1500 start), with 2024-25 held out — the same split as the
college tuner so the two holdout numbers read the same way:

| | K | home edge | regression | playoff K | cap | rest bonus |
|---|---|---|---|---|---|---|
| v2 feature engine (538's constants) | 20 | 55 | 0.25 | 1.2 | — | — |
| **tuned** | **20** | **48** | **0.40** | **1.0** | **45** | **20** |

| log loss | 2005-23 (fit, 5,128 games) | 2024-25 (holdout, 570 games) |
|---|---|---|
| v2 constants | 0.6327 | 0.6273 |
| tuned | 0.6295 | **0.6236** |
| always pick home (p = 0.55) | — | 0.6910 |

Two of the moves are worth a sentence. The off-season regression tuned to
40%, well above 538's third: an NFL roster turns over more from January
to September than the copied constant assumed, and the tuner said so on
its own. And the playoff multiplier tuned to 1.0 — postseason results
carry no extra information about team strength once the margin
multiplier has had its say.

On the holdout the engine picks 66.1% straight up. For scale, the closing
line runs ~0.61 log loss and ~66% on the same kind of sample
(`NFL/model/v2/README.md`), so this is a rating system that lands a hair
behind the market with no knowledge of it — which is the same conclusion
every other measurement in this repo has reached, stated one more way.

The margin fit in `daily/scoring.py` comes out at **~24 Elo per point** of
spread (v2 assumed 25).

## Known limitations (in priority order)

1. **No quarterback adjustment.** Elo's worst blind spot — it cannot tell
   the backup is starting until the results say so. `NFL/model/v2/
   elo_variants.py` prices the starter from prior-season EPA and is worth
   ~0.005 log loss standalone; it needs the roster cache the weekly
   report refreshes, and it is the obvious next input here.
2. **Tiebreakers are approximate.** The season sim breaks standings ties
   on division / conference record and then at random; head-to-head,
   common games and strength of victory are not modeled. Late in a tight
   race the seeding odds are the numbers to trust least.
3. **Ties are not simulated.** One a season; the sim plays every game to
   a decision.
4. **No market benchmark on the card.** The odds feed already snapshots
   NFL lines twice a day; putting the closing line's pick next to the
   Elo's on the ledger is a grading change, not a data change.
