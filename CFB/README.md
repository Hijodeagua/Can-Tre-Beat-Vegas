# College football model

The FBS sibling of `mlb/` and `soccer/clubs/`: a betting-blind Elo over every
FBS-involved game since 2001, a daily pipeline that grades what played,
predicts what's next and Monte Carlos the rest of the season, the model card
at `whosyurgoat.app/vegas/cfb`, and a twice-weekly update email. Same
walk-forward discipline, same "the baseline is the yardstick" honesty.

## Layout

```
CFB/
├── data/
│   ├── fetch_schedule.py   # cfbfastR-data (ESPN-derived) -> data/college_football/games.csv
│   └── teams.py            # name aliases, conference short names, the pooled-FCS constant
├── model/
│   ├── elo.py              # CfbEloEngine + replay(); the college-specific rules live here
│   ├── tune.py             # coordinate-descent grid search -> artifacts/tuned_params.json
│   └── artifacts/tuned_params.json
├── daily/                  # the daily pipeline — see daily/README.md
├── ingest.py               # Sports Reference CSV exports -> tidy aggregates (the manual-export path)
└── DATA_PULL_PLAN.md       # the original CFR shopping list; superseded for the spine, still the Tier-2 roadmap

data/college_football/
├── games.csv               # the spine: 2001-present, FBS-involved games, home perspective
├── predictions/            # slate_{D}.csv per run + grades.csv, the running ledger
├── raw/, agg/              # Sports Reference exports + ingest.py outputs (scoring stats)
```

```bash
python3 -m CFB.data.fetch_schedule --all   # rebuild the spine (current season only without --all)
python3 -m CFB.model.tune                  # refit the parameters (3 s)
python3 -m CFB.model.elo                   # print the current top 25
python3 -m CFB.daily.run --skip-fetch      # one full daily run against committed data
python3 -m pytest tests/test_cfb_elo.py tests/test_cfb_daily.py tests/test_cfb_ingest.py
```

## Data

`data/college_football/games.csv` comes from
[cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data), one
ESPN-derived CSV per season served off raw.githubusercontent.com with no
key — the college analogue of the nflverse file the NFL model reads. It has
what the plan's Sports Reference exports don't: a **per-season conference
for both teams** (realignment-proof), an **FBS/FCS division tag** for both
teams, the neutral-site flag, and scores that update through the current
season. Kickoff dates are converted to US/Eastern so a 10 pm ET Saturday
game is graded as Saturday.

Known gaps, so nobody re-discovers them: the upstream files carry bowls and
the CFP only from 2024 (older seasons are regular season only), and the
neutral flag is blank for 2001-02. Neither matters for a regular-season
walk-forward model. ESPN's public scoreboard is read for the trailing week
as a best-effort fill-in when cfbfastR's nightly reprocess lags.

## The engine (`model/elo.py`)

Shared skeleton with the other two engines (logistic expectation, home
edge in Elo points, K scaled by a ln-damped margin that shrinks when the
favourite wins), plus the four things `DATA_PULL_PLAN.md` §1 said college
needs:

| Rule | What it does |
|---|---|
| Conference-aware regression | Each August a program regresses toward `conf_weight` × its **new** conference's mean + (1 − w) × 1500. Independents and programs that left FBS regress toward 1500. |
| Pooled FCS opponent | Every non-FBS program is one fixed rating; the game updates only the FBS side. |
| FBS entry rating | A program's first FBS game starts it well below 1500. |
| Margin cap | \|margin\| is clipped before the multiplier. |

Tuned by `model/tune.py` on one-step-ahead log loss over every FBS-involved
game, seasons 2005-2023 (2001-04 are burn-in from the flat 1500 start),
with 2024-25 held out:

| | K | home edge | regression | conf weight | FCS pool | entry | cap |
|---|---|---|---|---|---|---|---|
| plan's guess | 32 | 70 | 0.40 | — | 1200 | — | 35 |
| **tuned** | **35** | **50** | **0.30** | **0.75** | **950** | **1250** | **80** |

| log loss | 2005-23 (fit) | 2024-25 (holdout, 1,861 games) |
|---|---|---|
| plan's guesses | 0.5052 | 0.5106 |
| tuned | 0.4954 | **0.4988** |

Read that holdout number against the right yardstick. On the same 2024-25
games, always-pick-home (p = 0.632, coin flip at neutral sites) scores
0.652. Restricted to the 1,614 FBS-vs-FBS games the model runs 0.554
against 0.678 for always-home, picking 70.1% straight up. None of it is
comparable to the NFL's 0.61 or MLB's 0.68 — college has far more talent
spread and far more games that are decided before kickoff — which is why
the pipeline's graded ledger carries an FBS-vs-FBS-only row.

The margin fit in `daily/scoring.py` comes out at **~18 Elo per point** of
spread (the plan guessed 27-32 by analogy with the NFL's 25; the tuned K
spreads ratings wider).

## Known limitations (in priority order)

1. **One FCS rating.** North Dakota State and an FCS bottom-feeder are the
   same opponent; beating NDSU under-credits, losing to it over-punishes.
   The plan's §3.4 accepted this; revisit with an FCS spine if it shows up
   in the ledger.
2. **No preseason talent prior.** Regression is toward the conference
   mean; returning production, recruiting composite and coaching changes
   (plan §4, §6) are the next inputs, and CFBD is the source for them.
3. **No playoff model.** The 12-team field is a committee ranking. The
   sim reports expected wins, bowl / undefeated odds and conference titles
   and deliberately stops there.
4. **No market benchmark.** The NCAAF Odds API entry (plan §7) is a
   four-line config change with a real quota cost — see the root README.
