# Squad-quality features — plan and build notes

> **Status: built** (except PFF, dropped — no subscription). See
> "6. Build results" at the bottom for what the features actually did, and
> for the two constraints that changed the plan during implementation.


Scope: per-team, per-game roster-quality features — draft pedigree counts,
honors (All-Pro / Pro Bowl / NFL Top 100), and PFF-based player and unit
grades. This documents sources, what is scrapeable vs. manual, the feature
engineering, and the leakage rules. Also folds in the two pipeline questions
(QB starter handling, interim coaches) per the review notes at the bottom.

## 1. Data sources — what's free, what's scrapeable, what's manual

| Ingredient | Source | Access | Coverage | Verdict |
|---|---|---|---|---|
| Weekly active rosters (player-game level) | nflverse `roster_weekly` releases | free CSV download, no scraping | 2002-present | **automated** |
| Draft round per player | nflverse `draft_picks` + `players` (gsis/pfr id crosswalk) | free CSV download | 1980-present | **automated** |
| All-Pro selections | PFR `/years/{yr}/allpro.htm` | scrapeable HTML table, ~25 pages one-time | 1970s-present | **one-time script, run locally** |
| Pro Bowl selections | PFR `/years/{yr}/probowl.htm` | scrapeable, same shape | same | **one-time script, run locally** |
| NFL Top 100 (player-voted) | Wikipedia annual list pages | scrapeable, stable tables | **2011-present only** | **one-time script** |
| PFF player grades / top-tier players | PFF Premium Stats | **paywalled (PFF+)** | 2006-present | **manual export/upload** |
| PFF unit averages (OL, DL, DB, etc.) | PFF Premium Stats team pages | **paywalled (PFF+)** | 2006-present | **manual export/upload** |

Notes on the awkward ones:

- **PFR scraping**: tables are plain HTML and PFR even offers "Share & Export
  → Get table as CSV". But PFR now aggressively rate-limits and
  Cloudflare-blocks bot traffic — the fetch must run *locally, slowly*
  (~1 request / 3-4s, ~50 pages total for All-Pro + Pro Bowl since 2002),
  and the output gets committed as static CSVs. Do **not** put PFR scraping
  in CI. If scraping is blocked entirely, the fallback is manual CSV export
  from those same pages (10 minutes of clicking).
- **Joining PFR to nflverse**: the nflverse `players` table carries a
  `pfr_id` for most modern players, so All-Pro/Pro Bowl rows join on PFR id,
  not name-matching. Top 100 (Wikipedia) has no ids → name+position+season
  fuzzy join; expect a handful of manual fixes per year.
- **PFF**: there is no public API and grades are behind PFF+. Two honest
  options: (a) you export CSVs from PFF+ (player grades per team-season, and
  their team position-group grades) into a drop folder we define; (b) skip
  PFF and use open substitutes (nflverse EPA-based unit proxies: OL = sack +
  pressure rate allowed, DB = EPA/dropback allowed, etc.). Plan assumes (a)
  with (b) as fallback so the pipeline never blocks on the subscription.
- **Prediction-market of it**: none of this is needed for the books
  comparison — it only feeds the model.

## 2. Storage layout

```
data/rosters/
├── nflverse/            # auto-downloaded, refreshed weekly in CI
│   ├── roster_weekly_{season}.csv
│   ├── draft_picks.csv
│   └── players.csv
├── awards/              # committed static CSVs from one-time local scrapes
│   ├── allpro.csv       # season, pfr_id, player, pos, team
│   ├── probowl.csv
│   └── top100.csv       # season, rank, player, pos, team  (2011+)
└── pff/                 # manual drops from PFF+ (optional)
    ├── player_grades_{season}.csv
    └── unit_grades_{season}.csv
```

## 3. Feature engineering (per team, per game)

Computed from the **active roster for that week** (nflverse `status == ACT`),
then merged home/away plus a diff column, same pattern as the existing form
features:

| Feature | Definition | Leakage rule |
|---|---|---|
| `n_first_rounders` | count of active players with draft round = 1 | none needed (draft is fixed at entry) |
| `n_top2_rounders` | draft round ≤ 2 | same |
| `allpro_score` | Σ over active players of max(1.0 if selected in last 3 seasons, 0.75 if last 5, 0.5 if ever) | **selections from seasons strictly before the game's season only** — 2025 All-Pro is announced Jan 2026 |
| `n_probowlers` | count with ≥1 Pro Bowl in last 5 seasons (strictly prior) | same lag rule |
| `n_top100` | count on the most recent Top 100 list | list is published pre-season, so the *current* season's list is legal from week 1; 2011+ only, NaN before |
| `pff_top_tier` | count of active players with prior-season PFF grade ≥ 80 (their "high quality" band) | prior season's grade only |
| `pff_ol_avg`, `pff_dl_avg`, `pff_db_avg`, `pff_lb_avg`, `pff_recv_avg`, `pff_qb` | mean prior-season PFF grade by position group | prior season only; in-season weekly grades are a later upgrade |

Interpretation caveat to carry into the writeup: all of these are slow-moving
season-scale facts that the closing line prices instantly. The realistic goal
is (a) better *openers*-vs-us testing later, and (b) giving the tree models
non-market structure to interact with `qb_change` / injuries — not beating
the close by counting Pro Bowlers.

Weighting ambiguity resolved: a player's All-Pro weight is the **max** of the
three bands (not additive) — a 2024 selection scores 1.0, not 1.0+0.75+0.5.
Multiple selections don't stack within a band; a player with selections in
2024 and 2019 scores 1.0.

## 4. Build order

1. `data_jobs/rosters/fetch_nflverse.py` — download roster/draft/players
   CSVs (same release-asset pattern as `schedule.py --refresh`); add to the
   weekly-report workflow.
2. `data_jobs/rosters/scrape_awards.py` — local-only PFR scraper with 4s
   delay + Wikipedia Top 100; writes `data/rosters/awards/*.csv` for commit.
3. `NFL/model/v2/squad.py` — build the per-team-week quality table, apply
   the leakage lags, expose `add_squad_features(games_df)`.
4. Wire into `dataset.py` behind a flag; rerun the bake-off + ablation with
   the new columns; only then decide if they stay.
5. PFF drop-folder ingestion, if/when you export from PFF+.

Effort guess: steps 1-4 ≈ one working session; the scrape (step 2) is the
only part that can fight back.

## 5. Fold-ins from the pipeline questions

**QB handling — already correct, one upgrade queued.** The model uses
nflverse's per-game `home_qb_name`/`away_qb_name`, which is the **actual
starter of that specific game**, not the season snap-count leader (verified:
2023 CLE shows 5 different starters game-by-game; 2025 CIN shows
Burrow 8 / Flacco 6 / Browning 3). `qb_change` flags a different starter than
the team's previous game. What's missing is *magnitude* — Burrow→Browning and
Burrow→Flacco flag identically. Upgrade (belongs in this batch since it needs
the same roster/grade plumbing): starter's prior-season PFF QB grade (or
open-data fallback: career EPA/dropback from nflverse pbp), plus
`qb_quality_drop` = prior starter grade − current starter grade.

**Interim head coach — measured, near-zero marginal signal, include as a
flag anyway.** Built the flag from `games.csv` (coach ≠ the team's week-1
coach): 243 games since 2002 (3.7%). Interim-coach teams go 38.2% SU (they're
bad teams — that's why the coach was fired) and **53.7% ATS** (n=246), the
folk "fired-coach bounce" — but that's z≈1.2 from a coin flip, not
significant. Adding home/away interim flags to a spread-only logistic makes
held-out log loss marginally *worse* (0.60801 → 0.60826): the spread already
prices it. Verdict: ship the flag in this batch (it's free from data we
already have, and trees may find interactions with rest/QB features), but
expect nothing from it standalone.

---

## 6. Build results

### What changed from the plan

**PFF dropped** — no subscription, so no player grades and no unit averages.
`qb_epa_prior` (passing EPA per attempt, prior seasons only, from nflverse
weekly stats) is the open-data stand-in for the QB-grade signal.

**PFR and Wikipedia are blocked from this environment.** The egress proxy
answers 403 to CONNECT for both hosts, so the honors scrape could not be run
here. `data_jobs/rosters/scrape_awards.py` is written and unit-tested against
fixture HTML, but it **must be run on a local machine**:

```bash
python3 -m data_jobs.rosters.scrape_awards --first 2002 --last 2025
python3 -m data_jobs.rosters.scrape_awards --report   # sanity-check coverage
```

It saves every fetched page under `data/rosters/awards/raw/` before parsing,
so if a parser is wrong for some year's markup you can fix it and re-run with
`--reparse` without touching the network again. Until that runs,
`allpro_score`, `n_probowlers` and `n_top100` are emitted as NaN and every
model treats them as missing — nothing else is affected.

One trap worth recording: `draft_picks.csv` has `allpro` and `probowls`
columns, which look like a free substitute for the scrape. They are **career
totals as of today**, so using them for a 2015 game would count selections
from 2016-2025. They are not used.

### What shipped

| Feature | Status |
|---|---|
| `n_first_rounders`, `n_top2_rounders`, `pct_drafted` | **live** — 96% of games |
| `allpro_score`, `n_probowlers`, `n_top100` | wired, NaN until the local scrape runs |
| `qb_epa_prior`, `qb_quality_drop` | **live** — 76% of games (rookies/low-volume QBs are NaN by design) |
| `home/away_is_interim_coach` | **live** — 100% of games |
| PFF unit grades | dropped |

QB magnitude works as intended. 2025 Cincinnati, game by game: Burrow
(0.177 EPA/att prior) → Browning (0.096, drop +0.080) → Flacco (0.008, drop
+0.088) → Burrow (drop −0.168, i.e. an upgrade). Previously all four
transitions were an identical `qb_change = 1`.

### Do they earn their place?

Walk-forward, 2015-2025, 3,018 out-of-sample games (log loss):

| model | top10 | top10 + squad | full45 | full45 + squad |
|---|---|---|---|---|
| logistic | **0.6140** | 0.6155 | 0.6178 | 0.6196 |
| extra trees | 0.6193 | 0.6178 | 0.6205 | 0.6198 |
| random forest | 0.6264 | 0.6208 | 0.6225 | 0.6204 |
| xgboost | 0.6241 | 0.6234 | 0.6272 | 0.6256 |
| lightgbm | 0.6324 | 0.6311 | 0.6339 | 0.6314 |

**Squad features help every tree model and hurt logistic**, at both feature-set
sizes — the same pattern the ablation study found for weak features generally.
Tested one at a time on top of the top-10 set, no single squad feature moves
logistic by more than 0.0003 (noise), while every one improves Extra Trees by
0.0003-0.0012.

The best top-3 hit rate in the whole study is now Extra Trees + squad at
**79.1%**, against the market's 79.6% and logistic's 78.2% — but that is ~430
picks with a standard error near 2 points, so it is not a real separation.

**Recommendation: keep the features, keep logistic/top-10 as the picks model.**
The features are cheap, correct, and leakage-safe; they are also further
evidence that the closing line has already priced roster quality. The one
scenario that would change the model choice is the honors scrape landing and
moving Extra Trees clearly past logistic — worth re-running the grid then.

### Re-running

```bash
python3 -m data_jobs.rosters.fetch_nflverse         # ~440 MB, gitignored
python3 -m NFL.model.v2.squad --build               # -> data/rosters/squad_features.csv
python3 -m NFL.model.v2.compare_models --target win --half-life 6 --with-squad --save
```

### 7. Is the tree advantage real? (bootstrap)

The squad features clearly help the tree models, so the follow-up question is
whether Extra Trees + squad should *replace* logistic for the weekly picks.
Answer: directionally yes, statistically not established.

Walk-forward 2015-2025, 675 top-3 picks (3 per week x 225 weeks):

| model | log loss | top-3 hit |
|---|---|---|
| market (closing) | **0.6121** | **79.6%** |
| logistic / top10 | 0.6140 | 78.2% |
| ensemble (logistic + ET, averaged) | 0.6150 | 78.8% |
| extra trees / top10 + squad | 0.6178 | 79.1% |
| random forest / top10 + squad | 0.6208 | 79.0% |

Bootstrapping 5,000 resamples **of whole weeks** (picks within a week are not
independent, so resampling individual games would understate the error):

| vs logistic/top10 | mean diff | 95% CI | P(better) |
|---|---|---|---|
| extra trees + squad | +0.88 pp | [-1.02, +2.72] | 0.80 |
| random forest + squad | +0.74 pp | [-1.35, +2.95] | 0.73 |
| ensemble | +0.59 pp | [-0.60, +1.83] | 0.79 |
| market | +1.34 pp | [+0.00, +2.76] | 0.96 |

Read: the trees are *probably* a bit better at picking the top 3 (80% odds),
but the interval straddles zero, and by season they beat logistic in 6 of 11,
tie 2, lose 3. The only comparison that clears significance is the market
beating us, which has been the finding all along.

The ensemble is the interesting row: middle of the pack on both metrics but
with by far the tightest interval (±1.2 pp vs ±1.9 pp), which is what
averaging two decorrelated models is supposed to buy. It is the option that is
least likely to be much worse in any given season.

Left as-is for now: the shipped picks model is still logistic/top10. Switching
to the ensemble is a reasonable call and is a two-line change in
`weekly_nfl_report.py`, but it should be a deliberate decision rather than
chasing a 0.6 pp difference inside its own confidence interval.

### 8. Top 100 landed (2011-2025) — verdict unchanged

The Wikipedia Top 100 scrape came in via PR #16: 1,500 rows, exactly 100 per
season, 2011-2025.

**Two fixes were needed before it was usable.**

1. **`pfr_id` was 0% populated.** Wikipedia carries no player ids, so without
   resolution every Top 100 entry silently fails to join a roster and
   `n_top100` would have been a column of zeros. Exact name matching got
   95.9%; the 62 misses were entirely systematic — spaced initials
   (`B. J. Raji` vs `B.J. Raji`), accents (`Pierre Garçon`), and suffixes
   (`Chris Harris Jr.`). Folding accents, dropping suffixes and stripping all
   separators took it to 99.7%, and three genuine name changes
   (Darius/Shaquille Leonard, Justin/Nnamdi Madubuike, Tariq/Riq Woolen) were
   added to an explicit alias map rather than fuzzy-matched — a wrong id here
   credits a star to the wrong roster. **Final: 1,500/1,500.**

2. **Missing honors were reading as zero, not unknown.** `groupby.sum()` folds
   an all-NaN group to `0.0`, so seasons before 2011 were asserting "this
   roster had no Top 100 players", and — once `top100.csv` existed but the PFR
   scrape still didn't — `allpro_score` and `n_probowlers` became all-zero
   columns for all 13,540 team-weeks. Each honor is now NaN'd independently.
   Both cases have regression tests.

Sanity check after the fix: 78-91 of each year's 100 listed players are on a
week-1 active roster (the rest injured, retired, or on the practice squad),
averaging 2.4-2.7 per team. 2025 leaders are Philadelphia (9) and Detroit (7).

**Does it move the model?** No.

| model | top10 | top10 + squad |
|---|---|---|
| logistic | **0.6140** | 0.6156 |
| extra trees | 0.6193 | 0.6179 |
| random forest | 0.6264 | 0.6203 |
| xgboost | 0.6241 | 0.6227 |
| lightgbm | 0.6324 | 0.6301 |

Same pattern as before Top 100 existed: helps every tree, hurts logistic.
Dropping `n_top100_diff` back out changes Extra Trees by -0.0001 and Random
Forest by +0.0000; dropping *all three* honor columns changes Extra Trees by
-0.0001. The re-run bootstrap actually moved slightly against the trees
(Extra Trees vs logistic now +0.58 pp, 95% CI [-1.31, +2.49], P(better)=0.70,
down from +0.88 pp / P=0.80).

Conclusion: the Top 100 data is correct, cheap and worth keeping wired, but it
carries no signal the closing line hasn't already priced. Picks model stays
logistic/top10. The All-Pro and Pro Bowl scrapes are unlikely to change this,
so treat them as low priority.

### 9. Top 100 by position, by rank, and at quarterback

Follow-up analysis in `NFL/model/v2/top100_analysis.py`; tables and the
rendered report in `artifacts/top100/`. 2011-2025, 4,096 games. Every rate is
reported straight up **and** against the closing spread, because those answer
different questions — "do better rosters win" (yes, and it pays nothing) vs
"does the market underprice them" (the only version worth money).

**Position groups.** Split each roster's Top 100 count by position group and
compare games where the home team had an advantage at that group against games
where it had a deficit:

| group | win-rate gap | spread already moves | cover gap | 95% CI |
|---|---|---|---|---|
| QB | +22.6 pp | 7.2 pts | +1.7 pp | [-3.4, +6.7] |
| OL | +11.6 pp | 4.1 pts | +1.3 pp | [-4.6, +7.2] |
| WR | +10.6 pp | 2.8 pts | **+3.4 pp** | [-1.5, +8.2] |
| TE | +10.2 pp | 3.5 pts | +1.6 pp | [-5.1, +8.5] |
| DB | +8.1 pp | 2.3 pts | +0.5 pp | [-4.7, +5.6] |
| LB | +7.3 pp | 2.1 pts | -1.5 pp | [-6.5, +3.5] |
| RB | +7.2 pp | 1.7 pts | +1.5 pp | [-4.1, +7.2] |
| DL | -4.5 pp | -0.4 pts | -2.4 pp | [-7.4, +2.5] |

No group weighs out: all eight intervals cross zero. The two middle columns are
the real story — the win gap and the spread swing move together, group by
group. Wide receiver is the least-bad candidate for a genuine leftover edge
(+3.4 pp, P(real)=0.91) because its spread swing is small relative to its win
gap. Defensive line runs the other way and is the one group where more Top 100
talent goes with a slightly worse record.

**Rank weighting** (flat count vs linear `(101-rank)/100` vs log
`1/log2(rank+1)`): log weighting describes reality better — correlation with
final margin 0.214 vs 0.201 for a plain count — but it also correlates *more*
with the closing spread (0.508 vs 0.448). The books weight by rank too, and
slightly better than we do, so against the number weighting is marginally
worse (0.010 vs 0.022, both ~zero). Plain counts stay.

**Quarterbacks.**

- Top 100 QB vs unlisted QB (n=1,925): **64.0% SU**, **50.5% ATS**, and his
  team lays 4.3 points on average. The largest single roster effect in the
  data, priced almost exactly.
- Both listed (n=509): the better-ranked QB wins **57.4%** (CI [53.1, 61.7],
  clears 50% cleanly) but covers only **51.1%** (CI [46.9, 55.5]).
- By tier, SU falls monotonically across the top three tiers — 68.4% (1-10),
  60.4% (11-25), 53.8% (26-50) — with the 51-100 tier breaking rank at 55.3%.
  So yes, higher-ranked quarterbacks demonstrably win more.
- ATS by tier flattens: 53.2% / 51.1% / 47.9% / 49.3%. The top-10 tier is the
  most interesting number in this whole batch — it clears the 52.4% break-even
  with P(ATS>50%)=0.95 — but its interval [49.5, 56.8] contains both
  break-even and the coin flip on 729 games.
- Bucketed by rank gap the head-to-head is not monotone and the 0-10 bucket
  inverts (better-ranked QB goes 47.0% on 83 games). The list ordering carries
  real information in broad strokes and essentially none in fine ones.

**Verdict: no model change.** Position-split counts, rank-weighted scores and
QB-tier flags all get absorbed by the closing spread, same as the raw Top 100
count did. Two threads to revisit with more seasons: wide receiver (+3.4 pp)
and top-10 quarterbacks (53.2% ATS). Each season adds ~270 games.
