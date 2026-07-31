# Squad-quality features — implementation plan (no code yet)

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
