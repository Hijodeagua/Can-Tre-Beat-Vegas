# College football Elo — data pull plan

> **Status: Tier 0-1 built, from a different source.** The spine, the
> per-season conference map and the FBS/FCS tags all come from cfbfastR-data
> (`CFB/data/fetch_schedule.py`) rather than Sports Reference exports, and
> the engine, tuner and daily pipeline are in `CFB/model/` and `CFB/daily/`
> (see `CFB/README.md`). §1, §3.4 and §8 below are implemented as written;
> §2-3's CFR pulls are superseded; §4-7 remain the roadmap for what to add
> next. Kept as the design record.
>
> Original framing: the shopping list for a CFB rating engine that mirrors
> `NFL/model/v2/` — same walk-forward discipline, same "the market is the
> benchmark" honesty — on a sport where the data is shaped very differently.

Source is [Sports Reference / College Football](https://www.sports-reference.com/cfb/)
(CFR). Every pull below is a **manual export**: each CFR table has
*Share & Export → Get table as CSV*, so a season page is one click, not a
scraper. That matters — CFR rate-limits hard and Cloudflare-blocks bot traffic,
same as PFR (see the note in `NFL/model/v2/SQUAD_QUALITY_PLAN.md` §1). Do not
put CFR scraping in CI.

---

## 0. The one-paragraph version

**Pull `/cfb/years/{YEAR}-schedule.html` for 2000-2025. That's 26 clicks and it
builds the whole Elo.** Everything else on this page is refinement. Do Tier 0
first, get ratings on the board, then decide whether Tier 1 is worth the next
hour based on what the ratings look like.

---

## 1. Why CFB Elo is not NFL Elo

The NFL engine (`NFL/model/v2/elo.py`) makes five assumptions that are all
wrong for college. Naming them up front, because they determine which pulls
are worth making.

| Assumption | Holds in NFL | In CFB |
|---|---|---|
| 32 teams, everyone connected via a balanced schedule | yes | **no** — 136 FBS teams in 2025, conference-heavy schedules, weak inter-conference connectivity. The rating graph is sparse. |
| Roster continuity year to year | mostly | **no** — graduation + transfer portal churn far exceeds NFL free agency. 25% season regression is far too gentle. |
| Talent is compressed; a 20-point spread is extreme | yes | **no** — 40+ point spreads are routine. Elo's 400-point logistic saturates and MOV blows up ratings. |
| Every opponent is in the rating pool | yes | **no** — ~120 games a season are against FCS teams that have no other games in your data. |
| Home field is roughly constant | ~2.2 pts everywhere | **no** — bigger on average (~2.5-3.0) and genuinely varies by venue (altitude, crowd, travel). |

Every Tier 1 pull below exists to patch one of these.

---

## 2. Tier 0 — the spine (do this first)

### 2.1 Season schedules and results

| | |
|---|---|
| **URL** | `https://www.sports-reference.com/cfb/years/{YEAR}-schedule.html` |
| **Pages** | 26 (2000-2025) |
| **Rows** | ~850-950 per season, **~23,000 total** |
| **Gives you** | date, week, day, kickoff time, both teams, both scores, neutral-site flag, notes (bowl / conference championship / CFP round) |
| **Unlocks** | the entire Elo. Nothing else is required. |

**Size check:** ~23,000 games against the NFL spine's 7,276 played games. CFB
gives roughly **3x the training data** — which is the one structural advantage
college has over the NFL model.

**Three gotchas, all of them real:**

1. **It's Winner/Loser format, not Home/Away.** The table lists the winning
   team first with a location marker between the two teams — blank when the
   winner was at home, `@` when the winner was on the road, `N` for a neutral
   site. You must reconstruct home/away yourself. Get this wrong and your HFA
   estimate inverts. **Write a unit test on this transform before anything
   else.** Sanity check: home teams should win ~57-60% of non-neutral games.
2. **Team names are full school names**, not codes — `Texas A&M`,
   `Miami (FL)`, `Ole Miss`, `Louisiana-Monroe`. Non-FBS opponents appear as
   plain unlinked text. You need a canonical crosswalk (see §3.3) and you need
   it before you can join anything.
3. **Names change across seasons.** Schools rename (`Louisiana-Lafayette` →
   `Louisiana`), and CFR is not always retroactive. Build the crosswalk from
   the school index, not from the schedule pages.

**Season range.** Pull 2000+ and treat **2000-2004 as burn-in only** — Elo needs
several seasons for ratings to leave 1500 and mean anything. Evaluate on 2005+.
Don't go earlier than 2000: pre-BCS-era realignment and a materially different
sport make the extra rows more noise than signal.

**Storage:** `data/cfb/schedules/cfb_schedule_{year}.csv`, then a single
normalised `cfb_games.csv` with the NFL spine's column names
(`season, week, gameday, game_type, home_team, away_team, home_score,
away_score, location, neutral`) so the existing `elo.py` engine runs on it with
minimal changes.

---

## 3. Tier 1 — makes the ratings sane (~1 hour)

These are the pulls that fix the five broken assumptions in §1.

### 3.1 Season standings / ratings — the preseason prior

| | |
|---|---|
| **URL** | `https://www.sports-reference.com/cfb/years/{YEAR}-standings.html` and `.../{YEAR}-ratings.html` |
| **Pages** | 26-52 |
| **Rows** | ~130 teams × 26 seasons ≈ 3,400 |
| **Gives you** | conference membership, W-L (overall and conference), **SRS**, **SOS**, and on the ratings page **OSRS / DSRS**, plus AP pre/high/post rank |
| **Unlocks** | conference-aware season regression, a preseason prior, per-season conference mapping, and a free benchmark |

This is the **highest-value Tier 1 pull** and there are four separate reasons:

- **Conference-aware regression.** Regressing every team toward a flat 1500
  each September is wrong in a sport where a Sun Belt team and Alabama are 30
  points apart in true strength. Regress toward the team's **conference mean**
  instead. That needs a per-season conference map — which is exactly what the
  standings page is.
- **Realignment.** Conference membership changes materially every year
  (2024-25 alone moved Texas, Oklahoma, USC, UCLA, Oregon, Washington, and the
  entire Pac-12 remnant). A static conference map is a bug. Pull it per season.
- **Preseason prior.** Prior-season SRS, shrunk, is a defensible day-one rating
  for a team with no games yet.
- **Free benchmark.** SRS is a well-understood public rating. Your Elo should
  correlate highly with it and ideally predict better out of sample. This is
  the CFB analogue of "beat the closing line" — a fixed external yardstick that
  keeps the project honest. **Pull it specifically so you have something to
  lose to.**

### 3.2 Weekly polls

| | |
|---|---|
| **URL** | `https://www.sports-reference.com/cfb/years/{YEAR}-polls.html` |
| **Pages** | 26 |
| **Gives you** | AP and Coaches poll, every week |
| **Unlocks** | week-1 priors, and a second cheap benchmark |

The **preseason** AP poll is the useful column: it is a crowd forecast made
before any games, available for the top ~25 only, and it is a genuinely decent
prior. Worth having as a comparison for whatever prior you build — if your
model's week-1 ratings disagree wildly with the preseason poll, that is
information, not necessarily an error, but you want to see it.

Poll rank also gives you a "ranked opponent" flag, which is a real scheduling
feature in CFB in a way it isn't in the NFL.

### 3.3 School index — the crosswalk

| | |
|---|---|
| **URL** | `https://www.sports-reference.com/cfb/schools/` |
| **Pages** | 1 |
| **Rows** | ~350 |
| **Gives you** | every school CFR knows, its URL slug, and the year range it was active |
| **Unlocks** | the team ID crosswalk you need before any join works |

One page, and it saves hours later. Build
`data/cfb/crosswalk.csv` = `cfr_name, slug, canonical_id, aliases, fbs_from,
fbs_to`. Every other file joins through this.

### 3.4 FCS handling — a decision, not a pull

~120 games a season are FBS-vs-FCS, and the FCS opponent has no other games in
your dataset. Three options:

| Approach | Verdict |
|---|---|
| Drop those games | Loses ~13% of the schedule and, worse, systematically removes the games good teams win. Biases ratings. |
| **Pool all FCS into one synthetic team with a fitted rating** | **Recommended.** One extra free parameter (~1150-1250 Elo). Standard practice in public CFB Elos. |
| Rate FCS teams individually | Requires pulling FCS schedules too (a separate CFR section). Not worth it until Tier 0-1 plateaus. |

Pooling has one known flaw worth writing down: it treats North Dakota State the
same as an FCS bottom-feeder, so beating NDSU under-credits and losing to them
over-punishes. Accept it, note it, revisit later.

---

## 4. Tier 2 — features beyond pure Elo

Only worth pulling once Tier 0-1 produce ratings you trust.

### 4.1 Team season stats

| | |
|---|---|
| **URL** | `.../{YEAR}-team-offense.html`, `.../{YEAR}-team-defense.html` |
| **Pages** | 52 |
| **Gives you** | per-team season totals — plays, yards, yards/play, pass/rush splits, first downs, turnovers, penalties, points |
| **Unlocks** | season-level efficiency priors; **not** rolling in-season form |

**Important limitation:** these are *season aggregates*, so they leak. You
cannot use 2025 season-total offense as a feature for a 2025 game. They are
usable only as **prior-season** features (a 2024 aggregate predicting 2025
games), which is a real but weak signal. Rolling in-season form requires
game-level stats, which means Tier 3.

### 4.2 QB stats — the highest-leverage Tier 2 pull

| | |
|---|---|
| **URL** | `.../{YEAR}-passing.html` (also `-rushing.html`, `-receiving.html`) |
| **Pages** | 26-78 |
| **Unlocks** | a CFB version of `elo_variants.qb` |

The NFL model found the QB adjustment worth 0.005 log loss standalone and
nothing at all once market features were present (`elo_variants/summary.csv`:
qb_talent 0.6319 vs base 0.6374). **Expect it to matter considerably more in
CFB**, for two reasons: QB turnover is far higher, and there is no efficient
market pricing it into your features on most games. This is the single most
plausible place a CFB model finds real signal that the NFL model couldn't.

Caveat to verify on pull: CFR's player pages may be leaders-only rather than
complete rosters in some seasons. Check row counts before designing around it.

### 4.3 Coaches

| | |
|---|---|
| **URL** | `https://www.sports-reference.com/cfb/coaches/` and per-school year-by-year |
| **Unlocks** | a coaching-change flag |

A first-year head coach is a genuine regression signal in college in a way it
mostly isn't in the NFL. Cheap to encode as a binary. Worth it.

---

## 5. Tier 3 — expensive, probably skip

### 5.1 Box scores

`.../boxscores/index.cgi?month=M&day=D&year=Y` → individual game pages with
per-game team stats.

**~23,000 pages.** Do not do this. If you want game-level advanced stats, use
CFBD (§6) instead — it has EPA and success rate, which CFR does not have at
all, and it hands them over in JSON.

---

## 6. What CFR does not have — and where to get it

Being explicit about this, because three of the four gaps matter more in
college than the equivalents did in the NFL.

| Missing | Why it matters in CFB | Where to get it |
|---|---|---|
| **Betting lines** | The NFL project's central finding is that the closing line beats every model tested. Building a CFB model with no market benchmark repeats a mistake already paid for. **This is the critical gap.** | **The Odds API — already wired up in this repo.** See §7. Historical lines: CFBD. |
| **EPA / success rate / play-by-play** | The obvious efficiency layer; CFR has none of it | [CollegeFootballData.com](https://collegefootballdata.com) (CFBD) — free API key, JSON, well-documented |
| **Recruiting rankings** | Talent composite is a strong CFB preseason prior with no NFL analogue | 247Sports composite, or CFBD's `/recruiting/teams` endpoint |
| **Returning production** | Best single-number preseason prior in public CFB modelling | CFBD `/player/returning`, or SP+ |

**The honest recommendation:** CFR is the right source for the **spine and
historical depth** — it goes back further and its season pages are one-click
exports. CFBD is the right source for **everything modern and advanced**, and
it is an actual API rather than a click-and-export. Use both. Do not try to
force CFR to be the whole stack.

---

## 7. The time-sensitive item

The 2026 college season kicks off in roughly two weeks. The repo already runs
`.github/workflows/unified-odds.yml` twice daily against The Odds API, and
`data_jobs/odds_api/config.py` holds a `SUPPORTED_SPORTS` dict that takes a new
entry in about four lines:

```python
"ncaaf": {
    "api_key": "americanfootball_ncaaf",
    "name": "NCAAF",
    "data_dir": "data/ncaaf",
    "file_prefix": "ncaaf_odds_api_data",
},
```

**Quota check.** The free tier is 500 credits/month. A sport-call costs 3
credits (h2h + spreads + totals). Current burn is 3/cycle × 2 cycles/day ≈
**180/month for NFL alone** (NBA calls return zero games in the offseason and
cost nothing). Adding NCAAF in season would be another ~180, for **~360 of
500** — feasible, but with less headroom than it looks, since NBA resumes in
October and would add ~180 more. Something has to give by November: gate
offseason sports off, drop to one snapshot a day, or drop a market.

**Why this is urgent and the CFR pulls are not:** CFR's historical pages will
be there in December. Live 2026 lines will not — every day without the NCAAF
feed is a day of market data you can never backfill for free. Do this before
the CFR clicking.

---

## 8. Parameters to fit, not inherit

The NFL constants in `elo.py` are tuned for the NFL. Every one of them needs
re-fitting. Starting guesses below are priors to grid-search around, not values
to adopt.

| Parameter | NFL value | CFB starting guess | How to fit |
|---|---|---|---|
| `K_FACTOR` | 20 | 20-40 | grid on walk-forward log loss |
| `HFA_ELO` | 55 (~2.2 pts) | 65-80 (~2.6-3.2 pts) | regress margin on home indicator; consider per-venue |
| `SEASON_REGRESSION` | 0.25 | **0.35-0.50** | grid; also test regression toward conference mean vs. flat 1500 |
| `ELO_PER_POINT` | 25 | fit, likely 27-32 | regress actual margin on `elo_diff` |
| MOV cap | none (log damping only) | **cap \|margin\| at 28-35 before damping** | grid |
| FCS pooled rating | n/a | ~1150-1250 | free parameter, fit on log loss |

The season-regression parameter is the one that matters most, and it is the
one most likely to be badly wrong if copied from the NFL.

---

## 9. Build order

1. **Add NCAAF to the odds job.** Time-sensitive; §7.
2. Pull 26 schedule pages. Normalise to the NFL spine schema.
3. Write and test the Winner/Loser → home/away transform. Verify home win rate
   lands at 57-60% on non-neutral games.
4. Run the existing `elo.py` unchanged. Look at the board. It will be wrong in
   instructive ways.
5. Pull standings/ratings. Add per-season conference maps, conference-aware
   regression, and pooled FCS.
6. Grid-search §8. Compare against CFR's SRS and against the preseason AP poll.
7. Only then: Tier 2, and only if step 6 plateaus.

---

## 10. Storage layout

```
data/cfb/
├── schedules/
│   ├── cfb_schedule_2000.csv … cfb_schedule_2025.csv   # raw CFR exports
│   └── cfb_games.csv                                    # normalised spine
├── standings/
│   ├── cfb_standings_{year}.csv
│   └── cfb_ratings_{year}.csv                           # SRS / OSRS / DSRS
├── polls/cfb_polls_{year}.csv
├── stats/  cfb_team_offense_{year}.csv, cfb_team_defense_{year}.csv
├── players/cfb_passing_{year}.csv
├── coaches/cfb_coaches.csv
└── crosswalk.csv                                        # the join key for all of it
```

Raw exports get committed as-is and never edited in place; normalisation is a
script, so a bad transform is always recoverable without re-clicking 26 pages.
