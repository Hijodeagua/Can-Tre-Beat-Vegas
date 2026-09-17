# Advanced metrics layer

Second-stage features on top of the Elo engines for club soccer, the NFL
and college football. The Elo engines are untouched; each sport gets a
raw fetch layer (cached, not committed), a processed table (committed),
a leakage-safe pregame feature builder, a chronological ablation, and a
freshness gate so a dead feed degrades the forecast to Elo instead of
feeding it stale numbers.

Rules that hold everywhere:

- **Pre-game only.** Every rolling stat is shifted so a match never sees
  its own result (`common/pregame.py`, tested in `tests/test_common_pregame.py`).
- **Chronological splits only.** Fit on earlier seasons, score later ones.
  Imputers, scalers and models are fit on training rows only
  (`common/evaluate.py` wraps them in one sklearn `Pipeline`).
- **Log loss and Brier decide.** Accuracy is printed because people ask.
  A feature group has to beat the baseline on the *same rows*; the number
  reported is the paired gain in standard errors (`paired_se`).
- **Freshness gate.** `common/freshness.py` reports each feed's newest
  row on the run date; a feed past its tolerance is logged STALE and the
  feature it drives is off for that run.
- **No paid APIs.** Everything below is a public file or an unauthenticated
  endpoint.

Regenerate any table here with the commands in each section. Shared code:
`common/pregame.py` (rolling / EWMA / shrink / rest), `common/freshness.py`,
`common/evaluate.py` (walk-forward, fixed split, paired test).

---

## Club soccer

### Sources

| Feed | Endpoint | Refresh | Processed table | Raw cache |
|---|---|---|---|---|
| Understat league data | `https://understat.com/getLeagueData/{league}/{season}` (headers: `User-Agent`, `X-Requested-With: XMLHttpRequest`, `Referer`), legacy `datesData`/`teamsData` HTML blobs as fallback | daily Actions job, current season only; backfill 2014-15 → on demand | `soccer/clubs/data/understat_matches.csv` (+ `xg_matches.csv` in the legacy 6-column shape) | `data/soccer_clubs/raw/understat/` |
| football-data.co.uk shots | unchanged (`fetch_shots.py`) | daily | `shots_matches.csv` | — |

The Understat host is blocked from the dev sandbox (proxy 403), so the
fetcher has only been exercised against fixtures. The daily workflow now
runs `python -m soccer.clubs.data.understat --probe epl <year>` before the
pipeline so the first Actions run prints the live response shape; the
parser accepts a JSON object, a JSON list, and the legacy HTML blob.

Per match the processed table keeps: xG, npxG, xPts, PPDA numerator and
denominator (attacking and defensive), deep completions, deep completions
allowed, goals. Shots and shots on target come from the football-data feed.

Commands:

```
python -m soccer.clubs.data.understat                  # last + current season refresh (Actions)
python -m soccer.clubs.data.understat --all            # backfill 2014-15 → current
python -m soccer.clubs.data.understat --probe epl 2025 # print the live response shape
python -m soccer.clubs.model.eval_advanced             # ablation, both scopes
```

### Pregame features (`soccer/clubs/model/advanced.py`)

Every value is a difference, home minus away, from each club's previous
league matches only. Rolling = mean of the last 10; EWMA = half-life 5
matches; both need 5 prior matches, and a club whose last match is more
than 130 days old reads NaN (imputed to the training median).

| Group | Features |
|---|---|
| xG form (EWMA) | `xg_for_ewm_diff`, `xg_against_ewm_diff`, `npxg_for_ewm_diff`, `npxg_against_ewm_diff`, `xg_net_ewm_diff` |
| xG form (rolling) | the rolling-10 versions of the above |
| Matchup | `home_att_vs_away_def` (home's home-only xG for vs away's away-only xG against), `away_att_vs_home_def`, `xg_per_shot_diff` |
| Territory | `deep_diff`, `deep_share_diff` (labelled a field-tilt *proxy*: deep completions / (deep for + deep against), not possession), `ppda_diff` (PPDA = opponent passes / defensive actions; lower = more pressing), `xpts_form_diff` |
| Rest | `rest_diff` (days since previous match), `congestion_diff` (matches in the previous 14 days), `uefa_diff` (European tie within 7 days), from the whole calendar including UEFA rows |

### Evaluation

Train < 2023-24, validation 2023-24 (picks C and the compact set), test
2024-25 → 2026-27, multinomial logistic on the shipping feature set plus
one group. Artifacts: `soccer/clubs/model/artifacts/advanced_eval_{all,top5}.json`.

Coverage on the test seasons (top-5 scope):

| season | rows | live xG form | npxG / territory | rest |
|---|---|---|---|---|
| 2023-24 | 1752 | 98.6% | 0% | 100% |
| 2024-25 | 1752 | 89.1% | 0% | 100% |
| 2025-26 | 1751 | 0% | 0% | 100% |
| 2026-27 | 146 | 0% | 0% | 98.6% |

npxG, xPts, PPDA and deep completions are 0% because the processed
Understat table cannot be fetched from the sandbox; the legacy xG file
only carries xG, and it ended on 2025-01-04.

Test, all leagues (n = 7,827) and top-5 only (n = 3,649):

| model | log loss (all) | SE vs base | log loss (top5) | SE vs base |
|---|---|---|---|---|
| base (Elo + economics + xG net + SoT net) | 1.01755 | — | 0.98353 | — |
| + xG EWMA | 1.01750 | +0.34 | 0.98360 | −0.18 |
| + xG rolling | 1.01762 | −0.47 | 0.98374 | −0.67 |
| + matchup | 1.01787 | −1.80 | 0.98439 | −2.07 |
| + territory | 1.01755 | +1.19 (all-NaN columns) | 0.98353 | −0.71 |
| + rest | 1.01739 | +0.59 | 0.98331 | +0.38 |
| combined | 1.01778 | −0.96 | = base | — |

Nothing clears +2 SE. That is the expected result of scoring a
chance-creation layer on seasons where the feed was dead for 57% of the
rows: the model learns coefficients on training seasons with live xG and
then meets imputed medians.

**Were the earlier xG results distorted by stale data?** Yes, in the
direction of hiding the signal, not inventing it. The staleness guard
zeroed `xg_net_diff` on 2,088 of 3,649 top-5 test matches. On all test
rows the feature is worth +0.35 SE; on the 1,561 matches where both clubs
had live form it is worth +1.78 SE (0.97941 vs 0.98013). The signal the
2023-24 validation found (+2.1 SE) is still there wherever the feed is.

### Production decision

- Promoted: nothing new. The shipping feature set is unchanged.
- The Understat fetcher is replaced (`fetch_xg.py` is a shim over
  `understat.py`), so once Actions lands one refresh `xg_net_diff` comes
  back on live slates through the existing 130-day guard.
- The daily run logs freshness for all three feeds and writes them to
  `web/public/data/soccer/latest.json` under `feeds`.
- Collected, not promoted: npxG, xPts, PPDA, deep completions, xG per
  shot, home/away attack splits, rest and congestion. Re-run
  `eval_advanced` after a season of live Understat rows before promoting
  any of them; the current numbers say nothing about them either way.

Leakage safeguards specific to this layer: form uses only rows dated
strictly before the match (`test_form_is_strictly_pre_match_and_warms_up`),
home splits only see home matches, rest and congestion come from the full
calendar (UEFA included) but only from dates before the match, and the
staleness guard is applied per row, per side.
