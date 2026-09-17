/**
 * What each live model actually looks at, in the shape the forecast tabs
 * show it: home Elo, away Elo, then the sport-specific features.
 *
 * The long-form version — tuned parameters, fitted coefficients, which
 * features are dormant and why — is `docs/MODEL_FEATURES.md`, and this is
 * the same content trimmed to what fits beside a forecast table. Keep the
 * two in step; the markdown is the reference, this is the reminder, and
 * `SHEET_URL` sends a reader from one to the other.
 *
 * `dormant` marks a feature that is wired up but contributing nothing
 * right now (a dead upstream feed, an upload that hasn't landed). It is
 * shown struck through rather than dropped: a reader comparing this list
 * against the markdown should see the same features in the same order,
 * and a feature quietly disappearing from a list is how a model comes to
 * look better than it is.
 */

import { REPO_URL } from '@/app/lib/modelDocs';

export const SHEET_URL = `${REPO_URL}/blob/main/docs/MODEL_FEATURES.md`;

export interface ModelFeature {
  name: string;
  /** What it is, in a few words. */
  detail: string;
  /** Wired up but feeding 0 today — shown, struck through, with the reason. */
  dormant?: string;
}

export interface ForecastModel {
  /** Heading, e.g. "Soccer model features". */
  title: string;
  /** How the forecast is produced, in one sentence. */
  engine: string;
  features: ModelFeature[];
  /** What the measured importance run says this model is missing or
   * wasting — the critique, not a disclaimer. Kept next to the feature
   * list because a feature list without one reads as a boast. */
  gaps: string;
}

const ELO_PAIR = (side: string, home: string): ModelFeature[] => [
  { name: 'Home Elo', detail: `${side} Elo, home side, ${home}` },
  { name: 'Away Elo', detail: `${side} Elo, away side` },
];

export const SOCCER_FEATURES: ForecastModel = {
  title: 'Soccer model features',
  engine:
    'Random forest over {home win, draw, away win} on every feature below — the two Elos as ' +
    'their own inputs beside the venue-adjusted gap, so a rating level can matter on its own — ' +
    'refit from the replay each run, with scorelines from independent Poisson goal rates, then ' +
    'the rest of the season replayed with live in-sim Elo.',
  features: [
    ...ELO_PAIR('Club', 'each fed to the model on its own, not only as a gap'),
    { name: 'Elo gap', detail: 'the venue-adjusted difference, alongside the two ratings' },
    { name: 'Transfer spend', detail: 'gross spend this season, home − away, z-scored in league-season' },
    { name: 'Net spend', detail: 'spend − sales, same normalisation' },
    { name: 'Squad value', detail: 'market-value differential' },
    {
      name: 'Wage bill',
      detail: 'wage-bill differential',
      dormant: 'no wage uploads yet, so the column is constant',
    },
    { name: 'xG form', detail: 'rolling xG net over 10 league matches, home − away — Understat, live again since 2026-09-17' },
    { name: 'Shot form', detail: 'rolling shots-on-target net (football-data.co.uk)' },
    {
      name: 'xG / npxG form, both flavours',
      detail: 'exponentially weighted (5-match half-life) and rolling-10 xG and non-penalty xG for and against, home − away',
    },
    {
      name: 'Attack vs defence',
      detail: 'the home side’s home-only xG attack against the away side’s away-only xG defence, and the reverse; npxG versions too; xG per shot',
    },
    {
      name: 'Territory',
      detail: 'deep completions, deep share (a field-tilt proxy — Understat publishes no possession), PPDA (pressing), xPts form',
    },
    { name: 'Rest and congestion', detail: 'rest days, matches in the last 14 days, a European tie in the last 7 — from the whole calendar' },
  ],
  gaps:
    'Measured on 2024-25 onward, the full 31-feature set beats the old seven-feature model by about ' +
    '0.0008 log loss (+1.4 SE) with a logistic; a random forest scores 0.0017 worse than that ' +
    'logistic (noise-level) and ships anyway because it can use npxG, PPDA and deep completions ' +
    'non-linearly once the Understat backfill lands — those columns were empty when this was ' +
    'measured. Possession itself is in no free feed the site has; deep share stands in for it. ' +
    'A rating level matters on its own: a big favourite above 1625 Elo wins 77% of the time, ' +
    'the same gap below 1325 wins 56%.',
};

export const NFL_FEATURES: ForecastModel = {
  title: 'Football model features',
  engine:
    'Win probability from a random forest over the betting-blind Elo — the two ratings as ' +
    'their own inputs beside the Elo probability — and every efficiency feature built from ' +
    'nflverse play-by-play, refit each run. The score model is refit from the engine’s own ' +
    'replay, then the rest of the season is replayed with live in-sim Elo through the ' +
    'seven-team bracket.',
  features: [
    ...ELO_PAIR('Team', 'each fed to the forest on its own, plus the Elo win probability'),
    {
      name: 'Opponent-adjusted ratings',
      detail:
        'offence and defence, both sides, ridge-adjusted for every opponent faced (half-life 5 weeks, prior season at half weight): EPA/play, success rate, dropback and rush EPA and success, early-down EPA and success, explosive rate, points and EPA per drive, red-zone TD rate, third-down conversion — plus every cross-unit matchup net',
    },
    {
      name: 'Form (EWMA, 5-game half-life)',
      detail: 'PROE (all and neutral), pass rate, neutral pace, drives, plays/yards per drive, series success, red-zone trips/points/EPA, third-down EPA/distance/short/long, sack rate for and against, net special-teams EPA, raw EPA for and against',
    },
    {
      name: 'Fallback',
      detail: 'Elo alone when the play-by-play feed is more than 10 days behind or a side has no rating',
    },
    { name: 'Rest', detail: '+20 Elo off a bye (10+ days) at prediction time' },
    { name: 'Margin of victory', detail: 'ln-damped, capped at 45 points, shrunk when the favourite wins' },
    { name: 'Postseason weight', detail: 'playoff results update at K × playoff multiplier' },
    { name: 'Season carryover', detail: '40% regression to 1500, franchise moves carry their rating' },
    { name: 'Ties', detail: 'scored as a half win, plain K' },
  ],
  gaps:
    'Ablation on held-out seasons says margin of victory and season regression carry this ' +
    'model, home advantage is worth very little, and the bye-week bonus is worth less than ' +
    'nothing — the model scores better without it. On the clean 2024–25 window the full ' +
    '41-input forest scores 0.61687 to Elo’s 0.62411 (+1.2 SE), tied with a logistic on the ' +
    'same inputs; boosting overfits and is worse than Elo alone. A rating level matters on ' +
    'its own: a favourite by 75+ Elo wins 79% of the time above 1575 and 59% below 1425. ' +
    'The season simulation and the expected score still run on Elo alone.',
};

export const CFB_FEATURES: ForecastModel = {
  title: 'Football model features',
  engine:
    'Win probability from a random forest over the betting-blind Elo — the two ratings as ' +
    'their own inputs beside the Elo probability — and every efficiency feature from the ' +
    'SportsDataverse weekly summaries, refit each run. The score model is refit from the ' +
    'engine’s own replay, then the rest of the regular season is replayed with live in-sim ' +
    'Elo. The 12-team playoff field is deliberately not modelled.',
  features: [
    ...ELO_PAIR('Program', 'each fed to the forest on its own, plus the Elo win probability'),
    {
      name: 'Opponent-adjusted EPA',
      detail:
        'home and away offence and defence, from the snapshot through the previous week (never the current one), blended with the prior season’s regressed final until a program has games; plus the matchup net',
    },
    {
      name: 'Efficiency, both sides',
      detail: 'success rate, early-down EPA, explosive rate, havoc, EPA per drive, drives per game, plays and yards per drive, red-zone and third-down success, third-down distance, pass rate, pass and rush EPA — each with its cross-unit matchup net',
    },
    {
      name: 'Fallback',
      detail: 'Elo alone for games with an FCS side, a side without a strength row, or any week the weekly feed is two weeks behind',
    },
    {
      name: 'Conference regression',
      detail: '30% toward a 0.75/0.25 blend of the new conference’s mean and 1500, so realignment is handled by construction',
    },
    { name: 'Pooled FCS opponent', detail: 'every non-FBS side is one synthetic team at 950; only the FBS side updates' },
    { name: 'FBS entry rating', detail: 'a first FBS game starts a program at 1250, not at average' },
    { name: 'Margin of victory', detail: 'ln-damped, capped at 80 points' },
  ],
  gaps:
    'Ablation on held-out seasons makes the pooled FCS rating the single most load-bearing ' +
    'component — one synthetic 950-rated team standing in for every non-FBS opponent, about ' +
    '13% of the schedule, is the crudest thing in the model and it matters more than home ' +
    'advantage. On 2024–25 the full 33-input forest scores 0.49347 to Elo’s 0.49768 overall ' +
    '(+1.1 SE) and 0.54657 to 0.55358 on FBS-vs-FBS games (+1.6 SE), the best of the three ' +
    'learners tried. A favourite by 250+ Elo wins 95% of the time above 1650 and 89% below ' +
    '1350. The season simulation still runs on Elo alone.',
};

export const MLB_FEATURES: ForecastModel = {
  title: 'Baseball model features',
  engine:
    'Elo with the starting pitcher, rest and travel priced in Elo points, run environment from ' +
    'exponentially weighted attack/defense rates, then the rest of the season replayed.',
  features: [
    ...ELO_PAIR('Team', 'plus +24 home advantage'),
    {
      name: 'Starting pitcher',
      detail: '3.0 Elo per point of rolling game score above the team’s own staff, with a TBD fallback',
    },
    { name: 'Rest', detail: '+2.3 Elo per rest day, capped at 3 days' },
    { name: 'Travel', detail: '−0.31 × miles^⅓, capped at −4 Elo' },
    { name: 'Run rates', detail: 'EWMA runs scored/allowed, half-life 20 games, shrunk to the league mean' },
  ],
  gaps:
    'The whole rating engine buys about 0.013 nats of log loss over always-picking-home, ' +
    'which is what baseball looks like at the game level. The pitcher, rest and travel ' +
    'adjustments are applied by the daily pipeline rather than the replay, so they are not ' +
    'in the measured breakdown yet — that is the next job here, ahead of park factors, ' +
    'bullpen quality or lineup handedness.',
};
