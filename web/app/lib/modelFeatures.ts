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
    'Multinomial logistic over {home win, draw, away win} on the features below — the two ' +
    'Elos reach it as one venue-adjusted gap — with scorelines from independent Poisson goal ' +
    'rates, then the rest of the season replayed with live in-sim Elo.',
  features: [
    ...ELO_PAIR('Club', 'plus the pool’s home advantage'),
    { name: 'Elo gap', detail: 'the venue-adjusted difference — what the model actually reads' },
    { name: 'Transfer spend', detail: 'gross spend this season, home − away, z-scored in league-season' },
    { name: 'Net spend', detail: 'spend − sales, same normalisation' },
    { name: 'Squad value', detail: 'market-value differential' },
    {
      name: 'Wage bill',
      detail: 'wage-bill differential',
      dormant: 'no wage uploads yet, so its fitted weight is 0.000',
    },
    {
      name: 'xG form',
      detail: 'rolling xG net over 10 league matches, home − away',
      dormant: 'the Understat feed stops at 2025-01-04, so the staleness guard voids it',
    },
    { name: 'Shot form', detail: 'rolling shots-on-target net — the chance-creation feed that is live' },
  ],
  gaps:
    'Measured on held-out seasons, the Elo gap is worth about four times the next feature ' +
    'and squad value is the only economics feature doing any work — the two transfer-spend ' +
    'features are constant on every recent row, so three of seven contribute nothing today. ' +
    'Rolling goal form and form-as-surprise were both tested and neither cleared the bar — ' +
    'the goals are already in the rating, and together the two fight each other — so the ' +
    'candidates worth chasing carry what the score line does not: possession, chance ' +
    'quality, and the corners, cards and fouls sitting unparsed in files already downloaded.',
};

export const NFL_FEATURES: ForecastModel = {
  title: 'Football model features',
  engine:
    'Win probability from a betting-blind Elo plus one efficiency layer: each side’s ' +
    'opponent-adjusted success rate on offence and defence, from nflverse play-by-play, in a ' +
    'regularised logistic refit each run. The score model is refit from the engine’s own ' +
    'replay, then the rest of the season is replayed with live in-sim Elo through the ' +
    'seven-team bracket.',
  features: [
    ...ELO_PAIR('Team', 'plus +48 home advantage — zero at a neutral site'),
    {
      name: 'Adjusted success rate',
      detail:
        'home and away offence and defence, ridge-adjusted for every opponent faced (half-life 5 weeks, prior season at half weight), plus the cross-unit matchup net; falls back to Elo alone when the play-by-play feed is more than 10 days behind',
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
    'nothing — the model scores better without it. The success-rate layer is worth about ' +
    '0.005 log loss over Elo alone across 2015–2025 (+3 SE); adjusted EPA, pass/rush splits, ' +
    'drive, red-zone, third-down and pace metrics were all collected and tested and none ' +
    'beat it on the clean 2024–25 window, so they stay out. The season simulation and the ' +
    'expected score still run on Elo alone.',
};

export const CFB_FEATURES: ForecastModel = {
  title: 'Football model features',
  engine:
    'Win probability from Elo alone — betting-blind, no closing line — with the score model ' +
    'refit from the engine’s own replay each run, then the rest of the regular season replayed ' +
    'with live in-sim Elo. The 12-team playoff field is deliberately not modelled.',
  features: [
    ...ELO_PAIR('Program', 'plus +50 home advantage — zero at a neutral site'),
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
    'advantage. No efficiency input: cfbFastR ships EPA per play from the same repo as the ' +
    'schedule spine.',
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
