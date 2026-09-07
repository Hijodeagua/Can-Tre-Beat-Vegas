/**
 * How every model on the board is built — one entry per model, each step
 * naming the file that does it, so a reader can follow the pipeline from
 * the data source to the graded pick without reading the whole repo.
 *
 * Paths are repo-relative and resolve to the file on GitHub at build time
 * (`fileUrl`). Keep this in step with the READMEs: the README says why,
 * this page says where.
 */

export const REPO_URL = 'https://github.com/Hijodeagua/Can-Tre-Beat-Vegas';

export function fileUrl(path: string): string {
  return `${REPO_URL}/blob/main/${path}`;
}

export interface DocStep {
  /** What this step does, in one or two sentences. */
  title: string;
  detail: string;
  /** Repo-relative paths, in the order a reader should open them. */
  files: string[];
}

export interface ModelDoc {
  key: string;
  name: string;
  emoji: string;
  /** The model in one sentence. */
  summary: string;
  /** The headline numbers a reader should know, as label/value pairs. */
  facts: { label: string; value: string }[];
  steps: DocStep[];
  /** Where the writeup lives. */
  readme: string;
}

export const MODEL_DOCS: ModelDoc[] = [
  {
    key: 'nfl',
    name: 'NFL Elo',
    emoji: '🏈',
    summary:
      'A betting-blind Elo over every NFL game since 1999, replayed from scratch each morning, ' +
      'with a daily pipeline that grades picks, predicts the next week and simulates the season ' +
      'through the playoff bracket.',
    facts: [
      { label: 'Spine', value: 'nflverse games file, 1999–present' },
      { label: 'Tuned', value: '2005–2023 · holdout 2024–25' },
      { label: 'Holdout log loss', value: '0.624 (always-home 0.691)' },
      { label: 'Parameters', value: 'K 20 · home +48 · bye +20 · cap 45 · regression 40%' },
    ],
    steps: [
      {
        title: 'Data',
        detail:
          'One row per game with scores, kickoff date, neutral flag, game type and rest days. ' +
          'The closing line is in the file but never reaches the rating.',
        files: ['NFL/model/schedule.py', 'data/schedules/nflverse_games.csv'],
      },
      {
        title: 'Rating engine',
        detail:
          'Logistic expectation on the rating gap; K scaled by a capped, log-damped margin that ' +
          'shrinks when the favourite wins; a rest edge for a side off its bye; playoff K; ' +
          'fractional regression to 1500 each off-season; STL/SD/OAK carried onto LA/LAC/LV.',
        files: ['NFL/elo/engine.py', 'NFL/elo/teams.py'],
      },
      {
        title: 'Tuning',
        detail:
          'Coordinate descent on one-step-ahead log loss over every game, 2005–2023, with ' +
          '2024–25 never touched. The tuned values are committed and read by the engine.',
        files: ['NFL/elo/tune.py', 'NFL/elo/artifacts/tuned_params.json'],
      },
      {
        title: 'Score model',
        detail:
          'Expected margin is linear in the Elo gap (about 24 Elo per point, refit each run); ' +
          'the expected total comes from each team’s recent points for and against, shrunk to ' +
          'the league mean.',
        files: ['NFL/daily/scoring.py', 'NFL/daily/state.py'],
      },
      {
        title: 'Slate and grading',
        detail:
          'The next NFL week’s unplayed games get a win probability, the model’s own line and an ' +
          'expected score; grading is idempotent by game id against a paired always-pick-home ' +
          'baseline, and the first morning a game appears on a slate is the pick that counts.',
        files: ['NFL/daily/predict.py', 'NFL/daily/grade.py', 'NFL/daily/config.py'],
      },
      {
        title: 'Season simulation',
        detail:
          'Ten thousand replays of the remaining schedule with live in-sim Elo, then seeds 1–7 per ' +
          'conference and the full bracket; played playoff games are honoured.',
        files: ['NFL/daily/simulate.py'],
      },
      {
        title: 'Publishing',
        detail:
          'Site JSON, the Tue/Thu update email, and the archive copy of that email, run daily by ' +
          'GitHub Actions.',
        files: ['NFL/daily/export_site.py', 'NFL/daily/emails.py', 'NFL/daily/run.py',
          '.github/workflows/nfl-daily.yml'],
      },
    ],
    readme: 'NFL/elo/README.md',
  },
  {
    key: 'cfb',
    name: 'College Football Elo',
    emoji: '🎓',
    summary:
      'An FBS Elo over every FBS-involved game since 2001 with conference-aware season ' +
      'regression, one pooled FCS opponent and a capped margin, plus the same daily grade / ' +
      'predict / simulate loop.',
    facts: [
      { label: 'Spine', value: 'cfbfastR-data (ESPN-derived), 2001–present' },
      { label: 'Tuned', value: '2005–2023 · holdout 2024–25' },
      { label: 'Holdout log loss', value: '0.499 (always-home 0.652)' },
      { label: 'Parameters', value: 'K 35 · home +50 · regression 30% toward 75% conference mean' },
    ],
    steps: [
      {
        title: 'Data',
        detail:
          'One ESPN-derived CSV per season with per-season conference for both teams, an FBS/FCS ' +
          'tag, the neutral flag and scores; ESPN’s scoreboard fills the trailing week.',
        files: ['CFB/data/fetch_schedule.py', 'CFB/data/teams.py', 'data/college_football/games.csv'],
      },
      {
        title: 'Rating engine and tuning',
        detail:
          'Shared Elo skeleton plus the four college rules: regression toward the new ' +
          'conference’s mean, a fixed pooled FCS rating, an FBS entry rating and a margin cap.',
        files: ['CFB/model/elo.py', 'CFB/model/tune.py', 'CFB/model/artifacts/tuned_params.json'],
      },
      {
        title: 'Daily pipeline',
        detail:
          'Replay, score model, two-day slate, idempotent grading with the paired always-home ' +
          'baseline, and a rest-of-regular-season Monte Carlo (expected wins, bowl, conference ' +
          'title; the 12-team playoff is a committee pick and is not modelled).',
        files: ['CFB/daily/state.py', 'CFB/daily/scoring.py', 'CFB/daily/predict.py',
          'CFB/daily/grade.py', 'CFB/daily/simulate.py', 'CFB/daily/run.py'],
      },
      {
        title: 'Publishing',
        detail: 'Site JSON, the Mon/Thu update email and its archive copy.',
        files: ['CFB/daily/export_site.py', 'CFB/daily/emails.py', '.github/workflows/cfb-daily.yml'],
      },
    ],
    readme: 'CFB/README.md',
  },
  {
    key: 'soccer',
    name: 'Club Soccer',
    emoji: '⚽',
    summary:
      'One Elo pool per country spanning the top flight and its second division, glued across ' +
      'countries by European cross-play, feeding a three-way outcome model with squad-economics ' +
      'and xG-form features and a Poisson score model.',
    facts: [
      { label: 'Leagues', value: 'EPL, Bundesliga, La Liga, Serie A, Ligue 1 and their second tiers; MLS unglued' },
      { label: 'Outcome model', value: 'multinomial logistic on six home-minus-away differentials' },
      { label: 'Validated', value: 'two-season holdout from 2024-25' },
      { label: 'Picks', value: 'three-way W/D/L, so log loss reads against ~1.10' },
    ],
    steps: [
      {
        title: 'Data',
        detail:
          'openfootball results per league, UEFA fixtures, Transfermarkt spend aggregates, ' +
          'market-value uploads and Understat xG per match.',
        files: ['soccer/clubs/data/leagues.py', 'soccer/clubs/data/fetch_results.py',
          'soccer/clubs/data/fetch_uefa.py', 'soccer/clubs/data/fetch_transfers.py',
          'soccer/clubs/data/fetch_xg.py'],
      },
      {
        title: 'Rating engine',
        detail:
          'Per-league tuned K, home edge, season regression and entry ratings; promoted and ' +
          'relegated clubs carry their rating through a tier blend; UEFA matches move ratings ' +
          'across pools at a reduced weight.',
        files: ['soccer/clubs/model/elo.py', 'soccer/clubs/model/europe.py', 'soccer/clubs/model/tune.py',
          'soccer/clubs/model/artifacts/tuned_params.json'],
      },
      {
        title: 'Features',
        detail:
          'The venue-adjusted Elo gap, transfer spend and net spend, squad value and wage bill ' +
          '(each z-scored within league-season and taken home minus away), and rolling xG net form.',
        files: ['soccer/clubs/model/features.py', 'soccer/clubs/model/xg.py'],
      },
      {
        title: 'Outcome model',
        detail:
          'A multinomial logistic over win / draw / loss, refit from the replay history on every ' +
          'run; the training script holds out two seasons and reports per-league metrics.',
        files: ['soccer/clubs/model/train.py', 'soccer/clubs/daily/state.py'],
      },
      {
        title: 'Sides vs. differences study',
        detail:
          'Four alternatives — the production differentials, every feature split into home and ' +
          'away values, sides plus each side’s goals scored and allowed, and a Poisson goals ' +
          'regressor — trained before 2026 and tested on this year’s matches, with importances.',
        files: ['soccer/clubs/model/compare_sides.py', 'soccer/clubs/model/report_sides_2026.py',
          'soccer/clubs/model/render_sides_report.py', 'reports/soccer/sides_vs_diffs_2026.html'],
      },
      {
        title: 'Score model, slate, grading, simulation',
        detail:
          'Independent Poisson goal rates from the Elo expectancy and each league’s goals total; ' +
          'a two-day slate; per-match grading; a Monte Carlo of every top-flight table.',
        files: ['soccer/clubs/daily/scoring.py', 'soccer/clubs/daily/predict.py',
          'soccer/clubs/daily/grade.py', 'soccer/clubs/daily/simulate.py', 'soccer/clubs/daily/run.py'],
      },
      {
        title: 'Publishing',
        detail: 'Site JSON, the Mon/Thu update email and its archive copy.',
        files: ['soccer/clubs/daily/export_site.py', 'soccer/clubs/daily/emails.py',
          '.github/workflows/soccer-daily.yml'],
      },
    ],
    readme: 'soccer/clubs/SPEC.md',
  },
  {
    key: 'mlb',
    name: 'MLB Elo',
    emoji: '⚾',
    summary:
      'A betting-blind team Elo with a starting-pitcher adjustment, rest and travel edges, a ' +
      'Poisson-style score model and a rest-of-season Monte Carlo, graded every morning.',
    facts: [
      { label: 'Spine', value: 'game logs 2009–present, MLB Stats API for finals and probables' },
      { label: 'Parameters', value: 'K 3 · home +24 · margin-of-victory weighted' },
      { label: 'Backtest', value: '2009–2025: 56.7% straight up, log loss 0.680 vs 0.691 always-home' },
      { label: 'Active model', value: 'v2-sp (starting-pitcher adjusted), cut over 2026-08-15' },
    ],
    steps: [
      {
        title: 'Data',
        detail: 'Game logs, the remaining schedule, and probable starters pulled daily.',
        files: ['mlb/build_games.py', 'mlb/daily/update_games.py', 'mlb/build_starts_statsapi.py'],
      },
      {
        title: 'Rating engine and tuning',
        detail: 'Team Elo with margin weighting; the starter adjustment prices each probable pitcher.',
        files: ['mlb/elo.py', 'mlb/tune_elo.py', 'mlb/pitcher_rating.py', 'mlb/adjustments.py',
          'mlb/daily/ratings.py', 'mlb/daily/sp_state.py'],
      },
      {
        title: 'Slate, grading, simulation',
        detail:
          'Today’s slate with simulated scores, yesterday graded against the paired always-home ' +
          'baseline, and the season Monte Carlo (division, playoffs, top seed).',
        files: ['mlb/daily/scoring.py', 'mlb/daily/simulate.py', 'mlb/daily/grade.py',
          'mlb/daily/run.py', 'mlb/daily/config.py'],
      },
      {
        title: 'Publishing',
        detail: 'Three morning emails (futures, slate, grade), the site JSON and the archive copies.',
        files: ['mlb/daily/emails.py', 'mlb/daily/export_site.py', 'mlb/daily/send_ledger.py',
          '.github/workflows/daily-report.yml'],
      },
    ],
    readme: 'mlb/daily/README.md',
  },
  {
    key: 'nfl-picks',
    name: 'NFL picks model (market-aware)',
    emoji: '📈',
    summary:
      'The older LightGBM / logistic picks model that also sees the closing line. It exists to ' +
      'answer “can we beat the close?” (no) and runs in the Tue/Fri weekly report; it is a ' +
      'different animal from the betting-blind Elo above.',
    facts: [
      { label: 'Features', value: '45 market, rating, form, schedule and context features' },
      { label: 'Validation', value: 'walk-forward, one retrain per season, 2010–2025' },
      { label: 'Result', value: '64.3% straight up vs 66.4% for the closing line; ATS 50.5%' },
    ],
    steps: [
      {
        title: 'Dataset and model',
        detail: 'Feature construction, the walk-forward trainer and the calibrated scorecard.',
        files: ['NFL/model/v2/dataset.py', 'NFL/model/v2/train.py', 'NFL/model/v2/scorecard.py',
          'NFL/model/v2/compare_models.py'],
      },
      {
        title: 'Importance study',
        detail:
          'Permutation and SHAP importances across five model families; read by the weekly ' +
          'models-check email, which draws them as charts.',
        files: ['NFL/model/v2/feature_importance.py', 'NFL/model/v2/artifacts/importance/',
          'data_jobs/reports/models_check.py', 'data_jobs/reports/email_charts.py'],
      },
      {
        title: 'Weekly report',
        detail: 'Picks, the ledger and the bookmaker leaderboard, Tuesday and Friday nights.',
        files: ['data_jobs/reports/weekly_nfl_report.py', '.github/workflows/weekly-nfl-report.yml'],
      },
    ],
    readme: 'NFL/model/v2/README.md',
  },
];

export const SHARED_DOCS: DocStep[] = [
  {
    title: 'Cross-sport summary',
    detail: 'Every model’s graded record, game-weighted, for the home page’s track-record strip.',
    files: ['data_jobs/build_summary.py', 'web/public/data/summary.json'],
  },
  {
    title: 'Email archive and send ledger',
    detail:
      'Every report email is archived as sent with a permalink footer and listed by league; ' +
      'the ledger keeps reruns from double-sending.',
    files: ['data_jobs/email_archive.py', 'data_jobs/email_ledger.py', 'web/public/emails/index.json'],
  },
];
