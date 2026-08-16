/**
 * The daily MLB pipeline's output.
 *
 * `public/data/mlb/latest.json` is rewritten and committed every morning by
 * `mlb/daily/export_site.py`; the site redeploys with each commit, so this is
 * imported at build time like the rest of the static data. Kept apart from
 * `data.ts` so the client-side tab view pulls in this file alone rather than
 * the odds slate and summary as well.
 */
import latest from '@/public/data/mlb/latest.json';

export interface MlbSlateRow {
  date: string;
  away: string;
  home: string;
  game_num: number;
  /** Probable starters. Present regardless of model version; only a model
   * input (via the *_sp_adj fields below) when the active model uses it. */
  away_sp?: string | null;
  home_sp?: string | null;
  /** Matchup-specific expected total runs (recent-form attack/defense). */
  pred_total?: number | null;
  p_home: number;
  pick: string;
  pick_prob: number;
  pred_home_score: number;
  pred_away_score: number;
  /** Which model version produced this row (v1-team-elo or v2-sp). */
  model_version?: string | null;
  /** Each starter's rating adjustment in Elo points, and how it was
   * derived — present only for a model version that uses starter identity
   * (mlb/pitcher_rating.py's fallback ladder: pitcher/thin/staff). */
  home_sp_adj?: number | null;
  away_sp_adj?: number | null;
  home_sp_mode?: string | null;
  away_sp_mode?: string | null;
  home_rt_adj?: number | null;
  away_rt_adj?: number | null;
}

export interface MlbFuturesRow {
  team: string;
  name?: string;
  division?: string;
  wins?: number;
  losses?: number;
  run_diff?: number;
  elo: number;
  mean_wins: number;
  mean_losses: number;
  division_pct: number;
  playoff_pct: number;
  top_seed_pct: number;
}

export interface MlbGradedRow extends MlbSlateRow {
  played?: boolean;
  home_score?: number | null;
  away_score?: number | null;
  pick_correct?: boolean | null;
}

export interface MlbHistoryRow {
  date: string;
  games: number;
  correct: number;
  accuracy: number | null;
  log_loss: number | null;
  brier: number | null;
  avg_margin_err: number | null;
  avg_total_err: number | null;
  cum_accuracy: number | null;
  cum_log_loss: number | null;
  link: string | null;
}

export interface MlbModelSummary {
  version: string;
  role: string;
  first_date: string;
  last_date: string;
  games: number;
  correct: number;
  accuracy: number | null;
  log_loss: number | null;
  brier: number | null;
  d_ll_vs_home_mean: number | null;
  d_ll_vs_home_se: number | null;
}

export interface MlbLatest {
  generated_at: string | null;
  date: string | null;
  model_version?: string | null;
  models?: MlbModelSummary[] | null;
  slate: MlbSlateRow[];
  futures: MlbFuturesRow[];
  graded_date: string | null;
  graded: MlbGradedRow[] | null;
  history: MlbHistoryRow[];
  team_names: Record<string, string>;
}

const data = latest as unknown as MlbLatest;

export function getMlbLatest(): MlbLatest {
  return data;
}

/** Franchise code -> club name, falling back to the code itself. */
export function teamName(code: string): string {
  return data.team_names?.[code] ?? code;
}

export const DIVISION_ORDER = [
  'AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West',
];

/** Yesterday's graded games that were actually played. */
export function playedGraded(): MlbGradedRow[] {
  return (data.graded ?? []).filter((g) => g.played);
}

/** The ledger row for the most recently graded day, if there is one. */
export function gradedLedgerRow(): MlbHistoryRow | undefined {
  return data.history.find((h) => h.date === data.graded_date);
}
