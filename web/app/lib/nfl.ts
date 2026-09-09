/**
 * The NFL daily pipeline's output.
 *
 * `public/data/nfl/latest.json` is rewritten and committed every morning by
 * `NFL/daily/export_site.py`; the site redeploys with each commit, so this is
 * imported at build time like the rest of the static data. Kept apart from
 * `cfb.ts`, `mlb.ts` and `soccer.ts` for the same reason those are kept apart
 * from `data.ts` — the NFL page pulls in only this file.
 */
import latest from '@/public/data/nfl/latest.json';

export interface NflRating {
  team: string;
  name: string;
  conference: string;
  division: string;
  elo: number;
  wins: number;
  losses: number;
  ties: number;
  div_wins: number;
  div_losses: number;
  pts_diff: number;
  /** Career games in the replay (1999-present, franchise continuous). */
  games: number;
  rank: number;
  /** Rating and rank on the morning of the season's first game — after
   * the off-season regression, before any result. */
  preseason_elo: number | null;
  preseason_rank: number | null;
}

export interface NflDivision {
  name: string;
  conference: string;
  teams: number;
  avgElo: number;
  preseasonAvgElo: number | null;
  /** Teams currently inside the top 10. */
  top10: number;
  bestTeam: string;
  bestElo: number;
  worstTeam: string;
  worstElo: number;
}

export interface NflSlateRow {
  game_id: string;
  date: string;
  weekday: string;
  gametime: string;
  season: number;
  week: number;
  game_type: string;
  week_label: string;
  home_team: string;
  away_team: string;
  home_name: string;
  away_name: string;
  home_division: string | null;
  away_division: string | null;
  div_game: boolean;
  neutral: boolean;
  home_rest: number | null;
  away_rest: number | null;
  elo_home_pre: number;
  elo_away_pre: number;
  p_home: number;
  pick: string;
  pick_prob: number;
  /** The model's own line, nflverse sign: positive = home favoured. */
  elo_spread: number;
  pred_home_score: number;
  pred_away_score: number;
  pred_total: number;
}

export interface NflGradedRow {
  game_id: string;
  date: string;
  week: number;
  week_label: string;
  home_team: string;
  away_team: string;
  neutral: boolean;
  pick: string;
  p_home: number;
  pick_prob: number;
  elo_spread: number;
  pred_home_score: number;
  pred_away_score: number;
  home_score: number;
  away_score: number;
  tie: boolean;
  pick_correct: boolean;
  log_loss: number;
  brier: number;
  d_ll: number;
}

export interface NflWindowStats {
  graded: number;
  correct?: number;
  ties?: number;
  accuracy?: number;
  log_loss?: number;
  brier?: number;
  home_log_loss?: number;
  avg_margin_err?: number;
  avg_total_err?: number;
  /** Paired per-game Δlog-loss vs. always-pick-home (negative = ahead). */
  d_ll_mean?: number;
  d_ll_se?: number | null;
}

export interface NflLedger extends NflWindowStats {
  first_date?: string;
  last_date?: string;
  by_week?: Record<string, NflWindowStats>;
  rolling?: Record<string, NflWindowStats>;
}

export interface NflFuturesTeam {
  team: string;
  name: string;
  conference: string;
  division: string;
  elo: number;
  wins: number;
  losses: number;
  ties: number;
  pts_diff: number;
  games_left: number;
  exp_wins: number;
  exp_losses: number;
  p_division: number;
  p_playoffs: number;
  p_top_seed: number;
  p_conf: number;
  p_sb: number;
}

export interface NflFutures {
  season: number;
  sims?: number;
  remaining_games?: number;
  status?: string;
  teams?: NflFuturesTeam[];
}

/** Every team's current-season Elo trajectory: the preseason rating, the
 * pre-game rating at each game date, closed with the live rating. */
export interface NflEloHistory {
  season: number;
  teams: Record<string, [string, number][]>;
}

export interface NflLatest {
  generated_at: string;
  run_date: string;
  season: number;
  week: number | null;
  week_label: string | null;
  params: Record<string, number>;
  ratings: NflRating[];
  divisions: Record<string, NflDivision>;
  slate: NflSlateRow[];
  graded_today: NflGradedRow[];
  graded_recent: NflGradedRow[];
  ledger: NflLedger;
  futures: NflFutures;
  elo_history?: NflEloHistory;
}

const data = latest as unknown as NflLatest;

export function getNflLatest(): NflLatest {
  return data;
}

/** Divisions ordered by average Elo, strongest first. */
export function orderedDivisions(): NflDivision[] {
  return Object.values(data.divisions ?? {}).sort((a, b) => b.avgElo - a.avgElo);
}

/** "Away @ Home", or "Away vs. Home (N)" at a neutral site. */
export function matchupLabel(row: { home_team: string; away_team: string; neutral: boolean }): string {
  return row.neutral
    ? `${row.away_team} vs. ${row.home_team} (N)`
    : `${row.away_team} @ ${row.home_team}`;
}

/** "12–5" or "12–4–1"; a win–loss record takes an en dash. */
export function fmtWlt(wins: number, losses: number, ties: number): string {
  return ties ? `${wins}–${losses}–${ties}` : `${wins}–${losses}`;
}

/**
 * The model's spread from the home side the way a bettor reads it: the
 * exporter's positive number means the home team is favoured, shown as a
 * negative line (`-3.5`); a dead-even line reads PK.
 */
export function fmtEloLine(spread: number | null | undefined): string {
  if (spread === null || spread === undefined || Number.isNaN(spread)) return '—';
  const line = -spread;
  if (Math.abs(line) < 0.05) return 'PK';
  return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
}
