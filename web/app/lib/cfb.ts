/**
 * The daily college-football pipeline's output.
 *
 * `public/data/cfb/latest.json` is rewritten and committed every morning by
 * `CFB/daily/export_site.py`; the site redeploys with each commit, so this is
 * imported at build time like the rest of the static data. Kept apart from
 * `mlb.ts` and `soccer.ts` for the same reason those are kept apart from
 * `data.ts` — the CFB page pulls in only this file.
 */
import latest from '@/public/data/cfb/latest.json';

export interface CfbRating {
  team: string;
  conference: string | null;
  conference_short: string;
  elo: number;
  wins: number;
  losses: number;
  conf_wins: number;
  conf_losses: number;
  pts_diff: number;
  /** Career FBS games in the replay (2001-present). */
  games: number;
  rank: number;
}

export interface CfbConference {
  name: string;
  short: string;
  teams: number;
  avgElo: number;
  top4Elo: number | null;
  bottom4Elo: number | null;
  bestTeam: string;
  bestElo: number;
  independent: boolean;
}

export interface CfbSlateRow {
  game_id: number;
  date: string;
  season: number;
  week: number;
  season_type: string;
  home_team: string;
  away_team: string;
  home_conference: string | null;
  away_conference: string | null;
  neutral: boolean;
  home_fcs: boolean;
  away_fcs: boolean;
  elo_home_pre: number;
  elo_away_pre: number;
  p_home: number;
  pick: string;
  pick_prob: number;
  pred_home_score: number;
  pred_away_score: number;
  pred_total: number;
  notes: string;
}

export interface CfbGradedRow {
  game_id: number;
  date: string;
  week: number;
  home_team: string;
  away_team: string;
  neutral: boolean;
  home_fcs: boolean;
  away_fcs: boolean;
  pick: string;
  p_home: number;
  pick_prob: number;
  pred_home_score: number;
  pred_away_score: number;
  home_points: number;
  away_points: number;
  pick_correct: boolean;
  log_loss: number;
  brier: number;
  d_ll: number;
}

export interface CfbWindowStats {
  graded: number;
  correct?: number;
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

export interface CfbLedger extends CfbWindowStats {
  first_date?: string;
  last_date?: string;
  fbs_only?: CfbWindowStats;
  by_week?: Record<string, CfbWindowStats>;
  rolling?: Record<string, CfbWindowStats>;
}

export interface CfbFuturesTeam {
  team: string;
  conference: string | null;
  elo: number;
  wins: number;
  losses: number;
  conf_wins: number;
  conf_losses: number;
  pts_diff: number;
  games_left: number;
  exp_wins: number;
  exp_losses: number;
  p_bowl: number;
  p_undefeated: number;
  /** Null for independents — no conference championship game. */
  p_ccg: number | null;
  p_conf_title: number | null;
}

export interface CfbFutures {
  season: number;
  sims?: number;
  remaining_games?: number;
  status?: string;
  teams?: CfbFuturesTeam[];
}

/** Every program's current-season Elo trajectory: pre-game rating at each
 * game date, closed with the live rating on the run date. */
export interface CfbEloHistory {
  season: number;
  teams: Record<string, [string, number][]>;
}

export interface CfbLatest {
  generated_at: string;
  run_date: string;
  season: number;
  week: number | null;
  params: Record<string, number>;
  top_n: number;
  ratings: CfbRating[];
  conferences: Record<string, CfbConference>;
  slate: CfbSlateRow[];
  graded_today: CfbGradedRow[];
  graded_recent: CfbGradedRow[];
  ledger: CfbLedger;
  futures: CfbFutures;
  elo_history?: CfbEloHistory;
}

const data = latest as unknown as CfbLatest;

export function getCfbLatest(): CfbLatest {
  return data;
}

/** Conferences ordered by average Elo, strongest first. */
export function orderedConferences(): CfbConference[] {
  return Object.values(data.conferences ?? {}).sort((a, b) => b.avgElo - a.avgElo);
}

/** "Away @ Home", or "Away vs. Home (N)" at a neutral site — the one
 * matchup spelling the CFB page and the home tab share. */
export function matchupLabel(row: { home_team: string; away_team: string; neutral: boolean }): string {
  return row.neutral
    ? `${row.away_team} vs. ${row.home_team} (N)`
    : `${row.away_team} @ ${row.home_team}`;
}
