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
  /** Rating and rank on the morning of the season's first game — after
   * the August regression, before any result. */
  preseason_elo: number | null;
  preseason_rank: number | null;
}

export interface CfbConference {
  name: string;
  short: string;
  teams: number;
  /** Elo bands: the average, the four strongest, the median, the four
   * weakest — a conference's average, ceiling, middle and floor. */
  avgElo: number;
  top4Elo: number | null;
  medianElo: number | null;
  bottom4Elo: number | null;
  preseasonAvgElo: number | null;
  /** Programs currently inside the site's top N. */
  topN: number;
  bestTeam: string;
  bestElo: number;
  worstTeam: string;
  worstElo: number;
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

/** Where the rest-of-season sim expects each program's rating to go:
 * [ISO date, median run, 10th pct, 90th pct] per checkpoint, opening on
 * today's actual rating so the chart can continue the history line. The
 * middle value is one real simulated season (the one that finished
 * mid-distribution), not an average — averaging returns today's rating
 * for everyone. Absent once there is nothing left to simulate. */
export interface CfbEloProjection {
  season: number;
  sims: number;
  from_date: string;
  teams: Record<string, [string, number, number, number][]>;
  /** A few whole simulated seasons per team, one rating per checkpoint.
   * Unlike the median runs above, path `i` of every team comes from the
   * same simulated season, so their crossings are one coherent board. */
  samples: Record<string, number[][]>;
}

/** One FBS-vs-FBS game between two different conferences (the FBS
 * independents count as one), with the Elo replay's pre-game home
 * probability — what the engine thought that morning, no hindsight. */
export interface CfbInterconferenceGame {
  game_id: number;
  season: number;
  week: number;
  season_type: string;
  date: string;
  home_team: string;
  away_team: string;
  home_conference: string;
  away_conference: string;
  neutral: boolean;
  elo_home_pre: number;
  elo_away_pre: number;
  /** Home Elo + home edge (0 at neutral sites) − away Elo. */
  elo_diff: number;
  p_home: number;
  home_points: number;
  away_points: number;
  /** Only on `fcs_games`: which side is the FCS program. */
  home_fcs?: boolean;
  away_fcs?: boolean;
}

/** The same grading over in-conference FBS games, as sums per season and
 * season type so any filter combination can be pooled. */
export interface CfbConferenceBaseline {
  season: number;
  season_type: string;
  games: number;
  correct: number;
  ll_sum: number;
  brier_sum: number;
}

export interface CfbInterconference {
  seasons: number[];
  games: CfbInterconferenceGame[];
  /** FBS vs. FCS over the same seasons — never pooled with `games`. */
  fcs_games?: CfbInterconferenceGame[];
  conference_baseline: CfbConferenceBaseline[];
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
  elo_projection?: CfbEloProjection;
  interconference?: CfbInterconference;
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
