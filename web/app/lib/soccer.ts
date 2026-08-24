/**
 * The daily club-soccer pipeline's output.
 *
 * `public/data/soccer/latest.json` is rewritten and committed by
 * `soccer/clubs/daily/run.py`; the site redeploys with each commit, so this
 * is imported at build time like the rest of the static data. Kept apart
 * from `mlb.ts` for the same reason that one is kept apart from `data.ts` —
 * the soccer page pulls in only this file.
 */
import latest from '@/public/data/soccer/latest.json';

export interface SoccerClubRating {
  team: string;
  elo: number;
  matches: number;
}

export interface SoccerLeagueRatings {
  name: string;
  tier: number;
  asOfSeason: string;
  clubs: SoccerClubRating[];
}

export interface SoccerSlateRow {
  date: string;
  league: string;
  season: string;
  home_team: string;
  away_team: string;
  elo_home_pre: number;
  elo_away_pre: number;
  p_H: number;
  p_D: number;
  p_A: number;
  pick: 'H' | 'D' | 'A';
  lambda_home: number;
  lambda_away: number;
  score_home: number;
  score_away: number;
}

export interface SoccerFuturesClub {
  team: string;
  elo: number;
  points: number;
  exp_points: number;
  p_title: number;
  p_top4: number;
  p_relegation: number;
}

export interface SoccerFuturesLeague {
  season: string;
  sims: number;
  remaining_matches: number;
  clubs: SoccerFuturesClub[];
}

/**
 * One league's cross-league ranking row. Current Elo always has a value
 * once the league has played matches; the squad-economics fields are each
 * null until a `market_values` upload exists for that league, and the two
 * "as of" seasons (`valueSeason` vs `squadStatsSeason`) can legitimately
 * differ — value/wage coverage is far ahead of the newer squad-size/age/
 * foreigners columns, which are still being backfilled league by league.
 */
export interface SoccerLeagueRanking {
  name: string;
  tier: number;
  avgElo: number | null;
  eloClubCount: number;
  valueSeason: string | null;
  avgSquadValueEurM: number | null;
  avgWageBillEurM: number | null;
  valueClubCount: number | null;
  squadStatsSeason: string | null;
  avgSquadSize: number | null;
  avgAge: number | null;
  avgForeigners: number | null;
  avgValuePerPlayerEurM: number | null;
  squadStatsClubCount: number | null;
}

export interface SoccerLatest {
  generated_at: string;
  run_date: string;
  ratings: Record<string, SoccerLeagueRatings>;
  league_rankings: Record<string, SoccerLeagueRanking>;
  slate: SoccerSlateRow[];
  graded_today: unknown[];
  ledger: { graded: number };
  futures: Record<string, SoccerFuturesLeague>;
}

const data = latest as unknown as SoccerLatest;

export function getSoccerLatest(): SoccerLatest {
  return data;
}

/**
 * League display order: top flights first (by rough Elo-average pedigree),
 * each immediately followed by its own second division — so the rankings
 * table reads country-pool by country-pool rather than alphabetically. MLS
 * goes last: it's on its own Elo scale (no UEFA glue, different
 * confederation), not one more country pool in the same sequence.
 */
export const LEAGUE_ORDER = [
  'epl', 'championship',
  'bundesliga', 'bundesliga_2',
  'la_liga', 'la_liga_2',
  'serie_a', 'serie_b',
  'ligue_1', 'ligue_2',
  'mls',
];

/** The ten UEFA-glued leagues share one Elo scale; MLS does not (separate
 * confederation, never plays them). Cross-league Elo comparisons — the
 * league-avg-Elo ranking, the global top/bottom-club list — only mean
 * something within this set. */
export const GLUED_LEAGUES = LEAGUE_ORDER.filter((k) => k !== 'mls');

export function orderedLeagueRankings(): [string, SoccerLeagueRanking][] {
  const rankings = data.league_rankings ?? {};
  return LEAGUE_ORDER.filter((k) => k in rankings).map((k) => [k, rankings[k]]);
}
