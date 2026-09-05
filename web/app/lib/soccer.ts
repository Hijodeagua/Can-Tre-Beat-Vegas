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
  /** Mean finishing position across sims — the Opta-style projected spot. */
  exp_position: number;
  p_title: number;
  /** P(top 4) = the UCL places. */
  p_top4: number;
  /** P(5th–6th) = the Europa League band. */
  p_uel: number;
  p_relegation: number;
}

export interface SoccerFuturesLeague {
  season: string;
  sims: number;
  remaining_matches: number;
  /** Present (with no clubs) when the season's fixtures aren't published. */
  status?: string;
  clubs?: SoccerFuturesClub[];
}

/** One club's current-season Elo trajectory: pre-match rating at each
 * match date, closed with the live rating on the run date — so the last
 * point moves every day the pipeline runs. */
export interface SoccerEloHistoryLeague {
  season: string;
  clubs: Record<string, [string, number][]>;
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
  /** Whether this pool trades Elo points with the others via UEFA
   * cross-play — false means `avgElo` sits on its own scale and the
   * `anchored*` fields (if any) are what's cross-league comparable. */
  glued: boolean;
  avgElo: number | null;
  /** Elo bands: mean of the league's 4 strongest clubs, the 10 clubs
   * centered on the median rank, and the 4 weakest — ceiling, midtable,
   * floor. Null where a band doesn't fit the league size. On the pool's
   * own scale — for an unglued league, not cross-league comparable. */
  top4Elo: number | null;
  mid10Elo: number | null;
  bottom4Elo: number | null;
  eloClubCount: number;
  /** Squad-value-implied Elo on the glued scale (see value_anchor.py),
   * for an unglued league only — null for a glued one (its avgElo above
   * is already comparable) and null for an unglued one with no fitted
   * anchor yet (too little value data). Never a substitute for a
   * measured rating: `anchorMethod` names how it was estimated, and
   * `anchorR2`/`anchorResidualStdElo` say how loose it is. */
  anchoredElo: number | null;
  anchoredTop4Elo: number | null;
  anchoredMid10Elo: number | null;
  anchoredBottom4Elo: number | null;
  anchorMethod: string | null;
  anchorFitClubs: number | null;
  anchorR2: number | null;
  anchorResidualStdElo: number | null;
  anchorValueSeason: string | null;
  valueSeason: string | null;
  avgSquadValueEurM: number | null;
  top3SquadValueEurM: number | null;
  avgWageBillEurM: number | null;
  top3WageBillEurM: number | null;
  /** Wage figures fall back to the newest season that has them, which can
   * trail valueSeason — null until any wage upload exists. */
  wageSeason: string | null;
  valueClubCount: number | null;
  squadStatsSeason: string | null;
  avgSquadSize: number | null;
  avgAge: number | null;
  avgForeigners: number | null;
  avgValuePerPlayerEurM: number | null;
  top3ValuePerPlayerEurM: number | null;
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
  elo_history?: Record<string, SoccerEloHistoryLeague>;
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

/** The UEFA-glued leagues share one Elo scale; an unglued league (MLS
 * today — see `glued` on each ranking row, sourced from the pipeline's
 * League registry) does not. Read from data rather than hardcoded, so a
 * future unglued league needs no change here. Cross-league *club* Elo
 * comparisons (the global top/bottom-club list) only mean something
 * within this set — the league table itself can still place an unglued
 * league via its value anchor; see `comparableElo`. */
export const GLUED_LEAGUES = LEAGUE_ORDER.filter(
  (k) => (data.league_rankings?.[k]?.glued ?? true) !== false,
);

export function orderedLeagueRankings(): [string, SoccerLeagueRanking][] {
  const rankings = data.league_rankings ?? {};
  return LEAGUE_ORDER.filter((k) => k in rankings).map((k) => [k, rankings[k]]);
}

/**
 * The Elo figure that's actually comparable across leagues: a glued
 * league's own average/bands, or an unglued league's squad-value anchor
 * when one has been fit. Null means "not placeable on the shared scale
 * yet" — never fall back to an unglued league's own average here, since
 * that number isn't on the same scale (that's the whole reason it's
 * unglued).
 */
export function comparableElo(r: SoccerLeagueRanking): number | null {
  return r.glued ? r.avgElo : r.anchoredElo;
}

export function comparableEloBands(
  r: SoccerLeagueRanking,
): { top4: number | null; mid10: number | null; bottom4: number | null } {
  return r.glued
    ? { top4: r.top4Elo, mid10: r.mid10Elo, bottom4: r.bottom4Elo }
    : { top4: r.anchoredTop4Elo, mid10: r.anchoredMid10Elo, bottom4: r.anchoredBottom4Elo };
}
