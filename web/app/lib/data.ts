/**
 * Static slate data and display formatters.
 *
 * The JSON in `public/data/` is written by `data_jobs/export_web_json.py`; the
 * types below mirror the dicts that module builds. Both files are imported at
 * build time rather than fetched, because the site is fully prerendered and the
 * data only changes when the odds job commits a new snapshot.
 *
 * Every numeric field is nullable: the exporter emits `None` whenever a book
 * did not post that market, so anything rendered from here has to tolerate a
 * missing value. The `fmt*` helpers all fall back to an em dash.
 */
import metaJson from '@/public/data/meta.json';
import slateJson from '@/public/data/slate.json';
import summaryJson from '@/public/data/summary.json';

export interface BookOdds {
  book: string;
  home_ml: number | null;
  away_ml: number | null;
  home_spread_odds: number | null;
  away_spread_odds: number | null;
  over_odds: number | null;
  under_odds: number | null;
}

export interface Consensus {
  home_ml: number | null;
  away_ml: number | null;
  home_spread: number | null;
  home_spread_odds: number | null;
  away_spread_odds: number | null;
  total: number | null;
  over_odds: number | null;
  under_odds: number | null;
  /** No-vig market probability for the home side. */
  home_win_prob: number | null;
}

export interface LineMovement {
  first_seen?: string;
  spread?: number | null;
  total?: number | null;
  home_ml?: number | null;
  /** Consensus minus opener. Absent when either end is missing. */
  spread_delta?: number;
  total_delta?: number;
}

export interface ModelPick {
  predicted_winner: string | null;
  home_win_prob: number | null;
  pred_spread: number | null;
  edge_vs_market: number | null;
}

export interface Game {
  game_id: string;
  commence_et: string;
  home_team: string;
  away_team: string;
  consensus: Consensus;
  books: BookOdds[];
  line_movement: LineMovement | null;
  model: ModelPick | null;
}

export interface Sport {
  key: string;
  name: string;
  snapshot: { file: string; pulled_at_et: string } | null;
  games: Game[];
}

export interface Slate {
  generated_at: string;
  window_hours: number;
  sports: Sport[];
}

export interface Meta {
  last_updated: string;
  window_hours: number;
  game_counts: Record<string, number>;
  snapshots: Record<string, { file: string; pulled_at_et: string }>;
  label: string;
}

/**
 * Cross-sport track record, written by `data_jobs/build_summary.py`.
 *
 * Every metric is nullable and stays null until a model has graded games —
 * the site renders those as em dashes. `roi` is null for every model by
 * design: nothing here stakes money.
 */
export interface SummaryMetrics {
  /** "57-51" with a plain hyphen; `fmtRecord` sets it as an en dash. */
  record: string | null;
  games: number;
  accuracy: number | null;
  log_loss: number | null;
  brier: number | null;
  roi: number | null;
  sports_reporting?: number;
}

export interface SummaryModel extends SummaryMetrics {
  /** Display name, matching a `name` in `app/lib/sports.ts`. */
  sport: string;
  emoji: string;
  status: 'in_season' | 'off_season';
  last_graded: string | null;
  /** Games on this sport's current slate. */
  slate_games: number;
  /** Month-precision season start, present only while off-season. */
  season_starts?: string | null;
}

export interface Summary {
  generated_at: string;
  overall: SummaryMetrics & { sports_reporting: number };
  models: SummaryModel[];
}

export function getSlate(): Slate {
  return slateJson as unknown as Slate;
}

export function getMeta(): Meta {
  return metaJson as unknown as Meta;
}

export function getSummary(): Summary {
  return summaryJson as unknown as Summary;
}

/** The odds-feed sport entry for a key, when the slate carries one. */
export function getSlateSport(key: string): Sport | undefined {
  return getSlate().sports.find((s) => s.key === key);
}

const DASH = '—';

/** American odds: -180, +164, or an em dash when the book had no price. */
export function fmtOdds(odds: number | null | undefined): string {
  if (odds === null || odds === undefined || Number.isNaN(odds)) return DASH;
  const rounded = Math.round(odds);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

/**
 * Point spread from the home team's perspective, signed the way a bettor reads
 * it: the exporter's positive number means the home team is favoured, which is
 * displayed as a negative line (`-3.5`).
 */
export function fmtSpread(spread: number | null | undefined): string {
  if (spread === null || spread === undefined || Number.isNaN(spread)) return DASH;
  const line = -spread;
  return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
}

/**
 * 0.6421 -> "64%". Re-exported from `format.ts` so the whole-number-with-`<1%`
 * rule has exactly one implementation across the site.
 */
export { fmtPct } from '@/app/lib/format';
