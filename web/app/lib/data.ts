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

export function getSlate(): Slate {
  return slateJson as unknown as Slate;
}

export function getMeta(): Meta {
  return metaJson as unknown as Meta;
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

/** 0.6421 -> "64%". */
export function fmtPct(prob: number | null | undefined): string {
  if (prob === null || prob === undefined || Number.isNaN(prob)) return DASH;
  return `${Math.round(prob * 100)}%`;
}
