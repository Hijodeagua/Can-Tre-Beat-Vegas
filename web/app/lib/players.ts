/**
 * Player-comparison data and maths.
 *
 * The files under `public/data/soccer/players/` are written by
 * `soccer/clubs/data/understat_players.py`: an `index.json` listing the
 * seasons and one `<year>.json` per season, each a column list plus rows.
 * Unlike every other data file on the site these are fetched in the
 * browser rather than imported at build time - a season is a few hundred
 * kB and the page only ever needs the seasons someone actually picks.
 *
 * Everything below the loaders is pure: a stat catalogue, the value of a
 * stat for a player in either display mode, and percentile rank inside a
 * peer pool. The page is those three things and a table.
 */

/** Must match `basePath` in next.config.mjs - `fetch` does not get it for free. */
const BASE = '/vegas';
const DIR = `${BASE}/data/soccer/players`;

export type PositionGroup = 'GK' | 'DF' | 'MF' | 'FW';

export interface PlayerSeason {
  /** `<season key>:<league>:<understat id>` - unique across the whole dataset. */
  key: string;
  season: number;
  seasonLabel: string;
  id: string;
  name: string;
  team: string;
  league: string;
  /** Understat's raw role string, e.g. "F M S". */
  pos: string;
  group: PositionGroup | null;
  games: number | null;
  min: number;
  goals: number | null;
  assists: number | null;
  shots: number | null;
  key_passes: number | null;
  xg: number | null;
  xa: number | null;
  npg: number | null;
  npxg: number | null;
  xgchain: number | null;
  xgbuildup: number | null;
  yellow: number | null;
  red: number | null;
}

export interface SeasonEntry {
  key: number;
  label: string;
  file: string;
  players: number;
  leagues: string[];
}

export interface PlayerIndex {
  generated_at: string;
  source: string;
  leagues: Record<string, string>;
  seasons: SeasonEntry[];
}

/** null when the export has never run (no index.json yet) - not an error. */
export async function loadIndex(): Promise<PlayerIndex | null> {
  const res = await fetch(`${DIR}/index.json`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`index.json: HTTP ${res.status}`);
  return (await res.json()) as PlayerIndex;
}

const seasonCache = new Map<number, Promise<PlayerSeason[]>>();

export function loadSeason(entry: SeasonEntry): Promise<PlayerSeason[]> {
  let hit = seasonCache.get(entry.key);
  if (!hit) {
    hit = fetch(`${DIR}/${entry.file}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${entry.file}: HTTP ${res.status}`);
        return res.json();
      })
      .then((body: { columns: string[]; rows: unknown[][] }) =>
        body.rows.map((values) => {
          const row: Record<string, unknown> = {};
          body.columns.forEach((c, i) => {
            row[c] = values[i];
          });
          return {
            ...(row as unknown as PlayerSeason),
            key: `${entry.key}:${row.league}:${row.id}`,
            season: entry.key,
            seasonLabel: entry.label,
            min: Number(row.min) || 0,
          };
        }),
      );
    hit.catch(() => seasonCache.delete(entry.key));
    seasonCache.set(entry.key, hit);
  }
  return hit;
}

/* ------------------------------------------------------------------ */
/* stat catalogue                                                      */
/* ------------------------------------------------------------------ */

export type Mode = 'p90' | 'total';

export interface StatDef {
  key: string;
  label: string;
  /** Longer explanation, shown as the row's tooltip. */
  hint?: string;
  section: 'Playing time' | 'Shooting' | 'Creation' | 'Possession chains' | 'Discipline';
  /**
   * `count` divides by 90s in per-90 mode; `raw` is shown as-is in both
   * (minutes, appearances, and ratios that are already rates).
   */
  kind: 'count' | 'raw';
  value: (p: PlayerSeason) => number | null;
  digits: { p90: number; total: number };
  /** Lower is better (cards): the percentile is flipped so a long bar is always good. */
  lowerBetter?: boolean;
  /** Left out of the similarity distance (volume and discipline say little about style). */
  noSimilarity?: boolean;
  /** No percentile: the number is context, not a skill (appearances). */
  noRank?: boolean;
}

const sum = (a: number | null, b: number | null) => (a === null || b === null ? null : a + b);
const sub = (a: number | null, b: number | null) => (a === null || b === null ? null : a - b);
const ratio = (a: number | null, b: number | null) =>
  a === null || b === null || b <= 0 ? null : a / b;

export const STATS: StatDef[] = [
  { key: 'games', label: 'Appearances', section: 'Playing time', kind: 'raw',
    value: (p) => p.games, digits: { p90: 0, total: 0 }, noSimilarity: true, noRank: true },
  { key: 'min', label: 'Minutes', section: 'Playing time', kind: 'raw',
    value: (p) => p.min, digits: { p90: 0, total: 0 }, noSimilarity: true },

  { key: 'goals', label: 'Goals', section: 'Shooting', kind: 'count',
    value: (p) => p.goals, digits: { p90: 2, total: 0 }, noSimilarity: true },
  { key: 'npg', label: 'Non-penalty goals', section: 'Shooting', kind: 'count',
    value: (p) => p.npg, digits: { p90: 2, total: 0 } },
  { key: 'xg', label: 'xG', hint: 'Expected goals, penalties included', section: 'Shooting',
    kind: 'count', value: (p) => p.xg, digits: { p90: 2, total: 1 }, noSimilarity: true },
  { key: 'npxg', label: 'npxG', hint: 'Non-penalty expected goals', section: 'Shooting',
    kind: 'count', value: (p) => p.npxg, digits: { p90: 2, total: 1 } },
  { key: 'shots', label: 'Shots', section: 'Shooting', kind: 'count',
    value: (p) => p.shots, digits: { p90: 2, total: 0 } },
  { key: 'npxg_shot', label: 'npxG per shot',
    hint: 'Average chance quality. Uses all shots, so penalty takers read slightly low.',
    section: 'Shooting', kind: 'raw', value: (p) => ratio(p.npxg, p.shots),
    digits: { p90: 2, total: 2 } },
  { key: 'g_minus_xg', label: 'npG − npxG',
    hint: 'Finishing over or under expectation. Noisy inside one season.',
    section: 'Shooting', kind: 'count', value: (p) => sub(p.npg, p.npxg),
    digits: { p90: 2, total: 1 }, noSimilarity: true },

  { key: 'assists', label: 'Assists', section: 'Creation', kind: 'count',
    value: (p) => p.assists, digits: { p90: 2, total: 0 }, noSimilarity: true },
  { key: 'xa', label: 'xA', hint: 'Expected assists: the xG of the shots a player set up',
    section: 'Creation', kind: 'count', value: (p) => p.xa, digits: { p90: 2, total: 1 } },
  { key: 'key_passes', label: 'Key passes', hint: 'Passes leading directly to a shot',
    section: 'Creation', kind: 'count', value: (p) => p.key_passes, digits: { p90: 2, total: 0 } },
  { key: 'npxg_xa', label: 'npxG + xA', section: 'Creation', kind: 'count',
    value: (p) => sum(p.npxg, p.xa), digits: { p90: 2, total: 1 }, noSimilarity: true },

  { key: 'xgchain', label: 'xGChain',
    hint: 'xG of every possession the player touched that ended in a shot',
    section: 'Possession chains', kind: 'count', value: (p) => p.xgchain,
    digits: { p90: 2, total: 1 } },
  { key: 'xgbuildup', label: 'xGBuildup',
    hint: 'xGChain without the shot and the key pass: involvement earlier in the move',
    section: 'Possession chains', kind: 'count', value: (p) => p.xgbuildup,
    digits: { p90: 2, total: 1 } },

  { key: 'yellow', label: 'Yellow cards', section: 'Discipline', kind: 'count',
    value: (p) => p.yellow, digits: { p90: 2, total: 0 }, lowerBetter: true, noSimilarity: true },
  { key: 'red', label: 'Red cards', section: 'Discipline', kind: 'count',
    value: (p) => p.red, digits: { p90: 2, total: 0 }, lowerBetter: true, noSimilarity: true },
];

/** A stat's value for a player in the chosen display mode; null when unknown. */
export function statValue(stat: StatDef, p: PlayerSeason, mode: Mode): number | null {
  const v = stat.value(p);
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  if (stat.kind === 'raw' || mode === 'total') return v;
  return p.min > 0 ? (v * 90) / p.min : null;
}

export function fmtStat(stat: StatDef, value: number | null, mode: Mode): string {
  if (value === null) return '—';
  const digits = stat.kind === 'raw' ? stat.digits.total : stat.digits[mode];
  const s = value.toFixed(digits);
  // "-0.00" is a zero; and a signed stat reads better with its plus.
  const clean = Number(s) === 0 ? (0).toFixed(digits) : s;
  return stat.key === 'g_minus_xg' && Number(clean) > 0 ? `+${clean}` : clean;
}

/* ------------------------------------------------------------------ */
/* peer pools and percentiles                                          */
/* ------------------------------------------------------------------ */

export type MinutesRule = 'auto' | number;
export type Scope = 'top5' | 'league';

/**
 * The minutes floor for a season's pool. `auto` is 900 for a finished
 * season and a third of the most-used player's minutes for one in
 * progress (rounded to 50), so a November comparison is not a pool of
 * nobody.
 */
export function minutesFloor(rule: MinutesRule, season: PlayerSeason[]): number {
  if (rule !== 'auto') return rule;
  let max = 0;
  for (const p of season) if (p.min > max) max = p.min;
  return Math.min(900, Math.max(90, Math.round(max / 3 / 50) * 50));
}

/**
 * The players a subject is ranked against: same season, same position
 * group, at or above the minutes floor, in all five leagues or only the
 * subject's own. The subject is always a member, so a player under the
 * floor still gets a rank (the page marks it as a small sample).
 */
export function peerPool(
  subject: PlayerSeason,
  season: PlayerSeason[],
  opts: { group: PositionGroup | null; floor: number; scope: Scope },
): PlayerSeason[] {
  const pool = season.filter(
    (p) =>
      p.group === opts.group &&
      p.min >= opts.floor &&
      (opts.scope === 'top5' || p.league === subject.league),
  );
  if (!pool.some((p) => p.key === subject.key)) pool.push(subject);
  return pool;
}

/**
 * Percentile rank, 0-100: the share of the pool strictly below the value
 * plus half of those level with it (mid-rank, so a pool where everyone
 * has zero red cards reads 50, not 0 or 100). Flipped for lower-is-better
 * stats. null when the subject has no value or the pool is too small to
 * mean anything.
 */
export function percentile(
  stat: StatDef,
  subject: PlayerSeason,
  pool: PlayerSeason[],
  mode: Mode,
): number | null {
  if (stat.noRank) return null;
  const v = statValue(stat, subject, mode);
  if (v === null) return null;
  let below = 0;
  let level = 0;
  let n = 0;
  for (const p of pool) {
    const x = statValue(stat, p, mode);
    if (x === null) continue;
    n += 1;
    if (x < v) below += 1;
    else if (x === v) level += 1;
  }
  if (n < MIN_POOL) return null;
  const pct = ((below + level / 2) / n) * 100;
  return Math.round(stat.lowerBetter ? 100 - pct : pct);
}

/** Below this many ranked peers a percentile is not shown. */
export const MIN_POOL = 10;

/**
 * Nearest statistical neighbours inside the pool: Euclidean distance over
 * z-scored per-90 style stats (shot volume and quality, creation,
 * possession-chain involvement). Returns the closest first, with a 0-100
 * similarity that is only meaningful relative to the others in the list.
 */
export function similarPlayers(
  subject: PlayerSeason,
  pool: PlayerSeason[],
  count = 8,
): { player: PlayerSeason; score: number }[] {
  const stats = STATS.filter((s) => !s.noSimilarity);
  const cols = stats.map((stat) => {
    const xs = pool.map((p) => statValue(stat, p, 'p90'));
    const known = xs.filter((x): x is number => x !== null);
    const mean = known.reduce((a, b) => a + b, 0) / Math.max(1, known.length);
    const sd = Math.sqrt(
      known.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, known.length),
    );
    return { stat, mean, sd };
  });
  const vector = (p: PlayerSeason) =>
    cols.map(({ stat, mean, sd }) => {
      const x = statValue(stat, p, 'p90');
      return x === null || sd === 0 ? 0 : (x - mean) / sd;
    });
  const me = vector(subject);
  const out = pool
    .filter((p) => p.key !== subject.key && p.id !== subject.id)
    .map((p) => {
      const v = vector(p);
      const d = Math.sqrt(v.reduce((a, x, i) => a + (x - me[i]) ** 2, 0));
      return { player: p, d };
    })
    .sort((a, b) => a.d - b.d)
    .slice(0, count);
  // exp(-d / sqrt(k)) maps "identical" to 100 and falls off with distance
  // in units of one standard deviation per stat.
  const k = Math.sqrt(cols.length);
  return out.map(({ player, d }) => ({ player, score: Math.round(100 * Math.exp(-d / k)) }));
}

/** "José Giménez" matches "jose gim". */
export function fold(text: string): string {
  return text.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
}

export const GROUP_LABEL: Record<PositionGroup, string> = {
  GK: 'Goalkeepers',
  DF: 'Defenders',
  MF: 'Midfielders',
  FW: 'Forwards',
};
