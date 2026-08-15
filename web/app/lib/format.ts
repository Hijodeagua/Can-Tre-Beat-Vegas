/**
 * Display formatters shared by every data surface.
 *
 * The rule these all exist to enforce: **missing data is an em dash, never a
 * zero and never blank.** A model with no graded games has not gone 0–0 at 0%
 * accuracy — it has no record at all, and rendering one would publish a score
 * it never earned. Every helper here falls back to the dash.
 */

export const DASH = '—';

/** True when a value is absent in the way the JSON exporters emit absence. */
export function missing(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value));
}

/**
 * Whole-number percentage. Anything above zero but below half a point renders
 * `<1%` rather than rounding down to `0%`, which would claim a real zero.
 */
export function fmtPct(value: number | null | undefined): string {
  if (missing(value)) return DASH;
  const v = value as number;
  if (v > 0 && v < 0.005) return '<1%';
  return `${Math.round(v * 100)}%`;
}

/**
 * Percentage to one decimal, for the two display figures the design sets that
 * way: the headline accuracy and the running-accuracy chip. Tables stay on
 * whole numbers via `fmtPct` — this is a headline treatment, not a second
 * rounding rule for data.
 */
export function fmtPctPrecise(value: number | null | undefined): string {
  if (missing(value)) return DASH;
  return `${((value as number) * 100).toFixed(1)}%`;
}

/** Fixed-precision metric (log-loss, Brier), or a dash. */
export function fmtNum(value: number | null | undefined, digits = 3): string {
  return missing(value) ? DASH : (value as number).toFixed(digits);
}

/** Signed integer, used for run differential. */
export function fmtSigned(value: number | null | undefined): string {
  if (missing(value)) return DASH;
  const v = value as number;
  return v > 0 ? `+${v}` : `${v}`;
}

/**
 * "57-51" -> "57–51". The pipeline writes a plain hyphen; the site sets it as
 * an en dash, which is what a win–loss record takes.
 */
export function fmtRecord(record: string | null | undefined): string {
  if (missing(record) || record === '') return DASH;
  return (record as string).replace('-', '–');
}

/**
 * ROI is an em dash by design — nothing here stakes money, so there is no
 * return to report. Kept as a function so the rule lives in one place rather
 * than as a dash hardcoded into each page.
 */
export function fmtRoi(): string {
  return DASH;
}

/**
 * Simulated score with the winner's runs first, which is how every score on
 * the site reads. `pickedHome` says which side the model picked.
 */
export function fmtSimScore(
  homeScore: number | null | undefined,
  awayScore: number | null | undefined,
  pickedHome: boolean,
): string {
  if (missing(homeScore) || missing(awayScore)) return DASH;
  const [winner, loser] = pickedHome
    ? [homeScore as number, awayScore as number]
    : [awayScore as number, homeScore as number];
  return `${winner}–${loser}`;
}

/** "2026-08-15T10:15:48+00:00" -> "2026-08-15 10:15:48". */
export function fmtTimestamp(value: string | null | undefined): string {
  if (missing(value) || value === '') return DASH;
  return (value as string)
    .replace('T', ' ')
    .replace('Z', '')
    .replace(/[+-]\d{2}:\d{2}$/, '')
    .trim();
}
