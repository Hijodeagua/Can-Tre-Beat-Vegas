/**
 * Presentation config for every forecasting model on the board.
 *
 * This array is the only place a sport is named. The nav, the track-record
 * ladder, the in-season slate section and the "up next season" cards are all
 * rendered by mapping over it, so bringing NFL online when the season starts
 * is a change to this file plus its route — not a new page and not a redesign.
 *
 * The *numbers* live in `public/data/summary.json` (written by
 * `data_jobs/build_summary.py`); what lives here is the fixed identity of each
 * sport: its glyph, its accent, and the copy that never comes from data.
 */

/** Where a sport's in-season slate is read from. */
export type SlateSource = 'mlb' | 'odds';

export interface SportConfig {
  /** Matches the `sport` key in summary.json and the slate.json sport key. */
  key: string;
  /** Display name, also the summary.json `sport` value. */
  name: string;
  /** Each sport owns one fixed glyph — the design system's whole icon set. */
  emoji: string;
  /** Accent colour; drives banner, table header, zebra rows and chips. */
  accent: string;
  /** Ink that stays legible on top of `accent`. */
  accentInk: string;
  /** Background for the glyph square in the ladder and on cards. */
  tint: string;
  /** Border colour for the dashed "up next season" card. */
  dashBorder: string;
  /** Dedicated route, or null while the sport has no page of its own. */
  href: string | null;
  /** Nav label; null when the sport is not yet in the nav. */
  navLabel: string | null;
  /** Which loader supplies this sport's slate when it is in season. */
  slateSource: SlateSource;
  /** One-line description of the model, shown above its slate. */
  blurb: string;
  /**
   * Copy for the off-season card. Templated rather than fixed because the
   * feed timestamp and season start are both real data.
   */
  offseasonLead: (opts: { seasonStarts: string | null; pulledAt: string | null; windowHours: number }) => string;
  /** Why the sport has no graded record yet. */
  offseasonNote: string;
}

/** "2026-09" -> "September 2026". Returns null for a missing/odd value. */
export function monthYear(value: string | null | undefined): string | null {
  if (!value) return null;
  const m = /^(\d{4})-(\d{2})$/.exec(value);
  if (!m) return null;
  const month = Number(m[2]);
  if (month < 1 || month > 12) return null;
  const names = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
  ];
  return `${names[month - 1]} ${m[1]}`;
}

/** "2026-08-15 06:17:47" -> "2026-08-15". */
function pullDate(pulledAt: string | null): string | null {
  return pulledAt ? pulledAt.slice(0, 10) : null;
}

/**
 * Off-season copy for a sport fed by the odds API.
 *
 * Shared by NFL and NBA because the only true difference between them is the
 * verb. The handoff gave NBA a "last pull ... at the end of the previous
 * season" line, which described a feed that had been idle since June — the
 * unified odds job now pulls both sports on the same schedule, so that
 * sentence would print a stale-feed claim next to today's date. The copy is
 * built from the timestamp instead of asserting anything about it.
 */
function oddsOffseasonLead(
  verb: string,
  { seasonStarts, pulledAt, windowHours }: { seasonStarts: string | null; pulledAt: string | null; windowHours: number },
): string {
  const when = monthYear(seasonStarts);
  const opener = when ? `${verb} ${when}.` : `${verb} date to be confirmed.`;
  const pull = pullDate(pulledAt);
  return pull
    ? `${opener} The odds feed is already live — last pull ${pull}, no games inside the ${windowHours}-hour window.`
    : `${opener} The odds feed has not been pulled yet.`;
}

export const SPORTS: SportConfig[] = [
  {
    key: 'soccer',
    name: 'Soccer',
    emoji: '⚽',
    accent: '#3ddc84',
    accentInk: '#06120b',
    tint: 'color-mix(in srgb, #3ddc84 22%, white)',
    dashBorder: 'color-mix(in srgb, #3ddc84 60%, white)',
    href: '/soccer',
    navLabel: '⚽ Soccer',
    slateSource: 'mlb', // unused: soccer isn't in the odds feed or the cross-sport ladder yet.
    blurb:
      'Country-pool club Elo across Europe’s top 5 leagues plus their second divisions, ' +
      'glued together by Champions/Europa/Conference League cross-play, with squad market ' +
      'value and transfer spend as extra outcome-model features.',
    offseasonLead: () => 'Not on the cross-sport ladder yet — soccer runs daily but is not graded there.',
    offseasonNote:
      'Club Elo + squad economics across 10 leagues. Not wired into the graded track-record ' +
      'ladder above; see the Soccer page for its own rankings and daily slate.',
  },
  {
    key: 'mlb',
    name: 'MLB',
    emoji: '⚾',
    accent: '#3ddc84',
    accentInk: '#06120b',
    tint: 'color-mix(in srgb, #3ddc84 22%, white)',
    dashBorder: 'color-mix(in srgb, #3ddc84 60%, white)',
    href: '/mlb',
    navLabel: '⚾ MLB',
    slateSource: 'mlb',
    blurb:
      'Betting-blind Elo (K=3, +24 home, margin-of-victory weighted). Backtest 2009–2025: ' +
      '56.7% straight up, log-loss 0.680 vs 0.691 for always-pick-home.',
    offseasonLead: ({ seasonStarts }) => {
      const when = monthYear(seasonStarts);
      return when ? `Opening day ${when}.` : 'Season start to be confirmed.';
    },
    offseasonNote:
      'Team-level Elo with a rest-of-season Monte Carlo. Grading resumes with the first played game.',
  },
  {
    key: 'nfl',
    name: 'NFL',
    emoji: '🏈',
    accent: '#ffd23f',
    accentInk: '#06120b',
    tint: 'color-mix(in srgb, #ffd23f 30%, white)',
    dashBorder: 'color-mix(in srgb, #ffd23f 70%, white)',
    href: null,
    navLabel: null,
    slateSource: 'odds',
    blurb:
      'LightGBM straight-up and against-the-spread models on schedule, results and team stats.',
    offseasonLead: (opts) => oddsOffseasonLead('Kickoff', opts),
    offseasonNote:
      'LightGBM straight-up and against-the-spread models are trained but not yet wired into the ' +
      'slate, so there is no graded record to show.',
  },
  {
    key: 'nba',
    name: 'NBA',
    emoji: '🏀',
    accent: '#ff2e88',
    accentInk: '#06120b',
    tint: 'color-mix(in srgb, #ff2e88 18%, white)',
    dashBorder: 'color-mix(in srgb, #ff2e88 45%, white)',
    href: null,
    navLabel: null,
    slateSource: 'odds',
    blurb:
      'Logistic-regression winner model plus a ridge score model on rolling team stats.',
    offseasonLead: (opts) => oddsOffseasonLead('Tip-off', opts),
    offseasonNote:
      'Logistic-regression winner model plus a ridge score model on rolling team stats. Picks ' +
      'post to the slate; grading starts with the first played game.',
  },
];

export function sportByKey(key: string): SportConfig | undefined {
  return SPORTS.find((s) => s.key === key.toLowerCase());
}

/** Looks up by the summary.json `sport` field ("MLB"), which is the display name. */
export function sportByName(name: string): SportConfig | undefined {
  return SPORTS.find((s) => s.name.toLowerCase() === name.toLowerCase());
}

/**
 * Sports with a page of their own. The nav stays flat while this is three or
 * fewer and collapses into a `Sports ▾` dropdown at four, so adding the fourth
 * sport never grows the header row.
 */
export const NAV_SPORTS = SPORTS.filter((s) => s.href && s.navLabel);
export const NAV_COLLAPSE_AT = 4;

/** Inline style that re-tints everything inside a section to one sport. */
export function accentVars(sport: Pick<SportConfig, 'accent' | 'accentInk'>): React.CSSProperties {
  return {
    '--sport-accent': sport.accent,
    '--sport-accent-ink': sport.accentInk,
  } as React.CSSProperties;
}
