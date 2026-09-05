import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Data sources — Can Tre Beat Vegas',
  description:
    'Where the odds, the schedules and the results behind every model on the board come from.',
};

/**
 * Deliberately minimal — three rows and a disclaimer, in place of the old
 * methodology writeup. The models explain themselves on their own pages; this
 * page only answers where the numbers came from.
 *
 * The route keeps its `/methodology` path so existing links do not break.
 */

const SOURCES = [
  {
    name: 'THE ODDS API',
    what:
      'Moneyline, spread and totals from DraftKings, FanDuel, BetMGM, BetRivers, Bovada ' +
      'and others — pulled twice daily, in American odds.',
  },
  {
    name: 'PRO FOOTBALL REF',
    what:
      'Shout out Pro Football Reference — the schedule, results and team stats behind ' +
      'the NFL model.',
  },
  {
    name: 'NFLVERSE',
    what:
      'The nflverse games file — every NFL game since 1999 with scores, rest days and the ' +
      'closing line — is the spine of the NFL Elo. Only the scores and rest days reach the ' +
      'ratings; the line is kept for grading the market, never for making a pick.',
  },
  {
    name: 'CFBFASTR-DATA',
    what:
      'ESPN-derived college football schedules and results, 2001–present, with per-season ' +
      'conference membership and FBS/FCS tags — the spine of the college Elo. ESPN’s public ' +
      'scoreboard fills in the trailing week’s finals.',
  },
  {
    name: 'MLB RESULTS',
    what:
      'Daily scores and season-to-date records feed the Elo ratings and the ' +
      'rest-of-season Monte Carlo.',
  },
];

export default function DataSourcesPage() {
  return (
    <div>
      <h2 className="pixel m-0 text-[18px] leading-[1.4]" style={{ color: 'var(--th-ink)' }}>
        DATA{' '}
        <span
          className="px-[6px] py-[2px]"
          style={{ background: 'var(--th-highlight)', color: 'var(--th-highlight-ink)' }}
        >
          SOURCES
        </span>
      </h2>

      <p
        className="mt-4 max-w-[620px] text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)' }}
      >
        A personal project by Tre. Every model here is experimental and none of it is
        betting advice.
      </p>

      <div className="mt-6 grid gap-3">
        {SOURCES.map((source) => (
          <div
            key={source.name}
            className="flex flex-wrap items-baseline gap-4 rounded-lg border p-4"
            style={{
              borderColor: 'var(--th-border)',
              borderLeft: '4px solid var(--accent-vegas)',
              background: 'var(--th-card)',
            }}
          >
            <span
              className="pixel min-w-[150px] text-[10px]"
              style={{ color: 'var(--th-ink)' }}
            >
              {source.name}
            </span>
            <span
              className="flex-1 text-[14px] leading-normal"
              style={{ color: 'var(--th-muted)' }}
            >
              {source.what}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-6 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Odds snapshots are committed to the repo twice daily, so each game&apos;s opener is
        the first snapshot it appeared in.
      </p>
    </div>
  );
}
