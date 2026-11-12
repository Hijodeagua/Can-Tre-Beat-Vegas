import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Data sources — Can Tre Beat Vegas',
  description:
    'Where the odds, the schedules and the results behind every model on the board come ' +
    'from, and what every stat on the site means.',
};

/**
 * Deliberately minimal — the source rows, a disclaimer and a KPI glossary,
 * in place of the old methodology writeup. The models explain themselves on
 * their own pages; this page answers where the numbers came from and what
 * each column header means.
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

/** Every stat the sport pages report, grouped by what it measures. Keep the
 * term spelled exactly as the column header that shows it. */
const GLOSSARY: { group: string; terms: { term: string; what: string }[] }[] = [
  {
    group: 'Ratings',
    terms: [
      { term: 'Elo', what: 'A team’s strength rating. 1500 is average; a 100-point gap makes the higher-rated side roughly a 64% favorite on a neutral field. Winning moves it up, losing moves it down, by more when the result was unexpected or the margin was big (capped).' },
      { term: 'Pre / Preseason avg', what: 'Rank or average Elo on the morning of the season’s first game — after the offseason regression toward the conference or league mean, before any result.' },
      { term: 'Δ pre', what: 'Elo gained or lost since that preseason rating.' },
      { term: 'Avg / Top-4 / Median / Bottom-4 / Mid-10 Avg Elo', what: 'A conference’s or league’s average rating, then its ceiling (four best teams), middle, floor (four worst) and, for soccer, the middle ten clubs.' },
      { term: 'In top N', what: 'How many of a conference’s or league’s teams sit inside the site’s top-N board.' },
      { term: 'Pt diff / Run diff', what: 'Points (or runs) scored minus allowed, season to date.' },
      { term: 'SP adj', what: 'MLB only: the Elo bump or penalty for the day’s probable starter versus his staff average, from his rolling game score.' },
    ],
  },
  {
    group: 'Game predictions',
    terms: [
      { term: 'Model pick / Win prob', what: 'The side the model favors and its chance of winning, from the two Elo ratings plus home edge (none at neutral sites, marked N).' },
      { term: 'H / D / A', what: 'Soccer: home win, draw and away win probabilities.' },
      { term: 'Elo line', what: 'The model’s own point spread from the home side, from the Elo gap alone. It never sees a sportsbook number — compare it against the market yourself.' },
      { term: 'Exp. score / Sim score / Predicted', what: 'The Elo-implied margin carved out of a matchup-specific total, winner first. An average outcome, not a literal final.' },
      { term: 'Exp. total', what: 'Expected combined points (or runs) from each team’s recent scoring for and against.' },
    ],
  },
  {
    group: 'Grading the model',
    terms: [
      { term: 'Accuracy / Record', what: 'How often the model’s pick won. Easy to inflate with lopsided games, so read it next to log-loss.' },
      { term: 'Log-loss', what: 'The main score for probability forecasts: it punishes confident misses hardest. Lower is better; 0.693 is a coin flip on every game.' },
      { term: 'Brier', what: 'Mean squared gap between the win probability and what happened (1 or 0). Lower is better; 0.25 is a coin flip.' },
      { term: 'Δ LL vs always-home (± SE)', what: 'Per-game log-loss minus that of a dumb “always pick the home team at the historical home rate” forecast on the same games, ± one standard error. Negative means the model is beating the baseline.' },
      { term: 'Cum. acc. / Cum. LL', what: 'Accuracy and log-loss accumulated across every graded day, weighted by games.' },
      { term: 'Window', what: 'The span a row is graded over — last 7 days, last 30, season to date.' },
    ],
  },
  {
    group: 'Season forecasts',
    terms: [
      { term: 'Proj / Proj W / Proj W–L', what: 'Expected final wins and losses from the rest-of-season Monte Carlo — the remaining schedule replayed thousands of times with ratings updating inside each simulation.' },
      { term: 'Bowl / Undefeated', what: 'College: chance of reaching six wins (bowl eligible) and of finishing the regular season unbeaten.' },
      { term: 'CCG / Conf title', what: 'College: chance of reaching the conference championship game, and of winning it.' },
      { term: 'Playoffs / Seed / #1 seed / Super Bowl', what: 'Pro: chance of making the postseason, the likeliest seed, of earning the top seed and of winning it all.' },
      { term: 'Pts / xPts / xPos', what: 'Soccer: points already banked, expected final points and expected final table position.' },
      { term: 'Title / UCL / UEL / Rel', what: 'Soccer: chance of finishing 1st, in the Champions League places (top 4), the Europa League places (5th–6th), or relegated (bottom 3).' },
      { term: 'Shield / Cup', what: 'MLS: chance of the best regular-season record (Supporters’ Shield) and of winning MLS Cup.' },
    ],
  },
  {
    group: 'Interconference (college)',
    terms: [
      { term: 'Exp W', what: 'Sum of the model’s pre-game win probabilities across a conference’s games against other conferences — the wins Elo said it should have.' },
      { term: 'W vs exp', what: 'Actual wins minus Exp W. Above zero: the conference outperformed its ratings (underrated, or lucky); below: it underperformed.' },
      { term: 'As fav / As dog', what: 'Record in the games where the model favored the conference’s team, and where it didn’t.' },
      { term: 'Avg margin / Exp margin / Margin vs exp', what: 'Average actual margin per game, the Elo-implied margin, and the gap between them in points per game.' },
      { term: 'Model acc.', what: 'How often the model’s pick was right in that conference’s interconference games.' },
    ],
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

      <h2
        id="kpis"
        className="pixel m-0 mt-12 text-[18px] leading-[1.4]"
        style={{ color: 'var(--th-ink)' }}
      >
        WHAT THE{' '}
        <span
          className="px-[6px] py-[2px]"
          style={{ background: 'var(--th-highlight)', color: 'var(--th-highlight-ink)' }}
        >
          KPIS
        </span>{' '}
        MEAN
      </h2>
      <p
        className="mt-4 max-w-[620px] text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)' }}
      >
        Every stat on the sport pages, named the way its column header names it.
      </p>

      <div className="mt-6 grid gap-6">
        {GLOSSARY.map((section) => (
          <section key={section.group}>
            <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
              {section.group}
            </h3>
            <dl
              className="mt-3 grid gap-0 overflow-hidden rounded-lg border"
              style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
            >
              {section.terms.map((t, i) => (
                <div
                  key={t.term}
                  className="flex flex-col gap-1 p-3 sm:flex-row sm:gap-4"
                  style={i ? { borderTop: '1px solid var(--th-border)' } : undefined}
                >
                  <dt
                    className="text-[13px] font-semibold sm:w-[220px] sm:shrink-0"
                    style={{ color: 'var(--th-ink)' }}
                  >
                    {t.term}
                  </dt>
                  <dd className="m-0 text-[14px] leading-normal" style={{ color: 'var(--th-muted)' }}>
                    {t.what}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </div>
  );
}
