import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getSoccerLatest, type SoccerResultsFreshness } from '@/app/lib/soccer';
import { accentVars, sportByKey } from '@/app/lib/sports';
import SoccerTabs from './SoccerTabs';

export const metadata: Metadata = {
  title: 'Club Soccer Elo — Can Tre Beat Vegas',
  description:
    'Cross-league club Elo rankings, squad economics and daily picks for Europe’s top 5 ' +
    'leagues plus their second divisions, and MLS.',
};

/**
 * "Results as of" line plus a warning per stale league. The pipeline
 * decides staleness (`results_freshness` in latest.json, thresholds in
 * soccer/clubs/daily/config.py); the page only prints it, so the site,
 * the email and the workflow gate always agree on which leagues are
 * frozen. Rendered above the tabs because a six-week-old table looks
 * exactly like a current one without it.
 */
function ResultsFreshness({ fresh }: { fresh: SoccerResultsFreshness | undefined }) {
  if (!fresh || !fresh.leagues) return null;
  const stale = (fresh.stale_leagues ?? []).map((k) => fresh.leagues[k]).filter(Boolean);
  const bits = Object.entries(fresh.leagues)
    .filter(([, e]) => e.newest_played)
    .map(([k, e]) => `${e.name} ${e.newest_played}${e.stale ? ' (stale)' : ''}`);
  return (
    <div className="mt-3 text-[12px]" style={{ color: 'var(--th-muted)' }}>
      <p className="m-0">
        <b>Results as of {fresh.as_of ?? '—'}</b>
        {bits.length ? ` · newest final per league: ${bits.join(' · ')}` : ''}
      </p>
      {stale.map((e) => (
        <p
          key={e.name}
          className="mt-2 rounded-md border px-3 py-2 text-[13px]"
          style={{
            borderColor: '#f0a000',
            background: 'color-mix(in srgb, #f0a000 14%, var(--th-card))',
            color: 'var(--th-text)',
          }}
        >
          ⚠ <b>{e.name} results are stale.</b> {e.reason} The {e.name} ratings, grades and
          forecasts on this page stop at {e.newest_played ?? 'that date'} and are not current.
        </p>
      ))}
    </div>
  );
}

export default function SoccerPage() {
  const soccer = getSoccerLatest();
  const sport = sportByKey('soccer');
  if (!sport) throw new Error('Soccer is missing from the sports config');

  return (
    <div style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h2 className="pixel m-0 text-[14px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} CLUB SOCCER ELO
        </h2>
        {soccer.run_date && (
          <span
            className="pixel text-[8px] tracking-[0.08em]"
            style={{ color: 'var(--sport-accent-ink)' }}
          >
            RUN {soccer.run_date}
          </span>
        )}
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        {sport.blurb} Ten country-pool leagues, one Elo scale: Premier League ↔ Championship,
        Bundesliga ↔ 2. Bundesliga, La Liga ↔ Segunda, Serie A ↔ Serie B, Ligue 1 ↔ Ligue 2 —
        each pair shares a pool with a promotion/relegation carry, and every pool exchanges
        rating points through Champions/Europa/Conference League matches. MLS runs alongside
        them — its own single-tier pool, ratings and squad economics — but a different
        confederation means it never plays the ten above, so its Elo stays on its own scale.
      </p>

      <ResultsFreshness fresh={soccer.results_freshness} />

      <SoccerTabs />

      {soccer.generated_at && (
        <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Data generated {fmtTimestamp(soccer.generated_at)}
          {soccer.run_date ? ` · run date ${soccer.run_date}` : ''}.
        </p>
      )}
    </div>
  );
}
