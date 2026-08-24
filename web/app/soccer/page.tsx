import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getSoccerLatest } from '@/app/lib/soccer';
import { accentVars, sportByKey } from '@/app/lib/sports';
import SoccerTabs from './SoccerTabs';

export const metadata: Metadata = {
  title: 'Club Soccer Elo — Can Tre Beat Vegas',
  description:
    'Cross-league club Elo rankings, squad economics and daily picks for Europe’s top 5 ' +
    'leagues plus their second divisions, and MLS.',
};

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
