import type { Metadata } from 'next';
import Link from 'next/link';
import { accentVars, sportByKey } from '@/app/lib/sports';
import CompareClient from './CompareClient';

export const metadata: Metadata = {
  title: 'Player Comparison — Can Tre Beat Vegas',
  description:
    'Side-by-side player comparison for Europe’s top 5 leagues: per-90 numbers with ' +
    'percentile ranks against positional peers, and statistically similar players.',
};

export default function ComparePage() {
  const sport = sportByKey('soccer');
  if (!sport) throw new Error('Soccer is missing from the sports config');

  return (
    <div style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h2 className="pixel m-0 text-[14px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} PLAYER COMPARISON
        </h2>
        <Link
          href="/soccer"
          className="pixel text-[8px] tracking-[0.08em] no-underline"
          style={{ color: 'var(--sport-accent-ink)' }}
        >
          CLUB ELO PAGE
        </Link>
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        Put up to five player-seasons side by side. Every number comes with a percentile bar:
        where that player ranks among players at the same position, in the same season, across
        the Premier League, La Liga, Bundesliga, Serie A and Ligue 1.
      </p>

      <CompareClient />
    </div>
  );
}
