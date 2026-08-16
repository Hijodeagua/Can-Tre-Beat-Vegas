import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getMlbLatest } from '@/app/lib/mlb';
import { accentVars, sportByKey } from '@/app/lib/sports';
import MlbTabs from './MlbTabs';

export const metadata: Metadata = {
  title: 'MLB Daily Model — Can Tre Beat Vegas',
  description:
    'Betting-blind Elo: daily picks with simulated scores, rest-of-season Monte Carlo futures, and a graded track record.',
};

export default function MlbPage() {
  const mlb = getMlbLatest();
  const sport = sportByKey('mlb');
  if (!sport) throw new Error('MLB is missing from the sports config');

  return (
    // One accent set here tints the banner, the active tab and every table on
    // all four tabs.
    <div style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h2 className="pixel m-0 text-[14px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} MLB DAILY MODEL
        </h2>
        {mlb.date && (
          <span
            className="pixel text-[8px] tracking-[0.08em]"
            style={{ color: 'var(--sport-accent-ink)' }}
          >
            RUN {mlb.date}
          </span>
        )}
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        Betting-blind Elo (K=3, +24 home, margin-of-victory weighted). Every morning:
        rest-of-season futures, today&apos;s slate with simulated scores, and
        yesterday&apos;s graded report card. Backtest 2009–2025: 56.7% straight up,
        log-loss 0.680 vs 0.691 for always-pick-home.
      </p>

      <MlbTabs />

      {mlb.generated_at && (
        <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Data generated {fmtTimestamp(mlb.generated_at)}
          {mlb.date ? ` · run date ${mlb.date}` : ''}.
        </p>
      )}
    </div>
  );
}
