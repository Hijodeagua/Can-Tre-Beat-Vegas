import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getNflLatest } from '@/app/lib/nfl';
import { accentVars, sportByKey } from '@/app/lib/sports';
import NflTabs from './NflTabs';

export const metadata: Metadata = {
  title: 'NFL Elo — Can Tre Beat Vegas',
  description:
    'Betting-blind NFL Elo: power ratings, this week\'s picks with the model\'s own line ' +
    'and expected scores, division / playoff / Super Bowl odds, and a graded track record.',
};

export default function NflPage() {
  const nfl = getNflLatest();
  const sport = sportByKey('nfl');
  if (!sport) throw new Error('NFL is missing from the sports config');
  const p = nfl.params;

  return (
    <div style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h2 className="pixel m-0 text-[14px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} NFL ELO
        </h2>
        {nfl.run_date && (
          <span
            className="pixel text-[8px] tracking-[0.08em]"
            style={{ color: 'var(--sport-accent-ink)' }}
          >
            {nfl.season}{nfl.week_label ? ` · ${nfl.week_label.toUpperCase()}` : ''} · RUN {nfl.run_date}
          </span>
        )}
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        {sport.blurb} Every game since 1999 on one Elo scale, franchise-continuous across the
        St. Louis, San Diego and Oakland moves: K={p.k}, +{p.home_advantage} Elo at home, +{p.rest_bonus}{' '}
        for a side off its bye, margin of victory capped at {p.margin_cap} points before it feeds
        the update, and {Math.round(p.season_regression * 100)}% regression to 1500 every off-season.
        Parameters were fit on 2005–2023 and checked on 2024–25 held out; the daily job replays
        all ~7,000 games from scratch each morning, so there is no rating state that can drift.
        Unlike the LightGBM picks model in the weekly report, this one never sees a market number.
      </p>

      <NflTabs />

      {nfl.generated_at && (
        <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Data generated {fmtTimestamp(nfl.generated_at)}
          {nfl.run_date ? ` · run date ${nfl.run_date}` : ''}.
        </p>
      )}
    </div>
  );
}
