import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getCfbLatest } from '@/app/lib/cfb';
import { accentVars, sportByKey } from '@/app/lib/sports';
import CfbTabs from './CfbTabs';

export const metadata: Metadata = {
  title: 'College Football Elo — Can Tre Beat Vegas',
  description:
    'Betting-blind FBS Elo: a top 25, weekly picks with expected scores, ' +
    'rest-of-season win and conference-title forecasts, and a graded track record.',
};

export default function CfbPage() {
  const cfb = getCfbLatest();
  const sport = sportByKey('cfb');
  if (!sport) throw new Error('CFB is missing from the sports config');

  return (
    <div style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h2 className="pixel m-0 text-[14px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} COLLEGE FOOTBALL ELO
        </h2>
        {cfb.run_date && (
          <span
            className="pixel text-[8px] tracking-[0.08em]"
            style={{ color: 'var(--sport-accent-ink)' }}
          >
            {cfb.season}{cfb.week ? ` · WEEK ${cfb.week}` : ''} · RUN {cfb.run_date}
          </span>
        )}
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        {sport.blurb} Every FBS program since 2001 on one Elo scale: ratings regress each
        August toward the team&apos;s <em>new</em> conference&apos;s mean (so realignment is
        handled by construction), every FCS opponent is one pooled rating, and margin of
        victory is capped before it feeds the update. Parameters were fit on 2005–2023 and
        checked on 2024–25 held out; the daily job replays all ~20,000 games from scratch
        each morning, so there is no rating state that can drift.
      </p>

      <CfbTabs />

      {cfb.generated_at && (
        <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Data generated {fmtTimestamp(cfb.generated_at)}
          {cfb.run_date ? ` · run date ${cfb.run_date}` : ''}.
        </p>
      )}
    </div>
  );
}
