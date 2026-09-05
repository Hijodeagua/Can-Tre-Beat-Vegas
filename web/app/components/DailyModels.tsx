'use client';

import { useState } from 'react';
import Link from 'next/link';
import MlbSlateTable from '@/app/components/MlbSlateTable';
import ThemedTable from '@/app/components/ThemedTable';
import { fmtPct } from '@/app/lib/format';
import type { MlbSlateRow } from '@/app/lib/mlb';

/**
 * The home page's daily-model section: one banner, a sport pill per daily
 * model, and the selected model's slate + chips below it. Replaces the old
 * one-section-per-sport stack so the reader clicks ⚾/⚽/🎓/🏈 instead of
 * scrolling past whichever table they didn't come for.
 *
 * Deliberately dumb: every number and row is prepared by the server page
 * and passed in as serializable props — this component only holds the
 * which-tab-is-open state.
 */

export interface DailyChip {
  text: string;
  highlight: boolean;
}

export interface SoccerSlateLine {
  date: string;
  league: string;
  home: string;
  away: string;
  pH: number;
  pD: number;
  pA: number;
  pick: 'H' | 'D' | 'A';
  scoreH: number;
  scoreA: number;
}

/** One row of a two-way pick slate — college football and the NFL share it. */
export interface GameSlateLine {
  key: string;
  date: string;
  matchup: string;
  pick: string;
  pickProb: number;
  score: string;
}

export interface DailyTab {
  key: string;
  name: string;
  emoji: string;
  accent: string;
  accentInk: string;
  runDate: string;
  blurb: string;
  detailHref: string | null;
  detailLabel: string;
  chips: DailyChip[];
  mlbSlate?: MlbSlateRow[];
  soccerSlate?: SoccerSlateLine[];
  gameSlate?: GameSlateLine[];
  /** Fixtures beyond the rows shown (soccer and CFB slates can run long). */
  moreCount?: number;
  /** Where the rest of a truncated slate lives ("the CFB page"). */
  morePage?: string;
}

const SOCCER_COLUMNS = [
  { header: 'Date' },
  { header: 'League' },
  { header: 'Match', strong: true },
  { header: 'H / D / A' },
  { header: 'Pick', strong: true },
  { header: 'Sim score' },
];

const GAME_COLUMNS = [
  { header: 'Date' },
  { header: 'Matchup', strong: true },
  { header: 'Model pick', strong: true },
  { header: 'Win prob' },
  { header: 'Exp. score' },
];

function games(tab: DailyTab): number {
  return (
    tab.mlbSlate?.length
    ?? ((tab.soccerSlate?.length ?? tab.gameSlate?.length ?? 0) + (tab.moreCount ?? 0))
  );
}

export default function DailyModels({ tabs }: { tabs: DailyTab[] }) {
  const first = tabs.find((t) => games(t) > 0) ?? tabs[0];
  const [active, setActive] = useState(first.key);
  const tab = tabs.find((t) => t.key === active) ?? first;
  const n = games(tab);

  return (
    <section
      className="mt-8"
      style={{
        '--sport-accent': tab.accent,
        '--sport-accent-ink': tab.accentInk,
      } as React.CSSProperties}
    >
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-t-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="pixel m-0 mr-1 text-[12px]" style={{ color: 'var(--sport-accent-ink)' }}>
            DAILY MODELS
          </h3>
          {tabs.map((t) => {
            const on = t.key === tab.key;
            return (
              <button
                key={t.key}
                onClick={() => setActive(t.key)}
                aria-pressed={on}
                className={`rounded-full px-3 py-1 text-[13px] ${on ? 'font-semibold' : ''}`}
                style={
                  on
                    ? { background: 'var(--sport-accent-ink)', color: 'var(--sport-accent)' }
                    : {
                        color: 'var(--sport-accent-ink)',
                        border: '1px solid var(--sport-accent-ink)',
                        opacity: 0.75,
                      }
                }
              >
                {t.emoji} {t.name}
              </button>
            );
          })}
        </div>
        <span
          className="pixel text-[8px] tracking-[0.08em]"
          style={{ color: 'var(--sport-accent-ink)' }}
        >
          {n} GAME{n === 1 ? '' : 'S'}
          {tab.runDate ? ` · RUN ${tab.runDate}` : ''}
        </span>
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        {tab.blurb}
      </p>

      <div className="mt-3">
        {n === 0 ? (
          <div
            className="rounded-lg border border-dashed p-8 text-center text-[14px]"
            style={{
              borderColor: 'var(--th-border)',
              background: 'var(--th-card)',
              color: 'var(--th-muted)',
            }}
          >
            No {tab.name} games on today&apos;s slate.
          </div>
        ) : tab.mlbSlate ? (
          <MlbSlateTable slate={tab.mlbSlate} />
        ) : tab.gameSlate ? (
          <ThemedTable
            columns={GAME_COLUMNS}
            rows={tab.gameSlate.map((r) => ({
              key: r.key,
              cells: [r.date, r.matchup, r.pick, fmtPct(r.pickProb), r.score],
            }))}
            note={
              (tab.moreCount ?? 0) > 0
                ? `${tab.moreCount} more games in the window — the full slate lives on ${tab.morePage ?? 'the sport page'}.`
                : undefined
            }
          />
        ) : (
          <ThemedTable
            columns={SOCCER_COLUMNS}
            rows={(tab.soccerSlate ?? []).map((r) => ({
              key: `${r.home}-${r.away}-${r.date}`,
              cells: [
                r.date,
                r.league,
                `${r.home} v ${r.away}`,
                `${fmtPct(r.pH)} / ${fmtPct(r.pD)} / ${fmtPct(r.pA)}`,
                r.pick,
                `${r.scoreH}–${r.scoreA}`,
              ],
            }))}
            note={
              (tab.moreCount ?? 0) > 0
                ? `${tab.moreCount} more fixtures in the window — the full slate lives on the soccer page.`
                : undefined
            }
          />
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {tab.chips.map((chip) => (
          <span
            key={chip.text}
            className={`rounded-full px-3 py-1 text-[12px] ${chip.highlight ? 'font-semibold' : ''}`}
            style={
              chip.highlight
                ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
                : { background: 'var(--th-chip)', color: 'var(--th-muted)' }
            }
          >
            {chip.text}
          </span>
        ))}
        {tab.detailHref && (
          <Link
            href={tab.detailHref}
            className="text-[14px] underline-offset-2 hover:underline"
            style={{ color: 'var(--th-muted)' }}
          >
            {tab.detailLabel}
          </Link>
        )}
      </div>
    </section>
  );
}
