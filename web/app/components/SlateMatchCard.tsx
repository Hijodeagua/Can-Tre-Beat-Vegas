'use client';

/**
 * One fixture on the slate, as a card rather than a table row.
 *
 * A row can carry a prediction; it cannot carry the case for one. The
 * card leads with what the model concluded — both sides' Elo, the
 * W/D/L split, the pick and the simulated score — and opens to show
 * every per-side number the prediction was built from, grouped the way
 * the pipeline groups them.
 *
 * Two things the layout is deliberate about:
 *
 * - **Both halves of every differential.** The model is trained on
 *   home-minus-away and nothing else, so "+0.4 xG created" is all it
 *   ever sees. That is a different match when it is 1.9 against 1.5 than
 *   when it is 0.7 against 0.3, and only the two columns say which.
 * - **A missing reading is a dash, not a zero.** A club short of the
 *   rolling window's warm-up, or whose feed has gone stale, has no
 *   number — printing 0 would claim it is average at something nobody
 *   has measured.
 *
 * The edge column names the favoured side in words rather than by
 * colour, and says nothing at all for a metric whose direction is not a
 * judgement (rest days, squad value).
 */

import { Fragment, useState } from 'react';
import { DASH, fmtPct } from '@/app/lib/format';
import type { SoccerSlateMetric, SoccerSlateRow } from '@/app/lib/soccer';

/** Decimals per metric. Elo and the counting metrics are whole numbers;
 * rates that live between 0 and 1 need three to be readable at all. */
function decimalsFor(key: string): number {
  if (key === 'elo') return 0;
  if (key === 'rest' || key === 'congestion14' || key === 'uefa7') return 0;
  if (key === 'xg_per_shot_ewm' || key === 'xg_per_shot_r10') return 3;
  if (key.startsWith('deep_share')) return 3;
  return 2;
}

function fmtStat(v: number | undefined, key: string): string {
  if (v === undefined || v === null || Number.isNaN(v)) return DASH;
  return v.toFixed(decimalsFor(key));
}

/** Which side the metric favours, or '' when the metric makes no claim
 * or either side is missing a reading. */
function edge(
  metric: SoccerSlateMetric, home: number | undefined, away: number | undefined,
): string {
  if (metric.higherIsBetter === null) return '';
  if (home === undefined || away === undefined) return '';
  if (home === away) return 'level';
  const homeAhead = metric.higherIsBetter ? home > away : home < away;
  return homeAhead ? 'home' : 'away';
}

function ProbabilityBar({ row }: { row: SoccerSlateRow }) {
  const segments = [
    { key: 'H', p: row.p_H, color: 'var(--viz-1)' },
    { key: 'D', p: row.p_D, color: 'var(--viz-3)' },
    { key: 'A', p: row.p_A, color: 'var(--viz-2)' },
  ];
  return (
    <div>
      <div
        className="flex h-2 w-full overflow-hidden rounded"
        role="img"
        aria-label={`Home ${fmtPct(row.p_H)}, draw ${fmtPct(row.p_D)}, away ${fmtPct(row.p_A)}`}
      >
        {segments.map((s) => (
          <div key={s.key} style={{ width: `${s.p * 100}%`, background: s.color }} />
        ))}
      </div>
      <div
        className="mt-1 flex justify-between text-[11px] tabular-nums"
        style={{ color: 'var(--th-muted)' }}
      >
        <span>H {fmtPct(row.p_H)}</span>
        <span>D {fmtPct(row.p_D)}</span>
        <span>A {fmtPct(row.p_A)}</span>
      </div>
    </div>
  );
}

export default function SlateMatchCard({
  row, metrics,
}: {
  row: SoccerSlateRow;
  metrics: SoccerSlateMetric[];
}) {
  const [open, setOpen] = useState(false);
  const sides = row.sides;
  const pickLabel =
    row.pick === 'H' ? row.home_team : row.pick === 'A' ? row.away_team : 'Draw';

  // Only groups with at least one reading on either side; a feed that
  // has not landed leaves its whole group out rather than printing a
  // block of dashes.
  const groups: [string, SoccerSlateMetric[]][] = [];
  for (const m of metrics) {
    if (!sides) break;
    const has = sides.home[m.key] !== undefined || sides.away[m.key] !== undefined;
    if (!has) continue;
    const last = groups[groups.length - 1];
    if (last && last[0] === m.group) last[1].push(m);
    else groups.push([m.group, [m]]);
  }

  return (
    <div
      className="rounded-lg border p-3"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-[11px] tabular-nums" style={{ color: 'var(--th-faint)' }}>
          {row.date}
        </span>
        <span className="text-[14px] font-semibold" style={{ color: 'var(--th-ink)' }}>
          {row.home_team}
        </span>
        <span className="text-[12px]" style={{ color: 'var(--th-faint)' }}>v</span>
        <span className="text-[14px] font-semibold" style={{ color: 'var(--th-ink)' }}>
          {row.away_team}
        </span>
      </div>

      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[12px] tabular-nums"
           style={{ color: 'var(--th-muted)' }}>
        <span>Elo {row.elo_home_pre.toFixed(0)} v {row.elo_away_pre.toFixed(0)}</span>
        <span>Pick <b style={{ color: 'var(--th-ink)' }}>{pickLabel}</b></span>
        <span>Sim score {row.score_home}–{row.score_away}</span>
      </div>

      <div className="mt-2">
        <ProbabilityBar row={row} />
      </div>

      {groups.length > 0 && (
        <>
          <button
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="mt-3 rounded-full px-3 py-1 text-[12px]"
            style={{ color: 'var(--th-muted)', border: '1px solid var(--th-border)' }}
          >
            {open ? 'Hide the numbers behind it' : 'Show the numbers behind it'}
          </button>

          {open && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full border-collapse text-[12px]">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-left font-semibold"
                        style={{ color: 'var(--th-muted)' }}>
                      Metric
                    </th>
                    <th className="px-2 py-1 text-right font-semibold"
                        style={{ color: 'var(--th-muted)' }}>
                      {row.home_team}
                    </th>
                    <th className="px-2 py-1 text-right font-semibold"
                        style={{ color: 'var(--th-muted)' }}>
                      {row.away_team}
                    </th>
                    <th className="px-2 py-1 text-left font-semibold"
                        style={{ color: 'var(--th-muted)' }}>
                      Edge
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {groups.map(([group, items]) => (
                    <Fragment key={group}>
                      <tr>
                        <td
                          colSpan={4}
                          className="px-2 pb-1 pt-3 text-[10px] uppercase tracking-wide"
                          style={{ color: 'var(--th-faint)' }}
                        >
                          {group}
                        </td>
                      </tr>
                      {items.map((m) => {
                        const h = sides!.home[m.key];
                        const a = sides!.away[m.key];
                        const e = edge(m, h, a);
                        return (
                          <tr key={m.key} style={{ borderTop: '1px solid var(--th-border)' }}>
                            <td className="px-2 py-1" style={{ color: 'var(--th-ink)' }}>
                              {m.label}
                            </td>
                            <td className="px-2 py-1 text-right tabular-nums"
                                style={{ color: 'var(--th-ink)' }}>
                              {fmtStat(h, m.key)}
                            </td>
                            <td className="px-2 py-1 text-right tabular-nums"
                                style={{ color: 'var(--th-ink)' }}>
                              {fmtStat(a, m.key)}
                            </td>
                            <td className="px-2 py-1" style={{ color: 'var(--th-muted)' }}>
                              {e === 'home' ? row.home_team
                                : e === 'away' ? row.away_team
                                : e === 'level' ? 'level' : ''}
                            </td>
                          </tr>
                        );
                      })}
                    </Fragment>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
                The model is trained on the difference between these two columns, never on
                a column by itself. A dash is a reading that does not exist — a club short
                of the rolling window&apos;s warm-up, or a feed that has gone stale — not a
                zero.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
