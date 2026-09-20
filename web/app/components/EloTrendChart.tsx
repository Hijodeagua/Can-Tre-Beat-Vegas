'use client';

import { useId, useMemo, useState } from 'react';

/**
 * Elo trajectories for one league, drawn as inline SVG — the site has no
 * chart dependency and doesn't need one for a line chart.
 *
 * Two halves, switched by the control above the plot:
 *
 * - "Season to date" is what has happened. The daily pipeline exports each
 *   team's pre-match rating at every match date plus today's live rating
 *   (`elo_history` in latest.json), so the chart moves every day the job
 *   runs, not just on matchdays.
 * - "Projection" adds where the rest-of-season Monte Carlo expects each
 *   rating to go (`elo_projection`), continuing each line from today's
 *   value. It is the same simulation behind the forecast table, so the two
 *   can't disagree. A page with no projection in its data (nothing left to
 *   simulate) just doesn't get the control.
 *
 *   What it draws is simulated seasons, never an average of them. The bold
 *   line is that side's *median run* — the one simulated season whose final
 *   rating came out in the middle of its distribution, so it wanders the
 *   way a season wanders; picked per side, so two bold lines are not the
 *   same simulated season. Behind it are three whole seasons that *are*
 *   shared across sides, so their crossings are a coherent league.
 *
 *   No mean appears anywhere: an Elo update is K × (actual − expected) and
 *   the sim draws results at its own expected rate, so a side's expected
 *   rating change is about zero and averaging thousands of seasons hands
 *   back today's rating for everybody. A chart led by that average reports
 *   that the table stops moving, which is the one thing the simulation
 *   rules out.
 *
 * Twenty clubs can't wear twenty distinguishable hues, so the league's
 * best `highlight` and worst `highlight` by current Elo — the two ends
 * anyone reads a table for — carry the categorical palette and an
 * end-of-line label, and the rest of the league recedes to thin muted
 * lines. The two groups share those five hues rather than inventing five
 * more: the group is carried by the marker (filled at the top, ringed at
 * the bottom) and by the rank in every label, and the groups sit at
 * opposite ends of the y axis. Identity is never colour-alone — every
 * coloured line is direct-labelled, with a leader line wherever the label
 * had to be nudged clear of its neighbour, and the full ratings table
 * sits on the same page.
 */

/** The x value of a point. An ISO date for the European leagues, whose
 * fixtures are published with dates; a match count for MLS, whose
 * remaining fixtures have none (see `xScale`). */
export type EloX = string | number;

export interface EloSeries {
  team: string;
  points: [EloX, number][]; // [ISO date or match count, elo]
}

/** One team's projected Elo: [x, median run, 10th pct, 90th pct].
 * The first point is today's actual rating, so the line joins the
 * history. The middle value is a real simulated season, not an average —
 * see the sim's `_projection_block`. */
export interface EloProjectionSeries {
  team: string;
  points: [EloX, number, number, number][];
  /** A few whole simulated seasons, each one rating per `points` entry.
   * Path `i` of every team comes from the same simulated season. */
  samples?: number[][];
}

// Categorical slots 1-5 from the validated reference palette, themed in
// globals.css; order is the CVD-safety mechanism — assign in order, never
// cycle.
const SERIES_COLORS = ['var(--viz-1)', 'var(--viz-2)', 'var(--viz-3)',
  'var(--viz-4)', 'var(--viz-5)'];
const MUTED_LINE = 'var(--th-muted)';
const MUTED_OPACITY = 0.3;

const W = 760;
const H = 380;
const M = { top: 16, right: 168, bottom: 28, left: 46 };
const LABEL_GAP = 13;

type Mode = 'actual' | 'projected';

function shortName(team: string): string {
  return team
    .replace(/^(AFC|FC|AC|AS|SS|SSC|US|CF|RC|RCD|SL|VfB|VfL|TSG|SV|1\.\s*FC)\s+/i, '')
    .replace(/\s+(FC|AFC|CF|SC|AC|BC|Calcio|Balompié)$/i, '');
}

interface Hover {
  x: number;
  y: number;
  team: string;
  date: EloX;
  elo: number;
  /** Set on a projected point: the 10th–90th-percentile band there. */
  band?: [number, number];
}

interface Pt { d: EloX; e: number; px: number; py: number; band?: [number, number] }

/** Default x scale: ISO dates to epoch milliseconds. A numeric x (MLS's
 * match count) is already its own scale and passes straight through. */
function ms(x: EloX): number {
  return typeof x === 'number' ? x : new Date(`${x}T00:00:00Z`).getTime();
}

export default function EloTrendChart({
  series,
  projection,
  highlight = 5,
  xLabel = String,
  nowLabel = 'today',
}: {
  series: EloSeries[];
  projection?: EloProjectionSeries[];
  highlight?: number;
  /** How an x value reads on the axis and in the tooltip. Dates print
   * themselves; MLS passes a formatter that turns 28 into "28 GP". */
  xLabel?: (x: EloX) => string;
  /** What the divider between record and simulation is called. "today"
   * for a date axis; MLS says "played" because its divider is a match
   * count, not a moment. */
  nowLabel?: string;
}) {
  const [hover, setHover] = useState<Hover | null>(null);
  const [mode, setMode] = useState<Mode>('actual');
  const clipId = `elo-plot-${useId()}`;

  const projected = useMemo(() => {
    const out = new Map<string, EloProjectionSeries>();
    for (const p of projection ?? []) {
      if (p.points.length > 1) out.set(p.team, p);
    }
    return out;
  }, [projection]);

  const showProjection = mode === 'projected' && projected.size > 0;

  const model = useMemo(() => {
    const withFinal = series
      .filter((s) => s.points.length > 0)
      .map((s) => ({ ...s, final: s.points[s.points.length - 1][1] }));
    withFinal.sort((a, b) => b.final - a.final);
    if (withFinal.length === 0) return null;

    // The two ends of the table. Capped at half the league so a short
    // series list can't put one team in both groups.
    const band = Math.max(0, Math.min(highlight, Math.floor(withFinal.length / 2)));

    const ranked = withFinal.map((s, rank) => {
      const inTop = rank < band;
      const inBottom = rank >= withFinal.length - band;
      const slot = inTop ? rank : rank - (withFinal.length - band);
      const proj = showProjection ? projected.get(s.team) : undefined;
      return {
        team: s.team,
        rank,
        group: inTop ? 'top' : inBottom ? 'bottom' : null,
        color: inTop || inBottom ? SERIES_COLORS[slot] : null,
        points: s.points,
        proj: proj?.points,
        samples: proj?.samples ?? [],
      };
    });

    // Scales. The projection extends the x domain and can widen the y
    // domain, so both are computed over whatever the current mode draws.
    const ts: number[] = [];
    const elos: number[] = [];
    for (const s of ranked) {
      for (const [d, e] of s.points) {
        ts.push(ms(d));
        elos.push(e);
      }
      for (const [d, mean] of s.proj ?? []) {
        ts.push(ms(d));
        elos.push(mean);
      }
      // Sample seasons are drawn, so they set the scale; the percentile
      // band is not (it appears on hover, clipped to the plot).
      if (s.color) {
        for (const path of s.samples) elos.push(...path.slice(0, (s.proj ?? []).length));
      }
    }
    const t0 = Math.min(...ts);
    const t1 = Math.max(...ts);
    const pad = Math.max(8, (Math.max(...elos) - Math.min(...elos)) * 0.06);
    const e0 = Math.min(...elos) - pad;
    const e1 = Math.max(...elos) + pad;

    const x = (d: EloX) =>
      t1 === t0
        ? M.left + (W - M.left - M.right) / 2
        : M.left + ((ms(d) - t0) / (t1 - t0)) * (W - M.left - M.right);
    const y = (e: number) => M.top + ((e1 - e) / (e1 - e0)) * (H - M.top - M.bottom);

    const drawn = ranked.map((s) => ({
      ...s,
      pts: s.points.map(([d, e]) => ({ d, e, px: x(d), py: y(e) })) as Pt[],
      projPts: (s.proj ?? []).map(([d, mean, lo, hi]) => ({
        d, e: mean, px: x(d), py: y(mean), band: [lo, hi] as [number, number],
      })) as Pt[],
      // One polyline per simulated season, on the projection's own dates.
      // Truncated to the dates it has, so a short or malformed path draws
      // what it covers instead of throwing.
      samplePts: s.samples.map((path) =>
        path.slice(0, (s.proj ?? []).length)
          .map((e, i) => ({ px: x(s.proj![i][0]), py: y(e) })),
      ),
    }));

    // End labels for the highlighted teams, anchored on whichever line
    // ends furthest right and nudged apart without leaving the plot —
    // a leader line is drawn wherever the nudge moved one.
    const labels = drawn
      .filter((s) => s.color)
      .map((s) => {
        const end = (s.projPts.length ? s.projPts : s.pts)[
          (s.projPts.length ? s.projPts : s.pts).length - 1];
        return {
          team: s.team,
          label: `${s.rank + 1} ${shortName(s.team)}`,
          color: s.color!,
          group: s.group!,
          x: end.px,
          endY: end.py,
          y: end.py,
        };
      })
      .sort((a, b) => a.endY - b.endY);
    for (let i = 1; i < labels.length; i++) {
      labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP);
    }
    const floor = H - M.bottom - 4;
    if (labels.length && labels[labels.length - 1].y > floor) {
      labels[labels.length - 1].y = floor;
      for (let i = labels.length - 2; i >= 0; i--) {
        labels[i].y = Math.min(labels[i].y, labels[i + 1].y - LABEL_GAP);
      }
    }

    // Recessive y grid: 4 round-numbered lines.
    const step = Math.max(10, Math.round((e1 - e0) / 4 / 10) * 10);
    const gridStart = Math.ceil(e0 / step) * step;
    const grid: number[] = [];
    for (let v = gridStart; v < e1; v += step) grid.push(v);

    const byX = (a: EloX, b: EloX) => ms(a) - ms(b);
    const actualDates = series.flatMap((s) => s.points.map(([d]) => d)).sort(byX);
    const today = actualDates[actualDates.length - 1];
    const projDates = drawn.flatMap((s) => s.projPts.map((p) => p.d)).sort(byX);
    return {
      drawn,
      labels,
      grid,
      x,
      y,
      firstDate: actualDates[0],
      today,
      dense: drawn.length > 40,
      lastDate: projDates.length ? projDates[projDates.length - 1] : today,
    };
  }, [series, projected, showProjection, highlight]);

  if (!model) return null;

  const onMove = (evt: React.MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const mx = ((evt.clientX - rect.left) / rect.width) * W;
    const my = ((evt.clientY - rect.top) / rect.height) * H;
    let best: Hover | null = null;
    let bestDist = 24 * 24; // generous hit target, larger than the marks
    for (const s of model.drawn) {
      for (const p of [...s.pts, ...s.projPts]) {
        const dist = (p.px - mx) ** 2 + (p.py - my) ** 2;
        if (dist < bestDist) {
          bestDist = dist;
          best = { x: p.px, y: p.py, team: s.team, date: p.d, elo: p.e, band: p.band };
        }
      }
    }
    setHover(best);
  };

  const pack = model.drawn.filter((s) => !s.color);
  const highlighted = model.drawn.filter((s) => s.color);
  const hovered = hover ? model.drawn.find((s) => s.team === hover.team) : undefined;
  const line = (pts: Pt[]) => pts.map((p) => `${p.px},${p.py}`).join(' ');

  return (
    <div className="relative">
      {projected.size > 0 && (
        <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Chart view">
          {([['actual', 'Season to date'], ['projected', 'Projection']] as const).map(
            ([key, label]) => (
              <button
                key={key}
                onClick={() => { setMode(key); setHover(null); }}
                aria-pressed={mode === key}
                className={`rounded-full px-3 py-1 text-[13px] ${
                  mode === key ? 'font-semibold' : 'hover:bg-slate-100'
                }`}
                style={
                  mode === key
                    ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
                    : { color: 'var(--th-muted)', border: '1px solid var(--th-border)' }
                }
              >
                {label}
              </button>
            ),
          )}
        </div>
      )}

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={
          showProjection
            ? 'Elo ratings this season with the projected rest of season'
            : 'Elo ratings over the current season'
        }
        className="w-full"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={M.left}
              y={M.top}
              width={W - M.left - M.right}
              height={H - M.top - M.bottom}
            />
          </clipPath>
        </defs>

        {model.grid.map((v) => (
          <g key={v}>
            <line
              x1={M.left}
              x2={W - M.right}
              y1={model.y(v)}
              y2={model.y(v)}
              stroke="var(--th-row)"
              strokeWidth={1}
            />
            <text
              x={M.left - 6}
              y={model.y(v) + 3.5}
              textAnchor="end"
              fontSize={10}
              fill="var(--th-faint)"
            >
              {v}
            </text>
          </g>
        ))}

        {(!showProjection || model.x(model.today) - M.left > 110) && (
          <text x={M.left} y={H - 8} fontSize={10} fill="var(--th-faint)">
            {xLabel(model.firstDate)}
          </text>
        )}
        {showProjection && (
          <>
            {/* Where the record stops and the simulation starts. */}
            <line
              x1={model.x(model.today)}
              x2={model.x(model.today)}
              y1={M.top}
              y2={H - M.bottom}
              stroke="var(--th-border)"
              strokeWidth={1}
            />
            <text
              x={model.x(model.today)}
              y={H - 8}
              textAnchor="middle"
              fontSize={10}
              fill="var(--th-faint)"
            >
              {xLabel(model.today)} · {nowLabel}
            </text>
          </>
        )}
        <text x={W - M.right} y={H - 8} textAnchor="end" fontSize={10} fill="var(--th-faint)">
          {xLabel(model.lastDate)}
        </text>

        {/* Muted pack first, then the highlighted bands, then their lines. */}
        {pack.map((s) => (
          <g key={s.team}>
            <polyline
              points={line(s.pts)}
              fill="none"
              stroke={MUTED_LINE}
              strokeOpacity={model.dense ? MUTED_OPACITY / 2 : MUTED_OPACITY}
              strokeWidth={1.25}
            />
          </g>
        ))}

        {/* The hovered line's percentile band, clipped to the plot so it
            can't widen the y scale or spill into the label column. The
            pack's own projections aren't drawn — with three sample seasons
            per labelled side on the plot, a second screen of grey dotted
            lines is noise — so the hovered one is drawn here with its band
            rather than leaving it floating. */}
        {hovered && hovered.projPts.length > 0 && !hovered.color && (
          <polyline
            points={line(hovered.projPts)}
            fill="none"
            stroke={MUTED_LINE}
            strokeOpacity={MUTED_OPACITY}
            strokeWidth={1.25}
            strokeDasharray="2 3"
          />
        )}
        {hovered && hovered.projPts.length > 0 && (
          <polygon
            clipPath={`url(#${clipId})`}
            points={[
              ...hovered.projPts.map((p) => `${p.px},${model.y(p.band![1])}`),
              ...[...hovered.projPts].reverse().map((p) => `${p.px},${model.y(p.band![0])}`),
            ].join(' ')}
            fill={hovered.color ?? MUTED_LINE}
            fillOpacity={0.18}
          />
        )}

        {/* Individual simulated seasons, under the mean: these are what
            show the table actually moving. */}
        {highlighted.map((s) => (
          <g key={`samples-${s.team}`} clipPath={`url(#${clipId})`}>
            {s.samplePts.map((path, i) => (
              <polyline
                key={i}
                points={path.map((p) => `${p.px},${p.py}`).join(' ')}
                fill="none"
                stroke={s.color!}
                strokeOpacity={0.4}
                strokeWidth={1}
              />
            ))}
          </g>
        ))}

        {highlighted.map((s) => (
          <g key={s.team}>
            <polyline
              points={line(s.pts)}
              fill="none"
              stroke={s.color!}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.projPts.length > 0 && (
              <polyline
                points={line(s.projPts)}
                fill="none"
                stroke={s.color!}
                strokeWidth={2}
                strokeOpacity={0.8}
                strokeDasharray="5 4"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            )}
            {s.pts.map((p) => (
              <circle
                key={p.d}
                cx={p.px}
                cy={p.py}
                r={3.5}
                fill={s.group === 'top' ? s.color! : 'var(--th-card)'}
                stroke={s.group === 'top' ? 'var(--th-card)' : s.color!}
                strokeWidth={s.group === 'top' ? 1.5 : 2}
              />
            ))}
          </g>
        ))}

        {model.labels.map((l) => (
          <g key={l.team}>
            {Math.abs(l.y - l.endY) > 2 && (
              <polyline
                points={`${l.x},${l.endY} ${l.x + 4},${l.y} ${l.x + 6},${l.y}`}
                fill="none"
                stroke="var(--th-border)"
                strokeWidth={1}
              />
            )}
            <circle
              cx={l.x + 11}
              cy={l.y}
              r={3.5}
              fill={l.group === 'top' ? l.color : 'var(--th-card)'}
              stroke={l.color}
              strokeWidth={l.group === 'top' ? 0 : 2}
            />
            <text
              x={l.x + 18}
              y={l.y + 3.5}
              fontSize={11}
              fontWeight={600}
              fill="var(--th-ink)"
            >
              {l.label}
            </text>
          </g>
        ))}

        {hover && (
          <circle
            cx={hover.x}
            cy={hover.y}
            r={5}
            fill="none"
            stroke="var(--th-ink)"
            strokeWidth={1.5}
          />
        )}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute rounded-md border px-2 py-1 text-[12px]"
          style={{
            left: `${(hover.x / W) * 100}%`,
            top: `${(hover.y / H) * 100}%`,
            transform: `translate(${hover.x > W - M.right - 120 ? '-105%' : '10px'}, -120%)`,
            borderColor: 'var(--th-border)',
            background: 'var(--th-card)',
            color: 'var(--th-ink)',
            whiteSpace: 'nowrap',
          }}
        >
          <b>{hover.team}</b> · {xLabel(hover.date)} ·{' '}
          {hover.band ? 'median run ' : 'Elo '}
          {Math.round(hover.elo)}
          {hover.band && ` (${Math.round(hover.band[0])}–${Math.round(hover.band[1])})`}
        </div>
      )}

      <p className="mt-1 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Labelled lines are the top {model.labels.filter((l) => l.group === 'top').length} and
        bottom {model.labels.filter((l) => l.group === 'bottom').length} by current Elo, numbered
        by rank — filled markers at the top of the table, ringed at the bottom; grey is the rest
        of the pack{showProjection ? ', which stops at today — hover any grey line to project it' : ''}.
        {showProjection
          ? ' Right of the “today” rule every line is a simulated season, never an average of them: the dashes are each side’s median run — the one season that finished mid-distribution, picked per side — and the thin lines are three whole seasons shared across sides, so those crossings are one coherent league. An average would be flat, since each game’s expected Elo change is about zero. Hovering a projected point washes in that line’s 10th–90th percentile band and prints the range.'
          : ''}{' '}
        Hover any point for the exact value.
      </p>
    </div>
  );
}
