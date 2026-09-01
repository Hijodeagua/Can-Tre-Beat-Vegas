'use client';

import { useMemo, useState } from 'react';

/**
 * Current-season Elo trajectories for one league, drawn as inline SVG —
 * the site has no chart dependency and doesn't need one for a line chart.
 *
 * The daily pipeline exports each club's pre-match rating at every match
 * date plus today's live rating (`elo_history` in latest.json), so the
 * chart moves every day the job runs, not just on matchdays.
 *
 * Twenty clubs can't wear twenty distinguishable hues, so the top
 * HIGHLIGHTS clubs by current Elo get the categorical palette + an
 * end-of-line label, and the rest recede to thin muted lines. Identity is
 * therefore never color-alone: every colored line is direct-labeled, and
 * the full ratings table sits on the same tab.
 */

export interface EloSeries {
  team: string;
  points: [string, number][]; // [ISO date, elo]
}

// Categorical slots 1-6 (light mode) from the validated reference palette;
// order is the CVD-safety mechanism — assign in order, never cycle.
const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'];
const MUTED_LINE = 'var(--th-muted)';
const MUTED_OPACITY = 0.3;
const HIGHLIGHTS = SERIES_COLORS.length;

const W = 760;
const H = 380;
const M = { top: 16, right: 168, bottom: 28, left: 46 };

function shortName(team: string): string {
  return team
    .replace(/^(AFC|FC|AC|AS|SS|SSC|US|CF|RC|RCD|SL|VfB|VfL|TSG|SV|1\.\s*FC)\s+/i, '')
    .replace(/\s+(FC|AFC|CF|SC|AC|BC|Calcio|Balompié)$/i, '');
}

interface Hover {
  x: number;
  y: number;
  team: string;
  date: string;
  elo: number;
}

export default function EloTrendChart({ series }: { series: EloSeries[] }) {
  const [hover, setHover] = useState<Hover | null>(null);

  const model = useMemo(() => {
    const withFinal = series
      .filter((s) => s.points.length > 0)
      .map((s) => ({ ...s, final: s.points[s.points.length - 1][1] }));
    withFinal.sort((a, b) => b.final - a.final);

    const allPoints = withFinal.flatMap((s) => s.points);
    if (allPoints.length === 0) return null;

    const ts = allPoints.map(([d]) => new Date(`${d}T00:00:00Z`).getTime());
    const elos = allPoints.map(([, e]) => e);
    const t0 = Math.min(...ts);
    const t1 = Math.max(...ts);
    const pad = Math.max(8, (Math.max(...elos) - Math.min(...elos)) * 0.06);
    const e0 = Math.min(...elos) - pad;
    const e1 = Math.max(...elos) + pad;

    const x = (d: string) => {
      const t = new Date(`${d}T00:00:00Z`).getTime();
      return t1 === t0
        ? M.left + (W - M.left - M.right) / 2
        : M.left + ((t - t0) / (t1 - t0)) * (W - M.left - M.right);
    };
    const y = (e: number) => M.top + ((e1 - e) / (e1 - e0)) * (H - M.top - M.bottom);

    const drawn = withFinal.map((s, rank) => ({
      team: s.team,
      color: rank < HIGHLIGHTS ? SERIES_COLORS[rank] : null,
      pts: s.points.map(([d, e]) => ({ d, e, px: x(d), py: y(e) })),
    }));

    // End labels for the highlighted clubs, nudged apart vertically.
    const labels = drawn
      .filter((s) => s.color)
      .map((s) => ({
        team: shortName(s.team),
        color: s.color!,
        x: s.pts[s.pts.length - 1].px,
        y: s.pts[s.pts.length - 1].py,
      }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
    }

    // Recessive y grid: 4 round-numbered lines.
    const step = Math.max(10, Math.round((e1 - e0) / 4 / 10) * 10);
    const gridStart = Math.ceil(e0 / step) * step;
    const grid: number[] = [];
    for (let v = gridStart; v < e1; v += step) grid.push(v);

    const dates = Array.from(new Set(allPoints.map(([d]) => d))).sort();
    return { drawn, labels, grid, y, x, dates, firstDate: dates[0], lastDate: dates[dates.length - 1] };
  }, [series]);

  if (!model) return null;

  const onMove = (evt: React.MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const mx = ((evt.clientX - rect.left) / rect.width) * W;
    const my = ((evt.clientY - rect.top) / rect.height) * H;
    let best: Hover | null = null;
    let bestDist = 24 * 24; // generous hit target, larger than the marks
    for (const s of model.drawn) {
      for (const p of s.pts) {
        const dist = (p.px - mx) ** 2 + (p.py - my) ** 2;
        if (dist < bestDist) {
          bestDist = dist;
          best = { x: p.px, y: p.py, team: s.team, date: p.d, elo: p.e };
        }
      }
    }
    setHover(best);
  };

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Club Elo ratings over the current season"
        className="w-full"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
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

        <text x={M.left} y={H - 8} fontSize={10} fill="var(--th-faint)">
          {model.firstDate}
        </text>
        <text x={W - M.right} y={H - 8} textAnchor="end" fontSize={10} fill="var(--th-faint)">
          {model.lastDate}
        </text>

        {/* Muted pack first, colored highlights on top. */}
        {model.drawn.filter((s) => !s.color).map((s) => (
          <polyline
            key={s.team}
            points={s.pts.map((p) => `${p.px},${p.py}`).join(' ')}
            fill="none"
            stroke={MUTED_LINE}
            strokeOpacity={MUTED_OPACITY}
            strokeWidth={1.25}
          />
        ))}
        {model.drawn.filter((s) => s.color).map((s) => (
          <g key={s.team}>
            <polyline
              points={s.pts.map((p) => `${p.px},${p.py}`).join(' ')}
              fill="none"
              stroke={s.color!}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.pts.map((p) => (
              <circle key={p.d} cx={p.px} cy={p.py} r={2.5} fill={s.color!} />
            ))}
          </g>
        ))}

        {model.labels.map((l) => (
          <text
            key={l.team}
            x={l.x + 8}
            y={l.y + 3.5}
            fontSize={11}
            fontWeight={600}
            fill="var(--th-ink)"
          >
            <tspan fill={l.color}>●</tspan> {l.team}
          </text>
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
          <b>{hover.team}</b> · {hover.date} · Elo {Math.round(hover.elo)}
        </div>
      )}
    </div>
  );
}
