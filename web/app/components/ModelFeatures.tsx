'use client';

/**
 * The "what this forecast is actually looking at" block that sits with
 * every forecast table: home Elo, away Elo, and the sport-specific
 * features, from `app/lib/modelFeatures.ts`.
 *
 * It ships collapsed — a reader who came for the odds shouldn't have to
 * scroll past a feature list to reach them — and open it names every
 * feature, with a dormant one struck through and its reason given rather
 * than dropped from the list.
 *
 * The note under it is the part worth keeping even if the rest is
 * collapsed: these forecasts are thousands of replayed seasons, so the
 * ratings and the table move inside every run. The averaged rating is
 * flat only because a fair game's expected Elo change is about zero —
 * which is why the chart above draws individual simulated seasons and not
 * just the mean.
 */

import { SHEET_URL, type ForecastModel } from '@/app/lib/modelFeatures';

export default function ModelFeatures({
  model,
  sims,
  unit = 'season',
}: {
  model: ForecastModel;
  /** Replays behind this forecast, for the one-line summary. */
  sims?: number;
  /** What one replay covers, e.g. "season". */
  unit?: string;
}) {
  return (
    <details
      className="mt-6 rounded-lg border"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
    >
      <summary
        className="cursor-pointer list-none px-4 py-3 text-[13px] font-semibold"
        style={{ color: 'var(--th-ink)' }}
      >
        {model.title}
        <span className="ml-2 font-normal" style={{ color: 'var(--th-faint)' }}>
          — what this forecast reads
        </span>
      </summary>
      <div className="px-4 pb-4">
        <p className="m-0 text-[12px]" style={{ color: 'var(--th-muted)' }}>
          {model.engine}
        </p>
        <ul className="m-0 mt-3 grid list-none gap-1 p-0 text-[12px] sm:grid-cols-2">
          {model.features.map((f) => (
            <li key={f.name} style={{ color: 'var(--th-muted)' }}>
              <span
                className="font-semibold"
                style={{
                  color: f.dormant ? 'var(--th-faint)' : 'var(--th-ink)',
                  textDecoration: f.dormant ? 'line-through' : undefined,
                }}
              >
                {f.name}
              </span>{' '}
              — {f.detail}
              {f.dormant && (
                <span style={{ color: 'var(--th-faint)' }}> · dormant: {f.dormant}</span>
              )}
            </li>
          ))}
        </ul>
        <p className="mb-0 mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          {sims ? `${sims.toLocaleString()} replayed ${unit}s, rerun daily. ` : ''}
          Ratings move inside every replay, so the table on screen is one point in a
          distribution, not a fixed outcome — the averaged rating barely moves only because a
          fair game&apos;s expected Elo change is about zero, which is why the chart draws
          individual simulated {unit}s rather than the average alone.{' '}
          <a
            href={SHEET_URL}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2"
            style={{ color: 'var(--th-muted)' }}
          >
            Full feature sheet
          </a>
          .
        </p>
      </div>
    </details>
  );
}
