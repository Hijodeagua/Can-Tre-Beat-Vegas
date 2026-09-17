/**
 * What a model's inputs are worth, as a signed bar per input.
 *
 * The quantity is a delta against a baseline — how much worse the model
 * scores with that input taken away — so the form is a diverging bar on
 * one shared scale, zero placed where the data puts it. Blue means the
 * input earns its place; red means the model scored *better* without it,
 * which is a thing worth being able to see rather than hiding behind an
 * absolute value. There is no third hue at the midpoint: zero is the
 * surface.
 *
 * Bars are proportional across the whole axis, including the negative
 * side, so a barely-negative input reads as a sliver rather than being
 * inflated to something visible. That is the honest rendering: it changed
 * almost nothing.
 *
 * Every bar is direct-labelled with its value, because the numbers here
 * are small (thousandths of a nat) and no axis tick would let a reader
 * recover them. A permutation run also carries its spread across repeats,
 * drawn as a whisker — a bar whose whisker crosses zero has not been
 * measured apart from noise.
 */

import type { ImportanceModel } from '@/app/lib/importance';

const TRACK = 320; // px of bar track at full width

function fmt(v: number): string {
  const s = v >= 0 ? '+' : '−';
  return `${s}${Math.abs(v).toFixed(v !== 0 && Math.abs(v) < 0.001 ? 4 : 3)}`;
}

export default function ImportanceChart({ model }: { model: ImportanceModel }) {
  const values = model.features.map((f) => f.value);
  const max = Math.max(0, ...values);
  const min = Math.min(0, ...values);
  const span = max - min || 1;
  // Where zero sits on the shared axis, as a fraction of the track.
  const zero = (-min / span) * TRACK;

  return (
    <div>
      <p className="m-0 text-[12px]" style={{ color: 'var(--th-muted)' }}>
        {model.metric}. Baseline log loss <b>{model.baseline.toFixed(5)}</b> over{' '}
        {model.n.toLocaleString()} games, {model.window}.
      </p>

      <ul className="m-0 mt-3 grid list-none gap-1.5 p-0">
        {model.features.map((f) => {
          const length = (Math.abs(f.value) / span) * TRACK;
          const negative = f.value < 0;
          const color = f.constant
            ? 'var(--th-faint)'
            : negative
              ? 'var(--viz-neg)'
              : 'var(--viz-pos)';
          return (
            <li key={f.name} className="flex items-center gap-3 text-[12px]">
              <span
                className="w-[150px] shrink-0 text-right font-semibold"
                style={{
                  color: f.constant ? 'var(--th-faint)' : 'var(--th-ink)',
                  fontFamily: f.name.includes('_')
                    ? 'ui-monospace, SFMono-Regular, Menlo, monospace'
                    : undefined,
                }}
                title={f.detail}
              >
                {f.name}
              </span>
              <span
                className="relative hidden h-[14px] shrink-0 sm:block"
                style={{ width: TRACK }}
              >
                {/* Zero baseline — a solid hairline, one step off the surface. */}
                <span
                  className="absolute top-[-2px] h-[18px] w-px"
                  style={{ left: zero, background: 'var(--th-border)' }}
                />
                <span
                  className="absolute top-0 h-[14px]"
                  style={{
                    left: negative ? zero - length : zero,
                    width: Math.max(length, f.value === 0 ? 0 : 1),
                    background: color,
                    borderRadius: negative ? '4px 0 0 4px' : '0 4px 4px 0',
                  }}
                />
                {f.sd != null && f.sd > 0 && (
                  <span
                    className="absolute top-[6px] h-px"
                    style={{
                      left: zero + length - (f.sd / span) * TRACK,
                      width: ((2 * f.sd) / span) * TRACK,
                      background: 'var(--th-ink)',
                      opacity: 0.45,
                    }}
                  />
                )}
              </span>
              <span
                className="shrink-0 tabular-nums"
                style={{ color: f.constant ? 'var(--th-faint)' : 'var(--th-muted)' }}
              >
                {fmt(f.value)}
                {f.sd != null && f.sd > 0 && ` ±${f.sd.toFixed(3)}`}
              </span>
              {f.constant && (
                <span style={{ color: 'var(--th-faint)' }}>no data over this window</span>
              )}
            </li>
          );
        })}
      </ul>

      <p className="mb-0 mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        <b>{model.method}</b> — {model.caveat}
      </p>
    </div>
  );
}
