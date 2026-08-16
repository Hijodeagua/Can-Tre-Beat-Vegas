import Link from 'next/link';
import StatTile from '@/app/components/StatTile';
import type { SummaryModel } from '@/app/lib/data';
import { DASH, fmtPct, fmtPctPrecise, fmtRecord } from '@/app/lib/format';
import type { SportConfig } from '@/app/lib/sports';

/**
 * One model's sport-level summary: its glyph, its season status, whatever
 * record it has earned, and a way into the sport.
 *
 * Two shapes of the same thing, so every sport is described by one component
 * regardless of which half of the page it lands on:
 *
 * - `row`  — a rung of the cross-sport ladder inside the track-record card.
 * - `card` — the dashed "up next season" card for a model that is off-season.
 *
 * Nothing sport-specific is written here. The glyph, accent and copy come from
 * the `SportConfig`, the numbers from `summary.json`, which is what lets NFL
 * and NBA slot in as data rather than as new markup. The matchup-level card is
 * a different object entirely and stays `GameCard`.
 */

interface ModelCardProps {
  sport: SportConfig;
  model: SummaryModel;
  variant: 'row' | 'card';
  /** Lead copy for the card variant, already resolved against real data. */
  lead?: string;
}

export default function ModelCard({ sport, model, variant, lead }: ModelCardProps) {
  return variant === 'row' ? (
    <LadderRow sport={sport} model={model} />
  ) : (
    <UpNextCard sport={sport} model={model} lead={lead} />
  );
}

function Glyph({ sport, size }: { sport: SportConfig; size: 'sm' | 'lg' }) {
  const sm = size === 'sm';
  return (
    <span
      className={`grid shrink-0 place-items-center ${
        sm ? 'h-[28px] w-[28px] rounded-sm text-[15px]' : 'h-[34px] w-[34px] rounded-md text-[18px]'
      }`}
      style={{ background: sport.tint }}
      aria-hidden="true"
    >
      {sport.emoji}
    </span>
  );
}

function statusLabel(model: SummaryModel): string {
  return model.status === 'in_season' ? 'In season' : 'Off-season';
}

function LadderRow({ sport, model }: { sport: SportConfig; model: SummaryModel }) {
  const inner = (
    /*
     * Three columns on a normal screen. On a phone the metrics drop to their
     * own full-width row rather than forcing the page to scroll sideways —
     * only tables do that here.
     */
    <div
      className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-2 rounded-md border p-3 sm:grid-cols-[132px_1fr_auto]"
      style={{ background: 'var(--th-inset)', borderColor: 'var(--th-row)' }}
    >
      <div className="flex items-center gap-2">
        <Glyph sport={sport} size="sm" />
        <span className="pixel text-[10px]" style={{ color: 'var(--th-ink)' }}>
          {sport.name}
        </span>
      </div>
      <span
        className="text-right text-[12px] uppercase tracking-wide sm:text-left"
        style={{ color: 'var(--th-faint)' }}
      >
        {statusLabel(model)}
      </span>
      <div
        className="col-span-2 flex flex-wrap gap-x-5 gap-y-1 text-[14px] sm:col-span-1"
        style={{ color: 'var(--th-ink)' }}
      >
        <span className="font-semibold">{fmtRecord(model.record)}</span>
        {/* One decimal on the ladder, matching the headline it sits under. */}
        <span>{fmtPctPrecise(model.accuracy)}</span>
        <span style={{ color: 'var(--th-faint)' }}>{model.last_graded ?? DASH}</span>
      </div>
    </div>
  );

  // A sport without a page of its own is not a link — there is nowhere to go
  // until its route exists.
  if (!sport.href) return inner;
  return (
    <Link
      href={sport.href}
      className="block rounded-md hover:brightness-[0.98]"
      aria-label={`${sport.name} model — ${statusLabel(model)}`}
    >
      {inner}
    </Link>
  );
}

function UpNextCard({
  sport,
  model,
  lead,
}: {
  sport: SportConfig;
  model: SummaryModel;
  lead?: string;
}) {
  return (
    <div
      className="rounded-lg border-2 border-dashed p-5"
      style={{ borderColor: sport.dashBorder, background: 'var(--th-card)' }}
    >
      <div className="flex items-center gap-2">
        <Glyph sport={sport} size="lg" />
        <h4 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
          {sport.name}
        </h4>
        <span
          className="ml-auto text-[11px] uppercase tracking-wide"
          style={{ color: 'var(--th-faint)' }}
        >
          {statusLabel(model)}
        </span>
      </div>

      {lead && (
        <p className="mt-3 text-[14px] leading-normal" style={{ color: 'var(--th-muted)' }}>
          {lead}
        </p>
      )}

      {/*
        Both tiles read from the model's real figures. A sport that has never
        been graded has nulls here, which format as em dashes — not zeros.
      */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        <StatTile label="Last season" value={fmtRecord(model.record)} />
        <StatTile label="Accuracy" value={fmtPct(model.accuracy)} />
      </div>

      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {sport.offseasonNote}
      </p>
    </div>
  );
}
