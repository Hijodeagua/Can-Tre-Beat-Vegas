/**
 * A small labelled figure — the hero metrics beside the headline and the
 * placeholder pair on each off-season card. Values arrive pre-formatted, so a
 * missing metric shows up here as the em dash its formatter produced.
 */
export default function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-sm border p-2 text-center"
      style={{ borderColor: 'var(--th-row)', background: 'var(--th-inset)' }}
    >
      <div
        className="text-[11px] uppercase tracking-wide"
        style={{ color: 'var(--th-faint)' }}
      >
        {label}
      </div>
      <div className="mt-0.5 text-[18px] font-bold" style={{ color: 'var(--th-ink)' }}>
        {value}
      </div>
    </div>
  );
}
