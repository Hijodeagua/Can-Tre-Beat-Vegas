import type { ReactNode } from 'react';
import { TABLE_HEADER_FONT } from '@/app/lib/theme';

/**
 * The one table on the site. Replaces the old per-tab `<table>` markup and
 * serves all five surfaces: the home slate, and MLB's futures, slate, grade
 * and history tabs.
 *
 * Header and zebra tints come from `--sport-accent`, which the enclosing
 * section sets (see `accentVars` in app/lib/sports.ts). That is the whole
 * mechanism behind "an NFL table is yellow with no code change" — this
 * component never learns which sport it is rendering.
 *
 * Deliberately a plain presentational component with no data access, so it
 * renders from both the server pages and the client-side MLB tab view.
 */

export interface ThemedColumn {
  header: string;
  /** Renders this column's cells semibold — the model pick, a key percentage. */
  strong?: boolean;
}

export interface ThemedRow {
  key: string;
  cells: ReactNode[];
}

export interface ThemedTableProps {
  columns: ThemedColumn[];
  rows: ThemedRow[];
  /**
   * The explanatory line under the table. Every data surface keeps one — a
   * number without its caveat is worse than no number.
   */
  note?: ReactNode;
  headerFont?: 'pixel' | 'plain';
}

export default function ThemedTable({
  columns,
  rows,
  note,
  headerFont = TABLE_HEADER_FONT,
}: ThemedTableProps) {
  const pixel = headerFont === 'pixel';
  // Press Start 2P never goes below 8px; Inter takes the plain variant at 12.
  const headerClass = pixel
    ? 'pixel text-[8px] font-normal'
    : 'text-[12px] font-medium';

  return (
    <div
      className="overflow-x-auto rounded-lg border"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
    >
      <table className="w-full border-collapse text-[14px]" style={{ color: 'var(--th-ink)' }}>
        <thead>
          <tr style={{ background: 'var(--th-head-bg)' }}>
            {columns.map((col, i) => (
              <th
                key={`${col.header}-${i}`}
                scope="col"
                className={`whitespace-nowrap px-4 py-2 text-left uppercase tracking-wide ${headerClass}`}
                style={{ color: 'var(--th-head-ink)' }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.key}
              style={{
                borderTop: '1px solid var(--th-row)',
                // Faint zebra in the sport's own hue rather than a grey.
                background: i % 2 === 1 ? 'var(--th-zebra)' : 'transparent',
              }}
            >
              {row.cells.map((cell, j) => (
                <td
                  key={j}
                  className={`whitespace-nowrap px-4 py-2 ${
                    columns[j]?.strong ? 'font-semibold' : ''
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {note && (
        <p className="m-0 px-4 py-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          {note}
        </p>
      )}
    </div>
  );
}
