'use client';

import { useMemo, useState, type ReactNode } from 'react';
import { TABLE_HEADER_FONT } from '@/app/lib/theme';

/**
 * ThemedTable's sortable sibling: identical chrome (accent header, zebra
 * rows, the note line), plus click-to-sort headers with aria-sort.
 *
 * Kept separate from ThemedTable rather than bolted onto it because
 * sorting needs client state, and ThemedTable's whole point is being a
 * plain presentational component the server pages can render.
 *
 * Each row carries `values` alongside its rendered `cells`: the raw
 * sortable value per column (string or number; null sorts last in either
 * direction). The first click on a column sorts descending for numbers
 * and ascending for strings — the direction you almost always want for
 * "biggest first" stats vs. names.
 */

export interface SortableColumn {
  header: string;
  strong?: boolean;
  /** Right-align + first click sorts descending. */
  numeric?: boolean;
}

export interface SortableRow {
  key: string;
  cells: ReactNode[];
  values: (string | number | null)[];
}

export interface SortableThemedTableProps {
  columns: SortableColumn[];
  rows: SortableRow[];
  note?: ReactNode;
  headerFont?: 'pixel' | 'plain';
  /** Column index to sort by on first render; omit to keep the given order. */
  initialSort?: { column: number; dir: 'asc' | 'desc' };
}

type SortState = { column: number; dir: 'asc' | 'desc' } | null;

function compareValues(a: string | number | null, b: string | number | null): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1; // nulls last regardless of direction
  if (b === null) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b));
}

export default function SortableThemedTable({
  columns,
  rows,
  note,
  headerFont = TABLE_HEADER_FONT,
  initialSort,
}: SortableThemedTableProps) {
  const [sort, setSort] = useState<SortState>(initialSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const dir = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const cmp = compareValues(a.values[sort.column], b.values[sort.column]);
      // compareValues already pins nulls last; don't let dir flip that.
      if (a.values[sort.column] === null || b.values[sort.column] === null) return cmp;
      return cmp * dir;
    });
  }, [rows, sort]);

  const clickHeader = (i: number) => {
    setSort((prev) => {
      if (prev?.column === i) {
        return { column: i, dir: prev.dir === 'asc' ? 'desc' : 'asc' };
      }
      return { column: i, dir: columns[i].numeric ? 'desc' : 'asc' };
    });
  };

  const pixel = headerFont === 'pixel';
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
            {columns.map((col, i) => {
              const active = sort?.column === i;
              const ariaSort = active
                ? sort!.dir === 'asc' ? 'ascending' : 'descending'
                : undefined;
              return (
                <th
                  key={`${col.header}-${i}`}
                  scope="col"
                  aria-sort={ariaSort}
                  className={`whitespace-nowrap px-4 py-2 text-left uppercase tracking-wide ${headerClass}`}
                  style={{ color: 'var(--th-head-ink)' }}
                >
                  <button
                    type="button"
                    onClick={() => clickHeader(i)}
                    className="inline-flex cursor-pointer items-center gap-1 uppercase tracking-wide"
                    style={{
                      font: 'inherit',
                      color: 'inherit',
                      background: 'none',
                      border: 'none',
                      padding: 0,
                    }}
                    title={`Sort by ${col.header}`}
                  >
                    {col.header}
                    <span aria-hidden="true" style={{ opacity: active ? 1 : 0.35 }}>
                      {active ? (sort!.dir === 'asc' ? '▲' : '▼') : '↕'}
                    </span>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={row.key}
              style={{
                borderTop: '1px solid var(--th-row)',
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
