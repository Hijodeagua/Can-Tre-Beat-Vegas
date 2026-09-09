'use client';

/**
 * The archive list: a league pill row, a type pill row, and the emails as
 * a sortable table with a link into each archived HTML. Reads `?league=`
 * on mount so the footer link in every email lands on its own league.
 */
import { useEffect, useState } from 'react';
import SortableThemedTable from '@/app/components/SortableThemedTable';
import { getEmailIndex, leagueName, typeLabel, type ArchivedEmail } from '@/app/lib/emails';
import { sportByKey } from '@/app/lib/sports';

const index = getEmailIndex();
const LEAGUES = ['all', ...Object.keys(index.leagues ?? {})];
const MODELS_ACCENT = { accent: '#ffd23f', accentInk: '#06120b' };

const COLUMNS = [
  { header: 'Date' },
  { header: 'League', strong: true },
  { header: 'Email' },
  { header: 'Subject' },
  { header: 'Open' },
];

function Pills({
  keys, active, onSelect, label,
}: {
  keys: string[];
  active: string;
  onSelect: (k: string) => void;
  label: (k: string) => string;
}) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {keys.map((k) => (
        <button
          key={k}
          onClick={() => onSelect(k)}
          aria-pressed={active === k}
          className={`rounded-full px-3 py-1 text-[13px] ${
            active === k ? 'font-semibold' : 'hover:bg-slate-100'
          }`}
          style={
            active === k
              ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
              : { color: 'var(--th-muted)', border: '1px solid var(--th-border)' }
          }
        >
          {label(k)}
        </button>
      ))}
    </div>
  );
}

function leagueGlyph(key: string): string {
  return sportByKey(key)?.emoji ?? (key === 'models' ? '📊' : '');
}

export default function EmailArchive() {
  const [league, setLeague] = useState('all');
  const [type, setType] = useState('all');

  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get('league');
    if (requested && LEAGUES.includes(requested)) setLeague(requested);
  }, []);

  const select = (k: string) => {
    setLeague(k);
    setType('all');
    window.history.replaceState(null, '', k === 'all' ? window.location.pathname : `?league=${k}`);
  };

  const inLeague = league === 'all'
    ? index.emails
    : index.emails.filter((e) => e.league === league);
  const types = ['all', ...Array.from(new Set(inLeague.map((e) => e.type))).sort()];
  const rows: ArchivedEmail[] = type === 'all' ? inLeague : inLeague.filter((e) => e.type === type);

  // The section wears the selected league's accent; "all" and the models
  // check use the site's yellow so the pills always have a colour.
  const sport = sportByKey(league);
  const accent = sport ? { accent: sport.accent, accentInk: sport.accentInk } : MODELS_ACCENT;

  return (
    <section
      className="mt-6"
      style={{
        '--sport-accent': accent.accent,
        '--sport-accent-ink': accent.accentInk,
      } as React.CSSProperties}
    >
      <Pills
        keys={LEAGUES}
        active={league}
        onSelect={select}
        label={(k) => (k === 'all' ? 'All leagues' : `${leagueGlyph(k)} ${leagueName(k)}`)}
      />
      {types.length > 2 && (
        <Pills
          keys={types}
          active={type}
          onSelect={setType}
          label={(k) => (k === 'all' ? 'All emails' : typeLabel(k))}
        />
      )}
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {rows.length} email{rows.length === 1 ? '' : 's'}
        {league !== 'all' ? ` · ${leagueName(league)}` : ''}
        {type !== 'all' ? ` · ${typeLabel(type)}` : ''} · newest first.
      </p>
      <div className="mt-2">
        {rows.length === 0 ? (
          <div
            className="rounded-lg border border-dashed p-8 text-center text-[14px]"
            style={{
              borderColor: 'var(--th-border)',
              background: 'var(--th-card)',
              color: 'var(--th-muted)',
            }}
          >
            Nothing archived yet for this selection.
          </div>
        ) : (
          <SortableThemedTable
            columns={COLUMNS}
            initialSort={{ column: 0, dir: 'desc' }}
            rows={rows.map((e) => ({
              key: `${e.league}:${e.type}:${e.date}`,
              cells: [
                e.date,
                `${leagueGlyph(e.league)} ${leagueName(e.league)}`,
                typeLabel(e.type),
                e.subject || `${leagueName(e.league)} ${typeLabel(e.type)} — ${e.date}`,
                <a
                  key="open"
                  href={`/vegas/${e.public_path}`}
                  className="underline-offset-2 hover:underline"
                  style={{ color: 'var(--th-ink)' }}
                >
                  Open ↗
                </a>,
              ],
              values: [e.date, leagueName(e.league), typeLabel(e.type), e.subject, e.date],
            }))}
            note="Archived copies are byte-for-byte what was delivered (emails from before the archive existed were copied in without the permalink footer). The MLB grade email is dated by the day it graded, one day before it was sent."
          />
        )}
      </div>
    </section>
  );
}
