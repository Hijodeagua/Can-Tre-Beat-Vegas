'use client';

/**
 * The player-comparison view: a search box, up to five player-seasons as
 * columns, one row per stat with a percentile bar against positional
 * peers, and a "similar players" list for the first column.
 *
 * State lives in the URL (`?p=2025:epl:1250,2024:la_liga:2371`) so a
 * comparison can be linked. Season files are fetched on demand - see
 * app/lib/players.ts for why this page does not import its data.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fmtStat, fold, GROUP_LABEL, loadIndex, loadSeason, MIN_POOL, minutesFloor, peerPool,
  percentile, similarPlayers, STATS, statValue,
  type MinutesRule, type Mode, type PlayerIndex, type PlayerSeason, type PositionGroup,
  type Scope, type SeasonEntry,
} from '@/app/lib/players';

const MAX_PLAYERS = 5;
/** One series slot per column, assigned in order and never cycled (see globals.css). */
const SLOT_COLORS = ['var(--viz-1)', 'var(--viz-2)', 'var(--viz-3)', 'var(--viz-4)', 'var(--viz-5)'];
const MINUTES_OPTIONS: MinutesRule[] = ['auto', 300, 600, 900, 1500];
const GROUPS: PositionGroup[] = ['FW', 'MF', 'DF', 'GK'];

type GroupRule = 'auto' | PositionGroup;

function readKeys(): string[] {
  if (typeof window === 'undefined') return [];
  const raw = new URLSearchParams(window.location.search).get('p');
  return raw ? raw.split(',').filter(Boolean).slice(0, MAX_PLAYERS) : [];
}

function writeKeys(keys: string[]) {
  const url = new URL(window.location.href);
  if (keys.length) url.searchParams.set('p', keys.join(','));
  else url.searchParams.delete('p');
  // The keys are plain [0-9a-z_:,] - keep them readable rather than %-encoded.
  window.history.replaceState(null, '', url.toString().replace(/%3A/gi, ':').replace(/%2C/gi, ','));
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="mt-6 rounded-lg border border-dashed p-8 text-center text-[14px]"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-muted)' }}
    >
      {children}
    </div>
  );
}

function Segmented<T extends string | number>({
  label, options, value, onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[12px] uppercase tracking-wide" style={{ color: 'var(--th-faint)' }}>
        {label}
      </span>
      <div className="flex flex-wrap gap-1" role="group" aria-label={label}>
        {options.map((o) => {
          const on = o.value === value;
          return (
            <button
              key={String(o.value)}
              type="button"
              aria-pressed={on}
              onClick={() => onChange(o.value)}
              className="rounded-md border px-2.5 py-1 text-[13px]"
              style={{
                borderColor: on ? 'var(--th-ink)' : 'var(--th-border)',
                background: on ? 'var(--th-ink)' : 'var(--th-card)',
                color: on ? '#ffffff' : 'var(--th-muted)',
                fontWeight: on ? 600 : 400,
              }}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Season picker + typeahead. Picking a result hands the player to `onAdd`. */
function PlayerSearch({
  index, disabled, taken, onAdd,
}: {
  index: PlayerIndex;
  disabled: boolean;
  taken: Set<string>;
  onAdd: (p: PlayerSeason) => void;
}) {
  const [seasonKey, setSeasonKey] = useState(index.seasons[0].key);
  const [players, setPlayers] = useState<PlayerSeason[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const entry = index.seasons.find((s) => s.key === seasonKey);
    if (!entry) return;
    let live = true;
    setPlayers(null);
    setFailed(false);
    loadSeason(entry).then(
      (rows) => live && setPlayers(rows),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [index, seasonKey]);

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', away);
    return () => document.removeEventListener('mousedown', away);
  }, []);

  const matches = useMemo(() => {
    const q = fold(query.trim());
    if (!players || q.length < 2) return [];
    const words = q.split(/\s+/);
    return players
      .filter((p) => {
        const hay = `${fold(p.name)} ${fold(p.team)}`;
        return words.every((w) => hay.includes(w));
      })
      .slice(0, 8); // rows arrive sorted by minutes, so regulars come first
  }, [players, query]);

  const pick = (p: PlayerSeason) => {
    if (taken.has(p.key)) return;
    onAdd(p);
    setQuery('');
    setOpen(false);
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <select
        aria-label="Season"
        value={seasonKey}
        onChange={(e) => setSeasonKey(Number(e.target.value))}
        className="rounded-md border px-2 py-2 text-[14px]"
        style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-ink)' }}
      >
        {index.seasons.map((s) => (
          <option key={s.key} value={s.key}>{s.label}</option>
        ))}
      </select>

      <div ref={box} className="relative min-w-[220px] flex-1">
        <input
          type="search"
          role="combobox"
          aria-expanded={open && matches.length > 0}
          aria-controls="player-results"
          aria-label="Search players"
          disabled={disabled || failed}
          value={query}
          placeholder={
            disabled ? `Remove a player to add another (max ${MAX_PLAYERS})`
              : failed ? 'This season failed to load'
                : players ? 'Search a player or club…' : 'Loading season…'
          }
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setCursor(0);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setCursor((c) => Math.min(c + 1, matches.length - 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setCursor((c) => Math.max(c - 1, 0));
            } else if (e.key === 'Enter' && matches[cursor]) {
              e.preventDefault();
              pick(matches[cursor]);
            } else if (e.key === 'Escape') {
              setOpen(false);
            }
          }}
          className="w-full rounded-md border px-3 py-2 text-[14px]"
          style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-ink)' }}
        />
        {open && query.trim().length >= 2 && players && (
          <ul
            id="player-results"
            role="listbox"
            className="absolute left-0 right-0 z-20 m-0 mt-1 max-h-[340px] list-none overflow-y-auto rounded-md border p-1"
            style={{ borderColor: 'var(--th-border)', background: 'var(--th-bar)' }}
          >
            {matches.length === 0 && (
              <li className="px-3 py-2 text-[13px]" style={{ color: 'var(--th-muted)' }}>
                No player matches “{query.trim()}” in this season.
              </li>
            )}
            {matches.map((p, i) => {
              const used = taken.has(p.key);
              return (
                <li key={p.key} role="option" aria-selected={i === cursor}>
                  <button
                    type="button"
                    disabled={used}
                    onMouseEnter={() => setCursor(i)}
                    onClick={() => pick(p)}
                    className="flex w-full items-baseline justify-between gap-3 rounded-sm px-3 py-2 text-left text-[14px]"
                    style={{
                      background: i === cursor ? 'var(--th-chip)' : 'transparent',
                      color: used ? 'var(--th-faint)' : 'var(--th-ink)',
                    }}
                  >
                    <span className="font-semibold">{p.name}</span>
                    <span className="text-[12px]" style={{ color: 'var(--th-muted)' }}>
                      {p.team} · {index.leagues[p.league] ?? p.league} · {p.group ?? 'sub'} ·{' '}
                      {p.min.toLocaleString()}′{used ? ' · added' : ''}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

function PercentileBar({ pct, color }: { pct: number | null; color: string }) {
  if (pct === null) {
    return <div className="h-[6px]" aria-hidden />;
  }
  return (
    <div
      className="h-[6px] overflow-hidden rounded-full"
      style={{ background: 'var(--th-row)' }}
      role="img"
      aria-label={`${pct}th percentile`}
    >
      <div className="h-full rounded-full" style={{ width: `${Math.max(pct, 2)}%`, background: color }} />
    </div>
  );
}

export default function CompareClient() {
  const [index, setIndex] = useState<PlayerIndex | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [keys, setKeys] = useState<string[]>([]);
  const [seasons, setSeasons] = useState<Record<number, PlayerSeason[]>>({});
  const [mode, setMode] = useState<Mode>('p90');
  const [scope, setScope] = useState<Scope>('top5');
  const [minutes, setMinutes] = useState<MinutesRule>('auto');
  const [groupRule, setGroupRule] = useState<GroupRule>('auto');
  const ready = useRef(false);

  useEffect(() => {
    setKeys(readKeys());
    ready.current = true;
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (ready.current) writeKeys(keys);
  }, [keys]);

  // Every season a selected player belongs to has to be in memory.
  useEffect(() => {
    if (!index) return;
    const wanted = new Set(keys.map((k) => Number(k.split(':')[0])));
    wanted.forEach((key) => {
      if (seasons[key]) return;
      const entry = index.seasons.find((s) => s.key === key);
      if (!entry) return;
      loadSeason(entry).then(
        (rows) => setSeasons((prev) => (prev[key] ? prev : { ...prev, [key]: rows })),
        (e: Error) => setError(e.message),
      );
    });
  }, [index, keys, seasons]);

  const selected = useMemo(() => {
    const out: PlayerSeason[] = [];
    for (const key of keys) {
      const rows = seasons[Number(key.split(':')[0])];
      const hit = rows?.find((p) => p.key === key);
      if (hit) out.push(hit);
    }
    return out;
  }, [keys, seasons]);

  const pending = keys.some((key) => {
    const season = Number(key.split(':')[0]);
    return !seasons[season] && index?.seasons.some((s) => s.key === season);
  });

  // One pool per column: the player's own season, and their own position
  // group unless a group is forced for everyone.
  const columns = useMemo(
    () =>
      selected.map((player, i) => {
        const season = seasons[player.season] ?? [];
        const group = groupRule === 'auto' ? player.group : groupRule;
        const floor = minutesFloor(minutes, season);
        const pool = peerPool(player, season, { group, floor, scope });
        return { player, group, floor, pool, color: SLOT_COLORS[i], small: player.min < floor };
      }),
    [selected, seasons, groupRule, minutes, scope],
  );

  const similar = useMemo(() => {
    const first = columns[0];
    if (!first || first.pool.length < MIN_POOL) return [];
    return similarPlayers(first.player, first.pool.filter((p) => p.min >= first.floor));
  }, [columns]);

  const add = (p: PlayerSeason) => {
    setKeys((prev) => (prev.includes(p.key) || prev.length >= MAX_PLAYERS ? prev : [...prev, p.key]));
  };
  const remove = (key: string) => setKeys((prev) => prev.filter((k) => k !== key));

  if (error) {
    return <Notice>Could not load the player data ({error}). Reload to try again.</Notice>;
  }
  if (index === undefined) return <Notice>Loading player data…</Notice>;
  if (index === null || index.seasons.length === 0) {
    return (
      <Notice>
        <b>No player data has been published yet.</b>
        <br />
        This page reads the season files written by{' '}
        <code>python -m soccer.clubs.data.understat_players</code>, which runs in the daily soccer
        workflow. They appear here after its first successful fetch.
      </Notice>
    );
  }

  const taken = new Set(keys);
  const leagueName = (k: string) => index.leagues[k] ?? k;
  const sections = Array.from(new Set(STATS.map((s) => s.section)));
  const cellBorder = { borderColor: 'var(--th-border)' };

  return (
    <div className="mt-6">
      <PlayerSearch index={index} disabled={keys.length >= MAX_PLAYERS} taken={taken} onAdd={add} />

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-3">
        <Segmented
          label="Numbers"
          value={mode}
          onChange={setMode}
          options={[{ value: 'p90', label: 'Per 90' }, { value: 'total', label: 'Totals' }]}
        />
        <Segmented
          label="Compared with"
          value={scope}
          onChange={setScope}
          options={[{ value: 'top5', label: 'Top 5 leagues' }, { value: 'league', label: 'Own league' }]}
        />
        <Segmented
          label="Position"
          value={groupRule}
          onChange={setGroupRule}
          options={[
            { value: 'auto' as GroupRule, label: 'Own' },
            ...GROUPS.map((g) => ({ value: g as GroupRule, label: g })),
          ]}
        />
        <Segmented
          label="Min. minutes"
          value={minutes}
          onChange={setMinutes}
          options={MINUTES_OPTIONS.map((m) => ({ value: m, label: m === 'auto' ? 'Auto' : String(m) }))}
        />
      </div>

      {columns.length === 0 && (
        <Notice>
          {pending
            ? 'Loading the linked players…'
            : 'Search for a player above to start a comparison. Add the same player from two seasons to see how they changed.'}
        </Notice>
      )}

      {columns.length > 0 && (
        <div
          className="mt-5 overflow-x-auto rounded-lg border"
          style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
        >
          <table className="w-full border-collapse text-[14px]" style={{ color: 'var(--th-ink)' }}>
            <thead>
              <tr>
                <th
                  scope="col"
                  className="sticky left-0 z-10 w-[150px] min-w-[130px] border-b px-3 py-3 text-left align-bottom text-[12px] font-medium"
                  style={{ ...cellBorder, background: 'var(--th-card)', color: 'var(--th-faint)' }}
                >
                  {mode === 'p90' ? 'Per 90 minutes' : 'Season totals'}
                  <br />
                  bar = percentile
                </th>
                {columns.map(({ player, group, pool, floor, color, small }) => (
                  <th
                    key={player.key}
                    scope="col"
                    className="min-w-[170px] border-b border-l px-3 py-3 text-left align-top font-normal"
                    style={cellBorder}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="inline-block h-[10px] w-[10px] shrink-0 rounded-full"
                          style={{ background: color }}
                          aria-hidden
                        />
                        <span className="text-[15px] font-semibold leading-tight">{player.name}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => remove(player.key)}
                        aria-label={`Remove ${player.name}`}
                        className="rounded-sm px-1.5 text-[16px] leading-none"
                        style={{ color: 'var(--th-faint)' }}
                      >
                        ×
                      </button>
                    </div>
                    <div className="mt-1 text-[12px]" style={{ color: 'var(--th-muted)' }}>
                      {player.team} · {player.seasonLabel}
                    </div>
                    <div className="text-[12px]" style={{ color: 'var(--th-muted)' }}>
                      {leagueName(player.league)} · {player.group ?? 'sub only'}
                    </div>
                    <div className="mt-1 text-[11px]" style={{ color: 'var(--th-faint)' }}>
                      {group
                        ? `vs ${pool.length} ${GROUP_LABEL[group].toLowerCase()}, ${floor}+ min`
                        : 'no position group — not ranked'}
                      {small && group ? ' · under the cutoff' : ''}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sections.map((section) => (
                <SectionRows
                  key={section}
                  section={section}
                  columns={columns}
                  mode={mode}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {columns.length > 0 && (
        <p className="mt-3 text-[12px] leading-normal" style={{ color: 'var(--th-muted)' }}>
          Each player is ranked inside their own season and position group, so a bar is a rank
          among peers, not a share of the leader’s number. The best raw figure in a row is bold.
          Card percentiles are flipped: a long bar means fewer cards. Understat has no goalkeeping
          or defensive-action data, so keepers and centre-backs are only described by what they do
          on the ball. Position comes from Understat’s most-used role for the season.
        </p>
      )}

      {similar.length > 0 && (
        <section className="mt-8">
          <h3 className="pixel m-0 text-[10px]" style={{ color: 'var(--th-ink)' }}>
            SIMILAR TO {columns[0].player.name.toUpperCase()}
          </h3>
          <p className="mt-2 text-[12px]" style={{ color: 'var(--th-muted)' }}>
            Closest statistical profiles in {columns[0].player.seasonLabel} among the same peer
            group, on per-90 shooting, creation and build-up numbers. Style, not quality.
          </p>
          <ul className="m-0 mt-3 grid list-none grid-cols-1 gap-2 p-0 sm:grid-cols-2">
            {similar.map(({ player, score }) => {
              const used = taken.has(player.key);
              const full = keys.length >= MAX_PLAYERS;
              return (
                <li
                  key={player.key}
                  className="flex items-center justify-between gap-3 rounded-md border px-3 py-2"
                  style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
                >
                  <div className="min-w-0">
                    <div className="truncate text-[14px] font-semibold">{player.name}</div>
                    <div className="truncate text-[12px]" style={{ color: 'var(--th-muted)' }}>
                      {player.team} · {leagueName(player.league)} · {player.min.toLocaleString()}′
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="text-[13px] tabular-nums" style={{ color: 'var(--th-muted)' }}>
                      {score}
                    </span>
                    <button
                      type="button"
                      disabled={used || full}
                      onClick={() => add(player)}
                      className="rounded-md border px-2.5 py-1 text-[13px]"
                      style={{
                        borderColor: 'var(--th-border)',
                        color: used || full ? 'var(--th-faint)' : 'var(--th-ink)',
                        background: 'var(--th-inset)',
                      }}
                    >
                      {used ? 'Added' : 'Compare'}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Data: {index.source}, top-flight league matches only · {index.seasons.length} season
        {index.seasons.length === 1 ? '' : 's'} · updated {index.generated_at.slice(0, 10)}.
      </p>
    </div>
  );
}

interface Column {
  player: PlayerSeason;
  group: PositionGroup | null;
  floor: number;
  pool: PlayerSeason[];
  color: string;
  small: boolean;
}

function SectionRows({ section, columns, mode }: { section: string; columns: Column[]; mode: Mode }) {
  const stats = STATS.filter((s) => s.section === section);
  const cellBorder = { borderColor: 'var(--th-border)' };
  return (
    <>
      <tr style={{ background: 'var(--th-head-bg)' }}>
        <th
          scope="colgroup"
          colSpan={columns.length + 1}
          className="pixel sticky left-0 px-3 py-2 text-left text-[8px] font-normal"
          style={{ color: 'var(--th-head-ink)' }}
        >
          {section.toUpperCase()}
        </th>
      </tr>
      {stats.map((stat) => {
        const values = columns.map((c) => statValue(stat, c.player, mode));
        const known = values.filter((v): v is number => v !== null);
        const best = known.length > 1
          ? (stat.lowerBetter ? Math.min(...known) : Math.max(...known))
          : null;
        const allLevel = known.length > 1 && known.every((v) => v === known[0]);
        return (
          <tr key={stat.key}>
            <th
              scope="row"
              title={stat.hint}
              className="sticky left-0 z-10 border-b px-3 py-2 text-left text-[13px] font-medium"
              style={{ ...cellBorder, background: 'var(--th-card)', color: 'var(--th-muted)' }}
            >
              {stat.label}
            </th>
            {columns.map((c, i) => {
              const v = values[i];
              const pct = c.group ? percentile(stat, c.player, c.pool, mode) : null;
              const lead = best !== null && !allLevel && v === best;
              return (
                <td key={c.player.key} className="border-b border-l px-3 py-2" style={cellBorder}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="tabular-nums" style={{ fontWeight: lead ? 700 : 400 }}>
                      {fmtStat(stat, v, mode)}
                    </span>
                    <span className="text-[12px] tabular-nums" style={{ color: 'var(--th-muted)' }}>
                      {pct === null ? '' : pct}
                    </span>
                  </div>
                  <div className="mt-1">
                    <PercentileBar pct={pct} color={c.color} />
                  </div>
                </td>
              );
            })}
          </tr>
        );
      })}
    </>
  );
}
