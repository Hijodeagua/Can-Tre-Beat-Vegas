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
  ordinal, percentile, PROFILE_STATS, similarPlayers, STATS, statValue,
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

/** "Goals" -> "goals" inside a sentence; "xG" and "npxG + xA" are left alone. */
function plain(label: string): string {
  return /^[A-Z][a-z -]+$/.test(label) ? label.toLowerCase() : label;
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

  // The first column is the subject: it gets the written summary and the
  // similar-players search. Both always read per-90 percentiles.
  const subject = columns[0];
  const report = useMemo(() => {
    if (!subject || !subject.group || subject.pool.length < MIN_POOL) return null;
    const ranked = STATS
      .filter((s) => !s.noRank && !s.lowerBetter && s.key !== 'min')
      .map((stat) => ({ stat, pct: percentile(stat, subject.player, subject.pool, 'p90') }))
      .filter((r): r is { stat: (typeof STATS)[number]; pct: number } => r.pct !== null)
      .sort((a, b) => b.pct - a.pct);
    return { top: ranked.slice(0, 3), low: ranked[ranked.length - 1] };
  }, [subject]);
  const similar = useMemo(() => {
    if (!subject || !subject.group || subject.pool.length < MIN_POOL) return null;
    return similarPlayers(subject.player, subject.pool);
  }, [subject]);

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

      {subject && report && report.top.length > 0 && subject.group && (
        <p
          className="mt-5 rounded-lg border px-4 py-3 text-[15px] leading-normal"
          style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-ink)' }}
        >
          <b>{subject.player.name}</b> is in the{' '}
          {report.top.map((r, i) => (
            <span key={r.stat.key}>
              {i > 0 && (i === report.top.length - 1 ? ' and ' : ', ')}
              <b>{ordinal(r.pct)}{i === 0 ? ' percentile' : ''}</b>
              {i === 0 ? ` of ${GROUP_LABEL[subject.group as PositionGroup].toLowerCase()}` : ''} for{' '}
              {plain(r.stat.label)}
            </span>
          ))}
          {report.low && report.low.pct < report.top[report.top.length - 1].pct
            ? `; lowest is ${plain(report.low.stat.label)} (${ordinal(report.low.pct)}).`
            : '.'}{' '}
          <span className="text-[13px]" style={{ color: 'var(--th-muted)' }}>
            Per 90, against {subject.pool.length}{' '}
            {GROUP_LABEL[subject.group as PositionGroup].toLowerCase()} with {subject.floor}+ minutes
            in {scope === 'top5' ? 'the top 5 leagues' : leagueName(subject.player.league)},{' '}
            {subject.player.seasonLabel}.
          </span>
        </p>
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

      {subject && similar && similar.similar.length > 0 && (
        <section className="mt-8">
          <h3 className="pixel m-0 text-[10px]" style={{ color: 'var(--th-ink)' }}>
            PLAYERS LIKE {subject.player.name.toUpperCase()}
          </h3>
          <p className="mt-2 text-[12px] leading-normal" style={{ color: 'var(--th-muted)' }}>
            Found from the percentiles: each number below is that player’s per-90 percentile
            among the same {subject.pool.length} peers, and the match score is 100 minus the
            typical gap to {subject.player.name}’s row. It measures a similar shape of game, not
            equal quality.
          </p>
          <div
            className="mt-3 overflow-x-auto rounded-lg border"
            style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
          >
            <table className="w-full border-collapse text-[13px]" style={{ color: 'var(--th-ink)' }}>
              <thead>
                <tr style={{ background: 'var(--th-head-bg)', color: 'var(--th-head-ink)' }}>
                  <th scope="col" className="px-3 py-2 text-left text-[12px] font-medium">Player</th>
                  <th scope="col" className="px-2 py-2 text-right text-[12px] font-medium">Match</th>
                  {PROFILE_STATS.map((stat) => (
                    <th
                      key={stat.key}
                      scope="col"
                      title={stat.hint ?? stat.label}
                      className="px-2 py-2 text-right text-[12px] font-medium"
                    >
                      {stat.short ?? stat.label}
                    </th>
                  ))}
                  <th scope="col" aria-label="Compare" className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                <ProfileRow
                  name={subject.player.name}
                  sub={`${subject.player.team} · ${leagueName(subject.player.league)}`}
                  profile={similar.subject}
                  color={subject.color}
                  highlight
                />
                {similar.similar.map(({ player, score, profile }) => {
                  const used = taken.has(player.key);
                  const full = keys.length >= MAX_PLAYERS;
                  return (
                    <ProfileRow
                      key={player.key}
                      name={player.name}
                      sub={`${player.team} · ${leagueName(player.league)} · ${player.min.toLocaleString()}′`}
                      profile={profile}
                      score={score}
                      color={subject.color}
                      action={
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
                      }
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
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

/**
 * One line of the similar-players table: a percentile per profile stat,
 * each shaded by its size so two rows with the same shape look alike at a
 * glance. The shade is the subject's series colour at pct% strength.
 */
function ProfileRow({
  name, sub, profile, score, color, action, highlight,
}: {
  name: string;
  sub: string;
  profile: (number | null)[];
  score?: number;
  color: string;
  action?: React.ReactNode;
  highlight?: boolean;
}) {
  const border = { borderColor: 'var(--th-border)' };
  return (
    <tr style={highlight ? { background: 'var(--th-zebra)' } : undefined}>
      <th scope="row" className="min-w-[170px] border-t px-3 py-2 text-left font-normal" style={border}>
        <div className="text-[14px] font-semibold">{name}</div>
        <div className="text-[12px]" style={{ color: 'var(--th-muted)' }}>{sub}</div>
      </th>
      <td className="border-t px-2 py-2 text-right font-semibold tabular-nums" style={border}>
        {score ?? ''}
      </td>
      {profile.map((pct, i) => (
        <td key={i} className="border-t px-1 py-2 text-right tabular-nums" style={border}>
          <span
            className="inline-block min-w-[34px] rounded-sm px-1.5 py-0.5 text-center"
            style={{
              background: pct === null
                ? 'transparent'
                : `color-mix(in srgb, ${color} ${Math.round(pct * 0.45)}%, transparent)`,
            }}
          >
            {pct ?? '—'}
          </span>
        </td>
      ))}
      <td className="border-t px-3 py-2 text-right" style={border}>{action}</td>
    </tr>
  );
}
