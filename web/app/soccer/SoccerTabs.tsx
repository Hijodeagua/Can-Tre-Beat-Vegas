'use client';

/**
 * Three-tab view of the daily club-soccer pipeline output, mirroring
 * `MlbTabs.tsx`. All data comes from `public/data/soccer/latest.json`.
 *
 * Rankings is the tab this page exists for — the cross-league table the
 * MLB page has no equivalent of, because MLB is one league and soccer is
 * ten sharing one Elo scale via the UEFA glue.
 */
import { useEffect, useState } from 'react';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtPct, missing } from '@/app/lib/format';
import {
  getSoccerLatest, orderedLeagueRankings, LEAGUE_ORDER, GLUED_LEAGUES,
  type SoccerSlateRow,
} from '@/app/lib/soccer';

const TABS = [
  { slug: 'rankings', label: 'League Rankings' },
  { slug: 'slate', label: "Today's Slate" },
  { slug: 'ratings', label: 'Club Ratings' },
] as const;

type TabSlug = (typeof TABS)[number]['slug'];

const data = getSoccerLatest();

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="mt-6 rounded-lg border border-dashed p-8 text-center text-[14px]"
      style={{
        borderColor: 'var(--th-border)',
        background: 'var(--th-card)',
        color: 'var(--th-muted)',
      }}
    >
      {children}
    </div>
  );
}

/** "42.7" -> "€42.7m"; null/NaN -> em dash. Squad economics are always
 * stored in EUR millions, so this is the one money formatter this page
 * needs. */
function fmtEurM(value: number | null | undefined, digits = 1): string {
  if (missing(value)) return DASH;
  return `€${(value as number).toFixed(digits)}m`;
}

function fmtNum1(value: number | null | undefined): string {
  if (missing(value)) return DASH;
  return (value as number).toFixed(1);
}

const RANKINGS_COLUMNS = [
  { header: 'League', strong: true },
  { header: 'Div' },
  { header: 'Avg Elo', strong: true },
  { header: 'Avg squad value' },
  { header: 'Avg wage bill' },
  { header: 'Avg squad size' },
  { header: 'Avg age' },
  { header: 'Avg foreigners' },
  { header: 'Avg value/player' },
];

const ELO_RANK_COLUMNS = [
  { header: '#' },
  { header: 'League', strong: true },
  { header: 'Div' },
  { header: 'Avg Elo', strong: true },
];

/** Every UEFA-glued league sorted by avg Elo, highest first — the flat
 * "who's the strongest league" ranking, as opposed to the table below it
 * which stays grouped by country pool. Leagues with no rated clubs yet sort
 * last. MLS is excluded here — see GLUED_LEAGUES. */
function LeagueEloRank() {
  const ranked = [...orderedLeagueRankings()]
    .filter(([key]) => GLUED_LEAGUES.includes(key))
    .sort((a, b) => (b[1].avgElo ?? -Infinity) - (a[1].avgElo ?? -Infinity));
  return (
    <ThemedTable
      columns={ELO_RANK_COLUMNS}
      rows={ranked.map(([key, r], i) => ({
        key,
        cells: [i + 1, r.name, r.tier === 1 ? '1st' : '2nd', r.avgElo == null ? DASH : Math.round(r.avgElo)],
      }))}
    />
  );
}

interface RankedClub {
  team: string;
  league: string;
  elo: number;
  matches: number;
}

/** Every club in every UEFA-glued league's current-season table, flattened
 * into one list — this is what "top/bottom Elo" means across the site,
 * since the UEFA glue keeps those ten leagues on one shared scale. MLS is
 * excluded — its Elo pool never exchanges points with them. */
function allClubsRanked(): RankedClub[] {
  const out: RankedClub[] = [];
  for (const key of GLUED_LEAGUES) {
    const table = data.ratings[key];
    if (!table) continue;
    for (const c of table.clubs) {
      out.push({ team: c.team, league: table.name, elo: c.elo, matches: c.matches });
    }
  }
  return out.sort((a, b) => b.elo - a.elo);
}

const CLUB_ELO_COLUMNS = [
  { header: '#' },
  { header: 'Club', strong: true },
  { header: 'League' },
  { header: 'Elo', strong: true },
];

/** Top-10 or bottom-10 club Elo, with the top/bottom 5 marked by a ★ —
 * one table covers both "top 10" and "top 5" (and the bottom pair) without
 * splitting into four near-duplicate tables. */
function ClubEloExtremes({
  title, clubs, markFirst,
}: {
  title: string;
  clubs: RankedClub[];
  markFirst: number;
}) {
  return (
    <div>
      <h4 className="pixel m-0 text-[10px]" style={{ color: 'var(--th-ink)' }}>
        {title}
      </h4>
      <div className="mt-2">
        <ThemedTable
          columns={CLUB_ELO_COLUMNS}
          rows={clubs.map((c, i) => ({
            key: `${c.team}-${c.league}`,
            cells: [
              i + 1,
              i < markFirst ? `★ ${c.team}` : c.team,
              c.league,
              Math.round(c.elo),
            ],
          }))}
        />
      </div>
    </div>
  );
}

function RankingsTab() {
  const rows = orderedLeagueRankings();
  if (rows.length === 0) return <Empty>Awaiting the first daily run.</Empty>;

  const anySquadStats = rows.some(([, r]) => r.squadStatsSeason);
  const allClubs = allClubsRanked();
  const top10 = allClubs.slice(0, 10);
  const bottom10 = allClubs.slice(-10).reverse();

  return (
    <div>
      <h3 className="pixel m-3 mt-6 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        Leagues by Avg Elo
      </h3>
      <LeagueEloRank />

      <h3 className="pixel m-3 mt-8 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        League Economics
      </h3>
      <ThemedTable
        columns={RANKINGS_COLUMNS}
        rows={rows.map(([key, r]) => ({
          key,
          cells: [
            r.name,
            r.tier === 1 ? '1st' : '2nd',
            r.avgElo == null ? DASH : Math.round(r.avgElo),
            fmtEurM(r.avgSquadValueEurM),
            fmtEurM(r.avgWageBillEurM),
            fmtNum1(r.avgSquadSize),
            fmtNum1(r.avgAge),
            fmtNum1(r.avgForeigners),
            fmtEurM(r.avgValuePerPlayerEurM, 2),
          ],
        }))}
        note={
          <>
            Avg Elo is each league&apos;s current club ratings, averaged — the same scale
            across all ten leagues via the Champions/Europa/Conference League cross-play, so
            a Bundesliga 1550 and a Segunda 1550 mean the same thing. Squad-value and
            wage-bill figures are that league&apos;s most recent{' '}
            <code>market_values</code> upload; squad size, age, foreigners and
            value-per-player come from a newer, still-growing set of uploads and read{' '}
            {DASH} until a league has one.
            {anySquadStats
              ? ' "As of" season varies by league and by column — coverage is being backfilled incrementally, not all at once.'
              : ''}
            {' '}MLS&apos;s Elo is on its own scale — a separate confederation means it
            never plays the ten leagues above and never exchanges rating points with them,
            so its number isn&apos;t comparable to theirs even though it shares this column.
          </>
        }
      />

      <h3 className="pixel m-3 mt-8 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        Highest &amp; Lowest Rated Clubs
      </h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <ClubEloExtremes title="Top 10 (★ = top 5)" clubs={top10} markFirst={5} />
        <ClubEloExtremes title="Bottom 10 (★ = bottom 5)" clubs={bottom10} markFirst={5} />
      </div>
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Every club currently rated across all ten leagues, one shared Elo scale via the UEFA
        glue — not per-league, the whole pool at once. A club&apos;s current-season table only;
        a newly promoted or relegated side with few matches at its new level can still be noisy.
      </p>
    </div>
  );
}

const SLATE_COLUMNS = [
  { header: 'League' },
  { header: 'Match', strong: true },
  { header: 'P(H) / P(D) / P(A)' },
  { header: 'Pick', strong: true },
  { header: 'Sim score' },
];

function leagueLabel(key: string): string {
  return data.ratings[key]?.name ?? key;
}

function SlateTab() {
  if (data.slate.length === 0) {
    return <Empty>No fixtures on the {data.run_date} slate.</Empty>;
  }
  const byLeague = new Map<string, SoccerSlateRow[]>();
  for (const row of data.slate) {
    const list = byLeague.get(row.league) ?? [];
    list.push(row);
    byLeague.set(row.league, list);
  }
  const orderedKeys = LEAGUE_ORDER.filter((k) => byLeague.has(k));

  return (
    <div>
      {orderedKeys.map((key) => {
        const rows = byLeague.get(key)!;
        return (
          <section key={key} className="mt-6">
            <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
              {leagueLabel(key)}
            </h3>
            <div className="mt-2">
              <ThemedTable
                columns={SLATE_COLUMNS}
                rows={rows.map((r) => ({
                  key: `${r.home_team}-${r.away_team}-${r.date}`,
                  cells: [
                    r.date,
                    `${r.home_team} v ${r.away_team}`,
                    `${fmtPct(r.p_H)} / ${fmtPct(r.p_D)} / ${fmtPct(r.p_A)}`,
                    r.pick,
                    `${r.score_home}–${r.score_away}`,
                  ],
                }))}
              />
            </div>
          </section>
        );
      })}
      <p className="mt-4 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        W/D/L outcome model on the Elo gap plus squad-economics differentials (transfer
        spend, squad value, wage bill); simulated score from independent Poisson goal rates.
      </p>
    </div>
  );
}

const RATINGS_COLUMNS = [
  { header: '#' },
  { header: 'Club', strong: true },
  { header: 'Elo', strong: true },
  { header: 'Matches' },
];

function RatingsTab() {
  const keys = LEAGUE_ORDER.filter((k) => data.ratings[k]?.clubs?.length);
  const [league, setLeague] = useState<string>(keys[0] ?? '');
  if (keys.length === 0) return <Empty>Awaiting the first daily run.</Empty>;

  const table = data.ratings[league];
  const sorted = [...table.clubs].sort((a, b) => b.elo - a.elo);

  return (
    <div>
      <div className="mt-4 flex flex-wrap gap-2">
        {keys.map((k) => (
          <button
            key={k}
            onClick={() => setLeague(k)}
            aria-pressed={league === k}
            className={`rounded-full px-3 py-1 text-[13px] ${
              league === k ? 'font-semibold' : 'hover:bg-slate-100'
            }`}
            style={
              league === k
                ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
                : { color: 'var(--th-muted)', border: '1px solid var(--th-border)' }
            }
          >
            {data.ratings[k].name}
          </button>
        ))}
      </div>
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        As of {table.asOfSeason} — {sorted.length} clubs.
      </p>
      <div className="mt-2">
        <ThemedTable
          columns={RATINGS_COLUMNS}
          rows={sorted.map((c, i) => ({
            key: c.team,
            cells: [i + 1, c.team, Math.round(c.elo), c.matches],
          }))}
        />
      </div>
    </div>
  );
}

export default function SoccerTabs() {
  const [tab, setTab] = useState<TabSlug>(TABS[0].slug);

  // Same `?tab=` convention as MlbTabs: read after mount so the default tab
  // still server-renders for crawlers and no-JS readers.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get('tab');
    const match = TABS.find((t) => t.slug === requested);
    if (match) setTab(match.slug);
  }, []);

  const select = (slug: TabSlug) => {
    setTab(slug);
    const url = slug === TABS[0].slug ? window.location.pathname : `?tab=${slug}`;
    window.history.replaceState(null, '', url);
  };

  return (
    <div className="mt-6">
      <div
        className="flex flex-wrap gap-2 border-b pb-2"
        style={{ borderColor: 'var(--th-border)' }}
      >
        {TABS.map((t) => {
          const active = tab === t.slug;
          return (
            <button
              key={t.slug}
              onClick={() => select(t.slug)}
              aria-pressed={active}
              className={`rounded-md px-3 py-1.5 text-[14px] ${
                active ? 'font-semibold' : 'hover:bg-slate-100'
              }`}
              style={
                active
                  ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
                  : { color: 'var(--th-muted)' }
              }
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === 'rankings' && <RankingsTab />}
      {tab === 'slate' && <SlateTab />}
      {tab === 'ratings' && <RatingsTab />}
    </div>
  );
}
