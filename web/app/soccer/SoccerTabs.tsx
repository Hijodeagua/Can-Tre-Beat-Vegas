'use client';

/**
 * Four-tab view of the daily club-soccer pipeline output, mirroring
 * `MlbTabs.tsx`. All data comes from `public/data/soccer/latest.json`.
 *
 * Rankings is the tab this page exists for — the cross-league table the
 * MLB page has no equivalent of, because MLB is one league and soccer is
 * ten sharing one Elo scale via the UEFA glue. Forecasts is the
 * Opta-style rest-of-season view: projected final table per top flight
 * with Title / UCL / UEL / Relegation odds, plus the daily-updating Elo
 * trend chart.
 */
import { useEffect, useState } from 'react';
import EloTrendChart from '@/app/components/EloTrendChart';
import SortableThemedTable from '@/app/components/SortableThemedTable';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtPct, missing } from '@/app/lib/format';
import {
  getSoccerLatest, orderedLeagueRankings, LEAGUE_ORDER, GLUED_LEAGUES,
  comparableElo, comparableEloBands, type SoccerSlateRow,
} from '@/app/lib/soccer';

const TABS = [
  { slug: 'rankings', label: 'League Rankings' },
  { slug: 'forecast', label: 'Forecasts' },
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
  { header: 'Avg Elo', strong: true, numeric: true },
  { header: 'Avg squad value', numeric: true },
  { header: 'Top-3 value', numeric: true },
  { header: 'Avg value/player', numeric: true },
  { header: 'Top-3 value/player', numeric: true },
  { header: 'Avg squad size', numeric: true },
  { header: 'Avg age', numeric: true },
  { header: 'Avg foreigners', numeric: true },
];

const ELO_RANK_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'League', strong: true },
  { header: 'Div' },
  { header: 'Avg Elo', strong: true, numeric: true },
  { header: 'Top-4 Avg Elo', numeric: true },
  { header: 'Mid-10 Avg Elo', numeric: true },
  { header: 'Bottom-4 Avg Elo', numeric: true },
];

/** Every league with a cross-league-comparable Elo, highest first — the
 * flat "who's the strongest league" ranking, as opposed to the table below
 * it which stays grouped by country pool. A glued league contributes its
 * own average; an unglued one (MLS today) contributes its squad-value
 * anchor when one has been fit — see `comparableElo`. A league with
 * neither (no rated clubs yet, or an unglued league with no anchor) sorts
 * last and is marked with a dagger either way, so "anchored" and "not yet
 * placeable" never look identical to a measured row. */
function LeagueEloRank() {
  const ranked = [...orderedLeagueRankings()]
    .sort((a, b) => (comparableElo(b[1]) ?? -Infinity) - (comparableElo(a[1]) ?? -Infinity));
  return (
    <SortableThemedTable
      columns={ELO_RANK_COLUMNS}
      rows={ranked.map(([key, r], i) => {
        const elo = comparableElo(r);
        const bands = comparableEloBands(r);
        const label = r.glued ? r.name : `${r.name}†`;
        return {
          key,
          cells: [
            i + 1,
            label,
            r.tier === 1 ? '1st' : '2nd',
            elo == null ? DASH : Math.round(elo),
            bands.top4 == null ? DASH : Math.round(bands.top4),
            bands.mid10 == null ? DASH : Math.round(bands.mid10),
            bands.bottom4 == null ? DASH : Math.round(bands.bottom4),
          ],
          values: [i + 1, r.name, r.tier, elo, bands.top4, bands.mid10, bands.bottom4],
        };
      })}
      note={
        <>
          Top-4 / Mid-10 / Bottom-4 are the mean Elo of each league&apos;s four strongest
          clubs, the ten clubs centered on the median rank, and its four weakest — the
          ceiling, the midtable, and the floor next to the overall average.{' '}
          <b>†</b> = not on the glued scale by shared matches — placed here by a squad-value
          anchor (regression of glued-league club Elo on ln(squad value), applied to this
          league&apos;s own values) rather than measured, so treat it as roughly which tier
          it plays at, not a precise figure; see League Economics below for the fit&apos;s
          R² and residual spread. Click any header to re-sort.
        </>
      }
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
  const anchoredRows = rows.filter(([, r]) => !r.glued && r.anchorMethod);
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
      <SortableThemedTable
        columns={RANKINGS_COLUMNS}
        rows={rows.map(([key, r]) => {
          const elo = comparableElo(r);
          const label = r.glued ? r.name : `${r.name}†`;
          return {
            key,
            cells: [
              label,
              r.tier === 1 ? '1st' : '2nd',
              elo == null ? DASH : Math.round(elo),
              fmtEurM(r.avgSquadValueEurM),
              fmtEurM(r.top3SquadValueEurM),
              fmtEurM(r.avgValuePerPlayerEurM, 2),
              fmtEurM(r.top3ValuePerPlayerEurM, 2),
              fmtNum1(r.avgSquadSize),
              fmtNum1(r.avgAge),
              fmtNum1(r.avgForeigners),
            ],
            values: [
              r.name,
              r.tier,
              elo,
              r.avgSquadValueEurM,
              r.top3SquadValueEurM,
              r.avgValuePerPlayerEurM,
              r.top3ValuePerPlayerEurM,
              r.avgSquadSize,
              r.avgAge,
              r.avgForeigners,
            ],
          };
        })}
        note={
          <>
            Avg Elo is each league&apos;s current club ratings, averaged — the same scale
            across all glued leagues via the Champions/Europa/Conference League cross-play, so
            a Bundesliga 1550 and a Segunda 1550 mean the same thing. Every Top-3 column
            is the mean of that league&apos;s three largest values for the metric — the
            league&apos;s ceiling next to its average. Squad-value figures are that
            league&apos;s most recent <code>market_values</code> upload; squad size, age,
            foreigners and value-per-player come from a newer, still-growing set of
            uploads and read {DASH} until a league has one.
            {anySquadStats
              ? ' "As of" season varies by league and by column — coverage is being backfilled incrementally, not all at once.'
              : ''}
            {anchoredRows.length > 0 && (
              <>
                {' '}
                <b>†</b> {anchoredRows.map(([, r]) => r.name).join(', ')} —
                {anchoredRows.length === 1 ? ' its' : ' their'} Elo above is a squad-value
                anchor, not a measured rating: a different confederation means{' '}
                {anchoredRows.length === 1 ? 'it' : 'they'} never {anchoredRows.length === 1 ? 'plays' : 'play'} the
                glued leagues, so there are no shared matches to calibrate against directly.
                Instead it&apos;s the Elo implied by regressing glued-league club Elo on
                ln(squad value) —{' '}
                {anchoredRows
                  .map(([, r]) => `R²${anchoredRows.length > 1 ? ` ${r.name}` : ''}=${r.anchorR2 ?? DASH} (±${r.anchorResidualStdElo ?? DASH} Elo, ${r.anchorFitClubs ?? DASH} clubs)`)
                  .join('; ')}{' '}
                — good for roughly which tier it plays at, not a precise figure.
              </>
            )}
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
        Every club currently rated across all {GLUED_LEAGUES.length} glued leagues, one shared
        Elo scale via the UEFA glue — not per-league, the whole pool at once. A club&apos;s
        current-season table only;
        a newly promoted or relegated side with few matches at its new level can still be noisy.
      </p>
    </div>
  );
}

/** Shared league-pill picker — Ratings and Forecasts both need one. */
function LeaguePills({
  keys, active, onSelect, labelFor,
}: {
  keys: string[];
  active: string;
  onSelect: (k: string) => void;
  labelFor: (k: string) => string;
}) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
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
          {labelFor(k)}
        </button>
      ))}
    </div>
  );
}

const FORECAST_COLUMNS = [
  { header: 'Proj', numeric: true },
  { header: 'Team', strong: true },
  { header: 'Pts', numeric: true },
  { header: 'xPts', numeric: true },
  { header: 'xPos', numeric: true },
  { header: 'Title', strong: true, numeric: true },
  { header: 'UCL', numeric: true },
  { header: 'UEL', numeric: true },
  { header: 'Rel', numeric: true },
];

function ForecastTab() {
  const keys = LEAGUE_ORDER.filter((k) => data.futures[k]?.clubs?.length);
  const [league, setLeague] = useState<string>(keys[0] ?? '');
  if (keys.length === 0) {
    return <Empty>No league has published fixtures to simulate yet.</Empty>;
  }

  const sim = data.futures[league];
  const clubs = sim.clubs ?? [];
  const history = data.elo_history?.[league];

  return (
    <div>
      <LeaguePills keys={keys} active={league} onSelect={setLeague} labelFor={leagueLabel} />
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {sim.season} · {sim.remaining_matches} matches left ·{' '}
        {sim.sims.toLocaleString()} season simulations, rerun every day.
      </p>
      <div className="mt-2">
        <SortableThemedTable
          columns={FORECAST_COLUMNS}
          rows={clubs.map((c, i) => ({
            key: c.team,
            cells: [
              i + 1,
              c.team,
              c.points,
              c.exp_points.toFixed(1),
              c.exp_position.toFixed(1),
              fmtPct(c.p_title),
              fmtPct(c.p_top4),
              fmtPct(c.p_uel),
              fmtPct(c.p_relegation),
            ],
            values: [
              i + 1,
              c.team,
              c.points,
              c.exp_points,
              c.exp_position,
              c.p_title,
              c.p_top4,
              c.p_uel,
              c.p_relegation,
            ],
          }))}
          note={
            <>
              Rest-of-season Monte Carlo: the remaining fixtures replayed{' '}
              {sim.sims.toLocaleString()} times with live in-sim Elo. Proj orders clubs by
              expected finishing position; Pts is points already banked; xPts / xPos are the
              expected final points and position. Title = 1st, UCL = top 4, UEL = 5th–6th,
              Rel = bottom 3. Click any header to re-sort.
            </>
          }
        />
      </div>

      {history && (
        <section className="mt-8">
          <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
            Elo Trend — {history.season}
          </h3>
          <div className="mt-3">
            <EloTrendChart
              series={Object.entries(history.clubs).map(([team, points]) => ({
                team,
                points: points as [string, number][],
              }))}
            />
          </div>
          <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
            Each point is a club&apos;s pre-match Elo at that date; the final point is the
            live rating as of the {data.run_date} run, so the chart adjusts daily. The top
            six clubs by current Elo are highlighted and labeled; the grey pack is the rest
            of the league. Hover any point for the exact value.
          </p>
        </section>
      )}
    </div>
  );
}

const SLATE_COLUMNS = [
  { header: 'Date' },
  { header: 'Match', strong: true },
  { header: 'P(H) / P(D) / P(A)', numeric: true },
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
              <SortableThemedTable
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
                  values: [
                    r.date,
                    `${r.home_team} v ${r.away_team}`,
                    r.p_H,
                    r.pick,
                    r.score_home + r.score_away,
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
  { header: '#', numeric: true },
  { header: 'Club', strong: true },
  { header: 'Elo', strong: true, numeric: true },
  { header: 'Matches', numeric: true },
];

function RatingsTab() {
  const keys = LEAGUE_ORDER.filter((k) => data.ratings[k]?.clubs?.length);
  const [league, setLeague] = useState<string>(keys[0] ?? '');
  if (keys.length === 0) return <Empty>Awaiting the first daily run.</Empty>;

  const table = data.ratings[league];
  const sorted = [...table.clubs].sort((a, b) => b.elo - a.elo);

  return (
    <div>
      <LeaguePills
        keys={keys}
        active={league}
        onSelect={setLeague}
        labelFor={(k) => data.ratings[k].name}
      />
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        As of {table.asOfSeason} — {sorted.length} clubs.
      </p>
      <div className="mt-2">
        <SortableThemedTable
          columns={RATINGS_COLUMNS}
          rows={sorted.map((c, i) => ({
            key: c.team,
            cells: [i + 1, c.team, Math.round(c.elo), c.matches],
            values: [i + 1, c.team, c.elo, c.matches],
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
      {tab === 'forecast' && <ForecastTab />}
      {tab === 'slate' && <SlateTab />}
      {tab === 'ratings' && <RatingsTab />}
    </div>
  );
}
