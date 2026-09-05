'use client';

/**
 * Four-tab view of the NFL daily pipeline output, mirroring `CfbTabs.tsx`.
 * All data comes from `public/data/nfl/latest.json`.
 *
 * Power Rankings is the 32-team board with each team's preseason rank
 * alongside, filterable by conference or division, plus the eight
 * divisions on one Elo scale. Forecasts is the rest-of-season Monte
 * Carlo — expected wins, division / playoff / #1 seed / conference /
 * Super Bowl odds — with the daily-updating Elo trend chart. This Week is
 * the slate with the model's own line; Grades is the ledger.
 */
import { useEffect, useState } from 'react';
import EloTrendChart from '@/app/components/EloTrendChart';
import SortableThemedTable from '@/app/components/SortableThemedTable';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtNum, fmtPct, fmtPctPrecise, fmtSigned, fmtSimScore, missing } from '@/app/lib/format';
import {
  fmtEloLine, fmtWlt, getNflLatest, matchupLabel, orderedDivisions,
  type NflSlateRow, type NflWindowStats,
} from '@/app/lib/nfl';

const TABS = [
  { slug: 'ratings', label: 'Power Rankings' },
  { slug: 'forecast', label: 'Forecasts' },
  { slug: 'slate', label: 'This Week' },
  { slug: 'grades', label: 'Grades' },
] as const;

type TabSlug = (typeof TABS)[number]['slug'];

const data = getNflLatest();
const TOP_HIGHLIGHT = 12;

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

function fmtDelta(mean: number | undefined, se: number | null | undefined): string {
  if (missing(mean)) return DASH;
  const m = mean as number;
  const head = `${m >= 0 ? '+' : ''}${m.toFixed(4)}`;
  return missing(se) ? head : `${head} ± ${(se as number).toFixed(4)}`;
}

/** Shared pill picker: all 32, a conference, or a division. */
function ScopePills({ active, onSelect }: { active: string; onSelect: (k: string) => void }) {
  const keys = ['All', 'AFC', 'NFC', ...Object.keys(data.divisions ?? {})];
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
          {k === 'All' ? 'All 32' : k}
        </button>
      ))}
    </div>
  );
}

function inScope<T extends { conference: string; division: string }>(rows: T[], scope: string): T[] {
  if (scope === 'All') return rows;
  if (scope === 'AFC' || scope === 'NFC') return rows.filter((r) => r.conference === scope);
  return rows.filter((r) => r.division === scope);
}

const RATINGS_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'Pre', numeric: true },
  { header: 'Team', strong: true },
  { header: 'Division' },
  { header: 'W–L' },
  { header: 'Div W–L' },
  { header: 'Pt diff', numeric: true },
  { header: 'Elo', strong: true, numeric: true },
  { header: 'Δ pre', numeric: true },
];

const DIVISION_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'Division', strong: true },
  { header: 'Avg Elo', strong: true, numeric: true },
  { header: 'Preseason avg', numeric: true },
  { header: 'In top 10', numeric: true },
  { header: 'Best team' },
  { header: 'Worst team' },
];

function fmtElo(v: number | null | undefined): string | number {
  return missing(v) ? DASH : Math.round(v as number);
}

function RatingsTab() {
  const [scope, setScope] = useState('All');
  if (data.ratings.length === 0) return <Empty>Awaiting the first daily run.</Empty>;
  const rows = inScope(data.ratings, scope);
  const divisions = orderedDivisions();

  return (
    <div>
      <ScopePills active={scope} onSelect={setScope} />
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {scope === 'All'
          ? 'All 32 teams by Elo — pick a conference or division to narrow the board.'
          : `${rows.length} teams · # is the league-wide Elo rank.`}
      </p>
      <div className="mt-2">
        <SortableThemedTable
          columns={RATINGS_COLUMNS}
          rows={rows.map((r) => {
            const delta = r.preseason_elo == null ? null : r.elo - r.preseason_elo;
            return {
              key: r.team,
              cells: [
                r.rank,
                r.preseason_rank ?? DASH,
                `${r.team} · ${r.name}`,
                r.division,
                fmtWlt(r.wins, r.losses, r.ties),
                `${r.div_wins}–${r.div_losses}`,
                fmtSigned(r.pts_diff),
                Math.round(r.elo),
                delta == null ? DASH : fmtSigned(Math.round(delta)),
              ],
              values: [
                r.rank, r.preseason_rank, r.name, r.division, r.wins,
                r.div_wins, r.pts_diff, r.elo, delta,
              ],
            };
          })}
          note="Ratings after every game played through the run date; Pre is the team's rank on the morning of the season's first game (after the off-season regression, before any result) and Δ pre the Elo moved since. Records and point differential are season-to-date, regular season. Click any header to re-sort."
        />
      </div>

      <h3 className="pixel m-3 mt-8 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        Divisions by Avg Elo
      </h3>
      <SortableThemedTable
        columns={DIVISION_COLUMNS}
        rows={divisions.map((d, i) => ({
          key: d.name,
          cells: [
            i + 1,
            d.name,
            fmtElo(d.avgElo),
            fmtElo(d.preseasonAvgElo),
            d.top10,
            `${d.bestTeam} (${Math.round(d.bestElo)})`,
            `${d.worstTeam} (${Math.round(d.worstElo)})`,
          ],
          values: [i + 1, d.name, d.avgElo, d.preseasonAvgElo, d.top10, d.bestElo, d.worstElo],
        }))}
        note="One Elo scale across the league, so the gap between divisions is a real gap in expected results: a 25-Elo edge is worth about one point of spread. Preseason avg is the same average on the morning of the season's first game."
      />
    </div>
  );
}

const FORECAST_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'Team', strong: true },
  { header: 'Division' },
  { header: 'W–L' },
  { header: 'Proj W–L', strong: true },
  { header: 'Division', numeric: true },
  { header: 'Playoffs', numeric: true },
  { header: '#1 seed', numeric: true },
  { header: 'Conf. title', numeric: true },
  { header: 'Super Bowl', strong: true, numeric: true },
];

function ForecastTab() {
  const [scope, setScope] = useState('All');
  const teams = data.futures.teams ?? [];
  if (teams.length === 0) {
    return <Empty>No remaining games to simulate.</Empty>;
  }
  const rows = inScope(teams, scope);
  const history = data.elo_history;
  const top = new Set(data.ratings.slice(0, TOP_HIGHLIGHT).map((r) => r.team));
  const series = history
    ? Object.entries(history.teams)
        .filter(([team]) => top.has(team))
        .map(([team, points]) => ({ team, points: points as [string, number][] }))
    : [];
  const sims = data.futures.sims ?? 0;

  return (
    <div>
      <ScopePills active={scope} onSelect={setScope} />
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {data.futures.season} · {data.futures.remaining_games ?? 0} regular-season games left ·{' '}
        {sims.toLocaleString()} season simulations, rerun every day.
      </p>
      <div className="mt-2">
        <SortableThemedTable
          columns={FORECAST_COLUMNS}
          rows={rows.map((t, i) => ({
            key: t.team,
            cells: [
              i + 1,
              `${t.team} · ${t.name}`,
              t.division,
              fmtWlt(t.wins, t.losses, t.ties),
              `${t.exp_wins.toFixed(1)}–${t.exp_losses.toFixed(1)}`,
              fmtPct(t.p_division),
              fmtPct(t.p_playoffs),
              fmtPct(t.p_top_seed),
              fmtPct(t.p_conf),
              fmtPct(t.p_sb),
            ],
            values: [
              i + 1, t.name, t.division, t.wins, t.exp_wins,
              t.p_division, t.p_playoffs, t.p_top_seed, t.p_conf, t.p_sb,
            ],
          }))}
          note={
            <>
              Rest-of-season Monte Carlo: the remaining regular-season games replayed{' '}
              {sims.toLocaleString()} times with live in-sim Elo, then the full seven-team
              bracket — Wild Card at the higher seed, the #1 seed off its bye, the Super Bowl at
              a neutral site. Ordered by Super Bowl odds. Standings ties break on division /
              conference record, then at random; the real tiebreaker ladder is not modeled. A
              playoff game already played is honoured whenever the bracket reproduces it. Click
              any header to re-sort.
            </>
          }
        />
      </div>

      {series.length > 0 && history && (
        <section className="mt-8">
          <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
            Elo Trend — {history.season} top {TOP_HIGHLIGHT}
          </h3>
          <div className="mt-3">
            <EloTrendChart series={series} />
          </div>
          <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
            Each point is a team&apos;s pre-game Elo at that date, opening with the
            post-regression preseason rating; the final point is the live rating as of the{' '}
            {data.run_date} run. The top six by current Elo are highlighted and labeled; the grey
            pack is the rest of the top {TOP_HIGHLIGHT}. Hover any point for the exact value.
          </p>
        </section>
      )}
    </div>
  );
}

const SLATE_COLUMNS = [
  { header: 'Date' },
  { header: 'ET' },
  { header: 'Matchup', strong: true },
  { header: 'Elo (away / home)' },
  { header: 'Model pick', strong: true },
  { header: 'Win prob', numeric: true },
  { header: 'Elo line (home)', numeric: true },
  { header: 'Exp. score' },
  { header: 'Exp. total', numeric: true },
];

function slateRow(r: NflSlateRow) {
  const pickedHome = r.pick === r.home_team;
  const div = r.div_game ? ' (div)' : '';
  return {
    key: r.game_id,
    cells: [
      `${r.weekday.slice(0, 3)} ${r.date}`,
      r.gametime || DASH,
      `${matchupLabel(r)}${div}`,
      `${Math.round(r.elo_away_pre)} / ${Math.round(r.elo_home_pre)}`,
      r.pick,
      fmtPct(r.pick_prob),
      fmtEloLine(r.elo_spread),
      fmtSimScore(r.pred_home_score, r.pred_away_score, pickedHome),
      r.pred_total.toFixed(1),
    ],
    values: [
      r.date, r.gametime, matchupLabel(r), r.elo_home_pre, r.pick, r.pick_prob,
      r.elo_spread, r.pred_total, r.pred_total,
    ],
  };
}

function SlateTab() {
  if (data.slate.length === 0) {
    return <Empty>No games on the {data.run_date} slate.</Empty>;
  }
  return (
    <div className="mt-4">
      <p className="mb-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {data.week_label ?? 'This week'} · {data.slate.length} game{data.slate.length === 1 ? '' : 's'}
        {' '}still to play as of the {data.run_date} run.
      </p>
      <SortableThemedTable
        columns={SLATE_COLUMNS}
        rows={data.slate.map(slateRow)}
        note="Every unplayed game of the next NFL week. Win probability straight from Elo with the home edge applied except at neutral sites (marked N) and the rest edge for a side off its bye. The Elo line is the model's own spread from the home side — it has never seen a market number, so read it as a betting-blind line to compare against whatever the books are hanging. Expected score is that margin carved out of a matchup-specific total from each team's recent points for/against, winner's points first, at one decimal — an average, not a literal final. Picks lock the first morning a game appears on a slate."
      />
    </div>
  );
}

const GRADE_COLUMNS = [
  { header: '' },
  { header: 'Date' },
  { header: 'Game' },
  { header: 'Pick' },
  { header: 'Elo line' },
  { header: 'Predicted' },
  { header: 'Final (home first)' },
  { header: 'Log-loss' },
];

const TRACKER_COLUMNS = [
  { header: 'Window' },
  { header: 'Record' },
  { header: 'Accuracy', strong: true },
  { header: 'Log-loss' },
  { header: 'Brier' },
  { header: 'Δ LL vs always-home (± SE)' },
];

function trackerRow(label: string, s: NflWindowStats | undefined) {
  if (!s || !s.graded) {
    return { key: label, cells: [label, DASH, DASH, DASH, DASH, DASH] };
  }
  const correct = s.correct ?? 0;
  const ties = s.ties ?? 0;
  return {
    key: label,
    cells: [
      label,
      fmtWlt(correct, s.graded - correct - ties, ties),
      fmtPctPrecise(s.accuracy),
      fmtNum(s.log_loss, 4),
      fmtNum(s.brier),
      fmtDelta(s.d_ll_mean, s.d_ll_se),
    ],
  };
}

function GradesTab() {
  const ledger = data.ledger;
  if (!ledger || !ledger.graded) {
    return (
      <Empty>
        Nothing graded yet — grading starts the morning after the first slate&apos;s games
        are played.
      </Empty>
    );
  }
  const recent = data.graded_recent ?? [];
  const correct = recent.filter((g) => g.pick_correct).length;
  const weeks = Object.entries(ledger.by_week ?? {});

  return (
    <div>
      <div className="mt-4">
        <ThemedTable
          columns={TRACKER_COLUMNS}
          rows={[
            trackerRow('Last 7 days', ledger.rolling?.['7d']),
            trackerRow('Last 30 days', ledger.rolling?.['30d']),
            trackerRow('Season to date', ledger),
            ...weeks.slice(-4).map(([label, s]) => trackerRow(label, s)),
          ]}
          note="Δ LL is the paired per-game log-loss difference against a fixed always-pick-home forecast on the same games (p = 0.55, a coin flip at neutral sites), ± one standard error; negative = beating the baseline. Log-loss reference points: 0.693 is a coin flip, 0.691 is always-pick-home; the tuned engine's 2024–25 holdout ran 0.624 over 570 games, and the closing line runs about 0.61. A tie counts as a wrong pick and scores both halves."
        />
      </div>

      {recent.length > 0 && (
        <>
          <p className="mt-6 text-[14px]" style={{ color: 'var(--th-ink)' }}>
            <b>
              Past 7 days: {correct}/{recent.length} correct ({fmtPct(correct / recent.length)})
            </b>
          </p>
          <div className="mt-3">
            <ThemedTable
              columns={GRADE_COLUMNS}
              rows={recent.map((g) => ({
                key: g.game_id,
                cells: [
                  g.pick_correct ? '✅' : g.tie ? '🤝' : '❌',
                  g.date,
                  matchupLabel(g),
                  `${g.pick} (${fmtPct(g.pick_prob)})`,
                  fmtEloLine(g.elo_spread),
                  fmtSimScore(g.pred_home_score, g.pred_away_score, g.pick === g.home_team),
                  `${g.home_score}–${g.away_score}`,
                  fmtNum(g.log_loss, 2),
                ],
              }))}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default function NflTabs() {
  const [tab, setTab] = useState<TabSlug>(TABS[0].slug);

  // Same `?tab=` convention as the other sport pages: read after mount so
  // the default tab still server-renders for crawlers and no-JS readers.
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

      {tab === 'ratings' && <RatingsTab />}
      {tab === 'forecast' && <ForecastTab />}
      {tab === 'slate' && <SlateTab />}
      {tab === 'grades' && <GradesTab />}
    </div>
  );
}
