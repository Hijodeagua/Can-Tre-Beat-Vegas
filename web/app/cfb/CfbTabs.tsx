'use client';

/**
 * Four-tab view of the daily college-football pipeline output, mirroring
 * `MlbTabs.tsx` and `SoccerTabs.tsx`. All data comes from
 * `public/data/cfb/latest.json`.
 *
 * Top 25 is the tab this page exists for — the Elo board, with the
 * conference-strength table underneath it (the college analogue of the
 * soccer league rankings). Forecasts is the rest-of-season Monte Carlo:
 * expected wins, bowl / undefeated odds and the conference title races,
 * plus the daily-updating Elo trend chart for the top 25.
 */
import { useEffect, useState } from 'react';
import EloTrendChart from '@/app/components/EloTrendChart';
import SortableThemedTable from '@/app/components/SortableThemedTable';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtNum, fmtPct, fmtPctPrecise, fmtSigned, fmtSimScore, missing } from '@/app/lib/format';
import {
  getCfbLatest, matchupLabel, orderedConferences,
  type CfbSlateRow, type CfbWindowStats,
} from '@/app/lib/cfb';

const TABS = [
  { slug: 'top25', label: 'Top 25' },
  { slug: 'forecast', label: 'Forecasts' },
  { slug: 'slate', label: 'This Week' },
  { slug: 'grades', label: 'Grades' },
] as const;

type TabSlug = (typeof TABS)[number]['slug'];

const data = getCfbLatest();

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

/** Shared pill picker — Top 25 and Forecasts both filter by conference. */
function ConferencePills({
  active, onSelect,
}: {
  active: string;
  onSelect: (k: string) => void;
}) {
  const keys = ['All', ...orderedConferences().map((c) => c.name)];
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
          {k === 'All' ? 'All FBS' : (data.conferences[k]?.short ?? k)}
        </button>
      ))}
    </div>
  );
}

const RATINGS_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'Team', strong: true },
  { header: 'Conf' },
  { header: 'W–L' },
  { header: 'Conf W–L' },
  { header: 'Pt diff', numeric: true },
  { header: 'Elo', strong: true, numeric: true },
];

const CONFERENCE_COLUMNS = [
  { header: '#', numeric: true },
  { header: 'Conference', strong: true },
  { header: 'Teams', numeric: true },
  { header: 'Avg Elo', strong: true, numeric: true },
  { header: 'Top-4 Elo', numeric: true },
  { header: 'Bottom-4 Elo', numeric: true },
  { header: 'Best team' },
];

function Top25Tab() {
  const [conf, setConf] = useState('All');
  if (data.ratings.length === 0) return <Empty>Awaiting the first daily run.</Empty>;

  const rows = conf === 'All'
    ? data.ratings.slice(0, data.top_n)
    : data.ratings.filter((r) => r.conference === conf);
  const conferences = orderedConferences();

  return (
    <div>
      <ConferencePills active={conf} onSelect={setConf} />
      <p className="mt-3 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {conf === 'All'
          ? `Top ${data.top_n} of ${data.ratings.length} FBS programs by Elo — pick a conference for its full board.`
          : `${rows.length} programs · # is the national Elo rank.`}
      </p>
      <div className="mt-2">
        <SortableThemedTable
          columns={RATINGS_COLUMNS}
          rows={rows.map((r) => ({
            key: r.team,
            cells: [
              r.rank,
              r.team,
              r.conference_short,
              `${r.wins}–${r.losses}`,
              `${r.conf_wins}–${r.conf_losses}`,
              fmtSigned(r.pts_diff),
              Math.round(r.elo),
            ],
            values: [r.rank, r.team, r.conference_short, r.wins, r.conf_wins, r.pts_diff, r.elo],
          }))}
          note="Ratings after every game played through the run date; records and point differential are season-to-date (regular season). Elo is on one scale across all of FBS. Click any header to re-sort."
        />
      </div>

      <h3 className="pixel m-3 mt-8 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        Conferences by Avg Elo
      </h3>
      <SortableThemedTable
        columns={CONFERENCE_COLUMNS}
        rows={conferences.map((c, i) => ({
          key: c.name,
          cells: [
            i + 1,
            c.name,
            c.teams,
            Math.round(c.avgElo),
            c.top4Elo == null ? DASH : Math.round(c.top4Elo),
            c.bottom4Elo == null ? DASH : Math.round(c.bottom4Elo),
            `${c.bestTeam} (${Math.round(c.bestElo)})`,
          ],
          values: [i + 1, c.name, c.teams, c.avgElo, c.top4Elo, c.bottom4Elo, c.bestTeam],
        }))}
        note="Top-4 / Bottom-4 are the mean Elo of each conference's four strongest and four weakest programs — its ceiling and its floor next to the average. Season regression pulls each program toward its conference mean every August (75% conference mean, 25% the 1500 base), which is what keeps a Sun Belt 1500 and an SEC 1500 meaning the same thing."
      />
    </div>
  );
}

const FORECAST_COLUMNS = [
  { header: 'Proj', numeric: true },
  { header: 'Team', strong: true },
  { header: 'Conf' },
  { header: 'W–L' },
  { header: 'Proj W', strong: true, numeric: true },
  { header: 'Bowl', numeric: true },
  { header: 'Undefeated', numeric: true },
  { header: 'CCG', numeric: true },
  { header: 'Conf title', strong: true, numeric: true },
];

function ForecastTab() {
  const [conf, setConf] = useState('All');
  const teams = data.futures.teams ?? [];
  if (teams.length === 0) {
    return <Empty>No remaining games to simulate.</Empty>;
  }
  const rows = conf === 'All'
    ? teams.slice(0, data.top_n)
    : teams.filter((t) => t.conference === conf);
  const history = data.elo_history;
  const top = new Set(data.ratings.slice(0, data.top_n).map((r) => r.team));
  const series = history
    ? Object.entries(history.teams)
        .filter(([team]) => top.has(team))
        .map(([team, points]) => ({ team, points: points as [string, number][] }))
    : [];
  const sims = data.futures.sims ?? 0;

  return (
    <div>
      <ConferencePills active={conf} onSelect={setConf} />
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
              t.team,
              data.conferences[t.conference ?? '']?.short ?? (t.conference ?? DASH),
              `${t.wins}–${t.losses}`,
              `${t.exp_wins.toFixed(1)}–${t.exp_losses.toFixed(1)}`,
              fmtPct(t.p_bowl),
              fmtPct(t.p_undefeated),
              t.p_ccg == null ? DASH : fmtPct(t.p_ccg),
              t.p_conf_title == null ? DASH : fmtPct(t.p_conf_title),
            ],
            values: [
              i + 1, t.team, t.conference, t.wins, t.exp_wins,
              t.p_bowl, t.p_undefeated, t.p_ccg, t.p_conf_title,
            ],
          }))}
          note={
            <>
              Rest-of-season Monte Carlo: the remaining regular-season games replayed{' '}
              {sims.toLocaleString()} times with live in-sim Elo. Proj orders teams by
              expected wins; Bowl = 6+ wins; CCG = reaching the conference championship game
              (top two by conference record, ties random); the title game is simulated at a
              neutral site. The 12-team playoff is a committee pick and is deliberately not
              modeled — independents show a dash. Click any header to re-sort.
            </>
          }
        />
      </div>

      {series.length > 0 && history && (
        <section className="mt-8">
          <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
            Elo Trend — {history.season} top {data.top_n}
          </h3>
          <div className="mt-3">
            <EloTrendChart series={series} />
          </div>
          <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
            Each point is a program&apos;s pre-game Elo at that date, opening with the
            post-regression preseason rating; the final point is the live rating as of the{' '}
            {data.run_date} run. The top six by current Elo are highlighted and labeled; the
            grey pack is the rest of the top {data.top_n}. Hover any point for the exact value.
          </p>
        </section>
      )}
    </div>
  );
}

const SLATE_COLUMNS = [
  { header: 'Date' },
  { header: 'Matchup', strong: true },
  { header: 'Elo (away / home)' },
  { header: 'Model pick', strong: true },
  { header: 'Win prob', numeric: true },
  { header: 'Exp. score' },
  { header: 'Exp. total', numeric: true },
];

function slateRow(r: CfbSlateRow) {
  const pickedHome = r.pick === r.home_team;
  const fcs = r.home_fcs || r.away_fcs ? ' (FCS)' : '';
  return {
    key: String(r.game_id),
    cells: [
      r.date,
      `${matchupLabel(r)}${fcs}`,
      `${Math.round(r.elo_away_pre)} / ${Math.round(r.elo_home_pre)}`,
      r.pick,
      fmtPct(r.pick_prob),
      fmtSimScore(r.pred_home_score, r.pred_away_score, pickedHome),
      r.pred_total.toFixed(1),
    ],
    values: [r.date, matchupLabel(r), r.elo_home_pre, r.pick, r.pick_prob, r.pred_total, r.pred_total],
  };
}

function SlateTab() {
  if (data.slate.length === 0) {
    return <Empty>No FBS games on the {data.run_date} slate.</Empty>;
  }
  return (
    <div className="mt-4">
      <SortableThemedTable
        columns={SLATE_COLUMNS}
        rows={data.slate.map(slateRow)}
        note="Games kicking off within two days of the run date (Eastern). Win probability straight from Elo with the home edge applied except at neutral sites (marked N). Expected score is the Elo-implied margin carved out of a matchup-specific total from each program's recent points for/against, winner's points first, at one decimal — an average, not a literal final. FCS opponents are one pooled rating, so those lines are the least informative on the board."
      />
    </div>
  );
}

const GRADE_COLUMNS = [
  { header: '' },
  { header: 'Date' },
  { header: 'Game' },
  { header: 'Pick' },
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

function trackerRow(label: string, s: CfbWindowStats | undefined) {
  if (!s || !s.graded) {
    return { key: label, cells: [label, DASH, DASH, DASH, DASH, DASH] };
  }
  return {
    key: label,
    cells: [
      label,
      `${s.correct}–${s.graded - (s.correct ?? 0)}`,
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

  return (
    <div>
      <div className="mt-4">
        <ThemedTable
          columns={TRACKER_COLUMNS}
          rows={[
            trackerRow('Last 7 days', ledger.rolling?.['7d']),
            trackerRow('Last 30 days', ledger.rolling?.['30d']),
            trackerRow('Season to date', ledger),
            trackerRow('Season, FBS vs. FBS only', ledger.fbs_only),
          ]}
          note="Δ LL is the paired per-game log-loss difference against a fixed always-pick-home forecast on the same games (p = 0.632, a coin flip at neutral sites), ± one standard error; negative = beating the baseline. Log-loss reference points: 0.693 is a coin flip; the tuned engine's 2024–25 holdout ran 0.499 across all FBS-involved games. Hit rate is inflated by FBS-vs-FCS games, which is why the FBS-only row exists."
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
                key: String(g.game_id),
                cells: [
                  g.pick_correct ? '✅' : '❌',
                  g.date,
                  matchupLabel(g),
                  `${g.pick} (${fmtPct(g.pick_prob)})`,
                  fmtSimScore(g.pred_home_score, g.pred_away_score, g.pick === g.home_team),
                  `${g.home_points}–${g.away_points}`,
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

export default function CfbTabs() {
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

      {tab === 'top25' && <Top25Tab />}
      {tab === 'forecast' && <ForecastTab />}
      {tab === 'slate' && <SlateTab />}
      {tab === 'grades' && <GradesTab />}
    </div>
  );
}
