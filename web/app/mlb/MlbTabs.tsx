'use client';

/**
 * Four-tab view of the daily MLB pipeline output. All data comes from
 * public/data/mlb/latest.json, which the daily GitHub Actions job rewrites
 * and commits; the site redeploys with each commit, so this is imported at
 * build time like the rest of the static data.
 *
 * Every table here is a `ThemedTable` tinted by the section's sport accent, so
 * the four tabs and the home slate are one component with different rows.
 */
import { useEffect, useState } from 'react';
import MlbSlateTable from '@/app/components/MlbSlateTable';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtNum, fmtPct, fmtPctPrecise, fmtSigned, fmtSimScore } from '@/app/lib/format';
import {
  DIVISION_ORDER, getMlbLatest, gradedLedgerRow, playedGraded, teamName,
} from '@/app/lib/mlb';

const TABS = [
  { slug: 'futures', label: 'Futures' },
  { slug: 'slate', label: "Today's Slate" },
  { slug: 'grade', label: "Yesterday's Grade" },
  { slug: 'history', label: 'History' },
] as const;

type TabSlug = (typeof TABS)[number]['slug'];

const data = getMlbLatest();

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

const AWAITING = (
  <Empty>
    Awaiting the first daily run — the pipeline publishes here every morning once the
    cron job fires.
  </Empty>
);

const FUTURES_COLUMNS = [
  { header: 'Team', strong: true },
  { header: 'W–L' },
  { header: 'Run diff' },
  { header: 'Elo' },
  { header: 'Proj W' },
  { header: 'Div', strong: true },
  { header: 'Playoffs' },
  { header: '#1 seed' },
];

function FuturesTab() {
  if (data.futures.length === 0) return AWAITING;
  return (
    <div>
      {DIVISION_ORDER.map((div) => {
        const teams = data.futures
          .filter((f) => f.division === div)
          .sort((a, b) => b.division_pct - a.division_pct)
          .slice(0, 5);
        if (teams.length === 0) return null;
        const leader = teams[0];
        return (
          <section key={div} className="mt-6">
            <div className="flex items-baseline gap-2">
              <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
                {div}
              </h3>
              <span className="text-[14px]" style={{ color: 'var(--th-faint)' }}>
                {teamName(leader.team)} {fmtPct(leader.division_pct)}
              </span>
            </div>
            <div className="mt-2">
              <ThemedTable
                columns={FUTURES_COLUMNS}
                rows={teams.map((t) => ({
                  key: t.team,
                  cells: [
                    teamName(t.team),
                    t.wins == null || t.losses == null ? DASH : `${t.wins}–${t.losses}`,
                    fmtSigned(t.run_diff),
                    Math.round(t.elo),
                    t.mean_wins.toFixed(1),
                    fmtPct(t.division_pct),
                    fmtPct(t.playoff_pct),
                    fmtPct(t.top_seed_pct),
                  ],
                }))}
              />
            </div>
          </section>
        );
      })}
      <p className="mt-4 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Rest-of-season Monte Carlo from current Elo, live rating updates within each sim.
        12-team playoff format; ties broken at random. Run differential is season-to-date
        as of the last pull. Five teams shown per division.
      </p>
    </div>
  );
}

function SlateTab() {
  if (data.slate.length === 0) {
    return data.date ? <Empty>No MLB games on the {data.date} slate.</Empty> : AWAITING;
  }
  return (
    <div className="mt-4">
      <MlbSlateTable slate={data.slate} />
    </div>
  );
}

const GRADE_COLUMNS = [
  { header: '' },
  { header: 'Game' },
  { header: 'Pick' },
  { header: 'Predicted' },
  { header: 'Actual (home first)' },
];

function GradeTab() {
  const graded = playedGraded();
  if (graded.length === 0) {
    return (
      <Empty>
        Nothing graded yet — grading starts the morning after the first slate is
        published.
      </Empty>
    );
  }
  const correct = graded.filter((g) => g.pick_correct).length;
  const ledger = gradedLedgerRow();

  return (
    <div>
      <p className="mt-4 text-[14px]" style={{ color: 'var(--th-ink)' }}>
        <b>
          {data.graded_date}: {correct}/{graded.length} correct ({fmtPct(correct / graded.length)})
        </b>
        {ledger && (
          <span style={{ color: 'var(--th-muted)' }}>
            {' '}
            · log-loss {fmtNum(ledger.log_loss)} · Brier {fmtNum(ledger.brier)} · running
            accuracy {fmtPctPrecise(ledger.cum_accuracy)}
          </span>
        )}
      </p>
      <div className="mt-3">
        <ThemedTable
          columns={GRADE_COLUMNS}
          rows={graded.map((g) => {
            const pickedHome = g.pick === g.home;
            return {
              key: `${g.away}-${g.home}-${g.game_num}`,
              cells: [
                g.pick_correct ? '✅' : '❌',
                `${teamName(g.away)} @ ${teamName(g.home)}`,
                `${teamName(g.pick)} (${fmtPct(g.pick_prob)})`,
                fmtSimScore(
                  Number(g.pred_home_score?.toFixed(1)),
                  Number(g.pred_away_score?.toFixed(1)),
                  pickedHome,
                ),
                g.home_score == null || g.away_score == null
                  ? DASH
                  : `${g.home_score}–${g.away_score}`,
              ],
            };
          })}
        />
      </div>
    </div>
  );
}

const MODEL_COLUMNS = [
  { header: 'Model' },
  { header: 'Role' },
  { header: 'Window' },
  { header: 'Record' },
  { header: 'Log-loss' },
  { header: 'Δ LL vs always-home (± SE)' },
];

function ModelVersions() {
  const models = data.models ?? [];
  if (models.length === 0) return null;
  return (
    <div className="mt-4">
      <ThemedTable
        columns={MODEL_COLUMNS}
        rows={models.map((m) => ({
          key: m.version,
          cells: [
            m.version,
            m.role,
            `${m.first_date} → ${m.last_date}`,
            `${m.correct}/${m.games} (${fmtPct(m.accuracy)})`,
            fmtNum(m.log_loss),
            m.d_ll_vs_home_mean == null || m.d_ll_vs_home_se == null
              ? DASH
              : `${m.d_ll_vs_home_mean >= 0 ? '+' : ''}${m.d_ll_vs_home_mean.toFixed(4)} ± ${m.d_ll_vs_home_se.toFixed(4)}`,
          ],
        }))}
        note="Each model version is graded in its own bucket — a model change never rewrites or contaminates an earlier record. The history table below is the active model’s ledger."
      />
    </div>
  );
}

const HISTORY_COLUMNS = [
  { header: 'Date' },
  { header: 'Games' },
  { header: 'Accuracy', strong: true },
  { header: 'Log-loss' },
  { header: 'Brier' },
  { header: 'Cum. acc.' },
  { header: 'Cum. LL' },
];

function HistoryTab() {
  if (data.history.length === 0) {
    return <Empty>No graded days yet — one row lands here every morning.</Empty>;
  }
  return (
    <>
      <ModelVersions />
      <div className="mt-4">
        <ThemedTable
          columns={HISTORY_COLUMNS}
          rows={data.history.map((h) => ({
            key: h.date,
            cells: [
              // The per-day snapshot hangs off the date rather than its own
              // column, which keeps the committed history pages reachable
              // without widening the table.
              h.link ? (
                // Plain <a>: the snapshot is a static file in /public, not a
                // Next route, so Link prefetching would 404 in dev.
                <a
                  href={h.link}
                  className="underline underline-offset-2"
                  target="_blank"
                  rel="noreferrer"
                  title={`Snapshot for ${h.date}`}
                >
                  {h.date}
                </a>
              ) : (
                h.date
              ),
              `${h.correct}/${h.games}`,
              fmtPct(h.accuracy),
              fmtNum(h.log_loss),
              fmtNum(h.brier),
              fmtPct(h.cum_accuracy),
              fmtNum(h.cum_log_loss),
            ],
          }))}
          note="Cumulative metrics are game-weighted across all graded days. Log-loss baseline: 0.693 = coin flip, ~0.691 = always pick home."
        />
      </div>
    </>
  );
}

export default function MlbTabs() {
  const [tab, setTab] = useState<TabSlug>(TABS[0].slug);

  /*
   * `?tab=slate` lets the home page's "Futures, grades and history →" land on
   * the slate the reader was already looking at.
   *
   * Read from `window` after mount rather than through `useSearchParams`: that
   * hook opts the whole subtree out of static prerendering, which would ship
   * this page as an empty shell for crawlers and no-JS readers. This way the
   * default tab is fully server-rendered and the query only redirects it.
   */
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get('tab');
    const match = TABS.find((t) => t.slug === requested);
    if (match) setTab(match.slug);
  }, []);

  const select = (slug: TabSlug) => {
    setTab(slug);
    // Keep the URL in step so a shared link reopens the same tab.
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

      {tab === 'futures' && <FuturesTab />}
      {tab === 'slate' && <SlateTab />}
      {tab === 'grade' && <GradeTab />}
      {tab === 'history' && <HistoryTab />}
    </div>
  );
}
