'use client';

/**
 * Four-tab view of the daily MLB pipeline output. All data comes from
 * public/data/mlb/latest.json, which the daily GitHub Actions job rewrites
 * and commits; the site redeploys with each commit, so this is imported at
 * build time like the rest of the static data.
 */
import { useState } from 'react';
import latest from '@/public/data/mlb/latest.json';

type SlateRow = {
  date: string;
  away: string;
  home: string;
  game_num: number;
  /** Probable starters — display only, not a model input. */
  away_sp?: string;
  home_sp?: string;
  /** Matchup-specific expected total runs (recent-form attack/defense). */
  pred_total?: number;
  p_home: number;
  pick: string;
  pick_prob: number;
  pred_home_score: number;
  pred_away_score: number;
};

type FuturesRow = {
  team: string;
  name?: string;
  division?: string;
  wins?: number;
  losses?: number;
  run_diff?: number;
  elo: number;
  mean_wins: number;
  mean_losses: number;
  division_pct: number;
  playoff_pct: number;
  top_seed_pct: number;
};

type GradedRow = SlateRow & {
  played?: boolean;
  home_score?: number | null;
  away_score?: number | null;
  pick_correct?: boolean | null;
};

type HistoryRow = {
  date: string;
  games: number;
  correct: number;
  accuracy: number | null;
  log_loss: number | null;
  brier: number | null;
  avg_margin_err: number | null;
  avg_total_err: number | null;
  cum_accuracy: number | null;
  cum_log_loss: number | null;
  link: string | null;
};

const data = latest as unknown as {
  generated_at: string | null;
  date: string | null;
  slate: SlateRow[];
  futures: FuturesRow[];
  graded_date: string | null;
  graded: GradedRow[] | null;
  history: HistoryRow[];
  team_names: Record<string, string>;
};

const TABS = ['Futures', "Today's Slate", "Yesterday's Grade", 'History'] as const;
type Tab = (typeof TABS)[number];

const DIVISION_ORDER = [
  'AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West',
];

function teamName(code: string): string {
  return data.team_names?.[code] ?? code;
}

function pct(x: number | null | undefined): string {
  if (x == null) return '—';
  if (x > 0 && x < 0.005) return '<1%';
  return `${Math.round(100 * x)}%`;
}

function num(x: number | null | undefined, digits = 3): string {
  return x == null ? '—' : x.toFixed(digits);
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-6 rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

const AWAITING = (
  <Empty>
    Awaiting the first daily run — the pipeline publishes here every morning
    once the cron job fires.
  </Empty>
);

function FuturesTab() {
  if (data.futures.length === 0) return AWAITING;
  return (
    <div>
      {DIVISION_ORDER.map((div) => {
        const teams = data.futures
          .filter((f) => f.division === div)
          .sort((a, b) => b.division_pct - a.division_pct);
        if (teams.length === 0) return null;
        const leader = teams[0];
        return (
          <section key={div} className="mt-6">
            <h3 className="text-lg font-bold">
              {div}
              <span className="ml-2 text-sm font-normal text-slate-400">
                {teamName(leader.team)} {pct(leader.division_pct)}
              </span>
            </h3>
            <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
                    <th className="px-3 py-2">Team</th>
                    <th className="px-3 py-2">W–L</th>
                    <th className="px-3 py-2">Run diff</th>
                    <th className="px-3 py-2">Elo</th>
                    <th className="px-3 py-2">Proj W</th>
                    <th className="px-3 py-2">Div</th>
                    <th className="px-3 py-2">Playoffs</th>
                    <th className="px-3 py-2">#1 seed</th>
                  </tr>
                </thead>
                <tbody>
                  {teams.map((t) => (
                    <tr key={t.team} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 font-semibold">{teamName(t.team)}</td>
                      <td className="px-3 py-2">
                        {t.wins}–{t.losses}
                      </td>
                      <td className="px-3 py-2">
                        {t.run_diff != null && t.run_diff > 0 ? '+' : ''}
                        {t.run_diff}
                      </td>
                      <td className="px-3 py-2">{Math.round(t.elo)}</td>
                      <td className="px-3 py-2">{t.mean_wins.toFixed(1)}</td>
                      <td className="px-3 py-2 font-semibold">{pct(t.division_pct)}</td>
                      <td className="px-3 py-2">{pct(t.playoff_pct)}</td>
                      <td className="px-3 py-2">{pct(t.top_seed_pct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
      <p className="mt-4 text-xs text-slate-400">
        Rest-of-season Monte Carlo from current Elo, live rating updates
        within each sim. 12-team playoff format; ties broken at random. Run
        differential is season-to-date as of the last pull.
      </p>
    </div>
  );
}

function SlateTab() {
  if (data.slate.length === 0) {
    return data.date ? (
      <Empty>No MLB games on the {data.date} slate.</Empty>
    ) : (
      AWAITING
    );
  }
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
            <th className="px-3 py-2">Matchup (home in bold)</th>
            <th className="px-3 py-2">SP (away vs. home)</th>
            <th className="px-3 py-2">Pick</th>
            <th className="px-3 py-2">Win prob</th>
            <th className="px-3 py-2">Exp. runs</th>
            <th className="px-3 py-2">Exp. total</th>
          </tr>
        </thead>
        <tbody>
          {data.slate.map((g) => {
            const pickHome = g.pick === g.home;
            const score = pickHome
              ? `${g.pred_home_score}–${g.pred_away_score}`
              : `${g.pred_away_score}–${g.pred_home_score}`;
            return (
              <tr
                key={`${g.away}-${g.home}-${g.game_num}`}
                className="border-b border-slate-100 last:border-0"
              >
                <td className="px-3 py-2">
                  {teamName(g.away)} @ <b>{teamName(g.home)}</b>
                  {g.game_num > 1 ? ` (G${g.game_num})` : ''}
                </td>
                <td className="px-3 py-2 text-xs text-slate-500">
                  {(g.away_sp || 'TBD') + ' vs. ' + (g.home_sp || 'TBD')}
                </td>
                <td className="px-3 py-2 font-semibold">{teamName(g.pick)}</td>
                <td className="px-3 py-2">{pct(g.pick_prob)}</td>
                <td className="px-3 py-2">{score}</td>
                <td className="px-3 py-2">
                  {g.pred_total != null ? g.pred_total.toFixed(1) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="px-3 py-2 text-xs text-slate-400">
        Probable starters shown for context only — the model is team-level
        Elo and does not use starter identity. Expected total is
        matchup-specific (each club&apos;s recent runs scored/allowed,
        ~20-game half-life, shrunk to league average); the Elo margin is
        carved out of it, so the pick and win probability stay pure Elo.
        Expected runs are shown at one decimal — averages over 10,000 sims
        (winner&apos;s first), not a literal final score; integer rounding
        collapses nearly every game onto 6–3.
      </p>
    </div>
  );
}

function GradeTab() {
  const graded = (data.graded ?? []).filter((g) => g.played);
  if (graded.length === 0) {
    return (
      <Empty>
        Nothing graded yet — grading starts the morning after the first slate
        is published.
      </Empty>
    );
  }
  const correct = graded.filter((g) => g.pick_correct).length;
  const todayRow = data.history.find((h) => h.date === data.graded_date);
  return (
    <div>
      <p className="mt-4 text-sm">
        <b>
          {data.graded_date}: {correct}/{graded.length} correct (
          {pct(correct / graded.length)})
        </b>
        {todayRow && (
          <span className="text-slate-500">
            {' '}
            · log-loss {num(todayRow.log_loss)} · Brier {num(todayRow.brier)} ·
            running accuracy {pct(todayRow.cum_accuracy)}
          </span>
        )}
      </p>
      <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-white">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
              <th className="px-3 py-2"></th>
              <th className="px-3 py-2">Game</th>
              <th className="px-3 py-2">Pick</th>
              <th className="px-3 py-2">Predicted</th>
              <th className="px-3 py-2">Actual (home first)</th>
            </tr>
          </thead>
          <tbody>
            {graded.map((g) => {
              const pickHome = g.pick === g.home;
              const predicted = pickHome
                ? `${g.pred_home_score}–${g.pred_away_score}`
                : `${g.pred_away_score}–${g.pred_home_score}`;
              return (
                <tr
                  key={`${g.away}-${g.home}-${g.game_num}`}
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-2">{g.pick_correct ? '✅' : '❌'}</td>
                  <td className="px-3 py-2">
                    {teamName(g.away)} @ <b>{teamName(g.home)}</b>
                  </td>
                  <td className="px-3 py-2">
                    {teamName(g.pick)} ({pct(g.pick_prob)})
                  </td>
                  <td className="px-3 py-2">{predicted}</td>
                  <td className="px-3 py-2">
                    {g.home_score}–{g.away_score}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HistoryTab() {
  if (data.history.length === 0) {
    return (
      <Empty>No graded days yet — one row lands here every morning.</Empty>
    );
  }
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
            <th className="px-3 py-2">Date</th>
            <th className="px-3 py-2">Games</th>
            <th className="px-3 py-2">Accuracy</th>
            <th className="px-3 py-2">Log-loss</th>
            <th className="px-3 py-2">Brier</th>
            <th className="px-3 py-2">Cum. acc.</th>
            <th className="px-3 py-2">Cum. LL</th>
            <th className="px-3 py-2">Detail</th>
          </tr>
        </thead>
        <tbody>
          {data.history.map((h) => (
            <tr key={h.date} className="border-b border-slate-100 last:border-0">
              <td className="px-3 py-2">{h.date}</td>
              <td className="px-3 py-2">
                {h.correct}/{h.games}
              </td>
              <td className="px-3 py-2 font-semibold">{pct(h.accuracy)}</td>
              <td className="px-3 py-2">{num(h.log_loss)}</td>
              <td className="px-3 py-2">{num(h.brier)}</td>
              <td className="px-3 py-2">{pct(h.cum_accuracy)}</td>
              <td className="px-3 py-2">{num(h.cum_log_loss)}</td>
              <td className="px-3 py-2">
                {h.link ? (
                  // Plain <a>: the snapshot is a static file in /public, not
                  // a Next route, so Link prefetching would 404 in dev.
                  <a
                    href={h.link}
                    className="text-blue-600 hover:underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    view
                  </a>
                ) : (
                  '—'
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-3 py-2 text-xs text-slate-400">
        Cumulative metrics are game-weighted across all graded days. Log-loss
        baseline: 0.693 = coin flip, ~0.691 = always pick home.
      </p>
    </div>
  );
}

export default function MlbTabs() {
  const [tab, setTab] = useState<Tab>(TABS[0]);
  return (
    <div className="mt-6">
      <div className="flex flex-wrap gap-2 border-b border-slate-200 pb-2">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={
              'rounded px-3 py-1.5 text-sm ' +
              (tab === t
                ? 'bg-slate-900 font-semibold text-white'
                : 'text-slate-600 hover:bg-slate-100')
            }
          >
            {t}
          </button>
        ))}
      </div>
      {tab === 'Futures' && <FuturesTab />}
      {tab === "Today's Slate" && <SlateTab />}
      {tab === "Yesterday's Grade" && <GradeTab />}
      {tab === 'History' && <HistoryTab />}
      {data.generated_at && (
        <p className="mt-8 text-xs text-slate-400">
          Data generated {data.generated_at.replace('T', ' ')}
          {data.date ? ` · run date ${data.date}` : ''}
        </p>
      )}
    </div>
  );
}
