'use client';

/**
 * Interconference tab: every FBS-vs-FBS game between two different
 * conferences, this season and last, graded two ways — how each
 * conference did against the model's expectation (wins and margin vs.
 * the Elo-implied numbers), and how well the model itself called these
 * games next to in-conference games. Inter-conference games are the only
 * thing tying the conferences to one scale, so they are where a
 * conference-strength miss would show.
 *
 * The Matchup Explorer picks a conference (and optionally one team) and
 * shows its record against each other conference, with FBS-vs-FCS games
 * (`fcs_games`) in a separate table that never feeds the FBS numbers.
 *
 * Everything is aggregated here from `data.interconference` so the
 * season and game-type filters can pool any combination.
 */
import { useState } from 'react';
import SortableThemedTable from '@/app/components/SortableThemedTable';
import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtNum, fmtPct, fmtPctPrecise } from '@/app/lib/format';
import {
  getCfbLatest, matchupLabel, orderedConferences,
  type CfbConferenceBaseline, type CfbInterconferenceGame,
} from '@/app/lib/cfb';

const data = getCfbLatest();

type SeasonFilter = 'this' | 'last' | 'both';
type TypeFilter = 'all' | 'regular' | 'postseason';

const EPS = 1e-6;

function shortConf(name: string): string {
  return data.conferences[name]?.short ?? name;
}

function signed1(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  const r = Math.round(v * 10) / 10;
  if (r === 0) return '0.0';
  return `${r > 0 ? '+' : ''}${r.toFixed(1)}`;
}

function Pills<T extends string>({
  options, active, onSelect,
}: {
  options: { key: T; label: string }[];
  active: T;
  onSelect: (k: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((o) => (
        <button
          key={o.key}
          onClick={() => onSelect(o.key)}
          aria-pressed={active === o.key}
          className={`rounded-full px-3 py-1 text-[13px] ${
            active === o.key ? 'font-semibold' : 'hover:bg-slate-100'
          }`}
          style={
            active === o.key
              ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
              : { color: 'var(--th-muted)', border: '1px solid var(--th-border)' }
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** One game from one conference's side of it. */
interface Side {
  conf: string;
  opp: string;
  team: string;
  oppTeam: string;
  p: number;          // model's win probability for this side
  won: boolean;
  margin: number;     // this side's points − opponent's
  expMargin: number;  // Elo-implied margin for this side
  correct: boolean;   // the model's pick was right (same for both sides)
}

function sides(g: CfbInterconferenceGame, eloPerPoint: number): [Side, Side] {
  const margin = g.home_points - g.away_points;
  const exp = g.elo_diff / eloPerPoint;
  const correct = (g.p_home >= 0.5 && margin > 0) || (g.p_home < 0.5 && margin < 0);
  return [
    {
      conf: g.home_conference, opp: g.away_conference, team: g.home_team, oppTeam: g.away_team,
      p: g.p_home, won: margin > 0,
      margin, expMargin: exp, correct,
    },
    {
      conf: g.away_conference, opp: g.home_conference, team: g.away_team, oppTeam: g.home_team,
      p: 1 - g.p_home, won: margin < 0,
      margin: -margin, expMargin: -exp, correct,
    },
  ];
}

interface Agg {
  games: number;
  wins: number;
  expWins: number;
  favW: number; favL: number;
  dogW: number; dogL: number;
  marginSum: number;
  expMarginSum: number;
  correct: number;
  llSum: number;
}

function emptyAgg(): Agg {
  return {
    games: 0, wins: 0, expWins: 0, favW: 0, favL: 0, dogW: 0, dogL: 0,
    marginSum: 0, expMarginSum: 0, correct: 0, llSum: 0,
  };
}

function add(a: Agg, s: Side) {
  const p = Math.min(1 - EPS, Math.max(EPS, s.p));
  a.games += 1;
  a.wins += s.won ? 1 : 0;
  a.expWins += s.p;
  if (s.p >= 0.5) {
    if (s.won) a.favW += 1; else a.favL += 1;
  } else if (s.won) a.dogW += 1; else a.dogL += 1;
  a.marginSum += s.margin;
  a.expMarginSum += s.expMargin;
  a.correct += s.correct ? 1 : 0;
  a.llSum += -Math.log(s.won ? p : 1 - p);
}

function gradeGames(games: CfbInterconferenceGame[]) {
  let correct = 0, ll = 0, brier = 0, homeWins = 0;
  for (const g of games) {
    const p = Math.min(1 - EPS, Math.max(EPS, g.p_home));
    const y = g.home_points > g.away_points ? 1 : 0;
    correct += (p >= 0.5) === (y === 1) ? 1 : 0;
    ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    brier += (p - y) ** 2;
    homeWins += y;
  }
  return { games: games.length, correct, ll, brier, homeWins };
}

const SUMMARY_COLUMNS = [
  { header: 'Games' },
  { header: 'Model record' },
  { header: 'Accuracy', strong: true },
  { header: 'Log-loss' },
  { header: 'Brier' },
];

const CONF_COLUMNS = [
  { header: 'Conference', strong: true },
  { header: 'G', numeric: true },
  { header: 'W–L' },
  { header: 'Exp W', numeric: true },
  { header: 'W vs exp', strong: true, numeric: true },
  { header: 'As fav' },
  { header: 'As dog' },
  { header: 'Avg margin', numeric: true },
  { header: 'Exp margin', numeric: true },
  { header: 'Margin vs exp', strong: true, numeric: true },
  { header: 'Model acc.', numeric: true },
  { header: 'Log-loss', numeric: true },
];

const EXPLORER_COLUMNS = [
  { header: 'Opponent', strong: true },
  { header: 'G', numeric: true },
  { header: 'W–L' },
  { header: 'Exp W', numeric: true },
  { header: 'W vs exp', strong: true, numeric: true },
  { header: 'Avg margin', numeric: true },
  { header: 'Margin vs exp', numeric: true },
];

const LOG_COLUMNS = [
  { header: '' },
  { header: 'Date' },
  { header: 'Opponent' },
  { header: 'Opp conf' },
  { header: 'Win prob' },
  { header: 'Result' },
  { header: 'Margin vs exp' },
];

const UPSET_COLUMNS = [
  { header: 'Date' },
  { header: 'Game' },
  { header: 'Confs' },
  { header: 'Winner', strong: true },
  { header: 'Winner win prob' },
  { header: 'Final (home first)' },
];

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="mt-6 rounded-lg border border-dashed p-8 text-center text-[14px]"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-muted)' }}
    >
      {children}
    </div>
  );
}

function H3({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="pixel m-3 mt-8 ml-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
      {children}
    </h3>
  );
}

function explorerRow(label: string, a: Agg) {
  const wVsExp = a.wins - a.expWins;
  const avgM = a.marginSum / a.games;
  const mVsExp = (a.marginSum - a.expMarginSum) / a.games;
  return {
    key: label,
    cells: [
      label, a.games, `${a.wins}–${a.games - a.wins}`, a.expWins.toFixed(1),
      signed1(wVsExp), signed1(avgM), signed1(mVsExp),
    ],
    values: [label, a.games, a.wins, a.expWins, wVsExp, avgM, mVsExp],
  };
}

function logRows(list: { g: CfbInterconferenceGame; s: Side }[]) {
  return list.map(({ g, s }) => ({
    key: String(g.game_id),
    cells: [
      s.won ? '✅' : '❌',
      g.date,
      `${g.neutral ? 'vs.' : s.team === g.home_team ? 'vs.' : '@'} ${s.oppTeam}${g.neutral ? ' (N)' : ''}${
        g.season_type === 'postseason' ? ' · bowl/playoff' : ''}`,
      shortConf(s.opp),
      fmtPct(s.p),
      `${s.won ? 'W' : 'L'} ${Math.max(g.home_points, g.away_points)}–${Math.min(g.home_points, g.away_points)}`,
      signed1(s.margin - s.expMargin),
    ],
  }));
}

/** Pick a conference (and optionally one of its teams) and see the record
 * against every other conference, with FBS-vs-FCS games in their own
 * table — they are never pooled into the FBS numbers. */
function MatchupExplorer({
  games, fcsGames, eloPerPoint,
}: {
  games: CfbInterconferenceGame[];
  fcsGames: CfbInterconferenceGame[];
  eloPerPoint: number;
}) {
  const ranked = orderedConferences().map((c) => c.name);
  const present = new Set(games.flatMap((g) => [g.home_conference, g.away_conference]));
  const confs = [
    ...ranked.filter((c) => present.has(c)),
    ...[...present].filter((c) => !ranked.includes(c)).sort(),
  ];
  const [conf, setConf] = useState(confs[0] ?? '');
  const [team, setTeam] = useState('All');
  const active = confs.includes(conf) ? conf : (confs[0] ?? '');

  // Every FBS side of a game that belongs to the picked conference.
  const fbsSides = games
    .flatMap((g) => sides(g, eloPerPoint).map((s) => ({ g, s })))
    .filter(({ s }) => s.conf === active);
  const fcsSides = fcsGames
    .map((g) => ({ g, s: sides(g, eloPerPoint)[g.home_fcs ? 1 : 0] }))
    .filter(({ s }) => s.conf === active);

  const teams = [...new Set([...fbsSides, ...fcsSides].map(({ s }) => s.team))].sort();
  const pickedTeam = teams.includes(team) ? team : 'All';
  const mine = <T extends { s: Side }>(list: T[]) =>
    pickedTeam === 'All' ? list : list.filter(({ s }) => s.team === pickedTeam);
  const fbs = mine(fbsSides);
  const fcs = mine(fcsSides);

  const byOpp = new Map<string, Agg>();
  const total = emptyAgg();
  for (const { s } of fbs) {
    if (!byOpp.has(s.opp)) byOpp.set(s.opp, emptyAgg());
    add(byOpp.get(s.opp)!, s);
    add(total, s);
  }
  const fcsByConf = new Map<string, Agg>();
  const fcsTotal = emptyAgg();
  for (const { s } of fcs) {
    if (!fcsByConf.has(s.opp)) fcsByConf.set(s.opp, emptyAgg());
    add(fcsByConf.get(s.opp)!, s);
    add(fcsTotal, s);
  }
  const subject = pickedTeam === 'All' ? active : pickedTeam;
  const confOptions = confs.map((c) => ({ key: c, label: shortConf(c) }));

  return (
    <section>
      <H3>Matchup Explorer</H3>
      <Pills options={confOptions} active={active} onSelect={(c) => { setConf(c); setTeam('All'); }} />
      <label className="mt-3 flex items-center gap-2 text-[13px]" style={{ color: 'var(--th-muted)' }}>
        Team
        <select
          value={pickedTeam}
          onChange={(e) => setTeam(e.target.value)}
          className="rounded-md border px-2 py-1 text-[13px]"
          style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-ink)' }}
        >
          <option value="All">All {shortConf(active)} teams</option>
          {teams.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </label>

      <h4 className="pixel m-0 mt-5 text-[10px]" style={{ color: 'var(--th-ink)' }}>
        {subject} vs. FBS conferences
      </h4>
      <div className="mt-2">
        {fbs.length === 0 ? (
          <Empty>No FBS interconference games for {subject} under this filter.</Empty>
        ) : (
          <SortableThemedTable
            columns={EXPLORER_COLUMNS}
            rows={[
              ...[...byOpp.entries()]
                .sort((x, y) => y[1].games - x[1].games)
                .map(([opp, a]) => explorerRow(opp, a)),
              { ...explorerRow('All FBS opponents', total), key: '__total' },
            ]}
            note="Told from the picked conference's (or team's) side. Exp W sums the model's pre-game win probabilities; W vs exp above zero means it won more than its Elo said it should. Margin vs exp is actual minus Elo-implied margin, points per game. FCS games are in their own table below and are not counted here."
          />
        )}
      </div>

      <h4 className="pixel m-0 mt-6 text-[10px]" style={{ color: 'var(--th-ink)' }}>
        {subject} vs. FCS (kept separate)
      </h4>
      <div className="mt-2">
        {fcs.length === 0 ? (
          <Empty>No FCS games for {subject} under this filter.</Empty>
        ) : (
          <SortableThemedTable
            columns={EXPLORER_COLUMNS}
            rows={[
              ...[...fcsByConf.entries()]
                .sort((x, y) => y[1].games - x[1].games)
                .map(([opp, a]) => explorerRow(opp, a)),
              { ...explorerRow('All FCS opponents', fcsTotal), key: '__total' },
            ]}
            note="FBS vs. FCS only, grouped by the FCS opponent's conference. Every FCS team is one pooled rating in the model, so these lines say little about any single FCS program, and they never feed the FBS tables above."
          />
        )}
      </div>

      {pickedTeam !== 'All' && (
        <>
          <h4 className="pixel m-0 mt-6 text-[10px]" style={{ color: 'var(--th-ink)' }}>
            {pickedTeam} game log
          </h4>
          {fbs.length > 0 && (
            <div className="mt-2">
              <ThemedTable columns={LOG_COLUMNS} rows={logRows(fbs)} note="FBS interconference games." />
            </div>
          )}
          {fcs.length > 0 && (
            <div className="mt-3">
              <ThemedTable columns={LOG_COLUMNS} rows={logRows(fcs)} note="FCS games, listed separately." />
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default function InterconferenceTab() {
  const ic = data.interconference;
  const season = data.season;
  const [seasonFilter, setSeasonFilter] = useState<SeasonFilter>('this');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('all');

  if (!ic || ic.games.length === 0) {
    return <Empty>No interconference games graded yet.</Empty>;
  }
  const eloPerPoint = data.params?.elo_per_point || 25;

  const seasonOk = (s: number) =>
    seasonFilter === 'both' ? (s === season || s === season - 1)
      : seasonFilter === 'this' ? s === season : s === season - 1;
  const typeOk = (t: string) => typeFilter === 'all' || t === typeFilter;

  const games = ic.games.filter((g) => seasonOk(g.season) && typeOk(g.season_type));
  const baseline = ic.conference_baseline.filter(
    (b) => seasonOk(b.season) && typeOk(b.season_type),
  );

  const seasonOptions: { key: SeasonFilter; label: string }[] = [
    { key: 'this', label: `This season (${season})` },
    { key: 'last', label: `Last season (${season - 1})` },
    { key: 'both', label: `Both (${season - 1}–${String(season).slice(2)})` },
  ];
  const typeOptions: { key: TypeFilter; label: string }[] = [
    { key: 'all', label: 'All games' },
    { key: 'regular', label: 'Regular season' },
    { key: 'postseason', label: 'Bowls & playoff' },
  ];

  const filters = (
    <div className="mt-4 grid gap-2">
      <Pills options={seasonOptions} active={seasonFilter} onSelect={setSeasonFilter} />
      <Pills options={typeOptions} active={typeFilter} onSelect={setTypeFilter} />
    </div>
  );

  if (games.length === 0) {
    return (
      <div>
        {filters}
        <Empty>No interconference games for this filter yet.</Empty>
      </div>
    );
  }

  // Model-level grading: interconference vs. in-conference.
  const ig = gradeGames(games);
  const bl = baseline.reduce(
    (acc: Omit<CfbConferenceBaseline, 'season' | 'season_type'>, b) => ({
      games: acc.games + b.games,
      correct: acc.correct + b.correct,
      ll_sum: acc.ll_sum + b.ll_sum,
      brier_sum: acc.brier_sum + b.brier_sum,
    }),
    { games: 0, correct: 0, ll_sum: 0, brier_sum: 0 },
  );
  const summaryRow = (label: string, n: number, c: number, ll: number, br: number) => ({
    key: label,
    cells: n
      ? [
          `${label} · ${n}`,
          `${c}–${n - c}`,
          fmtPctPrecise(c / n),
          fmtNum(ll / n, 4),
          fmtNum(br / n),
        ]
      : [`${label} · 0`, DASH, DASH, DASH, DASH],
  });

  // Per-conference and per-pair aggregation.
  const byConf = new Map<string, Agg>();
  for (const g of games) {
    const [h, a] = sides(g, eloPerPoint);
    for (const s of [h, a]) {
      if (!byConf.has(s.conf)) byConf.set(s.conf, emptyAgg());
      add(byConf.get(s.conf)!, s);
    }
  }
  const confRows = [...byConf.entries()]
    .sort((x, y) => (y[1].wins - y[1].expWins) - (x[1].wins - x[1].expWins))
    .map(([conf, a]) => {
      const wVsExp = a.wins - a.expWins;
      const avgM = a.marginSum / a.games;
      const expM = a.expMarginSum / a.games;
      return {
        key: conf,
        cells: [
          conf,
          a.games,
          `${a.wins}–${a.games - a.wins}`,
          a.expWins.toFixed(1),
          signed1(wVsExp),
          `${a.favW}–${a.favL}`,
          `${a.dogW}–${a.dogL}`,
          signed1(avgM),
          signed1(expM),
          signed1(avgM - expM),
          fmtPct(a.correct / a.games),
          fmtNum(a.llSum / a.games, 3),
        ],
        values: [
          conf, a.games, a.wins, a.expWins, wVsExp, a.favW, a.dogW, avgM, expM,
          avgM - expM, a.correct / a.games, a.llSum / a.games,
        ],
      };
    });

  const upsets = games
    .map((g) => {
      const homeWon = g.home_points > g.away_points;
      return { g, winner: homeWon ? g.home_team : g.away_team, p: homeWon ? g.p_home : 1 - g.p_home };
    })
    .filter((u) => u.p < 0.5)
    .sort((x, y) => x.p - y.p)
    .slice(0, 10);

  return (
    <div>
      {filters}

      <H3>Model on Interconference Games</H3>
      <ThemedTable
        columns={SUMMARY_COLUMNS}
        rows={[
          summaryRow('Interconference', ig.games, ig.correct, ig.ll, ig.brier),
          summaryRow('In-conference', bl.games, bl.correct, bl.ll_sum, bl.brier_sum),
        ]}
        note={
          <>
            FBS vs. FBS only — FCS opponents are left out of both rows. Probabilities are the
            Elo replay&apos;s pre-game numbers, the same ones the daily slate is built from, so
            nothing here uses hindsight; parameters were fit on 2005–2023, so both seasons are
            out of sample. If the model&apos;s conference scale were off, it would show up as
            the interconference row grading worse than the in-conference one — compare log-loss
            and Brier rather than accuracy, since non-conference schedules carry more lopsided
            Power-vs-Group-of-5 games that are easy to call. Home teams won{' '}
            {fmtPct(ig.homeWins / ig.games)} of these games.
          </>
        }
      />

      <H3>Conferences vs. the Model</H3>
      <SortableThemedTable
        columns={CONF_COLUMNS}
        rows={confRows}
        note={
          <>
            Each conference&apos;s record against every other conference. Exp W sums the
            model&apos;s pre-game win probabilities, so W vs exp above zero means the
            conference won more than its Elo said it should — the model is underrating it, or
            it was lucky. Margin vs exp is the same idea in points per game (actual margin
            minus the Elo-implied margin at {eloPerPoint.toFixed(1)} Elo per point). As fav /
            As dog split the record by whether the model favored the conference&apos;s team.
            Model acc. and Log-loss grade the model on that conference&apos;s games. Sorted by
            W vs exp; click any header to re-sort.
          </>
        }
      />

      <MatchupExplorer
        games={games}
        fcsGames={(ic.fcs_games ?? []).filter((g) => seasonOk(g.season) && typeOk(g.season_type))}
        eloPerPoint={eloPerPoint}
      />

      {upsets.length > 0 && (
        <>
          <H3>Biggest Interconference Upsets</H3>
          <ThemedTable
            columns={UPSET_COLUMNS}
            rows={upsets.map(({ g, winner, p }) => ({
              key: String(g.game_id),
              cells: [
                g.date,
                `${matchupLabel(g)}${g.season_type === 'postseason' ? ' · bowl/playoff' : ''}`,
                `${shortConf(g.away_conference)} / ${shortConf(g.home_conference)}`,
                winner,
                fmtPct(p),
                `${g.home_points}–${g.away_points}`,
              ],
            }))}
            note="The ten wins the model gave the lowest pre-game chance, under the current filter."
          />
        </>
      )}
    </div>
  );
}
