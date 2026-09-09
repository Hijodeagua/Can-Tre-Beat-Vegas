import GameCard from '@/app/components/GameCard';
import DailyModels, { type DailyTab } from '@/app/components/DailyModels';
import ModelCard from '@/app/components/ModelCard';
import StatTile from '@/app/components/StatTile';
import {
  getMeta, getSlate, getSlateSport, getSummary,
  type Sport, type SummaryModel,
} from '@/app/lib/data';
import { getCfbLatest, matchupLabel } from '@/app/lib/cfb';
import { getNflLatest, matchupLabel as nflMatchupLabel } from '@/app/lib/nfl';
import {
  fmtNum, fmtPct, fmtPctPrecise, fmtRecord, fmtRoi, fmtSimScore, fmtTimestamp,
} from '@/app/lib/format';
import { getMlbLatest, gradedLedgerRow, playedGraded } from '@/app/lib/mlb';
import { getSoccerLatest } from '@/app/lib/soccer';
import { accentVars, sportByName, type SportConfig } from '@/app/lib/sports';

/** Soccer fixtures shown inline on the home page before "see the full slate". */
const HOME_SOCCER_ROWS = 12;
/** Same cap for college football — a Saturday can run to 60+ games. */
const HOME_CFB_ROWS = 12;
/** An NFL week is at most 16 games, so the whole slate fits. */
const HOME_NFL_ROWS = 16;

/**
 * The model home: one board for every forecasting model.
 *
 * Everything sport-specific is a lookup into `SPORTS` (identity and copy) and
 * `summary.json` (numbers). Whichever models are in season render their slate
 * inline; the rest fall through to the dashed "up next season" grid. A sport
 * comes online here because its status in the summary flips — not because
 * anything on this page changes.
 */

export default function HomePage() {
  const summary = getSummary();
  const meta = getMeta();
  const slate = getSlate();

  // Only models this site knows how to draw. A sport in the summary with no
  // config is skipped rather than half-rendered.
  const models = summary.models
    .map((model) => ({ model, sport: sportByName(model.sport) }))
    .filter((m): m is { model: SummaryModel; sport: SportConfig } => Boolean(m.sport));

  const inSeason = models.filter((m) => m.model.status === 'in_season');
  const offSeason = models.filter((m) => m.model.status === 'off_season');
  const { overall } = summary;

  // The daily models (MLB, soccer, CFB, NFL) share one section with a sport
  // picker; the odds-feed sports keep a section each when their seasons start.
  const daily = inSeason.filter((m) => m.sport.slateSource !== 'odds');
  const oddsInSeason = inSeason.filter((m) => m.sport.slateSource === 'odds');
  const dailyTabs = daily.map(({ sport, model }) =>
    sport.slateSource === 'mlb'
      ? mlbTab(sport, model)
      : sport.slateSource === 'cfb'
        ? cfbTab(sport)
        : sport.slateSource === 'nfl'
          ? nflTab(sport)
          : soccerTab(sport),
  );

  return (
    <div>
      <h2 className="pixel m-0 text-[20px] leading-[1.4]" style={{ color: 'var(--th-ink)' }}>
        EVERY MODEL,
        <br />
        <span
          className="px-[6px] py-[2px]"
          style={{ background: 'var(--th-highlight)', color: 'var(--th-highlight-ink)' }}
        >
          ONE BOARD
        </span>
      </h2>
      <p
        className="mt-4 max-w-[640px] text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        One page for every forecasting model I run — whichever sport is in season sits on
        top, the rest wait their turn. {meta.label}
      </p>

      <TrackRecord overall={overall} models={models} />

      {dailyTabs.length > 0 && <DailyModels tabs={dailyTabs} />}

      {oddsInSeason.map(({ sport, model }) => (
        <InSeasonSection key={sport.key} sport={sport} model={model} slate={slate} />
      ))}

      {offSeason.length > 0 && (
        <section className="mt-8">
          <h3 className="pixel mb-3 mt-0 text-[12px]" style={{ color: 'var(--th-ink)' }}>
            UP NEXT SEASON
          </h3>
          <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(260px,1fr))]">
            {offSeason.map(({ sport, model }) => (
              <ModelCard
                key={sport.key}
                sport={sport}
                model={model}
                variant="card"
                lead={sport.offseasonLead({
                  seasonStarts: model.season_starts ?? null,
                  pulledAt: getSlateSport(sport.key)?.snapshot?.pulled_at_et ?? null,
                  windowHours: slate.window_hours,
                })}
              />
            ))}
          </div>
        </section>
      )}

      <FooterNote />
    </div>
  );
}

/** The cross-sport strip: one headline number, then a rung per model. */
function TrackRecord({
  overall,
  models,
}: {
  overall: ReturnType<typeof getSummary>['overall'];
  models: { model: SummaryModel; sport: SportConfig }[];
}) {
  const reporting = overall.sports_reporting;
  return (
    <section
      className="mt-6 rounded-lg border p-6"
      style={{
        borderColor: 'var(--th-border)',
        borderLeft: '6px solid var(--accent-vegas)',
        background: 'var(--th-card)',
      }}
    >
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <div
            className="pixel text-[8px] tracking-[0.08em]"
            style={{ color: 'var(--th-faint)' }}
          >
            HIGH SCORE — STRAIGHT UP VS. THE MARKET
          </div>
          <div className="mt-4 flex flex-wrap items-baseline gap-3">
            <span
              className="pixel text-[28px] leading-none"
              style={{ color: 'var(--th-score)' }}
            >
              {fmtPctPrecise(overall.accuracy)}
            </span>
            <span className="text-[14px]" style={{ color: 'var(--th-muted)' }}>
              {fmtRecord(overall.record)} · {overall.games} graded games ·{' '}
              {reporting} sport{reporting === 1 ? '' : 's'} reporting
            </span>
          </div>
        </div>
        <div className="grid gap-2 [grid-template-columns:repeat(3,minmax(96px,1fr))]">
          <StatTile label="Log-loss" value={fmtNum(overall.log_loss)} />
          <StatTile label="Brier" value={fmtNum(overall.brier)} />
          {/* Always a dash: nothing here stakes money. */}
          <StatTile label="ROI" value={fmtRoi()} />
        </div>
      </div>

      <div className="mt-5 flex flex-col gap-2">
        {models.map(({ sport, model }) => (
          <ModelCard key={sport.key} sport={sport} model={model} variant="row" />
        ))}
      </div>

      <p className="mt-4 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Record · accuracy · last graded day, game-weighted across every graded day. A model
        only reports once its picks have been graded. ROI is an em dash — nothing here
        stakes money. Log-loss baseline: 0.693 = coin flip, ~0.691 = always pick home.
        Soccer picks are three-way (win/draw/loss), so its log loss reads against a ~1.10
        baseline and stays out of the blended headline number.
      </p>
    </section>
  );
}

/** The MLB tab of the daily-models section: slate + chips, precomputed
 * server-side so the client component stays a dumb switcher. */
function mlbTab(sport: SportConfig, model: SummaryModel): DailyTab {
  const mlb = getMlbLatest();
  const graded = playedGraded();
  const correct = graded.filter((g) => g.pick_correct).length;
  const ledger = gradedLedgerRow();
  // The chip only wears the accent on a day the model actually won. A green
  // badge reading "6/14 ❌" would celebrate a losing day.
  const won = correct * 2 >= graded.length;

  const chips = [];
  if (graded.length > 0) {
    chips.push({
      text: `Yesterday ${correct}/${graded.length} correct ${won ? '✅' : '❌'}`,
      highlight: won,
    });
  }
  chips.push({
    text: `Running accuracy ${fmtPctPrecise(ledger?.cum_accuracy ?? model.accuracy)}`,
    highlight: false,
  });

  return {
    key: sport.key,
    name: sport.name,
    emoji: sport.emoji,
    accent: sport.accent,
    accentInk: sport.accentInk,
    runDate: mlb.date ?? '',
    blurb: sport.blurb,
    detailHref: sport.href ? `${sport.href}?tab=slate` : null,
    detailLabel: 'Futures, grades and history →',
    chips,
    mlbSlate: mlb.slate,
  };
}

/** The soccer tab: today's fixture window + the rolling tracker chips. */
function soccerTab(sport: SportConfig): DailyTab {
  const soccer = getSoccerLatest();
  const rows = soccer.slate.map((r) => ({
    date: r.date,
    league: soccer.ratings[r.league]?.name ?? r.league,
    home: r.home_team,
    away: r.away_team,
    pH: r.p_H,
    pD: r.p_D,
    pA: r.p_A,
    pick: r.pick,
    scoreH: r.score_home,
    scoreA: r.score_away,
  }));

  const ledger = soccer.ledger as {
    graded: number;
    accuracy?: number;
    rolling?: Record<string, { graded: number; accuracy?: number }>;
  };
  const chips = [];
  const week = ledger.rolling?.['7d'];
  if (week?.graded) {
    chips.push({
      text: `Last 7 days ${fmtPctPrecise(week.accuracy)} (${week.graded} graded)`,
      highlight: false,
    });
  }
  if (ledger.graded) {
    chips.push({
      text: `Running accuracy ${fmtPctPrecise(ledger.accuracy)} · ${ledger.graded} matches`,
      highlight: false,
    });
  }

  return {
    key: sport.key,
    name: sport.name,
    emoji: sport.emoji,
    accent: sport.accent,
    accentInk: sport.accentInk,
    runDate: soccer.run_date,
    blurb: sport.blurb,
    detailHref: sport.href ? `${sport.href}?tab=slate` : null,
    detailLabel: 'Forecasts, ratings and the full slate →',
    chips,
    soccerSlate: rows.slice(0, HOME_SOCCER_ROWS),
    moreCount: Math.max(0, rows.length - HOME_SOCCER_ROWS),
  };
}

/** The college-football tab: the two-day slate + the rolling tracker chips. */
function cfbTab(sport: SportConfig): DailyTab {
  const cfb = getCfbLatest();
  const rows = cfb.slate.map((r) => ({
    key: String(r.game_id),
    date: r.date,
    matchup: `${matchupLabel(r)}${r.home_fcs || r.away_fcs ? ' (FCS)' : ''}`,
    pick: r.pick,
    pickProb: r.pick_prob,
    score: fmtSimScore(r.pred_home_score, r.pred_away_score, r.pick === r.home_team),
  }));

  const ledger = cfb.ledger;
  const chips = [];
  const week = ledger.rolling?.['7d'];
  if (week?.graded) {
    chips.push({
      text: `Last 7 days ${fmtPctPrecise(week.accuracy)} (${week.graded} graded)`,
      highlight: false,
    });
  }
  if (ledger.graded) {
    chips.push({
      text: `Running accuracy ${fmtPctPrecise(ledger.accuracy)} · ${ledger.graded} games`,
      highlight: false,
    });
  }
  const top = cfb.ratings[0];
  if (top) {
    chips.push({ text: `Elo #1 ${top.team} (${Math.round(top.elo)})`, highlight: false });
  }

  return {
    key: sport.key,
    name: sport.name,
    emoji: sport.emoji,
    accent: sport.accent,
    accentInk: sport.accentInk,
    runDate: cfb.run_date,
    blurb: sport.blurb,
    detailHref: sport.href ? `${sport.href}?tab=slate` : null,
    detailLabel: 'Top 25, forecasts and grades →',
    chips,
    gameSlate: rows.slice(0, HOME_CFB_ROWS),
    moreCount: Math.max(0, rows.length - HOME_CFB_ROWS),
    morePage: 'the CFB page',
  };
}

/** The NFL tab: the next week's slate + the rolling tracker chips. */
function nflTab(sport: SportConfig): DailyTab {
  const nfl = getNflLatest();
  const rows = nfl.slate.map((r) => ({
    key: r.game_id,
    date: `${r.weekday.slice(0, 3)} ${r.date}`,
    matchup: `${nflMatchupLabel(r)}${r.div_game ? ' (div)' : ''}`,
    pick: r.pick,
    pickProb: r.pick_prob,
    score: fmtSimScore(r.pred_home_score, r.pred_away_score, r.pick === r.home_team),
  }));

  const ledger = nfl.ledger;
  const chips = [];
  const week = ledger.rolling?.['7d'];
  if (week?.graded) {
    chips.push({
      text: `Last 7 days ${fmtPctPrecise(week.accuracy)} (${week.graded} graded)`,
      highlight: false,
    });
  }
  if (ledger.graded) {
    chips.push({
      text: `Running accuracy ${fmtPctPrecise(ledger.accuracy)} · ${ledger.graded} games`,
      highlight: false,
    });
  }
  const top = nfl.ratings[0];
  if (top) {
    chips.push({ text: `Elo #1 ${top.team} (${Math.round(top.elo)})`, highlight: false });
  }
  const fav = nfl.futures.teams?.[0];
  if (fav) {
    chips.push({ text: `Super Bowl favourite ${fav.team} ${fmtPct(fav.p_sb)}`, highlight: false });
  }

  return {
    key: sport.key,
    name: sport.name,
    emoji: sport.emoji,
    accent: sport.accent,
    accentInk: sport.accentInk,
    runDate: nfl.run_date,
    blurb: sport.blurb,
    detailHref: sport.href ? `${sport.href}?tab=slate` : null,
    detailLabel: 'Power rankings, forecasts and grades →',
    chips,
    gameSlate: rows.slice(0, HOME_NFL_ROWS),
    moreCount: Math.max(0, rows.length - HOME_NFL_ROWS),
    morePage: 'the NFL page',
  };
}

/**
 * One odds-feed model's slate, inline. NFL and NBA reuse this block verbatim
 * in their seasons; the daily models render through DailyModels instead.
 */
function InSeasonSection({
  sport,
  slate,
}: {
  sport: SportConfig;
  model: SummaryModel;
  slate: ReturnType<typeof getSlate>;
}) {
  const oddsSport = slate.sports.find((s) => s.key === sport.key);
  const games = oddsSport?.games.length ?? 0;
  const runDate = slate.generated_at.slice(0, 10);

  return (
    <section className="mt-8" style={accentVars(sport)}>
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-t-lg px-4 py-3"
        style={{ background: 'var(--sport-accent)' }}
      >
        <h3 className="pixel m-0 text-[12px]" style={{ color: 'var(--sport-accent-ink)' }}>
          {sport.emoji} {sport.name.toUpperCase()} DAILY MODEL
        </h3>
        <span
          className="pixel text-[8px] tracking-[0.08em]"
          style={{ color: 'var(--sport-accent-ink)' }}
        >
          {games} GAME{games === 1 ? '' : 'S'}
          {runDate ? ` · RUN ${runDate}` : ''}
        </span>
      </div>

      <p
        className="mt-4 text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        {sport.blurb}
      </p>

      <div className="mt-3">
        {games === 0 ? (
          <EmptySlate hours={slate.window_hours} />
        ) : (
          <OddsSlate sport={oddsSport as Sport} />
        )}
      </div>
    </section>
  );
}

/** The odds-feed sports keep the matchup card — it carries per-book prices. */
function OddsSlate({ sport }: { sport: Sport }) {
  return (
    <div className="flex flex-col gap-4">
      {sport.games.map((game) => (
        <GameCard key={game.game_id} game={game} />
      ))}
    </div>
  );
}

function EmptySlate({ hours }: { hours: number }) {
  return (
    <div
      className="rounded-lg border border-dashed p-8 text-center text-[14px]"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)', color: 'var(--th-muted)' }}
    >
      No games in the next {hours} hours. The slate refreshes twice daily from the odds
      feed.
    </div>
  );
}

/** Provenance for every number above it. */
function FooterNote() {
  const mlb = getMlbLatest();
  const slate = getSlate();
  const feeds = slate.sports
    .filter((s) => s.snapshot?.pulled_at_et)
    .map((s) => `${s.snapshot?.pulled_at_et} ET (${s.name})`);
  const ungraded = mlb.history.length === 0;

  return (
    <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
      MLB data generated {fmtTimestamp(mlb.generated_at)}
      {feeds.length > 0 && <> · odds feed last pulled {feeds.join(' and ')}</>}.
      {/*
        The handoff was written against a snapshot with no graded rows. The
        caveat only holds while that is still true, so it is rendered from the
        data rather than left in as a standing claim.
      */}
      {ungraded && (
        <> The published file has no graded or history rows yet — the schema carries them,
        but no day has been graded.</>
      )}
    </p>
  );
}
