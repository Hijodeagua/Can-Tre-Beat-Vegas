import Link from 'next/link';
import GameCard from '@/app/components/GameCard';
import MlbSlateTable from '@/app/components/MlbSlateTable';
import ModelCard from '@/app/components/ModelCard';
import StatTile from '@/app/components/StatTile';
import {
  getMeta, getSlate, getSlateSport, getSummary,
  type Sport, type SummaryModel,
} from '@/app/lib/data';
import { fmtNum, fmtPctPrecise, fmtRecord, fmtRoi, fmtTimestamp } from '@/app/lib/format';
import { getMlbLatest, gradedLedgerRow, playedGraded } from '@/app/lib/mlb';
import { accentVars, sportByName, type SportConfig } from '@/app/lib/sports';

/**
 * The model home: one board for every forecasting model.
 *
 * Everything sport-specific is a lookup into `SPORTS` (identity and copy) and
 * `summary.json` (numbers). Whichever models are in season render their slate
 * inline; the rest fall through to the dashed "up next season" grid. When NFL
 * comes online in September its section appears here because its status in the
 * summary flips — not because anything on this page changes.
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

      {inSeason.map(({ sport, model }) => (
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
      </p>
    </section>
  );
}

/**
 * One in-season model's slate, inline. Rendered per in-season sport, so NFL and
 * NBA reuse this block verbatim in their seasons — the only branch is which
 * loader supplies the games.
 */
function InSeasonSection({
  sport,
  model,
  slate,
}: {
  sport: SportConfig;
  model: SummaryModel;
  slate: ReturnType<typeof getSlate>;
}) {
  const mlb = getMlbLatest();
  const isMlb = sport.slateSource === 'mlb';
  const oddsSport = isMlb ? undefined : slate.sports.find((s) => s.key === sport.key);
  const games = isMlb ? mlb.slate.length : (oddsSport?.games.length ?? 0);
  const runDate = isMlb ? mlb.date : slate.generated_at.slice(0, 10);

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
        ) : isMlb ? (
          <MlbSlateTable slate={mlb.slate} />
        ) : (
          <OddsSlate sport={oddsSport as Sport} />
        )}
      </div>

      {isMlb && <MlbChips sport={sport} model={model} />}
    </section>
  );
}

/** Yesterday's result and the running record, with a way into the full detail. */
function MlbChips({ sport, model }: { sport: SportConfig; model: SummaryModel }) {
  const graded = playedGraded();
  const correct = graded.filter((g) => g.pick_correct).length;
  const ledger = gradedLedgerRow();
  // The chip only wears the accent on a day the model actually won. A green
  // badge reading "6/14 ❌" would celebrate a losing day.
  const won = correct * 2 >= graded.length;

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      {graded.length > 0 && (
        <span
          className={`rounded-full px-3 py-1 text-[12px] ${won ? 'font-semibold' : ''}`}
          style={
            won
              ? { background: 'var(--sport-accent)', color: 'var(--sport-accent-ink)' }
              : { background: 'var(--th-chip)', color: 'var(--th-muted)' }
          }
        >
          Yesterday {correct}/{graded.length} correct {won ? '✅' : '❌'}
        </span>
      )}
      <span
        className="rounded-full px-3 py-1 text-[12px]"
        style={{ background: 'var(--th-chip)', color: 'var(--th-muted)' }}
      >
        Running accuracy {fmtPctPrecise(ledger?.cum_accuracy ?? model.accuracy)}
      </span>
      {sport.href && (
        // Lands on the Slate tab — the reader is already looking at the slate.
        <Link href={`${sport.href}?tab=slate`} className="text-[14px] underline-offset-2 hover:underline" style={{ color: 'var(--th-muted)' }}>
          Futures, grades and history →
        </Link>
      )}
    </div>
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
