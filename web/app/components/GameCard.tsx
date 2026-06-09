import { fmtOdds, fmtPct, fmtSpread, type Game } from '@/app/lib/data';

export default function GameCard({ game }: { game: Game }) {
  const { consensus, model, line_movement: move } = game;
  const homeFav =
    consensus.home_win_prob !== null && consensus.home_win_prob >= 0.5;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-base font-semibold">
          {game.away_team} <span className="font-normal text-slate-400">@</span>{' '}
          {game.home_team}
        </div>
        <div className="text-xs text-slate-400">{game.commence_et} ET</div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 text-center sm:grid-cols-4">
        <Stat
          label={`${shortName(game.home_team)} ML`}
          value={fmtOdds(consensus.home_ml)}
          accent={homeFav ? 'text-fav' : undefined}
        />
        <Stat
          label={`${shortName(game.away_team)} ML`}
          value={fmtOdds(consensus.away_ml)}
          accent={!homeFav ? 'text-fav' : undefined}
        />
        <Stat
          label="Spread (home)"
          value={fmtSpread(consensus.home_spread)}
        />
        <Stat label="Total" value={consensus.total?.toFixed(1) ?? '—'} />
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        {consensus.home_win_prob !== null && (
          <Chip>
            Market: {shortName(game.home_team)}{' '}
            {fmtPct(consensus.home_win_prob)} (no-vig)
          </Chip>
        )}
        {move && (move.spread_delta !== undefined || move.total_delta !== undefined) && (
          <Chip>
            Since open{move.first_seen ? ` (${move.first_seen.slice(0, 10)})` : ''}:
            {move.spread_delta !== undefined &&
              ` spread ${signed(move.spread_delta)}`}
            {move.spread_delta !== undefined && move.total_delta !== undefined && ' · '}
            {move.total_delta !== undefined && ` total ${signed(move.total_delta)}`}
          </Chip>
        )}
      </div>

      <div className="mt-3 rounded border border-slate-100 bg-slate-50 p-3 text-sm">
        {model && model.predicted_winner ? (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-semibold">
              🤖 Model pick: {model.predicted_winner}
            </span>
            {model.home_win_prob !== null && (
              <span className="text-slate-500">
                {shortName(game.home_team)} {fmtPct(model.home_win_prob)}
              </span>
            )}
            {model.pred_spread !== null && (
              <span className="text-slate-500">
                pred margin {fmtSpread(model.pred_spread)}
              </span>
            )}
            {model.edge_vs_market !== null && (
              <span
                className={`font-semibold ${
                  Math.abs(model.edge_vs_market) >= 0.05
                    ? model.edge_vs_market > 0
                      ? 'text-fav'
                      : 'text-dog'
                    : 'text-slate-500'
                }`}
              >
                edge vs market {signed(Math.round(model.edge_vs_market * 1000) / 10)}pp
              </span>
            )}
          </div>
        ) : (
          <span className="text-slate-400">
            🤖 No model pick for this game yet.
          </span>
        )}
      </div>

      {game.books.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs font-semibold text-slate-500 hover:text-slate-700">
            Book-by-book odds ({game.books.length} sportsbooks)
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-left text-slate-400">
                  <th className="py-1 pr-2 font-medium">Book</th>
                  <th className="py-1 pr-2 font-medium">Home ML</th>
                  <th className="py-1 pr-2 font-medium">Away ML</th>
                  <th className="py-1 pr-2 font-medium">Home spread</th>
                  <th className="py-1 pr-2 font-medium">Away spread</th>
                  <th className="py-1 pr-2 font-medium">Over</th>
                  <th className="py-1 font-medium">Under</th>
                </tr>
              </thead>
              <tbody>
                {game.books.map((b) => (
                  <tr key={b.book} className="border-b border-slate-100">
                    <td className="py-1 pr-2 font-medium text-slate-600">{b.book}</td>
                    <td className="py-1 pr-2">{fmtOdds(b.home_ml)}</td>
                    <td className="py-1 pr-2">{fmtOdds(b.away_ml)}</td>
                    <td className="py-1 pr-2">{fmtOdds(b.home_spread_odds)}</td>
                    <td className="py-1 pr-2">{fmtOdds(b.away_spread_odds)}</td>
                    <td className="py-1 pr-2">{fmtOdds(b.over_odds)}</td>
                    <td className="py-1">{fmtOdds(b.under_odds)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded border border-slate-100 bg-slate-50 p-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-400">
        {label}
      </div>
      <div className={`mt-0.5 text-lg font-bold ${accent ?? 'text-slate-900'}`}>
        {value}
      </div>
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">
      {children}
    </span>
  );
}

/** "New York Knicks" -> "Knicks" for compact stat labels. */
function shortName(team: string): string {
  const parts = team.split(' ');
  return parts[parts.length - 1];
}

function signed(n: number): string {
  return n > 0 ? `+${n}` : `${n}`;
}
