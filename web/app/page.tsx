import GameCard from '@/app/components/GameCard';
import { getMeta, getSlate } from '@/app/lib/data';

const SPORT_EMOJI: Record<string, string> = {
  nfl: '🏈',
  nba: '🏀',
  worldcup: '⚽',
};

export default function SlatePage() {
  const slate = getSlate();
  const meta = getMeta();
  const totalGames = slate.sports.reduce((n, s) => n + s.games.length, 0);

  return (
    <div>
      <h2 className="text-2xl font-bold">The Next {slate.window_hours} Hours</h2>
      <p className="mt-1 text-sm text-slate-500">
        Every game on the board in the next {slate.window_hours} hours — what
        the bookies say, how the line has moved, and what my models say when
        they have an opinion. {meta.label}
      </p>

      {totalGames === 0 ? (
        <div className="mt-6 rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500">
          No games in the next {slate.window_hours} hours. The slate refreshes
          twice daily from the odds feed.
        </div>
      ) : (
        slate.sports
          .filter((sport) => sport.games.length > 0)
          .map((sport) => (
            <section key={sport.key} className="mt-8">
              <div className="flex items-baseline justify-between">
                <h3 className="text-lg font-bold">
                  {SPORT_EMOJI[sport.key] ?? ''} {sport.name}
                  <span className="ml-2 text-sm font-normal text-slate-400">
                    {sport.games.length} game{sport.games.length === 1 ? '' : 's'}
                  </span>
                </h3>
                {sport.snapshot?.pulled_at_et && (
                  <span className="text-xs text-slate-400">
                    odds as of {sport.snapshot.pulled_at_et} ET
                  </span>
                )}
              </div>
              <div className="mt-3 flex flex-col gap-4">
                {sport.games.map((game) => (
                  <GameCard key={game.game_id} game={game} />
                ))}
              </div>
            </section>
          ))
      )}

      {slate.sports.some((s) => s.games.length === 0) && totalGames > 0 && (
        <p className="mt-8 text-xs text-slate-400">
          No games in the window for:{' '}
          {slate.sports
            .filter((s) => s.games.length === 0)
            .map((s) => s.name)
            .join(', ')}
          .
        </p>
      )}

      {meta.last_updated && (
        <p className="mt-8 text-xs text-slate-400">
          Slate generated {meta.last_updated.replace('T', ' ').replace('Z', ' UTC')}
        </p>
      )}
    </div>
  );
}
