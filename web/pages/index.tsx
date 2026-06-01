import type { GetStaticProps } from 'next';
import { getPicksToday, type PicksToday, type Pick } from '@/lib/data';

interface Props {
  data: PicksToday;
}

export const getStaticProps: GetStaticProps<Props> = async () => {
  return { props: { data: getPicksToday() } };
};

function LeanBadge({ lean }: { lean: Pick['model_lean'] }) {
  const map: Record<string, string> = {
    cover: 'bg-green-600/20 text-green-400 border-green-600/40',
    fade: 'bg-red-600/20 text-red-400 border-red-600/40',
    push: 'bg-slate-600/20 text-slate-400 border-slate-600/40',
  };
  return (
    <span className={`rounded border px-2 py-0.5 text-xs font-semibold uppercase ${map[lean] ?? map.push}`}>
      {lean}
    </span>
  );
}

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`;
}

function spread(s: number) {
  return s > 0 ? `+${s}` : `${s}`;
}

export default function TodaysPicks({ data }: Props) {
  const { picks, generated_at } = data;

  return (
    <div>
      <h2 className="text-2xl font-bold">Today&apos;s Picks</h2>
      <p className="mt-1 text-sm text-slate-400">
        Model win probability and against-the-spread (ATS) probability for each
        upcoming game, with the model&apos;s lean relative to the Vegas line.
      </p>

      {picks.length > 0 ? (
        <div className="mt-6 overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3">Matchup</th>
                <th className="px-4 py-3 text-right">Vegas spread</th>
                <th className="px-4 py-3 text-right">Win prob</th>
                <th className="px-4 py-3 text-right">ATS prob</th>
                <th className="px-4 py-3 text-center">Lean</th>
              </tr>
            </thead>
            <tbody>
              {picks.map((p) => (
                <tr key={p.game_id} className="border-b border-slate-800/60 last:border-0">
                  <td className="px-4 py-3">
                    <div className="font-medium">
                      {p.away_team} <span className="text-slate-500">@</span> {p.home_team}
                    </div>
                    <div className="text-xs text-slate-500">
                      {new Date(p.game_time).toLocaleString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                        hour: 'numeric',
                        minute: '2-digit',
                      })}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {p.home_team.split(' ').pop()} {spread(p.vegas_spread)}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">{pct(p.model_win_prob)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{pct(p.model_ats_prob)}</td>
                  <td className="px-4 py-3 text-center">
                    <LeanBadge lean={p.model_lean} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="mt-6 rounded-lg border border-dashed border-slate-700 p-8 text-center text-sm text-slate-400">
          No picks yet — check back when the next slate of games is posted.
        </div>
      )}

      {generated_at && (
        <p className="mt-4 text-xs text-slate-600">
          Win prob = P(home wins) · ATS prob = P(home covers). Generated{' '}
          {new Date(generated_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}.
        </p>
      )}
    </div>
  );
}
