import type { GetStaticProps } from 'next';
import { getRecord, type RecordFile, type RecordEntry } from '@/lib/data';

interface Props {
  data: RecordFile;
}

export const getStaticProps: GetStaticProps<Props> = async () => {
  return { props: { data: getRecord() } };
};

// Flat-stakes ROI at standard -110 ATS pricing: a win returns +0.909 units,
// a loss -1 unit, pushes/no-bets are excluded from staked capital.
const WIN_PAYOUT = 100 / 110;

function summarize(entries: RecordEntry[]) {
  const decided = entries.filter((e) => e.result === 'win' || e.result === 'loss');
  const wins = decided.filter((e) => e.result === 'win').length;
  const losses = decided.filter((e) => e.result === 'loss').length;
  const pushes = entries.filter((e) => e.result === 'push').length;
  const atsRate = decided.length ? wins / decided.length : 0;
  const profit = wins * WIN_PAYOUT - losses;
  const roi = decided.length ? profit / decided.length : 0;
  return { decided: decided.length, wins, losses, pushes, atsRate, profit, roi };
}

function ResultBadge({ result }: { result: RecordEntry['result'] }) {
  const map: Record<string, string> = {
    win: 'bg-green-600/20 text-green-400',
    loss: 'bg-red-600/20 text-red-400',
    push: 'bg-slate-600/20 text-slate-400',
    no_bet: 'bg-slate-700/20 text-slate-500',
  };
  const label = result === 'no_bet' ? 'no bet' : result ?? 'pending';
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${map[result ?? ''] ?? 'bg-slate-700/20 text-slate-500'}`}>
      {label}
    </span>
  );
}

export default function Record({ data }: Props) {
  const graded = data.picks.filter((e) => e.result !== null);
  const s = summarize(graded);

  return (
    <div>
      <h2 className="text-2xl font-bold">Season Record</h2>
      <p className="mt-1 text-sm text-slate-400">
        Every graded pick against the spread, with running accuracy and
        flat-stakes ROI at standard -110 pricing.
      </p>

      {graded.length > 0 ? (
        <>
          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Record (W–L)" value={`${s.wins}–${s.losses}`} />
            <Stat label="ATS win rate" value={`${(s.atsRate * 100).toFixed(1)}%`} />
            <Stat
              label="ROI (flat)"
              value={`${s.roi >= 0 ? '+' : ''}${(s.roi * 100).toFixed(1)}%`}
              accent={s.roi >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <Stat
              label="Units"
              value={`${s.profit >= 0 ? '+' : ''}${s.profit.toFixed(2)}`}
              accent={s.profit >= 0 ? 'text-green-400' : 'text-red-400'}
            />
          </div>
          {s.pushes > 0 && (
            <p className="mt-2 text-xs text-slate-500">{s.pushes} push(es) excluded from win rate and ROI.</p>
          )}

          <div className="mt-6 overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3">Pick</th>
                  <th className="px-4 py-3">Final</th>
                  <th className="px-4 py-3 text-center">Result</th>
                </tr>
              </thead>
              <tbody>
                {graded
                  .slice()
                  .sort((a, b) => b.pick_date.localeCompare(a.pick_date))
                  .map((e) => (
                    <tr key={e.game_id} className="border-b border-slate-800/60 last:border-0">
                      <td className="px-4 py-3 text-slate-400">{e.pick_date}</td>
                      <td className="px-4 py-3 font-medium">{e.pick}</td>
                      <td className="px-4 py-3 text-slate-400">{e.final_score ?? '—'}</td>
                      <td className="px-4 py-3 text-center">
                        <ResultBadge result={e.result} />
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="mt-6 rounded-lg border border-dashed border-slate-700 p-8 text-center text-sm text-slate-400">
          No graded picks yet — results appear here once games are final.
        </div>
      )}

      {data.updated_at && (
        <p className="mt-4 text-xs text-slate-600">
          Last updated{' '}
          {new Date(data.updated_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}.
        </p>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-center">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${accent ?? 'text-slate-100'}`}>{value}</div>
    </div>
  );
}
