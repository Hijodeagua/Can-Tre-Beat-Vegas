import type { Metadata } from 'next';
import MlbTabs from './MlbTabs';

export const metadata: Metadata = {
  title: 'MLB Daily Model — Can Tre Beat Vegas',
  description:
    'Betting-blind Elo: daily picks with simulated scores, rest-of-season Monte Carlo futures, and a graded track record.',
};

export default function MlbPage() {
  return (
    <div>
      <h2 className="text-2xl font-bold">⚾ MLB Daily Model</h2>
      <p className="mt-1 text-sm text-slate-500">
        Betting-blind Elo (K=3, +24 home, margin-of-victory weighted). Every
        morning: rest-of-season futures, today&apos;s slate with simulated
        scores, and yesterday&apos;s graded report card. Backtest 2009–2025:
        56.7% straight up, log-loss 0.680 vs 0.691 for always-pick-home.
      </p>
      <MlbTabs />
    </div>
  );
}
