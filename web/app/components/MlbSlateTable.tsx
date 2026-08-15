import ThemedTable from '@/app/components/ThemedTable';
import { fmtPct, fmtSimScore } from '@/app/lib/format';
import { teamName, type MlbSlateRow } from '@/app/lib/mlb';

/**
 * Today's MLB picks. Shared verbatim by the home dashboard and the MLB page's
 * Slate tab so the two can never drift apart.
 */

const COLUMNS = [
  { header: 'Matchup (home second)' },
  { header: 'Model pick', strong: true },
  { header: 'Win prob' },
  { header: 'Sim score' },
];

/*
 * The scores are means over 10,000 sims, kept at one decimal. Rounding them to
 * whole runs collapses nearly every game onto the same 6–3, which reads like a
 * literal predicted final score rather than an average — the decimal is what
 * keeps the column honest.
 */
const NOTE =
  'Score is the mean of 10,000 sims, conditioned on the picked side winning ' +
  '(winner’s runs first), shown at one decimal — integer rounding collapses ' +
  'nearly every game onto 6–3. The model is team-level Elo and does not use ' +
  'starter identity.';

export default function MlbSlateTable({ slate }: { slate: MlbSlateRow[] }) {
  const rows = slate.map((g) => {
    const pickedHome = g.pick === g.home;
    return {
      key: `${g.away}-${g.home}-${g.game_num}`,
      cells: [
        `${teamName(g.away)} @ ${teamName(g.home)}${g.game_num > 1 ? ` (G${g.game_num})` : ''}`,
        teamName(g.pick),
        fmtPct(g.pick_prob),
        fmtSimScore(
          Number(g.pred_home_score?.toFixed(1)),
          Number(g.pred_away_score?.toFixed(1)),
          pickedHome,
        ),
      ],
    };
  });

  return <ThemedTable columns={COLUMNS} rows={rows} note={NOTE} />;
}
