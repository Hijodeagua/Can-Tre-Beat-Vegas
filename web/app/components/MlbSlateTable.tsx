import ThemedTable from '@/app/components/ThemedTable';
import { DASH, fmtPct, fmtSimScore, missing } from '@/app/lib/format';
import { teamName, type MlbSlateRow } from '@/app/lib/mlb';

/**
 * Today's MLB picks. Shared verbatim by the home dashboard and the MLB page's
 * Slate tab so the two can never drift apart.
 *
 * Whether starter identity is a model input varies by which model version is
 * active (mlb/daily/config.py's ACTIVE_MODEL), so the SP columns and the
 * closing note are both driven off whether the rows actually carry a
 * *_sp_adj value rather than a hardcoded claim — the same test
 * mlb/daily/emails.py's slate table uses, so the email and the site can
 * never quietly disagree about which model produced a slate.
 */

/*
 * The scores are means over 10,000 sims, kept at one decimal. Rounding them to
 * whole runs collapses nearly every game onto the same 6–3, which reads like a
 * literal predicted final score rather than an average — the decimal is what
 * keeps the column honest.
 */
const BASE_NOTE =
  'Score is the mean of 10,000 sims, conditioned on the picked side winning ' +
  '(winner’s runs first), shown at one decimal — integer rounding collapses ' +
  'nearly every game onto 6–3.';

const SP_NOTE =
  `${BASE_NOTE} Starter identity IS a model input: each probable’s rolling ` +
  'game score (exponentially weighted, 20-start half-life) is compared to ' +
  'his team’s staff average, and the difference enters the team’s Elo at ' +
  '3.0 points per game-score point — the SP adj column, so the effect is ' +
  'auditable per game. Rookies/thin history fall back to a shrunk staff ' +
  'rating; TBD uses the staff rating. Rest and travel add up to a few Elo ' +
  'points each.';

const NO_SP_NOTE =
  `${BASE_NOTE} The model is team-level Elo and does not use starter identity.`;

/** Signed, one-decimal Elo delta, or a dash — matches format.ts's rule that
 * missing data reads as an em dash, never a zero. */
function fmtEloAdj(value: number | null | undefined): string {
  if (missing(value)) return DASH;
  const v = value as number;
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
}

export default function MlbSlateTable({ slate }: { slate: MlbSlateRow[] }) {
  const hasSpAdj = slate.some((g) => !missing(g.home_sp_adj));

  const columns = [
    { header: 'Matchup (home second)' },
    { header: 'SP (away vs. home)' },
    ...(hasSpAdj ? [{ header: 'SP adj (Elo, away/home)' }] : []),
    { header: 'Model pick', strong: true },
    { header: 'Win prob' },
    { header: 'Sim score' },
  ];

  const rows = slate.map((g) => {
    const pickedHome = g.pick === g.home;
    const spAdjCell = `${fmtEloAdj(g.away_sp_adj)} / ${fmtEloAdj(g.home_sp_adj)}`;
    return {
      key: `${g.away}-${g.home}-${g.game_num}`,
      cells: [
        `${teamName(g.away)} @ ${teamName(g.home)}${g.game_num > 1 ? ` (G${g.game_num})` : ''}`,
        `${g.away_sp || 'TBD'} vs. ${g.home_sp || 'TBD'}`,
        ...(hasSpAdj ? [spAdjCell] : []),
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

  return (
    <ThemedTable columns={columns} rows={rows} note={hasSpAdj ? SP_NOTE : NO_SP_NOTE} />
  );
}
