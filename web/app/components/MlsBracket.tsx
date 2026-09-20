'use client';

/**
 * The MLS Cup Playoffs bracket, drawn from the forecast's simulations.
 *
 * A bracket is a picture of *one* season and the forecast is 20,000 of
 * them, so this does not draw a prediction. It draws the bracket the
 * expected finishing positions imply, and puts against each slot how
 * often that club actually lands on that seed — so the shape is a real
 * bracket while the numbers keep it honest about being a distribution.
 *
 * Seeding by expected finish rather than by each slot's most likely
 * occupant is deliberate, and not a rounding choice. Slot modes are
 * marginals: taken independently they cheerfully put one club at the top
 * of two different slots and leave another out of the bracket entirely,
 * which is true of the marginals and nonsense as a bracket. Ordering by
 * expected finish gives every club exactly one slot, and the runners-up
 * line under each one says who else genuinely competes for it.
 *
 * Reading left to right is reading the format: seeds 8 and 9 meet in a
 * single Wild Card match, the survivor takes the 8 seed into a best-of-3
 * Round One against the 1 seed, and the three rounds after that are
 * single matches at the higher seed's ground. The top seed's reward is a
 * harder opponent than a bye, which is most of why the Shield favourite
 * is rarely the Cup favourite.
 */

import { fmtPct } from '@/app/lib/format';
import type { SoccerMlsBracket, SoccerMlsClub } from '@/app/lib/soccer';

/** The clubs of one conference in expected-finish order — the seeding the
 * bracket is drawn on. Index i is seed i+1. */
function seeding(clubs: SoccerMlsClub[], conference: 'East' | 'West') {
  return clubs
    .filter((c) => c.conference === conference)
    .sort((a, b) => a.exp_conf_seed - b.exp_conf_seed);
}

/** The rounds a club can reach, and the per-club field that holds how
 * often it does. `p_playoffs` is the entry condition rather than a round,
 * so the first column is qualification itself. */
const ROUNDS: { key: keyof SoccerMlsClub; label: string; note: string }[] = [
  { key: 'p_playoffs', label: 'Qualify', note: 'top 9 in the conference' },
  { key: 'p_conf_semi', label: 'Conf Semifinal', note: 'won a best-of-3 Round One' },
  { key: 'p_conf_final', label: 'Conf Final', note: 'one match at the higher seed' },
  { key: 'p_conf_title', label: 'MLS Cup', note: 'won the conference' },
  { key: 'p_cup', label: 'Champion', note: 'won MLS Cup' },
];

function shortName(team: string): string {
  return team
    .replace(/^(AFC|FC|AC|CF|SC)\s+/i, '')
    .replace(/\s+(FC|AFC|CF|SC)$/i, '');
}

/** The bracket's two halves. Each holds the two Round One series whose
 * winners meet in that half's Conference Semifinal, so the grouping on
 * screen is the grouping in the format rather than just a layout. The 1
 * seed's opponent is whoever survives 8 v 9, so that slot is drawn as the
 * Wild Card pair it comes from. */
const HALVES: [number, number | [number, number]][][] = [
  [[1, [8, 9]], [4, 5]],
  [[2, 7], [3, 6]],
];

function SeedCell({
  slot, bracket, conference, order,
}: {
  slot: number;
  bracket: SoccerMlsBracket;
  conference: 'East' | 'West';
  order: SoccerMlsClub[];
}) {
  const club = order[slot - 1];
  if (!club) {
    return (
      <div className="text-[12px]" style={{ color: 'var(--th-faint)' }}>
        {slot} —
      </div>
    );
  }
  // How often this club actually finishes on this seed — the honest
  // number behind a slot the expected finish assigned it.
  const held = club.seed_distribution[slot - 1] ?? 0;
  // Who else competes for the slot, the club holding it aside.
  const entry = bracket.conferences[conference].seeds.find((s) => s.seed === slot);
  const rest = (entry?.candidates ?? [])
    .filter((c) => c.team !== club.team)
    .slice(0, 2);
  const lead = { team: club.team, p: held };
  return (
    <div
      className="rounded border px-2 py-1"
      style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
    >
      <div className="flex items-baseline gap-1.5">
        <span
          className="text-[10px] tabular-nums"
          style={{ color: 'var(--th-faint)' }}
        >
          {slot}
        </span>
        <span className="text-[12px] font-semibold" style={{ color: 'var(--th-ink)' }}>
          {shortName(lead.team)}
        </span>
        <span className="ml-auto text-[11px] tabular-nums" style={{ color: 'var(--th-muted)' }}>
          {fmtPct(lead.p)}
        </span>
      </div>
      {rest.length > 0 && (
        <div className="mt-0.5 text-[10.5px]" style={{ color: 'var(--th-faint)' }}>
          {rest.map((c) => `${shortName(c.team)} ${fmtPct(c.p)}`).join(' · ')}
        </div>
      )}
    </div>
  );
}

function ConferenceBracket({
  bracket, conference, clubs,
}: {
  bracket: SoccerMlsBracket;
  conference: 'East' | 'West';
  clubs: SoccerMlsClub[];
}) {
  const side = bracket.conferences[conference];
  const order = seeding(clubs, conference);
  return (
    <section className="mt-5">
      <h4 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
        {conference}ern Conference
      </h4>
      <p className="mt-1 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Most likely to reach MLS Cup: <b>{side.favorite}</b> {fmtPct(side.p_favorite)}
      </p>
      <div className="mt-2 grid items-start gap-3 sm:grid-cols-2">
        {HALVES.map((half, i) => (
          <div
            key={i}
            className="rounded-lg border p-2"
            style={{ borderColor: 'var(--th-border)' }}
          >
            <div className="mb-2 text-[10px] uppercase tracking-wide"
                 style={{ color: 'var(--th-faint)' }}>
              Conference Semifinal {i + 1} · winners of these two meet
            </div>
            <div className="grid items-start gap-2">
              {half.map(([high, low]) => (
                <div
                  key={high}
                  className="rounded border p-1.5"
                  style={{ borderColor: 'var(--th-border)' }}
                >
                  <div className="mb-1 text-[10px] uppercase tracking-wide"
                       style={{ color: 'var(--th-faint)' }}>
                    Round One · best of 3
                  </div>
                  <div className="grid gap-1.5">
                    <SeedCell slot={high} bracket={bracket} conference={conference}
                              order={order} />
                    {Array.isArray(low) ? (
                      <div
                        className="rounded border border-dashed p-1.5"
                        style={{ borderColor: 'var(--th-border)' }}
                      >
                        <div className="mb-1 text-[10px] uppercase tracking-wide"
                             style={{ color: 'var(--th-faint)' }}>
                          Wild Card · {low[0]} hosts {low[1]}
                        </div>
                        <div className="grid gap-1.5">
                          {low.map((sd) => (
                            <SeedCell key={sd} slot={sd} bracket={bracket}
                                      conference={conference} order={order} />
                          ))}
                        </div>
                      </div>
                    ) : (
                      <SeedCell slot={low} bracket={bracket} conference={conference}
                                order={order} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
        Slots are filled by expected finishing position, and the figure beside a
        club is how often it actually lands on that exact seed — rarely more than
        a third of the time, because a seed is a fine distinction. The line
        underneath is who else competes for the slot. The two semifinal winners
        meet in the Conference Final, at the higher seed&apos;s ground.
      </p>
    </section>
  );
}

export default function MlsBracket({
  bracket, clubs,
}: {
  bracket: SoccerMlsBracket;
  clubs: SoccerMlsClub[];
}) {
  // Round-by-round survival, deepest run first — the same numbers as the
  // conference tables, read across the bracket instead of down a table.
  const byCup = [...clubs].sort((a, b) => b.p_cup - a.p_cup).slice(0, 10);
  return (
    <div>
      <ConferenceBracket bracket={bracket} conference="East" clubs={clubs} />
      <ConferenceBracket bracket={bracket} conference="West" clubs={clubs} />

      <section className="mt-6">
        <h4 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
          How far each club gets
        </h4>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr>
                <th className="px-2 py-1 text-left font-semibold"
                    style={{ color: 'var(--th-muted)' }}>
                  Club
                </th>
                {ROUNDS.map((r) => (
                  <th
                    key={r.label}
                    title={r.note}
                    className="px-2 py-1 text-right font-semibold"
                    style={{ color: 'var(--th-muted)' }}
                  >
                    {r.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {byCup.map((c) => (
                <tr key={c.team} style={{ borderTop: '1px solid var(--th-border)' }}>
                  <td className="px-2 py-1" style={{ color: 'var(--th-ink)' }}>
                    <b>{shortName(c.team)}</b>{' '}
                    <span style={{ color: 'var(--th-faint)' }}>
                      {c.conference === 'East' ? 'E' : 'W'}
                    </span>
                  </td>
                  {ROUNDS.map((r) => (
                    <td
                      key={r.label}
                      className="px-2 py-1 text-right tabular-nums"
                      style={{ color: 'var(--th-ink)' }}
                    >
                      {fmtPct(c[r.key] as number)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Each column is the share of simulated seasons in which the club is still
          alive at that round. The drop from Qualify to Champion is the format
          talking: four rounds, the first of them a best-of-3, and a drawn match in
          any of them settled by a shootout.
        </p>
      </section>

      {bracket.finals.length > 0 && (
        <section className="mt-6">
          <h4 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
            Most likely MLS Cup matchups
          </h4>
          <ul className="mt-2 grid gap-1 text-[12px]" style={{ color: 'var(--th-ink)' }}>
            {bracket.finals.map((f) => (
              <li key={`${f.east}-${f.west}`} className="flex gap-2">
                <span className="tabular-nums" style={{ color: 'var(--th-muted)' }}>
                  {fmtPct(f.p)}
                </span>
                <span>
                  {shortName(f.east)} <span style={{ color: 'var(--th-faint)' }}>v</span>{' '}
                  {shortName(f.west)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[12px]" style={{ color: 'var(--th-faint)' }}>
            Even the likeliest single final is a long way under 10%: there are 81
            possible pairings and the bracket is short enough that most of them
            stay live.
          </p>
        </section>
      )}
    </div>
  );
}
