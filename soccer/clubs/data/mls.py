"""
MLS season structure: conference alignment, the shape of a regular season,
and the rest-of-season fixture list reconstructed from the results log.

Why this module exists at all
-----------------------------
Every other league in `soccer/clubs/` gets its remaining fixtures for free:
openfootball publishes the whole season up front, so `results.csv` already
carries unplayed rows with blank scores and `daily/simulate.py` just reads
them. MLS does not. Its upstream (philo92/mls-elo, see `fetch_mls.py`) is
an Elo-history log of *played* matches — there is no fixture list in it,
which is exactly why the rest-of-season Monte Carlo has always reported
"no_fixtures" for MLS.

The unusual thing about MLS is that most of the remaining schedule is
recoverable anyway, because the league's format pins it down:

- 30 clubs, 15 per conference, **no** promotion/relegation.
- 34 regular-season matches each: 17 home, 17 away.
- Every club plays each of its **14 conference rivals twice, once at
  home and once away** (28 matches). So an intra-conference fixture is
  remaining if and only if that *ordered* (home, away) pair has not been
  played yet — an exact reconstruction, no guessing.
- The other 6 are cross-conference, **3 at home and 3 away**, against 6
  different opponents out of the 15 in the other conference. Which six a
  club draws is a scheduling decision the results log does not reveal
  until the matches are played, so the remaining cross-conference
  fixtures are *not* individually recoverable — but each club's remaining
  count of them, split home and away, is (3 minus what it has played).

`remaining_fixtures()` returns both parts separately for that reason: the
intra-conference list is fact, the cross-conference part is a set of
per-club home/away quotas that the forecast fills by sampling a valid
pairing per simulation (`model/mls_forecast.py`). Sampling rather than
fixing one arbitrary pairing is the honest treatment — the identity of
those ~25 opponents is genuinely unknown, so it belongs in the sim's
variance rather than in a single made-up schedule.

Every invariant above is checked against the committed data by
`tests/test_mls_forecast.py`, and `verify_structure()` re-checks them at
runtime so a future expansion club (or a move to a 36-match season) fails
loudly here instead of quietly skewing the odds.

Playoff format
--------------
The bracket constants describe the format MLS has used since 2023 and
kept for 2025: 9 clubs per conference qualify, 8 v 9 is a single Wild
Card match at 8's home, Round One is a best-of-3 series with the higher
seed hosting games 1 and 3, and the Conference Semifinal, Conference
Final and MLS Cup are single matches hosted by the higher seed. Any
match level in this format that is drawn in regulation goes straight to
a shootout — Round One never plays extra time, and the single-elimination
rounds play extra time first, but either way no match is left drawn.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import pandas as pd

LEAGUE_KEY = "mls"

# 2026 alignment. Verified against the committed 2024-26 results: every
# club in those seasons appears here, and the played-match counts split
# 28-intra / 6-inter per club exactly as the format says they should.
EAST = [
    "Atlanta United FC",
    "CF Montréal",
    "Charlotte FC",
    "Chicago Fire FC",
    "Columbus Crew",
    "D.C. United",
    "FC Cincinnati",
    "Inter Miami CF",
    "Nashville SC",
    "New England Revolution",
    "New York City FC",
    "New York Red Bulls",
    "Orlando City SC",
    "Philadelphia Union",
    "Toronto FC",
]
WEST = [
    "Austin FC",
    "Colorado Rapids",
    "FC Dallas",
    "Houston Dynamo FC",
    "LA Galaxy",
    "Los Angeles FC",
    "Minnesota United FC",
    "Portland Timbers",
    "Real Salt Lake",
    "San Diego FC",
    "San Jose Earthquakes",
    "Seattle Sounders FC",
    "Sporting Kansas City",
    "St. Louis City SC",
    "Vancouver Whitecaps FC",
]

CONFERENCE: dict[str, str] = {t: "East" for t in EAST}
CONFERENCE.update({t: "West" for t in WEST})

# Season shape, per club.
MATCHES_PER_CLUB = 34
INTRA_MATCHES = 28          # 14 conference rivals, home and away
INTER_MATCHES = 6           # 6 different cross-conference opponents
INTER_HOME = 3              # of which 3 at home, 3 away
INTER_AWAY = 3

# Playoff format (see module docstring).
PLAYOFF_SPOTS = 9           # per conference
WILD_CARD_SEEDS = (8, 9)    # single match at the 8 seed's home
ROUND_ONE_WINS = 2          # best-of-3: first to 2 wins
ROUND_ONE_HOSTS = (True, False, True)  # higher seed hosts games 1 and 3

# Points for a win / draw. MLS shootout wins in Round One decide the
# *series*, not the table — the regular season has no shootouts.
WIN_POINTS = 3
DRAW_POINTS = 1


def conference(team: str) -> str:
    """East/West for a club, or "" for one this module has never heard of
    (an expansion side upstream added before the alignment here caught up).
    Callers that need a hard failure use `verify_structure`."""
    return CONFERENCE.get(team, "")


def season_results(results: pd.DataFrame, season: str) -> pd.DataFrame:
    """Played MLS rows for one season, score columns as ints.

    Upstream includes playoff matches in the same log as regular-season
    ones with nothing to tell them apart, so this filters them out
    structurally: a playoff match is always a repeat of an ordered
    (home, away) pair that the round robin already used, or a
    cross-conference pair beyond the 6 each club is scheduled, and it is
    always played after every club has reached 34 matches. Keeping only
    the first occurrence of each ordered pair, capped at the scheduled
    match count per club, leaves the regular season exactly.
    """
    sub = results[
        (results["league"] == LEAGUE_KEY)
        & (results["season"] == season)
        & results["home_score"].notna()
        & results["away_score"].notna()
    ].sort_values("date", kind="stable")

    seen: set[tuple[str, str]] = set()
    played = collections.Counter()
    keep = []
    for r in sub.itertuples():
        pair = (r.home_team, r.away_team)
        if pair in seen:
            continue  # a playoff rematch of a fixture already in the table
        if (played[r.home_team] >= MATCHES_PER_CLUB
                or played[r.away_team] >= MATCHES_PER_CLUB):
            continue  # both clubs' regular seasons are already complete
        seen.add(pair)
        played[r.home_team] += 1
        played[r.away_team] += 1
        keep.append(r.Index)

    out = sub.loc[keep].copy()
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    return out


@dataclass(frozen=True)
class Standing:
    """One club's regular-season record. Ordered by the MLS tiebreakers."""

    team: str
    played: int
    points: int
    wins: int
    goals_for: int
    goals_against: int

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def sort_key(self) -> tuple:
        """MLS's published order: points, then wins, then goal difference,
        then goals for. Descending on all four, so negate for a plain
        ascending sort. The tiebreakers below goals for (disciplinary
        points, then away/home goal splits, then a draw of lots) are not
        modeled — the forecast breaks a four-way tie at random instead."""
        return (-self.points, -self.wins, -self.goal_diff, -self.goals_for)


def standings(results: pd.DataFrame, season: str) -> dict[str, Standing]:
    """Regular-season table as of the last played match."""
    rec = {
        t: {"played": 0, "points": 0, "wins": 0, "gf": 0, "ga": 0}
        for t in CONFERENCE
    }
    for r in season_results(results, season).itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        for team, gf, ga in ((r.home_team, hs, as_), (r.away_team, as_, hs)):
            d = rec.setdefault(
                team, {"played": 0, "points": 0, "wins": 0, "gf": 0, "ga": 0})
            d["played"] += 1
            d["gf"] += gf
            d["ga"] += ga
            if gf > ga:
                d["points"] += WIN_POINTS
                d["wins"] += 1
            elif gf == ga:
                d["points"] += DRAW_POINTS
    return {
        t: Standing(t, d["played"], d["points"], d["wins"], d["gf"], d["ga"])
        for t, d in rec.items()
    }


@dataclass(frozen=True)
class RemainingSchedule:
    """What is left of a regular season.

    `intra` is exact — every ordered same-conference pair the round robin
    still owes. `inter_home` / `inter_away` are per-club counts of
    cross-conference matches left to play at home and away; which
    opponents they land against is not knowable from the results log, so
    the forecast samples a pairing consistent with these counts per
    simulation.
    """

    season: str
    intra: list[tuple[str, str]]
    inter_home: dict[str, int]
    inter_away: dict[str, int]
    played_inter: set[frozenset[str]]

    @property
    def inter_count(self) -> int:
        return sum(self.inter_home.values())

    def __len__(self) -> int:
        return len(self.intra) + self.inter_count


def remaining_fixtures(results: pd.DataFrame, season: str) -> RemainingSchedule:
    """Reconstruct the rest of a regular season from the played matches."""
    played = season_results(results, season)
    intra_played: set[tuple[str, str]] = set()
    inter_home = collections.Counter()
    inter_away = collections.Counter()
    played_inter: set[frozenset[str]] = set()
    for r in played.itertuples():
        h, a = r.home_team, r.away_team
        if conference(h) and conference(h) == conference(a):
            intra_played.add((h, a))
        else:
            inter_home[h] += 1
            inter_away[a] += 1
            played_inter.add(frozenset((h, a)))

    intra = [
        (h, a)
        for group in (EAST, WEST)
        for h in group
        for a in group
        if h != a and (h, a) not in intra_played
    ]
    return RemainingSchedule(
        season=season,
        intra=sorted(intra),
        inter_home={t: INTER_HOME - inter_home[t] for t in CONFERENCE},
        inter_away={t: INTER_AWAY - inter_away[t] for t in CONFERENCE},
        played_inter=played_inter,
    )


def verify_structure(results: pd.DataFrame, season: str) -> list[str]:
    """Check the reconstruction against the format. Returns a list of
    problems — empty means the season fits the shape this module assumes,
    and the remaining fixtures are therefore trustworthy.

    Run before publishing odds. A non-empty result means MLS changed
    something (an expansion club, a 36-match season, an unbalanced
    conference) and the constants above need updating; it does not mean
    the data is corrupt.
    """
    problems = []
    played = season_results(results, season)
    seen = set(played["home_team"]) | set(played["away_team"])
    unknown = sorted(seen - set(CONFERENCE))
    if unknown:
        problems.append(f"clubs missing from the conference map: {unknown}")
        return problems
    if len(EAST) != len(WEST):
        problems.append(f"conferences uneven: {len(EAST)} East, {len(WEST)} West")

    rem = remaining_fixtures(results, season)
    gp = collections.Counter()
    for r in played.itertuples():
        gp[r.home_team] += 1
        gp[r.away_team] += 1

    home_left = collections.Counter(h for h, _ in rem.intra)
    away_left = collections.Counter(a for _, a in rem.intra)
    for t in CONFERENCE:
        for label, n in (("inter home", rem.inter_home[t]),
                         ("inter away", rem.inter_away[t])):
            if n < 0:
                problems.append(f"{t}: {label} remaining is {n} (over the scheduled {INTER_HOME})")
        total = (gp[t] + home_left[t] + away_left[t]
                 + rem.inter_home[t] + rem.inter_away[t])
        if total != MATCHES_PER_CLUB:
            problems.append(
                f"{t}: played {gp[t]} + remaining lands on {total} matches, "
                f"not {MATCHES_PER_CLUB}")

    # The two conferences' cross-conference quotas have to clear against
    # each other, or no valid pairing of the remaining inter fixtures
    # exists at all.
    for host, guest, label in ((EAST, WEST, "East-home"), (WEST, EAST, "West-home")):
        h = sum(rem.inter_home[t] for t in host)
        a = sum(rem.inter_away[t] for t in guest)
        if h != a:
            problems.append(
                f"{label} cross-conference quotas do not balance: {h} vs {a}")
    return problems
