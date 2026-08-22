"""
Parser for the openfootball Football.TXT match format.

One parser serves both the country league repos (england, deutschland, …)
used for current-season fixtures/results and the champions-league repo
(cl.txt / el.txt / conf.txt) used for the UEFA cross-league glue.

The format, per line:

    = English Premier League 2026/27          header (ignored)
    # Teams 20                                comment (ignored)
    ▪ Matchday 1  /  ▪ Group A  /  ▪ Final    round marker
    Fri Aug 21 2026                           date; year carries forward
      Sat Aug 22                              (year inferred, Dec→Jan rolls)
        20:00  Arsenal FC v Fulham FC  2-1 (1-0)   match
               Juventus (ITA) v Malmö FF (SWE) 2-0  UEFA files carry country codes

Score semantics worth knowing: `(1-0)` after the score is half-time (never
the result), `2-1 a.e.t. (…)` means the match ended 2-1 after extra time,
and `4-1 pen. 1-1 a.e.t. (…)` puts the *shootout* first — the real on-the-
night score is the one before `a.e.t.`. Elo cares about the played result,
so a pens decider parses as its extra-time score (usually a draw).
"""

import re
from dataclasses import dataclass
from typing import Optional

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}

DATE_RE = re.compile(
    r"^\s*\[?(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?P<mon>[A-Z][a-z]{2})[a-z]*\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\]?\s*$"
)
ROUND_RE = re.compile(r"^\s*[▪»]\s*(?P<round>.+?)\s*$")
MATCH_RE = re.compile(
    r"^\s*(?:\d{1,2}[.:]\d{2}\s+)?"          # optional kickoff time
    r"(?P<team1>\S.*?)\s+v\s+(?P<team2>\S.*?)"
    r"(?:\s\s+(?P<tail>.*?))?\s*$"            # 2+ spaces before score/annotations
)
SCORE_RE = re.compile(r"(?<![\d(])(\d+)-(\d+)(?![\d)])")
COUNTRY_RE = re.compile(r"\s*\((?P<code>[A-Z]{3})\)\s*$")


@dataclass
class TxtMatch:
    date: str                     # ISO, e.g. "2026-08-22"
    round: str
    team1: str
    team2: str
    country1: Optional[str]       # 3-letter code when present (UEFA files)
    country2: Optional[str]
    score1: Optional[int]         # None while unplayed
    score2: Optional[int]


def _split_team(raw: str) -> tuple[str, Optional[str]]:
    m = COUNTRY_RE.search(raw)
    if m:
        return raw[: m.start()].strip(), m.group("code")
    return raw.strip(), None


def _parse_score(tail: str) -> tuple[Optional[int], Optional[int]]:
    """Final on-the-night score from the annotation tail, or (None, None)."""
    if not tail:
        return None, None
    tail = tail.split("@")[0]                       # drop venue
    tail = re.sub(r"\([^)]*\)", " ", tail)          # drop half-time / breakdowns
    if "pen" in tail:
        # "<pens> pen. <score> a.e.t." — the real score follows "pen."
        tail = tail.split("pen", 1)[1].lstrip(".").lstrip()
    scores = SCORE_RE.findall(tail)
    if not scores:
        return None, None
    return int(scores[0][0]), int(scores[0][1])


def parse(text: str, season: str) -> list[TxtMatch]:
    """Parse one Football.TXT file. A date line without an explicit year is
    resolved from `season` ("2026-27") by month alone — July–December is the
    season's first year, January–June the second. (Group-stage files restart
    their date sequence per group, so carrying the year forward between date
    lines mis-years every group after the first; month inference doesn't.)"""
    start_year = int(season[:4])
    cur_date: Optional[str] = None
    cur_round = ""
    out: list[TxtMatch] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("=", "#", "(", "--")):
            continue

        m = ROUND_RE.match(line)
        if m:
            cur_round = m.group("round")
            continue

        m = DATE_RE.match(line)
        if m:
            month = MONTHS[m.group("mon")]
            if m.group("year"):
                year = int(m.group("year"))
            else:
                year = start_year if month >= 7 else start_year + 1
            cur_date = f"{year:04d}-{month:02d}-{int(m.group('day')):02d}"
            continue

        m = MATCH_RE.match(line)
        if m and cur_date and " v " in f" {stripped} ":
            team1, c1 = _split_team(m.group("team1"))
            team2raw = m.group("team2")
            tail = m.group("tail") or ""
            # A scoreless fixture line can swallow trailing words into team2;
            # names never contain digits, so split any score off team2 itself.
            sm = SCORE_RE.search(team2raw)
            if sm and not tail:
                tail = team2raw[sm.start():]
                team2raw = team2raw[: sm.start()]
            team2, c2 = _split_team(team2raw)
            s1, s2 = _parse_score(tail)
            if not team1 or not team2:
                continue
            out.append(
                TxtMatch(cur_date, cur_round, team1, team2, c1, c2, s1, s2)
            )
    return out
