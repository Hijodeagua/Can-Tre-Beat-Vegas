"""Starting-pitcher rating: Bill James game score, rolling game score (rGS)
books, and the pregame Elo adjustment.

Game score (James original):

    50 + outs + 2*(complete innings after the 4th) + K
       - 2*H - 4*ER - 2*(unearned runs) - BB

Rolling state, all exponentially weighted by start count with a tunable
half-life (debiased EW mean, i.e. pandas ewm(adjust=True), so a pitcher's
first few starts are an honest small-sample average rather than being
anchored to an arbitrary seed):

- per-pitcher rGS,
- per-team staff rGS (every game score produced by that team's starters),
- league rGS (long half-life; the shrinkage target and cold-start seed).

Pregame adjustment to the team's Elo rating before computing win probability:

    adj = C * (effective_rGS - staff_rGS)

with the fallback ladder (never drops a game):

    named starter with >= min_starts career starts -> pitcher rGS
    named starter with thin history               -> staff rGS shrunk toward
                                                     the league mean
    TBD / unknown                                 -> staff rGS (adj = 0)

Leakage contract: results enter through `record_start`, which only buffers
them; `advance_to(date)` commits strictly-earlier starts. `pregame_adj`
raises LeakageError if the committed state contains any start dated on or
after the game being predicted, so a mis-ordered replay fails loudly instead
of silently peeking. Doubleheader game 2 therefore does NOT see game 1's
result - date granularity cannot prove G1 finished first, so the book stays
strictly conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_HALF_LIFE = 10.0   # starts; tunable (Phase 4 grid)
DEFAULT_C = 4.7            # Elo points per rGS point (538's published value)
MIN_CAREER_STARTS = 5
ROOKIE_SHRINK = 0.5        # weight kept on staff rGS for thin-history starters
LEAGUE_HALF_LIFE = 500.0   # starts; slow era baseline
LEAGUE_SEED = 50.0         # James's league-average game score by construction


class LeakageError(RuntimeError):
    """A rGS input would include a start on/after the game being predicted."""


def game_score(outs: int, h: int, r: int, er: int, bb: int, so: int) -> int:
    """Bill James game score for one start. `outs` in thirds of an inning."""
    complete_innings_after_4th = max(0, outs // 3 - 4)
    unearned = r - er
    return (50 + outs + 2 * complete_innings_after_4th + so
            - 2 * h - 4 * er - 2 * unearned - bb)


@dataclass
class _EwMean:
    """Debiased exponentially weighted mean: value = num/den with weights
    (1-alpha)^k over past observations, newest first."""
    decay: float
    num: float = 0.0
    den: float = 0.0
    n: int = 0
    last_date: str = ""

    def add(self, x: float, date: str) -> None:
        self.num = x + self.decay * self.num
        self.den = 1.0 + self.decay * self.den
        self.n += 1
        self.last_date = max(self.last_date, date)

    def value(self, default: float) -> float:
        return self.num / self.den if self.den > 0 else default


def _decay(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)


@dataclass
class PitcherBook:
    """Walk-forward rGS state for all pitchers, staffs, and the league."""
    half_life: float = DEFAULT_HALF_LIFE
    c: float = DEFAULT_C
    min_starts: int = MIN_CAREER_STARTS
    rookie_shrink: float = ROOKIE_SHRINK
    league_half_life: float = LEAGUE_HALF_LIFE

    pitchers: dict = field(default_factory=dict)   # uid -> _EwMean
    staffs: dict = field(default_factory=dict)     # team -> _EwMean
    league: _EwMean = None  # type: ignore[assignment]
    _pending: list = field(default_factory=list)   # (date, team, uid, gs)

    def __post_init__(self):
        if self.league is None:
            self.league = _EwMean(_decay(self.league_half_life))

    # -- results intake ----------------------------------------------------

    def record_start(self, date: str, team: str, uid: str,
                     gs: float) -> None:
        """Buffer a finished start; it becomes visible only once
        advance_to(some later date) commits it."""
        self._pending.append((date, team, uid, gs))

    def advance_to(self, date: str) -> None:
        """Commit every buffered start strictly earlier than `date`."""
        keep = []
        for rec in sorted(self._pending):
            if rec[0] < date:
                self._commit(*rec)
            else:
                keep.append(rec)
        self._pending = keep

    def _commit(self, date: str, team: str, uid: str, gs: float) -> None:
        d = _decay(self.half_life)
        self.pitchers.setdefault(uid, _EwMean(d)).add(gs, date)
        self.staffs.setdefault(team, _EwMean(d)).add(gs, date)
        self.league.add(gs, date)

    # -- pregame query ------------------------------------------------------

    def pregame_adj(self, team: str, uid: str | None, date: str) -> dict:
        """Elo adjustment for `team` starting `uid` (None/'' = TBD) on
        `date`. Returns adj plus the inputs for audit columns."""
        league = self.league.value(LEAGUE_SEED)
        staff_ew = self.staffs.get(team)
        staff = staff_ew.value(league) if staff_ew else league

        for ew, label in ((self.pitchers.get(uid) if uid else None, uid),
                          (staff_ew, team), (self.league, "league")):
            if ew and ew.last_date and ew.last_date >= date:
                raise LeakageError(
                    f"rGS for {label} contains a start dated {ew.last_date} "
                    f">= game date {date}")

        p_ew = self.pitchers.get(uid) if uid else None
        if p_ew and p_ew.n >= self.min_starts:
            mode, effective = "pitcher", p_ew.value(league)
        elif uid:
            mode = "thin"
            effective = league + self.rookie_shrink * (staff - league)
        else:
            mode, effective = "staff", staff

        return {
            "adj": self.c * (effective - staff),
            "mode": mode,
            "effective_rgs": effective,
            "staff_rgs": staff,
            "league_rgs": league,
            "career_starts": p_ew.n if p_ew else 0,
        }
