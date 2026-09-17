"""Leakage-safe pregame form: rolling and exponentially weighted team
statistics that a row is only allowed to see from *earlier* games.

The one rule, enforced in one place: a team-game row's feature is
computed from the rows strictly before it in that team's own sequence.
`shifted_rolling` and `shifted_ewm` both `shift(1)` inside each team
group before rolling, so the game being predicted never contributes to
its own feature. `assert_no_same_game_leak` is the test-time check that
this held for a whole table.

Half-life convention: an EWMA with half-life h weights a game k games ago
by 0.5 ** (k / h). Five is the default everywhere here (≈ a month of
soccer, a third of an NFL season), so the sports read the same way.

Small samples are shrunk toward a prior — the league mean of the same
statistic over the training window — with a weight of `prior_games`
pseudo-observations, so a team two games into a season is mostly the
league and a team twelve games in is mostly itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HALF_LIFE = 5.0


def shifted_rolling(df: pd.DataFrame, group: str, cols: list[str], window: int,
                    min_periods: int = 1, order: list[str] | None = None,
                    suffix: str | None = None) -> pd.DataFrame:
    """Per-`group` rolling mean of `cols` over the previous `window` rows.

    `df` must already be in chronological order within each group (pass
    `order` to sort it first). The current row is excluded by
    construction.
    """
    out = df.sort_values(order) if order else df
    g = out.groupby(group, sort=False)[cols]
    rolled = g.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    rolled.columns = [f"{c}_{suffix or f'r{window}'}" for c in cols]
    return pd.concat([out, rolled], axis=1).sort_index() if order else pd.concat([out, rolled], axis=1)


def shifted_ewm(df: pd.DataFrame, group: str, cols: list[str],
                half_life: float = DEFAULT_HALF_LIFE, min_periods: int = 1,
                order: list[str] | None = None, suffix: str | None = None) -> pd.DataFrame:
    """Per-`group` exponentially weighted mean of `cols`, previous rows
    only, half-life in games."""
    out = df.sort_values(order) if order else df
    g = out.groupby(group, sort=False)[cols]
    ewm = g.transform(lambda s: s.shift(1).ewm(halflife=half_life, min_periods=min_periods).mean())
    ewm.columns = [f"{c}_{suffix or 'ewm'}" for c in cols]
    return pd.concat([out, ewm], axis=1).sort_index() if order else pd.concat([out, ewm], axis=1)


def shifted_count(df: pd.DataFrame, group: str, order: list[str] | None = None,
                  name: str = "games_before") -> pd.Series:
    """How many earlier rows the group has — the sample size behind a
    shifted statistic, for shrinkage."""
    out = df.sort_values(order) if order else df
    n = out.groupby(group, sort=False).cumcount()
    return n.reindex(df.index) if order else n


def shrink(value: pd.Series, n: pd.Series, prior: float | pd.Series,
           prior_games: float) -> pd.Series:
    """Shrink a per-team estimate from `n` games toward `prior` with
    `prior_games` pseudo-observations. NaN (no games yet) becomes the
    prior outright."""
    n = n.fillna(0).astype(float)
    v = value.where(value.notna(), prior)
    return (n * v + prior_games * prior) / (n + prior_games)


def days_since_previous(df: pd.DataFrame, group: str, date_col: str,
                        order: list[str] | None = None) -> pd.Series:
    """Rest: days between this row's date and the group's previous row."""
    out = df.sort_values(order or [date_col])
    d = pd.to_datetime(out[date_col])
    prev = out.assign(_d=d).groupby(group, sort=False)["_d"].shift(1)
    rest = (d - prev).dt.days
    return rest.reindex(df.index)


def count_in_previous_days(df: pd.DataFrame, group: str, date_col: str,
                           days: int) -> pd.Series:
    """How many of the group's earlier rows fall within `days` before this
    one (fixture congestion). O(n · window) per group, fine at league
    scale."""
    d = pd.to_datetime(df[date_col])
    out = pd.Series(0, index=df.index, dtype=int)
    for _, idx in df.groupby(group, sort=False).groups.items():
        dates = d.loc[idx].sort_values()
        vals = dates.to_numpy()
        counts = np.zeros(len(vals), dtype=int)
        j = 0
        for i in range(len(vals)):
            while vals[i] - vals[j] > np.timedelta64(days, "D"):
                j += 1
            counts[i] = i - j   # rows in (date - days, date), excluding self
        out.loc[dates.index] = counts
    return out


def assert_no_same_game_leak(df: pd.DataFrame, group: str, order: list[str],
                             raw: str, feature: str) -> None:
    """Every feature value must be reproducible from strictly earlier raw
    values of the same group. Raises AssertionError with the first row
    that isn't.

    Works for any shifted mean: recomputes the *first* non-null feature of
    each group and checks it equals the group's first raw value (the only
    thing a shift(1) could have seen), and checks the first row of every
    group is null.
    """
    s = df.sort_values(order)
    for key, g in s.groupby(group, sort=False):
        if g[feature].notna().iloc[0]:
            raise AssertionError(
                f"{feature}: first row for {group}={key!r} is not null — "
                f"the row saw its own game")
        if len(g) > 1 and g[feature].notna().iloc[1]:
            got, want = g[feature].iloc[1], g[raw].iloc[0]
            if not np.isclose(got, want, equal_nan=True):
                raise AssertionError(
                    f"{feature}: second row for {group}={key!r} is {got}, "
                    f"expected the first raw value {want}")
