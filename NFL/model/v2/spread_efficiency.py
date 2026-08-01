"""Is our predicted spread an efficient forecast — and is the market's?

MAE answers "how wrong". Efficiency answers the question that decides whether
any of this is tradable: **is there information left in the forecast error that
something observable could have predicted?** A forecast can be less accurate
than a rival and still carry information the rival lacks; it can also be
unbiased on average and badly biased inside every stratum. Neither shows up in
a mean absolute error.

Four tests, in increasing order of how much they matter to a bettor.

## 1. Mincer-Zarnowitz

    margin = a + b * forecast + e

An efficient forecast has ``a = 0, b = 1``. ``b < 1`` means the forecast is too
aggressive — it should be shrunk toward zero; ``b > 1`` means too timid. Tested
jointly with a cluster-robust Wald statistic.

## 2. Forecast encompassing

    margin = a + b1 * ours + b2 * vegas

If ``b1 = 0`` the market **encompasses** our model: everything we know, they
already knew, and our forecast is redundant. If ``b2 = 0`` we encompass them.
Both non-zero means each carries information the other lacks, and the fitted
weights are the optimal way to combine them. This is the honest version of "our
model is nearly as good" — nearly as good and redundant is worth nothing;
worse but independent is worth something.

## 3. Market-residual regression — the one that pays

    (margin - vegas) = a + b * (ours - vegas)

Regress the bookmaker's *error* on our *disagreement*. Under market efficiency
with respect to our information set, ``b = 0``: knowing we differ by 3 points
tells you nothing about which way the line missed. ``b > 0`` means each point
of disagreement buys ``b`` points of expected edge against the closing number,
which converts straight into a cover rate. This is algebraically the optimal
combination weight on our forecast, so it does double duty with test 2.

## 4. Stratified bias

An unbiased forecast must be unbiased *conditionally*. We slice the residual by
favourite size, home/away, week, venue and rest to find pockets where either
forecast is systematically off.

Standard errors are clustered on ``season-week`` throughout: games in the same
week share weather, news cycles and correlated line moves, so treating 3,028
games as independent would overstate significance by roughly the square root of
the games-per-week.

Usage
    python3 -m NFL.model.v2.spread_efficiency
    python3 -m NFL.model.v2.spread_efficiency --model ridge --since 2025-10-04
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "spread"

# -110 both ways: you must cover this often to break even.
BREAK_EVEN = 110 / 210


# --------------------------------------------------------------------------
# cluster-robust OLS
# --------------------------------------------------------------------------

class ClusterOLS:
    """OLS with one-way cluster-robust (Liang-Zeger) standard errors."""

    def __init__(self, y, X, clusters, names: list[str]):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.names = names
        n, k = self.X.shape
        XtX_inv = np.linalg.pinv(self.X.T @ self.X)
        self.beta = XtX_inv @ self.X.T @ self.y
        self.resid = self.y - self.X @ self.beta

        codes = pd.factorize(pd.Series(clusters))[0]
        G = codes.max() + 1
        meat = np.zeros((k, k))
        for g in range(G):
            m = codes == g
            Xg, ug = self.X[m], self.resid[m]
            s = Xg.T @ ug
            meat += np.outer(s, s)
        # Standard finite-sample correction (Cameron & Miller).
        c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        self.vcov = c * XtX_inv @ meat @ XtX_inv
        self.se = np.sqrt(np.diag(self.vcov))
        self.n, self.k, self.G = n, k, G
        ss_res = float(self.resid @ self.resid)
        ss_tot = float(((self.y - self.y.mean()) ** 2).sum())
        self.r2 = 1 - ss_res / ss_tot if ss_tot else np.nan

    def t_against(self, i: int, value: float = 0.0) -> tuple[float, float]:
        """t-stat and two-sided p for beta_i == value, on G-1 df.

        A zero standard error means the residual is degenerate — two identical
        forecasts, say. That is "no detectable difference", not a divide-by-zero.
        """
        if self.se[i] == 0 or not np.isfinite(self.se[i]):
            return 0.0, 1.0
        t = (self.beta[i] - value) / self.se[i]
        return float(t), float(2 * stats.t.sf(abs(t), df=self.G - 1))

    def wald(self, R: np.ndarray, r: np.ndarray) -> tuple[float, float]:
        """Joint test R @ beta == r. Returns (chi2, p) on q df."""
        R, r = np.atleast_2d(R), np.atleast_1d(r)
        d = R @ self.beta - r
        stat = float(d @ np.linalg.pinv(R @ self.vcov @ R.T) @ d)
        return stat, float(stats.chi2.sf(stat, df=R.shape[0]))

    def row(self, label: str) -> dict:
        out = {"spec": label, "n": self.n, "clusters": self.G, "r2": round(self.r2, 4)}
        for i, nm in enumerate(self.names):
            out[nm] = round(float(self.beta[i]), 4)
            out[f"se_{nm}"] = round(float(self.se[i]), 4)
        return out


def _blocks(d: pd.DataFrame) -> pd.Series:
    return d["season"].astype(str) + "_" + d["week"].astype(str)


# --------------------------------------------------------------------------
# the four tests
# --------------------------------------------------------------------------

def mincer_zarnowitz(d: pd.DataFrame, forecast: str, label: str) -> dict:
    """margin = a + b * forecast. Efficient iff (a, b) == (0, 1)."""
    X = np.column_stack([np.ones(len(d)), d[forecast]])
    m = ClusterOLS(d["margin"], X, _blocks(d), ["a", "b"])
    t_b, p_b = m.t_against(1, 1.0)
    t_a, p_a = m.t_against(0, 0.0)
    chi2, p_joint = m.wald(np.eye(2), np.array([0.0, 1.0]))
    return {"forecast": label, "n": m.n,
            "a": round(float(m.beta[0]), 3), "se_a": round(float(m.se[0]), 3),
            "b": round(float(m.beta[1]), 3), "se_b": round(float(m.se[1]), 3),
            "t(a=0)": round(t_a, 2), "t(b=1)": round(t_b, 2),
            "p(b=1)": round(p_b, 4), "joint_chi2": round(chi2, 2),
            "p_joint": round(p_joint, 4),
            "efficient": "yes" if p_joint > 0.05 else "no"}


def encompassing(d: pd.DataFrame, pred: str = "pred_margin") -> tuple[dict, ClusterOLS]:
    """margin = a + b1 * ours + b2 * vegas. b1 = 0 => the market encompasses us."""
    X = np.column_stack([np.ones(len(d)), d[pred], d["spread_line"]])
    m = ClusterOLS(d["margin"], X, _blocks(d), ["a", "b_ours", "b_vegas"])
    t1, p1 = m.t_against(1, 0.0)
    t2, p2 = m.t_against(2, 0.0)
    return {"n": m.n, "clusters": m.G,
            "b_ours": round(float(m.beta[1]), 4), "se_ours": round(float(m.se[1]), 4),
            "t_ours": round(t1, 2), "p_ours": round(p1, 4),
            "b_vegas": round(float(m.beta[2]), 4), "se_vegas": round(float(m.se[2]), 4),
            "t_vegas": round(t2, 2), "p_vegas": round(p2, 4),
            "r2": round(m.r2, 4)}, m


def market_residual(d: pd.DataFrame, pred: str = "pred_margin") -> dict:
    """(margin - vegas) = a + b * (ours - vegas).

    ``b`` is expected points of edge per point of disagreement, and the optimal
    weight to place on our forecast when combining. ``b = 0`` is market
    efficiency with respect to everything our model knows.
    """
    disagree = d[pred] - d["spread_line"]
    resid = d["margin"] - d["spread_line"]
    X = np.column_stack([np.ones(len(d)), disagree])
    m = ClusterOLS(resid, X, _blocks(d), ["a", "b"])
    t, p = m.t_against(1, 0.0)

    # What b implies for betting, and what actually happened.
    graded = resid != 0
    side_won = np.where(disagree[graded] > 0, resid[graded] > 0, resid[graded] < 0)
    lean = graded & (disagree.abs() >= 1.0)
    lean_won = np.where(disagree[lean] > 0, resid[lean] > 0, resid[lean] < 0)
    return {"n": m.n, "clusters": m.G,
            "b": round(float(m.beta[1]), 4), "se_b": round(float(m.se[1]), 4),
            "t": round(t, 2), "p": round(p, 4), "r2": round(m.r2, 4),
            "ats_all": round(float(side_won.mean()), 4), "n_ats": int(graded.sum()),
            "ats_disagree_1pt": round(float(lean_won.mean()), 4),
            "n_lean": int(lean.sum()),
            "break_even": round(BREAK_EVEN, 4)}


def required_disagreement(d: pd.DataFrame, b: float,
                          pred: str = "pred_margin") -> dict:
    """How far from the line must we be before the edge covers the vig?

    With ``b`` points of expected edge per point of disagreement and residuals
    roughly normal with SD sigma around the close, a disagreement of ``x``
    implies a cover probability of ``Phi(b*x/sigma)``. Setting that equal to the
    -110 break-even and solving gives the disagreement a bet needs before it is
    worth making — and, usually, how few games ever get there.
    """
    resid = d["margin"] - d["spread_line"]
    disagree = (d[pred] - d["spread_line"]).abs()
    sigma = float(resid.std())
    z = float(stats.norm.ppf(BREAK_EVEN))
    need = z * sigma / b if b > 0 else float("inf")
    return {"resid_sd": round(sigma, 2), "b": round(b, 4),
            "z_for_break_even": round(z, 4),
            "disagreement_needed": round(need, 1),
            "games_that_far_off": int((disagree >= need).sum()) if np.isfinite(need) else 0,
            "of_n": len(d)}


def implied_vs_actual(d: pd.DataFrame, b: float, pred: str = "pred_margin",
                      bins=(0, 1, 2, 3, 5, 7, 40)) -> pd.DataFrame:
    """Cover rate the edge model predicts, against the cover rate observed."""
    resid = d["margin"] - d["spread_line"]
    disagree = d[pred] - d["spread_line"]
    sigma = float(resid.std())
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (disagree.abs() >= lo) & (disagree.abs() < hi) & (resid != 0)
        if m.sum() < 40:
            continue
        won = np.where(disagree[m] > 0, resid[m] > 0, resid[m] < 0)
        rows.append({
            "disagreement": f"{lo}-{hi}", "n": int(m.sum()),
            "implied_cover": round(float(
                stats.norm.cdf(b * disagree[m].abs().mean() / sigma)), 4),
            "actual_cover": round(float(won.mean()), 4),
            "se": round(float(np.sqrt(.25 / m.sum())), 4),
            "break_even": round(BREAK_EVEN, 4)})
    return pd.DataFrame(rows)


def diebold_mariano(d: pd.DataFrame, pred: str = "pred_margin",
                    loss: str = "abs") -> dict:
    """Equal-accuracy test between our forecast and the closing line.

    Loss differential per game, averaged, with the standard error clustered on
    week — a negative statistic favours us, positive favours the book.
    """
    e_ours = d["margin"] - d[pred]
    e_vegas = d["margin"] - d["spread_line"]
    if loss == "abs":
        diff = e_ours.abs() - e_vegas.abs()
    else:
        diff = e_ours ** 2 - e_vegas ** 2
    m = ClusterOLS(diff, np.ones((len(d), 1)), _blocks(d), ["mean"])
    t, p = m.t_against(0, 0.0)
    return {"loss": loss, "mean_diff": round(float(m.beta[0]), 4),
            "se": round(float(m.se[0]), 4), "dm_stat": round(t, 2),
            "p": round(p, 4),
            "verdict": ("vegas better" if t > 0 and p < .05 else
                        "ours better" if t < 0 and p < .05 else "indistinguishable")}


def stratified_bias(d: pd.DataFrame, pred: str = "pred_margin") -> pd.DataFrame:
    """Conditional bias. An unbiased forecast is unbiased in every slice."""
    d = d.copy()
    d["fav_size"] = pd.cut(d["spread_line"].abs(), [-.01, 3, 7, 10, 60],
                           labels=["0-3", "3-7", "7-10", "10+"])
    d["half"] = np.where(d["week"] <= 9, "weeks 1-9", "weeks 10+")
    d["side"] = np.where(d["spread_line"] > 0, "home favoured", "away favoured")

    rows = []
    for col in ("fav_size", "half", "side"):
        for lvl, g in d.groupby(col, observed=True):
            if len(g) < 60:
                continue
            for label, f in (("ours", pred), ("vegas", "spread_line")):
                e = g["margin"] - g[f]
                blk = _blocks(g)
                m = ClusterOLS(e, np.ones((len(g), 1)), blk, ["mean"])
                t, p = m.t_against(0, 0.0)
                rows.append({"cut": col, "level": str(lvl), "forecast": label,
                             "games": len(g), "mean_resid": round(float(e.mean()), 3),
                             "se": round(float(m.se[0]), 3), "t": round(t, 2),
                             "p": round(p, 4),
                             "flag": "*" if p < .05 else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def load_preds(model: str, since: str | None) -> pd.DataFrame:
    p = ARTIFACTS / f"preds_{model}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 -m NFL.model.v2.spread_model "
                         f"--evaluate` first")
    d = pd.read_csv(p).dropna(subset=["spread_line", "margin", "pred_margin"])
    if since:
        from .dataset import load_games
        g = load_games()[["game_id", "gameday"]]
        d = d.merge(g, on="game_id", how="left")
        d = d[pd.to_datetime(d["gameday"]) >= since]
    return d.reset_index(drop=True)


def _fmt(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="extratrees")
    ap.add_argument("--since", default=None,
                    help="ISO date; restrict to games on or after it")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    d = load_preds(args.model, args.since)
    span = f"{d['season'].min()}-{d['season'].max()}"
    print(f"=== spread efficiency: {args.model}, {len(d)} games, {span} "
          f"{'since ' + args.since if args.since else ''} ===\n")

    print("--- 1. Mincer-Zarnowitz: margin = a + b*forecast, efficient iff (0, 1) ---")
    mz = [mincer_zarnowitz(d, "pred_margin", f"ours ({args.model})"),
          mincer_zarnowitz(d, "spread_line", "closing spread")]
    print(_fmt(mz), "\n")

    print("--- 2. Encompassing: margin = a + b1*ours + b2*vegas ---")
    enc, _ = encompassing(d)
    print(_fmt([enc]))
    verdict = ("the market encompasses our model (b_ours indistinguishable from 0)"
               if enc["p_ours"] > .05 else
               "our model carries information the market lacks")
    print(f"  -> {verdict}\n")

    print("--- 3. Market residual on our disagreement (the tradable test) ---")
    mr = market_residual(d)
    print(_fmt([mr]))
    edge = mr["b"]
    print(f"  -> each point of disagreement is worth {edge:+.3f} points against "
          f"the close\n")

    print("--- 3b. Is that edge economically usable? ---")
    rq = required_disagreement(d, edge)
    print(_fmt([rq]))
    iva = implied_vs_actual(d, edge)
    print(iva.to_string(index=False), "\n")

    print("--- 4. Diebold-Mariano, equal predictive accuracy ---")
    print(_fmt([diebold_mariano(d, loss="abs"), diebold_mariano(d, loss="sq")]), "\n")

    print("--- 5. Conditional bias by stratum ---")
    sb = stratified_bias(d)
    print(sb.to_string(index=False))

    if args.save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.since}" if args.since else ""
        pd.DataFrame(mz).to_csv(ARTIFACTS / f"eff_mz_{args.model}{suffix}.csv", index=False)
        pd.DataFrame([enc]).to_csv(ARTIFACTS / f"eff_encompass_{args.model}{suffix}.csv", index=False)
        pd.DataFrame([mr]).to_csv(ARTIFACTS / f"eff_residual_{args.model}{suffix}.csv", index=False)
        sb.to_csv(ARTIFACTS / f"eff_bias_{args.model}{suffix}.csv", index=False)
        print(f"\nsaved -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
