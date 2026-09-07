"""Four club-soccer outcome models on this year's matches — the data
behind reports/soccer/sides_vs_diffs_2026.html.

Every model is trained on league matches dated before TEST_FROM
(2026-01-01) and tested on every match played this calendar year (the
back half of the 2025-26 European seasons, MLS 2026, the first weeks of
2026-27). Nothing in the test window touches a fit.

    A  diffs · logistic         the production model: multinomial logistic on
                                the venue-adjusted Elo gap and five
                                home-minus-away differentials
    B  sides · logistic         the same information split into its home and
                                away values (home Elo, away Elo, home spend z,
                                away spend z, …) — no differences anywhere
    C  sides + goals · logistic B plus each side's rolling goals-scored and
                                goals-allowed factors (last GOALS_HALF_LIFE
                                matches, shrunk toward the league mean)
    D  goals regressor          C's inputs feeding two Poisson regressions —
                                home goals and away goals — with W/D/L read
                                off the joint score grid

For every model: 2026 log loss, accuracy, multiclass Brier, McFadden R²
against the class-frequency baseline, expected calibration error, the
home-win calibration slope, how it handles draws, per-league log loss,
the paired per-match Δlog-loss against A with a standard error,
permutation importance (Δ log loss when one feature is shuffled,
PERM_REPEATS repeats) and SHAP mean |φ| (exact linear attribution on the
logit or log-rate: |coef · (x − x̄)|). D also reports goals MAE / RMSE
per side and the exact-score hit rate.

Writes artifacts/sides_vs_diffs_2026.json; `render_sides_report.py`
turns it into the HTML page.

    python -m soccer.clubs.model.report_sides_2026
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from soccer.clubs.daily.config import MAX_GOALS
from soccer.clubs.daily.scoring import score_grid
from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model.compare_sides import ARTIFACTS, CLASSES, DIFFS, SIDES, build_table
from soccer.clubs.model.train import MAX_ITER

TEST_FROM = "2026-01-01"
PERM_REPEATS = 10
GOALS_HALF_LIFE = 10.0     # matches
GOALS_PRIOR = 8.0          # matches of league-mean prior in the shrinkage
GOAL_FEATURES = ["home_att", "home_def", "away_att", "away_def"]
OUT = ARTIFACTS / "sides_vs_diffs_2026.json"

MODELS = {
    "A": {"name": "Differences · logistic", "features": DIFFS, "kind": "logistic",
          "short": "diffs"},
    "B": {"name": "Sides · logistic", "features": SIDES, "kind": "logistic",
          "short": "sides"},
    "C": {"name": "Sides + goals form · logistic", "features": SIDES + GOAL_FEATURES,
          "kind": "logistic", "short": "sides+goals"},
    "D": {"name": "Goals regressor · Poisson", "features": SIDES + GOAL_FEATURES,
          "kind": "poisson", "short": "goals"},
}


# --- goals form ---------------------------------------------------------------

def attach_goal_form(table: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward attack / defence factors per side: an exponentially
    weighted goals-for and goals-against rate over the club's league
    matches (half-life GOALS_HALF_LIFE), shrunk toward the league's running
    mean goals per team-match with a GOALS_PRIOR-match prior, expressed as
    a ratio to that mean. 1.0 = league-average; strictly pre-match."""
    gamma = 0.5 ** (1.0 / GOALS_HALF_LIFE)
    t = table.sort_values(["date", "league", "home_team"], kind="stable")
    w: dict = {}; gf: dict = {}; ga: dict = {}
    lw: dict = {}; lpts: dict = {}
    out = {c: np.zeros(len(t)) for c in GOAL_FEATURES}

    def factor(league, team, store):
        L = lpts[league] / lw[league] if lw.get(league, 0) > 20 else 1.35
        n = w.get((league, team), 0.0)
        rate = (store.get((league, team), 0.0) + GOALS_PRIOR * L) / (n + GOALS_PRIOR)
        return rate / L

    for i, r in enumerate(t.itertuples(index=False)):
        lg, h, a = r.league, r.home_team, r.away_team
        out["home_att"][i] = factor(lg, h, gf); out["home_def"][i] = factor(lg, h, ga)
        out["away_att"][i] = factor(lg, a, gf); out["away_def"][i] = factor(lg, a, ga)
        hs, as_ = float(r.home_score), float(r.away_score)
        for team, f, g in ((h, hs, as_), (a, as_, hs)):
            k = (lg, team)
            w[k] = w.get(k, 0.0) * gamma + 1.0
            gf[k] = gf.get(k, 0.0) * gamma + f
            ga[k] = ga.get(k, 0.0) * gamma + g
        lw[lg] = lw.get(lg, 0.0) * 0.998 + 2.0
        lpts[lg] = lpts.get(lg, 0.0) * 0.998 + hs + as_
    for c in GOAL_FEATURES:
        t[c] = out[c]
    return t.sort_index()


# --- models -------------------------------------------------------------------

class PoissonOutcome:
    """Two Poisson regressions (home goals, away goals) on standardised
    features; outcome probabilities from the joint independent-Poisson
    score grid, the same construction the daily score model uses."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.home = PoissonRegressor(alpha=1e-4, max_iter=3000, tol=1e-8)
        self.away = PoissonRegressor(alpha=1e-4, max_iter=3000, tol=1e-8)
        self.classes_ = np.array(CLASSES)

    def fit(self, X, hs, as_):
        Z = self.scaler.fit_transform(X)
        self.home.fit(Z, hs); self.away.fit(Z, as_)
        return self

    def lambdas(self, X):
        Z = self.scaler.transform(X)
        return self.home.predict(Z), self.away.predict(Z)

    def predict_proba(self, X):
        lh, la = self.lambdas(X)
        probs = np.empty((len(X), 3))
        for i, (h, a) in enumerate(zip(lh, la)):
            g = score_grid(max(h, 0.05), max(a, 0.05))
            # grid[i, j] = P(home = i, away = j): above the diagonal the away
            # side scored more (A), on it a draw, below it the home side (H).
            probs[i] = [np.triu(g, 1).sum(), np.trace(g), np.tril(g, -1).sum()]   # A, D, H
        return probs

    def coef_on_lograte(self):
        """Coefficients per standardised feature, home-goal and away-goal rates."""
        return self.home.coef_, self.away.coef_


def fit_model(spec: dict, train: pd.DataFrame):
    X = train[spec["features"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec["kind"] == "logistic":
            return LogisticRegression(max_iter=MAX_ITER, tol=1e-10).fit(X, train["outcome"])
        return PoissonOutcome().fit(X, train["home_score"].to_numpy(), train["away_score"].to_numpy())


def proba(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return np.clip(p[:, order], 1e-9, 1.0)


# --- KPIs ---------------------------------------------------------------------

def kpis(p: np.ndarray, y: np.ndarray, freq: np.ndarray) -> dict:
    idx = np.array([CLASSES.index(c) for c in y])
    onehot = np.eye(3)[idx]
    ll_rows = -np.log(p[np.arange(len(y)), idx])
    ll_null = float(-np.log(np.clip(freq[idx], 1e-9, 1)).mean())
    pick = p.argmax(axis=1)
    conf = p.max(axis=1)
    # ECE on the picked class, 10 equal-width bins.
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pick[m] == idx[m]).mean() - conf[m].mean())
    # Home-win calibration slope: logit(P(H)) refit on the home-win indicator.
    ph = np.clip(p[:, 2], 1e-6, 1 - 1e-6)
    logit = np.log(ph / (1 - ph)).reshape(-1, 1)
    slope = float(LogisticRegression(C=1e6).fit(logit, (idx == 2).astype(int)).coef_[0][0])
    draws = pick == 1
    return {
        "n": int(len(y)),
        "accuracy": float((pick == idx).mean()),
        "log_loss": float(ll_rows.mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "mcfadden_r2": float(1 - ll_rows.mean() / ll_null),
        "ece": float(ece),
        "home_cal_slope": slope,
        "draw_picks": int(draws.sum()),
        "draw_pick_accuracy": float((idx[draws] == 1).mean()) if draws.any() else None,
        "mean_p_draw": float(p[:, 1].mean()),
        "pick_mix": {c: int((pick == i).sum()) for i, c in enumerate(CLASSES)},
    }, ll_rows


# --- importances --------------------------------------------------------------

def permutation(model, spec, X: np.ndarray, y: np.ndarray, base_ll: float,
                rng: np.random.Generator) -> list[dict]:
    idx = np.array([CLASSES.index(c) for c in y])
    out = []
    for j, f in enumerate(spec["features"]):
        deltas = []
        for _ in range(PERM_REPEATS):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pp = proba(model, Xp)
            deltas.append(float(-np.log(pp[np.arange(len(y)), idx]).mean() - base_ll))
        out.append({"feature": f, "mean": float(np.mean(deltas)), "std": float(np.std(deltas))})
    return out


def shap_linear(model, spec, X_test: np.ndarray, X_train: np.ndarray) -> list[dict]:
    if spec["kind"] == "logistic":
        centered = X_test - X_train.mean(axis=0)
        phi = np.abs(centered[None, :, :] * model.coef_[:, None, :]).mean(axis=(0, 1))
    else:
        Z = model.scaler.transform(X_test)
        ch, ca = model.coef_on_lograte()
        phi = 0.5 * (np.abs(Z * ch).mean(axis=0) + np.abs(Z * ca).mean(axis=0))
    return [{"feature": f, "mean_abs": float(phi[i])} for i, f in enumerate(spec["features"])]


def direction(model, spec) -> list[dict]:
    """Signed association per feature: the H-class coefficient for a
    logistic; for the Poisson, home-goal minus away-goal log-rate
    coefficient (positive = tilts the match toward the home side)."""
    if spec["kind"] == "logistic":
        h = list(model.classes_).index("H")
        coefs = model.coef_[h]
    else:
        ch, ca = model.coef_on_lograte()
        coefs = ch - ca
    return [{"feature": f, "coef": float(coefs[i])} for i, f in enumerate(spec["features"])]


# --- main ---------------------------------------------------------------------

def main() -> None:
    table = attach_goal_form(build_table())
    train = table[table["date"] < TEST_FROM]
    test = table[table["date"] >= TEST_FROM]
    y_train, y_test = train["outcome"].to_numpy(), test["outcome"].to_numpy()
    freq = np.array([(y_train == c).mean() for c in CLASSES])
    print(f"train {len(train)} (< {TEST_FROM}); test {len(test)} matches in 2026")

    rng = np.random.default_rng(17)
    results, rows = {}, {}
    for key, spec in MODELS.items():
        m = fit_model(spec, train)
        Xte, Xtr = test[spec["features"]].to_numpy(), train[spec["features"]].to_numpy()
        p = proba(m, Xte)
        k, ll_rows = kpis(p, y_test, freq)
        rows[key] = ll_rows
        by_league = {}
        for lg in LEAGUES:
            mask = (test["league"] == lg).to_numpy()
            if mask.any():
                pick = p[mask].argmax(axis=1)
                idx = np.array([CLASSES.index(c) for c in y_test[mask]])
                by_league[lg] = {"n": int(mask.sum()), "log_loss": float(ll_rows[mask].mean()),
                                 "accuracy": float((pick == idx).mean())}
        entry = {**spec, "kpis": k, "by_league": by_league,
                 "permutation": permutation(m, spec, Xte, y_test, k["log_loss"], rng),
                 "shap": shap_linear(m, spec, Xte, Xtr),
                 "direction": direction(m, spec)}
        if spec["kind"] == "logistic":
            entry["converged"] = bool(int(m.n_iter_[0]) < MAX_ITER)
        else:
            lh, la = m.lambdas(Xte)
            hs, as_ = test["home_score"].to_numpy(), test["away_score"].to_numpy()
            exact = 0
            for h, a, th, ta in zip(lh, la, hs, as_):
                g = score_grid(max(h, 0.05), max(a, 0.05))
                i, j = np.unravel_index(int(g.argmax()), g.shape)
                exact += int(i == th and j == ta)
            entry["goals"] = {
                "home_mae": float(np.abs(lh - hs).mean()), "home_rmse": float(np.sqrt(((lh - hs) ** 2).mean())),
                "away_mae": float(np.abs(la - as_).mean()), "away_rmse": float(np.sqrt(((la - as_) ** 2).mean())),
                "total_mae": float(np.abs(lh + la - hs - as_).mean()),
                "mean_pred_home": float(lh.mean()), "mean_actual_home": float(hs.mean()),
                "mean_pred_away": float(la.mean()), "mean_actual_away": float(as_.mean()),
                "exact_score_rate": exact / len(test),
                "naive_home_mae": float(np.abs(hs - hs.mean()).mean()),
                "naive_away_mae": float(np.abs(as_ - as_.mean()).mean()),
            }
        results[key] = entry
        print(f"  {key} {spec['name']:32s} ll {k['log_loss']:.4f}  acc {k['accuracy']:.3f}  "
              f"brier {k['brier']:.4f}  R2 {k['mcfadden_r2']:.4f}")

    paired = {}
    for key in ("B", "C", "D"):
        d = rows[key] - rows["A"]
        paired[key] = {"all": {"n": int(len(d)), "d_ll_mean": float(d.mean()),
                               "d_ll_se": float(d.std(ddof=1) / np.sqrt(len(d)))},
                       "by_league": {}}
        for lg in LEAGUES:
            mask = (test["league"] == lg).to_numpy()
            if mask.sum() > 1:
                dd = d[mask]
                paired[key]["by_league"][lg] = {"n": int(len(dd)), "d_ll_mean": float(dd.mean()),
                                                "d_ll_se": float(dd.std(ddof=1) / np.sqrt(len(dd)))}

    ll_null = float(-np.log(np.clip(freq[[CLASSES.index(c) for c in y_test]], 1e-9, 1)).mean())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_from": TEST_FROM,
        "train_matches": int(len(train)), "test_matches": int(len(test)),
        "test_date_range": [str(test["date"].min()), str(test["date"].max())],
        "test_outcomes": {c: int((y_test == c).sum()) for c in CLASSES},
        "train_frequencies": {c: float(freq[i]) for i, c in enumerate(CLASSES)},
        "frequency_log_loss": ll_null,
        "league_names": {k: v.name for k, v in LEAGUES.items()},
        "perm_repeats": PERM_REPEATS,
        "goals_half_life": GOALS_HALF_LIFE, "goals_prior": GOALS_PRIOR,
        "models": results, "paired": paired,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
