"""Weekly models check: one email walking through every forecasting
model's current performance — MLB, soccer, NFL — plus feature
importances (permutation + SHAP) and each feature's association with the
model, for the models that have a feature vector to attribute.

    python -m data_jobs.reports.models_check [--date YYYY-MM-DD]

What each section reports, and where it comes from:

- MLB — the daily Elo pipeline's own graded ledger
  (data/mlb/predictions/grades.csv): cumulative accuracy/log loss and the
  Δlog-loss edge vs. the always-pick-home baseline. The model is Elo +
  home advantage + the starting-pitcher adjustment — there is no ML
  feature vector, so there is nothing for SHAP/permutation to attribute;
  the honest check is the graded record vs. baseline, and the email says
  exactly that.
- Soccer — the club pipeline's ledger (cumulative + rolling 7d/30d), and
  live-computed importances for the outcome model: the last ~15% of the
  replay history (by date) is held out, the multinomial logistic is refit
  on the rest, then (a) permutation importance = Δ held-out log loss when
  one feature is shuffled, and (b) exact linear-SHAP mean |φ| (for a
  linear model φ_ij = coef_j · (x_ij − x̄_j), no shap package needed).
  Associations come from the home-win coefficients' signs.
- NFL — the v2 walk-forward scorecard (artifacts/scorecard/kpis.csv) and
  the committed importance study (artifacts/importance/): the uniform
  cross-model ranking plus the production-family (LGBM) permutation +
  SHAP table. Those CSVs are regenerated offline by
  NFL/model/v2/feature_importance.py, not recomputed here — the training
  window is 24 seasons and this is a weekly email.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "reports" / "models_check"
MLB_GRADES = REPO / "data" / "mlb" / "predictions" / "grades.csv"
NFL_SCORECARD = REPO / "NFL" / "model" / "v2" / "artifacts" / "scorecard" / "kpis.csv"
NFL_IMPORTANCE = REPO / "NFL" / "model" / "v2" / "artifacts" / "importance"
SUMMARY_JSON = REPO / "web" / "public" / "data" / "summary.json"

HOLDOUT_FRAC = 0.15    # most recent slice of the soccer replay, by date
PERM_REPEATS = 5

STYLE_TABLE = (
    "border-collapse:collapse;width:100%;max-width:680px;"
    "font-family:Arial,Helvetica,sans-serif;font-size:13px;"
)
STYLE_TH = (
    "text-align:left;padding:5px 8px;border-bottom:2px solid #333;"
    "background:#f4f4f4;white-space:nowrap;"
)
STYLE_TD = "padding:5px 8px;border-bottom:1px solid #ddd;white-space:nowrap;"
STYLE_H2 = "font-size:16px;margin:26px 0 2px 0;"
STYLE_H3 = "margin:14px 0 4px 0;"
STYLE_NOTE = "color:#666;font-size:12px;margin:4px 0 0 0;"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th style='{STYLE_TH}'>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='{STYLE_TD}'>{c}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table style='{STYLE_TABLE}'><tr>{head}</tr>{body}</table>"


def _fmt(x, digits=4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "&mdash;"
    return f"{x:.{digits}f}"


# --- MLB --------------------------------------------------------------------

def mlb_section() -> str:
    if not MLB_GRADES.exists():
        return "<p>No MLB graded ledger found.</p>"
    g = pd.read_csv(MLB_GRADES)
    if g.empty:
        return "<p>MLB ledger is empty.</p>"
    last = g.iloc[-1]
    recent = g.tail(7)
    rows = [
        ["Season to date",
         f"{int(last['cum_correct'])}/{int(last['cum_games'])}",
         f"{100 * last['cum_accuracy']:.1f}%",
         _fmt(last["cum_log_loss"]), _fmt(last["cum_brier"])],
        [f"Last {len(recent)} graded days",
         f"{int(recent['correct'].sum())}/{int(recent['games'].sum())}",
         f"{100 * recent['correct'].sum() / max(recent['games'].sum(), 1):.1f}%",
         _fmt((recent["log_loss"] * recent["games"]).sum() / max(recent["games"].sum(), 1)),
         "&mdash;"],
    ]
    edge = (
        f"Edge vs. always-pick-home: &Delta;log-loss "
        f"<b>{last['cum_d_ll_mean']:+.4f} &plusmn; {last['cum_d_ll_se']:.4f}</b> "
        f"per game (positive = model beats the baseline). "
        if pd.notna(last.get("cum_d_ll_mean")) else ""
    )
    return (
        _table(["Window", "Record", "Accuracy", "Log loss", "Brier"], rows)
        + f"<p style='{STYLE_NOTE}'>{edge}Graded through {last['date']}. "
        f"The MLB model is betting-blind Elo + home advantage + the "
        f"starting-pitcher adjustment &mdash; it has no ML feature vector, so "
        f"there are no SHAP/permutation importances to report; the record vs. "
        f"baseline above is the model check.</p>"
    )


# --- Soccer -----------------------------------------------------------------

def soccer_performance() -> str:
    from soccer.clubs.daily import grade

    today = date.today().isoformat()
    ledger = grade.ledger_summary(today)
    if not ledger.get("graded"):
        return "<p>No soccer picks graded yet.</p>"

    def row(label: str, s: dict) -> list[str]:
        if not s or not s.get("graded"):
            return [label, "0", "&mdash;", "&mdash;"]
        return [label, str(s["graded"]), f"{100 * s['accuracy']:.1f}%",
                _fmt(s["log_loss"])]

    rolling = ledger.get("rolling", {})
    rows = [
        row("Last 7 days", rolling.get("7d", {})),
        row("Last 30 days", rolling.get("30d", {})),
        row("Season to date", ledger),
    ]
    rows += [
        row(f"&nbsp;&nbsp;{lg}", s)
        for lg, s in sorted(ledger.get("by_league", {}).items())
    ]
    return _table(["Window / league", "Graded", "Pick accuracy", "Log loss"], rows)


def soccer_importances() -> str:
    """Refit on a date split and measure held-out permutation + linear-SHAP
    importance for the club outcome model."""
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression

    from soccer.clubs.daily.state import FEATURES, MAX_ITER, build_state

    state = build_state()
    hist = state.history.sort_values("date")
    cut = hist["date"].iloc[int(len(hist) * (1 - HOLDOUT_FRAC))]
    train = hist[hist["date"] < cut]
    hold = hist[hist["date"] >= cut]

    model = LogisticRegression(max_iter=MAX_ITER, tol=1e-10)
    model.fit(train[FEATURES], train["outcome"])

    perm = permutation_importance(
        model, hold[FEATURES], hold["outcome"], scoring="neg_log_loss",
        n_repeats=PERM_REPEATS, random_state=17)

    # Exact SHAP for a linear model: phi_ij = coef_j * (x_ij - mean(x_j)),
    # per class; report the mean |phi| over holdout rows and classes.
    X = hold[FEATURES].to_numpy()
    centered = X - train[FEATURES].to_numpy().mean(axis=0)
    phi = np.abs(centered[None, :, :] * model.coef_[:, None, :])  # class,row,feat
    shap_mean = phi.mean(axis=(0, 1))

    home_idx = list(model.classes_).index("H")
    home_coef = model.coef_[home_idx]

    order = np.argsort(-perm.importances_mean)
    rows = []
    for i in order:
        assoc = "raises home-win odds" if home_coef[i] > 0 else "lowers home-win odds"
        rows.append([
            f"<b>{FEATURES[i]}</b>",
            f"{perm.importances_mean[i]:+.4f} &plusmn; {perm.importances_std[i]:.4f}",
            _fmt(shap_mean[i]),
            f"{home_coef[i]:+.3f} ({assoc})",
        ])
    return (
        _table(["Feature", "Permutation &Delta;log-loss", "Mean |SHAP|",
                "Home-win coef (association)"], rows)
        + f"<p style='{STYLE_NOTE}'>Multinomial logistic refit on matches "
        f"before {cut} and evaluated on the {len(hold)} most recent "
        f"({int(HOLDOUT_FRAC * 100)}%) held-out matches. Permutation = how much "
        f"held-out log loss degrades when the feature is shuffled "
        f"({PERM_REPEATS} repeats); SHAP is the exact linear attribution "
        f"|coef &middot; (x &minus; x&#772;)| averaged over rows and classes. "
        f"The coefficient column is the H-class weight on the z-scored "
        f"feature &mdash; its sign is the direction of association.</p>"
    )


# --- NFL --------------------------------------------------------------------

def nfl_section() -> str:
    parts = []
    if NFL_SCORECARD.exists():
        k = pd.read_csv(NFL_SCORECARD)
        rows = [
            [r["model"], str(int(r["n"])), f"{100 * r['accuracy']:.1f}%",
             _fmt(r["log_loss"]), _fmt(r["brier"]),
             _fmt(r.get("cal_slope"), 3), _fmt(r.get("ece"), 3)]
            for _, r in k.iterrows()
        ]
        parts.append(
            _table(["Model", "Games", "Accuracy", "Log loss", "Brier",
                    "Cal slope", "ECE"], rows)
            + f"<p style='{STYLE_NOTE}'>Walk-forward out-of-sample scorecard "
            f"(NFL/model/v2). The market row is the closing line &mdash; the "
            f"bar the model is judged against.</p>"
        )
    else:
        parts.append("<p>No NFL scorecard artifact found.</p>")

    uniform = NFL_IMPORTANCE / "uniform_ranking.csv"
    if uniform.exists():
        u = pd.read_csv(uniform).head(15)
        rows = [
            [str(int(r["uniform_rank"])), f"<b>{r['feature']}</b>",
             f"{r['avg_rank']:.1f}", _fmt(r.get("avg_norm_score"), 3)]
            for _, r in u.iterrows()
        ]
        parts.append(
            f"<h3 style='{STYLE_H3}'>Top 15 features &mdash; uniform ranking "
            f"across 5 model families</h3>"
            + _table(["#", "Feature", "Avg (perm+SHAP) rank", "Norm. score"], rows)
        )

    lgbm = NFL_IMPORTANCE / "perm_lgbm.csv"
    if lgbm.exists():
        l = pd.read_csv(lgbm).head(10)
        rows = [
            [f"<b>{r['feature']}</b>",
             f"{r['perm_importance']:+.4f} &plusmn; {r['perm_std']:.4f}",
             _fmt(r["shap_mean_abs"])]
            for _, r in l.iterrows()
        ]
        parts.append(
            f"<h3 style='{STYLE_H3}'>Production family (LGBM) &mdash; top 10 by "
            f"permutation</h3>"
            + _table(["Feature", "Permutation &Delta;log-loss", "Mean |SHAP|"], rows)
            + f"<p style='{STYLE_NOTE}'>Fit 2002&ndash;2023 with production "
            f"recency weighting, measured on a held-out 2024&ndash;25 window "
            f"the model never saw. Regenerated offline by "
            f"NFL/model/v2/feature_importance.py; beeswarm plots showing each "
            f"feature's direction of association live in "
            f"NFL/model/v2/artifacts/importance/.</p>"
        )
    if len(parts) == 1 and not NFL_SCORECARD.exists():
        return "<p>No NFL artifacts available.</p>"
    return "".join(parts)


# --- assembly ---------------------------------------------------------------

def status_line() -> str:
    if not SUMMARY_JSON.exists():
        return ""
    s = json.loads(SUMMARY_JSON.read_text())
    bits = []
    for m in s.get("models", []):
        st = m.get("status", "")
        label = f"{m['emoji']} {m['sport']}: "
        if st == "in_season" and m.get("record"):
            label += f"{m['record']} ({100 * m['accuracy']:.1f}%)"
        else:
            label += st.replace("_", " ")
            if m.get("season_starts"):
                label += f" (back {m['season_starts']})"
        bits.append(label)
    return " &middot; ".join(bits)


def build_html(run_date: str) -> str:
    body = (
        f"<p style='color:#444;font-size:13px;'>{status_line()}</p>"
        f"<h2 style='{STYLE_H2}'>&#9918; MLB</h2>" + mlb_section()
        + f"<h2 style='{STYLE_H2}'>&#9917; Soccer</h2>"
        + soccer_performance()
        + f"<h3 style='{STYLE_H3}'>Outcome-model feature importances</h3>"
        + soccer_importances()
        + f"<h2 style='{STYLE_H2}'>&#127944; NFL</h2>" + nfl_section()
    )
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2 style=\"margin-bottom:2px;\">Weekly Models Check</h2>"
        f'<p style="margin-top:0;color:#666;font-size:13px;">Week of {run_date} '
        f"&middot; performance + feature importances for every live model.</p>"
        f"{body}"
        f'<p style="color:#999;font-size:11px;margin-top:24px;">'
        f"Can Tre Beat Vegas &middot; "
        f'<a href="https://whosyurgoat.app/vegas">whosyurgoat.app/vegas</a></p>'
        f"</div>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    run_date = args.date

    html = build_html(run_date)
    out = OUT_DIR / run_date
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.html").write_text(html, encoding="utf-8")

    from data_jobs.email_ledger import plan
    emails = {
        "models": {
            "path": str((out / "check.html").relative_to(REPO)),
            "subject": f"📊 Models Check — week of {run_date}",
            "date": run_date,
        }
    }
    emails = plan(emails, OUT_DIR / "sent.json")
    manifest = {"date": run_date, "emails": emails}
    (OUT_DIR / "manifest_latest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {json.dumps(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
