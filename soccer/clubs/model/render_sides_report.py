"""Render artifacts/sides_vs_diffs_2026.json as a self-contained HTML
page: reports/soccer/sides_vs_diffs_2026.html.

The page is both the comparison and a short guide to how each of the
four models is put together, with a link to every file involved, so it
reads on its own. Charts are inline SVG drawn by a few lines of script
from the embedded JSON; no libraries.

    python -m soccer.clubs.model.render_sides_report
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from soccer.clubs.model.compare_sides import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "reports" / "soccer" / "sides_vs_diffs_2026.html"
REPO_URL = "https://github.com/Hijodeagua/Can-Tre-Beat-Vegas/blob/main/"

FEATURE_LABELS = {
    "elo_gap": "Elo gap (home + venue − away)",
    "spend_diff_z": "Transfer spend z, home − away",
    "net_diff_z": "Net spend z, home − away",
    "value_diff_z": "Squad value z, home − away",
    "wage_diff_z": "Wage bill z, home − away",
    "xg_net_diff": "xG net form, home − away",
    "elo_home_adj": "Home Elo (+ venue edge)",
    "elo_away_pre": "Away Elo",
    "home_spend_z": "Home transfer spend z",
    "away_spend_z": "Away transfer spend z",
    "home_net_z": "Home net spend z",
    "away_net_z": "Away net spend z",
    "home_value_z": "Home squad value z",
    "away_value_z": "Away squad value z",
    "home_wage_z": "Home wage bill z",
    "away_wage_z": "Away wage bill z",
    "home_xg_net": "Home xG net form",
    "away_xg_net": "Away xG net form",
    "home_att": "Home goals-scored factor",
    "home_def": "Home goals-allowed factor",
    "away_att": "Away goals-scored factor",
    "away_def": "Away goals-allowed factor",
}

MODEL_FILES = {
    "A": ["soccer/clubs/model/train.py", "soccer/clubs/daily/state.py",
          "soccer/clubs/model/features.py", "soccer/clubs/model/xg.py"],
    "B": ["soccer/clubs/model/compare_sides.py"],
    "C": ["soccer/clubs/model/report_sides_2026.py"],
    "D": ["soccer/clubs/model/report_sides_2026.py", "soccer/clubs/daily/scoring.py"],
}

MODEL_HOW = {
    "A": ("The production model. Each input is one number per match: the home side's value minus "
          "the away side's. Elo enters as the venue-adjusted gap (home rating plus the league's "
          "home edge, minus the away rating); transfer spend, net spend, squad value and wage bill "
          "are z-scored within their league-season and differenced; xG form is the difference of "
          "the two sides' rolling ten-match xG net. A multinomial logistic maps those six numbers "
          "to P(home), P(draw), P(away)."),
    "B": ("Exactly the information in A, but nothing is subtracted. The model sees the home rating "
          "and the away rating as two inputs, the home spend z and the away spend z as two inputs, "
          "and so on — twelve features. If a strong home side and a weak away side were worth "
          "something different from the reverse gap, this is the model that could learn it."),
    "C": ("B plus what the production model does not see: each side's own recent scoring. For "
          "every club a walk-forward, exponentially weighted goals-scored rate and goals-allowed "
          "rate over its league matches (half-life ten matches, shrunk toward the league's running "
          "mean with an eight-match prior), expressed as a ratio to that mean so 1.0 is league "
          "average. Four more features, sixteen in all."),
    "D": ("The same sixteen inputs, but instead of predicting the outcome directly the model "
          "predicts goals: one Poisson regression for home goals and one for away goals, on "
          "standardised features. The two predicted rates fill the joint independent-Poisson score "
          "grid the daily score model already uses, and P(home), P(draw), P(away) are read off it — "
          "so the win probability is a by-product of a score prediction rather than the target."),
}


def _lbl(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def _links(files: list[str]) -> str:
    return " · ".join(f'<a class="file" href="{REPO_URL}{f}">{escape(f)}</a>' for f in files)


def _f(x, d=4) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def build(data: dict) -> str:
    models = data["models"]
    order = ["A", "B", "C", "D"]
    best_ll = min(models[k]["kpis"]["log_loss"] for k in order)
    leagues = list(data["league_names"].items())
    paired = data["paired"]
    payload = json.dumps({
        "order": order,
        "names": {k: models[k]["name"] for k in order},
        "labels": FEATURE_LABELS,
        "perm": {k: models[k]["permutation"] for k in order},
        "shap": {k: models[k]["shap"] for k in order},
        "direction": {k: models[k]["direction"] for k in order},
        "paired": paired,
        "leagues": data["league_names"],
    })

    # --- verdict tiles
    tiles = []
    for k in order:
        kp = models[k]["kpis"]
        d = paired.get(k, {}).get("all")
        delta = ("reference" if d is None else
                 f"{d['d_ll_mean']:+.4f} ± {d['d_ll_se']:.4f} vs A")
        tiles.append(
            f'<div class="tile s{k}"><div class="tile-k">{k} · {escape(models[k]["name"])}</div>'
            f'<div class="tile-v">{kp["log_loss"]:.4f}</div>'
            f'<div class="tile-s">log loss · {100 * kp["accuracy"]:.1f}% picks right</div>'
            f'<div class="tile-d">{delta}</div></div>')

    # --- model cards
    cards = []
    for k in order:
        m = models[k]
        feats = ", ".join(_lbl(f) for f in m["features"])
        cards.append(
            f'<article class="card s{k}"><h3><span class="tag">{k}</span> {escape(m["name"])}</h3>'
            f'<p>{MODEL_HOW[k]}</p>'
            f'<p class="feats"><b>{len(m["features"])} inputs:</b> {escape(feats)}</p>'
            f'<p class="files">{_links(MODEL_FILES[k])}</p></article>')

    # --- KPI table
    kpi_rows = [
        ("Matches scored", lambda kp: f"{kp['n']:,}"),
        ("Log loss (lower is better)", lambda kp: _f(kp["log_loss"])),
        ("Accuracy", lambda kp: f"{100 * kp['accuracy']:.1f}%"),
        ("Brier (three-way)", lambda kp: _f(kp["brier"])),
        ("McFadden R² vs. class frequencies", lambda kp: _f(kp["mcfadden_r2"])),
        ("Expected calibration error", lambda kp: _f(kp["ece"], 3)),
        ("Home-win calibration slope (1 = perfect)", lambda kp: _f(kp["home_cal_slope"], 3)),
        ("Mean P(draw)", lambda kp: f"{100 * kp['mean_p_draw']:.1f}%"),
        ("Draws picked", lambda kp: f"{kp['draw_picks']}"),
        ("Picks: home / draw / away", lambda kp: f"{kp['pick_mix']['H']} / {kp['pick_mix']['D']} / {kp['pick_mix']['A']}"),
    ]
    trs = []
    for label, fn in kpi_rows:
        cells = "".join(f"<td>{fn(models[k]['kpis'])}</td>" for k in order)
        trs.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    head = "".join(f"<th>{k}<span class='sub'>{escape(models[k]['short'])}</span></th>" for k in order)
    kpi_table = (f"<table class='kpi'><thead><tr><th></th>{head}</tr></thead>"
                 f"<tbody>{''.join(trs)}</tbody></table>")

    g = models["D"]["goals"]
    goals_table = (
        "<table class='kpi narrow'><thead><tr><th></th><th>Home goals</th><th>Away goals</th></tr></thead><tbody>"
        f"<tr><th scope='row'>Mean absolute error</th><td>{g['home_mae']:.3f}</td><td>{g['away_mae']:.3f}</td></tr>"
        f"<tr><th scope='row'>… guessing the 2026 mean instead</th><td>{g['naive_home_mae']:.3f}</td><td>{g['naive_away_mae']:.3f}</td></tr>"
        f"<tr><th scope='row'>Root mean squared error</th><td>{g['home_rmse']:.3f}</td><td>{g['away_rmse']:.3f}</td></tr>"
        f"<tr><th scope='row'>Mean predicted</th><td>{g['mean_pred_home']:.3f}</td><td>{g['mean_pred_away']:.3f}</td></tr>"
        f"<tr><th scope='row'>Mean actual</th><td>{g['mean_actual_home']:.3f}</td><td>{g['mean_actual_away']:.3f}</td></tr>"
        f"<tr><th scope='row'>Exact score hit rate</th><td colspan='2'>{100 * g['exact_score_rate']:.1f}%</td></tr>"
        f"<tr><th scope='row'>Total goals, mean absolute error</th><td colspan='2'>{g['total_mae']:.3f}</td></tr>"
        "</tbody></table>")

    # --- per-league table
    lg_rows = []
    for key, name in leagues:
        cells = []
        vals = {k: models[k]["by_league"].get(key) for k in order}
        if not any(vals.values()):
            continue   # no 2026 matches in this league
        best = min(v["log_loss"] for v in vals.values() if v)
        n = next(v["n"] for v in vals.values() if v)
        for k in order:
            v = vals[k]
            cls = " class='best'" if v and abs(v["log_loss"] - best) < 1e-9 else ""
            cells.append(f"<td{cls}>{_f(v['log_loss']) if v else '—'}</td>")
        lg_rows.append(f"<tr><th scope='row'>{escape(name)}<span class='sub'>{n} matches</span></th>{''.join(cells)}</tr>")
    league_table = (f"<table class='kpi'><thead><tr><th>League</th>{head}</tr></thead>"
                    f"<tbody>{''.join(lg_rows)}</tbody></table>")

    # --- coefficient mirror for B
    dir_b = {r["feature"]: r["coef"] for r in models["B"]["direction"]}
    pairs = [("elo_home_adj", "elo_away_pre", "Elo"), ("home_spend_z", "away_spend_z", "Transfer spend"),
             ("home_net_z", "away_net_z", "Net spend"), ("home_value_z", "away_value_z", "Squad value"),
             ("home_xg_net", "away_xg_net", "xG form")]
    mirror = []
    for h, a, name in pairs:
        ch, ca = dir_b[h], dir_b[a]
        ratio = "—" if abs(ca) < 1e-9 else f"{-ch / ca:.2f}"
        mirror.append(f"<tr><th scope='row'>{name}</th><td>{ch:+.4f}</td><td>{ca:+.4f}</td><td>{ratio}</td></tr>")
    mirror_table = ("<table class='kpi narrow'><thead><tr><th>Feature</th><th>Home coef.</th><th>Away coef.</th>"
                    "<th>Home ÷ (−away)</th></tr></thead><tbody>" + "".join(mirror) + "</tbody></table>")

    tr = data["test_date_range"]
    to = data["test_outcomes"]
    tf = data["train_frequencies"]
    d_c = paired["C"]["all"]

    return f"""<title>Sides vs. Differences</title>
<meta name="description" content="Four club-soccer outcome models on every 2026 match: the production differentials, the same features as home and away sides, sides plus goals form, and a Poisson goals regressor.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#fbfaf7; --card:#ffffff; --ink:#17201b; --muted:#5b655f; --faint:#8a938d; --rule:#d9ddd8;
  --accent:#1f7a4d; --accent-ink:#ffffff; --tint:#e6f2ea;
  --sA:#2a78d6; --sB:#eb6834; --sC:#1baf7a; --sD:#eda100; --pos:#2a78d6; --neg:#e34948; --mid:#c9c8c3;
  --display:'Archivo',ui-sans-serif,system-ui,sans-serif; --body:'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --ground:#13181a; --card:#1b2124; --ink:#eef1ec; --muted:#a7b0aa; --faint:#7d867f; --rule:#2f3835;
  --accent:#3ddc84; --accent-ink:#06120b; --tint:#1c2a22;
  --sA:#3987e5; --sB:#d95926; --sC:#199e70; --sD:#c98500; --pos:#3987e5; --neg:#e66767; --mid:#4a524d;
}} }}
:root[data-theme="dark"] {{
  --ground:#13181a; --card:#1b2124; --ink:#eef1ec; --muted:#a7b0aa; --faint:#7d867f; --rule:#2f3835;
  --accent:#3ddc84; --accent-ink:#06120b; --tint:#1c2a22;
  --sA:#3987e5; --sB:#d95926; --sC:#199e70; --sD:#c98500; --pos:#3987e5; --neg:#e66767; --mid:#4a524d;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--body); font-size:15px; line-height:1.55; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 24px 80px; }}
h1,h2,h3 {{ font-family:var(--display); text-wrap:balance; margin:0; }}
h1 {{ font-size:40px; font-weight:800; letter-spacing:-0.01em; line-height:1.05; }}
h2 {{ font-size:22px; font-weight:700; margin-top:56px; padding-top:14px; border-top:2px solid var(--ink); }}
h3 {{ font-size:16px; font-weight:700; }}
h4 {{ font-family:var(--body); font-size:12px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); margin:18px 0 6px; }}
p {{ max-width:68ch; margin:10px 0; }}
.eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:0.08em; text-transform:uppercase; color:var(--accent); margin-bottom:12px; }}
.lede {{ font-size:17px; color:var(--muted); max-width:66ch; }}
.meta {{ font-family:var(--mono); font-size:12px; color:var(--faint); margin-top:14px; }}
a {{ color:var(--ink); text-decoration-thickness:1px; text-underline-offset:2px; }}
a.file {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
a:focus-visible, button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-top:28px; }}
.tile {{ background:var(--card); border:1px solid var(--rule); border-top:4px solid var(--s); padding:14px 16px; }}
.tile-k {{ font-size:12px; font-weight:600; color:var(--muted); }}
.tile-v {{ font-family:var(--mono); font-size:30px; font-weight:500; margin-top:6px; font-variant-numeric:tabular-nums; }}
.tile-s {{ font-size:12px; color:var(--muted); }}
.tile-d {{ font-family:var(--mono); font-size:12px; margin-top:8px; color:var(--ink); font-variant-numeric:tabular-nums; }}
.sA {{ --s:var(--sA); }} .sB {{ --s:var(--sB); }} .sC {{ --s:var(--sC); }} .sD {{ --s:var(--sD); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin-top:18px; }}
.card {{ background:var(--card); border:1px solid var(--rule); padding:16px 18px; }}
.card h3 {{ display:flex; align-items:center; gap:10px; }}
.tag {{ display:inline-grid; place-items:center; width:26px; height:26px; background:var(--s); color:#fff; font-family:var(--mono); font-size:13px; border-radius:4px; }}
.card p {{ font-size:14px; }}
.feats {{ color:var(--muted); font-size:13px !important; }}
.files {{ margin-top:8px; }}
table.kpi {{ border-collapse:collapse; width:100%; margin-top:14px; font-variant-numeric:tabular-nums; }}
table.kpi.narrow {{ max-width:560px; }}
table.kpi th, table.kpi td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule); font-size:14px; }}
table.kpi th {{ font-weight:600; color:var(--muted); }}
table.kpi thead th {{ border-bottom:2px solid var(--ink); color:var(--ink); font-family:var(--mono); }}
table.kpi th[scope=row] {{ text-align:left; color:var(--ink); font-weight:500; }}
table.kpi td {{ font-family:var(--mono); }}
table.kpi td.best {{ background:var(--tint); font-weight:500; }}
.sub {{ display:block; font-family:var(--body); font-size:11px; font-weight:400; color:var(--faint); }}
.scroll {{ overflow-x:auto; }}
.grid3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }}
.chart {{ background:var(--card); border:1px solid var(--rule); padding:12px 14px 6px; }}
.chart h4 {{ margin:0 0 8px; }}
svg {{ width:100%; height:auto; display:block; font-family:var(--mono); }}
svg text {{ fill:var(--ink); font-size:11px; }}
svg .lab {{ fill:var(--muted); }}
svg .axis {{ stroke:var(--mid); stroke-width:1; }}
svg .whisk {{ stroke:var(--ink); stroke-width:1.2; }}
svg .bar:hover, svg .dot:hover {{ opacity:0.75; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); margin-top:10px; }}
.legend span::before {{ content:""; display:inline-block; width:10px; height:10px; border-radius:2px; background:var(--s); margin-right:6px; vertical-align:-1px; }}
#tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--ground); font-family:var(--mono); font-size:12px; padding:6px 8px; border-radius:4px; opacity:0; transition:opacity .08s; max-width:280px; z-index:9; }}
@media (prefers-reduced-motion: reduce) {{ #tip {{ transition:none; }} }}
.note {{ font-size:13px; color:var(--muted); }}
.model-block {{ margin-top:28px; }}
.model-block h3 {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
</style>

<main>
<div class="eyebrow">Club soccer · outcome model study</div>
<h1>Sides vs. Differences</h1>
<p class="lede">Four ways to turn the same ratings, squad economics and form into a win / draw / loss probability,
each trained on the {data['train_matches']:,} league matches played before {data['test_from']} and tested on the
{data['test_matches']:,} matches played this year ({tr[0]} to {tr[1]}: the back half of the 2025-26 European
seasons, MLS 2026 and the first weeks of 2026-27). Nothing in the test window touches a fit.</p>
<p class="meta">2026 outcomes: {to['H']} home wins · {to['D']} draws · {to['A']} away wins.
Training class frequencies {100 * tf['H']:.0f} / {100 * tf['D']:.0f} / {100 * tf['A']:.0f}% give a log loss of {data['frequency_log_loss']:.4f}
on these matches — the number every model has to beat. Generated {data['generated_at']}.</p>

<div class="tiles">{''.join(tiles)}</div>
<p class="note">Log loss is the three-way score (a coin flip between three outcomes is 1.099); the paired delta is the mean
per-match difference against model A over the same {data['test_matches']:,} matches, ± one standard error.
The best of the four, model C, is {abs(d_c['d_ll_mean']):.4f} ahead of production on {d_c['d_ll_se']:.4f} of standard error — well inside one SE.</p>

<h2>The four models</h2>
<p>The question underneath: does a model need to see the home side and the away side separately, or is the gap between
them all there is? Model B answers the first half, C asks what each side's own scoring adds, and D asks whether
predicting goals is a better road to the outcome than predicting the outcome.</p>
<div class="cards">{''.join(cards)}</div>

<h2>Model KPIs on the 2026 matches</h2>
<div class="scroll">{kpi_table}</div>
<p class="note">Brier is the three-way version (sum of squared errors over the three probabilities; 0.667 is a uniform
guess). McFadden R² is 1 − log loss / the class-frequency log loss: the share of the uncertainty the model
removes. Expected calibration error compares the picked class's probability with how often that pick was right, in ten
bins. The home-win slope refits the model's own P(home) logit against the home-win indicator; 1 is perfectly calibrated,
above 1 under-confident. None of the four ever picks a draw: the draw probability never exceeds both win probabilities,
which is the usual shape of three-way soccer models, so accuracy is the accuracy of the home-or-away call.</p>

<h4>Model D as a goals predictor</h4>
<div class="scroll">{goals_table}</div>
<p class="note">The Poisson rates are lower on the away side than the 2026 average, which is why D's home-win calibration
slope sits above the others: it leans slightly too far toward the home side.</p>

<h2>By league</h2>
<div class="scroll">{league_table}</div>
<p class="note">Log loss per league; the best of the four in each row is shaded. Second divisions and MLS carry no xG
form (0 for every model), which flattens the differences there.</p>
<div class="chart" style="margin-top:18px"><h4>Paired Δ log loss vs. model A, by league (± 1 SE)</h4>
<div id="paired"></div>
<div class="legend"><span class="sB">B sides</span><span class="sC">C sides + goals form</span><span class="sD">D goals regressor</span></div></div>
<p class="note">Negative means better than production. Every whisker crosses zero.</p>

<h2>Feature importance</h2>
<p>Two views per model, both measured on the 2026 matches. <b>Permutation</b> is how much log loss degrades when one
feature is shuffled across matches ({data['perm_repeats']} repeats, ± one standard deviation): what the model
loses without it. <b>SHAP</b> is the mean absolute attribution, |coefficient × (value − training mean)|, averaged over
matches and outcome classes: how much each feature moves a typical prediction. <b>Direction</b> is the home-win
coefficient for the logistic models and, for D, the home-goal minus away-goal log-rate coefficient — positive tilts the
match toward the home side.</p>
<p class="note">The wage-bill features are zero for every row in the data and the transfer-spend features are zero for
every 2026 match (no spend rows are loaded for the 2025-26 windows yet), so their importance reads as exactly 0 —
absent, not unimportant.</p>
<div id="importance"></div>

<h2>What the sides model learned</h2>
<p>Given the freedom to weight the home value and the away value separately, model B's coefficients pair off as
mirror images: for the two features that carry most of the signal, Elo and squad value, the home coefficient is
close to the negative of the away coefficient, which is the difference rediscovered. The pairs that don't mirror
(net spend, xG form) are the ones with the least weight and the sparsest data.</p>
<div class="scroll">{mirror_table}</div>

<h2>Reading the result</h2>
<p>All four land within a thousandth of a log-loss point of each other on {data['test_matches']:,} matches, and no
difference clears one standard error. Splitting differences into sides changes nothing (B vs. A: {paired['B']['all']['d_ll_mean']:+.4f});
adding each side's goals scored and allowed is the one change that moves the number in the right direction
(C vs. A: {paired['C']['all']['d_ll_mean']:+.4f} ± {paired['C']['all']['d_ll_se']:.4f}), and it is the one worth carrying into a longer walk-forward test,
since a single year is a small sample for a difference this size. Predicting goals and deriving the outcome (D)
matches the direct models on the outcome and, as a goals predictor, beats guessing the mean by about 0.07 goals a side.</p>

<h2>How to reproduce</h2>
<p class="files">Compute: <a class="file" href="{REPO_URL}soccer/clubs/model/report_sides_2026.py">soccer/clubs/model/report_sides_2026.py</a>
· render: <a class="file" href="{REPO_URL}soccer/clubs/model/render_sides_report.py">soccer/clubs/model/render_sides_report.py</a>
· data: <a class="file" href="{REPO_URL}soccer/clubs/model/artifacts/sides_vs_diffs_2026.json">soccer/clubs/model/artifacts/sides_vs_diffs_2026.json</a>
· season-by-season version: <a class="file" href="{REPO_URL}soccer/clubs/model/compare_sides.py">soccer/clubs/model/compare_sides.py</a></p>
<p class="note">Ratings come from the UEFA-glued replay (<a class="file" href="{REPO_URL}soccer/clubs/model/europe.py">europe.py</a>) with each league's tuned parameters
(<a class="file" href="{REPO_URL}soccer/clubs/model/artifacts/tuned_params.json">tuned_params.json</a>); the production model's own two-season validation is in
<a class="file" href="{REPO_URL}soccer/clubs/model/train.py">train.py</a>. The full map of every model on the board is at
<a href="https://whosyurgoat.app/vegas/models">whosyurgoat.app/vegas/models</a>.</p>
</main>
<div id="tip" role="status" aria-live="polite"></div>

<script id="data" type="application/json">{payload}</script>
<script>
(function () {{
  const D = JSON.parse(document.getElementById('data').textContent);
  const tip = document.getElementById('tip');
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a, t) => {{ const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; return e; }};
  const show = (e, txt) => {{ e.addEventListener('mouseenter', () => {{ tip.textContent = txt; tip.style.opacity = 1; }});
    e.addEventListener('mousemove', (ev) => {{ tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; }});
    e.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; }}); }};
  const lab = (f) => D.labels[f] || f;
  const fmt = (v, d) => (v >= 0 ? '+' : '') + v.toFixed(d);

  // Horizontal bars from a zero baseline; optional ± whisker; value at the tip.
  function hbar(rows, color, opts) {{
    const W = 420, L = 190, R = 70, rh = 20, H = rows.length * rh + 8;
    const top = Math.max(...rows.map(r => Math.max(r.v, 0) + (r.sd || 0)), 1e-9);
    const sx = (v) => L + Math.max(0, Math.min(1, v / top)) * (W - L - R);
    const svg = el('svg', {{ viewBox: `0 0 ${{W}} ${{H}}` }});
    svg.appendChild(el('line', {{ x1: L, y1: 0, x2: L, y2: H, class: 'axis' }}));
    rows.forEach((r, i) => {{
      const y = i * rh + 4;
      svg.appendChild(el('text', {{ x: L - 8, y: y + 13, 'text-anchor': 'end', class: 'lab' }}, lab(r.f)));
      const w = Math.max(0, sx(r.v) - L);
      const bar = el('rect', {{ x: L, y: y + 3, width: w, height: 13, fill: color, rx: 0, class: 'bar' }});
      show(bar, `${{lab(r.f)}}: ${{opts.signed ? fmt(r.v, opts.d) : r.v.toFixed(opts.d)}}${{r.sd != null ? ' ± ' + r.sd.toFixed(opts.d) : ''}}`);
      svg.appendChild(bar);
      if (r.sd) {{
        const x1 = sx(Math.max(0, r.v - r.sd)), x2 = sx(r.v + r.sd);
        svg.appendChild(el('line', {{ x1, y1: y + 9.5, x2, y2: y + 9.5, class: 'whisk' }}));
      }}
      svg.appendChild(el('text', {{ x: sx(r.v + (r.sd || 0)) + 6, y: y + 13 }}, (opts.signed ? fmt(r.v, opts.d) : r.v.toFixed(opts.d))));
    }});
    return svg;
  }}

  // Diverging bars around a midline.
  function diverging(rows, d) {{
    const W = 420, L = 190, R = 60, rh = 20, H = rows.length * rh + 8, mid = L + (W - L - R) / 2, half = (W - L - R) / 2;
    const top = Math.max(...rows.map(r => Math.abs(r.v)), 1e-9);
    const svg = el('svg', {{ viewBox: `0 0 ${{W}} ${{H}}` }});
    svg.appendChild(el('line', {{ x1: mid, y1: 0, x2: mid, y2: H, class: 'axis' }}));
    rows.forEach((r, i) => {{
      const y = i * rh + 4, w = Math.abs(r.v) / top * half;
      svg.appendChild(el('text', {{ x: L - 8, y: y + 13, 'text-anchor': 'end', class: 'lab' }}, lab(r.f)));
      const bar = el('rect', {{ x: r.v < 0 ? mid - w : mid, y: y + 3, width: w, height: 13, fill: r.v < 0 ? css('--neg') : css('--pos'), class: 'bar' }});
      show(bar, `${{lab(r.f)}}: ${{fmt(r.v, d)}} (${{r.v < 0 ? 'toward the away side' : 'toward the home side'}})`);
      svg.appendChild(bar);
      svg.appendChild(el('text', {{ x: r.v < 0 ? mid + 6 : mid + w + 6, y: y + 13 }}, fmt(r.v, d)));
    }});
    return svg;
  }}

  const imp = document.getElementById('importance');
  D.order.forEach(k => {{
    const color = css('--s' + k);
    const block = document.createElement('div'); block.className = 'model-block s' + k;
    block.innerHTML = `<h3><span class="tag">${{k}}</span>${{D.names[k]}}</h3>`;
    const grid = document.createElement('div'); grid.className = 'grid3';
    const mk = (title, svg) => {{ const c = document.createElement('div'); c.className = 'chart'; c.innerHTML = `<h4>${{title}}</h4>`; c.appendChild(svg); grid.appendChild(c); }};
    const perm = [...D.perm[k]].sort((a, b) => b.mean - a.mean).map(r => ({{ f: r.feature, v: r.mean, sd: r.std }}));
    const order = perm.map(r => r.f);
    const byF = (arr, key) => order.map(f => {{ const r = arr.find(x => x.feature === f); return {{ f, v: r[key] }}; }});
    mk('Permutation · Δ log loss when shuffled', hbar(perm, color, {{ d: 4, signed: true }}));
    mk('SHAP · mean |attribution|', hbar(byF(D.shap[k], 'mean_abs'), color, {{ d: 3, signed: false }}));
    mk('Direction · home-side coefficient', diverging(byF(D.direction[k], 'coef'), 3));
    block.appendChild(grid); imp.appendChild(block);
  }});

  // Paired dot-and-whisker plot by league.
  (function () {{
    const keys = ['ALL', ...Object.keys(D.leagues)];
    const models = ['B', 'C', 'D'];
    const W = 900, L = 150, R = 30, rh = 26, H = keys.length * rh + 30;
    const vals = [];
    keys.forEach(lg => models.forEach(m => {{ const p = lg === 'ALL' ? D.paired[m].all : D.paired[m].by_league[lg]; if (p) vals.push(p.d_ll_mean - p.d_ll_se, p.d_ll_mean + p.d_ll_se); }}));
    const ext = Math.max(...vals.map(Math.abs)) * 1.05;
    const sx = (v) => L + (v + ext) / (2 * ext) * (W - L - R);
    const svg = el('svg', {{ viewBox: `0 0 ${{W}} ${{H}}` }});
    svg.appendChild(el('line', {{ x1: sx(0), y1: 0, x2: sx(0), y2: H - 20, class: 'axis' }}));
    [-ext, -ext / 2, 0, ext / 2, ext].forEach(t => svg.appendChild(el('text', {{ x: sx(t), y: H - 4, 'text-anchor': 'middle', class: 'lab' }}, fmt(t, 3))));
    keys.forEach((lg, i) => {{
      const y = i * rh + 8;
      svg.appendChild(el('text', {{ x: L - 10, y: y + 12, 'text-anchor': 'end', class: 'lab' }}, lg === 'ALL' ? 'All 2026 matches' : D.leagues[lg]));
      models.forEach((m, j) => {{
        const p = lg === 'ALL' ? D.paired[m].all : D.paired[m].by_league[lg];
        if (!p) return;
        const cy = y + 4 + j * 6, c = css('--s' + m);
        svg.appendChild(el('line', {{ x1: sx(p.d_ll_mean - p.d_ll_se), y1: cy, x2: sx(p.d_ll_mean + p.d_ll_se), y2: cy, stroke: c, 'stroke-width': 1.5 }}));
        const dot = el('circle', {{ cx: sx(p.d_ll_mean), cy, r: 4, fill: c, stroke: css('--card'), 'stroke-width': 1.5, class: 'dot' }});
        show(dot, `${{m}} vs A · ${{lg === 'ALL' ? 'all' : D.leagues[lg]}}: ${{fmt(p.d_ll_mean, 4)}} ± ${{p.d_ll_se.toFixed(4)}} (n=${{p.n}})`);
        svg.appendChild(dot);
      }});
    }});
    document.getElementById('paired').appendChild(svg);
  }})();
}})();
</script>
"""


def main() -> None:
    data = json.loads((ARTIFACTS / "sides_vs_diffs_2026.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(data), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
