"""Generate the Top 100 review page, computing every SVG coordinate from data."""
import json
from pathlib import Path

D = json.load(open('/tmp/claude-0/-home-user-Can-Tre-Beat-Vegas/bcfe308a-5c0a-50d9-995a-664419533232/scratchpad/chartdata.json'))
OUT = Path('/tmp/claude-0/-home-user-Can-Tre-Beat-Vegas/bcfe308a-5c0a-50d9-995a-664419533232/scratchpad/top100_report.html')

pos = sorted(D['position'], key=lambda r: -r['su_gap'])
tiers = D['tiers']
scen = D['scenarios']
vs = D['vs_unlisted']
h2h = D['h2h']
rw = {r['weighting']: r for r in D['rank_weighting']}


# ---------- chart 1: SU gap vs spread swing (diverging, zero not at edge) ----
def chart_su_spread():
    W, ROW, TOP = 760, 34, 22
    ZERO, SCALE = 176, 22.0           # px at value 0, px per unit
    n = len(pos)
    H = TOP + n * ROW + 46
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Win-rate gap and '
         'spread swing by position group. Quarterback is largest on both; '
         'defensive line is negative on both.">']
    # gridlines every 5 units from -5 to +25
    for v in range(-5, 26, 5):
        x = ZERO + v * SCALE
        cls = 'zero' if v == 0 else 'grid'
        s.append(f'<line class="{cls}" x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+n*ROW}"/>')
        s.append(f'<text class="lbl-sm" x="{x:.1f}" y="{TOP+n*ROW+18}" text-anchor="middle">{v}</text>')
    s.append(f'<text class="lbl-sm" x="{ZERO+10*SCALE:.0f}" y="{H-6}" text-anchor="middle">'
             'percentage points (win gap) &#183; points (spread swing)</text>')
    for i, r in enumerate(pos):
        y = TOP + i * ROW
        s.append(f'<text class="lbl" x="8" y="{y+16}">{r["group"]}</text>')
        for j, (key, colr) in enumerate((('su_gap', 'series-1'), ('spread_swing', 'series-2'))):
            v = r[key] * (100 if key == 'su_gap' else 1)
            w = abs(v) * SCALE
            x = ZERO if v >= 0 else ZERO - w
            fill = f'var(--{colr})' if r['su_gap'] > 0 else 'var(--neg)'
            op = '' if j == 0 else ' opacity="0.85"'
            s.append(f'<rect x="{x:.1f}" y="{y+2+j*11}" width="{w:.1f}" height="9" rx="3" fill="{fill}"{op}/>')
        v = r['su_gap'] * 100
        if v >= 0:
            s.append(f'<text class="val" x="{ZERO+v*SCALE+8:.1f}" y="{y+15}">+{v:.1f} pp</text>')
        else:
            s.append(f'<text class="val" x="{ZERO-abs(v)*SCALE-8:.1f}" y="{y+15}" '
                     f'text-anchor="end" fill="var(--neg)">{v:.1f} pp</text>')
    s.append('</svg>')
    return '\n'.join(s)


# ---------- chart 2: cover gap with CI ----------
def chart_cover_gap():
    W, ROW, TOP = 760, 30, 20
    ZERO, SCALE = 390, 38.0
    p = sorted(pos, key=lambda r: -r['cover_gap'])
    n = len(p)
    H = TOP + n * ROW + 44
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cover-rate gap by '
         'position group with 95% confidence intervals. Every interval crosses zero.">']
    for v in (-8, -4, 0, 4, 8):
        x = ZERO + v * SCALE
        cls = 'zero' if v == 0 else 'grid'
        s.append(f'<line class="{cls}" x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+n*ROW}"/>')
        lab = '0' if v == 0 else f'{v:+d} pp'
        s.append(f'<text class="lbl-sm" x="{x:.1f}" y="{TOP+n*ROW+18}" text-anchor="middle">{lab}</text>')
    s.append(f'<text class="lbl-sm" x="{ZERO}" y="{H-6}" text-anchor="middle">'
             'cover-rate gap, advantage minus deficit (95% bootstrap CI)</text>')
    for i, r in enumerate(p):
        y = TOP + i * ROW + 14
        lo, hi, mid = r['lo'] * 100, r['hi'] * 100, r['cover_gap'] * 100
        x1, x2, xm = ZERO + lo * SCALE, ZERO + hi * SCALE, ZERO + mid * SCALE
        col = 'var(--pos)' if mid > 0.5 else ('var(--neg)' if mid < -0.5 else 'var(--neutral)')
        s.append(f'<text class="lbl" x="8" y="{y+4}">{r["group"]}</text>')
        s.append(f'<line class="err" x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}"/>')
        s.append(f'<line class="err" x1="{x1:.1f}" y1="{y-5}" x2="{x1:.1f}" y2="{y+5}"/>')
        s.append(f'<line class="err" x1="{x2:.1f}" y1="{y-5}" x2="{x2:.1f}" y2="{y+5}"/>')
        s.append(f'<circle cx="{xm:.1f}" cy="{y}" r="5" fill="{col}"/>')
        s.append(f'<text class="val" x="{x2+10:.1f}" y="{y+4}">{mid:+.1f}</text>')
    s.append('</svg>')
    return '\n'.join(s)


# ---------- chart 3: four QB matchup states ----------
def chart_scenarios():
    W, ROW, TOP = 760, 48, 22
    LO, HI = 38.0, 72.0
    X0, X1 = 168, 700
    SCALE = (X1 - X0) / (HI - LO)
    order = ['Home only', 'Both Top 100', 'Neither', 'Away only']
    rows = {r['label']: r for r in scen}
    n = len(order)
    H = TOP + n * ROW + 42
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Home win rate and home '
         'cover rate across four Top 100 quarterback matchup states. Win rates swing '
         'from 42 to 70 percent while cover rates stay near 50 percent.">']
    for v in (40, 50, 60, 70):
        x = X0 + (v - LO) * SCALE
        cls = 'zero' if v == 50 else 'grid'
        dash = ' stroke-dasharray="4 4"' if v == 50 else ''
        s.append(f'<line class="{cls}" x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+n*ROW}"{dash}/>')
        s.append(f'<text class="lbl-sm" x="{x:.1f}" y="{TOP+n*ROW+18}" text-anchor="middle">{v}%</text>')
    s.append(f'<text class="lbl-sm" x="{(X0+X1)/2:.0f}" y="{H-6}" text-anchor="middle">'
             'rate from the home team&#8217;s side &#183; dashed line = 50%</text>')
    for i, lab in enumerate(order):
        r = rows[lab]
        y = TOP + i * ROW
        s.append(f'<text class="lbl" x="8" y="{y+18}">{lab}</text>')
        s.append(f'<text class="lbl-sm" x="8" y="{y+32}">n = {r["games"]:,}</text>')
        for j, (key, colr) in enumerate((('home_win', 'series-1'), ('home_cover', 'series-2'))):
            v = r[key] * 100
            w = max((v - LO) * SCALE, 2)
            s.append(f'<rect x="{X0}" y="{y+4+j*13}" width="{w:.1f}" height="11" rx="3" '
                     f'fill="var(--{colr})"/>')
            s.append(f'<text class="{"val" if j==0 else "lbl-sm"}" x="{X0+w+8:.1f}" '
                     f'y="{y+13+j*13}">{v:.1f}%</text>')
    s.append('</svg>')
    return '\n'.join(s)


# ---------- chart 4: QB tier SU vs ATS ----------
def chart_tiers():
    W, ROW, TOP = 760, 52, 22
    LO, HI = 44.0, 72.0
    X0, X1 = 150, 660
    SCALE = (X1 - X0) / (HI - LO)
    n = len(tiers)
    H = TOP + n * ROW + 42
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Straight-up and '
         'against-the-spread rates by quarterback rank tier. Straight-up declines '
         'from 68 percent to 54 percent while cover rates stay near the coin flip.">']
    for v in (50, 55, 60, 65, 70):
        x = X0 + (v - LO) * SCALE
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+n*ROW}"/>')
        s.append(f'<text class="lbl-sm" x="{x:.1f}" y="{TOP+n*ROW+18}" text-anchor="middle">{v}%</text>')
    xbe = X0 + (52.4 - LO) * SCALE
    s.append(f'<line class="zero" x1="{xbe:.1f}" y1="{TOP-6}" x2="{xbe:.1f}" y2="{TOP+n*ROW}" '
             'stroke-dasharray="4 4"/>')
    s.append(f'<text class="lbl-sm" x="{xbe:.1f}" y="{TOP-10}" text-anchor="middle">52.4% break-even</text>')
    s.append(f'<text class="lbl-sm" x="{(X0+X1)/2:.0f}" y="{H-6}" text-anchor="middle">'
             'bars show 95% bootstrap intervals</text>')
    for i, r in enumerate(tiers):
        y = TOP + i * ROW
        s.append(f'<text class="lbl" x="8" y="{y+20}">Rank {r["tier"]}</text>')
        s.append(f'<text class="lbl-sm" x="8" y="{y+34}">n = {r["games"]:,}</text>')
        for j, (k, klo, khi, colr) in enumerate(
                (('su', 'su_lo', 'su_hi', 'series-1'), ('ats', 'ats_lo', 'ats_hi', 'series-2'))):
            v, lo, hi = r[k] * 100, r[klo] * 100, r[khi] * 100
            xv = X0 + (v - LO) * SCALE
            xl, xh = X0 + (lo - LO) * SCALE, X0 + (hi - LO) * SCALE
            yy = y + 12 + j * 18
            s.append(f'<line class="err" x1="{xl:.1f}" y1="{yy}" x2="{xh:.1f}" y2="{yy}"/>')
            s.append(f'<line class="err" x1="{xl:.1f}" y1="{yy-4}" x2="{xl:.1f}" y2="{yy+4}"/>')
            s.append(f'<line class="err" x1="{xh:.1f}" y1="{yy-4}" x2="{xh:.1f}" y2="{yy+4}"/>')
            s.append(f'<circle cx="{xv:.1f}" cy="{yy}" r="5.5" fill="var(--{colr})"/>')
            s.append(f'<text class="val" x="{xh+10:.1f}" y="{yy+4}">{v:.1f}%</text>')
    s.append('</svg>')
    return '\n'.join(s)


CSS = """
  .viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f3f0;--line:#e2e0da;
    --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#83817b;
    --series-1:#2a78d6;--series-2:#eb6834;--pos:#2a78d6;--neg:#d03b3b;--neutral:#b8b6b0;}
  @media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;--surface-1:#1a1a19;--surface-2:#232322;--line:#3a3a37;
    --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#96958c;
    --series-1:#3987e5;--series-2:#d95926;--pos:#3987e5;--neg:#e66767;--neutral:#55544f;}}
  :root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#232322;
    --line:#3a3a37;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#96958c;
    --series-1:#3987e5;--series-2:#d95926;--pos:#3987e5;--neg:#e66767;--neutral:#55544f;}
  .viz-root{background:var(--surface-1);color:var(--text-primary);
    font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    max-width:900px;margin:0 auto;padding:32px 20px 72px;}
  h1{font-size:27px;line-height:1.25;margin:0 0 6px;letter-spacing:-.02em}
  h2{font-size:20px;margin:46px 0 4px;letter-spacing:-.01em}
  h3{font-size:15px;margin:28px 0 10px;color:var(--text-secondary);font-weight:600}
  p{margin:10px 0;color:var(--text-secondary)}
  .sub{color:var(--text-muted);font-size:13px;margin-bottom:26px}
  .lede{color:var(--text-primary);font-size:16px}
  strong{color:var(--text-primary);font-weight:650}
  code{background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:12.5px}
  .kpis{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0 4px}
  .kpi{flex:1 1 190px;background:var(--surface-2);border:1px solid var(--line);
    border-radius:10px;padding:14px 16px}
  .kpi .n{font-size:26px;font-weight:660;letter-spacing:-.02em}
  .kpi .l{font-size:12.5px;color:var(--text-muted);margin-top:2px}
  .legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:4px 0 14px;
    font-size:13px;color:var(--text-secondary)}
  .swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;
    vertical-align:-1px}
  table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
  th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--text-muted);font-weight:600;font-size:12px;text-transform:uppercase;
    letter-spacing:.03em}
  tbody tr:last-child td{border-bottom:none}
  .tw{overflow-x:auto}
  .muted{color:var(--text-muted)}
  svg{display:block;max-width:100%;height:auto;margin:6px 0 4px}
  .grid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 3}
  .zero{stroke:var(--text-muted);stroke-width:1.5}
  .lbl{fill:var(--text-secondary);font-size:12px}
  .lbl-sm{fill:var(--text-muted);font-size:11px}
  .val{fill:var(--text-primary);font-size:12px;font-weight:600}
  .err{stroke:var(--text-muted);stroke-width:1.5}
  .note{border-left:3px solid var(--series-2);padding:2px 0 2px 14px;margin:16px 0;
    color:var(--text-secondary);font-size:14px}
  .verdict{border-left:3px solid var(--series-1);padding:2px 0 2px 14px;margin:16px 0}
"""

wr = next(r for r in pos if r['group'] == 'WR')
qbp = next(r for r in pos if r['group'] == 'QB')
dl = next(r for r in pos if r['group'] == 'DL')
t10 = tiers[0]

tier_rows = "".join(
    f"<tr><td>{r['tier']}</td><td>{r['games']:,}</td>"
    f"<td>{r['su']*100:.1f}%</td><td class='muted'>[{r['su_lo']*100:.1f}, {r['su_hi']*100:.1f}]</td>"
    f"<td>{r['ats']*100:.1f}%</td><td class='muted'>[{r['ats_lo']*100:.1f}, {r['ats_hi']*100:.1f}]</td>"
    f"<td>{r['p_ats']:.2f}</td></tr>" for r in tiers)

gap_rows = "".join(
    f"<tr><td>{g['bucket']}</td><td>{g['games']}</td>"
    f"<td>{g['su']*100:.1f}%</td><td>{g['ats']*100:.1f}%</td></tr>" for g in h2h['gaps'])

HTML = f"""<title>Top 100 talent: position, rank, and quarterbacks</title>
<style>{CSS}</style>
<div class="viz-root">
<h1>Top 100 talent: position, rank, and quarterbacks</h1>
<p class="sub">2011&#8211;2025 &#183; 4,096 games &#183; every rate shown straight up
<em>and</em> against the closing spread</p>

<p class="lede">Three questions: does any <strong>position group</strong> of Top 100 talent
move games more than others, do <strong>higher-ranked players</strong> outperform lower-ranked
ones, and how do games break down when a <strong>Top 100 quarterback</strong> faces one who
isn&#8217;t listed.</p>

<div class="note">Two numbers everywhere, because they answer different questions.
<strong>SU</strong> asks &#8220;do better rosters win?&#8221; &#8212; obviously yes, and it pays
nothing. <strong>ATS</strong> asks &#8220;does the market <em>underprice</em> them?&#8221;
&#8212; the only version worth money, break-even 52.4%.</div>

<h2>1. Does any position group weigh out?</h2>
<p>Games split by whether the home team had more Top 100 players at that position than the
visitor, or fewer. Blue is how far that advantage moves the result; orange is how far the
closing spread has already moved to meet it.</p>
<div class="legend">
  <span><span class="swatch" style="background:var(--series-1)"></span>Win-rate gap (pp)</span>
  <span><span class="swatch" style="background:var(--series-2)"></span>Spread swing (points)</span>
</div>
{chart_su_spread()}
<p>The ordering is intuitive &#8212; quarterback dwarfs everything, then the lines and pass
catchers &#8212; but the two bars move <em>together</em>. A Top 100 QB edge is worth
<strong>{qbp['su_gap']*100:.1f} points</strong> of win rate, and the spread has already moved
<strong>{qbp['spread_swing']:.1f} points</strong> to account for it. Defensive line is the odd
one out: more Top 100 defensive linemen goes with a slightly <em>worse</em> record, and the
market barely moves for it either.</p>

<h3>What survives the spread</h3>
<p>Subtract the market and what is left is the cover-rate gap.</p>
{chart_cover_gap()}
<div class="verdict"><strong>No position group weighs out.</strong> All eight intervals cross
zero. Wide receiver is the largest at {wr['cover_gap']*100:+.1f} pp and defensive line the most
negative at {dl['cover_gap']*100:+.1f} pp, but with ~800 games a side the standard error is
~2.5 pp. If you want the least-bad candidate for a real effect it is <strong>WR</strong> &#8212;
worth watching, not worth acting on.</div>

<h3>Rank weighting</h3>
<p>The list is ordered, so a #3 should outweigh a #97. Three schemes tested: flat count, linear
<code>(101&#8722;rank)/100</code>, and logarithmic <code>1/log&#8322;(rank+1)</code>.</p>
<div class="tw"><table>
<thead><tr><th>Weighting</th><th>corr. with margin</th><th>corr. with spread</th>
<th>corr. with margin vs spread</th></tr></thead>
<tbody>
<tr><td>count (flat)</td><td>{rw['count']['corr_with_margin']:.3f}</td>
<td>{rw['count']['corr_with_spread']:.3f}</td>
<td><strong>{rw['count']['corr_with_ats_margin']:.3f}</strong></td></tr>
<tr><td>linear</td><td>{rw['linear']['corr_with_margin']:.3f}</td>
<td>{rw['linear']['corr_with_spread']:.3f}</td>
<td>{rw['linear']['corr_with_ats_margin']:.3f}</td></tr>
<tr><td>logarithmic</td><td><strong>{rw['log']['corr_with_margin']:.3f}</strong></td>
<td><strong>{rw['log']['corr_with_spread']:.3f}</strong></td>
<td>{rw['log']['corr_with_ats_margin']:.3f}</td></tr>
</tbody></table></div>
<p>Rank weighting is a better description of reality &#8212; log-weighted talent correlates
more strongly with final margin ({rw['log']['corr_with_margin']:.3f} vs
{rw['count']['corr_with_margin']:.3f}). The catch is the middle column: it correlates more
strongly with the <em>closing spread</em> too ({rw['log']['corr_with_spread']:.3f} vs
{rw['count']['corr_with_spread']:.3f}). The books weight by rank as well, slightly better than
we do, so against the number weighting makes things marginally worse.</p>

<h2>2. Top 100 QB vs everyone else</h2>
<div class="kpis">
  <div class="kpi"><div class="n">{vs['su']*100:.1f}%</div><div class="l">Listed QB wins straight up<br>
    <span class="muted">n = {vs['games']:,} &#183; CI [{vs['su_lo']*100:.1f}, {vs['su_hi']*100:.1f}]</span></div></div>
  <div class="kpi"><div class="n">{vs['ats']*100:.1f}%</div><div class="l">&#8230;but covers the spread<br>
    <span class="muted">CI [{vs['ats_lo']*100:.1f}, {vs['ats_hi']*100:.1f}] &#183; break-even 52.4</span></div></div>
  <div class="kpi"><div class="n">&#8722;{vs['avg_spread']:.1f}</div><div class="l">Points already laid by the
    listed QB&#8217;s team</div></div>
</div>
<p>A Top 100 quarterback facing an unlisted one wins nearly two of every three games. The market
knows: those teams lay {vs['avg_spread']:.1f} points on average, and the cover rate lands at
<strong>{vs['ats']*100:.1f}%</strong> &#8212; a coin flip, two points under the juice.</p>

<h3>All four matchup states</h3>
<div class="legend">
  <span><span class="swatch" style="background:var(--series-1)"></span>Home win rate</span>
  <span><span class="swatch" style="background:var(--series-2)"></span>Home cover rate</span>
</div>
{chart_scenarios()}
<p>The blue bars swing 28 points top to bottom &#8212; a Top 100 quarterback is the single most
powerful roster fact in football. The orange bars sit in a 3-point band around the coin flip and
never reach break-even.</p>

<h2>3. Do higher-ranked players beat lower-ranked ones?</h2>
<p>First by tier: every game started by a listed quarterback, grouped by that year&#8217;s rank.</p>
<div class="legend">
  <span><span class="swatch" style="background:var(--series-1)"></span>Wins straight up</span>
  <span><span class="swatch" style="background:var(--series-2)"></span>Covers the spread</span>
</div>
{chart_tiers()}
<div class="tw"><table>
<thead><tr><th>QB rank</th><th>games</th><th>wins SU</th><th>95% CI</th><th>covers ATS</th>
<th>95% CI</th><th>P(ATS &gt; 50%)</th></tr></thead>
<tbody>{tier_rows}</tbody></table></div>
<p>Straight up the ordering is clean and monotone across the top three tiers:
{tiers[0]['su']*100:.1f}% &#8594; {tiers[1]['su']*100:.1f}% &#8594; {tiers[2]['su']*100:.1f}%,
with only the bottom tier breaking rank. So <strong>yes &#8212; higher-ranked quarterbacks
demonstrably win more.</strong></p>
<p>Against the spread it flattens. The top-10 tier at <strong>{t10['ats']*100:.1f}%</strong> is
the most interesting number here: it clears break-even, with P(ATS &gt; 50%) =
{t10['p_ats']:.2f}. But the interval [{t10['ats_lo']*100:.1f}, {t10['ats_hi']*100:.1f}] contains
both 52.4% and the coin flip, so on {t10['games']:,} games it is suggestive, not bankable.</p>

<h3>Head-to-head: both quarterbacks listed</h3>
<div class="kpis">
  <div class="kpi"><div class="n">{h2h['su']*100:.1f}%</div><div class="l">Better-ranked QB wins<br>
    <span class="muted">n = {h2h['games']} &#183; CI [{h2h['su_lo']*100:.1f}, {h2h['su_hi']*100:.1f}]
    &#183; P = {h2h['p_su']:.2f}</span></div></div>
  <div class="kpi"><div class="n">{h2h['ats']*100:.1f}%</div><div class="l">&#8230;and covers<br>
    <span class="muted">CI [{h2h['ats_lo']*100:.1f}, {h2h['ats_hi']*100:.1f}] &#183;
    P(&gt;50%) = {h2h['p_ats']:.2f}</span></div></div>
</div>
<p>When two listed quarterbacks meet, the better-ranked one wins <strong>{h2h['su']*100:.1f}%</strong>
and that <em>is</em> statistically solid &#8212; the interval clears 50% cleanly. Against the
spread it falls to {h2h['ats']*100:.1f}% and the interval swallows break-even.</p>
<div class="tw"><table>
<thead><tr><th>Rank gap</th><th>games</th><th>better QB wins SU</th><th>better QB covers</th></tr></thead>
<tbody>{gap_rows}</tbody></table></div>
<p>Bucketed by gap it is not monotone, and the 0&#8211;10 bucket inverts &#8212; when the list
separates two quarterbacks by fewer than ten places, the &#8220;better&#8221; one goes
{h2h['gaps'][0]['su']*100:.1f}%. That is the honest read on list precision: the ordering carries
real information in broad strokes and essentially none in fine ones. At 83&#8211;155 games a
bucket, do not over-read any single row.</p>

<h2>What this changes</h2>
<div class="verdict">
<p style="margin-top:0"><strong>Nothing in the model, and that is the finding.</strong>
Position-split counts, rank-weighted talent scores and QB-tier flags all describe outcomes well
and all get absorbed by the closing spread. The pattern repeats at every level of granularity:
raw effect large and obvious, market-relative effect indistinguishable from zero.</p>
<p style="margin-bottom:0">Two threads worth keeping: <strong>wide receiver</strong> is the only
position group whose leftover edge points meaningfully positive
({wr['cover_gap']*100:+.1f} pp), and <strong>top-10 quarterbacks</strong> cover at
{t10['ats']*100:.1f}%. Both sit inside their own confidence intervals today. Each season adds
~270 games; two or three more should settle whether either is real.</p>
</div>

<p class="sub" style="margin-top:28px">NFL Top 100 (player-voted) 2011&#8211;2025, 1,500
selections, all id-resolved to nflverse. Weekly active rosters and closing lines from nflverse;
position groups from the nflverse player master. Intervals are 20,000-sample bootstraps.
Reproduce with <code>python3 -m NFL.model.v2.top100_analysis --save</code>.</p>
</div>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({len(HTML):,} bytes)")
