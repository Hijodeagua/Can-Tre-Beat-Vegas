"""Email-safe bar charts for the report emails.

Mail clients strip <svg> and <script>, so a chart here is a table whose
bar cells hold a fixed-height <div> with an inline width — the one
construction Gmail, Outlook and Apple Mail all render. Two forms:

- `hbar_chart`: magnitude, one hue (blue), bars from a shared zero
  baseline, the value labeled at the tip in text ink. Used for
  permutation importance, mean |SHAP| and normalised importance scores.
- `diverging_chart`: polarity around zero — a negative arm (red, growing
  left) and a positive arm (blue, growing right) off a grey midline.
  Used for the sign of a feature's association.

Marks follow the site's chart rules: thin bars, rounded at the data end
and square at the baseline, a 2px surface gap between rows, values in
text tokens rather than the series colour. The blue/red pair passes the
CVD and normal-vision separation checks on a white surface.
"""

from __future__ import annotations

from html import escape

BLUE = "#2a78d6"       # magnitude bars, positive arm
RED = "#e34948"        # negative arm
MIDLINE = "#c9c8c3"    # diverging baseline
INK = "#0b0b0b"
MUTED = "#52514e"
FONT = "font-family:Arial,Helvetica,sans-serif;"

BAR_HEIGHT = 14
LABEL_WIDTH = 190
TRACK_WIDTH = 300


def _fmt(v: float, digits: int, signed: bool) -> str:
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def hbar_chart(rows: list[tuple[str, float, str | None]], *, digits: int = 4,
               color: str = BLUE, max_value: float | None = None,
               unit: str = "", signed: bool = False) -> str:
    """`rows` are (label, value, note); note (e.g. "± 0.0048") sits after the
    value. Values are clipped at zero for the bar (a negative permutation
    score is noise and draws as an empty track, with the number still
    shown). Bars scale to `max_value` or the largest row."""
    if not rows:
        return "<p style='color:#666;font-size:12px;'>Nothing to chart.</p>"
    top = max_value if max_value is not None else max(v for _, v, _ in rows)
    top = top if top > 0 else 1.0
    out = [
        f"<table role='presentation' cellpadding='0' cellspacing='0' "
        f"style='border-collapse:collapse;{FONT}font-size:12px;'>"
    ]
    for label, value, note in rows:
        width = max(0.0, min(1.0, value / top)) * TRACK_WIDTH
        tip = _fmt(value, digits, signed) + unit + (f" <span style='color:{MUTED};'>{note}</span>" if note else "")
        bar = (
            f"<div style='width:{width:.0f}px;height:{BAR_HEIGHT}px;background:{color};"
            f"border-radius:0 4px 4px 0;font-size:0;line-height:0;'>&nbsp;</div>"
            if width >= 1 else
            f"<div style='width:2px;height:{BAR_HEIGHT}px;background:{MIDLINE};font-size:0;'>&nbsp;</div>"
        )
        out.append(
            f"<tr><td style='padding:0 10px 2px 0;text-align:right;white-space:nowrap;"
            f"color:{INK};width:{LABEL_WIDTH}px;'>{escape(label)}</td>"
            f"<td style='padding:0 0 2px 0;width:{TRACK_WIDTH}px;border-left:1px solid {MIDLINE};'>{bar}</td>"
            f"<td style='padding:0 0 2px 8px;white-space:nowrap;color:{INK};'>{tip}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def diverging_chart(rows: list[tuple[str, float, str | None]], *, digits: int = 3,
                    max_abs: float | None = None) -> str:
    """Bars left (red) for negative values and right (blue) for positive
    ones off a shared grey midline; `note` is the direction word shown at
    the tip. Both arms share one scale so lengths compare."""
    if not rows:
        return "<p style='color:#666;font-size:12px;'>Nothing to chart.</p>"
    scale = max_abs if max_abs is not None else max(abs(v) for _, v, _ in rows)
    scale = scale if scale > 0 else 1.0
    half = TRACK_WIDTH // 2
    out = [
        f"<table role='presentation' cellpadding='0' cellspacing='0' "
        f"style='border-collapse:collapse;{FONT}font-size:12px;'>"
    ]
    for label, value, note in rows:
        width = min(1.0, abs(value) / scale) * half
        empty = f"<div style='width:1px;height:{BAR_HEIGHT}px;font-size:0;'>&nbsp;</div>"
        if value < 0 and width >= 1:
            left = (f"<div style='width:{width:.0f}px;height:{BAR_HEIGHT}px;background:{RED};"
                    f"border-radius:4px 0 0 4px;margin-left:auto;font-size:0;'>&nbsp;</div>")
            right = empty
        elif value > 0 and width >= 1:
            left = empty
            right = (f"<div style='width:{width:.0f}px;height:{BAR_HEIGHT}px;background:{BLUE};"
                     f"border-radius:0 4px 4px 0;font-size:0;'>&nbsp;</div>")
        else:
            left = right = empty
        tip = f"{value:+.{digits}f}" + (f" <span style='color:{MUTED};'>{escape(note)}</span>" if note else "")
        out.append(
            f"<tr><td style='padding:0 10px 2px 0;text-align:right;white-space:nowrap;"
            f"color:{INK};width:{LABEL_WIDTH}px;'>{escape(label)}</td>"
            f"<td style='padding:0 0 2px 0;width:{half}px;text-align:right;'>{left}</td>"
            f"<td style='padding:0 0 2px 0;width:{half}px;border-left:2px solid {MIDLINE};'>{right}</td>"
            f"<td style='padding:0 0 2px 8px;white-space:nowrap;color:{INK};'>{tip}</td></tr>"
        )
    out.append(
        f"<tr><td></td><td style='padding:2px 0 0 0;text-align:right;font-size:11px;color:{MUTED};'>"
        f"&larr; lowers&nbsp;</td><td style='padding:2px 0 0 0;font-size:11px;color:{MUTED};'>"
        f"&nbsp;raises &rarr;</td><td></td></tr></table>"
    )
    return "".join(out)


def legend(items: list[tuple[str, str]]) -> str:
    """A swatch + label row, for charts that use more than one colour."""
    cells = "".join(
        f"<span style='display:inline-block;width:10px;height:10px;background:{c};"
        f"border-radius:2px;margin:0 4px 0 12px;'></span>{escape(t)}"
        for c, t in items
    )
    return f"<p style='{FONT}font-size:11px;color:{MUTED};margin:2px 0 6px 0;'>{cells}</p>"
