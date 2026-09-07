"""Tests for the email-safe bar charts: bars scale to the largest value,
negative magnitudes draw an empty track with the number still shown, and
the diverging chart puts negatives on the left arm and positives on the
right with both arms on one scale."""

import re

from data_jobs.reports.email_charts import BLUE, RED, TRACK_WIDTH, diverging_chart, hbar_chart, legend


def _widths(html: str, color: str) -> list[int]:
    return [int(w) for w in re.findall(rf"width:(\d+)px;height:\d+px;background:{color}", html)]


def test_hbar_scales_to_largest_and_labels_the_tip():
    html = hbar_chart([("spread_line", 0.026, "± 0.005"), ("elo_diff", 0.013, None)], digits=3)
    widths = _widths(html, BLUE)
    assert widths[0] == TRACK_WIDTH and widths[1] == TRACK_WIDTH // 2
    assert "0.026" in html and "± 0.005" in html and "0.013" in html
    assert "<svg" not in html and "<script" not in html


def test_hbar_negative_value_draws_no_bar_but_keeps_the_number():
    html = hbar_chart([("good", 0.02, None), ("noise", -0.001, None)], digits=4, signed=True)
    assert len(_widths(html, BLUE)) == 1
    assert "-0.0010" in html and "+0.0200" in html


def test_hbar_respects_a_fixed_maximum():
    html = hbar_chart([("a", 0.5, None)], max_value=1.0)
    assert _widths(html, BLUE) == [TRACK_WIDTH // 2]


def test_diverging_arms_and_shared_scale():
    html = diverging_chart([("up", 0.4, "raises"), ("down", -0.2, "lowers"), ("flat", 0.0, None)])
    assert _widths(html, BLUE) == [TRACK_WIDTH // 2]
    assert _widths(html, RED) == [TRACK_WIDTH // 4]
    assert "+0.400" in html and "-0.200" in html and "+0.000" in html
    assert "raises" in html and "lowers" in html


def test_empty_rows_and_legend():
    assert "Nothing to chart" in hbar_chart([])
    assert "Nothing to chart" in diverging_chart([])
    assert "raises" in legend([(BLUE, "raises")]) and BLUE in legend([(BLUE, "raises")])
