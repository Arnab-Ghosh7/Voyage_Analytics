"""Tests for the dashboard design system.

Rewritten to match the actual module. The previous version imported a ``Theme``
class that has never existed in ``src/frontend/theme.py`` and asserted a dark
palette (``#0E1117``) against what is a light design — so it failed at import,
not on a real regression.

``theme`` imports streamlit at module level, so these tests skip cleanly when the
dashboard extras are not installed (the CI serving image does not include them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

theme = pytest.importorskip(
    "src.frontend.theme",
    reason="streamlit not installed — dashboard extras are optional",
)

HEX = ("NAVY", "TEAL", "GOLD", "BLUE", "GREEN", "ORANGE", "INK", "MUTED", "PAGE_BG")


# ------------------------------------------------------------------ tokens --
@pytest.mark.parametrize("name", HEX)
def test_colour_tokens_are_valid_hex(name):
    value = getattr(theme, name)
    assert isinstance(value, str)
    assert value.startswith("#") and len(value) == 7, f"{name}={value!r}"
    int(value[1:], 16)                      # raises if not hexadecimal


def test_chart_palette_is_usable():
    assert len(theme.CHART) >= 5
    assert all(c.startswith("#") and len(c) == 7 for c in theme.CHART)
    assert len(set(theme.CHART)) == len(theme.CHART), "palette has duplicates"


def test_palette_is_light_not_dark():
    """The dashboard is a light design; guard against a silent theme flip."""
    r, g, b = (int(theme.PAGE_BG[i:i + 2], 16) for i in (1, 3, 5))
    assert (r + g + b) / 3 > 200, "PAGE_BG should be a light background"


def test_gradients_are_css_gradients():
    for key in ("blue", "green", "orange", "navy", "hero"):
        assert key in theme.GRADIENTS
        assert theme.GRADIENTS[key].startswith("linear-gradient(")


# -------------------------------------------------------------- components --
def test_metric_card_renders_values():
    html = theme.metric_card("✈️", "Flight Price", "93.6%", "R² Score",
                             "footer text", theme.GRADIENTS["blue"], 0.936)
    for fragment in ("Flight Price", "93.6%", "R² Score", "footer text", "mcard"):
        assert fragment in html


def test_metric_card_clamps_the_progress_bar():
    """A score outside 0–1 must not produce a bar wider than its track."""
    assert "width:100%" in theme.metric_card("x", "t", "v", "l", "f", "g", 4.2)
    assert "width:0%" in theme.metric_card("x", "t", "v", "l", "f", "g", -1.0)


def test_metric_card_warning_variant():
    ok = theme.metric_card("x", "t", "v", "l", "f", "g", 0.5)
    warn = theme.metric_card("x", "t", "v", "l", "f", "g", 0.5,
                             status="At chance", warn=True)
    assert "pill warn" in warn and "pill warn" not in ok
    assert "At chance" in warn


def test_panel_and_row_helpers_produce_balanced_markup():
    panel = theme.panel_open("Title", "subtitle", badge="Running")
    assert 'class="panel"' in panel and "Title" in panel and "Running" in panel

    row = theme.prow("🗄️", "Data Validation", "22 checks passed", ok=True)
    assert "Data Validation" in row and "22 checks passed" in row
    assert row.count("<div") == row.count("</div>"), "unbalanced divs"


def test_prow_marks_incomplete_steps_differently():
    assert theme.prow("x", "n", "d", ok=True) != theme.prow("x", "n", "d", ok=False)
