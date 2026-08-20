import pytest
from src.frontend.theme import Theme


def test_theme_palette_and_styles():
    assert len(Theme.PALETTE) >= 6
    assert Theme.BG == "#0E1117"
    assert Theme.CARD == "#1E232F"
    assert Theme.TEXT == "#FAFAFA"


def test_theme_chart_colors():
    assert len(Theme.CHART) >= 5
    assert all(c.startswith("#") for c in Theme.CHART)
