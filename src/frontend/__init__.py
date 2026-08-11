"""Streamlit dashboard package.

The app itself lives in :mod:`src.frontend.app`. Streamlit executes that module
as a script, so it is deliberately *not* imported here — importing it would run
the whole UI on ``import src.frontend``.

Launch with::

    streamlit run src/frontend/app.py
    python -m src.frontend          # convenience wrapper around the same thing
"""

__all__ = ["APP_PATH"]

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
