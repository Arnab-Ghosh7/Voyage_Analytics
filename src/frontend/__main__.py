"""Launch the dashboard with ``python -m src.frontend``.

Streamlit needs to own the process, so this hands off to its CLI rather than
importing the app module directly.
"""
from __future__ import annotations

import sys

from streamlit.web import cli as stcli

from . import APP_PATH


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    sys.argv = ["streamlit", "run", str(APP_PATH), *args]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
