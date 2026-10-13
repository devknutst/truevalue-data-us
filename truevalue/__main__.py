"""Allows the package to be run as ``python -m truevalue``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Started as a plain script (e.g. PyCharm "Script path"): relative imports
    # need a parent package, so put the project root on sys.path and import
    # absolutely instead.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from truevalue.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
