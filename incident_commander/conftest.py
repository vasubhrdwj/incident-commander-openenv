"""Pytest bootstrap: ensure ``incident_commander`` resolves without an editable install.

The package uses a non-trivial ``pyproject.toml`` mapping
(``package-dir = {"incident_commander" = "."}``) so that the same files
can run both as an installed package (``import incident_commander.models``)
and inside the Docker container's flat working dir (``import models``).
That mapping requires a working editable install for tests to find the
package — and the editable install can fall out of the venv after an
unrelated ``uv pip install`` call.

Adding the repo root (one level up from this file) to ``sys.path`` lets
``from incident_commander.X import ...`` resolve regardless of install
state, which makes the smoke suite robust to environment churn.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
