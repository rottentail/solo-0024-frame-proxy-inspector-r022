"""Run every random-access experiment in order with isolated Python.

Usage (from the repository root):

    python3 -I experiments/random_access/run_all.py

Each experiment writes a machine-readable JSON report under
``generated/`` and raises on any failed invariant. Nothing here imports
from ``src/``; the production service is not exercised or modified.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = (
    "exp1_reorder_provenance.py",
    "exp2_shared_pts_identity.py",
    "exp3_rap_coverage_gate.py",
    "exp4_contact_sheet_flows.py",
)


def main() -> int:
    sys.path.insert(0, str(HERE))
    for name in EXPERIMENTS:
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    print("\nAll random-access experiments passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
