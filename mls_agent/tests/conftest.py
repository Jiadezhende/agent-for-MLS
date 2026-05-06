"""Shared pytest fixtures for mls_agent tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable so ``import mls_agent`` works
# without requiring pip install -e.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
