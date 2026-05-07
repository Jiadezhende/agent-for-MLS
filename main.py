"""Root CLI entry — thin shim over operator_opt_pipe.main.

Phase-2 evaluator runs ``bash run.sh`` which invokes ``python main.py``
from the repository root. This file forwards to the actual implementation
in ``operator_opt_pipe.main`` so the entry point stays stable while the
package internals can move freely.
"""
from __future__ import annotations

import sys

from operator_opt_pipe.main import main


if __name__ == "__main__":
    sys.exit(main())
