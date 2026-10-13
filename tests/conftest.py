"""Make the shared test helpers (``tests/_oracles.py``) importable from any test subdir."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
