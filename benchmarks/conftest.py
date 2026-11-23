"""Lets the smoke tests import the benchmark scripts (benchmarks/ on sys.path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
