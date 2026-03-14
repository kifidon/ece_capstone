"""
Generate test data: runs MoveNet on videos and writes combined CSV.
Run: python build_mlp/scripts/generate_test_data.py (from project root)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from build_mlp.movenet import generate_test_data

_BASE = Path(__file__).resolve().parent.parent
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(_BASE / "data" / "poses_test.csv"))

if __name__ == "__main__":
    generate_test_data(OUTPUT_PATH)
