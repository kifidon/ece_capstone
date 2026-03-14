#!/usr/bin/env python3
"""
Train the pose classifier.
Run: python -m build_mlp.run_train  (from project root)
   or: python build_mlp/run_train.py
"""

import os
import sys
from pathlib import Path

# Add project root so "build_mlp" can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_mlp.train import train

# Paths relative to build_mlp/
_BASE = Path(__file__).resolve().parent
CSV_PATH = os.environ.get("CSV_PATH", str(_BASE / "data" / "poses_test.csv"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(_BASE / "checkpoints"))

if __name__ == "__main__":
    train(csv_path=CSV_PATH, checkpoint_dir=CHECKPOINT_DIR)
