#!/usr/bin/env python3
"""
Train the pose classifier.
Run: python train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train

CSV_PATH = os.environ.get("CSV_PATH", "data/poses_test.csv")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

if __name__ == "__main__":
    train(
        csv_path=CSV_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
    )
