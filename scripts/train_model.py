"""
Train the pose classifier on the generated CSV data.
Run:
    python scripts/train_model.py              # LOVO (default, one fold per video)
    python scripts/train_model.py --folds 5    # 5-fold GroupKFold
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import train

CSV_DIR = os.path.join(PROJECT_ROOT, "data", "csvs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def find_latest_csv(csv_dir: str) -> str:
    csvs = sorted(f for f in os.listdir(csv_dir) if f.startswith("poses_data") and f.endswith(".csv"))
    if not csvs:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)
    return os.path.join(csv_dir, csvs[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pose classifier")
    parser.add_argument("--folds", type=int, default=None,
                        help="Number of CV folds. Default: LOVO (one per video)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv(CSV_DIR)
    print(f"Training on: {csv_path}")
    train(csv_path, checkpoint_dir=CHECKPOINT_DIR, n_folds=args.folds)


if __name__ == "__main__":
    main()
