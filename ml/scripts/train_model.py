"""
Train the pose classifier on the generated CSV data.
Run:
    python scripts/train_model.py --single-split   # one train/val split by video (not LOVO)
    python scripts/train_model.py                  # LOVO (default, one fold per video)
    python scripts/train_model.py --folds 5        # 5-fold GroupKFold
    python scripts/train_model.py --mixup-alpha 0.4
    python scripts/train_model.py --mixup-sweep
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import DEFAULT_MIXUP_SWEEP_ALPHAS, train

CSV_DIR = os.path.join(PROJECT_ROOT, "data", "csvs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def find_latest_csv(csv_dir: str) -> str:
    csvs = sorted(f for f in os.listdir(csv_dir) if f.startswith("poses_data") and f.endswith(".csv"))
    if not csvs:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)
    return os.path.join(csv_dir, csvs[-1])


def _alpha_checkpoint_subdir(alpha: float) -> str:
    """Safe folder name for a Beta concentration α (e.g. 0.2 -> mixup_alpha_0p2)."""
    s = f"{alpha:g}".replace(".", "p")
    return f"mixup_alpha_{s}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pose classifier")
    parser.add_argument("--folds", type=int, default=None,
                        help="Number of CV folds. Default: LOVO (one per video). Ignored with --single-split.")
    parser.add_argument(
        "--single-split",
        action="store_true",
        help="Single train/validation split by video (default 80%% train / 20%% val). Skips LOVO.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.4,
        help="Fraction of videos in the validation set when using --single-split (default 0.2).",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=None,
        help="Mixup: λ ~ Beta(α,α) on train batches only. Omit for no mixup.",
    )
    parser.add_argument(
        "--mixup-sweep",
        action="store_true",
        help=f"Run CV once per α in {DEFAULT_MIXUP_SWEEP_ALPHAS} (α=0 = no mixup baseline).",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv(CSV_DIR)
    print(f"Training on: {csv_path}")

    if args.mixup_sweep:
        sweep_results: list[tuple[float, float, float]] = []
        for alpha in DEFAULT_MIXUP_SWEEP_ALPHAS:
            sub = _alpha_checkpoint_subdir(alpha)
            out_dir = os.path.join(CHECKPOINT_DIR, sub)
            mix = None if alpha <= 0 else alpha
            print(f"\n=== Mixup sweep: Beta α = {alpha} (λ ~ Beta(α,α)); train mixup = {mix is not None} ===\n")
            summary = train(
                csv_path,
                checkpoint_dir=out_dir,
                n_folds=args.folds,
                mixup_alpha=mix,
                single_split=args.single_split,
                val_fraction=args.val_fraction,
            )
            if summary:
                sweep_results.append(
                    (alpha, summary["mean_accuracy"], summary["std_accuracy"]),
                )
        print("\n=== Mixup sweep summary (mean ± std val accuracy across folds) ===")
        for alpha, mean_acc, std_acc in sweep_results:
            label = "baseline (no mixup)" if alpha <= 0 else f"α={alpha:g}"
            print(f"  {label:28s}  {mean_acc:.4f} ± {std_acc:.4f}")
        best = max(sweep_results, key=lambda t: t[1])
        print(
            f"\nBest mean accuracy: α = {best[0]} ({best[1]:.4f} ± {best[2]:.4f})",
        )
        return

    mix = args.mixup_alpha
    if mix is not None and mix <= 0:
        mix = None

    train(
        csv_path,
        checkpoint_dir=CHECKPOINT_DIR,
        n_folds=args.folds,
        mixup_alpha=mix,
        single_split=args.single_split,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
