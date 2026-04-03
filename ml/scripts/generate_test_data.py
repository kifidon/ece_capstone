"""
Generate pose CSV: run MoveNet on all videos in data/Raw_Clips and write combined CSV.
Run: python scripts/generate_test_data.py
"""

import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.movenet import MoveNetProcessor

RAW_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "Raw_Clips")
CSV_DIR = os.path.join(PROJECT_ROOT, "data", "csvs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

POSE_CLASS_PREFIXES = ("Reaching", "Standing", "Sitting", "Lying")
VIDEO_EXTENSIONS = (".mp4", ".mov")

# After MoveNet, upsample minority classes (e.g. Lying, Reaching) so each class
# matches the frame count of the largest class. Rows are duplicated with unique
# frame_id suffixes (_osamp_*); training splits still group by video_id correctly.
BALANCE_TO_MAX_CLASS = True
BALANCE_RANDOM_STATE = 42


def get_class(filename: str) -> str:
    """Map filename stem to pose class (e.g. LyingV2.mov -> Lying, Sitting04dupe.mp4 -> Sitting)."""
    stem = os.path.splitext(filename)[0]
    for prefix in POSE_CLASS_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return re.sub(r"[\d.]+.*$", "", stem)


def build_video_class_pairs(video_dir: str) -> list[tuple[str, str]]:
    """Scan video_dir for video files and return (path, class_name) pairs."""
    pairs = []
    for f in sorted(os.listdir(video_dir)):
        if not f.lower().endswith(VIDEO_EXTENSIONS):
            continue
        path = os.path.join(video_dir, f)
        cls = get_class(f)
        pairs.append((path, cls))
    return pairs


def count_frames_per_class(pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Count total frames per class from video files."""
    class_frames: dict[str, int] = defaultdict(int)
    for path, cls in pairs:
        cap = cv2.VideoCapture(path)
        class_frames[cls] += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return dict(class_frames)


def balance_pose_csv_to_max_class(
    csv_path: str,
    *,
    random_state: int = BALANCE_RANDOM_STATE,
) -> None:
    """
    For each class with fewer rows than the majority class, sample rows with
    replacement until counts match. Writes the balanced table back to csv_path.
    """
    df = pd.read_csv(csv_path)
    if df.empty or "class_name" not in df.columns:
        return

    counts = df["class_name"].value_counts()
    target = int(counts.max())
    chunks: list[pd.DataFrame] = []

    for cls in counts.index:
        sub = df[df["class_name"] == cls].copy()
        n = len(sub)
        if n < target:
            need = target - n
            extra = sub.sample(n=need, replace=True, random_state=random_state)
            extra = extra.reset_index(drop=True)
            extra["frame_id"] = (
                extra["frame_id"].astype(str)
                + "_osamp_"
                + pd.Series(np.arange(need), dtype=str)
            )
            sub = pd.concat([sub, extra], ignore_index=True)
        chunks.append(sub)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    out.to_csv(csv_path, index=False)
    print(
        f"Balanced CSV: each class -> {target} frames "
        f"(before balance: min {int(counts.min())}, max {int(counts.max())})."
    )


def plot_frames_per_class(class_frames: dict[str, int], output_path: str) -> None:
    """Save a bar chart of frames per class."""
    classes = sorted(class_frames.keys())
    counts = [class_frames[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Number of Frames", fontsize=13)
    ax.set_title("Number of Frames per Class", fontsize=15)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(CSV_DIR, f"poses_data_{date_str}.csv")
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = build_video_class_pairs(RAW_CLIPS_DIR)
    if not pairs:
        print(f"No video files ({', '.join(VIDEO_EXTENSIONS)}) found in {RAW_CLIPS_DIR}")
        sys.exit(1)

    print(f"Found {len(pairs)} videos:")
    for path, cls in pairs:
        print(f"  {cls:12s} <- {os.path.basename(path)}")

    class_frames = count_frames_per_class(pairs)
    chart_path = os.path.join(OUTPUT_DIR, f"frames_per_class_{date_str}.png")
    plot_frames_per_class(class_frames, chart_path)

    print(f"\nLoading MoveNet...")
    processor = MoveNetProcessor()

    print(f"Processing videos -> {csv_path}")
    processor._process_videos_to_csv(pairs, csv_path)

    if BALANCE_TO_MAX_CLASS:
        print("\nBalancing class counts (minority classes upsampled with replacement)...")
        balance_pose_csv_to_max_class(csv_path)

    print("Done.")


if __name__ == "__main__":
    main()
