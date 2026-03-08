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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.movenet import MoveNetProcessor

RAW_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "Raw_Clips")
CSV_DIR = os.path.join(PROJECT_ROOT, "data", "csvs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def get_class(filename: str) -> str:
    """Extract class name from video filename (e.g. 'Sitting04dupe.mp4' -> 'Sitting')."""
    base = filename.replace(".mp4", "")
    return re.sub(r"[\d.]+.*$", "", base)


def build_video_class_pairs(video_dir: str) -> list[tuple[str, str]]:
    """Scan video_dir for .mp4 files and return (path, class_name) pairs."""
    pairs = []
    for f in sorted(os.listdir(video_dir)):
        if not f.endswith(".mp4"):
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
        print(f"No .mp4 files found in {RAW_CLIPS_DIR}")
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
    print("Done.")


if __name__ == "__main__":
    main()
