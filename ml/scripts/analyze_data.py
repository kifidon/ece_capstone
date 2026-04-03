"""
Analyze raw video clips: count frames per video and plot frames per class.
"""

import os
import re
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "Raw_Clips")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Longest first so names like "ReachingV2" match "Reaching", not a shorter prefix.
POSE_CLASS_PREFIXES = ("Reaching", "Standing", "Sitting", "Lying")
VIDEO_EXTENSIONS = (".mp4")


def get_class(filename: str) -> str:
    """Map filename stem to pose class (e.g. LyingV2.mov -> Lying, Sitting04dupe.mp4 -> Sitting)."""
    stem = os.path.splitext(filename)[0]
    for prefix in POSE_CLASS_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return re.sub(r"[\d.]+.*$", "", stem)


def count_frames(video_dir: str) -> dict[str, int]:
    """Return {filename: frame_count} for every video in video_dir."""
    counts = {}
    for f in sorted(os.listdir(video_dir)):
        if not f.lower().endswith(VIDEO_EXTENSIONS):
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, f))
        counts[f] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return counts


def aggregate_by_class(video_frames: dict[str, int]) -> dict[str, int]:
    """Sum frame counts by class."""
    class_frames: dict[str, int] = defaultdict(int)
    for video, count in video_frames.items():
        class_frames[get_class(video)] += count
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
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_frames = count_frames(RAW_CLIPS_DIR)
    for name, n in video_frames.items():
        print(f"{name}: {n} frames")

    class_frames = aggregate_by_class(video_frames)
    print("\nFrames per class:")
    for cls in sorted(class_frames):
        print(f"  {cls}: {class_frames[cls]}")

    plot_frames_per_class(class_frames, os.path.join(OUTPUT_DIR, "frames_per_class.png"))


if __name__ == "__main__":
    main()
