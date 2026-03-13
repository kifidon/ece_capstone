"""
Run MoveNet on a video and print keypoints as JSON.
Usage: python scripts/test_movenet.py <video_path> [--source-fps 30] [--target-fps 10]
"""

import argparse
import json
import os
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.movenet import MoveNetProcessor


def run(video_path: str, source_fps: int = 30, target_fps: int = 10) -> None:
    """Run MoveNet on the video and print keypoints (frame_0, frame_1, ...) as JSON."""
    if not os.path.isfile(video_path):
        print(f"Error: not a file: {video_path}", file=sys.stderr)
        sys.exit(1)
    processor = MoveNetProcessor()
    keypoints = processor.run_inference(video_path, source_fps=source_fps, target_fps=target_fps)
    with open("keypoints.json", "w") as f:
        json.dump(keypoints, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoveNet on a video and print keypoints as JSON.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--source-fps", type=int, default=30, help="Video framerate (default: 30)")
    parser.add_argument("--target-fps", type=int, default=10, help="Output keypoints at this fps (default: 10)")
    args = parser.parse_args()
    run(args.video_path, source_fps=args.source_fps, target_fps=args.target_fps)


if __name__ == "__main__":
    main()
