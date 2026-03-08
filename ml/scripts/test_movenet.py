"""
Tests for MoveNetProcessor.
Run: pytest scripts/test_movenet.py -v
"""

import json
import logging
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.movenet import MoveNetProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _make_video(path, num_frames, width=256, height=256, fps=30):
    """Create a synthetic video file with dummy frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def processor():
    """Shared processor instance to avoid loading model multiple times."""
    return MoveNetProcessor()


def test_single_frame_inference(processor):
    """Run inference on a single frame, assert output shape and format."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    try:
        _make_video(video_path, num_frames=1)
        json_str = processor.video_keypoints_to_json(video_path)

        data = json.loads(json_str)
        assert "frame_0" in data
        keypoints = data["frame_0"]
        assert len(keypoints) == 17
        for kp in keypoints:
            assert len(kp) == 2
            assert all(isinstance(c, (int, float)) for c in kp)
            assert kp[0] is not None and kp[1] is not None  # y, x
    finally:
        os.unlink(video_path)


def test_video_inference(processor):
    """Run inference on a video, assert output has expected frames and structure."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    try:
        # 15 frames at 30fps, downsampled to 10fps -> every 3rd frame -> 5 frames
        _make_video(video_path, num_frames=15)
        json_str = processor.video_keypoints_to_json(video_path, source_fps=30, target_fps=10)

        data = json.loads(json_str)
        assert len(data) == 5
        for i in range(5):
            key = f"frame_{i}"
            assert key in data
            keypoints = data[key]
            assert len(keypoints) == 17
            for kp in keypoints:
                assert len(kp) == 2
                assert all(isinstance(c, (int, float)) for c in kp)
    finally:
        os.unlink(video_path)
