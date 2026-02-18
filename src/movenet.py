import csv
import io
import json
import logging
import os
import sys

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# MoveNet Thunder expects 256x256 input; Lightning uses 192x192
INPUT_SIZE = 256
BATCH_SIZE = 10

# Hip indices for centering (left_hip=11, right_hip=12)
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# MoveNet keypoint order (17 landmarks)
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

class MoveNetProcessor:
    def __init__(self):
        self.movenet = self._load_movenet()

    def _load_movenet(self):
        """Download and load MoveNet Thunder from TensorFlow Hub."""
        url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        model = hub.load(url)
        return model.signatures["serving_default"]

    def _process_image(self, image):
        """
        Run pose estimation on one image.

        Args:
            image: numpy array, shape (height, width, 3), dtype uint8.

        Returns:
            keypoints: shape (17, 2) - 17 landmarks, each (y, x).
        """
        keypoints_list = self._process_batch([image])
        return keypoints_list[0]

    def _process_batch(self, images):
        """
        Run pose estimation on a batch of images.

        Args:
            images: list of numpy arrays, each shape (height, width, 3), dtype uint8.

        Returns:
            keypoints_list: list of arrays, each shape (17, 2).
        """
        if not images:
            return []
        batch = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in images])
        batch = tf.image.resize_with_pad(batch, INPUT_SIZE, INPUT_SIZE)
        batch = tf.cast(batch, dtype=tf.int32)
        outputs = self.movenet(batch)
        # output shape: (batch_size, 1, 17, 3)
        kp = outputs["output_0"].numpy().squeeze(axis=1)[..., :2]
        return [self._center_keypoints(kp[i]) for i in range(len(images))]

    def _center_keypoints(self, keypoints):
        """Subtract hip midpoint from all keypoints for translation invariance."""
        center = (
            keypoints[LEFT_HIP_IDX] + keypoints[RIGHT_HIP_IDX]
        ) / 2
        return np.array(keypoints) - center

    def _process_video(self, video_path, classification=None, batch_size=BATCH_SIZE):
        """
        Run pose estimation on a video file in batches of frames.

        Args:
            video_path: Path to the video file.
            classification: Optional class label for each frame (same for all frames).
            batch_size: Number of frames per batch (default 10).

        Yields:
            Tuples of (frame_id, keypoints) where keypoints shape is (17, 2).
        """
        video = cv2.VideoCapture(video_path)
        frame_buffer = []
        frame_id = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_buffer.append(frame)
            if len(frame_buffer) == batch_size:
                keypoints_list = self._process_batch(frame_buffer)
                for i, kp in enumerate(keypoints_list):
                    yield frame_id - len(frame_buffer) + i, kp
                frame_buffer = []
            frame_id += 1

        if frame_buffer:
            keypoints_list = self._process_batch(frame_buffer)
            start_id = frame_id - len(frame_buffer)
            for i, kp in enumerate(keypoints_list):
                yield start_id + i, kp

        video.release()

    def _process_videos_to_csv(
        self,
        video_class_pairs,
        output_path,
        batch_size=BATCH_SIZE,
    ):
        """
        Process multiple videos and save all keypoints to a single CSV.

        Args:
            video_class_pairs: List of (video_path, classification) tuples.
            output_path: Path to write the combined CSV file.
            batch_size: Number of frames per batch.
        """
        all_frame_ids = []
        all_keypoints = []
        all_class_names = []

        for video_idx, (video_path, classification) in enumerate(video_class_pairs):
            for fid, kp in self._process_video(video_path, classification, batch_size):
                base = os.path.basename(video_path)
                frame_id = f"{base}_frame{fid}"
                all_frame_ids.append(frame_id)
                all_keypoints.append(kp)
                all_class_names.append(classification)

        csv_str = self._build_keypoints_csv(
            all_keypoints,
            frame_ids=all_frame_ids,
            class_names=all_class_names,
        )
        with open(output_path, "w") as f:
            f.write(csv_str)
        logger.info("Saved %d frames from %d videos to %s", len(all_keypoints), len(video_class_pairs), output_path)

    def run_inference(self, video_path, source_fps=30, target_fps=10):
        """
        Extract pose keypoints from a video, downsampled to target_fps, as JSON.

        Args:
            video_path: Path to the video file.
            source_fps: Assumed video framerate (default 30).
            target_fps: Target framerate to downsample to (default 10).

        Returns:
            JSON : {"frame_0": [[y,x], [y,x], ...], "frame_1": [...], ...}
            Each frame has 17 keypoints, each as [y, x]. Same order as CSV/test data.
        """
        step = max(1, int(source_fps / target_fps))
        video = cv2.VideoCapture(video_path)
        result = {}
        frame_buffer = []
        source_frame_id = 0
        out_frame_idx = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            if source_frame_id % step == 0:
                frame_buffer.append(frame)
            source_frame_id += 1

            if len(frame_buffer) == BATCH_SIZE:
                keypoints_list = self._process_batch(frame_buffer)
                for kp in keypoints_list:
                    result[f"frame_{out_frame_idx}"] = kp.tolist()
                    out_frame_idx += 1
                frame_buffer = []

        if frame_buffer:
            keypoints_list = self._process_batch(frame_buffer)
            for kp in keypoints_list:
                result[f"frame_{out_frame_idx}"] = kp.tolist()
                out_frame_idx += 1
                out_frame_idx += 1

        video.release()
        return result

    def _build_keypoints_csv(
        self,
        keypoints_list,
        frame_ids=None,
        class_names=None,
        include_headers=True,
        include_class=True,
    ):
        """
        Build a CSV string from keypoints.

        Args:
            keypoints_list: Single array (17, 2) or list of such arrays (one per frame).
            frame_ids: Optional frame id per row (list or single value).
            class_names: Optional class label per row (list or single value).
            include_headers: Whether to include header row.
            include_class: Whether to include class_name column.

        Returns:
            Full CSV as string.
        """
        if hasattr(keypoints_list, "shape") and len(keypoints_list.shape) == 2:
            keypoints_list = [keypoints_list]
            frame_ids = [frame_ids] if frame_ids is not None else [None] * 1
            class_names = [class_names] if class_names is not None else [None] * 1
        frame_ids = frame_ids or [None] * len(keypoints_list)
        class_names = class_names or [None] * len(keypoints_list)

        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")

        headers = ["frame_id"]
        for name in KEYPOINT_NAMES:
            headers.extend([f"{name}_y", f"{name}_x"])
        if include_class:
            headers.append("class_name")
        if include_headers:
            writer.writerow(headers)

        for i, kp in enumerate(keypoints_list):
            fid = frame_ids[i] if i < len(frame_ids) else ""
            cname = class_names[i] if i < len(class_names) else ""
            row = [fid] + kp.flatten().tolist()
            if include_class:
                row.append(cname)
            writer.writerow(row)

        return out.getvalue()


def generate_test_data(output_path="data/poses_test.csv"):
    """
    Generate the full test CSV from video_class_pairs.

    Calls all required functions to process videos and write the combined CSV.
    """
    video_class_pairs = []

    if not video_class_pairs:
        logger.info("video_class_pairs is empty; no videos to process.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processor = MoveNetProcessor()
    processor._process_videos_to_csv(video_class_pairs, output_path)
