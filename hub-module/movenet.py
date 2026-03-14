import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import csv
import io
import logging

from camera_buffer import CameraRingBuffer
logger = logging.getLogger(__name__)

INPUT_SIZE = 256
KEYPOINT_NAMES = [
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
    "right_ankle"
]
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

class MoveNetProcessor:
    def __init__(self):
        self.movenet = self._load_movenet()

    def _load_movenet(self):
        """Download and load MoveNet Thunder from TensorFlow Hub."""
        url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        model = hub.load(url)
        return model.signatures["serving_default"]

    def _process_batch(self, images, centered=False):
        """
        Run pose estimation on a batch of images.

        Args:
            images: list of numpy arrays, each shape (height, width, 3), dtype uint8.

        Returns:
            keypoints_list: list of arrays, each shape (17, 3) — (y, x, confidence).
        """
        if not images:
            return []
        results = []
        for img in images:
            tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            tensor = tf.image.resize_with_pad(tensor[tf.newaxis], INPUT_SIZE, INPUT_SIZE)
            tensor = tf.cast(tensor, dtype=tf.int32)
            outputs = self.movenet(tensor)
            kp = outputs["output_0"].numpy().squeeze(axis=(0, 1))
            if centered: 
                results.append(self._center_keypoints(kp))
            else:
                results.append(kp)
        return results

    def _center_keypoints(self, keypoints):
        """Subtract hip midpoint from y,x columns; pass through confidence unchanged."""
        center = (
            keypoints[LEFT_HIP_IDX, :2] + keypoints[RIGHT_HIP_IDX, :2]
        ) / 2
        centered = np.array(keypoints, dtype=np.float64)
        centered[:, :2] -= center
        return centered

    def _process_video(self, video_path, classification=None):
        """
        Run pose estimation on a video file, one frame at a time.

        Args:
            video_path: Path to the video file.
            classification: Optional class label for each frame (same for all frames).

        Yields:
            Tuples of (frame_id, keypoints) where keypoints shape is (17, 3).
        """
        video = cv2.VideoCapture(video_path)
        frame_id = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            keypoints_list = self._process_batch([frame])
            if keypoints_list:
                yield frame_id, keypoints_list[0]
            frame_id += 1

        video.release()

    def run_inference_on_video(self, video_path, source_fps=30, target_fps=10):
        """
        Extract pose keypoints from a video, downsampled to target_fps, as JSON.

        Args:
            video_path: Path to the video file.
            source_fps: Assumed video framerate (default 30).
            target_fps: Target framerate to downsample to (default 10).

        Returns:
            JSON : {"frame_0": [[y, x, conf], ...], "frame_1": [...], ...}
            Each frame has 17 keypoints, each as [y, x, confidence].
        """
        step = max(1, int(source_fps / target_fps))
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {}
        
        video = cv2.VideoCapture(video_path)
        result = {}
        frame_buffer = []
        frame_id = 0
        out_frame_idx = 0

        ret, frame = video.read()
        # Read the video frames 
        while ret:
            if frame_id % step == 0:
                frame_buffer.append(frame)
            frame_id += 1
            ret, frame = video.read()

        # Process the video
        keypoints_list = self._process_batch(frame_buffer)
        for kp in keypoints_list:
            result[f"frame_{out_frame_idx}"] = kp.tolist()
            out_frame_idx += 1

        video.release()
        return result
    
    def run_inference(self, camera: CameraRingBuffer, source_fps=30, target_fps=10):
        step = max(1, int(source_fps / target_fps))
        frame_id = 0
        buffer = camera.get_buffer(seconds=10)
        if not buffer:
            return {}
        frames = [] 
        for _, frame in buffer: # store video frames downsampled to target_fps
            if frame_id % step == 0:
                frames.append(frame)
            frame_id += 1
        keypoints_list = self._process_batch(frames)
        return keypoints_list
    
movenet = MoveNetProcessor()