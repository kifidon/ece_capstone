"""
USB camera rolling buffer: always recording, keep only the last N seconds in memory.

Runs a background thread that captures from a V4L2/USB camera and pushes frames
into a fixed-size ring buffer. Callers can read the current buffer (e.g. on PIR
trigger) or export it to a video file.

Env:
  HUB_CAMERA_ENABLED    - Set to 1 / true / yes to enable the rolling buffer
  HUB_CAMERA_BUFFER_S   - Seconds to retain (default: 10)
  HUB_CAMERA_FPS        - Capture rate (default: 30)
  HUB_CAMERA_WIDTH      - Frame width (default: 640)
  HUB_CAMERA_HEIGHT     - Frame height (default: 480)
  (Device is always auto-selected: /dev/video* in order, then indices 0–3.)
"""

import glob
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# OpenCV is optional so hub can run without a camera 
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _enumerate_camera_candidates() -> list:
    """Prefer V4L2 device nodes, then numeric indices (covers path vs index quirks)."""
    paths = sorted(glob.glob("/dev/video*"))
    out: list = list(paths)
    for idx in range(4):
        out.append(idx)
    return out


def from_env():
    """Build CameraRingBuffer from env (HUB_CAMERA_*). Returns None if disabled or cv2 missing."""
    if os.environ.get("HUB_CAMERA_ENABLED", "").lower() not in ("1", "true", "yes"):
        return None
    if not HAS_CV2:
        logger.warning("HUB_CAMERA_ENABLED=1 but opencv-python-headless not installed; camera buffer disabled.")
        return None
    return CameraRingBuffer(
        buffer_seconds=int(os.environ.get("HUB_CAMERA_BUFFER_S", "10")),
        capture_fps=int(os.environ.get("HUB_CAMERA_FPS", "30")),
        width=int(os.environ.get("HUB_CAMERA_WIDTH", "640")),
        height=int(os.environ.get("HUB_CAMERA_HEIGHT", "480")),
    )


class CameraRingBuffer:
    """
    Captures from a USB camera in a background thread and keeps the last
    buffer_seconds of frames in a ring buffer (oldest dropped when full).
    """

    def __init__(
        self,
        buffer_seconds=20,
        capture_fps=30,
        width=640,
        height=480,
    ):
        if not HAS_CV2:
            raise RuntimeError("opencv-python-headless is required for camera buffer. pip install opencv-python-headless")
        # Set to the first device OpenCV opens successfully (see _capture_loop).
        self.device: int | str = "auto"
        self.buffer_seconds = buffer_seconds
        self.capture_fps = capture_fps
        self.width = width
        self.height = height
        self._max_frames = buffer_seconds * capture_fps
        self._deque = deque(maxlen=self._max_frames)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._cap = None
        self._opened_ok = False
        self._open_error: str | None = None
        self._status_lock = threading.Lock()

    def start(self):
        """Start the capture thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._status_lock:
            self._opened_ok = False
            self._open_error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Camera ring buffer started: probing /dev/video* then indices 0–3; %ds @ %d fps (%dx%d), max_frames=%d",
            self.buffer_seconds,
            self.capture_fps,
            self.width,
            self.height,
            self._max_frames,
        )

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info("Camera ring buffer stopped.")

    def _capture_loop(self):
        tried: list[str] = []
        self._cap = None
        for dev in _enumerate_camera_candidates():
            tried.append(repr(dev))
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                self._cap = cap
                self.device = dev
                logger.info("Using camera device %r", dev)
                break
            cap.release()

        if self._cap is None or not self._cap.isOpened():
            msg = f"Could not open any camera; tried: [{', '.join(tried)}]"
            logger.error(msg)
            with self._status_lock:
                self._open_error = msg
                self._opened_ok = False
            return

        with self._status_lock:
            self._open_error = None
            self._opened_ok = True
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Hint capture rate (many USB cams will approximate)
        self._cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
        interval = 1.0 / self.capture_fps

        while not self._stop.is_set():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Camera read failed; retrying.")
                time.sleep(0.5)
                continue
            ts = time.time()
            with self._lock:
                self._deque.append((ts, frame.copy()))
            elapsed = time.monotonic() - t0
            sleep = max(0.0, interval - elapsed)
            if sleep > 0:
                self._stop.wait(timeout=sleep)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_buffer(self, seconds=20):
        """
        Return a copy of the current ring buffer as a list of (timestamp, frame).
        Frames are numpy arrays (BGR); oldest first.
        If seconds is provided, return the last N seconds of frames.
        """
        with self._lock:
            return [(t, f.copy()) for t, f in self._deque if t > time.time() - seconds]

    def get_frame_count(self):
        """Number of frames currently in the buffer."""
        with self._lock:
            return len(self._deque)

    def is_capturing(self) -> bool:
        """True if the capture thread is running and the device opened successfully."""
        th = self._thread
        if th is None or not th.is_alive():
            return False
        with self._status_lock:
            return self._opened_ok

    def get_open_error(self) -> str | None:
        with self._status_lock:
            return self._open_error

    def save_buffer_to_video(self, path, fps=None):
        """
        Write the current buffer to a video file (e.g. on event).
        path: output file (.mp4 or .avi).
        fps: output fps (default: self.capture_fps).
        """
        fps = fps or self.capture_fps
        with self._lock:
            if not self._deque:
                logger.warning("save_buffer: buffer is empty")
                return False
            _, first = self._deque[0]
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, fps, (w, h))
            for _, frame in self._deque:
                out.write(frame)
            out.release()
        logger.info("Saved %d frames to %s", len(self._deque), path)
        return True
