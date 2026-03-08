import json
import logging

import numpy as np
import requests
from celery import shared_task
from cryptography.fernet import Fernet
from django.conf import settings

from .models import EdgeEvent, EdgeDevice
from .ml.ml import MLProcessor
from dashboard.models import CustomUser

logger = logging.getLogger(__name__)

ml_model = MLProcessor("minimal")


class EdgeEventProcessor():

    def __init__(self, *args, **kwargs):
        self._pose_estimator = ml_model
        self._keypoints = None | np.ndarray
        self._predictions = None | np.ndarray
        self._inference_result = None | str

    def _normalize_keypoints(self, keypoints: dict) -> np.ndarray:
        """
        Normalize the keypoints (Raw MoveNet output) so that they are centered (as done in model tests folder). Center keypoints around hip midpoint and flatten to (n_frames, 51).

        Returns:
            Numpy Array of shape (n_frames, 17, 3) where 17 is the number of keypoints and 3 is the y, x, and confidence
        Args:
            keypoints: {"frame_0": [[y,x,conf], ...], "frame_1": [...], ...}
        """
        sorted_frames = sorted(keypoints.keys(), key=lambda x: int(x.split("_")[1]))
        output = []
        LEFT_HIP_IDX = 11
        RIGHT_HIP_IDX = 12 # MoveNet Keypoint Indices
        for frame in sorted_frames:
            kp = np.array(keypoints[frame], dtype=np.float32)
            center = (kp[LEFT_HIP_IDX, :2] + kp[RIGHT_HIP_IDX, :2]) / 2
            kp[:, :2] -= center
            output.append(kp.flatten())
        return np.array(output, dtype=np.float32)

    CLASSIFICATION_RULES = [
        {
            "action": "reaching",
            "poses": ["Reaching"],
            "device_match": lambda d: d["type"] == "pir_sensor" and not d.get("special_use") and d.get("sensed"),
        },
        {
            "action": "medicine",
            "poses": ["Standing"],
            "device_match": lambda d: d["type"] == "pir_sensor" and d.get("special_use") == "medicine_cabinet" and d.get("sensed"),
        },
        {
            "action": "tv",
            "poses": ["Sitting", "Lying"],
            "device_match": lambda d: d["type"] == "smart_plug" and d.get("is_on"),
        },
    ]

    def _rule_based_classification(self, data: dict) -> str:
        """
        Match the pose inference result against device states to determine the event action.
        Iterates over all devices; the first matching rule wins.
        """
        devices = data.get("devices", [])
        action = "unknown"

        for rule in self.CLASSIFICATION_RULES:
            if self._inference_result in rule["poses"] and any(rule["device_match"](d) for d in devices):
                action = rule["action"]
                break

        return action

    def run_inference(self, data: dict) -> dict:
        """
        Run model inference on keypoints, then rule-based classification.
        Returns a dict with the fields to update on the event.
        """
        keypoints = data.get("keypoints", {})
        keypoints = self._normalize_keypoints(keypoints)
        self._predictions = self._pose_estimator.predict(keypoints)
        self._inference_result = self._pose_estimator.classify_pose(self._predictions)
        action = self._rule_based_classification(data)
        return {
            "inference_result": self._predictions.tolist(),
            "pose_classification": self._inference_result,
            "action": action,
            "keypoints": keypoints.tolist(),
        }


    def post_process_event(data: dict, user: CustomUser):
        """
        Call the AI API for all of a users is_processed=False events and determin which one are anomolus based on their historic data. Update the is_processed field and set is_alert for the ones the agent flags.

        This is a scheduled celery tasks that runs in the background every 2 hours
        """

        historic_events = EdgeEvent.objects.filter(
            device__user=user,
            device__is_active=True,
            is_processed=True,
            timestamp__gt= None, #TODO:  In the last 7 days
        )

        payload = EdgeEvent.objects.filter(
            device__user=user,
            device__is_active=True,
            is_processed=False,
        )

        # Serialize and call modal with Structured output, List of event IDS that are alerts

        pass


HUB_PORT = 5050


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def push_config_to_hub(self, device_id: str) -> None:
    """Encrypt and push user config to the hub. Retries up to 3 times on failure."""
    try:
        hub = EdgeDevice.objects.get(id=device_id)
    except EdgeDevice.DoesNotExist:
        logger.error("Device %s not found", device_id)
        return

    if not hub.ip_address:
        logger.warning(f"Cannot push config to hub {hub.serial_number}: no IP address registered.")
        return

    if not hub.user:
        logger.warning(f"Cannot push config to hub {hub.serial_number}: no user linked.")
        return

    payload = {
        "api_key": hub.api_key,
        "hub_device_id": str(hub.id),
        "kasa_username": hub.user.kasa_username,
        "kasa_password": hub.user.kasa_password,
    }

    f = Fernet(settings.FIELD_ENCRYPTION_KEY.encode())
    encrypted = f.encrypt(json.dumps(payload).encode()).decode()

    try:
        resp = requests.post(
            f"http://{hub.ip_address}:{HUB_PORT}/api/config",
            json={"encrypted": encrypted},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info(f"Config pushed to hub {hub.serial_number} at {hub.ip_address}")
    except requests.RequestException as exc:
        logger.error(f"Failed to push config to hub {hub.serial_number}: {exc}")
        raise self.retry(exc=exc)


@shared_task
def process_event(event_id: str) -> None:
    """Run pose inference + rule-based classification on a saved event."""
    try:
        event = EdgeEvent.objects.get(id=event_id)
    except EdgeEvent.DoesNotExist:
        logger.error("Event %s not found", event_id)
        return

    data = {
        "keypoints": event.keypoints,
        "devices": event.device_state,
    }

    processor = EdgeEventProcessor()
    results = processor.run_inference(data)

    event.inference_result = results["inference_result"]
    event.pose_classification = results["pose_classification"]
    event.action = results["action"]
    event.keypoints = results["keypoints"]
    event.is_keypoints_normalized = True
    event.save(update_fields=["inference_result", "pose_classification", "action", "keypoints", "is_keypoints_normalized"])
