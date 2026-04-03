import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import requests
from celery import shared_task
from cryptography.fernet import Fernet
from django.conf import settings
from google import genai
from .models import EdgeEvent, EdgeDevice
from dashboard.models import CustomUser
from pydantic import BaseModel, Field
from django.db.models import Q
from django.utils import timezone
from config.settings import GEMINI_API_KEY

if TYPE_CHECKING:
    from .ml.ml import MLProcessor

logger = logging.getLogger(__name__)


def _edge_events_for_user(user: CustomUser):
    """Same device scope as EdgeEventView: hub or peripheral tied to this user's hub."""
    device_ids = EdgeDevice.objects.filter(
        Q(user=user) | Q(hub_device__user=user)
    ).values_list("id", flat=True)
    return EdgeEvent.objects.filter(
        hub_device_id__in=device_ids,
        hub_device__is_active=True,
        is_deleted=False,
    )

# Load TensorFlow/Keras only when running inference (Celery worker), not when the web
# process imports this module for .delay() — critical for low-memory hosts (e.g. Render 512MB).
_ml_model_lock = threading.Lock()
_ml_model: Optional[Any] = None


def get_ml_model():
    """Singleton MLProcessor; imported lazily to avoid TF in Gunicorn workers."""
    global _ml_model
    with _ml_model_lock:
        if _ml_model is None:
            from .ml.ml import MLProcessor

            _ml_model = MLProcessor("mixup_08")
        return _ml_model


class EventClassification(BaseModel):
    event_id: str = Field(description="The event ID being classified")
    is_anomaly: bool = Field(description="True if the event is anomalous, false if it is normal")
    reasoning: str = Field(description="Brief reasoning (under 100 words) for why this event is or is not anomalous")


class AnomalyDetectionResponse(BaseModel):
    results: list[EventClassification] = Field(
        default_factory=list,
        description="One entry per new event with classification and reasoning.",
    )

class EdgeEventProcessor():

    def __init__(self, *args, **kwargs):
        self._pose_estimator = None  # resolved in run_inference via lazy loader
        self._keypoints: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._inference_result: Optional[str] = None

    def _normalize_keypoints(self, keypoints: dict) -> np.ndarray:
        """
        Normalize the keypoints (Raw MoveNet output) so that they are centered (as done in model tests folder). Center keypoints around hip midpoint and flatten to (n_frames, 51).

        Returns:
            Numpy Array of shape (n_frames, 17, 3) where 17 is the number of keypoints and 3 is the y, x, and confidence
        Args:
            keypoints: {"frame_0": [[y,x,conf], ...], "frame_1": [...], ...} — only this shape is supported.
        """
        if not isinstance(keypoints, dict) or not keypoints:
            raise ValueError(
                "keypoints must be a non-empty dict with frame_* keys, e.g. {'frame_0': [...], 'frame_1': [...]}"
            )
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
            "poses": ["reaching"],
            "device_match": lambda d: d.get("device_type") == "pir_sensor" and not d.get("special_use") and d.get("sensed"),
        },
        {
            "action": "medicine",
            "poses": ["standing"],
            "device_match": lambda d: d.get("device_type") == "pir_sensor" and d.get("special_use") == "medicine_cabinet" and d.get("sensed"),
        },
        {
            "action": "tv",
            "poses": ["sitting", "lying"],
            "device_match": lambda d: d.get("device_type") == "smart_plug" and d.get("is_on"),
        },
    ]

    def _rule_based_classification(self, data: dict) -> str:
        """
        Match the pose inference result against device states to determine the event action.
        Uses devices list: trigger device (with sensed=True) plus other devices. First matching rule wins.
        """
        devices = data.get("devices", [])
        action = "unknown"
        pose = (self._inference_result or "").lower()

        for rule in self.CLASSIFICATION_RULES:
            poses_lower = [p.lower() for p in rule["poses"]]
            if pose in poses_lower and any(rule["device_match"](d) for d in devices):
                action = rule["action"]
                break

        return action

    def run_inference(self, data: dict) -> dict:
        """
        Run model inference on keypoints, then rule-based classification.
        Returns a dict with the fields to update on the event.
        """
        self._pose_estimator = get_ml_model()
        keypoints = data.get("keypoints", {})
        keypoints = self._normalize_keypoints(keypoints)
        preds = self._pose_estimator.predict(keypoints)
        self._predictions = preds if isinstance(preds, list) else preds.tolist()
        self._inference_result = self._pose_estimator.classify_pose(
            np.array(self._predictions)
        )
        action = self._rule_based_classification(data)
        return {
            "inference_result": self._predictions,
            "pose_classification": self._inference_result,
            "action": action,
            "keypoints": keypoints.tolist(),
        }
        
    def _detect_anomalies(self, data: list[dict], new_events: list[dict]) -> dict:
        """
        Detect anomalies in the data using the AI API.
        """
        client = genai.Client()
        
        def to_jsonable(x):
            # Datetime -> ISO string
            if hasattr(x, "isoformat"):
                return x.isoformat()
            # UUID -> string
            try:
                import uuid
                if isinstance(x, uuid.UUID):
                    return str(x)
            except Exception:
                pass
            return x

        safe_historic = [{k: to_jsonable(v) for k, v in row.items()} for row in (data or [])]
        safe_new_events = [{k: to_jsonable(v) for k, v in row.items()} for row in (new_events or [])]
        
        prompt = (
            "You are an anomaly detection model. Your task is to detect true anomalies (alerts) in users' event data based on historical patterns.\n\n"
            "Definition of an anomaly (alert):\n"
            "- An event is only anomalous if it clearly deviates from an identifiable, consistent pattern or routine present in the user's historical baseline.\n"
            "- Consider meaningful patterns in the user's event history: timing, frequency, duration, sequence, or any predictable regularity.\n"
            "- Only mark an event as an anomaly if there is strong evidence that a pattern exists in the baseline and the new event disrupts that pattern in a significant way.\n"
            "- DO NOT mark events as anomalies if the historical events show no recognizable or strong patterns; if behavior is already highly variable or random, new events should not be flagged as alerts simply for being unusual.\n"
            "- Examples of valid anomalies: events occurring at a time of day never seen before, far higher/lower frequency than baseline, abnormal duration, or clear outliers in established routines.\n"
            "- Examples of non-anomalies: sporadic or random baseline, lack of clear trend in historical data, natural variability with no discernible pattern.\n\n"
            "User feedback fields in the data:\n"
            "- is_alert: whether a past event was flagged as an anomaly.\n"
            "- is_resolved: whether the user explicitly marked a past alert as 'not an issue'. "
            "A resolved event means the user reviewed it and confirmed it is NORMAL expected behavior for them. "
            "Treat resolved events as strong positive examples of the user's baseline routine. "
            "Events that are similar in action, pose, and timing to resolved events should have a much higher threshold before being flagged as anomalous — "
            "the user has told you this type of activity is acceptable. "
            "Conversely, unresolved alerts (is_alert=true, is_resolved=false) represent patterns the user considers genuinely concerning.\n\n"
            "You are given:\n"
            "  - A list of historical baseline events for a user (includes is_alert and is_resolved feedback).\n"
            "  - One or more new events to assess.\n\n"
            "Return ONLY JSON matching the schema. DO NOT include explanation or notes.\n"
            "IMPORTANT: Return exactly one entry in the 'results' array for EVERY new event. "
            "Set is_anomaly to true only for events you are confident are anomalous. "
            "For every event (anomalous or not), provide brief reasoning explaining your decision.\n\n"
            f"Historic events (JSON): {json.dumps(safe_historic, ensure_ascii=False)}\n\n"
            f"New events (JSON): {json.dumps(safe_new_events, ensure_ascii=False)}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": AnomalyDetectionResponse.model_json_schema(),
                # Reduce randomness for more deterministic structured output.
                "temperature": 0,
            },
        )
        parsed = AnomalyDetectionResponse.model_validate_json(response.text)
        logger.info(
            "Anomaly detection (Gemini): %d anomaly(es) flagged of %d new event(s)",
            len(parsed.results),
            len(safe_new_events),
        )
        return parsed.model_dump()
    

    def post_process_event(self):
        """
        Call the AI API for all of a users is_processed=False events and determin which one are anomolus based on their historic data. Update the is_processed field and set is_alert for the ones the agent flags.

        This is a scheduled celery tasks that runs in the background every hour
        """
        users = CustomUser.objects.all()
        for user in users:
            base = _edge_events_for_user(user)
            event_fields = (
                "id", "timestamp", "action", "pose_classification",
                "is_alert", "is_resolved",
            )
            historic_events = list(
                base.filter(
                    is_processed=True,
                    timestamp__gte=timezone.now() - timezone.timedelta(days=7),
                ).values(*event_fields)
            )
            events_to_process = list(
                base.filter(is_processed=False).values(*event_fields)
            )

            classifications = self._detect_anomalies(historic_events, events_to_process)
            event_ids = [event["id"] for event in events_to_process]
            if not event_ids:
                continue

            updates = []
            for result in classifications.get("results", []):
                updates.append(
                    EdgeEvent(
                        id=result["event_id"],
                        is_processed=True,
                        is_alert=result.get("is_anomaly", False),
                        alert_reasoning=result.get("reasoning", ""),
                    )
                )

            classified_ids = {u.id for u in updates}  # type: ignore[attr-defined]
            unclassified_ids = [eid for eid in event_ids if eid not in classified_ids]
            if unclassified_ids:
                EdgeEvent.objects.filter(id__in=unclassified_ids).update(is_processed=True)

            if updates:
                EdgeEvent.objects.bulk_update(updates, ["is_processed", "is_alert", "alert_reasoning"])
                alert_count = sum(1 for u in updates if u.is_alert)
                logger.info(
                    "Processed %d events for user %s (%d alerts, %d normal)",
                    len(updates), user.id, alert_count, len(updates) - alert_count,
                )

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

    headers = {}
    if hub.api_key:
        headers["X-API-Key"] = hub.api_key

    try:
        resp = requests.post(
            f"http://{hub.ip_address}:{HUB_PORT}/api/config",
            json={"encrypted": encrypted},
            headers=headers,
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

    # Use raw PIR payload from trigger_device; add device_type and sensed in backend for rule matching
    devices = list(event.device_state) if isinstance(event.device_state, list) else []
    trigger = event.trigger_device
    if isinstance(trigger, dict) and trigger:
        trigger = {"device_type": "pir_sensor", **trigger, "sensed": True}
        devices = [trigger] + devices

    data = {
        "keypoints": event.keypoints,
        "devices": devices,
    }

    processor = EdgeEventProcessor()
    results = processor.run_inference(data)

    event.inference_result = results["inference_result"]
    event.pose_classification = results["pose_classification"]
    event.action = results["action"]
    event.keypoints = results["keypoints"]
    event.is_keypoints_normalized = True
    event.save(update_fields=["inference_result", "pose_classification", "action", "keypoints", "is_keypoints_normalized"])


@shared_task
def run_post_process_events() -> None:
    """
    Periodic job: Gemini anomaly pass over unprocessed events (sets is_processed / is_alert).
    Registered in CELERY_BEAT_SCHEDULE — runs once per hour.
    """
    try:
        EdgeEventProcessor().post_process_event()
    except Exception:
        logger.exception("run_post_process_events failed")
        raise
