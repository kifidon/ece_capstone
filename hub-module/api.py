import logging
import time
from flask import Blueprint, request, jsonify
import httpx
import os
from camera_buffer import CameraRingBuffer
from movenet import MoveNetProcessor
from poller import DevicePoller

BACKEND_URL = os.getenv('BACKEND_URL')

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# Set by app via init_api() so the blueprint can use camera buffer and movenet
_camera: CameraRingBuffer | None = None
_movenet: MoveNetProcessor | None = None
_poller: DevicePoller | None = None
_hub_state: dict | None = None

def init_api(camera_buffer=None, movenet_processor=None, poller=None, hub_state=None):
    """Call from app.py after creating camera_buffer and loading movenet. Pass the same instances."""
    global _camera, _movenet, _poller, _hub_state
    _camera = camera_buffer
    _movenet = movenet_processor
    _poller = poller
    _hub_state = hub_state


# ESP motion payload (POST body) — recommended shape when ESP calls POST /api/motion-detected:
#   { "serial_number": "PIR-001", "timestamp": 1234567890.123, "battery_level": 95 }
# Optional: "location" or "special_use": "medicine_cabinet" for rule-based actions.
# Dev/test: "bypass_discovered_check": true skips in-memory poller lookup (requires HUB_ALLOW_MOTION_BYPASS=1).


def _motion_bypass_allowed() -> bool:
    return os.environ.get("HUB_ALLOW_MOTION_BYPASS", "").lower() in ("1", "true", "yes")


@api_bp.route("/motion-detected", methods=["POST"])
async def motion_detected():
    logger.info("Motion detected")
    data = request.get_json(silent=True) or {}

    serial_number = data.get("serial_number")
    if not serial_number:
        return jsonify({"error": "serial_number is required"}), 400

    bypass = bool(data.get("bypass_discovered_check"))
    if bypass:
        if not _motion_bypass_allowed():
            return jsonify(
                {
                    "error": "bypass_discovered_check requires HUB_ALLOW_MOTION_BYPASS=1 in the hub environment",
                }
            ), 403
        logger.warning(
            "motion-detected: bypass_discovered_check — skipping in-memory device lookup (dev/test only)"
        )
    else:
        dev = _poller.try_get_device_by_serial(serial_number)
        if not dev:
            return jsonify({"error": "Device not found"}), 404
        dev.last_seen = data.get("timestamp")
        dev.battery_level = data.get("battery_level")

    hub_id = (_hub_state or {}).get("hub_device_id")
    if not hub_id:
        return jsonify(
            {
                "error": "hub_device_id unknown — hub must check in with backend first (POST /api/hub/register/)",
            }
        ), 503

    # capture the next 5 seconds of video
    logger.info("Sleeping for 5 seconds to capture video")
    time.sleep(5)
    logger.info("Running inference")
    keypoints_list = _movenet.run_inference(
        camera=_camera,
        source_fps=30,
        target_fps=10,
    )
    logger.info("Inference complete")
    if not keypoints_list:
        return jsonify({"error": "No keypoints detected"}), 400
    keypoints_serializable = [kp.tolist() for kp in keypoints_list]

    logger.info("Polling smart plug status")
    smart_plugs = _poller.get_devices("smart_plug")
    smart_plug_json = []
    if _poller is not None and smart_plugs:
        for device in smart_plugs:
            smart_plug_json.append(_poller.get_device_status(device))
    logger.info("Building payload")
    trigger = dict(data)
    trigger.pop("bypass_discovered_check", None)
    payload = {
        "hub_device": str(hub_id),
        "trigger_device": trigger,
        "keypoints": keypoints_serializable,
        "devices": [],
    }
    payload["devices"].extend(smart_plug_json)
    payload["devices"].extend(
        [
            d.to_registration_payload()
            for d in _poller.get_devices("pir_sensor")
            if d.serial_number != serial_number
        ]
    )
    
    logger.info(f"Sending payload to server")
    # send to server (fire-and-forget, do not await response)
    async def forward_payload():
        async with httpx.AsyncClient() as client:
            try:
                await client.post(f"{BACKEND_URL}/api/events/", json=payload)
            except Exception as e:
                logger.warning(f"Fire-and-forget POST failed: {e}")

    import asyncio
    asyncio.create_task(forward_payload())
    return jsonify(payload), 200