import json
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

def init_api(camera_buffer=None, movenet_processor=None, poller=None):
    """Call from app.py after creating camera_buffer and loading movenet. Pass the same instances."""
    global _camera, _movenet, _poller
    _camera = camera_buffer
    _movenet = movenet_processor
    _poller = poller


# ESP motion payload (POST body) — recommended shape when ESP calls POST /api/motion-detected:
#   { "serial_number": "PIR-001", "timestamp": 1234567890.123, "battery_level": 95 }
# Optional: "location" or "special_use": "medicine_cabinet" for rule-based actions.


@api_bp.route("/motion-detected", methods=["POST"])
async def motion_detected():
    logger.info(f"Motion detected")
    data = request.get_json()
    
    # update seen device
    serial_number = data.get("serial_number")
    if not serial_number:
        return jsonify({"error": "serial_number is required"}), 400
    trigger_device = _poller.get_device_by_serial(serial_number)
    if not trigger_device:
        return jsonify({"error": "Device not found"}), 404

    trigger_device.last_seen = data.get("timestamp")
    trigger_device.battery_level = data.get("battery_level")

    # capture the next 5 seconds of video 
    logger.info(f"Sleeping for 5 seconds to capture video")
    start_time = time.time()
    time.sleep(5)
    logger.info(f"Running inference")
    keypoints_list = _movenet.run_inference(
        camera=_camera,
        source_fps=30,
        target_fps=10,
    )
    logger.info(f"Inference complete")
    if not keypoints_list:
        return jsonify({"error": "No keypoints detected"}), 400
    # keypoints_list is list of numpy arrays; convert to list of lists for JSON
    keypoints_serializable = [kp.tolist() for kp in keypoints_list]
    kp_json = json.dumps(keypoints_serializable)
    
    # poll Smart plug status if connected 
    logger.info(f"Polling smart plug status")
    smart_plugs = _poller.get_devices("smart_plug")
    smart_plug_json = []
    if _poller is not None and smart_plugs:
        for device in smart_plugs:
            smart_plug_json.append(_poller.get_device_status(device))
    logger.info(f"Building payload")
    payload = {
        "trigger_device": data,
        "keypoints": kp_json,
        "devices": []
    }
    payload["devices"].extend(smart_plug_json)
    payload["devices"].extend([d.to_registration_payload() for d in _poller.get_devices("pir_sensor") if d.serial_number != serial_number])
    
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