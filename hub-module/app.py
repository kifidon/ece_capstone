import json
import os
import platform
import logging
import logging.handlers
import threading
from functools import partial

from cryptography.fernet import Fernet
from flask import Flask, jsonify, request

from poller import DevicePoller
from wifi_manager import WifiManager
from decorators import require_api_key
from callbacks import on_wifi_connected, on_wifi_failed, apply_config, boot_checkin_loop
from captive_portal import captive_portal_bp, init_captive_portal
from camera_buffer import from_env as camera_buffer_from_env
from api import api_bp, init_api as init_api_blueprint
from movenet import movenet as movenet_processor


# Log to file (and keep console for systemd/journal)
LOG_FILE = os.environ.get("HUB_LOG_FILE", "/var/log/smarthub.log")
LOG_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
LOG_BACKUP_COUNT = 5

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_root.addHandler(_console)

try:
    _file = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    _file.setFormatter(_fmt)
    _root.addHandler(_file)
except OSError:
    _fallback = os.path.join(os.getcwd(), "smarthub.log")
    _file = logging.handlers.RotatingFileHandler(
        _fallback, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    _file.setFormatter(_fmt)
    _root.addHandler(_file)
    logging.getLogger(__name__).warning(f"Cannot write to {LOG_FILE}, using {_fallback}")

logger = logging.getLogger(__name__)

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_raw_key = os.environ.get("FIELD_ENCRYPTION_KEY")
ENCRYPTION_KEY = _raw_key.strip() if _raw_key else None


def get_serial_number() -> str:
    """
    Read the hardware serial number.
    On Raspberry Pi OS this comes from /proc/cpuinfo.
    Falls back to the primary MAC address on other platforms (e.g. macOS for debugging).
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass

    import uuid
    mac = uuid.getnode()
    return f"MAC-{mac:012x}"


HUB_SERIAL = get_serial_number()
logger.info(f"Hub serial: {HUB_SERIAL} (platform: {platform.system()})")

poller = DevicePoller()
wifi = WifiManager(HUB_SERIAL)
camera_buffer = camera_buffer_from_env()

hub_state = {
    "api_key": None,
    "hub_device_id": None,
    "is_provisioned": False,
}


_on_wifi_connected = partial(on_wifi_connected, BACKEND_URL, HUB_SERIAL, hub_state, wifi)
_on_wifi_failed = partial(on_wifi_failed, hub_state)

init_captive_portal(HUB_SERIAL, hub_state, wifi, poller, _on_wifi_connected, _on_wifi_failed)
app.register_blueprint(captive_portal_bp)
init_api_blueprint(camera_buffer=camera_buffer, movenet_processor=movenet_processor, poller=poller)
app.register_blueprint(api_bp, url_prefix="/api")


def init():
    """
    Startup sequence:
    1. If not HUB_SKIP_AP: try saved WiFi (NM) and Ethernet before starting the AP
    2. Start WiFi AP only if no LAN uplink (captive portal for WiFi creds + PIR provisioning)
    3. Start PIR sensor listener and Kasa poller background threads
    When HUB_SKIP_AP=1 (e.g. Pi already on WiFi), skip AP and run API + threads only.
    """
    logger.info("=== Hub Init ===")
    _skip_ap_raw = os.environ.get("HUB_SKIP_AP", "<unset>")
    logger.info("HUB_SKIP_AP=%r", _skip_ap_raw)
    skip_ap = str(_skip_ap_raw).lower() in ("1", "true", "yes")
    if skip_ap:
        logger.info("HUB_SKIP_AP set; skipping WiFi AP (running on existing network).")
    else:
        if wifi.try_existing_lan_before_ap():
            pass
        else:
            try:
                wifi.start_ap()
            except Exception as e:
                logger.warning("Failed to start WiFi AP (hub will run on existing network): %s", e)

    threading.Thread(target=poller.start_pir_listener, daemon=True).start()
    threading.Thread(target=poller.start_kasa_poller, daemon=True).start()
    if camera_buffer is not None:
        camera_buffer.start()

    threading.Thread(
        target=boot_checkin_loop,
        args=(BACKEND_URL, HUB_SERIAL, hub_state, wifi.hub_report_ipv4),
        daemon=True,
    ).start()


# --- API Routes (served after hub joins home WiFi) ---

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/config", methods=["POST"])
def receive_config():
    """
    Receives an encrypted config push from the backend.
    Payload is a Fernet-encrypted JSON blob keyed under {"encrypted": "..."}.
    """
    data = request.get_json()
    if not data or "encrypted" not in data:
        return jsonify({"error": "No encrypted config data provided"}), 400

    if not ENCRYPTION_KEY:
        return jsonify({"error": "Hub encryption key not configured"}), 500

    try:
        f = Fernet(ENCRYPTION_KEY.encode())
        decrypted = json.loads(f.decrypt(data["encrypted"].encode()))
    except Exception:
        return jsonify({"error": "Decryption failed"}), 400

    if hub_state["api_key"]:
        key = request.headers.get("X-API-Key")
        if key != hub_state["api_key"]:
            return jsonify({"error": "Unauthorized"}), 401

    apply_config(hub_state, poller, BACKEND_URL, HUB_SERIAL, decrypted)
    return jsonify({"status": "ok", "message": "Config applied."})


@app.route("/api/ping", methods=["GET"])
# @require_api_key(hub_state)
def ping():
    return jsonify({
        "message": "pong",
        "hub_id": hub_state["hub_device_id"],
        "is_provisioned": hub_state["is_provisioned"],
    })


@app.route("/api/devices", methods=["GET"])
# @require_api_key(hub_state)
def list_devices():
    live = [d.to_registration_payload() for d in poller.discovered_devices]
    return jsonify({"discovered": live})


@app.route("/api/camera/status", methods=["GET"])
# @require_api_key(hub_state)
def camera_status():
    """Return whether camera buffer is active and current frame count."""
    if camera_buffer is None:
        return jsonify({"enabled": False, "capturing": False, "frames": 0, "device": None, "error": None})
    dev = camera_buffer.device
    return jsonify({
        "enabled": True,
        "capturing": camera_buffer.is_capturing(),
        "frames": camera_buffer.get_frame_count(),
        "device": dev if isinstance(dev, str) else str(dev),
        "error": camera_buffer.get_open_error(),
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": f"Not found: {error}"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": f"Internal server error: {error}"}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": f"Bad request: {error}"}), 400

@app.errorhandler(Exception)
def unhandled_exception(error):
    return jsonify({"error": f"A problem occurred locally: {error}"}), 500


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=5050, debug=True)
