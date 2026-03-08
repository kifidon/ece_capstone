import json
import os
import platform
import logging
import threading
from functools import partial

from cryptography.fernet import Fernet
from flask import Flask, jsonify, request

from poller import DevicePoller
from wifi import WifiManager
from decorators import require_api_key
from callbacks import on_wifi_connected, on_wifi_failed, apply_config
from captive_portal import captive_portal_bp, init_captive_portal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ENCRYPTION_KEY = os.environ.get("FIELD_ENCRYPTION_KEY")


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

hub_state = {
    "api_key": None,
    "hub_device_id": None,
    "is_provisioned": False,
}


_on_wifi_connected = partial(on_wifi_connected, BACKEND_URL, HUB_SERIAL, hub_state)
_on_wifi_failed = partial(on_wifi_failed, hub_state)

init_captive_portal(HUB_SERIAL, hub_state, wifi, poller, _on_wifi_connected, _on_wifi_failed)
app.register_blueprint(captive_portal_bp)


def init():
    """
    Startup sequence:
    1. Start WiFi AP (captive portal for user to enter WiFi creds + PIR sensor provisioning)
    2. Start PIR sensor listener and Kasa poller background threads
    Hub stays in AP mode until user submits WiFi credentials via the captive portal.
    """
    logger.info("=== Hub Init ===")

    wifi.start_ap()

    threading.Thread(target=poller.start_pir_listener, daemon=True).start()
    threading.Thread(target=poller.start_kasa_poller, daemon=True).start()


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
@require_api_key(hub_state)
def ping():
    return jsonify({
        "message": "pong",
        "hub_id": hub_state["hub_device_id"],
        "is_provisioned": hub_state["is_provisioned"],
    })


@app.route("/api/devices", methods=["GET"])
@require_api_key(hub_state)
def list_devices():
    live = [d.to_registration_payload() for d in poller.discovered_devices]
    return jsonify({"discovered": live})


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
