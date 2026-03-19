import logging
import threading

from flask import Blueprint, request, redirect, render_template, jsonify

logger = logging.getLogger(__name__)

captive_portal_bp = Blueprint("captive_portal", __name__)

_hub_serial = None
_hub_state = None
_wifi = None
_poller = None
_on_wifi_connected = None
_on_wifi_failed = None


def init_captive_portal(hub_serial, hub_state, wifi, poller, on_connected, on_failed):
    """Wire up the blueprint with app-level dependencies."""
    global _hub_serial, _hub_state, _wifi, _poller, _on_wifi_connected, _on_wifi_failed
    _hub_serial = hub_serial
    _hub_state = hub_state
    _wifi = wifi
    _poller = poller
    _on_wifi_connected = on_connected
    _on_wifi_failed = on_failed


@captive_portal_bp.route("/wifi/scan", methods=["GET"])
def wifi_scan():
    """Return list of visible WiFi networks for the setup page (scan-and-select)."""
    networks = _wifi.scan_networks()
    return jsonify({"networks": networks})


@captive_portal_bp.route("/", methods=["GET"])
def setup_page():
    """Serve the WiFi setup page, or the failure page if the last attempt failed."""
    failed_ssid = _hub_state.pop("wifi_error", None)
    if failed_ssid:
        return render_template("setup_fail.html", ssid=failed_ssid)
    return render_template("setup.html", serial=_hub_serial)


@captive_portal_bp.route("/setup", methods=["POST"])
def setup_wifi():
    """User submits WiFi credentials from the captive portal form."""
    ssid = request.form.get("ssid", "").strip()
    password = request.form.get("password", "").strip()

    if not ssid or not password:
        return render_template("setup.html", serial=_hub_serial, status_msg="Please fill in both fields."), 400

    logger.info(f"WiFi credentials received for network: {ssid}")

    threading.Thread(
        target=_wifi.provision_and_switch,
        args=(ssid, password, _poller.discovered_devices, _on_wifi_connected, _on_wifi_failed),
        daemon=True,
    ).start()

    return render_template("setup_success.html", ssid=ssid, serial=_hub_serial)


@captive_portal_bp.route("/generate_204", methods=["GET"])
@captive_portal_bp.route("/hotspot-detect.html", methods=["GET"])
@captive_portal_bp.route("/connecttest.txt", methods=["GET"])
@captive_portal_bp.route("/ncsi.txt", methods=["GET"])
def captive_redirect():
    """Android/iOS/Windows captive portal detection endpoints."""
    return redirect("/", code=302)
