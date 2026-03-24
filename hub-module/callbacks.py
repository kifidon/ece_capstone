import logging
import time

import requests

logger = logging.getLogger(__name__)
from poller import DevicePoller


def pull_config_from_backend(
    backend_url: str,
    hub_serial: str,
    hub_state: dict,
) -> bool:
    """
    Hub polls the backend for its config (after check-in).
    If the hub has an api_key in hub_state, use it in the X-API-Key header.
    Returns True on success, False on failure.
    """
    if not hub_state.get("api_key"):
        logger.warning("Cannot pull config: no API key in hub_state yet (hub not claimed?)")
        return False

    url = f"{backend_url.rstrip('/')}/api/hub/config/?hub_serial={hub_serial}"
    headers = {"X-API-Key": hub_state["api_key"]}

    logger.info("Pulling config from backend at %s", url)

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        encrypted_payload = data.get("encrypted")
        if not encrypted_payload:
            logger.error("Backend returned no encrypted config")
            return False

        # Decrypt and parse (same logic as /api/config POST)
        from cryptography.fernet import Fernet
        import json
        import os

        encryption_key = os.environ.get("FIELD_ENCRYPTION_KEY", "").strip()
        if not encryption_key:
            logger.error("FIELD_ENCRYPTION_KEY not set; cannot decrypt config")
            return False

        f = Fernet(encryption_key.encode())
        decrypted = json.loads(f.decrypt(encrypted_payload.encode()))

        # Store in hub_state
        hub_state["api_key"] = decrypted.get("api_key", hub_state.get("api_key"))
        hub_state["hub_device_id"] = decrypted.get("hub_device_id")
        
        # Store Kasa creds for poller (could also return them)
        hub_state["kasa_username"] = decrypted.get("kasa_username")
        hub_state["kasa_password"] = decrypted.get("kasa_password")

        logger.info("Config pulled successfully from backend")
        return True

    except requests.RequestException as e:
        logger.error("Failed to pull config from backend: %s", e)
        return False
    except Exception as e:
        logger.error("Failed to decrypt config from backend: %s", e)
        return False


def checkin_with_backend(
    backend_url: str,
    hub_serial: str,
    hub_state: dict,
    local_ip: str | None = None,
) -> bool:
    """
    Tell the backend this hub is online and what IP it's reachable at.
    The backend needs this so it can push config when the user claims the device.
    If the hub is already claimed, the backend pushes config immediately.
    """
    payload: dict = {"serial_number": hub_serial}
    if local_ip:
        payload["local_ip"] = local_ip.strip()

    logger.info("Checking in with backend at %s (local_ip=%r)", backend_url, local_ip)

    try:
        resp = requests.post(
            f"{backend_url}/api/hub/register/",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        hub_state["hub_device_id"] = data.get("hub_device_id")
        hub_state["api_key"] = data.get("api_key")  # Store API key from check-in response
        logger.info(
            "Check-in complete. Status: %s, Hub ID: %s",
            data.get("status"),
            hub_state["hub_device_id"],
        )
        return True

    except requests.RequestException as e:
        logger.error("Failed to check in with backend: %s", e)
        return False


def boot_checkin_loop(backend_url: str, hub_serial: str, hub_state: dict, report_ipv4) -> None:
    """
    On every boot: retry check-in, then pull config.
    Hub will not start Kasa polling until config is pulled.
    """
    time.sleep(5)
    for attempt in range(1, 6):
        ip = report_ipv4() if callable(report_ipv4) else None
        if checkin_with_backend(backend_url, hub_serial, hub_state, local_ip=ip):
            logger.info("Boot check-in succeeded (attempt %s)", attempt)
            # Pull config from backend after successful check-in
            if pull_config_from_backend(backend_url, hub_serial, hub_state):
                logger.info("Config pulled from backend")
            else:
                logger.warning("Failed to pull config; will retry on next sync")
            return
        logger.warning("Boot check-in failed (attempt %s/5); retrying in 10s...", attempt)
        time.sleep(10)


def on_wifi_connected(backend_url, hub_serial, hub_state, wifi_manager, *_args):
    """Called after the hub successfully connects to the user's home WiFi."""
    ip = wifi_manager.hub_report_ipv4()
    checkin_with_backend(backend_url, hub_serial, hub_state, local_ip=ip)

    hub_state["is_provisioned"] = True
    logger.info("Hub is online and checked in with backend.")


def on_wifi_failed(hub_state, wifi_ssid):
    """Called when WiFi connection fails. AP is restarted so user can retry."""
    hub_state["wifi_error"] = wifi_ssid
    logger.warning(f"WiFi connection to '{wifi_ssid}' failed. User can retry via captive portal.")


def sync_devices_with_backend(backend_url: str, hub_serial: str, hub_state: dict, poller: DevicePoller):
    """Send all discovered devices to the backend to be registered under this hub."""
    api_key = hub_state.get("api_key")
    if not api_key:
        logger.warning("Cannot sync devices: no API key yet.")
        return

    devices = [d.to_registration_payload() for d in poller.discovered_devices]
    if not devices:
        logger.info("No discovered devices to sync.")
        return

    try:
        resp = requests.post(
            f"{backend_url}/api/hub/sync/",
            json={"hub_serial": hub_serial, "devices": devices},
            headers={"X-API-Key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Device sync complete: {data.get('created')} created, {data.get('updated')} updated")
    except requests.RequestException as e:
        logger.error(f"Failed to sync devices with backend: {e}")


def apply_config(hub_state: dict, poller: DevicePoller, backend_url: str, hub_serial: str, config: dict):
    """Apply configuration received from the backend, then sync discovered devices."""
    hub_state["api_key"] = config.get("api_key")
    hub_state["hub_device_id"] = config.get("hub_device_id")

    kasa_username = config.get("kasa_username")
    kasa_password = config.get("kasa_password")
    if kasa_username and kasa_password:
        poller.set_kasa_credentials(kasa_username, kasa_password)

    logger.info("Config applied successfully.")

    sync_devices_with_backend(backend_url, hub_serial, hub_state, poller)
