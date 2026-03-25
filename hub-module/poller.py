import asyncio
import json
import socket
import logging
import time

from kasa import Discover, Credentials

logger = logging.getLogger(__name__)

KASA_POLL_INTERVAL = 60
PIR_PROVISION_PORT = 9998


class Device:
    def __init__(self, serial_number: str, device_type: str, local_ip: str = None):
        self.serial_number = serial_number
        self.device_type = device_type
        self.local_ip = local_ip
        self.battery_level = None
        self.alias = None
        self.is_on = None
        self.last_seen = time.time()

    def to_registration_payload(self) -> dict:
        return {
            "device_type": self.device_type,
            "serial_number": self.serial_number,
            "battery_level": self.battery_level,
            "last_seen": self.last_seen,
            "alias": self.alias,
            "is_on": self.is_on,
        }

    def __repr__(self):
        name = self.alias or self.serial_number
        return f"<Device {self.device_type} serial={self.serial_number} ip={self.local_ip} name={name}>"


class DevicePoller:
    """Discovers PIR sensors (UDP broadcast) and Kasa smart plugs on the local network."""

    PIR_BROADCAST_PORT = 9999

    def __init__(self):
        self.discovered_devices: list[Device] = []
        self.kasa_credentials: Credentials | None = None
        self._sock = None

    def set_kasa_credentials(self, username: str, password: str):
        """Called when the backend pushes user config with Kasa credentials."""
        self.kasa_credentials = Credentials(username, password)
        logger.info("Kasa credentials configured from backend.")

    # --- PIR sensor discovery (UDP broadcast, runs forever) ---

    def start_pir_listener(self):
        """Listen for PIR sensor UDP broadcasts. Runs forever in a background thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", self.PIR_BROADCAST_PORT))

        logger.info(f"Listening for PIR sensor broadcasts on port {self.PIR_BROADCAST_PORT}...")

        while True:
            try:
                data, (sender_ip, _) = self._sock.recvfrom(1024)
            except OSError as e:
                logger.error("PIR recv failed: %s", e)
                time.sleep(0.5)
                continue

            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                # Port 9999 sees random LAN noise (SSDP, other apps); only ESP text beacons are UTF-8.
                logger.debug(
                    "Ignoring non-UTF-8 datagram on port %s from %s (%d bytes)",
                    self.PIR_BROADCAST_PORT,
                    sender_ip,
                    len(data),
                )
                continue

            parts = message.split(":")
            if len(parts) != 3:
                logger.debug("Skipping non-PIR-shaped message from %s", sender_ip)
                continue

            serial_number, device_type, battery_str = parts

            try:
                existing = next(
                    (d for d in self.discovered_devices if d.serial_number == serial_number),
                    None,
                )
                if existing:
                    try:
                        existing.battery_level = int(battery_str)
                    except ValueError:
                        pass
                    continue

                device = Device(
                    serial_number=serial_number,
                    device_type=device_type,
                    local_ip=sender_ip,
                )
                try:
                    device.battery_level = int(battery_str)
                except ValueError:
                    logger.warning("Invalid battery level from %s: %s", sender_ip, battery_str)
                self.discovered_devices.append(device)
                logger.info("Discovered %s", device)
            except Exception as e:
                logger.error("Error handling PIR broadcast from %s: %s", sender_ip, e)
                
    def get_devices(self, device_type: str | None = None) -> list[Device]:
        if device_type is None:
            return self.discovered_devices
        return [d for d in self.discovered_devices if d.device_type == device_type]

    def start_periodic_sync(self, backend_url: str, hub_serial: str, hub_state: dict):
        """
        Every 5 minutes, send all discovered devices (with latest state) to the backend.
        Runs in a background thread.
        """
        import requests
        
        interval = 5 * 60  # 5 minutes
        while True:
            time.sleep(interval)
            
            api_key = hub_state.get("api_key")
            if not api_key:
                logger.debug("Skipping periodic sync: no API key yet")
                continue
            
            devices_payload = [d.to_registration_payload() for d in self.discovered_devices]
            if not devices_payload:
                logger.debug("No devices to sync")
                continue
            
            payload = {
                "hub_serial": hub_serial,
                "devices": devices_payload,
            }
            
            try:
                url = f"{backend_url.rstrip('/')}/api/hub/sync/"
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"X-API-Key": api_key},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(
                    f"Periodic device sync: {data.get('created', 0)} created, {data.get('updated', 0)} updated"
                )
            except requests.RequestException as e:
                logger.error(f"Periodic sync failed: {e}")

    # --- Send WiFi credentials to PIR devices ---

    @staticmethod
    def send_wifi_to_pir_devices(devices: list, wifi_ssid: str, wifi_password: str):
        """
        Send WiFi credentials to all discovered PIR sensors over the AP network.
        Uses UDP unicast to each device's known IP on the provisioning port.
        ESP32 PIR firmware should be listening on PIR_PROVISION_PORT for this.
        """
        pir_devices = [d for d in devices if d.device_type == "pir_sensor" and d.local_ip]
        if not pir_devices:
            logger.info("No PIR devices to provision.")
            return

        payload = json.dumps({
            "wifi_ssid": wifi_ssid,
            "wifi_password": wifi_password,
        }).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for device in pir_devices:
                try:
                    sock.sendto(payload, (device.local_ip, PIR_PROVISION_PORT))
                    logger.info(f"Sent WiFi credentials to PIR {device.serial_number} at {device.local_ip}")
                except Exception as e:
                    logger.error(f"Failed to send WiFi creds to {device.serial_number}: {e}")
        finally:
            sock.close()

    # --- Kasa smart plug discovery (polls periodically) ---
    def run_kasa_discovery_once(self):
        """Run a single Kasa discovery cycle (blocking). Used for immediate discovery on boot."""
        if self.kasa_credentials is None:
            logger.warning("Cannot run Kasa discovery: no credentials set.")
            return
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._discover_kasa_devices())
        except Exception as e:
            logger.error(f"Kasa discovery error (one-shot): {e}")
        finally:
            loop.close()

    def start_kasa_poller(self):
        """Poll for Kasa smart plugs periodically. Runs forever in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            if self.kasa_credentials is None:
                logger.debug("No Kasa credentials yet — skipping discovery cycle.")
                time.sleep(KASA_POLL_INTERVAL)
                continue

            try:
                loop.run_until_complete(self._discover_kasa_devices())
            except Exception as e:
                logger.error(f"Kasa discovery error: {e}")
            time.sleep(KASA_POLL_INTERVAL)

    async def _discover_kasa_devices(self):
        logger.info("Running Kasa device discovery...")
        found = await Discover.discover(credentials=self.kasa_credentials)

        for ip, kasa_dev in found.items():
            try:
                await kasa_dev.update()
            except Exception as e:
                logger.warning(f"Failed to update Kasa device at {ip}: {e}")
                continue

            if not kasa_dev.is_plug:
                continue

            serial = kasa_dev.device_id

            existing = next(
                (d for d in self.discovered_devices if d.serial_number == serial),
                None,
            )
            if existing:
                existing.is_on = kasa_dev.is_on
                existing.alias = kasa_dev.alias
                continue

            device = Device(
                serial_number=serial,
                device_type="smart_plug",
                local_ip=ip,
            )
            device.alias = kasa_dev.alias
            device.is_on = kasa_dev.is_on
            self.discovered_devices.append(device)
            logger.info(f"Discovered Kasa plug: {device}")

        plug_count = len([d for d in self.discovered_devices if d.device_type == "smart_plug"])
        logger.info(f"Kasa discovery complete. {plug_count} plug(s) total.")

    async def _fetch_kasa_plug_status(self, device: Device) -> dict | None:
        """Query a single Kasa plug by IP; returns current state or None on failure."""
        if not device.local_ip or self.kasa_credentials is None:
            return None
        try:
            kasa_dev = await Discover.discover_single(
                device.local_ip,
                credentials=self.kasa_credentials,
            )
            await kasa_dev.update() 
            device.last_seen = time.time()
            device.is_on = bool(getattr(kasa_dev, "is_on", None))
            return device.to_registration_payload()
        
        except Exception as e:
            logger.warning(f"Failed to fetch status for {device.serial_number} at {device.local_ip}: {e}")
            return None

    def get_device_status(self, device: Device) -> dict:
        """
        Actively poll the device's current status. For smart_plug, connects to the
        device by IP and fetches fresh state; for pir_sensor, returns last-known state
        (PIRs push updates via UDP, there is no on-demand query).
        """
        if device.device_type == "smart_plug":
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(self._fetch_kasa_plug_status(device))
                if status is None:
                    raise RuntimeError(f"Failed to fetch status for {device.serial_number} at {device.local_ip}")
                return status
            
            finally:
                loop.close()
        elif device.device_type == "pir_sensor":
            return device.to_registration_payload()
        else:
            raise ValueError(f"Invalid device type: {device.device_type}")


    def get_device_by_serial(self, serial_number: str) -> Device:
        for device in self.discovered_devices:
            if device.serial_number == serial_number:
                return device
        raise ValueError(f"Device not found: {serial_number}")

    def try_get_device_by_serial(self, serial_number: str) -> Device | None:
        for device in self.discovered_devices:
            if device.serial_number == serial_number:
                return device
        return None

    def stop(self):
        if self._sock:
            self._sock.close()
