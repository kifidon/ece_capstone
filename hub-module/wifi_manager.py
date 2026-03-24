import os
import re
import subprocess
import logging
import platform
import time

logger = logging.getLogger(__name__)

AP_INTERFACE = "wlan0"
AP_IP = "192.168.50.1"
NM_UNMANAGE_WLAN0 = "/etc/NetworkManager/conf.d/10-unmanage-wlan0.conf"
DHCP_RANGE_START = "192.168.50.10"
DHCP_RANGE_END = "192.168.50.50"


class WifiManager:
    """
    Manages the hub's WiFi: AP mode with captive portal for initial setup,
    then switches to the user's home WiFi.
    Only runs real commands on Linux (Raspberry Pi OS).
    On macOS, logs what it would do (for development).
    """

    def __init__(self, hub_serial: str):
        self.ap_ssid = f"SmartHub-{hub_serial[-6:]}"
        self.is_ap_active = False
        self._is_linux = platform.system() == "Linux"

    def _ensure_wlan0_managed(self) -> None:
        """Allow NetworkManager to control wlan0 (required for station mode)."""
        if not self._is_linux or not os.path.isfile(NM_UNMANAGE_WLAN0):
            return
        try:
            os.remove(NM_UNMANAGE_WLAN0)
            subprocess.run(
                ["sudo", "systemctl", "reload", "NetworkManager"],
                check=True,
                capture_output=True,
                text=True,
            )
            time.sleep(2)
        except Exception as e:
            logger.warning("Could not make wlan0 managed: %s", e)

    def _ensure_wlan0_unmanaged_for_ap(self) -> None:
        """Restore NM drop-in so hostapd can own wlan0 without fighting NetworkManager."""
        if not self._is_linux:
            return
        if os.path.isfile(NM_UNMANAGE_WLAN0):
            return
        try:
            os.makedirs(os.path.dirname(NM_UNMANAGE_WLAN0), exist_ok=True)
            with open(NM_UNMANAGE_WLAN0, "w") as f:
                f.write("[keyfile]\nunmanaged-devices=interface-name:wlan0\n")
            subprocess.run(
                ["sudo", "systemctl", "reload", "NetworkManager"],
                check=False,
                capture_output=True,
                text=True,
            )
            time.sleep(2)
        except Exception as e:
            logger.warning("Could not restore wlan0 unmanaged for AP: %s", e)

    def _ethernet_has_ipv4(self) -> bool:
        """True if a non-wlan interface has a global IPv4 (e.g. eth0, end0)."""
        if not self._is_linux:
            return False
        try:
            r = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "scope", "global"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        for line in r.stdout.splitlines():
            m = re.match(r"^\d+:\s+(\S+)\s+inet\s+(\S+)/", line.strip())
            if not m:
                continue
            iface, addr = m.group(1), m.group(2)
            if iface in ("lo", AP_INTERFACE) or addr.startswith("127."):
                continue
            if addr == AP_IP:
                continue
            logger.info("Found IPv4 on %s (%s); treating as wired/uplink.", iface, addr)
            return True
        return False

    def _wlan0_station_ready(self) -> bool:
        """True if wlan0 is a connected client with an IP other than the AP address."""
        if not self._is_linux:
            return False
        try:
            r = subprocess.run(
                ["nmcli", "-t", "-f", "DEVICE,STATE", "device", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        state_line = None
        for line in r.stdout.splitlines():
            if line.startswith(f"{AP_INTERFACE}:"):
                state_line = line.split(":", 1)[1].lower()
                break
        if not state_line or "connected" not in state_line:
            return False
        try:
            r2 = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "dev", AP_INTERFACE, "scope", "global"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        if not r2.stdout.strip():
            return False
        if f" {AP_IP}/" in r2.stdout:
            return False
        return True

    def hub_report_ipv4(self) -> str | None:
        """
        Best-effort LAN IPv4 to send to the backend on check-in (not the AP address).
        Prefers wired interfaces over wlan.
        """
        if not self._is_linux:
            return None
        try:
            r = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "scope", "global"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
        entries: list[tuple[str, str]] = []
        for line in r.stdout.splitlines():
            m = re.match(r"^\d+:\s+(\S+)\s+inet\s+(\S+)/", line.strip())
            if not m:
                continue
            iface, addr = m.group(1), m.group(2)
            if iface == "lo" or addr.startswith("127."):
                continue
            if addr == AP_IP:
                continue
            entries.append((iface, addr))
        if not entries:
            return None

        def _iface_rank(iface: str) -> int:
            if iface.startswith(("eth", "end", "enp", "eno", "usb", "enx")):
                return 0
            if iface == AP_INTERFACE:
                return 2
            return 1

        entries.sort(key=lambda t: _iface_rank(t[0]))
        return entries[0][1]

    def try_existing_lan_before_ap(self) -> bool:
        """
        Prefer an existing uplink before starting the captive-portal AP:
        1) Saved WiFi (NetworkManager on wlan0)
        2) Ethernet (any other interface with a global IPv4)
        Returns True if the hub should skip start_ap().
        """
        if not self._is_linux:
            return False

        self._ensure_wlan0_managed()

        deadline = time.monotonic() + 35.0
        prompted_connect = False
        start = time.monotonic()
        while time.monotonic() < deadline:
            if self._wlan0_station_ready():
                logger.info("wlan0 connected with saved WiFi; skipping WiFi AP.")
                return True
            if self._ethernet_has_ipv4():
                logger.info("Ethernet uplink present; skipping WiFi AP.")
                return True
            if not prompted_connect and (time.monotonic() - start) >= 10.0:
                prompted_connect = True
                subprocess.run(
                    ["sudo", "nmcli", "device", "connect", AP_INTERFACE],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
            time.sleep(2)

        logger.info("No saved WiFi or Ethernet uplink in time; will start WiFi AP.")
        return False

    def _run(self, cmd: list[str], check=True):
        if not self._is_linux:
            logger.info(f"[macOS stub] Would run: {' '.join(cmd)}")
            return
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            if result.stderr:
                logger.error("%s stderr: %s", cmd[0], result.stderr.strip())
            if result.stdout:
                logger.error("%s stdout: %s", cmd[0], result.stdout.strip())
            if check:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result

    def start_ap(self):
        """Create WiFi AP with DNS redirect so all traffic hits the captive portal."""
        logger.info(f"Starting WiFi AP: {self.ap_ssid}")

        if not self._is_linux:
            logger.info(f"[macOS stub] AP '{self.ap_ssid}' would be created on {AP_INTERFACE}")
            self.is_ap_active = True
            return

        self._ensure_wlan0_unmanaged_for_ap()

        hostapd_conf = (
            f"interface={AP_INTERFACE}\n"       # WiFi interface to host the AP on
            f"ssid={self.ap_ssid}\n"            # Network name devices will see
            "hw_mode=g\n"                       # 2.4 GHz band (802.11g) — widest device compatibility
            "channel=7\n"                       # WiFi channel (1-11); 7 avoids common 1/6/11 overlap
            "wmm_enabled=0\n"                   # Disable WiFi Multimedia QoS (unnecessary for setup)
            "macaddr_acl=0\n"                   # No MAC address filtering — allow any device to connect
            "auth_algs=1\n"                     # Open authentication (no password required to join)
            "ignore_broadcast_ssid=0\n"         # Broadcast the SSID so it shows up in WiFi lists
        )
        with open("/tmp/hostapd.conf", "w") as f:
            f.write(hostapd_conf)

        # DNS redirect: resolve ALL domains to the hub's AP IP (captive portal)
        dnsmasq_conf = (
            f"interface={AP_INTERFACE}\n"                                           # Only serve DHCP/DNS on the AP interface
            f"dhcp-range={DHCP_RANGE_START},{DHCP_RANGE_END},255.255.255.0,24h\n"   # IP pool for connected devices, 24h lease
            "bind-interfaces\n"                                                     # Don't bind to wildcard — avoid conflicts with other interfaces
            f"listen-address={AP_IP}\n"                                             # Only listen on the AP's own IP
            f"address=/#/{AP_IP}\n"                                                 # Wildcard DNS: resolve ALL domains to the hub — forces captive portal
        )
        with open("/tmp/dnsmasq.conf", "w") as f:
            f.write(dnsmasq_conf)

        # Clean up any leftover hostapd/dnsmasq from a previous run (avoids "Address already in use")
        self._run(["sudo", "killall", "hostapd"], check=False)
        self._run(["sudo", "killall", "dnsmasq"], check=False)
        time.sleep(1)

        # Reset interface so driver releases previous mode (avoids "Match already configured")
        self._run(["sudo", "ip", "link", "set", AP_INTERFACE, "down"], check=False)
        self._run(["sudo", "iw", "dev", AP_INTERFACE, "set", "type", "__ap"], check=False)
        self._run(["sudo", "ip", "link", "set", AP_INTERFACE, "up"])
        self._run(["sudo", "ip", "addr", "flush", "dev", AP_INTERFACE])
        self._run(["sudo", "ip", "addr", "add", f"{AP_IP}/24", "dev", AP_INTERFACE])
        self._run(["sudo", "hostapd", "-B", "/tmp/hostapd.conf"])
        self._run(["sudo", "dnsmasq", "-C", "/tmp/dnsmasq.conf"])

        # Redirect port 80 -> 5050 so OS captive portal checks (which use :80) hit our app and trigger the popup
        self._run(
            [
                "sudo", "iptables", "-t", "nat", "-D", "PREROUTING",
                "-i", AP_INTERFACE, "-p", "tcp", "--dport", "80",
                "-j", "REDIRECT", "--to-ports", "5050",
            ],
            check=False,
        )
        self._run(
            [
                "sudo", "iptables", "-t", "nat", "-A", "PREROUTING",
                "-i", AP_INTERFACE, "-p", "tcp", "--dport", "80",
                "-j", "REDIRECT", "--to-ports", "5050",
            ],
        )

        self.is_ap_active = True
        logger.info(f"AP '{self.ap_ssid}' is active. Captive portal at http://{AP_IP}:5050/")

    def stop_ap(self):
        if not self.is_ap_active:
            return

        logger.info("Stopping WiFi AP...")
        self._run(
            [
                "sudo", "iptables", "-t", "nat", "-D", "PREROUTING",
                "-i", AP_INTERFACE, "-p", "tcp", "--dport", "80",
                "-j", "REDIRECT", "--to-ports", "5050",
            ],
            check=False,
        )
        self._run(["sudo", "killall", "hostapd"], check=False)
        self._run(["sudo", "killall", "dnsmasq"], check=False)
        self._run(["sudo", "ip", "addr", "flush", "dev", AP_INTERFACE], check=False)
        # Return interface to managed mode so NetworkManager can scan and connect
        self._run(["sudo", "ip", "link", "set", AP_INTERFACE, "down"], check=False)
        self._run(["sudo", "iw", "dev", AP_INTERFACE, "set", "type", "managed"], check=False)
        self._run(["sudo", "ip", "link", "set", AP_INTERFACE, "up"], check=False)
        self.is_ap_active = False
        logger.info("AP stopped.")

    def scan_networks(self) -> list[dict]:
        """
        Scan for visible WiFi networks. Returns list of {"ssid": str, "signal": int} sorted by signal (strongest first).
        May return empty when the interface is in AP mode (single-radio Pi); manual SSID entry still works.
        """
        if not self._is_linux:
            # Stub for macOS dev: return fake networks
            return [
                {"ssid": "HomeNetwork", "signal": -45},
                {"ssid": "Neighbor_5G", "signal": -72},
            ]
        try:
            result = subprocess.run(
                ["iw", "dev", AP_INTERFACE, "scan"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("iw scan failed (e.g. interface in AP mode): %s", result.stderr or result.stdout)
                return []
            # Parse BSS blocks: signal (dBm) and SSID; dedupe by SSID keeping strongest signal
            by_ssid: dict[str, int] = {}
            current_signal: int | None = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("signal:"):
                    try:
                        current_signal = int(float(line.replace("signal:", "").replace("dBm", "").strip()))
                    except (ValueError, TypeError):
                        current_signal = None
                elif line.startswith("SSID:"):
                    ssid = line[5:].strip()
                    if ssid and current_signal is not None:
                        if ssid not in by_ssid or by_ssid[ssid] < current_signal:
                            by_ssid[ssid] = current_signal
                    current_signal = None
            out = [{"ssid": s, "signal": sig} for s, sig in by_ssid.items()]
            out.sort(key=lambda x: -x["signal"])
            return out
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("WiFi scan error: %s", e)
            return []

    def connect_to_wifi(self, ssid: str, password: str) -> bool:
        """Connect the hub to the user's home WiFi network. Returns True on success."""
        logger.info(f"Connecting to WiFi network: {ssid}")

        if not self._is_linux:
            logger.info(f"[macOS stub] Would connect to '{ssid}'")
            return True

        # wlan0 is normally "unmanaged" so hostapd can use it for AP. For connecting to
        # home WiFi we need NM to manage it: remove the unmanage config and reload NM.
        self._ensure_wlan0_managed()

        # Give NM time to see the interface, then rescan so the target SSID is in the list
        time.sleep(1)
        subprocess.run(
            ["sudo", "nmcli", "device", "wifi", "rescan", "ifname", AP_INTERFACE],
            check=False, capture_output=True, text=True, timeout=15,
        )
        time.sleep(2)

        # Add connection with explicit key-mgmt (avoids "key-mgmt: property is missing");
        # "device wifi connect" with -- is not supported on all nmcli versions.
        subprocess.run(
            ["sudo", "nmcli", "connection", "delete", ssid],
            check=False, capture_output=True, text=True,
        )
        add_result = subprocess.run(
            [
                "sudo", "nmcli", "connection", "add", "type", "wifi", "con-name", ssid,
                "ifname", AP_INTERFACE, "ssid", ssid,
                "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password,
            ],
            check=False, capture_output=True, text=True, timeout=15,
        )
        if add_result.returncode != 0:
            logger.error("Failed to add connection: %s", add_result.stderr or add_result.stdout or "")
            result = add_result
        else:
            result = subprocess.run(
                ["sudo", "nmcli", "connection", "up", ssid, "ifname", AP_INTERFACE],
                check=False, capture_output=True, text=True, timeout=30,
            )
        if result.returncode != 0 and "No network with SSID" in (result.stderr or result.stdout or ""):
            logger.info("SSID not in scan yet, rescanning and retrying once...")
            subprocess.run(
                ["sudo", "nmcli", "device", "wifi", "rescan", "ifname", AP_INTERFACE],
                check=False, capture_output=True, text=True, timeout=15,
            )
            time.sleep(2)
            result = subprocess.run(
                ["sudo", "nmcli", "connection", "up", ssid, "ifname", AP_INTERFACE],
                check=False, capture_output=True, text=True, timeout=30,
            )

        if result.returncode == 0:
            logger.info(f"Connected to {ssid}")
            return True
        logger.error(f"Failed to connect to {ssid}: %s", result.stderr or result.stdout or "")
        self._ensure_wlan0_unmanaged_for_ap()
        return False

    def provision_and_switch(self, wifi_ssid: str, wifi_password: str, pir_devices: list, on_connected=None, on_failed=None):
        """
        Try connecting to home WiFi first. If it works, provision PIR sensors
        and tear down the AP. If it fails, keep the AP up so the user can retry.
        """
        self.stop_ap()

        if not self.connect_to_wifi(wifi_ssid, wifi_password):
            logger.warning("WiFi connection failed. Restarting AP for retry.")
            self.start_ap()
            if on_failed:
                on_failed(wifi_ssid)
            return

        from poller import DevicePoller
        DevicePoller.send_wifi_to_pir_devices(pir_devices, wifi_ssid, wifi_password)

        logger.info("Hub provisioning complete. Now on home WiFi.")

        if on_connected:
            on_connected(wifi_ssid, wifi_password)
