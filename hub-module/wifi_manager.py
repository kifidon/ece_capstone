import os
import subprocess
import logging
import platform
import time

logger = logging.getLogger(__name__)

AP_INTERFACE = "wlan0"
AP_IP = "192.168.50.1"
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
        nm_unmanage = "/etc/NetworkManager/conf.d/10-unmanage-wlan0.conf"
        if os.path.isfile(nm_unmanage):
            try:
                os.remove(nm_unmanage)
                subprocess.run(
                    ["sudo", "systemctl", "reload", "NetworkManager"],
                    check=True, capture_output=True, text=True,
                )
                time.sleep(2)
            except Exception as e:
                logger.warning("Could not make wlan0 managed: %s", e)

        # Give NM time to see the interface, then rescan so the target SSID is in the list
        time.sleep(1)
        subprocess.run(
            ["sudo", "nmcli", "device", "wifi", "rescan", "ifname", AP_INTERFACE],
            check=False, capture_output=True, text=True, timeout=15,
        )
        time.sleep(2)

        def do_connect():
            return subprocess.run(
                ["sudo", "nmcli", "device", "wifi", "connect", ssid,
                 "password", password, "ifname", AP_INTERFACE],
                check=False, capture_output=True, text=True, timeout=30,
            )

        result = do_connect()
        if result.returncode != 0 and "No network with SSID" in (result.stderr or result.stdout or ""):
            logger.info("SSID not in scan yet, rescanning and retrying once...")
            subprocess.run(
                ["sudo", "nmcli", "device", "wifi", "rescan", "ifname", AP_INTERFACE],
                check=False, capture_output=True, text=True, timeout=15,
            )
            time.sleep(2)
            result = do_connect()

        if result.returncode == 0:
            logger.info(f"Connected to {ssid}")
            return True
        logger.error(f"Failed to connect to {ssid}: %s", result.stderr or result.stdout or "")
        # Re-create unmanage config so next start_ap() can use wlan0 again
        if not os.path.isfile(nm_unmanage):
            try:
                os.makedirs(os.path.dirname(nm_unmanage), exist_ok=True)
                with open(nm_unmanage, "w") as f:
                    f.write("[keyfile]\nunmanaged-devices=interface-name:wlan0\n")
                subprocess.run(
                    ["sudo", "systemctl", "reload", "NetworkManager"],
                    check=False, capture_output=True, text=True,
                )
            except Exception:
                pass
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
