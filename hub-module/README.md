# Hub Module

Flask-based firmware for the Raspberry Pi CM4 Smart Hub. Runs as a systemd service on boot and handles device discovery, WiFi provisioning, and secure communication with the backend.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Raspberry Pi CM4 (Hub)                             │
│                                                     │
│  ┌──────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ Flask API │  │ DevicePoller│  │  WifiManager   │  │
│  │ (port 5050)│  │            │  │                │  │
│  │           │  │ PIR listener│  │ AP mode ←→     │  │
│  │ /api/*    │  │ (UDP:9999) │  │ Home WiFi      │  │
│  │ /health   │  │            │  │                │  │
│  │ / (portal)│  │ Kasa poller│  │ Captive portal │  │
│  └──────────┘  │ (async)    │  │ (hostapd +     │  │
│                └────────────┘  │  dnsmasq)       │  │
│                                └────────────────┘  │
└─────────────────────────────────────────────────────┘
         │                              │
         │ HTTPS (check-in/sync)        │ UDP (broadcast)
         ▼                              ▼
   Django Backend               PIR Sensors (ESP32)
```

## Boot Sequence

1. **Start WiFi AP** — creates a `SmartHub-XXXXXX` network (open, no password) using `hostapd` and `dnsmasq`. All DNS queries redirect to the hub to force the captive portal.
2. **Start background threads** — PIR sensor UDP listener (port 9999) and Kasa smart plug poller run continuously.
3. **User connects phone to AP** — OS detects captive portal and opens the setup page automatically.
4. **User submits WiFi credentials** — hub stops the AP, connects to home WiFi via `nmcli`, and pushes WiFi creds to discovered PIR sensors (UDP unicast, port 9998).
5. **Hub checks in with backend** — POSTs its serial number to `/api/hub/register/`. Backend stores the hub's IP.
6. **Backend pushes config** — if the hub is claimed by a user, the backend sends an encrypted config payload (API key, Kasa credentials) to the hub's `/api/config` endpoint.
7. **Hub syncs devices** — after receiving config, the hub sends all discovered devices to `/api/hub/sync/` so they're registered in the backend DB.

If WiFi connection fails, the AP restarts and the user sees a retry page.

## File Structure

| File | Purpose |
|---|---|
| `app.py` | Flask application, API routes, startup orchestration |
| `captive_portal.py` | Blueprint with captive portal routes (`/`, `/setup`, OS detection endpoints) |
| `poller.py` | `DevicePoller` — discovers PIR sensors (UDP) and Kasa smart plugs (`python-kasa`) |
| `wifi.py` | `WifiManager` — AP mode (hostapd/dnsmasq), home WiFi connection (nmcli) |
| `callbacks.py` | Backend communication: check-in, config application, device sync |
| `decorators.py` | `@require_api_key` decorator for protected Flask routes |
| `templates/` | Jinja2 HTML templates for captive portal pages |

## API Endpoints

### Captive Portal (no auth, served over AP)

| Method | Path | Description |
|---|---|---|
| GET | `/` | WiFi setup page (or failure/retry page) — scan list + pick network, then enter password |
| GET | `/wifi/scan` | JSON list of visible networks (for setup page; may be empty when Pi is in AP mode) |
| POST | `/setup` | Submit WiFi SSID and password |
| GET | `/generate_204` | Android captive portal detection → redirects to `/` |
| GET | `/hotspot-detect.html` | Apple captive portal detection → redirects to `/` |
| GET | `/connecttest.txt` | Windows captive portal detection → redirects to `/` |

### Hub API (auth required after provisioning)

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Health check |
| POST | `/api/config` | Fernet-encrypted payload, X-API-Key after first config | Receive config push from backend |
| GET | `/api/ping` | X-API-Key | Connectivity check, returns hub status |
| GET | `/api/devices` | X-API-Key | List all discovered devices |

## Environment Variables

| Variable | Description |
|---|---|
| `BACKEND_URL` | Django backend URL (default: `http://localhost:8000`) |
| `FIELD_ENCRYPTION_KEY` | Fernet symmetric key shared with backend for encrypting config payloads in transit |
| `HUB_LOG_FILE` | Log file path (default: `/var/log/smarthub.log`). Rotating, 2 MB × 5 backups. |
| `HUB_SKIP_AP` | Set to `1` to skip starting the WiFi AP (use when Pi is already on WiFi, e.g. dev). |
| `HUB_CAMERA_ENABLED` | Set to `1` to enable USB camera rolling buffer (default: off). |
| `HUB_CAMERA_DEVICE` | Camera device index or path (default: `0`). |
| `HUB_CAMERA_BUFFER_S` | Seconds to keep in ring buffer (default: `10`). |
| `HUB_CAMERA_FPS` | Capture rate in fps (default: `30`). |
| `HUB_CAMERA_WIDTH` / `HUB_CAMERA_HEIGHT` | Frame size (default: 640×480). |

## Security

- **Config payloads are Fernet-encrypted** (AES-128-CBC + HMAC-SHA256) before being sent over HTTP from the backend to the hub.
- **API key authentication** — protected endpoints require a valid `X-API-Key` header. The key is generated server-side and pushed to the hub during provisioning.
- **First config push is unauthenticated** — necessary because the hub doesn't have an API key until it receives the first config. Subsequent pushes require the key.
- **macOS development stubs** — WiFi/AP commands are stubbed on non-Linux platforms so the app can run locally for development.

## Device Discovery

### PIR Sensors
- ESP32-based sensors broadcast `serial:type:battery` on UDP port 9999.
- The hub listens continuously and maintains an in-memory list.
- During provisioning, the hub pushes home WiFi credentials to each sensor via UDP unicast on port 9998.

### Kasa Smart Plugs
- Discovered using the `python-kasa` library with user-provided Kasa account credentials.
- Polling runs every 60 seconds in a background thread.
- Only starts after Kasa credentials are received from the backend.

## Dependencies

```
flask
requests
python-kasa
cryptography
opencv-python-headless>=4.8.0
```

## Running Locally

```bash
cd hub-module
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
export FIELD_ENCRYPTION_KEY=<your-fernet-key>
python app.py
```

## Production (Raspberry Pi)

Run as a systemd service:

```ini
[Unit]
Description=SmartHub Flask API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/hub-module
Environment=BACKEND_URL=https://your-backend.com
Environment=FIELD_ENCRYPTION_KEY=<your-fernet-key>
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
