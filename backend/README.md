# Backend

Django REST Framework backend for the Smart Hub IoT platform. Manages users, devices, events, ML inference, and secure communication with hub hardware.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Django Backend                                              │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  REST API    │  │ Celery Tasks │  │  ML Pipeline        │ │
│  │             │  │              │  │                     │ │
│  │ /api/events │  │ push_config  │  │ MoveNet keypoints   │ │
│  │ /api/devices│  │ process_event│  │ → pose classifier   │ │
│  │ /api/devices│  │              │  │ → rule-based action │ │
│  │   /register │  │ (async via   │  │   classification    │ │
│  │   /claim    │  │  Redis)      │  │                     │ │
│  │   /sync     │  │              │  │                     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ PostgreSQL  │  │    Redis     │  │  Keras Model        │ │
│  │ (Supabase)  │  │ (RedisLabs)  │  │  (checkpoints/)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
         ▲                                      │
         │ HTTPS                                │ HTTP (encrypted)
         │                                      ▼
   Frontend (TBD)                      Hub (Raspberry Pi CM4)
```

## Data Model

### CustomUser (`dashboard`)
Extends Django's `AbstractUser` with IoT-specific fields.

| Field | Type | Description |
|---|---|---|
| `address` | CharField | User's address |
| `phone_number` | CharField | Contact number |
| `kasa_username` | CharField | TP-Link Kasa account email |
| `kasa_password` | **EncryptedCharField** | Kasa account password (encrypted at rest) |

### EdgeDevice (`api`)
Represents any device in the system — hubs, PIR sensors, or smart plugs.

| Field | Type | Description |
|---|---|---|
| `id` | UUIDField | Primary key |
| `device_type` | CharField | `smart_hub`, `pir_sensor`, or `smart_plug` |
| `hub_device` | ForeignKey (self) | Parent hub this device belongs to |
| `user` | ForeignKey (CustomUser) | Owner (set when user claims the hub) |
| `serial_number` | CharField | Unique hardware identifier |
| `api_key` | **EncryptedCharField** | Auto-generated, used for hub-to-backend auth |
| `ip_address` | GenericIPAddressField | Hub's public IP (set during check-in) |
| `battery_level` | IntegerField | Battery % (PIR sensors) |
| `is_active` | BooleanField | Whether the device is online |
| `is_provisioned` | BooleanField | Whether user config has been applied |
| `location` | CharField | Room assignment (living_room, bedroom, etc.) |
| `special_use` | CharField | Special function (e.g. medicine_cabinet) |

### EdgeEvent (`api`)
A single observation from the hub, containing pose keypoints and device states.

| Field | Type | Description |
|---|---|---|
| `id` | UUIDField | Primary key |
| `timestamp` | DateTimeField | Auto-set on creation |
| `hub_device` | ForeignKey (EdgeDevice) | Which hub created this event |
| `keypoints` | JSONField | MoveNet pose keypoint data |
| `device_state` | JSONField | Snapshot of all device states at event time |
| `action` | CharField | Classified action: `tv`, `medicine`, `reaching`, `unknown` |
| `pose_classification` | CharField | ML result: `lying`, `reaching`, `sitting`, `standing` |
| `inference_result` | JSONField | Raw model output probabilities |
| `is_processed` | BooleanField | Whether the AI anomaly detector has reviewed this event |
| `is_alert` | BooleanField | Flagged as anomalous by post-processing |
| `is_resolved` | BooleanField | Alert acknowledged by user |

## API Endpoints

### Device Management

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/hub/register/` | None | Hub checks in with serial number; backend stores IP |
| POST | `/api/hub/claim/` | JWT | User claims a device by serial number |
| POST | `/api/hub/sync/` | X-API-Key (hub) | Hub registers discovered PIR/Kasa devices |
| GET | `/api/hub/<serial>/config/` | None | Fallback: hub pulls its config |

### CRUD (via DRF ViewSets)

| Path | ViewSet | Description |
|---|---|---|
| `/api/devices/` | `EdgeDeviceView` | List/retrieve/update/delete devices |
| `/api/events/` | `EdgeEventView` | List/retrieve events; create triggers ML inference |

### Authentication
- **User auth**: JWT via `rest_framework_simplejwt` (60-min access, 7-day refresh)
- **Hub auth**: `X-API-Key` header verified against the hub's `api_key` field via `@require_hub_api_key` decorator

## Celery Tasks

| Task | Trigger | Description |
|---|---|---|
| `push_config_to_hub` | Device register (if claimed) or device claim | Encrypts config with Fernet, POSTs to hub's `/api/config`. Retries 3x with 5s delay. |
| `process_event` | Event creation | Normalizes MoveNet keypoints, runs pose classification model, applies rule-based action classification. |

## ML Pipeline

The `EdgeEventProcessor` in `tasks.py` handles inference:

1. **Keypoint normalization** — centers MoveNet output around hip midpoint, flattens to (n_frames, 51)
2. **Pose classification** — Keras model predicts one of 4 classes: Lying, Reaching, Sitting, Standing
3. **Rule-based action classification** — combines pose result with device states:
   - Reaching + PIR sensor triggered → `reaching`
   - Standing + medicine cabinet PIR triggered → `medicine`
   - Sitting/Lying + smart plug on → `tv`

Model config lives in `models.yml` at the backend root.

## Security

### Encryption at Rest
`EncryptedCharField` (in `utils/fields.py`) transparently encrypts sensitive fields using Fernet before writing to PostgreSQL. Used for:
- `EdgeDevice.api_key`
- `CustomUser.kasa_password`

### Encryption in Transit
Config payloads pushed to the hub are Fernet-encrypted before sending over HTTP, so secrets (API keys, Kasa credentials) are not exposed in plaintext.

### Key Management
A single `FIELD_ENCRYPTION_KEY` (Fernet key) is shared between the backend and hub firmware. Set via environment variable and loaded in `settings.py`.

## File Structure

```
backend/
├── config/
│   ├── settings.py          # Django settings, Celery, encryption key
│   ├── urls.py              # Root URL config → api/
│   ├── wsgi.py
│   └── celery.py
├── api/
│   ├── models.py            # EdgeDevice, EdgeEvent
│   ├── views.py             # REST endpoints + device provisioning views
│   ├── urls.py              # API URL routing
│   ├── serializers.py       # DRF serializers
│   ├── tasks.py             # Celery tasks (push_config, process_event, ML inference)
│   ├── decorators.py        # @require_hub_api_key
│   └── ml/
│       └── ml.py            # MLProcessor — Keras model loading and inference
├── dashboard/
│   └── models.py            # CustomUser (extends AbstractUser)
├── utils/
│   ├── __init__.py
│   └── fields.py            # EncryptedCharField (Fernet)
└── models.yml               # ML model paths and config
```

## Environment Variables

| Variable | Description |
|---|---|
| `DB_CONNECTION_STRING` | PostgreSQL connection string (Supabase) |
| `REDIS_URL` | Redis connection string (Celery broker + result backend) |
| `FIELD_ENCRYPTION_KEY` | Fernet key for database encryption and hub config encryption |

## Provisioning Flow

```
1. Hub powers on
   └─► Creates WiFi AP (captive portal)

2. User connects phone to AP
   └─► Enters home WiFi credentials
       └─► Hub connects to home WiFi
           └─► Hub POSTs to /api/hub/register/
               └─► Backend stores hub IP

3. User goes to website
   └─► Enters hub serial number
       └─► Frontend POSTs to /api/hub/claim/
           └─► Backend links device to user
               └─► Celery task: push_config_to_hub
                   └─► Encrypts API key + Kasa creds
                       └─► POSTs to hub's /api/config
                           └─► Hub applies config
                               └─► Hub POSTs to /api/hub/sync/
                                   └─► Backend creates EdgeDevice records
                                       for all discovered PIR/Kasa devices
```

## Running Locally

```bash
cd backend
pip install -r ../requirements.txt

# Set environment variables
export DB_CONNECTION_STRING=postgresql://...
export REDIS_URL=redis://...
export FIELD_ENCRYPTION_KEY=<your-fernet-key>

# Run migrations
python manage.py migrate

# Start Django
python manage.py runserver

# Start Celery worker (separate terminal)
celery -A config worker -l info
```

Generate a Fernet key:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Deploy to Render (one instance: Django + Celery worker + beat)

One **Web Service** runs **Gunicorn**, **Celery worker**, and **Celery beat** together via [Honcho](https://github.com/nickstenning/honcho) and `Procfile`.

1. **Repo root** includes `render.yaml` (Blueprint). In the Render dashboard: **New → Blueprint** → connect the repo, or create a **Web Service** manually:
   - **Runtime:** Docker  
   - **Dockerfile path:** `backend/Dockerfile`  
   - **Docker context:** `backend`

2. **Environment variables** (dashboard):
   | Variable | Notes |
   |----------|--------|
   | `SECRET_KEY` | Django secret (generate a long random string) |
   | `DEBUG` | `False` |
   | `ALLOWED_HOSTS` | Your service hostname, e.g. `your-app.onrender.com` (comma-separated if several) |
   | `DATABASE_URL` | Your existing Postgres URL (e.g. Supabase); same as `DB_CONNECTION_STRING` if you prefer that name |
   | `REDIS_URL` | Render Redis, Upstash, or Redis Cloud (Celery broker + result backend) |
| `GEMINI_API_KEY` | Used by hourly `run_post_process_events` (Gemini anomaly detection) |
| _(temp)_ | `POST /api/tasks/run-post-process/` is a **public** test hook to queue Gemini post-processing; **remove before production**. |

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/tasks/run-post-process/"
```
   | `FIELD_ENCRYPTION_KEY` | Same Fernet key as hub firmware |
   | `CORS_ALLOW_ALL_ORIGINS` | Currently **all origins allowed** in `settings.py` for convenience. Lock down before production (set `False` and configure `CORS_ALLOWED_ORIGINS` in code or env). |

3. **Health check:** use path `/admin/login/` (or add a simple `/health/` view later).

4. **Local Docker test:**
   ```bash
   cd backend
   docker build -t ece-backend .
   docker run --rm -p 8000:8000 -e PORT=8000 \
     -e DATABASE_URL=... -e REDIS_URL=... -e SECRET_KEY=dev -e ALLOWED_HOSTS='*' \
     -e FIELD_ENCRYPTION_KEY=... \
     ece-backend
   ```

**Note:** TensorFlow in the image makes builds slow and the image large. If the API does not need TF at runtime, trim `requirements.txt` for production to speed deploys.

**Out of memory on Render (free ~512MB):** The stack runs Gunicorn + Celery worker + beat in one container. TensorFlow is **lazy-loaded** only inside the Celery worker when `process_event` runs (not in web workers). `Procfile` uses **Gunicorn `--workers 1`** and **`celery worker --concurrency=1`** to avoid multiple TF copies. The Dockerfile sets `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` to cap thread overhead. If you still OOM, upgrade the Render plan RAM or split the **Celery worker** into a second service (its own 512MB) and run only `web` + `beat` on the web service (advanced).
