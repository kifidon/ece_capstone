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
| `type` | CharField | `smart_hub`, `pir_sensor`, or `smart_plug` |
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
| POST | `/api/devices/register/` | None | Hub checks in with serial number; backend stores IP |
| POST | `/api/devices/claim/` | JWT | User claims a device by serial number |
| POST | `/api/devices/sync/` | X-API-Key (hub) | Hub registers discovered PIR/Kasa devices |
| GET | `/api/devices/<serial>/config/` | None | Fallback: hub pulls its config |

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
           └─► Hub POSTs to /api/devices/register/
               └─► Backend stores hub IP

3. User goes to website
   └─► Enters hub serial number
       └─► Frontend POSTs to /api/devices/claim/
           └─► Backend links device to user
               └─► Celery task: push_config_to_hub
                   └─► Encrypts API key + Kasa creds
                       └─► POSTs to hub's /api/config
                           └─► Hub applies config
                               └─► Hub POSTs to /api/devices/sync/
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
