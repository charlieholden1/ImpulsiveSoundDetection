# Impulsive Sound Detection (ISD) System

**ENEB453 Web-Based Application Development — Spring 2026**  
**Team:** Mathew Ridgely · Sholom kott (Web App) · Charlie Holden (ML Pipeline)

---

## Overview

The Impulsive Sound Detection (ISD) system is a full-stack cyber-physical web application for real-time gunshot and glass-break detection in school environments. It combines a two-stage machine learning audio pipeline running on Raspberry Pi 5 compute nodes with a multi-node MQTT messaging architecture and a Node.js/Express web dashboard.

### Key Features

- **Two-stage ML pipeline** — RMS energy trigger (Stage 1) feeds YAMNet or a custom EfficientNetB0 CNN classifier (Stage 2)
- **Multi-node MQTT architecture** — RPi5 nodes publish detections, RMS frames, and heartbeats to a central broker
- **Live web dashboard** — four-tab interface: Live Feed, Event History, Query Lab, Maintenance Mode
- **Node management** — auto-discovery of new nodes, registration, renaming, diagnostics
- **Authenticated admin routes** — API key protection on all write/admin endpoints
- **Dockerized deployment** — single `docker compose up` starts the dashboard server

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                             │
│                                                                 │
│  ┌──────────────────┐    ┌────────────────────────────────┐    │
│  │  host_subscriber │    │   dashboard_server (Docker)    │    │
│  │  (Python)        │───▶│   Node.js + Express            │    │
│  │  writes host.db  │    │   sql.js reads host.db         │    │
│  └──────────────────┘    │   serves http://localhost:3000 │    │
│           ▲              └────────────────────────────────┘    │
│           │ MQTT subscribe                                      │
│  ┌────────┴─────────┐                                          │
│  │ Mosquitto Broker │                                          │
│  │ port 1883        │                                          │
│  └────────┬─────────┘                                          │
└───────────┼─────────────────────────────────────────────────────┘
            │ MQTT publish
  ┌─────────┴──────────────────────────────────────────┐
  │              RPi5 NODE NETWORK (LAN)                │
  │                                                     │
  │  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
  │  │  node_1    │  │  node_2    │  │  node_N    │   │
  │  │  RPi5      │  │  RPi5      │  │  RPi5      │   │
  │  │  MEMS Mic  │  │  MEMS Mic  │  │  MEMS Mic  │   │
  │  │  YAMNet    │  │  YAMNet    │  │  YAMNet    │   │
  │  └────────────┘  └────────────┘  └────────────┘   │
  └─────────────────────────────────────────────────────┘
```

### ML Pipeline (per node)

```
Audio Input → Pre-emphasis → 512-sample RMS frames
    → Rolling Baseline Comparison (Stage 1: energy trigger)
        → 0.975s window → YAMNet / EfficientNetB0 (Stage 2)
            → ClassificationResult → MQTT publish → host.db
```

---

## Repository Structure

```
ImpulsiveSoundDetection/
├── impulsive_sound_detection/          ← Python package (RPi + host)
│   ├── config.py                       ← All constants, node identity, MQTT settings
│   ├── pipeline.py                     ← Stage 1 + 2 orchestration
│   ├── stream_monitor.py               ← Stage 1: RMS energy trigger
│   ├── classifier.py                   ← Stage 2: YAMNet + CNN classifiers
│   ├── mqtt_bridge.py                  ← Node-side MQTT publisher
│   ├── host_subscriber.py              ← Host-side MQTT subscriber → host.db
│   ├── event_logger.py                 ← SQLite + JSONL event logging
│   ├── live_stream.py                  ← Live microphone capture
│   ├── main.py                         ← CLI entry point
│   ├── data_loader.py                  ← VOICe dataset loader
│   ├── augmentor.py                    ← Audio augmentation
│   ├── spectrogram_utils.py            ← FFT / LogMel / MFCC spectrogram rendering
│   ├── visualizer.py                   ← Detection plots (matplotlib)
│   ├── dashboard.py                    ← Terminal dashboard (ANSI)
│   ├── gui.py                          ← GUI dashboard (customtkinter)
│   └── dashboard_server/               ← Web dashboard (Docker)
│       ├── index.js                    ← Express server + sql.js + REST API
│       ├── public/index.html           ← Single-page dashboard (4 tabs)
│       ├── package.json
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── .env.example                ← Copy to .env and fill in values
│       └── .dockerignore
├── train/                              ← EfficientNetB0 training scripts
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── feature_sweep.py
│   └── evaluate.py
├── models/                             ← Trained .keras + .tflite models (not in git)
├── reports/                            ← Training metrics, confusion matrices
├── logs/                               ← host.db + per-node JSONL logs
├── test_mqtt.py                        ← Publish a fake detection over MQTT
├── test_pipeline.py                    ← Run full pipeline on a WAV file with MQTT
├── test_audio.py                       ← Generate synthetic spike/sine/sawtooth WAVs
├── simulate_live.py                    ← Stream a WAV at real-time speed (mic substitute)
└── check_db.py                         ← Inspect host.db schema and contents
```

---

## Prerequisites

### Host Machine
- Python 3.10 or 3.11 (64-bit)
- Docker Desktop (for the web dashboard)
- Mosquitto MQTT broker

### RPi5 Nodes
- Raspberry Pi 5 with MEMS microphone
- Python 3.10+ with all dependencies (see below)
- Network access to the host machine

---

## Setup — Host Machine

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/ImpulsiveSoundDetection.git
cd ImpulsiveSoundDetection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install tensorflow tensorflow-hub librosa audiomentations \
            sounddevice soundfile customtkinter matplotlib \
            numpy scipy scikit-learn rich paho-mqtt pillow
```

### 4. Install and start Mosquitto

**Windows:**
```powershell
winget install EclipseFoundation.Mosquitto
# Run as Administrator:
net start mosquitto
# Or run directly:
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

**Linux/macOS:**
```bash
sudo apt install mosquitto mosquitto-clients   # Debian/Ubuntu
brew install mosquitto                          # macOS
sudo systemctl start mosquitto
```

### 5. Create the logs directory

```bash
mkdir -p logs
```

### 6. Configure the dashboard environment

```bash
cd impulsive_sound_detection/dashboard_server
cp .env.example .env
```

Edit `.env`:
```
ISD_DB_PATH=C:\Github\ImpulsiveSoundDetection\logs\host.db
ADMIN_API_KEY=your-secret-key-here
PORT=3000
```

Generate a secure key:
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### 7. Update config.py for your machine

Edit `impulsive_sound_detection/config.py`:
```python
ISD_ROOT          = Path(r"C:\ImpulsiveSoundDetection")   # your data root
MQTT_BROKER_HOST  = "127.0.0.1"                           # or host LAN IP
NODE_ID           = "node_1"                               # unique per device
NODE_LOCATION     = "Hallway A"
```

---

## Running the System

### Terminal 1 — MQTT Broker
```bash
# If not running as a service:
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

### Terminal 2 — Host Subscriber (writes host.db)
```bash
cd ImpulsiveSoundDetection
python -m impulsive_sound_detection.host_subscriber --broker-host 127.0.0.1
```
Expected output:
```
Host database ready at C:\Github\ImpulsiveSoundDetection\logs\host.db
HostSubscriber running – waiting for node data …
```

### Terminal 3 — Web Dashboard (Docker)
```bash
cd impulsive_sound_detection/dashboard_server
docker compose up --build
```
Open **http://localhost:3000**

### Terminal 4 — Node Simulation (dev/test)
```bash
# Generate test audio
python test_audio.py

# Simulate a live node stream
python simulate_live.py --wav sine.wav --threshold-multiplier 1.5 \
    --mqtt --broker-host 127.0.0.1 --node-id node_sim --loop
```

### RPi5 Node — Live Microphone
```bash
python -m impulsive_sound_detection.main live \
    --mqtt --broker-host 192.168.1.100 \
    --node-id node_1 \
    --threshold-multiplier 2.0
```

---

## Database Schema

**`detection_events`** — every audio detection event

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| event_uuid | TEXT | UUID per detection |
| node_id | TEXT | Source RPi node |
| label | TEXT | YAMNet label (e.g. "Gunshot, gunfire") |
| confidence | REAL | 0.0–1.0 classifier confidence |
| is_suspicious | INTEGER | 1 if label matches suspicious set |
| severity | TEXT | LOW / MEDIUM / HIGH |
| timestamp_node | REAL | Unix time (stream-relative) |
| wall_clock_time | REAL | Unix time (actual wall clock at trigger) |
| received_at_host | REAL | Unix time when host subscriber received it |
| onset_index | INTEGER | Sample index of the trigger onset |
| session_id | TEXT | Session identifier |
| inserted_at | TEXT | SQLite insert timestamp |

**`node_status`** — one row per registered RPi node

| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT PK | Unique node identifier |
| location | TEXT | Human-readable location |
| status | TEXT | online / offline |
| last_seen | REAL | Unix time of last heartbeat |
| enabled | INTEGER | 1 = active, 0 = decommissioned |
| notes | TEXT | Admin notes |

**`rms_frames`** — throttled RMS energy samples (≈5/sec per node)

| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT | Source node |
| ts | REAL | Unix timestamp |
| rms | REAL | Frame RMS energy |
| baseline | REAL | Rolling 10s baseline |
| threshold | REAL | Dynamic trigger threshold |
| is_trigger | INTEGER | 1 if this frame caused a Stage 1 trigger |

**`localization_results`** — Sound Localization team output (stub)

| Column | Type | Description |
|--------|------|-------------|
| received_at | REAL | Unix timestamp |
| payload_json | TEXT | Raw JSON from localization module |

---

## REST API Reference

### Public Routes (no authentication required)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/events` | Latest detection events. `?limit=N&node=node_id` |
| GET | `/api/stats` | Summary counters (total, suspicious, nodes online, avg confidence) |
| GET | `/api/nodes` | Node list with per-node stats |
| GET | `/api/rms` | Recent RMS frames for chart. `?node=node_id` |
| GET | `/api/correlated` | Cross-node events within 2s (TDOA candidates) |
| GET | `/api/localization` | Latest localization result (stub) |
| GET | `/api/status` | DB mode (live/demo) and path |
| GET | `/api/history` | Paginated, filterable event history. See params below |
| POST | `/api/auth/verify` | Verify admin key. Body: `{"key":"..."}` |

**`/api/history` query parameters:**

| Param | Example | Description |
|-------|---------|-------------|
| page | `?page=2` | Page number (default 1) |
| per | `?per=50` | Results per page (max 200, default 50) |
| node | `?node=node_1` | Filter by node ID |
| label | `?label=Gunshot` | Filter by label substring |
| severity | `?severity=HIGH` | Filter by severity (HIGH/MEDIUM/LOW) |
| susp | `?susp=1` | Suspicious events only |
| from | `?from=2026-05-01 00:00:00` | From date (inserted_at) |
| to | `?to=2026-05-07 23:59:59` | To date (inserted_at) |

### Admin Routes (require `X-Admin-Key` header)

All admin routes require the HTTP header:
```
X-Admin-Key: <your ADMIN_API_KEY>
```

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/admin/query` | Run a read-only SELECT query. Body: `{"sql":"..."}` |
| GET | `/api/admin/nodes` | All nodes with admin fields (enabled, notes) |
| GET | `/api/admin/nodes/discovered` | Nodes detected in events but not registered |
| POST | `/api/admin/nodes` | Register a new node. Body: `{"node_id","location","notes"}` |
| PUT | `/api/admin/nodes/:id` | Update node. Body: `{"location","enabled","notes"}` |
| DELETE | `/api/admin/nodes/:id` | Remove a node record (preserves events) |
| POST | `/api/admin/nodes/:id/ping` | Mark node last_seen = now |
| POST | `/api/admin/nodes/:id/clear` | Delete all events + RMS frames for a node |

---

## Authentication

Admin routes are protected by API key authentication. The key is set in `.env` as `ADMIN_API_KEY` and is never exposed in source code or committed to git.

**How it works:**
1. User clicks the **Maintenance** or **Query Lab** tab for the first time
2. Dashboard prompts for the admin key via a modal dialog
3. Key is sent to `POST /api/auth/verify` — server returns 200 or 401
4. On success, key is stored in `sessionStorage` (cleared when browser tab closes)
5. All subsequent admin API calls include `X-Admin-Key: <key>` in the request header
6. Server middleware validates the header on every `/api/admin/*` route

**Security notes:**
- The key is stored in `sessionStorage`, not `localStorage` — it does not persist across sessions
- HTTPS should be used in production deployment to prevent key interception
- The default key `isd-admin-changeme` triggers a server warning on startup
- This is API-key authentication, appropriate for an IoT infrastructure dashboard used by network administrators

---

## Deployment

### Local (Docker)

```bash
cd impulsive_sound_detection/dashboard_server
docker build -t isd-dashboard .
docker compose up
```

The `docker-compose.yml` mounts the host `logs/` directory into the container at `/data` so the server can read `host.db` written by `host_subscriber.py`.

### Environment Variables in Docker

Environment variables are injected via `docker-compose.yml` and do not come from the `.env` file inside the container (`.env` is excluded by `.dockerignore`):

```yaml
environment:
  - ISD_DB_PATH=/data/host.db
  - ADMIN_API_KEY=your-secret-key-here
  - PORT=3000
```

---

## Testing

### Generate test audio files
```bash
python test_audio.py
# Creates spike.wav, sine.wav, sawtooth.wav
```

### Run a complete end-to-end test

With Mosquitto running and `host_subscriber.py` active:

```bash
# Publish a fake detection over MQTT
python test_mqtt.py

# Verify it hit the database
python check_db.py

# Stream a WAV through the full pipeline
python simulate_live.py --wav sine.wav --threshold-multiplier 1.5 \
    --mqtt --broker-host 127.0.0.1 --node-id node_sim
```

### Test admin API authentication
```bash
# Should return 401
curl -X GET http://localhost:3000/api/admin/nodes

# Should return node list
curl -X GET http://localhost:3000/api/admin/nodes \
     -H "X-Admin-Key: your-secret-key-here"
```

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript (ES2022), Chart.js |
| Backend | Node.js 20, Express.js 4 |
| Database | SQLite (via sql.js in server, via sqlite3 in Python) |
| ML | TensorFlow 2, TensorFlow Hub (YAMNet), EfficientNetB0 |
| Audio | librosa, sounddevice, soundfile, audiomentations |
| Messaging | MQTT (Mosquitto broker, paho-mqtt client) |
| Deployment | Docker, docker-compose |
| Auth | API key (X-Admin-Key header, sessionStorage) |
| Version Control | Git / GitHub |

---

## Limitations and Future Work

- **Authentication** — API key auth is appropriate for a LAN IoT dashboard but does not support multiple users with individual passwords. A full JWT/session system would be needed for public deployment.
- **HTTPS** — currently served over HTTP. Production deployment should use a reverse proxy (nginx) with TLS.
- **Sound Localization** — the TDOA localization module is stubbed out pending integration with the Sound Localization team.
- **CNN domain gap** — the EfficientNetB0 classifier achieves 98.8% F1 on synthetic spectrograms but shows performance degradation on real-world audio. Fine-tuning on real gunshot recordings is the next ML milestone.
- **RPi microphone** — MEMS microphone input on Windows dev machine was not accessible through sounddevice; WAV simulation was used for local testing.

---

## License

Code in `impulsive_sound_detection/` is released under MIT. 
Dataset files carry their own licenses — see `Gunshot Audio Spectrogram Dataset for Binary Class/README.md`.