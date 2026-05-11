# Impulsive Sound Detection (ISD) System

**ENEB453 Web-Based Application Development — Spring 2026**
**Team:** Mathew Ridgely · Skott (Web App) · Charlie Holden (ML Pipeline)
**Instructor:** Dr. Nestor Michael C. Tiglao

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Initial Setup — Host Machine](#initial-setup--host-machine)
6. [Initial Setup — RPi5 Node](#initial-setup--rpi5-node)
7. [Running the System](#running-the-system)
8. [Dashboard Overview](#dashboard-overview)
9. [Database Schema](#database-schema)
10. [REST API Reference](#rest-api-reference)
11. [Authentication & Security](#authentication--security)
12. [Local Test Cases](#local-test-cases)
13. [Integrated Test Cases](#integrated-test-cases)
14. [Troubleshooting](#troubleshooting)
15. [Scalability — Adding New Nodes](#scalability--adding-new-nodes)
16. [Scalability — Admin Account Management](#scalability--admin-account-management)
17. [Limitations & Future Work](#limitations--future-work)
18. [Technologies Used](#technologies-used)
19. [AI Tool Disclosure](#ai-tool-disclosure)

---

## Overview

The ISD system detects gunshots and glass breaks in real time using a network of Raspberry Pi 5 compute modules equipped with MEMS microphones. Each node runs a two-stage ML pipeline locally and publishes results over MQTT to a central host machine running a Node.js/Express web dashboard.

**Key capabilities:**
- Real-time acoustic event detection with sub-second latency from trigger to dashboard alert
- Multi-node coverage with per-node status monitoring and correlated event detection
- Server-Sent Events (SSE) push — suspicious detections appear on the dashboard instantly without polling
- Maintenance Mode for adding, renaming, diagnosing, and removing nodes at any time
- Bcrypt-authenticated admin routes with HTTPS transport

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HOST MACHINE                               │
│                                                                     │
│  ┌───────────────────┐     ┌──────────────────────────────────┐    │
│  │  host_subscriber  │     │   dashboard_server  (Docker)     │    │
│  │  (Python)         │────▶│   Node.js + Express              │    │
│  │  writes host.db   │     │   sql.js reads host.db           │    │
│  │  notifies SSE     │     │   https://localhost:3443         │    │
│  └───────────────────┘     └──────────────────────────────────┘    │
│           ▲                                                         │
│           │  MQTT subscribe (all topics)                            │
│  ┌────────┴──────────┐                                             │
│  │  Mosquitto Broker │                                             │
│  │  port 1883        │                                             │
│  └────────┬──────────┘                                             │
└───────────┼─────────────────────────────────────────────────────────┘
            │  MQTT publish over LAN
  ┌─────────┴────────────────────────────────────────────┐
  │                  RPi5 NODE NETWORK                    │
  │                                                       │
  │  ┌───────────┐   ┌───────────┐   ┌───────────┐      │
  │  │  node_1   │   │  node_2   │   │  node_N   │      │
  │  │  RPi5     │   │  RPi5     │   │  RPi5     │      │
  │  │  MEMS Mic │   │  MEMS Mic │   │  MEMS Mic │      │
  │  │  YAMNet   │   │  YAMNet   │   │  YAMNet   │      │
  │  └───────────┘   └───────────┘   └───────────┘      │
  └───────────────────────────────────────────────────────┘
```

### ML Pipeline (per node)

```
Microphone → RMS Frames (512 samples)
  → Stage 1: Rolling baseline comparison
    → Trigger (energy > N × baseline)
      → Stage 2: YAMNet / EfficientNetB0 CNN
        → ClassificationResult
          → MQTT publish → host_subscriber → host.db → SSE push → Dashboard
```

---

## Repository Structure

```
ImpulsiveSoundDetection/
├── impulsive_sound_detection/
│   ├── config.py                       ← All constants — edit per device
│   ├── pipeline.py                     ← Stage 1 + 2 orchestration
│   ├── stream_monitor.py               ← Stage 1: RMS energy trigger
│   ├── classifier.py                   ← Stage 2: YAMNet + CNN
│   ├── mqtt_bridge.py                  ← Node-side MQTT publisher
│   ├── host_subscriber.py              ← Host-side subscriber → host.db + SSE
│   ├── event_logger.py                 ← SQLite + JSONL logging
│   ├── live_stream.py                  ← Live microphone capture
│   ├── main.py                         ← CLI entry point
│   ├── spectrogram_utils.py            ← FFT / LogMel / MFCC rendering
│   ├── visualizer.py                   ← Detection plots
│   ├── dashboard.py                    ← Terminal dashboard
│   ├── gui.py                          ← GUI dashboard (customtkinter)
│   └── dashboard_server/
│       ├── index.js                    ← Express server + REST API + SSE
│       ├── public/index.html           ← Single-page dashboard (4 tabs)
│       ├── package.json
│       ├── Dockerfile
│       ├── docker-compose.yml          ← Edit: hash + cert paths
│       ├── generate-cert.ps1           ← Run once to create TLS certs
│       └── .dockerignore
├── train/                              ← EfficientNetB0 training scripts
├── models/                             ← Trained .keras models (not in git)
├── reports/                            ← Confusion matrices, ROC/PR curves
├── logs/                               ← host.db + JSONL logs (not in git)
├── test_mqtt.py                        ← Publish a fake MQTT detection
├── test_pipeline.py                    ← Run pipeline on WAV + publish
├── test_audio.py                       ← Generate synthetic WAV files
├── test_simulate_live.py               ← Stream WAV at real-time speed
├── check_db.py                         ← Inspect host.db contents
└── README.md
```

---

## Prerequisites

### Host Machine (Windows / Linux / macOS)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 or 3.11 | 64-bit |
| Node.js | 18+ | For bcrypt hash generation |
| Docker Desktop | Latest | For the web dashboard |
| Git | Any | OpenSSL bundled in Git for Windows |
| Mosquitto | 2.x | MQTT broker |

### RPi5 Nodes

| Requirement | Notes |
|-------------|-------|
| Raspberry Pi 5 | Any RAM config |
| MEMS microphone | I2S or USB interface |
| Python 3.10+ | Pre-installed on Pi OS |
| LAN access to host | Same network as host machine |

---

## Initial Setup — Host Machine

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/ImpulsiveSoundDetection.git
cd ImpulsiveSoundDetection
```

### 2. Create Python virtual environment

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

# Start as a service (run PowerShell as Administrator)
net start mosquitto

# Or run directly in a terminal
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

**Linux / macOS:**
```bash
sudo apt install mosquitto mosquitto-clients   # Debian/Ubuntu
brew install mosquitto                          # macOS
sudo systemctl enable --now mosquitto
```

**Allow remote connections** (required for RPi nodes on the LAN).
Create or edit `mosquitto.conf` (usually at `C:\Program Files\mosquitto\mosquitto.conf`):

```
listener 1883
allow_anonymous true
```

Restart after editing: `net stop mosquitto && net start mosquitto`

### 5. Create the logs directory

```bash
mkdir -p logs
```

### 6. Configure the dashboard

```powershell
cd impulsive_sound_detection\dashboard_server
```

Create the `.env` file from the template:

```powershell
Copy-Item .env.example .env
```

You will fill in the `ADMIN_PASSWORD_HASH` value in the next step.

#### 6a. Install Node dependencies and generate bcrypt hash

```powershell
npm install

# Generate a bcrypt hash of your chosen admin password
node -e "require('bcrypt').hash('your-password-here', 12).then(console.log)"
# Shorthand via package.json script:
# npm run hash your-password-here
```

This takes a few seconds. You will get a 60-character string starting with `$2b$12$`.

**⚠ `$` escaping required:** Docker Compose interpolates `$` signs in `.env` values, silently corrupting a bcrypt hash. Every `$` in the hash must be doubled to `$$` when you write it to `.env`. For example, the hash `$2b$12$ABC...` becomes `$$2b$$12$$ABC...` in `.env`.

The three `$` signs appear at the very start of the hash (`$2b$12$`). No `$` signs appear in the rest of the hash.

```powershell
# In .env — replace every leading $ with $$:
ADMIN_PASSWORD_HASH=$$2b$$12$$<rest of your hash>
```

#### 6b. Generate TLS certificate for HTTPS

```powershell
New-Item -ItemType Directory -Force -Path .\certs

& "C:\Program Files\Git\usr\bin\openssl.exe" req -x509 -nodes -newkey rsa:2048 `
  -keyout .\certs\server.key `
  -out    .\certs\server.crt `
  -days   365 `
  -subj   "/C=US/ST=MD/L=College Park/O=ISD System/CN=localhost" `
  -addext "subjectAltName=IP:127.0.0.1,IP:192.168.1.100,DNS:localhost"
```

Replace `192.168.1.100` with your actual host LAN IP (`ipconfig` to find it).

#### 6c. Edit docker-compose.yml

> **Critical:** Update the left side of the `logs` volume mount to match the **absolute Windows path** where you cloned the repo. Find your path with `(Get-Location).Path` or `pwd`. The right side (`/data`) must stay as-is.

```yaml
services:
  isd-dashboard:
    build: .
    container_name: isd-dashboard
    ports:
      - "3000:3000"
      - "3443:3443"
    volumes:
      - C:\your\actual\path\to\ImpulsiveSoundDetection\logs:/data   # ← update this
      - .\certs:/certs:ro
    environment:
      - ISD_DB_PATH=/data/host.db
      - ADMIN_PASSWORD_HASH=$2b$12$...paste your hash here...
      - PORT=3000
      - HTTPS_PORT=3443
      - CERT_PATH=/certs/server.crt
      - KEY_PATH=/certs/server.key
    restart: unless-stopped
```

#### 6d. Confirm config.py (no edit required for basic setup)

`ISD_ROOT` now defaults to the repo root automatically (resolved relative to `config.py`), so no manual path edit is needed for a standard clone. The value can still be overridden with the `ISD_ROOT` environment variable if needed.

Confirm `MQTT_BROKER_HOST` matches your intended broker:

```python
MQTT_BROKER_HOST = "127.0.0.1"   # change to host LAN IP when RPi nodes are active
```

If you have a non-standard layout or want to override, set the env var before running:

```powershell
$env:ISD_ROOT = "C:\your\path\to\ImpulsiveSoundDetection"
```

#### 6e. Add to .gitignore

```gitignore
impulsive_sound_detection/dashboard_server/certs/
impulsive_sound_detection/dashboard_server/.env
impulsive_sound_detection/dashboard_server/node_modules/
logs/host.db
logs/*.jsonl
models/**/*.keras
models/**/*.tflite
```

---

## Initial Setup — RPi5 Node

Repeat for each physical RPi. Each node **must** have a unique `NODE_ID`.

### 1. Install dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

cd ~
git clone https://github.com/<your-org>/ImpulsiveSoundDetection.git
cd ImpulsiveSoundDetection

python3 -m venv venv
source venv/bin/activate

pip install tensorflow tensorflow-hub librosa sounddevice soundfile \
            numpy scipy scikit-learn paho-mqtt rich
```

### 2. Configure the node

Edit `impulsive_sound_detection/config.py` on **this device only**:

```python
# Section 10 — NODE IDENTITY
NODE_ID       = "node_1"           # unique across ALL nodes — no duplicates
NODE_LOCATION = "Hallway A"        # human-readable location for the dashboard

# Section 11 — MQTT
MQTT_BROKER_HOST = "192.168.1.100" # LAN IP of the host machine running Mosquitto
```

> **Critical:** If two nodes share the same `NODE_ID`, their data will collide
> in the database and the dashboard will show incorrect event counts and locations.

### 3. Test microphone access

```bash
python -m impulsive_sound_detection.main live --threshold-multiplier 3.0
```

You should see the terminal dashboard updating with RMS values. Press Ctrl+C to stop.

### 4. Run as a systemd service (auto-start on boot)

Create `/etc/systemd/system/isd-node.service`:

```ini
[Unit]
Description=ISD Node Detection Pipeline
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/ImpulsiveSoundDetection
ExecStart=/home/pi/ImpulsiveSoundDetection/venv/bin/python \
          -m impulsive_sound_detection.main live \
          --mqtt --broker-host 192.168.1.100 \
          --node-id node_1 \
          --threshold-multiplier 2.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable isd-node
sudo systemctl start isd-node
sudo systemctl status isd-node
```

---

## Running the System

Start these **in order** every time. Docker Desktop must already be running.

**Terminal 1 — MQTT Broker**

If running Mosquitto as a Windows service (the default after `winget install`), it is already running. Skip this step or verify with:
```powershell
sc.exe query mosquitto    # STATE should be RUNNING
```
To run manually instead:
```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

**Terminal 2 — Host Subscriber**
```powershell
cd C:\path\to\ImpulsiveSoundDetection   # ← your actual repo path
python -m impulsive_sound_detection.host_subscriber `
    --broker-host 127.0.0.1 `
    --dashboard-url https://localhost:3443
```

> **Wait for startup:** TensorFlow loads on first run (~15 seconds). Do **not** start Terminal 3 until you see the "HostSubscriber running" message below — starting Docker before `host.db` is created puts the dashboard in DEMO mode.

Expected output (after ~15 s):
```
Host database ready at C:\path\to\ImpulsiveSoundDetection\logs\host.db
HostSubscriber running – waiting for node data …
```

> **Note:** You may see TensorFlow deprecation warnings and a paho-mqtt `Callback API version 1 is deprecated` warning on startup. These are expected and non-fatal.

**Terminal 3 — Web Dashboard**
```powershell
cd impulsive_sound_detection\dashboard_server
docker compose up --build
```

Expected output:
```
[DB] Live mode – loaded from /data/host.db
ISD Dashboard → https://localhost:3443  (HTTPS)
Database mode : LIVE
Reading from  : /data/host.db
HTTP redirect → http://localhost:3000  (redirects to HTTPS)
```

If you see `Database mode : DEMO` instead, `host.db` was not found when the container started — restart Docker after Terminal 2 finishes loading:
```powershell
docker compose down && docker compose up
```

Open **https://localhost:3443**. On first visit click **Advanced → Proceed to localhost** to accept the self-signed certificate.

**RPi5 Nodes** (or simulation on host):
```bash
# On each RPi
python -m impulsive_sound_detection.main live \
    --mqtt --broker-host 192.168.1.100 --node-id node_1

# Or simulate locally for testing
python test_simulate_live.py --wav sine.wav --threshold-multiplier 1.5 \
    --mqtt --broker-host 127.0.0.1 --node-id node_sim --loop
```

---

## Dashboard Overview

| Tab | Purpose |
|-----|---------|
| **Live Feed** | Real-time stats, auto-scaling RMS chart, gauge, event feed, node summary, correlated events |
| **History** | Filterable, sortable, paginated event log with CSV export |
| **Query Lab** | Read-only SQL editor with preset queries and CSV export |
| **Maintenance** | Node discovery, registration, editing, diagnostics (requires admin login) |

**Admin login:** When you first open the Maintenance or Query Lab tab you will be prompted for the admin password. Enter the plaintext password you hashed during setup. It is verified via bcrypt server-side — never stored. The session persists in `sessionStorage` until the browser tab is closed.

---

## Database Schema

**`detection_events`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| event_uuid | TEXT | UUID per detection |
| node_id | TEXT | Source RPi node |
| label | TEXT | Classifier label (e.g. "Gunshot, gunfire") |
| confidence | REAL | 0.0–1.0 |
| is_suspicious | INTEGER | 1 if label matches SUSPICIOUS_LABELS |
| severity | TEXT | LOW / MEDIUM / HIGH |
| timestamp_node | REAL | Unix time (stream-relative) |
| wall_clock_time | REAL | Unix time at trigger |
| received_at_host | REAL | Unix time when host received it |
| onset_index | INTEGER | Sample index of trigger |
| session_id | TEXT | Session identifier |
| inserted_at | TEXT | SQLite insert timestamp |

**`node_status`**

| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT PK | Unique node identifier |
| location | TEXT | Human-readable location |
| status | TEXT | online / offline |
| last_seen | REAL | Unix time of last heartbeat |
| enabled | INTEGER | 1 = active, 0 = decommissioned |
| notes | TEXT | Admin notes |

**`rms_frames`**

| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT | Source node |
| ts | REAL | Unix timestamp |
| rms | REAL | Frame RMS energy |
| baseline | REAL | Rolling 10s baseline |
| threshold | REAL | Dynamic trigger threshold |
| is_trigger | INTEGER | 1 if this frame triggered Stage 1 |

**`localization_results`** — stub, pending Sound Localization team integration

| Column | Type | Description |
|--------|------|-------------|
| received_at | REAL | Unix timestamp |
| payload_json | TEXT | Raw JSON payload |

---

## REST API Reference

### Public Routes (no auth)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/events` | Latest events. `?limit=N&node=id` |
| GET | `/api/stats` | Summary counters |
| GET | `/api/nodes` | Node list with stats |
| GET | `/api/rms` | RMS frames. `?node=id` |
| GET | `/api/correlated` | Cross-node events within 2s |
| GET | `/api/localization` | Latest localization result |
| GET | `/api/status` | DB mode and path |
| GET | `/api/events/stream` | SSE persistent push connection |
| GET | `/api/history` | Paginated event history (see params below) |
| POST | `/api/auth/verify` | Verify password. Body: `{"key":"..."}` |

**`/api/history` params:** `page`, `per`, `sort` (id/node_id/label/confidence/severity/is_suspicious/inserted_at), `dir` (ASC/DESC), `node`, `label`, `severity`, `susp=1`, `from`, `to`

### Admin Routes (require `X-Admin-Key: <password>` header)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/admin/query` | Read-only SQL. Body: `{"sql":"SELECT ..."}` |
| GET | `/api/admin/nodes` | All nodes with admin fields |
| GET | `/api/admin/nodes/discovered` | Nodes seen in events but not registered |
| POST | `/api/admin/nodes` | Register node. Body: `{"node_id","location","notes"}` |
| PUT | `/api/admin/nodes/:id` | Update node. Body: `{"location","enabled","notes"}` |
| DELETE | `/api/admin/nodes/:id` | Remove node record |
| POST | `/api/admin/nodes/:id/ping` | Update last_seen to now |
| POST | `/api/admin/nodes/:id/clear` | Delete all events + RMS for a node |

---

## Authentication & Security

**Bcrypt password hashing:**
- Admin password is hashed with bcrypt cost factor 12 (~250ms per verify)
- Only the hash is stored in `ADMIN_PASSWORD_HASH` — the plaintext password is never persisted anywhere
- `bcrypt.compare()` runs on every admin request server-side
- The ~250ms delay is intentional — it defeats brute-force attacks even if the hash is exposed

**HTTPS:**
- Self-signed TLS certificate covers localhost and LAN IP
- All traffic on port 3000 is redirected to HTTPS on port 3443 via `301`
- The `X-Admin-Key` header (containing the password) is encrypted in transit

**Query safety:**
- Query Lab only permits `SELECT` / `WITH` — write keywords blocked by word-boundary regex
- `ORDER BY` columns validated against a hardcoded whitelist
- Filter values escaped to prevent SQL injection

---

## Local Test Cases

Run these on the host machine only. Terminals 2 and 3 must be running.

### LT-1 · Demo mode fallback

```powershell
Rename-Item logs\host.db logs\host.db.bak
docker rm -f isd-dashboard && docker compose up --build
```

Dashboard should show **DEMO MODE** badge and simulated events every 4 seconds. Restore when done:
```powershell
Rename-Item logs\host.db.bak logs\host.db
docker rm -f isd-dashboard && docker compose up
```

---

### LT-2 · Synthetic audio pipeline test

```powershell
python test_audio.py                                        # generates spike/sine/sawtooth WAVs
python -m impulsive_sound_detection.main detect --wav spike.wav --no-viz
python -m impulsive_sound_detection.main detect --wav sine.wav --no-viz
```

Expected: `0 detections` for both files. The test WAVs are 5 seconds long and `STREAM_WARMUP_SEC = 5.0` in config.py, so Stage 1 never exits warmup before the file ends. This validates that the pipeline loads and runs without crashing. Use LT-4 (`test_simulate_live.py`) for a triggered detection test — it loops the file, allowing the baseline to stabilise across multiple passes.

---

### LT-3 · MQTT publish → database write

```powershell
python test_mqtt.py     # publishes a fake suspicious detection
python check_db.py      # should show new row with is_suspicious=1
```

---

### LT-4 · Simulate live stream

```powershell
python test_simulate_live.py --wav spike.wav --threshold-multiplier 0.5 `
    --mqtt --broker-host 127.0.0.1 --node-id node_sim --loop
```

> **`--loop` is required** for detections. The test WAVs are 5 s long and `STREAM_WARMUP_SEC = 5.0`, so Stage 1 never fires on the first pass. On loop 2+ the spike occurs after warmup and triggers YAMNet. Threshold `0.5` is needed because the baseline adapts to the spike level after the first loop. Use `sine.wav` for a smoother signal.

Expected: YAMNet classifies the spike window (likely `Outside`, `Rustling leaves`, or similar — the synthetic spike is not a real gunshot). The key check is that events appear in the database and on the Live Feed, not the specific label.

Watch the Live Feed — events appear for `node_sim`. RMS chart Y-axis scales dynamically.

---

### LT-5 · Authentication

```powershell
# 401 — no key
curl -k https://localhost:3443/api/admin/nodes

# 401 — wrong password
curl -k https://localhost:3443/api/admin/nodes -H "X-Admin-Key: wrongpassword"

# 200 — correct password
curl -k https://localhost:3443/api/admin/nodes -H "X-Admin-Key: your-actual-password"

# Verify endpoint
curl -k -X POST https://localhost:3443/api/auth/verify `
     -H "Content-Type: application/json" -d '{"key":"your-actual-password"}'
# Expected: {"ok":true}
```

---

### LT-6 · HTTPS redirect

```powershell
curl -v http://localhost:3000/
# Look for: Location: https://localhost:3443/
```

---

### LT-7 · Query Lab keyword blocking

| Query | Expected result |
|-------|----------------|
| `SELECT * FROM detection_events LIMIT 5` | Returns rows |
| `SELECT inserted_at FROM detection_events LIMIT 5` | Returns rows (inserted_at is not blocked) |
| `DELETE FROM detection_events` | Blocked: "Only SELECT / WITH queries are permitted" |
| `DROP TABLE node_status` | Blocked |
| `WITH x AS (SELECT 1) SELECT * FROM x` | Returns 1 row |
| `SELECT * FROM nonexistent_table` | SQLite error shown, no crash |

---

### LT-8 · History sort and filter

1. Click **CONF** header → sorts DESC (highest first), arrow shows ↓
2. Click **CONF** again → flips to ASC, arrow shows ↑
3. Click **SEVERITY** → sorts HIGH → MEDIUM → LOW (logical order, not alphabetical)
4. Apply **SUSPICIOUS ONLY** filter + **CONF DESC** sort simultaneously
5. Click **NEXT** page → sort preserved
6. Click **CSV ↓** → downloaded file matches current sort + filter
7. Click **CLEAR** → filters and sort reset to ID DESC

---

### LT-9 · Node registration edge cases

In Maintenance tab (after login):

| Action | Expected |
|--------|----------|
| Add node with blank node_id | "Node ID is required" toast |
| Add node with id that already exists | "Node already exists" error toast |
| Remove a node, restart Docker | Events still exist in History (only registration removed) |
| Disable a node via edit modal | Node appears with reduced opacity in sidebar |

---

### LT-10 · SSE connection

1. Open DevTools → Network → filter by `stream`
2. `GET /api/events/stream` should show as **pending** (persistent connection)
3. Run `python test_mqtt.py` → nav badge flashes **⚠ ALERT** within 1 second
4. Kill Terminal 2 (host_subscriber) → SSE disconnects, polling continues every 4s
5. Restart Terminal 2 → SSE reconnects automatically (exponential backoff)

---

## Integrated Test Cases

Run with at least one RPi5 node active on the LAN.

### IT-1 · End-to-end detection

1. Start broker, host_subscriber, Docker, and RPi node pipeline
2. Make a loud sudden noise near the RPi microphone
3. Within ≤1 second: dashboard nav badge flashes **⚠ ALERT**
4. Event appears at top of Live Feed immediately (SSE push, not 4s poll)
5. Terminal 2 logs the detection with correct node_id, label, severity

---

### IT-2 · Multi-node correlated event

1. Two RPi nodes running (node_1, node_2)
2. Make a noise loud enough for both to detect
3. Both trigger within 2 seconds of each other
4. Correlated Events panel shows both node IDs with Δt ≤ 2.0s

---

### IT-3 · New node auto-discovery

1. Configure a new RPi with a node_id not yet registered (e.g. `node_5`)
2. Start its pipeline — it publishes one detection
3. Open Maintenance tab → Node Discovery → click ↺ REFRESH
4. `node_5` appears with event count and last seen time
5. Click **REGISTER** → node moves to Registered Nodes list
6. Restart Docker → node_5 still registered (persisted to host.db)

---

### IT-4 · Node offline watchdog

1. RPi running and showing **● ON** in sidebar
2. Kill the RPi pipeline (`Ctrl+C` or `sudo systemctl stop isd-node`)
3. Wait ~60 seconds
4. Sidebar updates to **○ OFF** for that node

---

### IT-5 · Multiple concurrent browser tabs (SSE)

1. Open dashboard in 3 browser tabs
2. Docker logs show `[SSE] Client connected (total: 3)`
3. Run `python test_mqtt.py` → all 3 tabs flash alert simultaneously
4. Close one tab → Docker logs show `[SSE] Client disconnected (total: 2)`

---

## Troubleshooting

### Docker Desktop must be running before `docker compose up`

If you get `failed to connect to the docker API`, Docker Desktop is not started.
Launch it from the Start Menu or via:
```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```
Wait ~30 seconds for the whale icon to appear in the system tray, then retry.

---

### Port 3000 or 3443 already in use

```powershell
netstat -ano | findstr ":3000 "
# Note the PID in the last column, then:
docker ps    # check if another container is using it
```
Stop the conflicting container:
```powershell
docker stop <container-name>
```
Then re-run `docker compose down && docker compose up`.

---

### Dashboard shows DEMO MODE instead of LIVE DATA

`host.db` is not found at the path in `ISD_DB_PATH`, or Docker started before `host_subscriber.py` created it.

Most likely cause: Terminal 3 (Docker) was started before Terminal 2 finished loading (~15 s for TensorFlow). Fix: wait for the "HostSubscriber running" message, then:
```powershell
docker compose down && docker compose up
```

Also verify the left side of the volume mount in `docker-compose.yml` matches the actual Windows path to your `logs/` folder:
```powershell
# The path on the left must match where host.db is created
Test-Path C:\your\path\to\ImpulsiveSoundDetection\logs\host.db
```

---

### Docker shows HTTP only — no HTTPS

Cert files not mounted or env vars missing.

```powershell
dir .\certs\                              # must show server.key and server.crt
Select-String "CERT_PATH" .\docker-compose.yml    # must be uncommented

docker rm -f isd-dashboard && docker compose up --build
```

---

### Admin login always fails

The most common cause: the bcrypt hash in `.env` is being silently corrupted by Docker Compose `$` interpolation.

Docker Compose treats `$` in values as a variable substitution prefix, even in `.env` files. A bcrypt hash like `$2b$12$Abc...` has three `$` signs — Docker Compose replaces them with empty strings and the 60-character hash arrives in the container as a malformed 46-character string that will never match.

**Fix:** use `$$` for every `$` in the hash when writing to `.env`:

```powershell
# Generate hash
node -e "require('bcrypt').hash('your-password', 12).then(console.log)"
# Output: $2b$12$XYZ...

# Write to .env with $$ escaping for every $ sign
# Example output line: $2b$12$XYZ...
# In .env write: $$2b$$12$$XYZ...
```

Verify the hash length inside the container (must be exactly 60 chars):
```powershell
docker exec isd-dashboard sh -c 'echo "${#ADMIN_PASSWORD_HASH}"'
# Should print: 60
```

If it's less than 60, the hash is still being corrupted. Regenerate and re-escape.

Bcrypt is also case-sensitive — verify you type the exact same password you hashed.

---

### `ns.enabled` column error in Docker logs

Stale `host.db` from before the schema migration. Handled automatically on startup by `ensureAdminColumns()`. If it persists:

```powershell
# Nuclear option — delete and recreate
Remove-Item logs\host.db
# Restart host_subscriber — it recreates the schema automatically
```

---

### Nodes not appearing in sidebar or History dropdown

Usually caused by the column error above. Fix that first, then hard-refresh (`Ctrl+Shift+R`).

---

### MQTT events not reaching host_subscriber

```powershell
# Test the broker is reachable and passing messages
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -h 127.0.0.1 -t "isd/#" -v
# Then in another terminal:
& "C:\Program Files\mosquitto\mosquitto_pub.exe" -h 127.0.0.1 -t "isd/test" -m "hello"
# mosquitto_sub should print the message

# If nothing appears: broker is not running
net start mosquitto
```

---

### RPi cannot connect to broker

```bash
# From the RPi
ping 192.168.1.100                          # verify host is reachable
mosquitto_pub -h 192.168.1.100 -t test -m hello   # test MQTT
```

If ping works but MQTT doesn't: Mosquitto isn't accepting remote connections. Add `listener 1883` and `allow_anonymous true` to `mosquitto.conf` on the host then restart.

---

### bcrypt module not found

```powershell
cd impulsive_sound_detection\dashboard_server
npm install bcrypt --save
```

If compilation fails (node-gyp error), use the pure-JavaScript alternative:
```powershell
npm install bcryptjs --save
# Change require('bcrypt') to require('bcryptjs') in index.js — API is identical
```

---

### test_simulate_live.py doesn't trigger any detections

The signal level is too low for the threshold multiplier. Try:

```powershell
python test_simulate_live.py --wav spike.wav --threshold-multiplier 0.5 `
    --mqtt --broker-host 127.0.0.1 --node-id node_sim
```

Lower multiplier = more sensitive. Default is 3.0; `spike.wav` triggers reliably at 1.5.

---

## Scalability — Adding New Nodes

The system requires **no changes** to the host, broker, or dashboard code when new nodes are added.

### Physical setup

1. Follow [Initial Setup — RPi5 Node](#initial-setup--rpi5-node) on the new device
2. Set a unique `NODE_ID` in `config.py`
3. Set `MQTT_BROKER_HOST` to the host's LAN IP
4. Start the pipeline

### Dashboard registration

Once the node publishes its first event, it auto-appears in **Maintenance → Node Discovery**. Click **REGISTER** to add a location and notes. It immediately appears in the sidebar, filter dropdowns, and diagnostics selector.

**Pre-register before the RPi arrives:**

1. Open Maintenance → Add Node Manually
2. Enter the `NODE_ID` exactly as it will appear in `config.py`
3. Set location and any notes
4. Click **+ REGISTER NODE**

The dashboard will already have the record when the RPi first comes online.

### Broker scaling

For 10+ nodes add to `mosquitto.conf`:

```
max_connections 500
max_queued_messages 1000
persistence true
persistence_location /var/lib/mosquitto/
```

### Database scaling

SQLite handles up to ~20 nodes at typical detection rates. For larger deployments, migrate to PostgreSQL — only `host_subscriber.py` and `index.js` need changing; the schema and all queries stay identical.

---

## Scalability — Admin Account Management

The system currently uses a **single shared admin password**, appropriate for a small team on a LAN.

### Changing the admin password

```powershell
cd impulsive_sound_detection\dashboard_server

node -e "require('bcrypt').hash('new-password', 12).then(console.log)"
# Paste new hash into docker-compose.yml → ADMIN_PASSWORD_HASH

docker rm -f isd-dashboard && docker compose up --build
```

All active admin sessions are invalidated immediately on rebuild — users will be re-prompted on their next admin action.

### Renewing the TLS certificate (annually)

Self-signed certs expire after 365 days. Renew with the same command:

```powershell
& "C:\Program Files\Git\usr\bin\openssl.exe" req -x509 -nodes -newkey rsa:2048 `
  -keyout .\certs\server.key `
  -out    .\certs\server.crt `
  -days   365 `
  -subj   "/C=US/ST=MD/L=College Park/O=ISD System/CN=localhost" `
  -addext "subjectAltName=IP:127.0.0.1,IP:192.168.1.100,DNS:localhost"

docker rm -f isd-dashboard && docker compose up --build
```

### Future: per-user admin accounts

The architecture supports upgrading to multiple named admins with minimal changes:

1. Add an `admin_users` table to `host.db`:
   ```sql
   CREATE TABLE admin_users (
     id         INTEGER PRIMARY KEY AUTOINCREMENT,
     username   TEXT UNIQUE NOT NULL,
     hash       TEXT NOT NULL,
     created_at TEXT DEFAULT (datetime('now'))
   );
   ```

2. Update `requireAdmin` to look up the hash by username:
   ```javascript
   const user = get('SELECT hash FROM admin_users WHERE username=?', [username]);
   const match = user && await bcrypt.compare(password, user.hash);
   ```

3. Add routes: `POST /api/admin/users` (create), `DELETE /api/admin/users/:name` (remove), `POST /api/admin/users/:name/password` (change)

4. Update the login modal to include a username field alongside the password field

5. Optionally add JWT tokens to cache authentication and avoid bcrypt on every request

---

## Limitations & Future Work

| Limitation | Impact | Path Forward |
|------------|--------|-------------|
| Single shared admin password | All admins use same credential | Per-user accounts (see above) |
| Self-signed TLS certificate | Browser warning on first visit | CA-signed cert via Let's Encrypt (requires public domain) |
| CNN domain gap | EfficientNetB0 F1 drops on real audio vs synthetic test set | Fine-tune on ReaLISED dataset |
| Sound Localization stub | TDOA panel shows no live data | Awaiting Sound Localization team MQTT output on `isd/localization/result` |
| sql.js write-back | `db.export()` briefly blocks Node.js event loop on admin writes | Migrate to `better-sqlite3` native binding |
| Windows microphone | sounddevice could not access mic on Windows dev machine | Use RPi5 hardware; WAV simulation used for all local testing |
| SQLite at scale | May slow under 20+ nodes at high publish rates | Migrate to PostgreSQL |

---

## Technologies Used

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | HTML5, CSS3, JavaScript ES2022 | — |
| Charts | Chart.js | 4.4.0 |
| Backend | Node.js + Express.js | 20 / 4.x |
| Database (server) | SQLite via sql.js | 1.12.x |
| Database (Python) | SQLite via stdlib sqlite3 | — |
| Authentication | bcrypt (cost factor 12) | 5.1.x |
| TLS | OpenSSL self-signed + Node.js https module | 3.x |
| ML | TensorFlow + TensorFlow Hub (YAMNet) | 2.x |
| ML (CNN) | EfficientNetB0 via Keras | 2.x |
| Audio | librosa, sounddevice, soundfile, audiomentations | — |
| Messaging | MQTT (Mosquitto broker, paho-mqtt client) | 2.x |
| Containerization | Docker + docker-compose | Latest |
| Real-time push | Server-Sent Events (SSE) | native |
| Version Control | Git / GitHub | — |

---

## AI Tool Disclosure

This project was developed with assistance from **Claude** (Anthropic) for code generation, architecture design, debugging, and documentation. All AI-generated code was reviewed, understood, tested, and integrated by the team. The team remains fully responsible for the correctness, design, and originality of the system.

Areas where AI assistance was used: Express.js server architecture, REST API design, SSE push implementation, bcrypt authentication middleware, HTTPS/TLS configuration, frontend tab navigation and Chart.js integration, SQL query construction and validation, and this README.