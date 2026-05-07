"""
host_subscriber.py – Host-side MQTT subscriber for the web dashboard.

Subscribes to all node topics and writes incoming data to the host
SQLite database at C:\\ImpulsiveSoundDetection\\host.db, which the
Express dashboard server reads.

Topic layout subscribed
-----------------------
  isd/node/+/detection   – ClassificationResult JSON from any node
  isd/node/+/rms         – Throttled RMS frame from any node
  isd/node/+/heartbeat   – Online/offline ping from any node
  isd/localization/result – Sound Localization team output (stub)

Schema
------
The host.db schema mirrors the teammate's detection_events table from
event_logger.py, extended with received_at_host and wall_clock_time
for TDOA localization support.

Usage
-----
    python -m impulsive_sound_detection.host_subscriber \\
        --broker-host 192.168.1.100
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False
    mqtt = None  # type: ignore[assignment]

from . import config
from .classifier import ClassificationResult

logger = logging.getLogger(__name__)

HOST_DB_PATH: Path = config.ISD_ROOT / "host.db"
NODE_TIMEOUT_SEC: float = 30.0


class HostSubscriber:
    """MQTT subscriber that feeds multi-node data to the host dashboard.

    Parameters
    ----------
    broker_host : str
        IP / hostname of the MQTT broker.
    broker_port : int
        TCP port of the broker.
    on_detection : callable | None
        Optional callback ``(result: ClassificationResult) -> None``
        called on every received detection.
    on_rms : callable | None
        Optional callback ``(node_id: str, payload: dict) -> None``.
    on_localization : callable | None
        Optional callback ``(payload: dict) -> None`` for the Sound
        Localization team's future output.
    db_path : Path | None
        Path to the SQLite database.  Defaults to HOST_DB_PATH.
    """

    def __init__(
        self,
        broker_host: str = config.MQTT_BROKER_HOST,
        broker_port: int = config.MQTT_BROKER_PORT,
        on_detection: Optional[Callable[[ClassificationResult], None]] = None,
        on_rms: Optional[Callable[[str, dict], None]] = None,
        on_localization: Optional[Callable[[dict], None]] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        if not _PAHO_AVAILABLE:
            raise ImportError(
                "paho-mqtt is required.\n"
                "Install it with:  pip install paho-mqtt"
            )

        self._broker_host = broker_host
        self._broker_port = broker_port
        self._on_detection = on_detection
        self._on_rms = on_rms
        self._on_localization = on_localization
        self._db_path = db_path or HOST_DB_PATH

        self._client: mqtt.Client = mqtt.Client(
            client_id="isd-host-subscriber",
            clean_session=True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        self._connected = threading.Event()
        self._stop = threading.Event()

        self._node_last_seen: Dict[str, float] = {}
        self._node_status: Dict[str, str] = {}
        self._node_lock = threading.Lock()

        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()

        self._watchdog_thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        """Connect to the broker, init the database, and start loops."""
        self._init_db()

        logger.info(
            "Connecting to MQTT broker %s:%d …",
            self._broker_host, self._broker_port,
        )
        self._client.connect_async(self._broker_host, self._broker_port, 60)
        self._client.loop_start()

        if not self._connected.wait(timeout=15.0):
            self._client.loop_stop()
            raise ConnectionError(
                f"Could not connect to broker "
                f"{self._broker_host}:{self._broker_port}"
            )

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="node-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        logger.info("HostSubscriber running – waiting for node data …")

    def stop(self) -> None:
        """Disconnect and shut down cleanly."""
        self._stop.set()
        self._client.disconnect()
        self._client.loop_stop()
        if self._db:
            self._db.close()
        logger.info("HostSubscriber stopped")

    def block_until_stopped(self) -> None:
        """Block the calling thread until stop() is called."""
        self._stop.wait()

    # ── database ──────────────────────────────────────────────────────
    def _init_db(self) -> None:
        """Create or connect to the host SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uuid          TEXT    NOT NULL,
                node_id             TEXT    NOT NULL,
                label               TEXT    NOT NULL,
                confidence          REAL    NOT NULL,
                is_suspicious       INTEGER NOT NULL,
                severity            TEXT    NOT NULL DEFAULT 'LOW',
                timestamp_node      REAL    NOT NULL,
                timestamp_iso       TEXT,
                wall_clock_time     REAL    NOT NULL,
                received_at_host    REAL    NOT NULL,
                onset_index         INTEGER NOT NULL,
                session_id          TEXT,
                classifier_version  TEXT    DEFAULT 'unknown',
                inserted_at         TEXT    DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS rms_frames (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id     TEXT  NOT NULL,
                ts          REAL  NOT NULL,
                rms         REAL  NOT NULL,
                baseline    REAL  NOT NULL,
                threshold   REAL  NOT NULL,
                is_trigger  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS node_status (
                node_id     TEXT PRIMARY KEY,
                location    TEXT,
                status      TEXT NOT NULL DEFAULT 'unknown',
                last_seen   REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS localization_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at     REAL NOT NULL,
                payload_json    TEXT NOT NULL
            );
        """)
        self._db.commit()
        logger.info("Host database ready at %s", self._db_path)

    def _db_insert_detection(self, result: ClassificationResult) -> None:
        with self._db_lock:
            self._db.execute("""
                INSERT INTO detection_events
                (event_uuid, node_id, label, confidence, is_suspicious,
                 severity, timestamp_node, timestamp_iso, wall_clock_time,
                 received_at_host, onset_index, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.event_uuid,
                result.node_id,
                result.label,
                result.confidence,
                int(result.is_suspicious),
                result.severity,
                result.timestamp,
                None,  # timestamp_iso computed on host if needed
                result.wall_clock_time,
                result.received_at_host,
                result.onset_index,
                result.session_id,
            ))
            self._db.commit()

    def _db_insert_rms(self, node_id: str, payload: dict) -> None:
        with self._db_lock:
            self._db.execute("""
                INSERT INTO rms_frames (node_id,ts,rms,baseline,threshold,is_trigger)
                VALUES (?,?,?,?,?,?)
            """, (
                node_id,
                payload.get("ts", 0),
                payload.get("rms", 0),
                payload.get("baseline", 0),
                payload.get("threshold", 0),
                int(payload.get("is_trigger", False)),
            ))
            self._db.commit()

    def _db_upsert_node(self, node_id: str, location: str, status: str, ts: float) -> None:
        with self._db_lock:
            self._db.execute("""
                INSERT INTO node_status (node_id, location, status, last_seen)
                VALUES (?,?,?,?)
                ON CONFLICT(node_id) DO UPDATE SET
                    location=excluded.location,
                    status=excluded.status,
                    last_seen=excluded.last_seen
            """, (node_id, location, status, ts))
            self._db.commit()

    # ── paho callbacks ────────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            logger.info("Host subscriber connected to broker")
            self._connected.set()
            client.subscribe("isd/node/+/detection",  qos=1)
            client.subscribe("isd/node/+/rms",        qos=0)
            client.subscribe("isd/node/+/heartbeat",  qos=1)
            client.subscribe(config.MQTT_TOPIC_LOCALIZATION, qos=1)
        else:
            logger.error("Broker connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()
        if rc != 0:
            logger.warning("Unexpected disconnect rc=%d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        topic: str = msg.topic
        try:
            payload_str = msg.payload.decode("utf-8")
        except Exception:
            logger.warning("Non-UTF-8 payload on %s – ignored", topic)
            return

        try:
            if topic.endswith("/detection"):
                self._handle_detection(payload_str)
            elif topic.endswith("/rms"):
                self._handle_rms(topic, payload_str)
            elif topic.endswith("/heartbeat"):
                self._handle_heartbeat(topic, payload_str)
            elif topic == config.MQTT_TOPIC_LOCALIZATION:
                self._handle_localization(payload_str)
        except Exception:
            logger.exception("Error handling message on %s", topic)

    # ── message handlers ──────────────────────────────────────────────
    def _handle_detection(self, payload_str: str) -> None:
        received_at = time.time()
        result = ClassificationResult.from_mqtt_payload(payload_str, received_at)
        logger.info(
            "[%s] Detection: %s conf=%.3f suspicious=%s severity=%s",
            result.node_id, result.label, result.confidence,
            result.is_suspicious, result.severity,
        )
        self._db_insert_detection(result)
        if self._on_detection:
            self._on_detection(result)

    def _handle_rms(self, topic: str, payload_str: str) -> None:
        data = json.loads(payload_str)
        node_id = data.get("node_id", topic.split("/")[2])
        self._db_insert_rms(node_id, data)
        if self._on_rms:
            self._on_rms(node_id, data)

    def _handle_heartbeat(self, topic: str, payload_str: str) -> None:
        data = json.loads(payload_str)
        node_id  = data.get("node_id", topic.split("/")[2])
        status   = data.get("status", "unknown")
        location = data.get("location", "")
        ts       = data.get("ts", time.time())

        with self._node_lock:
            self._node_last_seen[node_id] = time.time()
            self._node_status[node_id] = status

        self._db_upsert_node(node_id, location, status, ts)

    def _handle_localization(self, payload_str: str) -> None:
        """
        ── SOUND LOCALIZATION STUB ──────────────────────────────────────
        The Sound Localization team will publish to:
            isd/localization/result

        Expected payload contract (TBD by that team):
        {
            "ts":               <float>  UTC timestamp of the incident,
            "likely_node":      <str>    nearest node_id,
            "likely_location":  <str>    human-readable location,
            "tdoa_matrix": {             time-difference-of-arrival (sec)
                "<node_a>:<node_b>": <float>, …
            },
            "confidence":       <float>  0–1
        }

        Wire into the dashboard by passing an on_localization callback
        that pushes the payload to the Express server.
        """
        received_at = time.time()
        logger.info("Localization result received")
        with self._db_lock:
            self._db.execute(
                "INSERT INTO localization_results (received_at, payload_json)"
                " VALUES (?,?)",
                (received_at, payload_str),
            )
            self._db.commit()
        if self._on_localization:
            try:
                self._on_localization(json.loads(payload_str))
            except Exception:
                logger.exception("on_localization callback failed")

    # ── node watchdog ─────────────────────────────────────────────────
    def _watchdog_loop(self) -> None:
        """Mark nodes offline if they stop sending heartbeats."""
        while not self._stop.wait(timeout=10.0):
            now = time.time()
            with self._node_lock:
                for node_id, last_seen in list(self._node_last_seen.items()):
                    if now - last_seen > NODE_TIMEOUT_SEC:
                        if self._node_status.get(node_id) != "offline":
                            logger.warning(
                                "[%s] Node timed out – marking offline", node_id
                            )
                            self._node_status[node_id] = "offline"
                            self._db_upsert_node(node_id, "", "offline", now)


# ── Standalone entry point ────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ISD Host Subscriber – writes MQTT data to SQLite."
    )
    p.add_argument("--broker-host", default=config.MQTT_BROKER_HOST)
    p.add_argument("--broker-port", type=int, default=config.MQTT_BROKER_PORT)
    p.add_argument("--db-path", default=str(HOST_DB_PATH))
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    args = _build_parser().parse_args()
    sub = HostSubscriber(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        db_path=Path(args.db_path),
    )
    sub.start()
    try:
        sub.block_until_stopped()
    except KeyboardInterrupt:
        pass
    finally:
        sub.stop()
