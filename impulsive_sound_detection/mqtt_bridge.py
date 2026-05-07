"""
mqtt_bridge.py – Node-side MQTT publisher for the RPi5 compute modules.

Each RPi node instantiates one MQTTBridge and passes it to
DetectionPipeline.  The bridge handles:

  • Publishing ClassificationResult detections on every trigger.
  • Publishing lightweight RMS heartbeat frames (throttled).
  • Publishing periodic online/offline heartbeats.
  • Auto-reconnect with exponential back-off.

Topic layout
------------
  isd/node/<node_id>/detection  – ClassificationResult JSON (on trigger)
  isd/node/<node_id>/rms        – {"node_id","ts","rms","baseline",
                                    "threshold","is_trigger"} (throttled)
  isd/node/<node_id>/heartbeat  – {"node_id","ts","status","location"}
                                    (every HEARTBEAT_INTERVAL_SEC)

Usage
-----
>>> from impulsive_sound_detection.mqtt_bridge import MQTTBridge
>>> bridge = MQTTBridge()
>>> bridge.connect()
>>> bridge.publish_detection(result)   # ClassificationResult
>>> bridge.publish_rms(rms, baseline, threshold)
>>> bridge.disconnect()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False
    mqtt = None  # type: ignore[assignment]

from . import config
from .classifier import ClassificationResult

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SEC: float = 10.0
_MAX_RECONNECT_DELAY_SEC: float = 60.0


class MQTTBridge:
    """Thin MQTT publisher wrapper for a single RPi node.

    Parameters
    ----------
    broker_host : str
        IP or hostname of the MQTT broker (host machine).
    broker_port : int
        TCP port of the broker (default 1883).
    node_id : str
        Unique node identifier (used in all topic strings).
    node_location : str
        Human-readable location label embedded in heartbeat messages.
    keepalive : int
        MQTT keepalive interval in seconds.
    """

    def __init__(
        self,
        broker_host: str = config.MQTT_BROKER_HOST,
        broker_port: int = config.MQTT_BROKER_PORT,
        node_id: str = config.NODE_ID,
        node_location: str = config.NODE_LOCATION,
        keepalive: int = config.MQTT_KEEPALIVE_SEC,
    ) -> None:
        if not _PAHO_AVAILABLE:
            raise ImportError(
                "paho-mqtt is required for MQTT mode.\n"
                "Install it with:  pip install paho-mqtt"
            )

        self._broker_host = broker_host
        self._broker_port = broker_port
        self._node_id = node_id
        self._node_location = node_location
        self._keepalive = keepalive

        self._topic_detection = f"isd/node/{node_id}/detection"
        self._topic_rms       = f"isd/node/{node_id}/rms"
        self._topic_heartbeat = f"isd/node/{node_id}/heartbeat"

        self._client: mqtt.Client = mqtt.Client(
            client_id=f"isd-node-{node_id}",
            clean_session=True,
        )
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # LWT – host sees "offline" if node crashes
        will_payload = json.dumps({
            "node_id":  node_id,
            "ts":       0,
            "status":   "offline",
            "location": node_location,
        })
        self._client.will_set(
            self._topic_heartbeat,
            payload=will_payload,
            qos=1,
            retain=True,
        )

        self._connected = threading.Event()
        self._reconnect_delay: float = 1.0

        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()

        self._rms_frame_counter: int = 0

    # ── connection ────────────────────────────────────────────────────
    def connect(self) -> None:
        """Connect to the broker and start the heartbeat thread."""
        logger.info(
            "[%s] Connecting to MQTT broker %s:%d …",
            self._node_id, self._broker_host, self._broker_port,
        )
        self._client.connect_async(
            self._broker_host, self._broker_port, self._keepalive
        )
        self._client.loop_start()

        if not self._connected.wait(timeout=10.0):
            self._client.loop_stop()
            raise ConnectionError(
                f"Could not connect to MQTT broker at "
                f"{self._broker_host}:{self._broker_port} within 10 s"
            )
        self._start_heartbeat()

    def disconnect(self) -> None:
        """Publish offline will, stop heartbeat, and disconnect."""
        self._stop_heartbeat()
        self._publish_heartbeat(status="offline")
        self._client.disconnect()
        self._client.loop_stop()
        logger.info("[%s] MQTT disconnected", self._node_id)

    # ── publish helpers ───────────────────────────────────────────────
    def publish_detection(self, result: ClassificationResult) -> None:
        """Publish a ClassificationResult to the detection topic.

        Uses result.to_json() which includes event_uuid, severity,
        node_id, wall_clock_time, and all other fields added by the
        teammate's classifier work.

        Parameters
        ----------
        result : ClassificationResult
            Detection to send.
        """
        if not self._connected.is_set():
            logger.warning("[%s] Not connected – detection not published", self._node_id)
            return
        payload = result.to_json()
        info = self._client.publish(self._topic_detection, payload=payload, qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning("[%s] Detection publish failed (rc=%d)", self._node_id, info.rc)
        else:
            logger.debug(
                "[%s] Detection published: suspicious=%s conf=%.3f severity=%s",
                self._node_id, result.is_suspicious, result.confidence, result.severity,
            )

    def publish_rms(
        self,
        rms: float,
        baseline: float,
        threshold: float,
        is_trigger: bool = False,
    ) -> None:
        """Publish a lightweight RMS frame (throttled).

        Only publishes every MQTT_RMS_PUBLISH_EVERY_N_FRAMES calls.

        Parameters
        ----------
        rms : float
            Current frame RMS energy.
        baseline : float
            Rolling baseline energy.
        threshold : float
            Dynamic trigger threshold.
        is_trigger : bool
            True if this frame caused a Stage 1 trigger.
        """
        self._rms_frame_counter += 1
        if self._rms_frame_counter % config.MQTT_RMS_PUBLISH_EVERY_N_FRAMES != 0:
            return
        if not self._connected.is_set():
            return
        payload = json.dumps({
            "node_id":    self._node_id,
            "ts":         round(time.time(), 4),
            "rms":        round(rms, 6),
            "baseline":   round(baseline, 6),
            "threshold":  round(threshold, 6),
            "is_trigger": is_trigger,
        })
        self._client.publish(self._topic_rms, payload=payload, qos=0)

    # ── internal ──────────────────────────────────────────────────────
    def _publish_heartbeat(self, status: str = "online") -> None:
        payload = json.dumps({
            "node_id":  self._node_id,
            "ts":       round(time.time(), 4),
            "status":   status,
            "location": self._node_location,
        })
        self._client.publish(self._topic_heartbeat, payload=payload, qos=1, retain=True)

    def _start_heartbeat(self) -> None:
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"mqtt-heartbeat-{self._node_id}",
            daemon=True,
        )
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=3.0)
            self._hb_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.is_set():
            self._publish_heartbeat(status="online")
            self._hb_stop.wait(timeout=HEARTBEAT_INTERVAL_SEC)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            logger.info(
                "[%s] MQTT connected to %s:%d",
                self._node_id, self._broker_host, self._broker_port,
            )
            self._connected.set()
            self._reconnect_delay = 1.0
        else:
            logger.error("[%s] MQTT connect failed rc=%d", self._node_id, rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()
        if rc != 0:
            logger.warning(
                "[%s] Unexpected MQTT disconnect (rc=%d) – retrying in %.0f s",
                self._node_id, rc, self._reconnect_delay,
            )
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, _MAX_RECONNECT_DELAY_SEC
            )
            try:
                client.reconnect()
            except Exception as exc:
                logger.error("[%s] Reconnect failed: %s", self._node_id, exc)
