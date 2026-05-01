"""
event_logger.py – Event logging to SQLite and JSONL.

Handles thread-safe persistence of detection events to both SQLite database
and JSONL file for the partner's web app integration.

Public API
----------
EventLogger
    Stateful logger that writes to both SQLite and JSONL.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .classifier import ClassificationResult

logger = logging.getLogger(__name__)


class EventLogger:
    """Thread-safe logger for detection events to SQLite + JSONL.

    Parameters
    ----------
    sqlite_path : Path
        Path to SQLite database file.
    jsonl_path : Path
        Path to JSONL log file.
    classifier_version : str
        Version of the classifier (for auditing).
    """

    def __init__(
        self,
        sqlite_path: Path = Path("logs/detections.db"),
        jsonl_path: Optional[Path] = None,
        classifier_version: str = "1.0",
    ) -> None:
        self._sqlite_path = Path(sqlite_path)
        self._jsonl_path = jsonl_path
        self._classifier_version = classifier_version
        self._lock = threading.Lock()
        self._session_id = self._generate_session_id()

        # Ensure directories exist
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self._init_sqlite()
        logger.info("EventLogger initialized. SQLite: %s, JSONL: %s, Session: %s",
                   self._sqlite_path, self._jsonl_path, self._session_id)

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _init_sqlite(self) -> None:
        """Create SQLite schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
            cursor = conn.cursor()

            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid      TEXT    NOT NULL UNIQUE,
                    timestamp_unix  REAL    NOT NULL,
                    timestamp_iso   TEXT    NOT NULL,
                    onset_index     INTEGER NOT NULL,
                    label           TEXT    NOT NULL,
                    confidence      REAL    NOT NULL,
                    is_suspicious   INTEGER NOT NULL,
                    severity        TEXT    NOT NULL DEFAULT 'LOW',
                    location_id     TEXT    DEFAULT NULL,
                    audio_clip_path TEXT    DEFAULT NULL,
                    session_id      TEXT    NOT NULL,
                    classifier_version TEXT NOT NULL DEFAULT '1.0',
                    created_at      TEXT    NOT NULL
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_events(timestamp_unix)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_suspicious ON detection_events(is_suspicious)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON detection_events(session_id)")

            conn.commit()
            conn.close()
            logger.info("SQLite schema initialized: %s", self._sqlite_path)
        except Exception as exc:
            logger.error("Failed to initialize SQLite: %s", exc)
            raise

    def log(
        self,
        result: ClassificationResult,
        location_id: Optional[str] = None,
        audio_clip_path: Optional[str] = None,
    ) -> None:
        """
        Log a detection event to SQLite and JSONL.

        Parameters
        ----------
        result : ClassificationResult
            The classification result to log.
        location_id : str, optional
            Location/room identifier.
        audio_clip_path : str, optional
            Path to the saved audio clip.
        """
        with self._lock:
            try:
                # Prepare data
                result.session_id = self._session_id
                timestamp_iso = datetime.fromtimestamp(
                    result.timestamp, tz=timezone.utc
                ).isoformat()
                created_at = datetime.now(tz=timezone.utc).isoformat()

                # Write to SQLite
                conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO detection_events (
                        event_uuid, timestamp_unix, timestamp_iso, onset_index,
                        label, confidence, is_suspicious, severity,
                        location_id, audio_clip_path, session_id,
                        classifier_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.event_uuid,
                    result.timestamp,
                    timestamp_iso,
                    result.onset_index,
                    result.label,
                    result.confidence,
                    1 if result.is_suspicious else 0,
                    result.severity,
                    location_id,
                    audio_clip_path,
                    self._session_id,
                    self._classifier_version,
                    created_at,
                ))

                conn.commit()
                conn.close()

                # Write to JSONL if path provided
                if self._jsonl_path:
                    self._write_jsonl(result, location_id, audio_clip_path, created_at)

                logger.debug("Event logged: %s (uuid=%s)", result.label, result.event_uuid)

            except Exception as exc:
                logger.error("Failed to log event: %s", exc)
                raise

    def _write_jsonl(
        self,
        result: ClassificationResult,
        location_id: Optional[str],
        audio_clip_path: Optional[str],
        created_at: str,
    ) -> None:
        """Write event to JSONL file."""
        try:
            event_dict = {
                "event_uuid": result.event_uuid,
                "timestamp_unix": result.timestamp,
                "timestamp_iso": datetime.fromtimestamp(
                    result.timestamp, tz=timezone.utc
                ).isoformat(),
                "onset_index": result.onset_index,
                "label": result.label,
                "confidence": round(result.confidence, 4),
                "is_suspicious": result.is_suspicious,
                "severity": result.severity,
                "location": {
                    "id": location_id,
                    "description": None,
                    "lat": None,
                    "lon": None,
                },
                "audio_clip_path": audio_clip_path,
                "session_id": result.session_id,
                "classifier_version": self._classifier_version,
                "created_at": created_at,
            }

            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(event_dict) + "\n")

        except Exception as exc:
            logger.error("Failed to write JSONL: %s", exc)
            raise

    def query_events(
        self,
        is_suspicious: Optional[bool] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """
        Query events from the database.

        Parameters
        ----------
        is_suspicious : bool, optional
            Filter by suspicious flag.
        session_id : str, optional
            Filter by session ID.
        limit : int
            Maximum number of events to return.

        Returns
        -------
        list
            List of event dictionaries.
        """
        try:
            conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM detection_events WHERE 1=1"
            params = []

            if is_suspicious is not None:
                query += " AND is_suspicious = ?"
                params.append(1 if is_suspicious else 0)

            if session_id is not None:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY timestamp_unix DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as exc:
            logger.error("Failed to query events: %s", exc)
            return []

    def close(self) -> None:
        """Close any open resources."""
        logger.info("EventLogger closed. Session: %s", self._session_id)
