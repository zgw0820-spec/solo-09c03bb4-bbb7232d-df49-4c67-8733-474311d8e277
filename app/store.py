"""SQLite persistence for diagnosis batches and incident events.

The database is the source of truth for results so they survive an API restart.
A batch is written once at start ("running") and once when it finishes. A batch
still "running" after a restart can no longer be trusted (its in-flight probe
tasks died with the old process), so startup marks it as needing a retest and
it keeps that label forever.

Batch ``seq`` values are assigned by the API and are strictly increasing. When
a slow batch finishes after a newer batch was created, its record is stored but
flagged ``superseded`` so it can never become the latest diagnosis result.

Incident events live in an append-only ``events`` table whose ``event_id`` is
the subscription cursor. The batch's terminal update, the event insert (if the
batch changes the notified incident state) and the notified-state marker are
written in **one transaction**: a crash lands either both or neither, so a
subscriber resuming from its cursor never misses or double-reports a change.
Events are never updated or deleted, so a page once returned stays valid.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import plan_event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id       TEXT PRIMARY KEY,
    seq            INTEGER NOT NULL DEFAULT 0,
    state_version  INTEGER NOT NULL,
    status         TEXT NOT NULL,
    outcome        TEXT,
    created_at     TEXT NOT NULL,
    completed_at   TEXT,
    summary        TEXT,
    superseded     INTEGER NOT NULL DEFAULT 0,
    superseded_by  INTEGER,
    result_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_batches_seq ON batches(seq DESC);
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    batch_id        TEXT NOT NULL,
    batch_seq       INTEGER NOT NULL,
    state_version   INTEGER NOT NULL,
    root_causes_json     TEXT NOT NULL,
    previous_root_causes_json TEXT NOT NULL,
    notified_json   TEXT NOT NULL,
    primary_root_cause TEXT,
    summary         TEXT NOT NULL,
    evidence_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_id ON events(event_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Meta key holding the JSON list of root-cause services the duty system has
# already been notified about. Updated only inside the finish_batch
# transaction, so it always matches the newest event.
NOTIFIED_KEY = "notified_root_causes"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    """Small, thread-safe wrapper around one SQLite file."""

    def __init__(self, path: str, max_stored_batches: int = 200) -> None:
        self.path = path
        self.max_stored_batches = max(1, max_stored_batches)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # -- writes ---------------------------------------------------------
    def create_running_batch(self, batch_id: str, seq: int, state_version: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO batches (batch_id, seq, state_version, status, created_at)"
                " VALUES (?, ?, ?, 'running', ?)",
                (batch_id, seq, state_version, utcnow()),
            )

    def finish_batch(
        self,
        batch_id: str,
        *,
        state_version: int,
        status: str,
        outcome: str,
        summary: str,
        result: dict[str, Any] | None,
        superseded_by: int | None = None,
    ) -> dict[str, Any] | None:
        """Store the terminal state; emit an incident event in the same commit.

        ``superseded_by`` is recorded only when this batch finished after a
        younger batch had already been created; that is the only case where a
        late response genuinely loses a race.

        When the batch is completed, credible and not superseded, and its
        root-cause set differs from the notified set, one incident event is
        inserted and the notified marker advanced — all inside this
        transaction, so batch completion and event generation are atomic.
        Returns the stored event (with its ``event_id`` cursor) or ``None``.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE batches SET status = ?, outcome = ?, summary = ?,"
                " completed_at = ?, superseded = ?, superseded_by = ?,"
                " result_json = ? WHERE batch_id = ?",
                (
                    status,
                    outcome,
                    summary,
                    utcnow(),
                    1 if superseded_by is not None else 0,
                    superseded_by,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    batch_id,
                ),
            )
            event = self._plan_and_insert_event(
                conn,
                status=status,
                superseded=superseded_by is not None,
                result=result,
            )
            conn.execute(
                "DELETE FROM batches WHERE status != 'running' AND batch_id NOT IN ("
                " SELECT batch_id FROM batches WHERE status != 'running'"
                " ORDER BY seq DESC LIMIT ?)",
                (self.max_stored_batches,),
            )
            return event

    # -- incident events ------------------------------------------------
    def _plan_and_insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        status: str,
        superseded: bool,
        result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Insert an incident event if this batch changes the notified state.

        Runs inside the caller's transaction: the notified marker is read and
        advanced here, so concurrent finishers cannot both emit or both skip.
        """
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (NOTIFIED_KEY,)
        ).fetchone()
        notified = json.loads(row["value"]) if row else []
        planned = plan_event(
            status=status, superseded=superseded, result=result, notified=notified
        )
        if planned is None:
            return None
        payload, new_notified = planned
        cur = conn.execute(
            "INSERT INTO events (event_type, batch_id, batch_seq, state_version,"
            " root_causes_json, previous_root_causes_json, notified_json,"
            " primary_root_cause, summary, evidence_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payload["event_type"],
                payload["batch_id"],
                payload["batch_seq"],
                payload["state_version"],
                json.dumps(payload["root_causes"], ensure_ascii=False),
                json.dumps(payload["previous_root_causes"], ensure_ascii=False),
                json.dumps(new_notified, ensure_ascii=False),
                payload["primary_root_cause"],
                payload["summary"],
                json.dumps(payload["evidence"], ensure_ascii=False),
                utcnow(),
            ),
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (NOTIFIED_KEY, json.dumps(new_notified, ensure_ascii=False)),
        )
        event = dict(payload)
        event["event_id"] = int(cur.lastrowid)
        return event

    def list_events(self, cursor: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        """Events with ``event_id > cursor``, oldest first (immutable pages)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_id > ?"
                " ORDER BY event_id ASC LIMIT ?",
                (max(0, int(cursor)), max(1, int(limit))),
            ).fetchall()
        return [_event_row_to_dict(row) for row in rows]

    def latest_event_id(self) -> int:
        """Highest assigned event id (0 when no event exists yet)."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(event_id), 0) AS m FROM events"
            ).fetchone()
        return int(row["m"])

    def count_events(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()
        return int(row["c"])

    def notified_root_causes(self) -> list[str]:
        """Root-cause services the duty system was last notified about."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (NOTIFIED_KEY,)
            ).fetchone()
        return json.loads(row["value"]) if row else []

    def mark_all_running_interrupted(self, reason: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE batches SET status = 'interrupted', outcome = 'needs_retest',"
                " summary = ?, completed_at = ? WHERE status = 'running'",
                (reason, utcnow()),
            )
            return cur.rowcount

    # -- meta counters --------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- reads ----------------------------------------------------------
    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return _row_to_dict(row)

    def latest_batch(self, *, completed_only: bool = False) -> dict[str, Any] | None:
        query = "SELECT * FROM batches"
        if completed_only:
            query += " WHERE status = 'completed'"
        query += " ORDER BY seq DESC LIMIT 1"
        with self._lock, self._connect() as conn:
            row = conn.execute(query).fetchone()
        return _row_to_dict(row)

    def max_seq(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM batches").fetchone()
        return int(row["m"])

    def list_batches(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batches ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    raw = data.pop("result_json", None)
    data["result"] = json.loads(raw) if raw else None
    return data


def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["root_causes"] = json.loads(data.pop("root_causes_json"))
    data["previous_root_causes"] = json.loads(data.pop("previous_root_causes_json"))
    data["evidence"] = json.loads(data.pop("evidence_json"))
    # The notified marker after this event; useful for auditing the feed.
    data["notified_root_causes"] = json.loads(data.pop("notified_json"))
    return data


__all__ = ["Store", "utcnow"]
