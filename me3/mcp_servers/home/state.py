"""SQLite-backed state for the home server: devices, reminders, timers/alarms, events, call log."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, label TEXT NOT NULL,
    due REAL NOT NULL, fired INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, message TEXT NOT NULL,
    created REAL NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS intent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, intent TEXT NOT NULL,
    slots TEXT NOT NULL, session TEXT);
"""

DEFAULT_DEVICES = {
    "lights_on": False,
    "brightness": 100,
    "color": "white",
    "temperature": 24,
}


class HomeState:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        for key, value in DEFAULT_DEVICES.items():
            self.db.execute(
                "INSERT OR IGNORE INTO devices VALUES (?, ?)", (key, json.dumps(value))
            )
        self.db.commit()

    # --- devices -------------------------------------------------------------
    def get(self, key: str):
        row = self.db.execute("SELECT value FROM devices WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO devices VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def devices(self) -> dict:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM devices")}

    # --- reminders -----------------------------------------------------------
    def add_reminder(self, task: str) -> int:
        cur = self.db.execute(
            "INSERT INTO reminders (task, created) VALUES (?, ?)", (task, time.time())
        )
        self.db.commit()
        return cur.lastrowid

    def reminders(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT task FROM reminders ORDER BY id")]

    # --- timers / alarms -------------------------------------------------------
    def schedule(self, kind: str, label: str, due: float) -> int:
        cur = self.db.execute(
            "INSERT INTO schedules (kind, label, due) VALUES (?, ?, ?)", (kind, label, due)
        )
        self.db.commit()
        return cur.lastrowid

    def pop_due(self, now: float) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT id, kind, label FROM schedules WHERE fired = 0 AND due <= ?", (now,)
        ).fetchall()
        for row_id, _, _ in rows:
            self.db.execute("UPDATE schedules SET fired = 1 WHERE id = ?", (row_id,))
        self.db.commit()
        return [(kind, label) for _, kind, label in rows]

    def pending_schedules(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT kind, label, due FROM schedules WHERE fired = 0 ORDER BY due"
        )
        return [{"kind": k, "label": l, "due": d} for k, l, d in rows]

    # --- events (pushed to the voice agent) -----------------------------------
    def add_event(self, kind: str, message: str) -> None:
        self.db.execute(
            "INSERT INTO events (kind, message, created) VALUES (?, ?, ?)",
            (kind, message, time.time()),
        )
        self.db.commit()

    def take_events(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, kind, message, created FROM events WHERE delivered = 0 ORDER BY id"
        ).fetchall()
        if rows:
            self.db.execute(
                f"UPDATE events SET delivered = 1 WHERE id IN ({','.join('?' * len(rows))})",
                [r[0] for r in rows],
            )
            self.db.commit()
        return [{"kind": k, "message": m, "created": c} for _, k, m, c in rows]

    # --- intent log (benchmark ground truth comparison) ------------------------
    def log_intent(self, intent: str, slots: dict, session: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO intent_log (ts, intent, slots, session) VALUES (?, ?, ?, ?)",
            (time.time(), intent, json.dumps(slots), session),
        )
        self.db.commit()

    def intents_since(self, ts: float) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts, intent, slots FROM intent_log WHERE ts >= ? ORDER BY id", (ts,)
        )
        return [{"ts": t, "intent": i, "slots": json.loads(s)} for t, i, s in rows]
