import json
from datetime import datetime, timezone

import aiosqlite

from socialcloud.gender_detection.config import DB_PATH

_db: aiosqlite.Connection | None = None


async def init_db():
    global _db
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            username         TEXT PRIMARY KEY,
            display_name     TEXT,
            gender           TEXT,
            status           TEXT DEFAULT 'pending',
            signals          TEXT DEFAULT '{}',
            pending_signals  TEXT DEFAULT '[]',
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at      TIMESTAMP
        )
    """)
    await _db.commit()


async def close_db():
    global _db
    if _db:
        await _db.close()
        _db = None


async def upsert_profile(
    username: str,
    display_name: str | None,
    signals: dict,
    gender: str | None,
    status: str,
):
    resolved_at = datetime.now(timezone.utc).isoformat() if status == "resolved" else None
    await _db.execute(
        """
        INSERT INTO profiles (username, display_name, gender, status, signals, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            gender=excluded.gender, status=excluded.status, signals=excluded.signals,
            resolved_at=excluded.resolved_at
    """,
        (username, display_name, gender, status, json.dumps(signals), resolved_at),
    )
    await _db.commit()


async def get_profile(username: str) -> dict | None:
    cursor = await _db.execute("SELECT * FROM profiles WHERE username = ?", (username,))
    row = await cursor.fetchone()
    if not row:
        return None
    return _row_to_dict(row)


async def get_resolved_profiles(gender: str | None = None, limit: int = 50) -> list[dict]:
    if gender:
        cursor = await _db.execute(
            "SELECT * FROM profiles WHERE status = 'resolved' AND gender = ? LIMIT ?",
            (gender, limit),
        )
    else:
        cursor = await _db.execute("SELECT * FROM profiles WHERE status = 'resolved' LIMIT ?", (limit,))
    return [_row_to_dict(r) for r in await cursor.fetchall()]


def _row_to_dict(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["signals"] = json.loads(d["signals"])
    return d
