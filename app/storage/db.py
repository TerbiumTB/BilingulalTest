"""SQLite-хранилище: сессии, ответы и агрегаты для дашборда админа."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    mode         TEXT,
    k            INTEGER,
    started_at   TEXT,
    finished_at  TEXT,
    questions    INTEGER,
    theta_ru     REAL,
    theta_en     REAL,
    ru_percent   INTEGER,
    en_percent   INTEGER,
    bilingualism INTEGER,
    honesty      INTEGER
);
CREATE TABLE IF NOT EXISTS answers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    idx         INTEGER,
    type        TEXT,
    lang        TEXT,
    level       TEXT,
    is_fake     INTEGER,
    caught_lie  INTEGER,
    prompt      TEXT,
    options_json TEXT,
    user_answer TEXT,
    correct     INTEGER,
    created_at  TEXT
);
"""


@contextmanager
def get_conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def save_answer(session_id: str, idx: int, question, result: dict,
                user_answer: str, created_at: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO answers
               (session_id, idx, type, lang, level, is_fake, caught_lie,
                prompt, options_json, user_answer, correct, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, idx, question.type, result["lang"], question.level,
                1 if result["is_fake"] else 0,
                None if result["caught_lie"] is None else int(result["caught_lie"]),
                question.prompt,
                json.dumps(question.options, ensure_ascii=False) if question.options else None,
                user_answer,
                None if result["correct"] is None else int(result["correct"]),
                created_at,
            ),
        )


def save_session(session, result: dict, finished_at: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, mode, k, started_at, finished_at, questions,
                theta_ru, theta_en, ru_percent, en_percent, bilingualism, honesty)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session.id, session.mode, session.k, session.started_at, finished_at,
                result["questions"], session.theta["RU"], session.theta["EN"],
                result["ru"]["percent"], result["en"]["percent"],
                result["bilingualism"], result["honesty"],
            ),
        )


def admin_stats() -> dict:
    with get_conn() as conn:
        sessions = conn.execute(
            "SELECT * FROM sessions WHERE finished_at IS NOT NULL ORDER BY finished_at DESC"
        ).fetchall()

        total = len(sessions)
        avg = {"ru": 0, "en": 0, "bilingualism": 0, "honesty": 0}
        if total:
            avg["ru"] = round(sum(s["ru_percent"] for s in sessions) / total)
            avg["en"] = round(sum(s["en_percent"] for s in sessions) / total)
            avg["bilingualism"] = round(sum(s["bilingualism"] for s in sessions) / total)
            avg["honesty"] = round(sum(s["honesty"] for s in sessions) / total)

        by_level = conn.execute(
            """SELECT level,
                      COUNT(*) AS n,
                      AVG(correct) AS acc
               FROM answers WHERE correct IS NOT NULL
               GROUP BY level ORDER BY level"""
        ).fetchall()

        by_type = conn.execute(
            """SELECT type,
                      COUNT(*) AS n,
                      AVG(correct) AS acc
               FROM answers WHERE correct IS NOT NULL
               GROUP BY type"""
        ).fetchall()

        fake_row = conn.execute(
            """SELECT COUNT(*) AS n, AVG(caught_lie) AS rate
               FROM answers WHERE is_fake = 1"""
        ).fetchone()

        recent = sessions[:15]

    return {
        "total_sessions": total,
        "avg": avg,
        "by_level": [
            {"level": r["level"], "n": r["n"], "accuracy": round((r["acc"] or 0) * 100)}
            for r in by_level
        ],
        "by_type": [
            {"type": r["type"], "n": r["n"], "accuracy": round((r["acc"] or 0) * 100)}
            for r in by_type
        ],
        "fake": {
            "total": fake_row["n"] or 0,
            "caught_rate": round((fake_row["rate"] or 0) * 100),
        },
        "recent": [
            {
                "id": s["id"],
                "mode": s["mode"],
                "finished_at": s["finished_at"],
                "ru": s["ru_percent"],
                "en": s["en_percent"],
                "bilingualism": s["bilingualism"],
                "honesty": s["honesty"],
            }
            for s in recent
        ],
    }
