from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import config
from . import engine
from .question_bank import Question


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    id: str
    mode: str
    k: int
    languages: list[str]
    started_at: str
    theta: dict = field(default_factory=lambda: {"RU": 2.0, "EN": 2.0})
    theta_count: dict = field(default_factory=lambda: {"RU": 0, "EN": 0})
    index: int = 0
    fake_total: int = 0
    fake_caught: int = 0
    current: Question | None = None
    finished: bool = False
    result: dict | None = None
    seen_words: set = field(default_factory=set)
    seen_gaps: set = field(default_factory=set)
    seen_texts: set = field(default_factory=set)
    # Результаты кросс-языковых упражнений (вставка слова + сжатие текста):
    # по ним считаются «межъязыковая гибкость» и «направленность перевода».
    cross_answers: list = field(default_factory=list)  # [{"type", "lang", "correct"}]


class SessionStore:
    def __init__(self):
        self._store: dict[str, Session] = {}

    def create(self, mode: str, k: int) -> Session:
        mode = mode if mode in ("bilingual", "ru", "en") else "bilingual"
        k = max(1, min(50, int(k)))
        session = Session(
            id=secrets.token_hex(8),
            mode=mode,
            k=k,
            languages=engine.languages_for_mode(mode),
            started_at=_now(),
        )
        self._store[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._store.get(session_id)


store = SessionStore()


def now() -> str:
    return _now()
