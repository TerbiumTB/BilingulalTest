from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Protocol

from .. import config


class WordProvider(Protocol):
    def get_word(self, language: str, level: str, type: str) -> str:
        """type ∈ {Real, Fake, Random}. Возвращает лемму."""
        ...

    def generate_words(self, language: str, level: str, number: int) -> list[str]:
        ...


class WordsUnavailable(RuntimeError):
    """Нет слов под запрошенные (язык, уровень, тип)."""


class FileWordProvider:
    """
    Индекс: (language, level, legitimate) -> [lemmas]. Если под точный
    уровень слов нет, берём ближайший по индексу уровень.
    """

    def __init__(self, words_dir: Path | None = None):
        self.words_dir = Path(words_dir or config.WORDS_DIR)
        # (lang, level, legitimate) -> list[str]
        self._index: dict[tuple[str, str, bool], list[str]] = {}
        self._load()

    def _load(self) -> None:
        for lang in config.LANGUAGES:
            path = self.words_dir / f"{lang.lower()}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lemma = (row.get("lemma") or "").strip()
                    level = (row.get("level") or "").strip().upper()
                    if not lemma or level not in config.LEVEL2IDX:
                        continue
                    legit = str(row.get("legitimate", "")).strip().lower() in (
                        "true", "1", "yes", "real",
                    )
                    self._index.setdefault((lang, level, legit), []).append(lemma)

    def _pool(self, language: str, legitimate: bool, level: str) -> list[str]:
        target = config.LEVEL2IDX[level]
        order = sorted(config.LEVELS, key=lambda lv: abs(config.LEVEL2IDX[lv] - target))
        for lv in order:
            pool = self._index.get((language, lv, legitimate))
            if pool:
                return pool
        return []

    def get_word(self, language: str, level: str, type: str) -> str:
        language = language.upper()
        level = level.upper()
        if type == config.WORD_RANDOM:
            legitimate = random.random() < config.REAL_WORD_RATIO
        else:
            legitimate = type == config.WORD_REAL
        pool = self._pool(language, legitimate, level)
        if not pool:
            raise WordsUnavailable(f"нет слов для {language}/{level}/{type}")
        return random.choice(pool)

    def generate_words(self, language: str, level: str, number: int) -> list[str]:
        language = language.upper()
        level = level.upper()
        pool = self._pool(language, True, level)
        if not pool:
            raise WordsUnavailable(f"нет слов для {language}/{level}")
        if number <= len(pool):
            return random.sample(pool, number)
        return [random.choice(pool) for _ in range(number)]


class ModelWordProvider:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or config.MODEL_API_URL).rstrip("/")

    def get_word(self, language: str, level: str, type: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/get_word",
            json={"Language": language, "Level": level, "Type": type},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["word"]

    def generate_words(self, language: str, level: str, number: int) -> list[str]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/generate_words",
            json={"Language": language, "Level": level, "Number": number},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["words"]


def get_provider() -> WordProvider:
    if config.WORD_PROVIDER == "model":
        return ModelWordProvider()
    return FileWordProvider()
