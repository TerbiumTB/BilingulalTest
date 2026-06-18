"""Конфигурация приложения.

Переключение источника слов и базовые константы теста. Всё через
переменные окружения, чтобы позже легко подменить файловый провайдер
на реальный HTTP-клиент модели Тимы без правки кода.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL2IDX = {lvl: i for i, lvl in enumerate(LEVELS)}
NUM_LEVELS = len(LEVELS)

LANGUAGES = ["RU", "EN"]

WORD_REAL = "Real"
WORD_FAKE = "Fake"
WORD_RANDOM = "Random"

DEFAULT_K = 10

REAL_WORD_RATIO = 0.7

WORD_PROVIDER = os.environ.get("WORD_PROVIDER", "file")
WORDS_DIR = Path(os.environ.get("WORDS_DIR", BASE_DIR / "data" / "words"))
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:9000")

DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data" / "bilingual.db"))

STATIC_DIR = BASE_DIR / "static"
