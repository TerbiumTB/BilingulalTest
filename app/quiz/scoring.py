from __future__ import annotations

from .. import config

MAX_THETA = float(config.NUM_LEVELS - 1)


def _cefr(theta: float) -> str:
    idx = round(min(MAX_THETA, max(0.0, theta)))
    return config.LEVELS[idx]


def _pct(theta: float) -> int:
    return round(theta / MAX_THETA * 100)


def compute_result(session) -> dict:
    theta_ru = session.theta["RU"]
    theta_en = session.theta["EN"]

    balance = 1 - abs(theta_ru - theta_en) / MAX_THETA
    weaker = min(theta_ru, theta_en) / MAX_THETA
    bilingualism = round(balance * weaker * 100)

    fake_total = session.fake_total
    caught = session.fake_caught
    honesty = round((1 - caught / fake_total) * 100) if fake_total else 100

    return {
        "mode": session.mode,
        "questions": session.index,
        "ru": {"percent": _pct(theta_ru), "cefr": _cefr(theta_ru)},
        "en": {"percent": _pct(theta_en), "cefr": _cefr(theta_en)},
        "bilingualism": bilingualism,
        "balance": round(balance * 100),
        "honesty": honesty,
        "honesty_flag": honesty < 60,
        "fake_total": fake_total,
        "fake_caught": caught,
    }
