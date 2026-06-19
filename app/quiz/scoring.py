from __future__ import annotations

from .. import config

MAX_THETA = float(config.NUM_LEVELS - 1)


def _pct(theta: float) -> int:
    """Сырой балл языка по адаптивной шкале θ (0..5) -> 0..100%."""
    return round(theta / MAX_THETA * 100)


def _percentile(value: float, pool: list[int] | None) -> int | None:
    """Насколько результат лучше тех, кто проходил тест раньше: доля прошлых
    сессий со строго меньшим баллом. None — если статистики ещё нет."""
    if not pool:
        return None
    return round(100 * sum(1 for x in pool if x < value) / len(pool))


def _flexibility(session) -> int | None:
    """Межъязыковая гибкость: среднее процента верных по двум упражнениям —
    вставка слова (sentence_gap) и сжатие текста (text_compression)."""
    by_type: dict[str, list[int]] = {}
    for a in session.cross_answers:
        d = by_type.setdefault(a["type"], [0, 0])
        d[0] += 1 if a["correct"] else 0
        d[1] += 1
    pcts = [c / t * 100 for c, t in by_type.values() if t]
    return round(sum(pcts) / len(pcts)) if pcts else None


def _direction_asymmetry(session) -> dict | None:
    """В какую сторону переводить легче. У кросс-языковых заданий проверяемый
    язык = язык вариантов: EN -> направление RU→EN, RU -> направление EN→RU.
    Нужны оба направления, иначе сравнивать не с чем -> None."""
    by_lang = {"RU": [0, 0], "EN": [0, 0]}
    for a in session.cross_answers:
        if a["lang"] in by_lang:
            by_lang[a["lang"]][0] += 1 if a["correct"] else 0
            by_lang[a["lang"]][1] += 1
    ce, te = by_lang["EN"]   # перевод В английский: RU→EN
    cr, tr = by_lang["RU"]   # перевод В русский:    EN→RU
    if not te or not tr:
        return None
    into_en = round(ce / te * 100)
    into_ru = round(cr / tr * 100)
    diff = into_en - into_ru
    easier = "RU→EN" if diff > 0 else "EN→RU" if diff < 0 else None
    return {"value": diff, "easier": easier, "into_en": into_en, "into_ru": into_ru}


def _interference_resistance(session) -> int | None:
    """Устойчивость к интерференции: доля верно отклонённых псевдослов.
    None — если ловушек в этой сессии не было."""
    if not session.fake_total:
        return None
    return round((1 - session.fake_caught / session.fake_total) * 100)


def compute_result(session, pool: dict | None = None) -> dict:
    pool = pool or {}
    theta_ru = session.theta["RU"]
    theta_en = session.theta["EN"]
    ru_pct = _pct(theta_ru)
    en_pct = _pct(theta_en)

    # --- метрики билингвальности ---
    balance = 100 - abs(ru_pct - en_pct)            # 100 = языки равны
    dominance = en_pct - ru_pct                      # >0 английский сильнее, <0 русский
    weaker = min(theta_ru, theta_en) / MAX_THETA
    bilingualism = round((balance / 100) * weaker * 100)

    interference = _interference_resistance(session)
    # honesty храним для дашборда/БД — это та же величина (устойчивость к интерференции).
    honesty = interference if interference is not None else 100

    return {
        "mode": session.mode,
        "questions": session.index,
        "ru": {"percent": ru_pct, "percentile": _percentile(ru_pct, pool.get("ru"))},
        "en": {"percent": en_pct, "percentile": _percentile(en_pct, pool.get("en"))},
        "bilingualism": bilingualism,
        "bilingualism_percentile": _percentile(bilingualism, pool.get("bilingualism")),
        "balance": balance,
        "dominance": dominance,
        "cross_language_flexibility": _flexibility(session),
        "direction_asymmetry": _direction_asymmetry(session),
        "interference_resistance": interference,
        "honesty": honesty,
        "honesty_flag": honesty < 60,
        "fake_total": session.fake_total,
        "fake_caught": session.fake_caught,
    }
