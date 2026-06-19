from __future__ import annotations

import random

from .. import config
from . import question_bank as qb
from .question_bank import Question
from .word_provider import WordProvider, WordsUnavailable

MIN_THETA, MAX_THETA = 0.0, float(config.NUM_LEVELS - 1)


def languages_for_mode(mode: str) -> list[str]:
    if mode == "ru":
        return ["RU"]
    if mode == "en":
        return ["EN"]
    return ["RU", "EN"]


def _step(n_answers: int) -> float:
    return max(0.4, 1.5 * (0.8 ** n_answers))


def _theta_to_level(theta: float) -> str:
    idx = round(min(MAX_THETA, max(MIN_THETA, theta)))
    return config.LEVELS[idx]


def pick_language(session) -> str:
    langs = session.languages
    return langs[session.index % len(langs)]


def pick_type(session) -> str:
    single = len(session.languages) == 1
    i = session.index
    if not single and i % 5 == 4:
        return qb.TYPE_SENTENCE_GAP
    if i % 5 == 2:
        return qb.TYPE_FAKE_SEEN
    return qb.TYPE_VOCAB


def next_question(session, provider: WordProvider) -> Question:
    qtype = pick_type(session)

    if qtype == qb.TYPE_SENTENCE_GAP:
        lang = pick_language(session)
        level = _theta_to_level(session.theta[lang])
        q = qb.make_sentence_gap(lang, level, avoid=session.seen_gaps)
        session.seen_gaps.add(q.extra.get("raw_sentence", q.prompt))
    else:
        lang = pick_language(session)
        level = _theta_to_level(session.theta[lang])
        try:
            if qtype == qb.TYPE_FAKE_SEEN:
                q = qb.make_fake_seen(provider, lang, level, avoid=session.seen_words)
            else:
                q = qb.make_vocab(provider, lang, level, avoid=session.seen_words)
        except WordsUnavailable:
            q = qb.make_vocab(provider, lang, level, avoid=session.seen_words)
        session.seen_words.add(f"{q.lang}:{q.prompt}")

    q.index = session.index + 1
    q.total = session.k
    return q


def grade(question: Question, answer: str) -> dict:
    answer = (answer or "").strip()

    if question.type == qb.TYPE_SENTENCE_GAP:
        return {
            "correct": answer == question.correct_answer,
            "is_fake": False,
            "caught_lie": None,
            "affects_theta": True,
            "lang": question.lang,
        }

    said_yes = answer == qb.ANS_YES
    if question.legitimate:
        return {
            "correct": said_yes,
            "is_fake": False,
            "caught_lie": None,
            "affects_theta": True,
            "lang": question.lang,
        }

    return {
        "correct": None,
        "is_fake": True,
        "caught_lie": said_yes,
        "affects_theta": False,
        "lang": question.lang,
    }


def apply_result(session, result: dict) -> None:
    if not result["affects_theta"] or result["correct"] is None:
        return
    lang = result["lang"]
    step = _step(session.theta_count[lang])
    delta = step if result["correct"] else -step
    session.theta[lang] = min(MAX_THETA, max(MIN_THETA, session.theta[lang] + delta))
    session.theta_count[lang] += 1
