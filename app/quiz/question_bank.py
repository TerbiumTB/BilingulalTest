from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from .word_provider import WordProvider, WordsUnavailable

TYPE_VOCAB = "vocab"
TYPE_FAKE_SEEN = "fake_seen"
TYPE_SENTENCE_GAP = "sentence_gap"
TYPE_TRANSLATE = "translate"

ANSWER_BINARY = "binary"
ANSWER_CHOICE = "choice"

ANS_YES = "yes"
ANS_NO = "no"


@dataclass
class Question:
    question_id: str
    type: str
    lang: str
    level: str
    prompt: str
    answer_kind: str
    options: list[str] | None = None
    legitimate: bool | None = None
    correct_answer: str | None = None
    affects_theta: bool = True
    index: int = 0
    total: int = 0
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "question_id": self.question_id,
            "type": self.type,
            "lang": self.lang,
            "level": self.level,
            "prompt": self.prompt,
            "answer_kind": self.answer_kind,
            "options": self.options,
            "instruction": self.extra.get("instruction", ""),
            "yes_label": self.extra.get("yes_label", "Да"),
            "no_label": self.extra.get("no_label", "Нет"),
            "index": self.index,
            "total": self.total,
        }


DIR_RU_TO_EN = "RU_to_EN"
DIR_EN_TO_RU = "EN_to_RU"

_DIR_LANG = {DIR_RU_TO_EN: "EN", DIR_EN_TO_RU: "RU"}
_DIR_INSTRUCTION = {
    DIR_RU_TO_EN: "Выберите английское слово, подходящее на место пропуска",
    DIR_EN_TO_RU: "Выберите русское слово, подходящее на место пропуска",
}

_FALLBACK_GAP_BANK: list[dict] = [
    {
        "level": "A1",
        "direction": DIR_RU_TO_EN,
        "sentence": "Мой младший брат каждый день идёт в ___ .",
        "answer": "school",
        "distractors": ["house", "water", "green", "run"],
    },
    {
        "level": "A2",
        "direction": DIR_RU_TO_EN,
        "sentence": "На выходных мы поедем на ___ .",
        "answer": "sea",
        "distractors": ["book", "chair", "loud", "buy"],
    },
    {
        "level": "B1",
        "direction": DIR_RU_TO_EN,
        "sentence": "Он принял трудное ___ и уволился.",
        "answer": "decision",
        "distractors": ["invitation", "celebration", "direction", "tradition"],
    },
]

_GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "words" / "gap_fill_bilingual_examples.txt"


def _load_gap_bank() -> list[dict]:
    try:
        raw = _GAP_FILE.read_text(encoding="utf-8")
        data = ast.literal_eval(raw)
    except (OSError, ValueError, SyntaxError):
        return list(_FALLBACK_GAP_BANK)

    bank: list[dict] = []
    for item in data:
        try:
            sentence = item["sentence"]
            answer = item["answer"]
            level = item["level"]
        except (KeyError, TypeError):
            continue
        if not sentence or not answer or level not in config.LEVEL2IDX:
            continue
        direction = item.get("direction", DIR_RU_TO_EN)
        if direction not in _DIR_LANG:
            direction = DIR_RU_TO_EN
        bank.append({
            "level": level,
            "direction": direction,
            "sentence": sentence,
            "answer": answer,
            "distractors": list(item.get("distractors", [])),
        })
    return bank or list(_FALLBACK_GAP_BANK)


SENTENCE_GAP_BANK: list[dict] = _load_gap_bank()


def _gap_pick(level: str, lang: str, avoid: set[str] | None = None) -> dict:
    avoid = avoid or set()
    pool_all = [t for t in SENTENCE_GAP_BANK if _DIR_LANG[t["direction"]] == lang]
    if not pool_all:  # на случай, если для языка нет примеров — берём любые
        pool_all = SENTENCE_GAP_BANK
    target = config.LEVEL2IDX[level]
    order = sorted(
        pool_all,
        key=lambda t: abs(config.LEVEL2IDX[t["level"]] - target),
    )
    fresh = [t for t in order if t["sentence"] not in avoid]
    pool = fresh or order
    best = pool[0]["level"]
    candidates = [t for t in pool if t["level"] == best]
    return random.choice(candidates)


def _new_id() -> str:
    return "q" + "".join(random.choice("0123456789abcdef") for _ in range(12))


def _pick_word(provider: WordProvider, lang: str, level: str, word_type: str,
               avoid: set[str], tries: int = 10) -> str:
    lemma = provider.get_word(lang, level, word_type)
    for _ in range(tries):
        if f"{lang}:{lemma}" not in avoid:
            return lemma
        lemma = provider.get_word(lang, level, word_type)
    return lemma


def make_vocab(provider: WordProvider, lang: str, level: str,
               avoid: set[str] | None = None) -> Question:
    legit = random.random() < config.REAL_WORD_RATIO
    word_type = config.WORD_REAL if legit else config.WORD_FAKE
    lemma = _pick_word(provider, lang, level, word_type, avoid or set())
    return Question(
        question_id=_new_id(),
        type=TYPE_VOCAB,
        lang=lang,
        level=level,
        prompt=lemma,
        answer_kind=ANSWER_BINARY,
        legitimate=legit,
        affects_theta=legit,
        extra={
            "instruction": "Знаете ли вы это слово?",
            "yes_label": "Знаю",
            "no_label": "Не знаю",
        },
    )


def make_fake_seen(provider: WordProvider, lang: str, level: str,
                   avoid: set[str] | None = None) -> Question:
    lemma = _pick_word(provider, lang, level, config.WORD_FAKE, avoid or set())
    return Question(
        question_id=_new_id(),
        type=TYPE_FAKE_SEEN,
        lang=lang,
        level=level,
        prompt=lemma,
        answer_kind=ANSWER_BINARY,
        legitimate=False,
        affects_theta=False,
        extra={
            "instruction": "Встречали ли вы такое слово?",
            "yes_label": "Встречал",
            "no_label": "Не встречал",
        },
    )


def make_sentence_gap(lang: str, level: str,
                      avoid: set[str] | None = None) -> Question:
    tpl = _gap_pick(level, lang, avoid)
    direction = tpl["direction"]
    tested_lang = _DIR_LANG[direction]
    options = [tpl["answer"]] + list(tpl["distractors"])
    random.shuffle(options)
    return Question(
        question_id=_new_id(),
        type=TYPE_SENTENCE_GAP,
        lang=tested_lang,
        level=tpl["level"],
        prompt=tpl["sentence"].replace("___", "______"),
        answer_kind=ANSWER_CHOICE,
        options=options,
        correct_answer=tpl["answer"],
        affects_theta=True,
        extra={
            "instruction": _DIR_INSTRUCTION[direction],
            "raw_sentence": tpl["sentence"],
        },
    )
