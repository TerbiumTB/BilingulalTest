from __future__ import annotations

import random
from dataclasses import dataclass, field

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


SENTENCE_GAP_BANK: list[dict] = [
    {
        "level": "A1",
        "sentence": "Мой младший брат каждый день идёт в ___ (школу).",
        "answer": "school",
        "distractors": ["house", "water", "green", "run"],
    },
    {
        "level": "A1",
        "sentence": "Утром я люблю пить горячий ___ (кофе).",
        "answer": "coffee",
        "distractors": ["table", "window", "happy", "fast"],
    },
    {
        "level": "A2",
        "sentence": "На выходных мы поедем на ___ (море).",
        "answer": "sea",
        "distractors": ["book", "chair", "loud", "buy"],
    },
    {
        "level": "A2",
        "sentence": "Она забыла дома свой ___ (зонт), а на улице дождь.",
        "answer": "umbrella",
        "distractors": ["bridge", "pencil", "garden", "silent"],
    },
    {
        "level": "B1",
        "sentence": "Чтобы получить эту работу, нужен опыт и хорошее ___ (образование).",
        "answer": "education",
        "distractors": ["furniture", "weather", "distance", "courage"],
    },
    {
        "level": "B1",
        "sentence": "Он принял трудное ___ (решение) и уволился.",
        "answer": "decision",
        "distractors": ["invitation", "celebration", "direction", "tradition"],
    },
    {
        "level": "B2",
        "sentence": "Её аргументы были вполне ___ (убедительными).",
        "answer": "convincing",
        "distractors": ["confusing", "annoying", "demanding", "surrounding"],
    },
    {
        "level": "B2",
        "sentence": "Компания решила ___ (расширить) своё присутствие на рынке.",
        "answer": "expand",
        "distractors": ["expend", "expel", "expose", "expire"],
    },
    {
        "level": "C1",
        "sentence": "Его замечание было довольно ___ (язвительным).",
        "answer": "sarcastic",
        "distractors": ["spacious", "strategic", "systematic", "sympathetic"],
    },
    {
        "level": "C2",
        "sentence": "Этот политик известен своей ___ (изворотливостью).",
        "answer": "slipperiness",
        "distractors": ["sloppiness", "stubbornness", "sluggishness", "shrewdness"],
    },
]


def _gap_pick(level: str, avoid: set[str] | None = None) -> dict:
    avoid = avoid or set()
    target = config.LEVEL2IDX[level]
    order = sorted(
        SENTENCE_GAP_BANK,
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
    tpl = _gap_pick(level, avoid)
    options = [tpl["answer"]] + list(tpl["distractors"])
    random.shuffle(options)
    return Question(
        question_id=_new_id(),
        type=TYPE_SENTENCE_GAP,
        lang="EN",
        level=tpl["level"],
        prompt=tpl["sentence"].replace("___", "______"),
        answer_kind=ANSWER_CHOICE,
        options=options,
        correct_answer=tpl["answer"],
        affects_theta=True,
        extra={
            "instruction": "Выберите английское слово, подходящее на место пропуска",
            "raw_sentence": tpl["sentence"],
        },
    )
