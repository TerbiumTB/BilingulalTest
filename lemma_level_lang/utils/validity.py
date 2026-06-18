from pymorphy3 import MorphAnalyzer
import spacy


class LemmaValidator:
    def __init__(self, en_model: str = "en_core_web_sm"):
        self.en_model_name = en_model
        self._morph_ru = MorphAnalyzer()
        self._nlp_en = spacy.load(self.en_model_name, disable=["parser", "ner"])

    def is_lemma(self, word: str, lang: str) -> bool:
        word = word.strip().lower()
        if not word:
            return False
        if lang == "RU":
            return self._is_lemma_ru(word)
        if lang == "ENG":
            return self._is_lemma_en(word)

    def _is_lemma_ru(self, word: str) -> bool:
        parses = self.morph_ru.parse(word)
        return any(p.is_known and p.normal_form == word for p in parses)

    def _is_lemma_en(self, word: str) -> bool:
        if not word.isalpha():
            return False
        doc = self.nlp_en(word)
        if len(doc) != 1:
            return False
        token = doc[0]
        return token.is_alpha and token.lemma_.lower() == word

    def lemma_rate(self, words: list[str], lang: str) -> float:
        if not words:
            return 0.0
        return sum(self.is_lemma(w, lang) for w in words) / len(words)

    def filter_lemmas(self, words: list[str], lang: str) -> list[str]:
        return [w for w in words if self.is_lemma(w, lang)]
