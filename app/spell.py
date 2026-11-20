"""Исправление опечаток и неверной раскладки. Трогаем только неизвестные слова."""
from __future__ import annotations

from rapidfuzz import fuzz, process

from .text import STOPWORDS, swap_layout, tokens


class Speller:
    def __init__(self, surface_words: set[str], known_stems: set[str], stemming: bool,
                 min_len: int = 4, min_score: float = 82.0):
        self.vocab = sorted(w for w in surface_words if len(w) >= 3 and not w.isdigit())
        self.known = known_stems
        self.stemming = stemming
        self.min_len = min_len
        self.min_score = min_score

    def _stem(self, w: str) -> str:
        t = tokens(w, self.stemming)
        return t[0] if t else w

    def is_known(self, word: str) -> bool:
        return word in STOPWORDS or self._stem(word) in self.known

    def correct(self, word: str) -> tuple[str, str, float] | None:
        """-> (исправленное слово, тип 'layout'|'typo', уверенность) или None."""
        if self.is_known(word) or word.isdigit():
            return None
        swapped = swap_layout(word)
        if swapped != word and self.is_known(swapped) and swapped not in STOPWORDS:
            return swapped, "layout", 100.0
        if len(word) < self.min_len:
            return None
        hit = process.extractOne(word, self.vocab, scorer=fuzz.ratio, score_cutoff=self.min_score)
        if not hit:
            return None
        cand, score, _ = hit
        # опечатка в первой букве редка: требуем более высокого сходства
        if cand[0] != word[0] and score < 92:
            return None
        if abs(len(cand) - len(word)) > 2:
            return None
        return cand, "typo", float(score)
