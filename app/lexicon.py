"""Словарь синонимов / сленга / аббревиатур -> группы эквивалентных фраз."""
from __future__ import annotations

from .text import tokens

Phrase = tuple[str, ...]


class Lexicon:
    def __init__(self, groups: list[list[str]], stemming: bool = True):
        # объединяем группы с общими фразами (union-find по фразам)
        parent: dict[Phrase, Phrase] = {}

        def find(x: Phrase) -> Phrase:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        surface: dict[Phrase, str] = {}
        for g in groups:
            ph = []
            for s in g:
                t = tuple(tokens(s, stemming))
                if t:
                    ph.append(t)
                    surface.setdefault(t, s)
                    parent.setdefault(t, t)
            for t in ph[1:]:
                parent[find(t)] = find(ph[0])

        clusters: dict[Phrase, list[Phrase]] = {}
        for t in parent:
            clusters.setdefault(find(t), []).append(t)

        self.group_of: dict[Phrase, int] = {}
        self.groups: list[list[Phrase]] = []
        for i, members in enumerate(clusters.values()):
            self.groups.append(sorted(members))
            for t in members:
                self.group_of[t] = i
        self.surface = surface
        self.max_len = max((len(t) for t in parent), default=1)
        self.stem_vocab = {s for t in parent for s in t}

    def find_phrases(self, stems: list[str]) -> list[tuple[int, int, int]]:
        """Жадный поиск самых длинных словарных фраз: [(start, end, group_id)]."""
        out, i = [], 0
        while i < len(stems):
            for n in range(min(self.max_len, len(stems) - i), 0, -1):
                t = tuple(stems[i:i + n])
                if t in self.group_of:
                    out.append((i, i + n, self.group_of[t]))
                    i += n
                    break
            else:
                i += 1
        return out

    def alternatives(self, group_id: int) -> list[Phrase]:
        return self.groups[group_id]

    def surface_of(self, phrase: Phrase) -> str:
        return self.surface.get(phrase, " ".join(phrase))
