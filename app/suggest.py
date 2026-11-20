"""Автодополнение: названия, вендоры, категории, словарные фразы и успешные запросы из журнала."""
from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz, process

from .engine import SearchEngine
from .store import SearchLog
from .text import normalize, words


class Suggester:
    def __init__(self, engine: SearchEngine, log: SearchLog | None = None):
        self.engine = engine
        self.log = log
        base: dict[str, tuple[str, float]] = {}
        pop = engine.signals.pop
        for p in engine.products:
            base[normalize(p.name)] = ("product", 1.0 + (pop.get(p.id) or 0.0))
            base.setdefault(normalize(p.vendor), ("vendor", 0.8))
            base.setdefault(normalize(p.category), ("category", 0.9))
        for ph, s in engine.lexicon.surface.items():
            base.setdefault(normalize(s), ("dictionary", 0.5))
        base.pop("", None)
        self.base = base

    def suggest(self, q: str, limit: int = 8) -> list[dict]:
        nq = " ".join(words(q))
        if not nq:
            return []
        cands: dict[str, tuple[str, float]] = dict(self.base)
        if self.log:  # только запросы, которые находили товары
            freq = Counter(s["normalized"] for s in self.log.searches if not s["zero_result"])
            for text, c in freq.items():
                cands[text] = ("history", 1.0 + min(c, 50) / 10)
        out = []
        for text, (kind, w) in cands.items():
            if text.startswith(nq):
                out.append((2.0 + w, text, kind))
            elif any(t.startswith(nq) for t in text.split()) or (" " in nq and nq in text):
                out.append((1.0 + w, text, kind))
        if len(out) < 3 and len(nq) >= 3:  # терпимость к опечаткам
            for text, score, _ in process.extract(nq, list(cands), scorer=fuzz.partial_ratio, limit=limit):
                if score >= 80 and all(text != o[1] for o in out):
                    kind, w = cands[text]
                    out.append((score / 100 + w * 0.1, text, kind + "~fuzzy"))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [{"text": t, "type": k} for _, t, k in out[:limit]]
