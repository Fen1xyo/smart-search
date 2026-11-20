"""Бизнес-сигналы и reranking с relevance gate.

Правило: бизнес-выгода (популярность, маржа) и контекст клиента двигают товар
только внутри полосы «хороших» кандидатов (rel >= gate * rel_top). Нерелевантный
товар не может обогнать подходящий, сколько бы маржи у него ни было.
Отсутствующий сигнал = нейтральное значение, а не выдумка.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .config import SearchConfig
from .data_loader import Product

EVENT_WEIGHTS = {"click": 1.0, "add_to_cart": 3.0, "purchase": 5.0}


class BusinessSignals:
    def __init__(self, products: list[Product], events: list[dict], neutral: float = 0.5):
        self.neutral = neutral
        pops = [p.popularity for p in products if p.popularity is not None and p.popularity >= 0]
        margins = [p.margin for p in products if p.margin is not None]
        max_lp = math.log1p(max(pops)) if pops else 0.0
        mn, mx = (min(margins), max(margins)) if margins else (0.0, 0.0)
        self.pop: dict[str, float | None] = {}
        self.margin: dict[str, float | None] = {}
        for p in products:
            self.pop[p.id] = (math.log1p(p.popularity) / max_lp) if (p.popularity is not None and max_lp) else None
            if p.margin is None:
                self.margin[p.id] = None
            else:
                self.margin[p.id] = (p.margin - mn) / (mx - mn) if mx > mn else neutral

        # Агрегаты поведения по товару и сегменту (без привязки к тексту запроса -> нет утечки из оценки)
        self.seg_share: dict[str, dict[str, float]] = defaultdict(dict)
        by_prod: dict[str, Counter] = defaultdict(Counter)
        for e in events:
            if e.get("product_id") and e.get("segment"):
                by_prod[e["product_id"]][e["segment"]] += EVENT_WEIGHTS.get(e.get("event_type"), 1.0)
        for pid, c in by_prod.items():
            tot = sum(c.values())
            for seg, v in c.items():
                self.seg_share[pid][seg] = v / tot
        self.category = {p.id: p.category for p in products}
        self.vendor = {p.id: p.vendor for p in products}

    def business(self, pid: str, cfg: SearchConfig) -> tuple[float, dict]:
        pop, mar = self.pop.get(pid), self.margin.get(pid)
        missing = [n for n, v in (("popularity", pop), ("margin", mar)) if v is None]
        pop = self.neutral if pop is None else pop
        mar = self.neutral if mar is None else mar
        val = cfg.popularity_share * pop + cfg.margin_share * mar
        return val, {"popularity_norm": round(pop, 3), "margin_norm": round(mar, 3), "missing": missing}

    def context(self, pid: str, segment: str | None, history: list[str]) -> tuple[float, list[str]]:
        """0..1, 0.5 = нейтрально. Причины — для объяснения в выдаче."""
        reasons, parts = [], []
        if segment:
            share = self.seg_share.get(pid, {}).get(segment)
            if share is not None:
                parts.append(share)
                reasons.append(f"segment:{segment}={share:.2f}")
        if history:
            cats = {self.category.get(h) for h in history}
            vens = {self.vendor.get(h) for h in history}
            aff = 1.0 if self.category.get(pid) in cats else (0.75 if self.vendor.get(pid) in vens else 0.5)
            parts.append(aff)
            if aff > 0.5:
                reasons.append("history:" + ("category" if aff == 1.0 else "vendor"))
        return (sum(parts) / len(parts) if parts else self.neutral), reasons


def rerank(cands: list[dict], signals: BusinessSignals, cfg: SearchConfig,
           segment: str | None = None, history: list[str] | None = None) -> list[dict]:
    """cands: [{'pid', 'rel_norm', ...}] -> отсортированный список с полями объяснения."""
    history = history or []
    gated, rest = [], []
    for c in cands:
        biz, info = signals.business(c["pid"], cfg)
        ctx, ctx_reasons = signals.context(c["pid"], segment, history)
        c["business"] = info
        c["context_reasons"] = ctx_reasons
        boost = 0.0
        if cfg.use_business_boost and c["rel_norm"] >= cfg.relevance_gate:
            boost = cfg.business_weight * (biz - cfg.neutral_signal)
            if segment or history:
                boost += cfg.context_weight * (ctx - cfg.neutral_signal)
            c["gated"] = True
            gated.append(c)
        else:
            c["gated"] = False
            rest.append(c)
        c["boost"] = round(boost, 4)
        c["score"] = c["rel_norm"] + boost
    gated.sort(key=lambda c: (-c["score"], c["pid"]))
    rest.sort(key=lambda c: (-c["score"], c["pid"]))
    return gated + rest  # tier 1 (прошли gate) всегда выше tier 2
