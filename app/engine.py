"""Поисковый конвейер: запрос -> концепты -> кандидаты -> гейты -> reranking -> объяснение."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config as C
from .config import SearchConfig
from .data_loader import Product, load_catalog, load_events, load_synonym_groups
from .index import FieldBM25, product_fields
from .lexicon import Lexicon
from .ranking import BusinessSignals, rerank
from .spell import Speller
from .text import GENERIC_TERMS, NEGATIONS, STOPWORDS, normalize, tokens, words


@dataclass
class Alt:
    phrase: tuple[str, ...]
    weight: float
    source: str          # original | typo | layout | synonym
    surface: str


@dataclass
class Concept:
    text: str
    alts: list[Alt]
    negative: bool = False
    weight: float = 1.0  # IDF-вес для coverage
    doc_freq: int = 0
    fields_by_doc: dict = field(default_factory=dict)


class SearchEngine:
    def __init__(self, products: list[Product], dict_groups: list[list[str]],
                 team_groups: list[list[str]] | None = None, events: list[dict] | None = None,
                 cfg: SearchConfig = C.DEFAULT_CONFIG):
        self.cfg = cfg
        self.products = products
        self.by_id = {p.id: p for p in products}
        self.pos_of = {p.id: i for i, p in enumerate(products)}
        st = cfg.use_stemming
        self.index = FieldBM25(products, cfg.field_weights, st, cfg.bm25_k1, cfg.bm25_b)
        groups = []
        if cfg.use_synonyms:
            groups += dict_groups
            if cfg.use_team_lexicon:
                groups += team_groups or []
        self.lexicon = Lexicon(groups, st)
        surface = {w for p in products for t in product_fields(p).values() for w in words(t)}
        surface |= {w for g in groups for s in g for w in words(s)}
        self.speller = Speller(surface, self.index.vocab | self.lexicon.stem_vocab, st,
                               cfg.typo_min_len, cfg.typo_min_score)
        self.exact_name = {" ".join(words(p.name)): p.id for p in products}
        self.exact_id = {p.id.lower(): p.id for p in products}
        # нечисловые токены названия: «Орбита Офис 50» -> {орбит, офис}
        self.name_stems = {p.id: {t for t in tokens(p.name, st) if not t.isdigit()} for p in products}
        self.signals = BusinessSignals(products, events or [], cfg.neutral_signal)
        self.categories = sorted({p.category for p in products if p.category})

    @classmethod
    def from_files(cls, cfg: SearchConfig = C.DEFAULT_CONFIG) -> "SearchEngine":
        dict_groups = load_synonym_groups(C.SYNONYMS_PATHS[:1])
        team_groups = load_synonym_groups(C.SYNONYMS_PATHS[1:])
        return cls(load_catalog(), dict_groups, team_groups, load_events(), cfg)

    # ------------------------------------------------------------------ разбор запроса
    def parse(self, q: str) -> dict:
        cfg = self.cfg
        ws = words(q)
        corrections, items = [], []  # items: (word, negative, source)
        negate_next = False
        for w in ws:
            if cfg.use_negation and w in NEGATIONS:
                negate_next = True
                continue
            if w in STOPWORDS:
                continue
            src = "original"
            if cfg.use_typo:
                fix = self.speller.correct(w)
                if fix:
                    corrections.append({"from": w, "to": fix[0], "kind": fix[1], "score": round(fix[2], 1)})
                    w, src = fix[0], fix[1]
            items.append((w, negate_next, src))
            negate_next = False

        stems = [tokens(w, cfg.use_stemming)[0] if tokens(w, cfg.use_stemming) else w for w, _, _ in items]
        spans = self.lexicon.find_phrases(stems) if cfg.use_synonyms else []
        span_at = {s: (s, e, g) for s, e, g in spans}
        concepts: list[Concept] = []
        synonyms_applied = []
        i = 0
        while i < len(items):
            if i in span_at:
                s, e, g = span_at[i]
                phrase = tuple(stems[s:e])
                src = "original" if all(items[j][2] == "original" for j in range(s, e)) else items[s][2]
                w0 = 1.0 if src == "original" else cfg.typo_weight
                alts = [Alt(phrase, w0, src, " ".join(items[j][0] for j in range(s, e)))]
                others = [ph for ph in self.lexicon.alternatives(g) if ph != phrase]
                alts += [Alt(ph, cfg.synonym_weight, "synonym", self.lexicon.surface_of(ph)) for ph in others]
                text = alts[0].surface
                synonyms_applied.append({"phrase": text, "expanded_to": [a.surface for a in alts[1:]]})
                concepts.append(Concept(text, alts, negative=any(items[j][1] for j in range(s, e))))
                i = e
            else:
                w, neg, src = items[i]
                w0 = 1.0 if src == "original" else cfg.typo_weight
                concepts.append(Concept(w, [Alt((stems[i],), w0, src, w)], negative=neg))
                i += 1

        # Не засчитываем концепт через синоним, который пользователь сам написал другим словом
        # («crm для продаж»: «продаж» не должен совпадать через «crm» — это двойной счёт).
        typed = {c.alts[0].phrase for c in concepts}
        for c in concepts:
            c.alts = [c.alts[0]] + [a for a in c.alts[1:] if a.phrase not in typed]

        for c in concepts:
            orig = c.alts[0].phrase
            c.weight = sum(self.index.idf(t) for t in orig) / len(orig)
            if cfg.use_generic_terms and all(t in GENERIC_TERMS for t in orig):
                c.weight *= cfg.generic_term_factor
        return {"words": ws, "corrections": corrections, "concepts": concepts,
                "synonyms_applied": synonyms_applied}

    # ------------------------------------------------------------------ поиск
    def search(self, q: str, k: int | None = None, category: str | None = None,
               segment: str | None = None, history: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        cfg = self.cfg
        k = k or cfg.default_k
        q = (q or "").strip()[:300]
        parsed = self.parse(q)
        concepts: list[Concept] = parsed["concepts"]
        pos = [c for c in concepts if not c.negative]
        neg = [c for c in concepts if c.negative]

        # точное совпадение ID / названия
        norm_q = " ".join(parsed["words"])
        exact = set()
        if norm_q in self.exact_name:
            exact.add(self.exact_name[norm_q])
        for w in parsed["words"]:
            if w in self.exact_id:
                exact.add(self.exact_id[w])

        # кандидаты
        cand_docs: set[int] = set()
        for c in pos:
            docs_c = set()
            for a in c.alts:
                for t in a.phrase:
                    docs_c.update(self.index.docs_with(t))
            cand_docs |= docs_c
        for pid in exact:
            cand_docs.add(self.pos_of[pid])

        total_w = sum(c.weight for c in pos) or 1.0
        scored = []
        concept_hit = {id(c): 0 for c in pos}
        for d in cand_docs:
            rel, cov_w, matches = 0.0, 0.0, []
            for c in pos:
                best, best_cov, best_m = 0.0, 0.0, None
                for a in c.alts:
                    parts = [self.index.term_score(d, t) for t in a.phrase]
                    present = [p for p in parts if p[0] > 0]
                    if not present:
                        continue
                    full = len(present) == len(parts)
                    if not full and a.source == "synonym":
                        continue  # синоним-фраза засчитывается только целиком
                    frac = len(present) / len(parts)
                    s = a.weight * sum(p[0] for p in present) * (1.0 if full else 0.5)
                    if s > best:
                        flds = sorted({f for p in present for f in p[1]})
                        best, best_cov = s, frac
                        best_m = {"term": c.text, "via": a.source, "matched": a.surface, "fields": flds}
                if best_m:
                    rel += best
                    cov_w += c.weight * best_cov
                    matches.append(best_m)
                    concept_hit[id(c)] += 1
            coverage = cov_w / total_w if pos else 0.0
            penalized = []
            for c in neg:
                if any(all(self.index.term_score(d, t)[0] > 0 for t in a.phrase) for a in c.alts):
                    rel *= cfg.negation_penalty
                    penalized.append(c.text)
            pid = self.products[d].id
            is_exact = pid in exact
            if rel <= 0 and not is_exact:
                continue
            if cfg.use_coverage_gate and coverage < cfg.coverage_threshold and not is_exact:
                continue
            scored.append({"pid": pid, "rel": rel, "coverage": coverage, "matches": matches,
                           "exact": is_exact, "negated": penalized})

        filtered_out = 0
        if category:
            before = len(scored)
            scored = [s for s in scored if self.by_id[s["pid"]].category.lower() == category.lower()]
            filtered_out = before - len(scored)

        top = max((s["rel"] for s in scored if not s["exact"]), default=0.0) or 1.0
        q_stems = {t for c in pos for a in c.alts for t in a.phrase}
        for s in scored:
            s["rel_norm"] = min(s["rel"] / top, 1.0) + (cfg.exact_name_bonus if s["exact"] else 0.0)
            ns = self.name_stems.get(s["pid"])
            s["name_covered"] = bool(cfg.use_name_coverage and ns and ns <= q_stems)
            if s["name_covered"]:
                s["rel_norm"] += cfg.name_coverage_bonus
        ranked = rerank(scored, self.signals, cfg, segment, history or [])

        items = [self._item(s, concepts) for s in ranked[:k]]
        unmatched = [c.text for c in pos if concept_hit[id(c)] == 0]
        latency = (time.perf_counter() - t0) * 1000
        did_you_mean = None
        if parsed["corrections"]:
            fixed = q.lower()
            for corr in parsed["corrections"]:
                fixed = fixed.replace(corr["from"], corr["to"])
            did_you_mean = fixed
        return {
            "query": q,
            "normalized": norm_q,
            "corrections": parsed["corrections"],
            "did_you_mean": did_you_mean,
            "synonyms_applied": parsed["synonyms_applied"],
            "negations": [c.text for c in neg],
            "unmatched_terms": unmatched,
            "context": {"category": category, "segment": segment, "history_len": len(history or []),
                        "filtered_out_by_category": filtered_out},
            "total_found": len(ranked),
            "zero_result": len(ranked) == 0,
            "items": items,
            "related_queries": self.related(parsed, items),
            "latency_ms": round(latency, 3),
            "data_notice": "SYNTHETIC" if any(p.is_mock for p in self.products) else None,
        }

    def _item(self, s: dict, concepts: list[Concept]) -> dict:
        p = self.by_id[s["pid"]]
        stems_q = {t for c in concepts if not c.negative for a in c.alts for t in a.phrase}
        confirmed = [a for a in p.attributes if set(tokens(a, self.cfg.use_stemming)) & stems_q]
        fields = sorted({f for m in s["matches"] for f in m["fields"]})
        return {
            **p.public(),
            "score": round(s["score"], 4),
            "relevance": round(s["rel"], 4),
            "relevance_norm": round(min(s["rel_norm"], 1.0), 4),
            "coverage": round(s["coverage"], 3),
            "exact_match": s["exact"],
            "name_fully_matched": s.get("name_covered", False),
            "passed_relevance_gate": s["gated"],
            "business_boost": s["boost"],
            "business": s["business"],
            "context_reasons": s["context_reasons"],
            "matched": s["matches"],
            "matched_fields": fields,
            "confirmed_attributes": confirmed,
            "negated": s["negated"],
        }

    def related(self, parsed: dict, items: list[dict]) -> list[str]:
        out: list[str] = []
        for syn in parsed["synonyms_applied"]:
            out += syn["expanded_to"][:2]
        for it in items[:3]:
            if it["category"] and it["category"] not in out:
                out.append(it["category"])
        if not items:  # для пустой выдачи — категории по частично совпавшим словам
            for c in parsed["concepts"]:
                for t in c.alts[0].phrase:
                    for d in self.index.docs_with(t)[:2]:
                        cat = self.products[d].category
                        if cat not in out:
                            out.append(cat)
        seen, res = set(), []
        for r in out:
            if normalize(r) not in seen and r:
                seen.add(normalize(r))
                res.append(r)
        return res[:6]
