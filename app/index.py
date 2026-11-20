"""BM25 по полям товара (BM25F-подобная сумма с весами полей)."""
from __future__ import annotations

import math
from collections import Counter

from .data_loader import Product
from .text import tokens

FIELDS = ("name", "vendor", "category", "attributes", "description")


def product_fields(p: Product) -> dict[str, str]:
    return {
        "name": p.name, "vendor": p.vendor, "category": p.category,
        "attributes": " , ".join(p.attributes), "description": p.description,
    }


class FieldBM25:
    def __init__(self, products: list[Product], field_weights: dict[str, float],
                 stemming: bool = True, k1: float = 1.2, b: float = 0.75):
        self.products = products
        self.fw = field_weights
        self.k1, self.b = k1, b
        self.tf: list[dict[str, Counter]] = []
        self.len: list[dict[str, int]] = []
        df: Counter = Counter()
        self.postings: dict[str, list[int]] = {}
        for i, p in enumerate(products):
            per_field = {f: Counter(tokens(t, stemming)) for f, t in product_fields(p).items()}
            self.tf.append(per_field)
            self.len.append({f: sum(c.values()) for f, c in per_field.items()})
            terms = {t for c in per_field.values() for t in c}
            df.update(terms)
            for t in terms:
                self.postings.setdefault(t, []).append(i)
        n = max(len(products), 1)
        self.avg = {f: (sum(l[f] for l in self.len) / n) or 1.0 for f in FIELDS}
        self.N = len(products)
        self.df = df
        self.vocab = set(df)

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1 + (self.N - d + 0.5) / (d + 0.5))

    def term_score(self, doc: int, term: str) -> tuple[float, list[str]]:
        """Вклад одного термина в документ + поля, где он найден."""
        s, fields = 0.0, []
        for f in FIELDS:
            tf = self.tf[doc][f].get(term, 0)
            if not tf:
                continue
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[doc][f] / self.avg[f]))
            s += self.fw.get(f, 1.0) * norm
            fields.append(f)
        return (self.idf(term) * s if s else 0.0), fields

    def docs_with(self, term: str) -> list[int]:
        return self.postings.get(term, [])
