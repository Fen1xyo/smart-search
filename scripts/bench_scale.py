"""Нагрузочная оценка latency на искусственно увеличенном каталоге.

Генерирует N товаров, перемешивая слова синтетического каталога (это НЕ реалистичный
каталог: оценка порядка величины, а не обещание). Запуск:
    python scripts/bench_scale.py --sizes 1000 10000 50000
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config as C  # noqa: E402
from app.data_loader import Product, load_catalog, load_eval, load_events, load_synonym_groups  # noqa: E402
from app.engine import SearchEngine  # noqa: E402
from app.text import words  # noqa: E402


def synth_catalog(base: list[Product], n: int, seed: int = 7) -> list[Product]:
    rnd = random.Random(seed)
    vocab = sorted({w for p in base for w in words(p.description) if len(w) > 3})
    out = list(base)
    for i in range(n - len(base)):
        b = rnd.choice(base)
        out.append(Product(
            id=f"G{i:06d}", name=f"{b.vendor}{i % 97} {rnd.choice(vocab).capitalize()} {i % 13}",
            vendor=f"{b.vendor}{i % 97}", category=b.category,
            description=" ".join(rnd.sample(vocab, 8)) + ". " + b.description,
            attributes=rnd.sample(b.attributes, k=max(1, len(b.attributes) - 1)),
            popularity=rnd.randint(0, 500), margin=round(rnd.uniform(0.1, 0.4), 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    base = load_catalog()
    dg, tg = load_synonym_groups(C.SYNONYMS_PATHS[:1]), load_synonym_groups(C.SYNONYMS_PATHS[1:])
    queries = [q["query"] for q in load_eval()]
    print("| товаров | индексация, с | память индекса, МБ | p50, мс | p95, мс | max, мс |")
    print("|---|---|---|---|---|---|")
    for n in args.sizes:
        cat = synth_catalog(base, n)
        tracemalloc.start()
        t0 = time.perf_counter()
        eng = SearchEngine(cat, dg, tg, load_events(), C.DEFAULT_CONFIG)
        build = time.perf_counter() - t0
        mem = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        lat = []
        for _ in range(args.runs):
            for q in queries:
                lat.append(eng.search(q)["latency_ms"])
        lat.sort()
        print(f"| {n:,} | {build:.1f} | {mem:.0f} | {statistics.median(lat):.1f} | "
              f"{lat[int(len(lat) * 0.95) - 1]:.1f} | {lat[-1]:.1f} |".replace(",", " "))


if __name__ == "__main__":
    main()
