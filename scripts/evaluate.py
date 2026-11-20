"""Оценка качества и ablation на размеченных запросах.

Запуск:  python scripts/evaluate.py            (все ablation, dev/test/all)
         python scripts/evaluate.py --sweep    (+ развёртка порога coverage на dev)

Метрики (только для запросов, где есть релевантные товары):
  Recall@10 = |relevant ∩ top10| / |relevant|
  MRR@10    = 1 / ранг первого релевантного (0, если нет в top10)
  nDCG@10   = бинарная релевантность, DCG = Σ 1/log2(rank+1), нормировка на идеальный DCG
Для запросов с пустым ожидаемым ответом (kind=zero_result):
  zero_detection = доля, где система вернула пустую выдачу (правильный «нет в каталоге»)
Доли нулевой выдачи:
  false_zero_rate = доля обычных запросов, где выдача пустая (плохо)
  zero_rate_all   = доля всех запросов с пустой выдачей
Latency: p50/p95 по N прогонам каждого запроса (тёплый процесс, без кэша).

Split: нечётные номера Q -> dev (на них настраивали), чётные -> test.
ВНИМАНИЕ: test-часть видели при чтении файла, это не слепой holdout.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG, SearchConfig  # noqa: E402
from app.data_loader import load_catalog, load_eval, load_events, load_synonym_groups  # noqa: E402
from app import config as C  # noqa: E402
from app.engine import SearchEngine  # noqa: E402

K = 10


def split_of(qid: str) -> str:
    num = int("".join(ch for ch in qid if ch.isdigit()) or 0)
    return "dev" if num % 2 == 1 else "test"


def ablation_configs() -> list[tuple[str, SearchConfig]]:
    off = dict(use_stemming=False, use_synonyms=False, use_team_lexicon=False, use_typo=False,
               use_negation=False, use_generic_terms=False, use_name_coverage=False,
               use_coverage_gate=False, use_business_boost=False, use_compound=False)
    steps = [
        ("A0 BM25 по полям (без морфологии)", {}),
        ("A1 + стемминг RU/EN", {"use_stemming": True}),
        ("A2 + словарь синонимов/сленга", {"use_synonyms": True}),
        ("A3 + лексикон команды (аббревиатуры)", {"use_team_lexicon": True}),
        ("A4 + опечатки/раскладка", {"use_typo": True}),
        ("A5 + отрицания, родовые слова, покрытие названия",
         {"use_negation": True, "use_generic_terms": True, "use_name_coverage": True}),
        ("A6 + coverage gate (детект «нет в каталоге»)", {"use_coverage_gate": True}),
        ("A7 + бизнес-буст с relevance gate (= v1)", {"use_business_boost": True}),
        ("A8 + составные слова (post-test fix)", {"use_compound": True}),
    ]
    out, cur = [], dict(off)
    for name, upd in steps:
        cur.update(upd)
        out.append((name, DEFAULT_CONFIG.with_(**cur)))
    return out


def evaluate(engine: SearchEngine, queries: list[dict], latency_runs: int = 0) -> dict:
    per_q, lat = [], []
    for q in queries:
        r = engine.search(q["query"], k=K)
        ids = [it["id"] for it in r["items"]]
        rel = set(q["relevant_product_ids"])
        row = {"query_id": q["query_id"], "query": q["query"], "split": split_of(q["query_id"]),
               "expected": sorted(rel), "got": ids[:5], "zero": r["zero_result"]}
        if rel:
            hits = [i for i, pid in enumerate(ids[:K]) if pid in rel]
            row["recall"] = len({ids[i] for i in hits}) / len(rel)
            row["rr"] = 1 / (hits[0] + 1) if hits else 0.0
            dcg = sum(1 / math.log2(i + 2) for i in hits)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), K)))
            row["ndcg"] = dcg / idcg
            row["ok"] = bool(hits) and hits[0] == 0
        else:
            row["ok"] = r["zero_result"]
        per_q.append(row)
        for _ in range(latency_runs):
            lat.append(engine.search(q["query"], k=K)["latency_ms"])

    def agg(rows):
        s = [r for r in rows if r["expected"]]
        z = [r for r in rows if not r["expected"]]
        m = {
            "n_search": len(s), "n_zero": len(z),
            "recall@10": statistics.mean(r["recall"] for r in s) if s else None,
            "mrr@10": statistics.mean(r["rr"] for r in s) if s else None,
            "ndcg@10": statistics.mean(r["ndcg"] for r in s) if s else None,
            "false_zero_rate": (sum(r["zero"] for r in s) / len(s)) if s else None,
            "zero_detection": (sum(r["zero"] for r in z) / len(z)) if z else None,
            "zero_rate_all": sum(r["zero"] for r in rows) / len(rows) if rows else None,
        }
        return m

    res = {split: agg([r for r in per_q if split == "all" or r["split"] == split])
           for split in ("dev", "test", "all")}
    if lat:
        lat.sort()
        res["latency_ms"] = {"p50": round(lat[len(lat) // 2], 3),
                             "p95": round(lat[int(len(lat) * 0.95) - 1], 3),
                             "max": round(lat[-1], 3), "n": len(lat)}
    res["per_query"] = per_q
    return res


def build(cfg: SearchConfig, data) -> SearchEngine:
    products, dict_groups, team_groups, events = data
    return SearchEngine(products, dict_groups, team_groups, events, cfg)


def boost_audit(data, queries) -> dict:
    """Сравнение с/без бизнес-буста: куда сдвинулся первый релевантный, были ли «вредные» подъёмы."""
    with_b = build(DEFAULT_CONFIG, data)
    no_b = build(DEFAULT_CONFIG.with_(use_business_boost=False), data)
    moved, harmful, changed_top1 = [], [], 0
    for q in queries:
        a, b = with_b.search(q["query"]), no_b.search(q["query"])
        ida, idb = [i["id"] for i in a["items"]], [i["id"] for i in b["items"]]
        if ida[:1] != idb[:1]:
            changed_top1 += 1
        rel = set(q["relevant_product_ids"])
        ra = next((i for i, p in enumerate(ida) if p in rel), None)
        rb = next((i for i, p in enumerate(idb) if p in rel), None)
        if ra != rb:
            moved.append({"query": q["query"], "rank_without": rb, "rank_with": ra})
        # «вредный» подъём: товар с rel_norm < gate оказался выше товара с rel_norm >= gate
        items = a["items"]
        for i, it in enumerate(items):
            if not it["passed_relevance_gate"] and any(x["passed_relevance_gate"] for x in items[i + 1:]):
                harmful.append(q["query"])
                break
    return {"queries_top1_changed": changed_top1, "relevant_rank_changes": moved,
            "harmful_promotions": harmful}


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--latency-runs", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    data = (load_catalog(), load_synonym_groups(C.SYNONYMS_PATHS[:1]),
            load_synonym_groups(C.SYNONYMS_PATHS[1:]), load_events())
    queries = load_eval()
    out = {"note": "SYNTHETIC: 30 товаров, 36 запросов, разметка командой. Не показатель качества на реальном каталоге.",
           "ablation": []}
    lines = ["# Результаты оценки (синтетика, 30 товаров / 36 запросов)", "",
             "Split: dev = нечётные Q (16 обычных + 2 «нет в каталоге»), test = чётные (16 + 2). "
             "Test-часть видели при чтении файла — это не слепой holdout.", "",
             "| Конфигурация | split | Recall@10 | MRR@10 | nDCG@10 | false zero | zero detect | zero rate (все) |",
             "|---|---|---|---|---|---|---|---|"]
    final = None
    for name, cfg in ablation_configs():
        r = evaluate(build(cfg, data), queries, latency_runs=0)
        out["ablation"].append({"name": name, **{s: r[s] for s in ("dev", "test", "all")}})
        for s in ("dev", "test", "all"):
            m = r[s]
            lines.append(f"| {name} | {s} | {fmt(m['recall@10'])} | {fmt(m['mrr@10'])} | {fmt(m['ndcg@10'])} | "
                         f"{fmt(m['false_zero_rate'])} | {fmt(m['zero_detection'])} | {fmt(m['zero_rate_all'])} |")
    final = evaluate(build(DEFAULT_CONFIG, data), queries, latency_runs=args.latency_runs)
    out["final"] = {k: v for k, v in final.items() if k != "per_query"}
    out["final_per_query"] = final["per_query"]
    lat = final["latency_ms"]
    lines += ["", f"**Latency итоговой конфигурации** (in-process, {lat['n']} вызовов): "
              f"p50 = {lat['p50']} мс, p95 = {lat['p95']} мс, max = {lat['max']} мс.", ""]

    audit = boost_audit(data, queries)
    out["boost_audit"] = audit
    lines += ["## Аудит бизнес-буста (итоговая конфигурация с/без буста)", "",
              f"- запросов, где сменился top-1: {audit['queries_top1_changed']}",
              f"- сдвиги ранга релевантного товара: {audit['relevant_rank_changes'] or 'нет'}",
              f"- «вредные» подъёмы (товар ниже relevance gate выше прошедшего gate): "
              f"{audit['harmful_promotions'] or 'нет'}", ""]

    if args.sweep:
        lines += ["## Развёртка порога coverage (только dev)", "",
                  "| порог | Recall@10 | MRR@10 | false zero | zero detect |", "|---|---|---|---|---|"]
        out["sweep_dev"] = []
        for thr in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            r = evaluate(build(DEFAULT_CONFIG.with_(coverage_threshold=thr), data),
                         [q for q in queries if split_of(q["query_id"]) == "dev"])["dev"]
            out["sweep_dev"].append({"threshold": thr, **r})
            lines.append(f"| {thr} | {fmt(r['recall@10'])} | {fmt(r['mrr@10'])} | "
                         f"{fmt(r['false_zero_rate'])} | {fmt(r['zero_detection'])} |")
        lines.append("")

    lines += ["## Ошибки итоговой конфигурации", ""]
    bad = [r for r in final["per_query"] if not r["ok"]]
    if not bad:
        lines.append("Нет: для всех запросов релевантный товар на 1-м месте, «нет в каталоге» — пустая выдача.")
    for r in bad:
        lines.append(f"- [{r['split']}] {r['query_id']} «{r['query']}»: ожидали {r['expected'] or 'пусто'}, "
                     f"получили {r['got'] or 'пусто'}")
    Path(args.out).mkdir(exist_ok=True)
    Path(args.out, "eval_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out, "eval_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
