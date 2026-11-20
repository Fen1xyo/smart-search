"""Журнал поисков/событий (JSONL) и аналитика, которая строится из журнала.

ПДн: не пишем IP, User-Agent, user_id, историю клиента. Текст запроса маскируется
(email, телефоны, длинные номера). Идентификатор сессии — только солёный хэш.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .text import mask_pii

ABANDON_WINDOW_S = 300  # «уход после неудачного поиска»: нет успешного поиска в сессии за 5 мин


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, int(round(p * len(v))) - 1))], 3)


class SearchLog:
    def __init__(self, log_dir: Path, salt: str | None = None):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.search_path = self.dir / "searches.jsonl"
        self.zero_path = self.dir / "zero_results.jsonl"
        self.event_path = self.dir / "events.jsonl"
        self.salt = salt or os.environ.get("IK3_LOG_SALT") or secrets.token_hex(8)
        self.lock = threading.Lock()
        self.searches: list[dict] = self._read(self.search_path)
        self.events: list[dict] = self._read(self.event_path)

    @staticmethod
    def _read(p: Path) -> list[dict]:
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _append(self, p: Path, rec: dict) -> None:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def anon(self, session: str | None) -> str | None:
        if not session:
            return None
        return hashlib.sha256((self.salt + session).encode()).hexdigest()[:12]

    def log_search(self, result: dict, session: str | None, k: int, cache_hit: bool) -> str:
        sid = uuid.uuid4().hex[:16]
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t": time.time(),
            "search_id": sid,
            "session": self.anon(session),
            "query": mask_pii(result["query"]),
            "normalized": mask_pii(result["normalized"]),
            "zero_result": result["zero_result"],
            "total_found": result["total_found"],
            "top_ids": [it["id"] for it in result["items"]],
            "corrections": [{"from": mask_pii(c["from"]), "to": c["to"], "kind": c["kind"]}
                            for c in result["corrections"]],
            "synonyms": [s["phrase"] for s in result["synonyms_applied"]],
            "unmatched_terms": [mask_pii(t) for t in result["unmatched_terms"]],
            "category": result["context"]["category"],
            "segment": result["context"]["segment"],
            "history_len": result["context"]["history_len"],
            "k": k,
            "latency_ms": result["latency_ms"],
            "cache_hit": cache_hit,
        }
        with self.lock:
            self.searches.append(rec)
            self._append(self.search_path, rec)
            if rec["zero_result"]:
                self._append(self.zero_path, rec)
        return sid

    def log_event(self, search_id: str | None, product_id: str, event_type: str,
                  position: int | None, session: str | None) -> dict:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "t": time.time(),
               "search_id": search_id, "product_id": product_id, "event_type": event_type,
               "position": position, "session": self.anon(session)}
        with self.lock:
            self.events.append(rec)
            self._append(self.event_path, rec)
        return rec

    def reset(self) -> None:
        with self.lock:
            self.searches, self.events = [], []
            for p in (self.search_path, self.zero_path, self.event_path):
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------------ аналитика
    def analytics(self, imported_events: list[dict] | None = None, top: int = 15) -> dict:
        with self.lock:
            S, E = list(self.searches), list(self.events)
        n = len(S)
        zero = [s for s in S if s["zero_result"]]
        lat = [s["latency_ms"] for s in S if not s.get("cache_hit")]
        q_counter = Counter(s["normalized"] for s in S)
        zero_counter = Counter(s["normalized"] for s in zero)
        last_zero = {}
        for s in zero:
            last_zero[s["normalized"]] = s
        unmet = Counter(t for s in S for t in s["unmatched_terms"])

        # уходы после неудачного поиска (proxy, по анонимной сессии)
        by_sess = defaultdict(list)
        for s in S:
            if s.get("session"):
                by_sess[s["session"]].append(s)
        abandoned = recovered = 0
        for ss in by_sess.values():
            ss.sort(key=lambda r: r["t"])
            for i, s in enumerate(ss):
                if not s["zero_result"]:
                    continue
                later = [x for x in ss[i + 1:] if x["t"] - s["t"] <= ABANDON_WINDOW_S]
                if any(not x["zero_result"] for x in later):
                    recovered += 1
                else:
                    abandoned += 1

        # воронка поиск -> клик -> корзина -> покупка (по search_id)
        ev_by_type = defaultdict(set)
        for e in E:
            if e.get("search_id"):
                ev_by_type[e["event_type"]].add(e["search_id"])
        clicked_products = Counter(e["product_id"] for e in E if e["event_type"] == "click")
        positions = [e["position"] for e in E if e["event_type"] == "click" and e.get("position") is not None]

        res = {
            "totals": {
                "searches": n,
                "unique_queries": len(q_counter),
                "zero_result_searches": len(zero),
                "zero_result_rate": round(len(zero) / n, 4) if n else None,
                "with_corrections_rate": round(sum(bool(s["corrections"]) for s in S) / n, 4) if n else None,
                "with_synonyms_rate": round(sum(bool(s["synonyms"]) for s in S) / n, 4) if n else None,
                "latency_p50_ms": _pct(lat, 0.5),
                "latency_p95_ms": _pct(lat, 0.95),
                "cache_hits": sum(bool(s.get("cache_hit")) for s in S),
            },
            "funnel": {
                "searches": n,
                "with_click": len(ev_by_type["click"]),
                "with_add_to_cart": len(ev_by_type["add_to_cart"]),
                "with_purchase": len(ev_by_type["purchase"]),
                "ctr": round(len(ev_by_type["click"]) / n, 4) if n else None,
                "search_to_order": round(len(ev_by_type["purchase"]) / n, 4) if n else None,
                "mean_click_position": round(sum(positions) / len(positions), 2) if positions else None,
            },
            "failed_search_outcome": {
                "zero_then_abandoned": abandoned,
                "zero_then_recovered": recovered,
                "note": "proxy по анонимной сессии: после пустой выдачи за 5 мин не было успешного поиска",
            },
            "top_queries": [{"query": q, "count": c, "zero_result": q in zero_counter}
                            for q, c in q_counter.most_common(top)],
            "zero_result_queries": [{"query": q, "count": c, "unmatched_terms": last_zero[q]["unmatched_terms"],
                                     "last_seen": last_zero[q]["ts"]} for q, c in zero_counter.most_common(top)],
            "unmet_demand_terms": [{"term": t, "count": c} for t, c in unmet.most_common(top)],
            "top_clicked_products": [{"product_id": p, "clicks": c} for p, c in clicked_products.most_common(top)],
        }
        if imported_events:
            by_t = Counter(e["event_type"] for e in imported_events)
            by_seg = Counter(e["segment"] for e in imported_events if e.get("segment"))
            by_p = Counter(e["product_id"] for e in imported_events if e["event_type"] == "purchase")
            res["imported_behavior"] = {
                "label": "SYNTHETIC — случайные события, не привязаны к запросам; только проверка импорта",
                "events": len(imported_events), "by_type": dict(by_t), "by_segment": dict(by_seg),
                "top_purchased": [{"product_id": p, "purchases": c} for p, c in by_p.most_common(5)],
            }
        return res
