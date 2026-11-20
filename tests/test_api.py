"""API, журнал, аналитика, ПДн."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["products"] == 30 and h["synthetic_data"] is True


def test_search_contract(client):
    r = client.get("/search", params={"q": "антивирс для пк", "k": 3}).json()
    for key in ("items", "corrections", "synonyms_applied", "latency_ms", "zero_result", "search_id",
                "unmatched_terms", "related_queries", "did_you_mean"):
        assert key in r
    it = r["items"][0]
    assert it["id"] == "P003"
    for key in ("score", "relevance", "coverage", "business", "matched", "matched_fields",
                "confirmed_attributes", "passed_relevance_gate", "business_boost"):
        assert key in it
    assert len(r["items"]) <= 3


def test_post_search_with_context(client):
    r = client.post("/search", json={"q": "почта", "segment": "SMB", "history": ["P016"]}).json()
    assert r["items"] and r["context"]["segment"] == "SMB"


def test_validation_errors(client):
    assert client.get("/search", params={"q": "   "}).status_code == 422
    assert "Пустой запрос" in client.get("/search", params={"q": ""}).text
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "я" * 301}).status_code == 422
    assert client.post("/search", json={"k": 5}).status_code == 422
    assert client.post("/event", json={"product_id": "NOPE", "event_type": "click"}).status_code == 404
    assert client.post("/event", json={"product_id": "P001", "event_type": "hack"}).status_code == 422


# 9. Повтор запросов: одинаковый ответ, кэш, каждый вызов в журнале
def test_repeat_queries(client):
    a = client.get("/search", params={"q": "Мониторинг серверов"}).json()
    b = client.get("/search", params={"q": "мониторинг   серверов"}).json()
    assert [i["id"] for i in a["items"]] == [i["id"] for i in b["items"]]
    assert b["cache_hit"] is True and a["search_id"] != b["search_id"]
    lines = (client.log_dir / "searches.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# 10. Нет персональных данных в журнале
def test_no_pii_in_logs(client):
    client.post("/search", json={
        "q": "антивирус ivan.petrov@corp.ru +7 (912) 345-67-89 инн 7712345678",
        "session": "user-session-SECRET-42", "history": ["P003", "P005"], "segment": "SMB"})
    client.post("/event", json={"product_id": "P003", "event_type": "click", "session": "user-session-SECRET-42"})
    raw = "".join((client.log_dir / f).read_text(encoding="utf-8") for f in ("searches.jsonl", "events.jsonl"))
    for secret in ("ivan.petrov", "corp.ru", "912", "345-67-89", "7712345678", "SECRET"):
        assert secret not in raw, secret
    rec = json.loads((client.log_dir / "searches.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for k in ("ip", "client_ip", "user_agent", "user_id", "history"):
        assert k not in rec
    assert "<email>" in rec["query"] and "<phone>" in rec["query"]


def test_zero_results_logged_separately_and_in_analytics(client):
    client.get("/search", params={"q": "компьютерные мыши", "session": "s1"})
    client.get("/search", params={"q": "компьютерные мыши", "session": "s2"})
    client.get("/search", params={"q": "мыши", "session": "s2"})
    client.get("/search", params={"q": "crm", "session": "s1"})  # s1 переформулировал успешно
    zero_lines = (client.log_dir / "zero_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(zero_lines) == 3
    a = client.get("/analytics").json()
    assert a["totals"]["searches"] == 4
    assert a["totals"]["zero_result_rate"] == 0.75
    zq = {z["query"]: z["count"] for z in a["zero_result_queries"]}
    assert zq["компьютерные мыши"] == 2
    terms = {t["term"]: t["count"] for t in a["unmet_demand_terms"]}
    assert terms["мыши"] == 3
    assert a["failed_search_outcome"]["zero_then_recovered"] == 1
    assert a["failed_search_outcome"]["zero_then_abandoned"] == 2
    assert "SYNTHETIC" in a["imported_behavior"]["label"]
    z = client.get("/analytics/zero-results").json()
    assert z["queries"][0]["query"] == "компьютерные мыши"


def test_funnel_from_events(client):
    r = client.get("/search", params={"q": "crm"}).json()
    client.post("/event", json={"search_id": r["search_id"], "product_id": "P008", "event_type": "click", "position": 0})
    client.post("/event", json={"search_id": r["search_id"], "product_id": "P008", "event_type": "purchase", "position": 0})
    client.get("/search", params={"q": "bi"})
    f = client.get("/analytics").json()["funnel"]
    assert f["with_click"] == 1 and f["with_purchase"] == 1 and f["ctr"] == 0.5


def test_suggest(client):
    s = [x["text"] for x in client.get("/suggest", params={"q": "стр"}).json()["suggestions"]]
    assert s and all(x.startswith("страж") for x in s[:3])
    s = [x["text"] for x in client.get("/suggest", params={"q": "тикет"}).json()["suggestions"]]
    assert "тикетница" in s
    # пустые запросы из журнала не подсказываются
    client.get("/search", params={"q": "компьютерные мыши"})
    s = [x["text"] for x in client.get("/suggest", params={"q": "компь"}).json()["suggestions"]]
    assert "компьютерные мыши" not in s


def test_business_boost_toggle(client):
    a = client.get("/search", params={"q": "почта", "business_boost": "false"}).json()
    assert a["business_boost_enabled"] is False and all(i["business_boost"] == 0 for i in a["items"])


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Умный поиск" in r.text


def test_eval_script_runs(tmp_path):
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--latency-runs", "1",
                          "--out", str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert res["final"]["all"]["recall@10"] is not None
