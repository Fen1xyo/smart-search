"""FastAPI: /health, /search, /suggest, /event, /analytics + демо-страница на /.

Запуск: uvicorn app.main:app --port 8000
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import config as C
from .data_loader import load_catalog, load_events, load_synonym_groups
from .engine import SearchEngine
from .store import SearchLog
from .suggest import Suggester

VERSION = "0.2.0"
STATIC = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    q: str = Field(..., description="Поисковый запрос", examples=["антивирс для пк"])
    k: int = Field(10, ge=1, le=50)
    category: str | None = Field(None, description="Фильтр по категории")
    segment: str | None = Field(None, description="Сегмент клиента, напр. SMB / Enterprise")
    history: list[str] = Field(default_factory=list, max_length=50,
                               description="ID товаров, которые клиент смотрел/покупал (не логируются)")
    session: str | None = Field(None, max_length=64, description="Анонимный id сессии; в журнал — только хэш")
    business_boost: bool = Field(True, description="false = ablation без бизнес-буста")

    @field_validator("q")
    @classmethod
    def _q(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Пустой запрос: передайте непустую строку q")
        if len(v) > 300:
            raise ValueError("Запрос длиннее 300 символов")
        return v


class EventRequest(BaseModel):
    search_id: str | None = Field(None, max_length=32)
    product_id: str = Field(..., max_length=64)
    event_type: Literal["click", "add_to_cart", "purchase"]
    position: int | None = Field(None, ge=0, le=100)
    session: str | None = Field(None, max_length=64)


def create_app(log_dir: Path | None = None, cache_size: int = 512) -> FastAPI:
    products = load_catalog()
    dict_groups = load_synonym_groups(C.SYNONYMS_PATHS[:1])
    team_groups = load_synonym_groups(C.SYNONYMS_PATHS[1:])
    events = load_events()
    engines = {
        True: SearchEngine(products, dict_groups, team_groups, events, C.DEFAULT_CONFIG),
        False: SearchEngine(products, dict_groups, team_groups, events,
                            C.DEFAULT_CONFIG.with_(use_business_boost=False)),
    }
    log = SearchLog(log_dir or C.LOG_DIR)
    suggester = Suggester(engines[True], log)
    cache: OrderedDict = OrderedDict()

    app = FastAPI(title="IK3 Smart Search — команда №2", version=VERSION,
                  description="MVP умного поиска. Данные по умолчанию — СИНТЕТИКА команды, не данные заказчика.")
    app.state.log = log
    app.state.engines = engines

    def run_search(req: SearchRequest) -> dict:
        eng = engines[req.business_boost]
        key = (" ".join(req.q.lower().split()), req.k, req.category, req.segment, tuple(req.history),
               req.business_boost)
        hit = key in cache
        if hit:
            cache.move_to_end(key)
            res = deepcopy(cache[key])
            res["latency_ms"] = 0.0
        else:
            res = eng.search(req.q, k=req.k, category=req.category, segment=req.segment, history=req.history)
            cache[key] = deepcopy(res)
            if len(cache) > cache_size:
                cache.popitem(last=False)
        res["search_id"] = log.log_search(res, req.session, req.k, cache_hit=hit)
        res["cache_hit"] = hit
        res["business_boost_enabled"] = req.business_boost
        return res

    @app.get("/health")
    def health():
        p = engines[True].products
        return {"status": "ok", "version": VERSION, "products": len(p),
                "synthetic_data": any(x.is_mock for x in p),
                "catalog": str(C.CATALOG_PATH.name),
                "margin_available": sum(x.margin is not None for x in p),
                "popularity_available": sum(x.popularity is not None for x in p)}

    @app.get("/search")
    def search_get(q: str = Query(..., description="Запрос"), k: int = Query(10, ge=1, le=50),
                   category: str | None = None, segment: str | None = None,
                   history: str | None = Query(None, description="ID через запятую"),
                   session: str | None = None, business_boost: bool = True):
        try:
            req = SearchRequest(q=q, k=k, category=category, segment=segment,
                                history=[h.strip() for h in (history or "").split(",") if h.strip()],
                                session=session, business_boost=business_boost)
        except ValueError as e:
            raise HTTPException(422, detail=_err(e))
        return run_search(req)

    @app.post("/search")
    def search_post(req: SearchRequest):
        return run_search(req)

    @app.get("/suggest")
    def suggest(q: str = Query("", max_length=100), limit: int = Query(8, ge=1, le=20)):
        return {"query": q, "suggestions": suggester.suggest(q, limit)}

    @app.post("/event")
    def event(req: EventRequest):
        if req.product_id not in engines[True].by_id:
            raise HTTPException(404, detail=f"Неизвестный product_id: {req.product_id}")
        log.log_event(req.search_id, req.product_id, req.event_type, req.position, req.session)
        return {"status": "logged"}

    @app.get("/analytics")
    def analytics(top: int = Query(15, ge=1, le=100)):
        return log.analytics(imported_events=events, top=top)

    @app.get("/analytics/zero-results")
    def zero_results(top: int = Query(50, ge=1, le=500)):
        a = log.analytics(top=top)
        return {"zero_result_rate": a["totals"]["zero_result_rate"],
                "queries": a["zero_result_queries"], "unmet_demand_terms": a["unmet_demand_terms"]}

    @app.get("/categories")
    def categories():
        return {"categories": engines[True].categories}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def _err(e: ValueError) -> str:
    errs = getattr(e, "errors", None)
    if callable(errs):
        return "; ".join(str(x.get("msg")) for x in errs())
    return str(e)


app = create_app()
