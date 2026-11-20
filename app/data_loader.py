"""Импорт каталога, словарей и событий.

Формат входа: JSON-массив, JSONL или CSV. Колонки сопоставляются через
`data/field_mapping.json`, поэтому реальный набор 26.11 подключается правкой
маппинга, а не кода. Отсутствующие поля остаются None — ничего не выдумываем.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config


@dataclass
class Product:
    id: str
    name: str
    description: str = ""
    category: str = ""
    vendor: str = ""
    attributes: list[str] = field(default_factory=list)
    popularity: float | None = None
    margin: float | None = None
    price: float | None = None
    is_mock: bool = False

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "vendor": self.vendor, "category": self.category,
            "description": self.description, "attributes": self.attributes, "price": self.price,
            "is_mock": self.is_mock,
        }


def read_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8-sig") as f:
        if suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if suffix == ".csv":
            return list(csv.DictReader(f))
        data = json.load(f)
    if isinstance(data, dict):  # {"items": [...]} или {"products": [...]}
        for key in ("items", "products", "data", "events", "queries"):
            if isinstance(data.get(key), list):
                return data[key]
    return data


def load_mapping(section: str) -> dict[str, list[str]]:
    p = config.FIELD_MAP_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get(section, {})
    return {}


def _pick(rec: dict, names: list[str]) -> Any:
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return rec[n]
    return None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


def _attrs(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, dict):  # {"ОС": "Linux"} -> ["ОС: Linux"]
        return [f"{k}: {val}" for k, val in v.items()]
    if isinstance(v, str):
        return [s.strip() for s in v.replace(";", ",").split(",") if s.strip()]
    return [str(x) for x in v]


def load_catalog(path: Path | None = None) -> list[Product]:
    m = load_mapping("catalog")
    g = lambda r, k, d: _pick(r, m.get(k, [d]))  # noqa: E731
    out, seen = [], set()
    for r in read_records(path or config.CATALOG_PATH):
        pid = g(r, "id", "product_id")
        name = g(r, "name", "name")
        if pid is None or name is None or str(pid) in seen:
            continue  # без ID товар нельзя вернуть — пропускаем, а не придумываем
        seen.add(str(pid))
        out.append(Product(
            id=str(pid), name=str(name),
            description=str(g(r, "description", "description") or ""),
            category=str(g(r, "category", "category") or ""),
            vendor=str(g(r, "vendor", "vendor") or ""),
            attributes=_attrs(g(r, "attributes", "attributes")),
            popularity=_num(g(r, "popularity", "popularity")),
            margin=_num(g(r, "margin", "margin")),
            price=_num(g(r, "price", "price")),
            is_mock=bool(r.get("is_mock", False)),
        ))
    return out


def load_synonym_groups(paths: list[Path] | None = None) -> list[list[str]]:
    """Словарь {термин: [варианты]} -> список групп эквивалентных фраз."""
    groups = []
    for p in paths or config.SYNONYMS_PATHS:
        p = Path(p)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for key, variants in data.items():
            if key.startswith("_") or not isinstance(variants, list):
                continue
            groups.append([key, *[str(v) for v in variants]])
    return groups


def load_events(path: Path | None = None) -> list[dict]:
    m = load_mapping("events")
    out = []
    for r in read_records(path or config.EVENTS_PATH):
        out.append({
            "query_id": _pick(r, m.get("query_id", ["query_id"])),
            "product_id": str(_pick(r, m.get("product_id", ["product_id"])) or ""),
            "event_type": _pick(r, m.get("event_type", ["event_type"])),
            "segment": _pick(r, m.get("segment", ["segment"])),
            "is_mock": bool(r.get("is_mock", False)),
        })
    return out


def load_eval(path: Path | None = None) -> list[dict]:
    return read_records(path or config.EVAL_PATH)
