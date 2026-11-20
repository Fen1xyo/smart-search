"""Все настраиваемые веса и пороги в одном месте.

Пути переопределяются переменными окружения, чтобы 26.11 подменить синтетику
реальными данными без правки кода (см. README, раздел «Реальные данные»).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v) if v else default


DATA_DIR = _p("IK3_DATA_DIR", ROOT / "data" / "synthetic")
CATALOG_PATH = _p("IK3_CATALOG", DATA_DIR / "catalog_synthetic.json")
SYNONYMS_PATHS = [
    _p("IK3_SYNONYMS", DATA_DIR / "synonyms_synthetic.json"),
    _p("IK3_TEAM_LEXICON", ROOT / "data" / "lexicon_team.json"),
]
EVENTS_PATH = _p("IK3_EVENTS", DATA_DIR / "behavior_events_synthetic.json")
EVAL_PATH = _p("IK3_EVAL", DATA_DIR / "eval_queries_synthetic.json")
FIELD_MAP_PATH = _p("IK3_FIELD_MAP", ROOT / "data" / "field_mapping.json")
LOG_DIR = _p("IK3_LOG_DIR", ROOT / "logs")


@dataclass(frozen=True)
class SearchConfig:
    # --- переключатели для ablation ---
    use_stemming: bool = True
    use_synonyms: bool = True
    use_team_lexicon: bool = True
    use_typo: bool = True
    use_negation: bool = True
    use_generic_terms: bool = True
    use_name_coverage: bool = True
    use_compound: bool = True   # составные слова; добавлено ПОСЛЕ просмотра test-ошибок (см. README)
    use_coverage_gate: bool = True
    use_business_boost: bool = True

    # --- BM25 ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    field_weights: dict = field(default_factory=lambda: {
        "name": 3.0, "vendor": 2.0, "category": 1.5, "attributes": 2.0, "description": 1.0,
    })
    synonym_weight: float = 0.7      # вклад совпадения через синоним относительно исходного слова
    typo_weight: float = 0.9         # вклад исправленного слова
    exact_name_bonus: float = 1.0    # добавка к rel_norm при точном совпадении ID / названия
    negation_penalty: float = 0.5    # множитель релевантности товара, совпавшего с «без X»
    name_coverage_bonus: float = 0.05  # +к rel_norm, если запрос покрывает все (нечисловые) слова названия
    compound_weight: float = 0.6     # вклад совпадения части составного слова
    compound_min_len: int = 5
    generic_term_factor: float = 0.2   # вес родовых слов («система», «менеджер») в coverage

    # --- опечатки ---
    typo_min_len: int = 4
    typo_min_score: float = 82.0     # RapidFuzz ratio, 0..100

    # --- гейты ---
    coverage_threshold: float = 0.5  # середина плато dev-развёртки 0.40–0.55 (см. reports)
    relevance_gate: float = 0.75     # бизнес-буст только для rel >= gate * rel_top

    # --- бизнес-буст (центрирован: нейтральный сигнал = 0) ---
    business_weight: float = 0.2     # итоговый сдвиг ∈ [-0.1; +0.1]
    popularity_share: float = 0.6
    margin_share: float = 0.4
    context_weight: float = 0.1      # сегмент / история клиента, тоже внутри гейта
    neutral_signal: float = 0.5

    default_k: int = 10

    def with_(self, **kw) -> "SearchConfig":
        return replace(self, **kw)


DEFAULT_CONFIG = SearchConfig()
