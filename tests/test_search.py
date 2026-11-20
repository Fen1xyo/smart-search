"""Поведенческие тесты движка. Запросы здесь НЕ совпадают с eval_queries_synthetic.json
(кроме явно отмеченных), чтобы проверять обобщение, а не заучивание набора оценки."""
import pytest

from app import config as C
from app.data_loader import Product
from app.engine import SearchEngine
from app.text import tokens

from conftest import ids


# 1. Точное наименование / ID
def test_exact_name(engine):
    r = engine.search("Орбита Офис Сервер")
    assert ids(r)[0] == "P002" and r["items"][0]["exact_match"]


def test_exact_id(engine):
    r = engine.search("P021")
    assert ids(r)[0] == "P021" and r["items"][0]["exact_match"]


def test_partial_name_prefers_fully_covered_name(engine):
    assert ids(engine.search("заря bi"))[0] == "P026"   # не SelfHost


# 2. Аббревиатуры
@pytest.mark.parametrize("q,expected", [
    ("вкс", {"P016", "P017"}),
    ("субд", {"P012", "P013"}),
    ("эдо", {"P022", "P023"}),
    ("iam", {"P020"}),
    ("wms", {"P025"}),
])
def test_abbreviation(engine, q, expected):
    r = engine.search(q)
    assert not r["zero_result"]
    assert set(ids(r, len(expected))) == expected


# 3. Синонимы (без буквального совпадения слов с карточкой)
@pytest.mark.parametrize("q,expected", [
    ("видеозвонки", {"P016", "P017"}),
    ("резервная копия", {"P006", "P007"}),
    ("система управления клиентами", {"P008", "P009"}),
    ("предотвращение утечек", {"P030"}),
])
def test_synonym(engine, q, expected):
    r = engine.search(q)
    assert set(ids(r, len(expected))) == expected, ids(r)
    assert r["synonyms_applied"]


# 4. Опечатки и раскладка
@pytest.mark.parametrize("q,expected,kind", [
    ("антивирс", "P003", "typo"),
    ("мониторинк серверов", "P018", "typo"),
    ("видеоконференци", "P016", None),        # недописанное слово — стемминг справляется
    ("vjybnjhbyu", "P018", "layout"),         # «мониторинг» в английской раскладке
    ("резервное копированее", "P006", None),   # стеммер снимает «-ее», исправление не нужно
    ("тикетнеца", "P028", "typo"),
])
def test_typo(engine, q, expected, kind):
    r = engine.search(q)
    assert ids(r)[0] == expected, (ids(r), r["corrections"])
    if kind:
        assert any(c["kind"] == kind for c in r["corrections"])
        assert r["did_you_mean"]


def test_typo_does_not_touch_known_words(engine):
    assert engine.search("почта")["corrections"] == []


# 5. Сленг
@pytest.mark.parametrize("q,expected", [
    ("тикетница", {"P028", "P029"}),
    ("срм", {"P008", "P009"}),
    ("бэкап", {"P006", "P007"}),
])
def test_slang(engine, q, expected):
    assert set(ids(engine.search(q), len(expected))) == expected


# 6. Несколько ограничений
def test_multi_constraint_negation(engine):
    r = engine.search("crm без облака")
    assert ids(r)[0] == "P009"
    cloud = next(i for i in r["items"] if i["id"] == "P008")
    assert cloud["negated"]


def test_multi_constraint_attributes(engine):
    r = engine.search("трекер задач on-prem linux api")
    assert ids(r)[0] == "P011"
    assert {"on-prem", "linux", "api"} <= set(r["items"][0]["confirmed_attributes"])


def test_multi_constraint_cloud_mail(engine):
    assert ids(engine.search("облачная почта и календарь"))[0] == "P015"


def test_category_context_filter(engine):
    r = engine.search("сервер", category="Кибербезопасность")
    assert r["items"] and all(i["category"] == "Кибербезопасность" for i in r["items"])
    assert r["context"]["filtered_out_by_category"] > 0


# 7. Несуществующий товар
@pytest.mark.parametrize("q", ["компьютерные мыши", "кофемашина для офиса", "лицензия windows 9999",
                               "ноутбук игровой с rtx", "система для автосервиса"])
def test_nonexistent(engine, q):
    r = engine.search(q)
    assert r["zero_result"] and r["items"] == []
    assert r["unmatched_terms"], "непокрытые слова должны попасть в отчёт о спросе"


# 8. Отсутствие поля маржи / популярности
def test_missing_margin(data):
    products, dg, tg, ev = data
    no_margin = [_clone(p, margin=None) for p in products]
    e = SearchEngine(no_margin, dg, tg, ev, C.DEFAULT_CONFIG)
    r = e.search("антивирус для рабочих станций")
    assert ids(r)[0] == "P003"
    for it in r["items"]:
        assert "margin" in it["business"]["missing"]
        assert it["business"]["margin_norm"] == C.DEFAULT_CONFIG.neutral_signal


def test_missing_all_business_fields(data):
    products, dg, tg, ev = data
    bare = [_clone(p, margin=None, popularity=None) for p in products]
    e = SearchEngine(bare, dg, tg, [], C.DEFAULT_CONFIG)
    r = e.search("crm")
    assert all(it["business_boost"] == 0 for it in r["items"])  # нейтрально, ничего не выдумано


def _clone(p: Product, **kw) -> Product:
    d = dict(p.__dict__)
    d.update(kw)
    return Product(**d)


# 9. Повтор запросов — детерминизм
def test_repeat_deterministic(engine):
    a = [ids(engine.search("локальный эдо")) for _ in range(5)]
    assert all(x == a[0] for x in a)


# 11. Отсутствие вредного нерелевантного буста
def test_no_harmful_boost(data):
    products, dg, tg, ev = data
    bait = Product(id="X999", name="Мега Офис Пакет", vendor="Мега", category="Офисное ПО",
                   description="Офисный пакет, работает на любом сервере.", attributes=["windows"],
                   popularity=10_000_000, margin=0.99, is_mock=True)
    e = SearchEngine(products + [bait], dg, tg, ev, C.DEFAULT_CONFIG)
    r = e.search("антивирус для серверов")
    order = ids(r)
    assert order[0] == "P004"
    if "X999" in order:
        bait_item = r["items"][order.index("X999")]
        assert not bait_item["passed_relevance_gate"] and bait_item["business_boost"] == 0
        assert order.index("X999") > order.index("P004")


def test_gate_invariant_on_all_eval_queries(engine):
    """Ни один товар ниже relevance gate не стоит выше прошедшего gate; буст ограничен."""
    from app.data_loader import load_eval
    cap = C.DEFAULT_CONFIG.business_weight / 2 + C.DEFAULT_CONFIG.context_weight / 2 + 1e-9
    for q in load_eval():
        items = engine.search(q["query"], segment="SMB", history=["P003"])["items"]
        seen_ungated = False
        for it in items:
            if not it["passed_relevance_gate"]:
                seen_ungated = True
                assert it["business_boost"] == 0
            else:
                assert not seen_ungated, q["query"]
            assert abs(it["business_boost"]) <= cap


def test_boost_only_reorders_close_candidates(data):
    """Буст двигает только близкие по релевантности товары; сильный отрыв не перекрывается."""
    products, dg, tg, ev = data
    on = SearchEngine(*data, cfg=C.DEFAULT_CONFIG)
    off = SearchEngine(*data, cfg=C.DEFAULT_CONFIG.with_(use_business_boost=False))
    for q in ["почта", "crm", "антивирус", "bi"]:
        a, b = on.search(q)["items"], off.search(q)["items"]
        rel = {i["id"]: i["relevance_norm"] for i in b}
        top_off = b[0]
        top_on = a[0]
        assert rel[top_on["id"]] >= rel[top_off["id"]] - C.DEFAULT_CONFIG.business_weight


# Морфология русского языка
@pytest.mark.parametrize("a,b", [("серверов", "сервер"), ("антивирусы", "антивирус"),
                                  ("локальная", "локальный"), ("документооборота", "документооборот")])
def test_morphology(a, b):
    assert tokens(a) == tokens(b)


def test_context_history_reorders_within_gate(engine):
    base = ids(engine.search("почта"))
    with_hist = engine.search("почта", history=["P016"])  # клиент смотрел облачную ВКС
    assert set(ids(with_hist)) == set(base)
    assert all("history:category" in it["context_reasons"] for it in with_hist["items"]
               if it["passed_relevance_gate"] and it["category"] == "Коммуникации")
