"""Импорт данных другой схемы (подготовка к реальному набору 26.11)."""
import json

from app import config as C
from app.data_loader import load_catalog
from app.engine import SearchEngine


def test_csv_with_other_columns_and_missing_fields(tmp_path, monkeypatch):
    csv_path = tmp_path / "cat.csv"
    csv_path.write_text(
        "sku;title;описание;категория;brand;характеристики\n".replace(";", ",") +
        'A-1,Тест Антивирус,"Защита рабочих станций",Кибербезопасность,Тест,"windows; linux"\n'
        "A-2,Тест Почта,Почтовый сервер,Коммуникации,Тест,\n"
        ",Без ID,будет пропущен,X,Y,\n", encoding="utf-8")
    prods = load_catalog(csv_path)
    assert [p.id for p in prods] == ["A-1", "A-2"]
    assert prods[0].attributes == ["windows", "linux"]
    assert prods[0].margin is None and prods[0].popularity is None
    e = SearchEngine(prods, [], [], [], C.DEFAULT_CONFIG)
    r = e.search("антивирус")
    assert r["items"][0]["id"] == "A-1"
    assert set(r["items"][0]["business"]["missing"]) == {"popularity", "margin"}


def test_jsonl_and_dict_attributes(tmp_path):
    p = tmp_path / "cat.jsonl"
    p.write_text(json.dumps({"id": 7, "name": "СУБД Тест", "attributes": {"ОС": "Linux"},
                             "margin": "0,3"}, ensure_ascii=False) + "\n", encoding="utf-8")
    prods = load_catalog(p)
    assert prods[0].id == "7" and prods[0].attributes == ["ОС: Linux"] and prods[0].margin == 0.3
