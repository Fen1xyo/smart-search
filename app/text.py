"""Нормализация русского/английского текста, стемминг, маскирование ПДн."""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")

STOPWORDS = {
    # ru
    "и", "в", "во", "на", "с", "со", "по", "для", "от", "до", "из", "к", "ко", "о", "об", "а",
    "или", "у", "за", "при", "про", "это", "мой", "мне", "нам", "наш", "свой", "своем", "своей",
    "свою", "своих", "своего", "нужен", "нужна", "нужно", "нужны", "хочу", "купить", "какой", "какая",
    "что", "как", "где",
    # en
    "the", "for", "and", "of", "to", "a", "an", "in", "with",  # "on" не стоп-слово: on-prem
}
# Родовые «головные» слова каталога ПО: несут мало информации о том, ЧТО ищут.
# В coverage их вес снижается (config.generic_term_factor), из BM25 не удаляются.
GENERIC_TERMS = {
    "систем", "программ", "софт", "решен", "платформ", "сервис", "приложен", "инструмент",
    "менеджер", "продукт", "средств", "комплекс", "модул", "пакет", "корпоративн",
    "system", "software", "platform", "tool", "manag", "solut",
}
NEGATIONS = {"без", "не", "кроме", "except", "without", "no"}

_CAMEL = re.compile(r"(?<=[a-zа-я])(?=[A-ZА-Я])")
_TOKEN = re.compile(r"[a-zа-я0-9]+")
_CYR = re.compile(r"[а-я]")

EN_KEYS = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
RU_KEYS = "йцукенгшщзхъфывапролджэячсмитьбюё"
_EN2RU = str.maketrans(EN_KEYS, RU_KEYS)
_RU2EN = str.maketrans(RU_KEYS, EN_KEYS)


def normalize(text: str) -> str:
    text = _CAMEL.sub(" ", text or "")          # OnPrem -> On Prem, SelfHost -> Self Host
    return text.lower().replace("ё", "е")


def words(text: str) -> list[str]:
    """Поверхностные слова (без стемминга, со стоп-словами)."""
    return _TOKEN.findall(normalize(text))


@lru_cache(maxsize=50_000)
def stem(word: str) -> str:
    if word.isdigit():
        return word
    if _CYR.search(word):
        return _RU.stemWord(word)
    return _EN.stemWord(word)


def plain(word: str) -> str:
    """Режим без стемминга (для ablation)."""
    return word


def tokens(text: str, stemming: bool = True) -> list[str]:
    f = stem if stemming else plain
    return [f(w) for w in words(text) if w not in STOPWORDS]


def swap_layout(word: str) -> str:
    """Слово, набранное в неверной раскладке: 'fynbdbhec' -> 'антивирус'."""
    if _CYR.search(word):
        return word.translate(_RU2EN)
    return word.translate(_EN2RU)


# --- ПДн: в журнал попадает только маскированный текст запроса ---
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\s\-()]*){10,}")
_LONGNUM = re.compile(r"\b\d{10,}\b")


def mask_pii(text: str) -> str:
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    return _LONGNUM.sub("<number>", text)
