import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config as C  # noqa: E402
from app.data_loader import load_catalog, load_events, load_synonym_groups  # noqa: E402
from app.engine import SearchEngine  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return (load_catalog(), load_synonym_groups(C.SYNONYMS_PATHS[:1]),
            load_synonym_groups(C.SYNONYMS_PATHS[1:]), load_events())


@pytest.fixture(scope="session")
def engine(data):
    return SearchEngine(*data, cfg=C.DEFAULT_CONFIG)


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app(log_dir=tmp_path)
    with TestClient(app) as c:
        c.log_dir = tmp_path
        yield c


def ids(res, n=None):
    items = res["items"] if n is None else res["items"][:n]
    return [i["id"] for i in items]
