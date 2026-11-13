# backend/tests/test_api.py
"""Tes API. Tes unit memakai model palsu dan data ILUSTRASI buatan sendiri (bukan data ReDial);
tes bertanda needs_data memakai server sungguhan (model dan data asli) dan dilewati jika data belum ada."""

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.nb import NBData
from backend.app.services.recommender import Recommender

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
HAS_DATA = all(
    p.exists()
    for p in [
        PROCESSED / "pairs.csv",
        PROCESSED / "conversations.csv",
        EMB / "train_emb.npy",
        EMB / "test_emb.npy",
    ]
)
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data olahan/embedding belum dibuat")


def unit(deg):
    rad = math.radians(deg)
    return [math.cos(rad), math.sin(rad)]


# ILUSTRASI: konteks pada sudut 0, 10, 20, 90 derajat
ITEMS = [
    [("m1", "Film 1"), ("m2", "Film 2")],
    [("m2", "Film 2"), ("m3", "Film 3")],
    [("m2", "Film 2"), ("m4", "Film 4")],
    [("m5", "Film 5")],
]


def make_data():
    angles = [0, 10, 20, 90]
    return NBData(
        emb=np.array([unit(a) for a in angles], dtype=np.float32),
        conv_ids=[f"c{i}" for i in range(4)],
        contexts=[f"konteks {i}" for i in range(4)],
        items=ITEMS,
    )


class FakeModel:
    """Pengganti model embedding: query selalu pada sudut 0 derajat; mencatat teks masukan."""

    def __init__(self):
        self.calls = []

    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(text)
        return np.array([1.0, 0.0])


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model):
    # TestClient tanpa "with" tidak menjalankan lifespan, jadi model asli tidak dimuat.
    app.state.recommender = Recommender(data=make_data(), model=model)
    return TestClient(app)


# ---------- dengan model palsu ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "all-MiniLM-L6-v2"
    assert body["datastore_conversations"] == 4


def test_recommend_ok_structure(client):
    r = client.post("/recommend", json={"query": "anything", "k": 3, "n": 4, "show_neighbors": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["k"] == 3 and body["n"] == 4
    assert body["top_similarity"] == pytest.approx(1.0, abs=1e-4)

    nbs = body["neighbors"]
    assert [n["rank"] for n in nbs] == [1, 2]
    assert [n["conversation_id"] for n in nbs] == ["c0", "c1"]
    assert nbs[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert nbs[1]["similarity"] == pytest.approx(math.cos(math.radians(10)), abs=1e-4)
    assert nbs[0]["items"] == ["Film 1", "Film 2"]

    recs = body["recommendations"]
    assert [x["movie_id"] for x in recs] == ["m2", "m1", "m3", "m4"]
    assert [x["title"] for x in recs] == ["Film 2", "Film 1", "Film 3", "Film 4"]
    assert [x["votes"] for x in recs] == [3, 1, 1, 1]
    assert [x["rank"] for x in recs] == [1, 2, 3, 4]
    assert recs[0]["explanation"] == "Disarankan di 3 dari 3 percakapan serupa (total similarity 2.92)."
    assert [s["conversation_id"] for s in recs[0]["supporters"]] == ["c0", "c1", "c2"]


def test_default_values(client):
    body = client.post("/recommend", json={"query": "x"}).json()
    assert body["k"] == 90 and body["n"] == 10
    assert len(body["neighbors"]) == 4      # datastore ilustrasi hanya 4 percakapan
    assert len(body["recommendations"]) == 5  # m1 sampai m5


def test_empty_query_returns_status_not_error(client, model):
    r = client.post("/recommend", json={"query": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "empty_query"
    assert body["recommendations"] == [] and body["neighbors"] == []
    assert body["top_similarity"] is None
    assert body["message"] != ""
    assert model.calls == []


def test_whitespace_query_is_empty(client):
    r = client.post("/recommend", json={"query": "   \n  "})
    assert r.status_code == 200
    assert r.json()["status"] == "empty_query"


def test_query_spaces_are_normalized(client, model):
    body = client.post("/recommend", json={"query": "  a   movie  "}).json()
    assert body["query"] == "a movie"
    assert model.calls == ["a movie"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "x", "k": 0},
        {"query": "x", "k": 201},
        {"query": "x", "n": 0},
        {"query": "x", "n": 51},
        {"query": "x", "show_neighbors": -1},
        {"query": "x", "show_neighbors": 51},
        {"query": "x" * 1001},
        {},
        {"query": None},
    ],
    ids=["k=0", "k=201", "n=0", "n=51", "show=-1", "show=51", "query-1001-chars", "missing-query", "query-null"],
)
def test_invalid_inputs_rejected(client, model, payload):
    r = client.post("/recommend", json=payload)
    assert r.status_code == 422
    assert model.calls == []


def test_show_neighbors_zero_hides_neighbors(client):
    body = client.post("/recommend", json={"query": "x", "k": 3, "show_neighbors": 0}).json()
    assert body["neighbors"] == []
    assert len(body["recommendations"]) > 0


def test_n_larger_than_distinct_items(client):
    body = client.post("/recommend", json={"query": "x", "k": 4, "n": 50}).json()
    assert len(body["recommendations"]) == 5


def test_cors_allows_vite_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_blocks_unknown_origin(client):
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_value_error_becomes_400(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(app.state.recommender, "recommend_text", boom)
    r = client.post("/recommend", json={"query": "x"})
    assert r.status_code == 400
    assert r.json()["detail"] == "boom"


# ---------- server sungguhan (model dan data asli) ----------

@pytest.fixture(scope="module")
def real_client():
    with TestClient(app) as c:  # "with" menjalankan lifespan: memuat model dan data asli
        yield c


@needs_data
def test_real_health(real_client):
    body = real_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["datastore_conversations"] == 9566


@needs_data
def test_real_recommend(real_client):
    r = real_client.post(
        "/recommend",
        json={"query": "I want a psychological thriller with a plot twist", "n": 5, "show_neighbors": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["k"] == 90
    assert len(body["neighbors"]) == 3 and len(body["recommendations"]) == 5
    sims = [n["similarity"] for n in body["neighbors"]]
    assert sims == sorted(sims, reverse=True)
    assert body["top_similarity"] == pytest.approx(sims[0], abs=1e-6)
    votes = [x["votes"] for x in body["recommendations"]]
    assert votes == sorted(votes, reverse=True)
    for x in body["recommendations"]:
        assert 1 <= x["votes"] <= 90
        assert 1 <= len(x["supporters"]) <= 5
        assert x["explanation"].startswith(f"Disarankan di {x['votes']} dari 90 percakapan serupa")


@needs_data
def test_real_empty_query(real_client):
    body = real_client.post("/recommend", json={"query": ""}).json()
    assert body["status"] == "empty_query"