# backend/tests/test_recommender.py
"""Tes pembungkus query teks. Memakai model palsu agar cepat; data ILUSTRASI buatan sendiri."""

import numpy as np
import pytest

from backend.app.services.nb import NBData
from backend.app.services.recommender import Recommender


class FakeModel:
    """Pengganti model embedding: selalu menghasilkan vektor yang sama dan mencatat teks masukan."""

    def __init__(self):
        self.calls = []

    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(text)
        return np.array([1.0, 0.0])


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def rec(model):
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    data = NBData(
        emb=emb,
        conv_ids=["c0", "c1"],
        contexts=["konteks 0", "konteks 1"],
        items=[[("m1", "Film 1")], [("m2", "Film 2")]],
    )
    return Recommender(data=data, model=model)


def test_empty_query_returns_status_without_error(rec, model):
    res = rec.recommend_text("")
    assert res.status == "empty_query"
    assert res.recommendations == [] and res.neighbors == []
    assert res.top_similarity is None
    assert model.calls == []


def test_whitespace_and_non_string_are_empty(rec):
    assert rec.recommend_text("   \n ").status == "empty_query"
    assert rec.recommend_text(None).status == "empty_query"


def test_spaces_are_normalized_before_encoding(rec, model):
    res = rec.recommend_text("  a   movie \n like  x ")
    assert res.query == "a movie like x"
    assert model.calls == ["a movie like x"]


def test_ok_result_fields(rec):
    res = rec.recommend_text("anything", k=2, n=1)
    assert res.status == "ok"
    assert len(res.neighbors) == 2
    assert len(res.recommendations) == 1
    assert res.top_similarity == pytest.approx(res.neighbors[0].similarity)
    assert res.recommendations[0].movie_id == "m1"


def test_invalid_k_or_n_raises(rec):
    with pytest.raises(ValueError):
        rec.recommend_text("x", k=0)
    with pytest.raises(ValueError):
        rec.recommend_text("x", n=0)