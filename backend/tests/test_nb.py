# backend/tests/test_nb.py
"""Tes fungsi inti NB.
Tes unit memakai data ILUSTRASI buatan sendiri (bukan data ReDial).
Tes bertanda needs_data memakai data ReDial asli dan dilewati jika file belum ada."""

import math
from pathlib import Path

import numpy as np
import pytest

from backend.app.services.nb import NBData, find_neighbors, load_data, recommend, vote

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


def make_data(angles, items):
    emb = np.array([unit(a) for a in angles], dtype=np.float32)
    n = len(angles)
    return NBData(
        emb=emb,
        conv_ids=[f"c{i}" for i in range(n)],
        contexts=[f"konteks {i}" for i in range(n)],
        items=items,
    )


# ILUSTRASI: konteks pada sudut 0, 10, 20, 90 derajat
ITEMS = [
    [("m1", "Film 1"), ("m2", "Film 2")],
    [("m2", "Film 2"), ("m3", "Film 3")],
    [("m2", "Film 2"), ("m4", "Film 4")],
    [("m5", "Film 5")],
]
QUERY = np.array([1.0, 0.0], dtype=np.float32)  # sudut 0 derajat


@pytest.fixture
def data():
    return make_data([0, 10, 20, 90], ITEMS)


# ---------- find_neighbors ----------

def test_neighbors_sorted_by_cosine(data):
    nbs = find_neighbors(QUERY, data, k=4)
    assert [n.row for n in nbs] == [0, 1, 2, 3]
    sims = [n.similarity for n in nbs]
    assert sims == sorted(sims, reverse=True)


def test_neighbors_similarity_values(data):
    nbs = find_neighbors(QUERY, data, k=4)
    expected = [1.0, math.cos(math.radians(10)), math.cos(math.radians(20)), 0.0]
    for nb, e in zip(nbs, expected):
        assert nb.similarity == pytest.approx(e, abs=1e-5)


def test_top_k_limits_result(data):
    assert len(find_neighbors(QUERY, data, k=2)) == 2


def test_k_larger_than_data_returns_all(data):
    assert len(find_neighbors(QUERY, data, k=10)) == 4


def test_neighbor_ties_keep_earlier_row_first():
    d = make_data([0, 0, 45], [[("a", "A")], [("b", "B")], [("c", "C")]])
    nbs = find_neighbors(QUERY, d, k=3)
    assert [n.row for n in nbs] == [0, 1, 2]


def test_query_is_normalized(data):
    a = find_neighbors(QUERY, data, k=4)
    b = find_neighbors(QUERY * 5, data, k=4)
    assert [n.row for n in a] == [n.row for n in b]
    assert [n.similarity for n in b] == pytest.approx([n.similarity for n in a], abs=1e-6)


def test_invalid_inputs_raise(data):
    with pytest.raises(ValueError):
        find_neighbors(QUERY, data, k=0)
    with pytest.raises(ValueError):
        find_neighbors(np.zeros(2, dtype=np.float32), data, k=3)
    with pytest.raises(ValueError):
        find_neighbors(np.ones(3, dtype=np.float32), data, k=3)


# ---------- vote ----------

def test_vote_counts_each_neighbor_once(data):
    ranked = vote(find_neighbors(QUERY, data, k=3), data)
    votes = {r.movie_id: r.votes for r in ranked}
    assert votes == {"m1": 1, "m2": 3, "m3": 1, "m4": 1}


def test_vote_ranks_by_votes_first(data):
    ranked = vote(find_neighbors(QUERY, data, k=3), data)
    assert ranked[0].movie_id == "m2"
    assert ranked[0].votes == 3


def test_vote_tie_break_by_similarity_sum(data):
    # m1, m3, m4 sama-sama 1 suara; urut menurut similarity tetangga pendukung (K9)
    ranked = vote(find_neighbors(QUERY, data, k=3), data)
    assert [r.movie_id for r in ranked] == ["m2", "m1", "m3", "m4"]


def test_vote_tie_break_by_movie_id():
    d = make_data([0], [[("m9", "Z"), ("m1", "A")]])
    ranked = vote(find_neighbors(QUERY, d, k=1), d)
    assert [r.movie_id for r in ranked] == ["m1", "m9"]


def test_vote_records_supporters(data):
    ranked = vote(find_neighbors(QUERY, data, k=3), data)
    m2 = next(r for r in ranked if r.movie_id == "m2")
    assert [cid for cid, _ in m2.supporters] == ["c0", "c1", "c2"]
    assert m2.sim_sum == pytest.approx(sum(s for _, s in m2.supporters))


def test_vote_without_neighbors_is_empty(data):
    assert vote([], data) == []


# ---------- recommend ----------

def test_recommend_returns_neighbors_and_top_n(data):
    nbs, recs = recommend(QUERY, data, k=3, n=2)
    assert len(nbs) == 3
    assert [r.movie_id for r in recs] == ["m2", "m1"]


def test_recommend_n_larger_than_distinct_items(data):
    _, recs = recommend(QUERY, data, k=4, n=100)
    assert len(recs) == 5  # m1 sampai m5


# ---------- tes dengan data ReDial asli ----------

@needs_data
def test_real_data_is_consistent():
    for split in ("train", "test"):
        d = load_data(split, PROCESSED, EMB)
        n = len(d.conv_ids)
        assert d.emb.shape[0] == n == len(d.contexts) == len(d.items)
        assert all(len(items) >= 1 for items in d.items)
        assert np.allclose(np.linalg.norm(d.emb, axis=1), 1.0, atol=1e-4)
        assert len(set(d.conv_ids)) == n


@needs_data
def test_real_items_unique_within_conversation():
    d = load_data("train", PROCESSED, EMB)
    for items in d.items:
        ids = [mid for mid, _ in items]
        assert len(ids) == len(set(ids))


@needs_data
def test_real_train_and_test_conversations_are_disjoint():
    train = load_data("train", PROCESSED, EMB)
    test = load_data("test", PROCESSED, EMB)
    assert not set(train.conv_ids) & set(test.conv_ids)


@needs_data
def test_real_self_query_finds_identical_embedding_first():
    d = load_data("train", PROCESSED, EMB)
    for row in (0, 1, 2):
        nb = find_neighbors(d.emb[row], d, k=1)[0]
        assert nb.similarity == pytest.approx(1.0, abs=1e-4)
        assert np.allclose(d.emb[nb.row], d.emb[row])