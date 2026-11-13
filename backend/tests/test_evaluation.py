# backend/tests/test_evaluation.py
"""Tes fungsi evaluasi. Tes unit memakai data ILUSTRASI buatan sendiri (bukan data ReDial);
tes bertanda needs_data memakai data ReDial asli dan dilewati jika file belum ada."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from backend.app.services.evaluation import (
    ceiling_at_k,
    evaluate_rankings,
    hit_at_k,
    mean_se,
    ndcg_at_k,
    popularity_ranking,
    rank_nb,
    recall_at_k,
    reciprocal_rank_at_k,
    relevant_sets,
    split_validation,
    subset,
)
from backend.app.services.nb import NBData, load_data

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


@pytest.fixture
def data():
    return make_data([0, 10, 20, 90], ITEMS)


# ---------- metrik per query ----------

def test_recall_at_k_by_hand():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)


def test_recall_ranking_shorter_than_k():
    assert recall_at_k(["a"], {"a", "b"}, 5) == pytest.approx(0.5)


def test_recall_empty_relevant_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_hit_at_k():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0
    assert hit_at_k(ranked, {"b", "z"}, 2) == 1.0


def test_ceiling_at_k():
    assert ceiling_at_k(4, 1) == pytest.approx(0.25)
    assert ceiling_at_k(4, 5) == pytest.approx(1.0)
    assert ceiling_at_k(1, 1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ceiling_at_k(0, 1)


# ---------- agregasi ----------

def test_mean_se_values():
    mean, se = mean_se([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert se == pytest.approx(1.0 / math.sqrt(3))


def test_mean_se_single_value():
    assert mean_se([0.5]) == (0.5, 0.0)
    with pytest.raises(ValueError):
        mean_se([])


def test_evaluate_rankings_by_hand():
    rankings = [["a", "b", "c"], ["x", "y", "z"]]
    relevants = [{"a", "c"}, {"q"}]
    res = evaluate_rankings(rankings, relevants, ks=(1, 3))
    assert res["n_queries"] == 2
    assert res["recall"][1] == pytest.approx((0.25, 0.25))
    assert res["recall"][3] == pytest.approx((0.5, 0.5))
    assert res["hit"][1] == pytest.approx((0.5, 0.5))
    assert res["hit"][3] == pytest.approx((0.5, 0.5))
    assert res["ceiling"][1][0] == pytest.approx(0.75)
    assert res["ceiling"][3][0] == pytest.approx(1.0)


def test_evaluate_rankings_invalid_inputs():
    with pytest.raises(ValueError):
        evaluate_rankings([["a"]], [{"a"}, {"b"}])
    with pytest.raises(ValueError):
        evaluate_rankings([], [])


# ---------- baseline popularitas ----------

def test_popularity_ranking_by_hand(data):
    # m2 muncul di 3 dialog; m1, m3, m4, m5 masing-masing 1 (seri: urut movie_id)
    assert popularity_ranking(data, n=20) == ["m2", "m1", "m3", "m4", "m5"]


def test_popularity_ranking_truncates(data):
    assert popularity_ranking(data, n=2) == ["m2", "m1"]


# ---------- pembagian validasi dan subset ----------

def test_split_validation_properties():
    train, val = split_validation(10, frac=0.2, seed=42)
    assert len(val) == 2 and len(train) == 8
    assert not set(train.tolist()) & set(val.tolist())
    assert sorted(set(train.tolist()) | set(val.tolist())) == list(range(10))
    assert train.tolist() == sorted(train.tolist())
    assert val.tolist() == sorted(val.tolist())


def test_split_validation_deterministic_and_seed_sensitive():
    a = split_validation(100, 0.2, 42)
    b = split_validation(100, 0.2, 42)
    c = split_validation(100, 0.2, 7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_validation_invalid():
    with pytest.raises(ValueError):
        split_validation(1)
    with pytest.raises(ValueError):
        split_validation(10, frac=0.0)
    with pytest.raises(ValueError):
        split_validation(10, frac=1.0)


def test_subset_rows(data):
    sub = subset(data, [0, 2])
    assert sub.emb.shape == (2, 2)
    assert sub.conv_ids == ["c0", "c2"]
    assert sub.contexts == ["konteks 0", "konteks 2"]
    assert sub.items == [ITEMS[0], ITEMS[2]]
    assert np.allclose(sub.emb[1], data.emb[2])


def test_validation_queries_not_in_datastore():
    d = make_data(list(range(0, 100, 10)), [[(f"m{i}", "F")] for i in range(10)])
    train_idx, val_idx = split_validation(len(d.conv_ids), 0.2, 42)
    store = subset(d, train_idx)
    queries = subset(d, val_idx)
    assert not set(store.conv_ids) & set(queries.conv_ids)
    assert len(store.conv_ids) + len(queries.conv_ids) == 10


# ---------- NB melalui jalur evaluasi ----------

def test_rank_nb_by_hand(data):
    queries = np.array([unit(0), unit(90)], dtype=np.float32)
    # k=1: query 0 derajat -> tetangga c0 (m1, m2 seri, urut movie_id); query 90 derajat -> c3 (m5)
    assert rank_nb(queries, data, k=1) == [["m1", "m2"], ["m5"]]


def test_nb_end_to_end_small(data):
    queries = np.array([unit(0), unit(90)], dtype=np.float32)
    rankings = rank_nb(queries, data, k=1)
    res = evaluate_rankings(rankings, [{"m1"}, {"m5"}], ks=(1, 5))
    assert res["recall"][1][0] == pytest.approx(1.0)
    assert res["recall"][5][0] == pytest.approx(1.0)


def test_relevant_sets(data):
    sets = relevant_sets(data)
    assert sets[0] == {"m1", "m2"}
    assert sets[3] == {"m5"}


# ---------- tes dengan data ReDial asli ----------

@needs_data
def test_real_split_sizes():
    d = load_data("train", PROCESSED, EMB)
    n = len(d.conv_ids)
    train_idx, val_idx = split_validation(n)
    assert len(train_idx) + len(val_idx) == n
    assert abs(len(val_idx) - 0.2 * n) <= 1
    ids_train = {d.conv_ids[i] for i in train_idx}
    ids_val = {d.conv_ids[i] for i in val_idx}
    assert not ids_train & ids_val


@needs_data
def test_real_popularity_matches_independent_count():
    d = load_data("train", PROCESSED, EMB)
    counts = Counter()
    for items in d.items:
        for mid in {m for m, _ in items}:
            counts[mid] += 1
    expected = sorted(counts, key=lambda m: (-counts[m], m))[:20]
    assert popularity_ranking(d, n=20) == expected


@needs_data
def test_real_relevant_sets_nonempty():
    for split in ("train", "test"):
        d = load_data(split, PROCESSED, EMB)
        assert all(len(s) >= 1 for s in relevant_sets(d))


# ---------- metrik ranking tambahan kelompok (bukan metrik asli paper) ----------

@pytest.mark.parametrize("ranked,relevant,k,expected", [
    (["a", "x", "b"], {"a", "b"}, 3, 1.5 / (1 + 1 / math.log2(3))),
    (["a", "b", "x"], {"a", "b"}, 3, 1.0),
    (["a", "b"], {"a", "b", "c"}, 2, 1.0),
    (["a"], {"a", "b"}, 5, 1 / (1 + 1 / math.log2(3))),
    (["x", "a"], {"a"}, 2, 1 / math.log2(3)),
    (["x", "a"], {"a"}, 1, 0.0),
    (["x"], {"a"}, 3, 0.0),
    ([], {"a"}, 3, 0.0),
    (["a", "a", "b"], {"a", "b"}, 3, 1.5 / (1 + 1 / math.log2(3))),
    (["a", "a", "b"], {"a", "b"}, 2, 1 / (1 + 1 / math.log2(3))),
    (["x", "x", "a"], {"a"}, 3, 0.5),
])
def test_ndcg_binary_by_hand(ranked, relevant, k, expected):
    assert ndcg_at_k(ranked, relevant, k) == pytest.approx(expected)


@pytest.mark.parametrize("ranked,relevant,k,expected", [
    (["a", "x", "b"], {"a", "b"}, 3, 1.0),
    (["x", "b", "a"], {"a", "b"}, 3, 0.5),
    (["x", "x", "a"], {"a"}, 3, 1 / 3),
    (["x", "x", "a"], {"a"}, 2, 0.0),
    (["a", "a", "b"], {"a", "b"}, 3, 1.0),
    (["a"], {"a", "b"}, 20, 1.0),
    (["x"], {"a"}, 3, 0.0),
    ([], {"a"}, 3, 0.0),
])
def test_reciprocal_rank_by_hand(ranked, relevant, k, expected):
    assert reciprocal_rank_at_k(ranked, relevant, k) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [ndcg_at_k, reciprocal_rank_at_k])
@pytest.mark.parametrize("k", [0, -1])
def test_ranking_metrics_invalid_k(metric, k):
    with pytest.raises(ValueError):
        metric(["a"], {"a"}, k)


@pytest.mark.parametrize("metric", [ndcg_at_k, reciprocal_rank_at_k])
def test_ranking_metrics_empty_ground_truth(metric):
    with pytest.raises(ValueError):
        metric(["a"], set(), 1)


def test_ranking_aggregation_preserves_old_metrics():
    # Query 1: NDCG=1, RR=1; query 2: NDCG=0.5, RR=1/3.
    rankings = [["a", "b", "x"], ["x", "y", "c"]]
    relevants = [{"a", "b"}, {"c"}]
    res = evaluate_rankings(rankings, relevants, ks=(1, 3))
    assert res["n_queries"] == 2
    assert res["ndcg"][3] == pytest.approx((0.75, 0.25))
    assert res["mrr"][3] == pytest.approx((2 / 3, 1 / 3))
    assert res["ndcg"][1] == pytest.approx((0.5, 0.5))
    assert res["mrr"][1] == pytest.approx((0.5, 0.5))
    assert res["recall"][1] == pytest.approx((0.25, 0.25))
    assert res["hit"][1] == pytest.approx((0.5, 0.5))
    assert res["ceiling"][1] == pytest.approx((0.75, 0.25))
    for metric in ("recall", "hit", "ceiling"):
        assert res[metric][3] == pytest.approx((1.0, 0.0))


def test_nb_and_popularity_ranking_metrics_integration(data):
    queries = np.array([unit(0), unit(90)], dtype=np.float32)
    relevants = [{"m1", "m2"}, {"m5"}]
    nb = evaluate_rankings(rank_nb(queries, data, k=1), relevants, ks=(1, 5))
    pop = evaluate_rankings([popularity_ranking(data)] * 2, relevants, ks=(1, 5))
    for metric in ("ndcg", "mrr"):
        assert nb[metric][5] == pytest.approx((1.0, 0.0))
        assert pop[metric][1] == pytest.approx((0.5, 0.5))
    # Popularitas = m2, m1, m3, m4, m5; query kedua mendapat posisi 5.
    assert pop["ndcg"][5] == pytest.approx(((1 + 1 / math.log2(6)) / 2,
                                             (1 - 1 / math.log2(6)) / 2))
    assert pop["mrr"][5] == pytest.approx((0.6, 0.4))
