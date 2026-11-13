# backend/app/services/evaluation.py
"""Fungsi evaluasi sesuai docs/evaluation_protocol.md (K12-K19).
Evaluasi memanggil nb.recommend, jalur kode yang sama dengan aplikasi."""

import math
from collections import Counter

import numpy as np

from .nb import NBData, recommend

KS = (1, 5, 20)                              # K13
N_LIST = 20                                  # K15
GRID_K = (1, 5, 10, 30, 60, 90, 120, 150)    # K17
VAL_FRAC = 0.2                               # K17
SEED = 42                                    # K17


def recall_at_k(ranked, relevant, k):
    """Recall@K satu query = |Top-K irisan relevan| / |relevan|."""
    if k < 1:
        raise ValueError("k harus >= 1")
    if not relevant:
        raise ValueError("Himpunan item relevan kosong")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked, relevant, k):
    """1.0 jika ada item relevan di Top-K, selain itu 0.0."""
    if k < 1:
        raise ValueError("k harus >= 1")
    if not relevant:
        raise ValueError("Himpunan item relevan kosong")
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def ceiling_at_k(num_relevant, k):
    """Recall@K tertinggi yang mungkin untuk satu query: min(K, |relevan|) / |relevan|."""
    if k < 1:
        raise ValueError("k harus >= 1")
    if num_relevant < 1:
        raise ValueError("Jumlah item relevan harus >= 1")
    return min(k, num_relevant) / num_relevant


def mean_se(values):
    """(rata-rata, standard error); standard error = simpangan baku sampel / akar n (K19)."""
    values = list(values)
    if not values:
        raise ValueError("Tidak ada nilai")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / math.sqrt(len(arr)))


def evaluate_rankings(rankings, relevants, ks=KS):
    """rankings: list daftar movie_id per query; relevants: list himpunan movie_id per query.
    Hasil: dict dengan (rata-rata, standard error) untuk recall, hit, dan ceiling per K."""
    if len(rankings) != len(relevants):
        raise ValueError("Jumlah peringkat dan jumlah himpunan relevan berbeda")
    if not rankings:
        raise ValueError("Tidak ada query")
    out = {"n_queries": len(rankings), "recall": {}, "hit": {}, "ceiling": {}}
    for k in ks:
        out["recall"][k] = mean_se(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants))
        out["hit"][k] = mean_se(hit_at_k(r, rel, k) for r, rel in zip(rankings, relevants))
        out["ceiling"][k] = mean_se(ceiling_at_k(len(rel), k) for rel in relevants)
    return out


def relevant_sets(data):
    """Item relevan per percakapan (K12): himpunan movie_id dialog itu."""
    return [{mid for mid, _ in items} for items in data.items]


def popularity_ranking(data, n=N_LIST):
    """Baseline popularitas (K16): film diurutkan menurut jumlah dialog datastore yang memuatnya;
    seri diurutkan menurut movie_id."""
    counts = Counter()
    for items in data.items:
        counts.update({mid for mid, _ in items})
    ordered = sorted(counts, key=lambda m: (-counts[m], m))
    return ordered[:n]


def split_validation(n, frac=VAL_FRAC, seed=SEED):
    """Membagi nomor baris 0..n-1 menjadi (indeks datastore, indeks query validasi), keduanya terurut."""
    if n < 2:
        raise ValueError("Butuh minimal 2 baris")
    if not 0 < frac < 1:
        raise ValueError("frac harus di antara 0 dan 1")
    n_val = min(max(int(round(n * frac)), 1), n - 1)
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def subset(data, rows):
    """Datastore baru dari baris tertentu (untuk mengeluarkan query validasi dari datastore)."""
    rows = [int(r) for r in rows]
    return NBData(
        emb=data.emb[rows],
        conv_ids=[data.conv_ids[i] for i in rows],
        contexts=[data.contexts[i] for i in rows],
        items=[data.items[i] for i in rows],
    )


def rank_nb(query_embs, datastore, k, n=N_LIST):
    """Peringkat NB (daftar movie_id) untuk tiap embedding query, memakai nb.recommend."""
    rankings = []
    for q in query_embs:
        _, recs = recommend(q, datastore, k=k, n=n)
        rankings.append([r.movie_id for r in recs])
    return rankings