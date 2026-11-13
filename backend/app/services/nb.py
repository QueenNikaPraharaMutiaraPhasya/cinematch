# backend/app/services/nb.py
"""Komponen NB (neighborhood-based) dari NBCRS, tanpa training.

Alur: embedding query -> cosine similarity ke konteks train -> top-k tetangga
      -> voting item (w_i = 1) -> peringkat Top-N.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class NBData:
    emb: np.ndarray        # (jumlah percakapan, dim), ternormalisasi
    conv_ids: list         # conversation_id per baris embedding
    contexts: list         # teks konteks per baris embedding
    items: list            # per baris: list[(movie_id, movie_title)]


@dataclass
class Neighbor:
    row: int
    conversation_id: str
    similarity: float
    context: str
    item_titles: list


@dataclass
class Recommendation:
    movie_id: str
    title: str
    votes: int
    sim_sum: float
    supporters: list = field(default_factory=list)  # list[(conversation_id, similarity)]


def load_data(split="train", processed_dir="data/processed", emb_dir="data/embeddings"):
    """Memuat embedding, konteks, dan item untuk satu split. Memeriksa kesejajaran baris."""
    processed_dir, emb_dir = Path(processed_dir), Path(emb_dir)
    conv = pd.read_csv(
        processed_dir / "conversations.csv",
        dtype={"conversation_id": str},
        keep_default_na=False,
    )
    pairs = pd.read_csv(
        processed_dir / "pairs.csv",
        dtype={"conversation_id": str, "movie_id": str},
        keep_default_na=False,
    )
    emb = np.load(emb_dir / f"{split}_emb.npy")

    conv = conv[conv["split"] == split].sort_values("emb_row").reset_index(drop=True)
    if len(conv) != emb.shape[0] or not (conv["emb_row"].to_numpy() == np.arange(len(conv))).all():
        raise ValueError(f"Baris embedding {split} tidak sejajar dengan conversations.csv")

    pairs = pairs[pairs["split"] == split]
    by_conv = {}
    for cid, mid, title in zip(pairs["conversation_id"], pairs["movie_id"], pairs["movie_title"]):
        by_conv.setdefault(cid, []).append((mid, title))

    items = [by_conv[cid] for cid in conv["conversation_id"]]
    return NBData(
        emb=emb,
        conv_ids=conv["conversation_id"].tolist(),
        contexts=conv["context"].tolist(),
        items=items,
    )


def find_neighbors(query_emb, data, k):
    """Top-k percakapan dengan cosine similarity tertinggi (seri: baris lebih awal dulu)."""
    if k < 1:
        raise ValueError("k harus >= 1")
    q = np.asarray(query_emb, dtype=np.float32)
    if q.shape != (data.emb.shape[1],):
        raise ValueError(f"Bentuk query {q.shape} tidak cocok dengan dim embedding {data.emb.shape[1]}")
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Embedding query nol")
    q = q / norm

    sims = data.emb @ q  # = cosine similarity, karena embedding data bernorma 1
    order = np.lexsort((np.arange(len(sims)), -sims))[:k]
    return [
        Neighbor(
            row=int(i),
            conversation_id=data.conv_ids[i],
            similarity=float(sims[i]),
            context=data.contexts[i],
            item_titles=[t for _, t in data.items[i]],
        )
        for i in order
    ]


def vote(neighbors, data):
    """Skor item = jumlah neighbor yang memuat item (w_i = 1).
    Seri: jumlah similarity tetangga pendukung, lalu movie_id (keputusan prototype K9)."""
    table = {}
    for nb in neighbors:
        for movie_id, title in data.items[nb.row]:
            rec = table.get(movie_id)
            if rec is None:
                rec = table[movie_id] = Recommendation(movie_id, title, 0, 0.0)
            rec.votes += 1
            rec.sim_sum += nb.similarity
            rec.supporters.append((nb.conversation_id, nb.similarity))
    return sorted(table.values(), key=lambda r: (-r.votes, -r.sim_sum, r.movie_id))


def recommend(query_emb, data, k=10, n=10):
    """Mengembalikan (daftar tetangga, Top-N rekomendasi)."""
    neighbors = find_neighbors(query_emb, data, k)
    ranked = vote(neighbors, data)
    return neighbors, ranked[:n]