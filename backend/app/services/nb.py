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
    score: float = 0.0                              # jumlah bobot w_i; skema "uniform": sama dengan votes


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


def neighbor_weights(sims, scheme="uniform", param=1.0):
    """Bobot w_i tiap tetangga pada Persamaan 1 NBCRS (score(v|q) = sum w_i * 1(v_i = v)).
    - "uniform": w_i = 1 (pengaturan paper; perilaku asli prototype)
    - "power":   w_i = max(sim, 0) ** param   (param > 1 mempertajam tetangga terdekat)
    - "softmax": w_i = exp((sim - sim_max) / param)  (param = temperatur; kecil = tajam)
    Tetap NB murni: tanpa pelatihan, hanya mengubah cara tetangga ditimbang."""
    sims = np.asarray(sims, dtype=np.float64)
    if scheme == "uniform":
        return np.ones_like(sims)
    if scheme == "power":
        return np.clip(sims, 0.0, None) ** param
    if scheme == "softmax":
        if param <= 0:
            raise ValueError("temperatur harus > 0")
        return np.exp((sims - sims.max()) / param)
    raise ValueError(f"skema bobot tidak dikenal: {scheme}")


def aggregate(rows, sims, data, scheme="uniform", param=1.0, with_supporters=False):
    """Inti voting: rows = indeks baris tetangga, sims = similarity-nya (urut menurun).
    Seri: jumlah similarity tetangga pendukung, lalu movie_id (keputusan prototype K9)."""
    weights = neighbor_weights(sims, scheme, param)
    table = {}
    for row, sim, w in zip(rows, sims, weights):
        row, sim = int(row), float(sim)
        for movie_id, title in data.items[row]:
            rec = table.get(movie_id)
            if rec is None:
                rec = table[movie_id] = Recommendation(movie_id, title, 0, 0.0)
            rec.votes += 1
            rec.sim_sum += sim
            rec.score += float(w)
            if with_supporters:
                rec.supporters.append((data.conv_ids[row], sim))
    return sorted(table.values(), key=lambda r: (-r.score, -r.sim_sum, r.movie_id))


def vote(neighbors, data, scheme="uniform", param=1.0):
    """Skor item = jumlah bobot tetangga yang memuat item. Default w_i = 1 (jumlah neighbor)."""
    return aggregate(
        [nb.row for nb in neighbors],
        [nb.similarity for nb in neighbors],
        data, scheme, param, with_supporters=True,
    )


def recommend(query_emb, data, k=10, n=10, scheme="uniform", param=1.0):
    """Mengembalikan (daftar tetangga, Top-N rekomendasi)."""
    neighbors = find_neighbors(query_emb, data, k)
    ranked = vote(neighbors, data, scheme, param)
    return neighbors, ranked[:n]