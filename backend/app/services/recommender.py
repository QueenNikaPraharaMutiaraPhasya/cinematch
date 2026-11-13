# backend/app/services/recommender.py
"""Pembungkus NB untuk query teks bebas. Model embedding dimuat sekali."""

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from .nb import NBData, load_data, recommend

MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RecommendationResult:
    status: str                      # "ok" atau "empty_query"
    query: str
    k: int
    n: int
    top_similarity: float | None = None
    neighbors: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    message: str = ""


class Recommender:
    def __init__(self, data: NBData | None = None, model=None):
        self.data = data if data is not None else load_data("train")
        self.model = (
            model if model is not None
            else SentenceTransformer(MODEL_NAME, device="cpu")
        )

    def encode(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)

    def recommend_text(self, query, k=10, n=10) -> RecommendationResult:
        if k < 1 or n < 1:
            raise ValueError("k dan n harus >= 1")
        if not isinstance(query, str) or not query.strip():
            return RecommendationResult(
                status="empty_query",
                query="",
                k=k,
                n=n,
                message="Query kosong. Tulis preferensi film Anda dalam bahasa Inggris.",
            )
        clean = " ".join(query.split())  # K10: hanya merapikan spasi
        neighbors, recs = recommend(self.encode(clean), self.data, k=k, n=n)
        return RecommendationResult(
            status="ok",
            query=clean,
            k=k,
            n=n,
            top_similarity=neighbors[0].similarity,
            neighbors=neighbors,
            recommendations=recs,
        )