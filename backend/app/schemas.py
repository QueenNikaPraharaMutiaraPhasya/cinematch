# backend/app/schemas.py
"""Bentuk permintaan dan balasan API CineMatch."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., max_length=1000, description="Preferensi film dalam bahasa Inggris")
    k: int = Field(90, ge=1, le=200, description="Jumlah percakapan tetangga untuk voting")
    n: int = Field(10, ge=1, le=50, description="Jumlah film yang ditampilkan")
    show_neighbors: int = Field(10, ge=0, le=50, description="Jumlah tetangga yang ditampilkan")


class NeighborOut(BaseModel):
    rank: int
    conversation_id: str
    similarity: float
    context: str
    items: list[str]


class SupporterOut(BaseModel):
    conversation_id: str
    similarity: float


class RecommendationOut(BaseModel):
    rank: int
    movie_id: str
    title: str
    votes: int
    sim_sum: float
    explanation: str
    supporters: list[SupporterOut]


class RecommendResponse(BaseModel):
    status: str                      # "ok" atau "empty_query"
    query: str
    k: int
    n: int
    top_similarity: float | None = None
    message: str = ""
    neighbors: list[NeighborOut] = Field(default_factory=list)
    recommendations: list[RecommendationOut] = Field(default_factory=list)