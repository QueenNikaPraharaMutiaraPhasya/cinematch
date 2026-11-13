# backend/app/main.py
"""API CineMatch (prototype implementasi konsep NBCRS).
Jalankan dari folder proyek (cinematch):  python -m uvicorn backend.app.main:app --port 8000"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    NeighborOut,
    RecommendationOut,
    RecommendRequest,
    RecommendResponse,
    SupporterOut,
)
from .services.recommender import MODEL_NAME, Recommender

MAX_SUPPORTERS = 5  # pendukung teratas yang ditampilkan per rekomendasi


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model dan datastore dimuat sekali saat server menyala
    app.state.recommender = Recommender()
    yield


app = FastAPI(title="CineMatch API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request):
    rec = request.app.state.recommender
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "datastore_conversations": len(rec.data.conv_ids),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(body: RecommendRequest, request: Request):
    rec = request.app.state.recommender
    try:
        res = rec.recommend_text(body.query, k=body.k, n=body.n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    neighbors = [
        NeighborOut(
            rank=i,
            conversation_id=nb.conversation_id,
            similarity=round(nb.similarity, 4),
            context=nb.context,
            items=nb.item_titles,
        )
        for i, nb in enumerate(res.neighbors[: body.show_neighbors], 1)
    ]
    recommendations = [
        RecommendationOut(
            rank=i,
            movie_id=r.movie_id,
            title=r.title,
            votes=r.votes,
            sim_sum=round(r.sim_sum, 4),
            explanation=(
                f"Disarankan di {r.votes} dari {res.k} percakapan serupa "
                f"(total similarity {r.sim_sum:.2f})."
            ),
            supporters=[
                SupporterOut(conversation_id=cid, similarity=round(s, 4))
                for cid, s in r.supporters[:MAX_SUPPORTERS]
            ],
        )
        for i, r in enumerate(res.recommendations, 1)
    ]
    return RecommendResponse(
        status=res.status,
        query=res.query,
        k=res.k,
        n=res.n,
        top_similarity=None if res.top_similarity is None else round(res.top_similarity, 4),
        message=res.message,
        neighbors=neighbors,
        recommendations=recommendations,
    )