# scripts/check_env.py
# Tujuan: memastikan library terpasang dan model embedding bisa dijalankan.

import sys

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Python       :", sys.version.split()[0])
print("numpy        :", np.__version__)
print("pandas       :", pd.__version__)
print("scikit-learn :", sklearn.__version__)
print("torch        :", torch.__version__)

print("\nMemuat model all-MiniLM-L6-v2 (pertama kali akan mengunduh sekitar 90 MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

kalimat = [
    "I want a psychological thriller with a plot twist.",
    "Recommend me a mind-bending thriller with a surprise ending.",
    "I am looking for a cute animated movie for kids.",
]
emb = model.encode(kalimat)
print("\nBentuk embedding:", emb.shape)

sim = cosine_similarity(emb)
print("\nMatriks cosine similarity (dibulatkan 3 desimal):")
print(np.round(sim, 3))