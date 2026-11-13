# scripts/build_embeddings.py
# Tujuan: mengubah konteks percakapan (train & test) menjadi embedding MiniLM
# dan menyimpannya. Data mentah dan pairs.csv TIDAK diubah.
# Hasil: data/embeddings/train_emb.npy, test_emb.npy,
#        data/processed/conversations.csv, docs/embedding_report.txt

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PAIRS_PATH = Path("data/processed/pairs.csv")
CONV_PATH = Path("data/processed/conversations.csv")
EMB_DIR = Path("data/embeddings")
REPORT_PATH = Path("docs/embedding_report.txt")
MODEL_NAME = "all-MiniLM-L6-v2"

lines = []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


# keep_default_na=False agar teks seperti "NA" atau "null" tidak diubah jadi kosong
pairs = pd.read_csv(
    PAIRS_PATH,
    dtype={"conversation_id": str, "movie_id": str},
    keep_default_na=False,
)

# Satu baris per percakapan
maks_konteks = pairs.groupby(["split", "conversation_id"])["context"].nunique().max()
log("Maks jumlah konteks berbeda dalam satu percakapan (harus 1):", maks_konteks)

conv = (
    pairs.drop_duplicates(subset=["split", "conversation_id"])
    [["split", "conversation_id", "context"]]
    .reset_index(drop=True)
)
conv["emb_row"] = conv.groupby("split").cumcount()
log("Percakapan per split:", conv["split"].value_counts().to_dict())

log("\nMemuat model", MODEL_NAME, "(CPU)")
model = SentenceTransformer(MODEL_NAME, device="cpu")
log("max_seq_length model:", getattr(model, "max_seq_length", "tidak diketahui"))

EMB_DIR.mkdir(parents=True, exist_ok=True)
embs = {}
for split in ("train", "test"):
    texts = conv.loc[conv["split"] == split, "context"].tolist()
    log(f"\nMengenkode {len(texts)} konteks {split} ...")
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(EMB_DIR / f"{split}_emb.npy", emb)
    embs[split] = emb

conv.to_csv(CONV_PATH, index=False, encoding="utf-8")

log("\n=== VALIDASI ===")
for split, emb in embs.items():
    n_conv = int((conv["split"] == split).sum())
    norms = np.linalg.norm(emb, axis=1)
    log(f"{split}: bentuk {emb.shape}, dtype {emb.dtype}, jumlah baris = jumlah percakapan? {emb.shape[0] == n_conv}")
    log(f"  nilai NaN/Inf: {int((~np.isfinite(emb)).sum())}, norma min/maks: {norms.min():.4f} / {norms.max():.4f}")

train_conv = conv[conv["split"] == "train"].reset_index(drop=True)
test_conv = conv[conv["split"] == "test"].reset_index(drop=True)

# Cek kesejajaran baris: enkode ulang 3 konteks train pertama, bandingkan dengan file
re_emb = model.encode(
    train_conv["context"].head(3).tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True,
)
log("\nEnkode ulang 3 konteks train pertama cocok dengan file tersimpan:",
    bool(np.allclose(re_emb, embs["train"][:3], atol=1e-4)))

# Uji kasar saja (BUKAN evaluasi): tetangga terdekat untuk konteks test pertama
sims = embs["train"] @ embs["test"][0]
top = np.argsort(-sims)[:3]
log("\n=== UJI KASAR: tetangga train terdekat untuk konteks test pertama ===")
log("Konteks test:", test_conv.loc[0, "context"][:200])
for rank, idx in enumerate(top, 1):
    log(f"  #{rank} sim={sims[idx]:.3f} | conv {train_conv.loc[idx, 'conversation_id']} | "
        f"{train_conv.loc[idx, 'context'][:150]}")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\nSelesai. Embedding di", EMB_DIR, "| laporan di", REPORT_PATH)