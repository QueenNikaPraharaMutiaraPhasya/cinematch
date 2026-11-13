# scripts/demo_query.py
# Tujuan: mencoba query teks bebas dan memeriksa panjang token. BUKAN evaluasi.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.nb import load_data
from backend.app.services.recommender import Recommender

K, N = 10, 5
OUT_PATH = Path("docs/demo_query_output.txt")
lines = []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


# Query ILUSTRASI (dibuat sendiri, bukan dari dataset)
QUERIES = [
    "I want a psychological thriller with a plot twist, but without too much violence.",
    "something like Inception",
    "a feel-good romantic comedy for a rainy evening",
    "hi",
    "asdfgh qwerty",
    "",
]

rec = Recommender()
log(f"Model siap. Datastore train: {len(rec.data.conv_ids)} percakapan | K={K}, N={N}")

# Pemeriksaan panjang token (batas model)
batas = rec.model.max_seq_length
tok = rec.model.tokenizer
log(f"\n=== PANJANG TOKEN (batas model {batas}) ===")
for split in ("train", "test"):
    d = rec.data if split == "train" else load_data("test")
    ids = tok(d.contexts, add_special_tokens=True, truncation=False)["input_ids"]
    panjang = [len(x) for x in ids]
    lebih = sum(1 for x in panjang if x > batas)
    log(f"{split}: {len(panjang)} konteks | token maks {max(panjang)} | melebihi batas: {lebih}")

for q in QUERIES:
    res = rec.recommend_text(q, k=K, n=N)
    log(f"\n=== QUERY: {q!r} ===")
    log("status:", res.status)
    if res.status != "ok":
        log("pesan:", res.message)
        continue
    log(f"top_similarity (tetangga #1): {res.top_similarity:.3f}")
    log("3 tetangga teratas:")
    for i, nb in enumerate(res.neighbors[:3], 1):
        log(f"  #{i} sim={nb.similarity:.3f} | {nb.context[:100]}")
    log(f"Top-{N} rekomendasi:")
    for i, r in enumerate(res.recommendations, 1):
        log(f"  {i}. {r.title} | suara {r.votes}/{K}")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\nSelesai. Hasil juga disimpan di", OUT_PATH)