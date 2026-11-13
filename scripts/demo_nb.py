# scripts/demo_nb.py
# Tujuan: mencoba retrieval + voting secara manual. BUKAN evaluasi.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.nb import find_neighbors, load_data, recommend

K, N = 10, 5
OUT_PATH = Path("docs/demo_nb_output.txt")
lines = []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


train = load_data("train")
test = load_data("test")
train_movie_ids = {mid for items in train.items for mid, _ in items}
log(f"Train: {len(train.conv_ids)} percakapan | Test: {len(test.conv_ids)} percakapan | K={K}, N={N}")

# Uji sanity: query = embedding train baris 0 -> tetangga #1 harus dirinya sendiri
nbs = find_neighbors(train.emb[0], train, k=3)
log("\n=== SANITY ===")
log("Tetangga #1 dari embedding train baris 0 adalah dirinya sendiri?",
    nbs[0].row == 0, "| sim =", round(nbs[0].similarity, 4))
sims = [n.similarity for n in nbs]
log("Urutan similarity menurun?", sims == sorted(sims, reverse=True))

for row in range(3):
    query = test.emb[row]
    neighbors, recs = recommend(query, train, k=K, n=N)

    log(f"\n=== TEST conv {test.conv_ids[row]} ===")
    log("Konteks:", test.contexts[row][:200])
    log("Item sebenarnya (hanya untuk perbandingan manual):")
    for mid, title in test.items[row]:
        status = "ada di train" if mid in train_movie_ids else "TIDAK ada di train"
        log(f"  - {title} [{status}]")

    log(f"\nTetangga (top-{K}):")
    for i, nb in enumerate(neighbors, 1):
        log(f"  #{i} sim={nb.similarity:.3f} | conv {nb.conversation_id} | {nb.context[:90]}")
        log(f"       item: {nb.item_titles[:3]}")

    log(f"\nTop-{N} rekomendasi:")
    for i, r in enumerate(recs, 1):
        log(f"  {i}. {r.title} | suara {r.votes}/{K} | jumlah sim pendukung {r.sim_sum:.3f}")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\nSelesai. Hasil juga disimpan di", OUT_PATH)