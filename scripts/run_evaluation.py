# scripts/run_evaluation.py
# Tujuan: menjalankan evaluasi sesuai docs/evaluation_protocol.md (K12-K19).
#   Tahap 1: memilih k pada validasi (dialog train 80/20, seed 42).
#   Tahap 2: mengevaluasi test SATU kali dengan k terpilih.
# Hasil: docs/evaluation_report.txt dan docs/evaluation_results.csv

import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.evaluation import (
    GRID_K,
    KS,
    N_LIST,
    SEED,
    evaluate_rankings,
    popularity_ranking,
    rank_nb,
    relevant_sets,
    split_validation,
    subset,
)
from backend.app.services.nb import load_data

REPORT_PATH = Path("docs/evaluation_report.txt")
CSV_PATH = Path("docs/evaluation_results.csv")
lines, records = [], []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


def pct(res, metric, k):
    mean, se = res[metric][k]
    return f"{100 * mean:6.2f} +- {100 * se:4.2f}"


def model_cells(res):
    return [pct(res, "recall", K) for K in KS] + [pct(res, "hit", K) for K in KS]


def table(rows):
    cols = [f"Recall@{K}" for K in KS] + [f"Hit@{K}" for K in KS]
    log(f"{'Model':<30}" + "".join(f"{c:>15}" for c in cols))
    for name, cells in rows:
        log(f"{name:<30}" + "".join(f"{c:>15}" for c in cells))


def add_records(split, model, k_nb, res):
    for metric in ("recall", "hit", "ceiling"):
        for K in KS:
            mean, se = res[metric][K]
            records.append(
                {
                    "split": split,
                    "model": model,
                    "k_neighbors": k_nb,
                    "metric": metric,
                    "K": K,
                    "mean_pct": round(100 * mean, 4),
                    "se_pct": round(100 * se, 4),
                    "n_queries": res["n_queries"],
                }
            )


log("EVALUASI CINEMATCH (prototype implementasi konsep NBCRS; bukan replikasi paper)")
log("Protokol: docs/evaluation_protocol.md (K12-K19)")
log(
    f"Python {platform.python_version()} | numpy {version('numpy')} | pandas {version('pandas')}"
    f" | torch {version('torch')} | sentence-transformers {version('sentence-transformers')}"
)
log(f"Embedding: all-MiniLM-L6-v2 | seed {SEED} | panjang daftar N = {N_LIST} | grid k = {GRID_K}")

train = load_data("train")
test = load_data("test")
n = len(train.conv_ids)
store_idx, val_idx = split_validation(n)
store, val = subset(train, store_idx), subset(train, val_idx)
assert not set(store.conv_ids) & set(val.conv_ids), "Query validasi masih ada di datastore!"
log(f"Train: {n} dialog -> datastore sementara {len(store.conv_ids)}, query validasi {len(val.conv_ids)}")
log(f"Test: {len(test.conv_ids)} dialog (query), datastore akhir = {n} dialog train")

# ---------------- Tahap 1: pemilihan k pada validasi ----------------
log("\n=== TAHAP 1: PEMILIHAN k PADA VALIDASI (K17) ===")
val_rel = relevant_sets(val)
grid_results, avg_len = [], {}
for k in GRID_K:
    rankings = rank_nb(val.emb, store, k=k, n=N_LIST)
    res = evaluate_rankings(rankings, val_rel)
    grid_results.append((k, res))
    avg_len[k] = sum(len(r) for r in rankings) / len(rankings)
    add_records("validation", "NB zero-shot", k, res)
    print(f"  selesai k={k}", flush=True)

pop_store = popularity_ranking(store, N_LIST)
pop_val = evaluate_rankings([pop_store] * len(val_rel), val_rel)
add_records("validation", "Popularitas", None, pop_val)

rows = [(f"NB zero-shot k={k}", model_cells(res)) for k, res in grid_results]
rows.append(("Popularitas (datastore 80%)", model_cells(pop_val)))
table(rows)
log("Satuan: persen; format rata-rata +- standard error. Query validasi:", len(val_rel))
log("Panjang daftar rata-rata per k:", {k: round(v, 1) for k, v in avg_len.items()})

best_k, best_res = max(grid_results, key=lambda kr: (kr[1]["recall"][20][0], -kr[0]))
log(f"\nk terpilih (Recall@20 validasi tertinggi; seri: k lebih kecil): k = {best_k}")

# ---------------- Tahap 2: evaluasi akhir pada test ----------------
log(f"\n=== TAHAP 2: EVALUASI AKHIR PADA TEST (K18), k = {best_k} ===")
test_rel = relevant_sets(test)
nb_test = evaluate_rankings(rank_nb(test.emb, train, k=best_k, n=N_LIST), test_rel)
pop_train = popularity_ranking(train, N_LIST)
assert len(pop_train) == N_LIST
pop_test = evaluate_rankings([pop_train] * len(test_rel), test_rel)
add_records("test", "NB zero-shot", best_k, nb_test)
add_records("test", "Popularitas", None, pop_test)

table(
    [
        (f"NB zero-shot k={best_k}", model_cells(nb_test)),
        ("Popularitas", model_cells(pop_test)),
        ("Plafon (K14)", [pct(nb_test, "ceiling", K) for K in KS] + ["-"] * len(KS)),
    ]
)
log("Satuan: persen; format rata-rata +- standard error. Query test:", len(test_rel))

# ---------------- Diagnostik (bukan metrik protokol) ----------------
log("\n=== DIAGNOSTIK (bukan metrik protokol) ===")
train_items = {mid for items in train.items for mid, _ in items}
total = sum(len(r) for r in test_rel)
reach = sum(1 for rel in test_rel for m in rel if m in train_items)
log(
    f"Pasangan test dengan item yang ada di train: {reach} dari {total} "
    f"(laporan preprocessing: 3636 dari 4001) -> cocok? {reach == 3636 and total == 4001}"
)
log("Plafon realistis (hanya film yang ada di train yang bisa ditebak), rata-rata persen:")
for K in KS:
    vals = [min(K, len(rel & train_items)) / len(rel) for rel in test_rel]
    log(f"  Recall@{K}: {100 * sum(vals) / len(vals):.2f}")

leaky = evaluate_rankings(rank_nb(val.emb, train, k=best_k, n=N_LIST), val_rel)
log(
    f"ILUSTRASI kebocoran (query validasi TIDAK dikeluarkan dari datastore, k={best_k}): "
    f"Recall@20 = {100 * leaky['recall'][20][0]:.2f} vs bersih {100 * best_res['recall'][20][0]:.2f}. "
    "Angka bocor tidak valid; ditampilkan hanya untuk menunjukkan mengapa query dikeluarkan."
)

log("\n=== KETERBATASAN ===")
log("- Hanya 1229 dari 1342 dialog test dipakai (filter K1-K3); tidak sebanding langsung dengan angka paper.")
log("- Satu split validasi dengan satu seed; tidak ada interval kepercayaan antar-seed.")
log("- Datastore saat tuning (80%) lebih kecil daripada datastore akhir (100%).")
log("- Top-N dapat memuat ID ganda untuk film yang sama (K5).")
log("- Pengaturan tidak boleh diubah setelah hasil test dilihat.")

pd.DataFrame(records).to_csv(CSV_PATH, index=False, encoding="utf-8")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\nSelesai. Laporan:", REPORT_PATH, "| tabel:", CSV_PATH)