# scripts/run_improvement.py
# Tujuan: mencari perbaikan NB murni (tanpa MB, tanpa pelatihan) pada ReDial.
#   Tahap 1: grid skema bobot tetangga x k (x encoder opsional) pada VALIDASI saja
#            (dialog train 80/20, seed 42, seleksi memakai Recall@20 validasi, sama seperti K17).
#   Tahap 2: test dievaluasi SEKALI untuk konfigurasi terpilih, dibandingkan dengan baseline
#            (all-MiniLM-L6-v2, w_i = 1, k = 90) memakai selisih berpasangan per dialog.
# Keluaran baru (laporan lama tidak ditimpa): docs/improvement_report.txt, docs/improvement_results.csv
#
# Jalankan dari root repo:
#   .\.venv\Scripts\python.exe scripts\run_improvement.py
#   .\.venv\Scripts\python.exe scripts\run_improvement.py --models all-mpnet-base-v2 BAAI/bge-small-en-v1.5

import argparse
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.evaluation import (
    KS,
    N_LIST,
    SEED,
    evaluate_rankings,
    mean_se,
    rank_nb,
    recall_at_k,
    relevant_sets,
    split_validation,
    subset,
)
from backend.app.services.nb import aggregate, load_data

BASE_MODEL = "all-MiniLM-L6-v2"
BASELINE = (BASE_MODEL, "uniform", 1.0, 90)  # konfigurasi protokol lama (k terpilih = 90)
K_GRID = (5, 10, 30, 60, 90, 120, 150, 200, 300)
SCHEMES = [
    ("uniform", 1.0),
    ("power", 2.0), ("power", 4.0), ("power", 8.0),
    ("softmax", 0.10), ("softmax", 0.05), ("softmax", 0.02),
]
EMB_DIR = Path("data/embeddings")
REPORT_PATH = Path("docs/improvement_report.txt")
CSV_PATH = Path("docs/improvement_results.csv")
lines, records = [], []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


def pct(res, metric, k):
    mean, se = res[metric][k]
    return f"{100 * mean:6.2f} +- {100 * se:4.2f}"


def cfg_name(cfg):
    model, scheme, param, k = cfg
    w = "w=1" if scheme == "uniform" else f"{scheme}({param:g})"
    return f"{model} | {w} | k={k}"


def get_embeddings(model_name, train, test):
    """Embedding (train, test) untuk satu encoder. Encoder dasar memakai file yang sudah ada;
    encoder lain meng-encode kolom konteks dan menyimpannya sebagai cache."""
    if model_name == BASE_MODEL:
        return train.emb, test.emb
    slug = model_name.replace("/", "_")
    out = []
    for split, data in (("train", train), ("test", test)):
        path = EMB_DIR / f"cache_{slug}_{split}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            from sentence_transformers import SentenceTransformer  # impor lambat: hanya bila perlu

            log(f"Meng-encode {len(data.contexts)} konteks {split} dengan {model_name} ...")
            model = SentenceTransformer(model_name, device="cpu")
            emb = model.encode(
                data.contexts, batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=True,
            ).astype(np.float32)
            np.save(path, emb)
        if emb.shape[0] != len(data.contexts):
            raise ValueError(f"Cache {path} tidak sejajar dengan data {split}")
        out.append(emb)
    return out[0], out[1]


def grid(model_name, q_emb, store, q_rel):
    """Grid skema x k pada query validasi. Jalur cepat: similarity dihitung sekali per encoder."""
    sims = q_emb @ store.emb.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, : max(K_GRID)]  # seri: baris awal dulu
    results = []
    for scheme, param in SCHEMES:
        for k in K_GRID:
            rankings = []
            for qi in range(len(q_emb)):
                idx = order[qi, :k]
                recs = aggregate(idx, sims[qi, idx], store, scheme, param)[:N_LIST]
                rankings.append([r.movie_id for r in recs])
            res = evaluate_rankings(rankings, q_rel)
            results.append(((model_name, scheme, param, k), res, rankings))
            print(f"  selesai {cfg_name((model_name, scheme, param, k))}", flush=True)
    return results


def add_records(split, cfg, res):
    model, scheme, param, k = cfg
    for metric in ("recall", "hit", "ndcg", "mrr"):
        for K in KS:
            mean, se = res[metric][K]
            records.append({
                "split": split, "encoder": model, "scheme": scheme, "param": param,
                "k_neighbors": k, "metric": metric, "K": K,
                "mean_pct": round(100 * mean, 4), "se_pct": round(100 * se, 4),
                "n_queries": res["n_queries"],
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="encoder tambahan (sentence-transformers)")
    args = ap.parse_args()
    models = [BASE_MODEL] + [m for m in args.models if m != BASE_MODEL]

    log("PERBAIKAN NB CINEMATCH (NB murni, tanpa MB; dataset ReDial; bukan replikasi paper)")
    log("Pasca-hoc: protokol lama sudah melihat test. Seleksi di sini HANYA memakai validasi;")
    log("test dievaluasi satu kali untuk konfigurasi terpilih. Evaluasi ini harus dilaporkan sebagai eksperimen lanjutan.")
    log(f"Python {platform.python_version()} | numpy {np.__version__} | seed {SEED} | N = {N_LIST}")
    log(f"Encoder: {models} | grid k = {K_GRID} | skema = {SCHEMES}")

    train, test = load_data("train"), load_data("test")
    n = len(train.conv_ids)
    store_idx, val_idx = split_validation(n)
    val_rel = relevant_sets(subset(train, val_idx))
    log(f"Train {n} -> datastore sementara {len(store_idx)}, query validasi {len(val_idx)}; test {len(test.conv_ids)}")

    embs = {m: get_embeddings(m, train, test) for m in models}

    # ---------------- Tahap 1: validasi ----------------
    log("\n=== TAHAP 1: GRID PADA VALIDASI (seleksi: Recall@20) ===")
    all_results = []
    for m in models:
        tr_emb, _ = embs[m]
        store = subset(replace(train, emb=tr_emb), store_idx)
        q_emb = tr_emb[val_idx]
        res_m = grid(m, q_emb, store, val_rel)
        all_results.extend(res_m)

        # Pemeriksaan: jalur cepat == jalur aplikasi (nb.recommend) untuk konfigurasi baseline/uniform
        if m == BASE_MODEL:
            cfg0 = (m, "uniform", 1.0, 90)
            fast = next(r for c, _, r in res_m if c == cfg0)[:50]
            slow = rank_nb(q_emb[:50], store, k=90, n=N_LIST)
            assert fast == slow, "Jalur cepat grid berbeda dari nb.recommend!"
            log("Pemeriksaan jalur cepat vs nb.recommend (50 query): cocok")

    for cfg, res, _ in all_results:
        add_records("validation", cfg, res)

    ranked = sorted(all_results, key=lambda r: (-r[1]["recall"][20][0], r[0][3], r[0][1] != "uniform"))
    log("\n10 konfigurasi terbaik pada validasi:")
    log(f"{'Konfigurasi':<58}{'Recall@1':>15}{'Recall@5':>15}{'Recall@20':>15}{'NDCG@20':>15}")
    for cfg, res, _ in ranked[:10]:
        log(f"{cfg_name(cfg):<58}{pct(res, 'recall', 1):>15}{pct(res, 'recall', 5):>15}"
            f"{pct(res, 'recall', 20):>15}{pct(res, 'ndcg', 20):>15}")
    base_val = next(r for r in all_results if r[0] == BASELINE)
    log(f"\nBaseline validasi {cfg_name(BASELINE)}: Recall@20 = {pct(base_val[1], 'recall', 20)}")
    best_cfg, best_val, _ = ranked[0]
    log(f"Terpilih (Recall@20 validasi tertinggi): {cfg_name(best_cfg)}")

    # ---------------- Tahap 2: test satu kali ----------------
    log("\n=== TAHAP 2: TEST (satu kali), datastore = seluruh train ===")
    test_rel = relevant_sets(test)
    per_query, test_res = {}, {}
    for name, cfg in (("baseline", BASELINE), ("terpilih", best_cfg)):
        model, scheme, param, k = cfg
        tr_emb, te_emb = embs[model]
        rankings = rank_nb(te_emb, replace(train, emb=tr_emb), k=k, n=N_LIST, scheme=scheme, param=param)
        test_res[name] = evaluate_rankings(rankings, test_rel)
        per_query[name] = {K: [recall_at_k(r, rel, K) for r, rel in zip(rankings, test_rel)] for K in KS}
        add_records("test", cfg, test_res[name])
        log(f"{name:<9} {cfg_name(cfg)}")
        log("          " + " | ".join(f"Recall@{K} {pct(test_res[name], 'recall', K)}" for K in KS)
            + f" | NDCG@20 {pct(test_res[name], 'ndcg', 20)} | MRR@20 {pct(test_res[name], 'mrr', 20)}")

    if best_cfg == BASELINE:
        log("\nTidak ada konfigurasi yang mengalahkan baseline pada validasi; tidak ada perbaikan yang diklaim.")
    else:
        log("\nSelisih berpasangan terpilih - baseline per dialog (poin persen; rata-rata +- standard error):")
        for K in KS:
            diff = np.array(per_query["terpilih"][K]) - np.array(per_query["baseline"][K])
            mean, se = mean_se(diff)
            flag = "  <- melewati 2 SE" if abs(mean) > 2 * se else "  (dalam derau)"
            log(f"  Recall@{K}: {100 * mean:+.2f} +- {100 * se:.2f}{flag}")

    log("\n=== KETERBATASAN ===")
    log("- Test sudah pernah dilihat pada protokol lama; ini eksperimen lanjutan, bukan hasil murni.")
    log("- Banyak konfigurasi dicoba pada satu split validasi dan satu seed: ada risiko overfit ke validasi.")
    log("- Jika encoder selain all-MiniLM-L6-v2 terpilih, ubah MODEL_NAME di recommender.py dan buat ulang embedding train.")
    log("- Angka tidak sebanding langsung dengan paper (unit sampel dan protokol berbeda).")

    pd.DataFrame(records).to_csv(CSV_PATH, index=False, encoding="utf-8")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\nSelesai. Laporan:", REPORT_PATH, "| tabel:", CSV_PATH)


if __name__ == "__main__":
    main()