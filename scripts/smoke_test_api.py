# scripts/smoke_test_api.py
# Tujuan: uji end-to-end terhadap server sungguhan yang sedang berjalan.
# Bukan evaluasi kualitas rekomendasi. Hasil: docs/smoke_test_report.txt
# Pemakaian (backend harus menyala):  python scripts\smoke_test_api.py

import statistics
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(errors="replace")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
REPORT_PATH = Path("docs/smoke_test_report.txt")
lines, failures = [], []
total = 0


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


def check(name, condition, detail=""):
    global total
    total += 1
    if not condition:
        failures.append(f"{name} {detail}".strip())
        log(f"  [GAGAL] {name} {detail}".rstrip())


def post(client, payload):
    t0 = time.perf_counter()
    r = client.post("/recommend", json=payload)
    return r, (time.perf_counter() - t0) * 1000


def check_ok_structure(label, body, k, n, show=10):
    nbs, recs = body["neighbors"], body["recommendations"]
    sims = [x["similarity"] for x in nbs]
    votes = [x["votes"] for x in recs]
    check(f"{label}: status ok", body["status"] == "ok", body["status"])
    check(f"{label}: similarity tetangga menurun", sims == sorted(sims, reverse=True))
    check(f"{label}: suara menurun", votes == sorted(votes, reverse=True))
    check(f"{label}: suara 1..k", all(1 <= v <= k for v in votes), str(votes))
    check(f"{label}: 1 <= jumlah rekomendasi <= n", 1 <= len(recs) <= n, str(len(recs)))
    check(f"{label}: jumlah tetangga = min(k, show)", len(nbs) == min(k, show), str(len(nbs)))
    check(f"{label}: rank berurutan", [r["rank"] for r in recs] == list(range(1, len(recs) + 1)))
    check(f"{label}: movie_id unik", len({r["movie_id"] for r in recs}) == len(recs))
    check(f"{label}: judul tidak kosong", all(r["title"].strip() for r in recs))
    check(f"{label}: pendukung 1..5", all(1 <= len(r["supporters"]) <= 5 for r in recs))
    if nbs:
        check(f"{label}: top_similarity = tetangga #1", body["top_similarity"] == nbs[0]["similarity"])


def main():
    with httpx.Client(base_url=BASE, timeout=60) as client:
        try:
            health = client.get("/health").json()
        except httpx.ConnectError:
            raise SystemExit(f"Tidak dapat terhubung ke {BASE}. Nyalakan backend dulu.")
        log(f"Server: {BASE} | {health}")
        check("health: status ok", health.get("status") == "ok")
        check("health: 9566 percakapan", health.get("datastore_conversations") == 9566)

        log("\n=== 1. Query normal dan kasus tepi (struktur hasil) ===")
        queries = [
            "I want a psychological thriller with a plot twist",
            "something like Inception",
            "a feel-good romantic comedy for a rainy evening",
            "hi",
            "asdfgh qwerty",
            "film horor yang menakutkan",
            "I want something fun \U0001F600 with caf\u00e9 scenes",
            "???",
            "line one\n\n  line two\tend",
            "<script>alert(1)</script>",
            "'; DROP TABLE movies; --",
        ]
        for q in queries:
            r, ms = post(client, {"query": q})
            check(f"{q!r}: HTTP 200", r.status_code == 200, str(r.status_code))
            if r.status_code != 200:
                continue
            body = r.json()
            check_ok_structure(repr(q)[:30], body, 90, 10)
            check(f"{q!r}: query dirapikan", body["query"] == " ".join(q.split()))
            top3 = "; ".join(f"{x['title']} ({x['votes']})" for x in body["recommendations"][:3])
            log(f"  {q!r}: {ms:.0f} ms | top_similarity={body['top_similarity']} | 3 teratas: {top3}")

        log("\n=== 2. Query kosong ===")
        for q in ["", "   ", "\n\t "]:
            r, _ = post(client, {"query": q})
            body = r.json()
            check(f"{q!r}: HTTP 200 dan empty_query", r.status_code == 200 and body["status"] == "empty_query")
            check(f"{q!r}: tanpa rekomendasi", body["recommendations"] == [] and body["top_similarity"] is None)
        log("  selesai")

        log("\n=== 3. Batas parameter ===")
        for k, n, show in [(1, 1, 10), (200, 50, 10), (5, 3, 0), (90, 50, 50)]:
            r, _ = post(client, {"query": "something like Inception", "k": k, "n": n, "show_neighbors": show})
            check(f"k={k},n={n},show={show}: HTTP 200", r.status_code == 200, str(r.status_code))
            if r.status_code == 200:
                check_ok_structure(f"k={k},n={n},show={show}", r.json(), k, n, show)
        log("  selesai")

        log("\n=== 4. Panjang query ===")
        r, _ = post(client, {"query": "a" * 1000})
        check("1000 karakter: HTTP 200", r.status_code == 200, str(r.status_code))
        r, _ = post(client, {"query": "a" * 1001})
        check("1001 karakter: HTTP 422", r.status_code == 422, str(r.status_code))
        log("  selesai")

        log("\n=== 5. Masukan tidak valid dan rute lain ===")
        bad = [
            {"query": "x", "k": 0},
            {"query": "x", "k": 201},
            {"query": "x", "n": 0},
            {"query": "x", "n": 51},
            {"query": "x", "show_neighbors": -1},
            {},
            {"query": None},
        ]
        for payload in bad:
            r, _ = post(client, payload)
            check(f"{payload}: HTTP 422", r.status_code == 422, str(r.status_code))
        r = client.post("/recommend", content="{bukan json", headers={"Content-Type": "application/json"})
        check("JSON rusak ditolak (400/422)", r.status_code in (400, 422), str(r.status_code))
        r = client.get("/recommend")
        check("GET /recommend: HTTP 405", r.status_code == 405, str(r.status_code))
        r = client.get("/tidak-ada")
        check("Rute tidak ada: HTTP 404", r.status_code == 404, str(r.status_code))
        log("  selesai")

        log("\n=== 6. Konsistensi dan latensi ===")
        payload = {"query": "something like Inception"}
        r1, _ = post(client, payload)
        r2, _ = post(client, payload)
        check("Query sama memberi hasil identik", r1.json() == r2.json())
        times = [post(client, payload)[1] for _ in range(20)]
        log(
            f"  Latensi 20 permintaan (ms): median {statistics.median(times):.0f}, "
            f"min {min(times):.0f}, maks {max(times):.0f}"
        )
        log("  Diukur di komputer ini (CPU saja), termasuk HTTP lokal dan pengkodean query; bukan klaim umum.")

    log(f"\n=== RINGKASAN: pemeriksaan {total}, gagal {len(failures)} ===")
    for f in failures:
        log("  -", f)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\nSelesai. Laporan:", REPORT_PATH)


main()