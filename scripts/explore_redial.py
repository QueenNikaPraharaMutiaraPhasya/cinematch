# scripts/explore_redial.py
# Tujuan: mengumpulkan fakta struktur data ReDial. Tidak mengubah data mentah.

import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/redial")
TOKEN = re.compile(r"@(\d+)")


def load(name):
    with (RAW_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mentions_of(row):
    m = row.get("movieMentions")
    return m if isinstance(m, dict) else {}


def norm_title(title):
    """Judul disederhanakan untuk perbandingan. Mengembalikan None jika bukan teks."""
    if not isinstance(title, str):
        return None
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)  # buang tahun di akhir
    return re.sub(r"\s+", " ", title).strip().lower()


def text_of(msg):
    t = msg.get("text")
    return t if isinstance(t, str) else ""


def summarize(rows, label):
    print(f"\n=== {label}: {len(rows)} dialog ===")

    # 1) movieMentions yang bentuknya tidak biasa
    aneh = [r for r in rows if not isinstance(r.get("movieMentions"), dict)]
    print("movieMentions yang bukan dict:", len(aneh))
    for r in aneh[:3]:
        print("  conversationId", r.get("conversationId"), "-> nilai:", repr(r.get("movieMentions")))

    # 2) judul yang bukan teks (mis. null)
    judul_kosong = [
        (r["conversationId"], mid, t)
        for r in rows
        for mid, t in mentions_of(r).items()
        if not isinstance(t, str)
    ]
    print("Entri movieMentions dengan judul bukan teks (mis. null):", len(judul_kosong))
    for cid, mid, t in judul_kosong[:5]:
        print(f"  conversationId {cid}, movie_id {mid}, judul = {t!r}")

    # 3) statistik jumlah film dan pesan
    n_film = pd.Series([len(mentions_of(r)) for r in rows])
    print("\nJumlah film per dialog (movieMentions):")
    print(n_film.describe().round(2).to_string())
    print("Dialog dengan 0 film:", int((n_film == 0).sum()))

    n_pesan = pd.Series([len(r["messages"]) for r in rows])
    print("\nJumlah pesan per dialog:")
    print(n_pesan.describe().round(2).to_string())

    # 4) token @ID yang tidak punya pasangan di movieMentions
    total, hilang = 0, []
    for r in rows:
        ids = set(mentions_of(r))
        for msg in r["messages"]:
            for tok in TOKEN.findall(text_of(msg)):
                total += 1
                if tok not in ids:
                    hilang.append((r["conversationId"], tok))
    print(f"\nToken @ID di teks pesan: {total}; tidak ada di movieMentions: {len(hilang)}")
    for cid, tok in hilang[:3]:
        print(f"  conversationId {cid}, token @{tok}")

    # 5) satu judul dengan beberapa ID dalam satu dialog
    kembar = 0
    for r in rows:
        peta = {}
        for movie_id, title in mentions_of(r).items():
            key = norm_title(title)
            if key is None:
                continue
            peta.setdefault(key, set()).add(movie_id)
        if any(len(ids) > 1 for ids in peta.values()):
            kembar += 1
    print(f"Dialog dengan >1 ID untuk judul yang sama (setelah tahun dibuang): {kembar}")


def readable(row, max_msgs=8):
    titles = mentions_of(row)
    seeker = row["initiatorWorkerId"]
    print(f"\n--- Dialog {row['conversationId']} (maks {max_msgs} pesan pertama) ---")
    for msg in row["messages"][:max_msgs]:
        who = "SEEKER     " if msg["senderWorkerId"] == seeker else "RECOMMENDER"
        text = TOKEN.sub(
            lambda m: f"<{titles.get(m.group(1), 'ID?' + m.group(1))}>", text_of(msg)
        )
        print(f"  [{who}] {text[:160]}")


train = load("train_data.jsonl")
test = load("test_data.jsonl")

summarize(train, "TRAIN")
summarize(test, "TEST")

train_ids = {i for r in train for i in mentions_of(r)}
test_ids = {i for r in test for i in mentions_of(r)}
print("\n=== Tumpang tindih film train vs test ===")
print("ID film unik di train:", len(train_ids))
print("ID film unik di test :", len(test_ids))
print("ID film test yang TIDAK pernah muncul di train:", len(test_ids - train_ids))

print("\n=== Isi field questions (dialog train pertama) ===")
for key in ("respondentQuestions", "initiatorQuestions"):
    val = train[0][key]
    if isinstance(val, dict):
        print(key, "(dict):", list(val.items())[:3])
    else:
        print(key, f"({type(val).__name__}):", val[:3] if val else val)

readable(train[0])

print("\n=== Intip movies_with_mentions.csv ===")
csv_path = RAW_DIR / "movies_with_mentions.csv"
try:
    df = pd.read_csv(csv_path)
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="latin-1")
print("Bentuk (baris, kolom):", df.shape)
print("Kolom:", list(df.columns))
print(df.head(3).to_string())