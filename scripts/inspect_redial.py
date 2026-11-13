# scripts/inspect_redial.py
# Tujuan: melihat isi data mentah ReDial. Tidak mengubah data apa pun.

import json
from pathlib import Path

RAW_DIR = Path("data/raw/redial")

files = sorted(RAW_DIR.rglob("*.jsonl"))
if not files:
    raise SystemExit(f"Tidak ada file .jsonl di {RAW_DIR}. Cek langkah unduh/ekstrak.")

print("File .jsonl yang ditemukan:")
all_rows = {}
for path in files:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    all_rows[path.name] = rows
    print(f"  {path.name}: {len(rows)} dialog")

# Detail dilihat dari file pertama
name, rows = next(iter(all_rows.items()))
first = rows[0]

print(f"\n=== Contoh dialog pertama dari {name} ===")
print("Kunci (field) tiap dialog:", list(first.keys()))
print("conversationId:", first.get("conversationId"))

mentions = first.get("movieMentions")
print("\nTipe movieMentions:", type(mentions).__name__)
if isinstance(mentions, dict):
    for movie_id, title in list(mentions.items())[:5]:
        print(f"  {movie_id} -> {title}")
elif isinstance(mentions, list):
    for item in mentions[:5]:
        print(" ", item)

messages = first.get("messages", [])
print(f"\nJumlah pesan: {len(messages)}")
if messages:
    print("Kunci tiap pesan:", list(messages[0].keys()))
    for m in messages[:3]:
        print("  -", str(m.get("text", ""))[:100])

kosong = sum(1 for r in rows if not r.get("movieMentions"))
print(f"\nDialog tanpa movieMentions di {name}: {kosong} dari {len(rows)}")