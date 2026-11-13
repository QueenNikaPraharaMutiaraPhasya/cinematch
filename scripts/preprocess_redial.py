# scripts/preprocess_redial.py
# Tujuan: mengubah data mentah ReDial menjadi tabel pasangan (conversation, item)
# sesuai keputusan K1-K7. Data mentah TIDAK diubah.
# Hasil: data/processed/pairs.csv dan docs/preprocess_report.txt

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/redial")
OUT_DIR = Path("data/processed")
REPORT_PATH = Path("docs/preprocess_report.txt")
TOKEN = re.compile(r"@(\d+)")
COLUMNS = ["split", "conversation_id", "context", "movie_id", "movie_title"]

lines = []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


def load(name):
    with (RAW_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(s):
    return " ".join(s.split())


def valid_title(t):
    return isinstance(t, str) and t.strip() != ""


def norm_title(title):
    """Judul tanpa tahun di akhir, huruf kecil. Hanya untuk perbandingan."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", clean_text(title)).lower()


def mentions_of(row):
    m = row.get("movieMentions")
    return m if isinstance(m, dict) else {}


def questions_of(row, key):
    q = row.get(key)
    return q if isinstance(q, dict) else {}


def suggested_ids(row, key):
    return {
        mid
        for mid, lab in questions_of(row, key).items()
        if isinstance(lab, dict) and lab.get("suggested") == 1
    }


def agreed_suggestions(row):
    """K1: saran jika suggested=1 di initiatorQuestions DAN respondentQuestions."""
    return suggested_ids(row, "initiatorQuestions") & suggested_ids(row, "respondentQuestions")


def text_of(msg):
    t = msg.get("text")
    return t if isinstance(t, str) else ""


def build_context(row, sugg, titles):
    """K3: pesan seeker sebelum pesan pertama yang menyebut film saran."""
    seeker = row.get("initiatorWorkerId")
    parts, inserted = [], []
    unresolved = 0

    def repl(m):
        nonlocal unresolved
        title = titles.get(m.group(1))
        if valid_title(title):
            inserted.append(title)
            return clean_text(title)
        unresolved += 1
        return ""

    for msg in row.get("messages", []):
        t = text_of(msg)
        if set(TOKEN.findall(t)) & sugg:
            break
        if msg.get("senderWorkerId") != seeker:
            continue
        parts.append(TOKEN.sub(repl, t))
    return clean_text(" ".join(parts)), inserted, unresolved


def process_split(rows, split):
    out, drop, status, stats, leaks = [], Counter(), {}, Counter(), []
    for r in rows:
        cid = str(r.get("conversationId"))
        titles = mentions_of(r)
        reason = None
        items, ctx, inserted = {}, "", []

        if not titles:
            reason = "movieMentions bukan dict atau kosong"
        else:
            sugg = agreed_suggestions(r)
            for mid in sorted(sugg):
                if mid not in titles:
                    stats["saran tanpa entri di movieMentions"] += 1
                elif not valid_title(titles[mid]):
                    stats["saran dengan judul tidak valid (null/kosong)"] += 1
                else:
                    items[mid] = clean_text(titles[mid])
            if not items:
                reason = "tanpa item saran valid"
            else:
                ctx, inserted, unresolved = build_context(r, sugg, titles)
                stats["token @ID tak terselesaikan di konteks"] += unresolved
                if not ctx:
                    reason = "konteks kosong"

        if reason:
            drop[reason] += 1
            status[cid] = "dibuang: " + reason
            continue

        inserted_norm = {norm_title(t) for t in inserted}
        for mid, title in items.items():
            if norm_title(title) in inserted_norm:
                stats["pasangan: judul item = judul film di konteks (ID beda)"] += 1
                if len(leaks) < 5:
                    leaks.append(f"conv {cid}: item {title!r}")
            out.append(
                {
                    "split": split,
                    "conversation_id": cid,
                    "context": ctx,
                    "movie_id": mid,
                    "movie_title": title,
                }
            )
        status[cid] = f"dipakai: {len(items)} item"

    return pd.DataFrame(out, columns=COLUMNS), drop, status, stats, leaks


def report_split(split, n_raw, df, drop, stats, leaks):
    n_dialog = df["conversation_id"].nunique()
    log(f"\n=== {split.upper()} ===")
    log("Dialog mentah:", n_raw)
    log("Dialog dibuang:", sum(drop.values()))
    for reason, n in drop.most_common():
        log(f"  - {reason}: {n}")
    log("Dialog dipakai:", n_dialog)
    log("Cek: mentah = dibuang + dipakai ->", n_raw == sum(drop.values()) + n_dialog)
    log("Baris pasangan (conversation, item):", len(df))
    log("Film unik sebagai item:", df["movie_id"].nunique())
    for name, n in stats.items():
        log(f"Catatan - {name}: {n}")
    for x in leaks:
        log("  contoh:", x)
    if len(df):
        log("Item per dialog:")
        log(df.groupby("conversation_id").size().describe().round(2).to_string())
        log("Panjang konteks (kata):")
        ctx_len = df.drop_duplicates("conversation_id")["context"].str.split().str.len()
        log(ctx_len.describe().round(1).to_string())


train_raw = load("train_data.jsonl")
test_raw = load("test_data.jsonl")
log("Tipe conversationId di data mentah:", type(train_raw[0].get("conversationId")).__name__)

train_df, train_drop, train_status, train_stats, train_leaks = process_split(train_raw, "train")
test_df, test_drop, test_status, test_stats, test_leaks = process_split(test_raw, "test")

report_split("train", len(train_raw), train_df, train_drop, train_stats, train_leaks)
report_split("test", len(test_raw), test_df, test_drop, test_stats, test_leaks)

pairs = pd.concat([train_df, test_df], ignore_index=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
pairs.to_csv(OUT_DIR / "pairs.csv", index=False, encoding="utf-8")

log("\n=== VALIDASI AKHIR ===")
log("Pasangan duplikat (split, conversation_id, movie_id):",
    int(pairs.duplicated(subset=["split", "conversation_id", "movie_id"]).sum()))
log("Konteks kosong:", int((pairs["context"].str.strip() == "").sum()))
log("Judul film kosong:", int((pairs["movie_title"].str.strip() == "").sum()))
log("conversation_id yang muncul di train DAN test:",
    len(set(train_df["conversation_id"]) & set(test_df["conversation_id"])))

train_items = set(train_df["movie_id"])
reach = int(test_df["movie_id"].isin(train_items).sum())
log(f"Pasangan test dengan film yang juga item di train: {reach} dari {len(test_df)}"
    " (batas atas teoretis, bukan hasil)")

log("\n=== PEMERIKSAAN ANOMALI (train) ===")
for cid in ("1973", "1382", "4186"):
    log(f"conv {cid}:", train_status.get(cid, "tidak ditemukan"))

log("\n=== CONTOH 3 BARIS PERTAMA ===")
show = pairs.head(3).copy()
show["context"] = show["context"].str.slice(0, 80)
log(show.to_string(index=False))

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\nSelesai. Tabel di", OUT_DIR / "pairs.csv", "| laporan di", REPORT_PATH)