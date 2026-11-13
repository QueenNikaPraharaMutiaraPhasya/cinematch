# scripts/explore_items.py
# Tujuan: mengumpulkan fakta untuk memutuskan definisi item dan konteks.
# Tidak mengubah data mentah. Hasil ditulis ke docs/explore_items_output.txt

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/redial")
OUT_PATH = Path("docs/explore_items_output.txt")
TOKEN = re.compile(r"@(\d+)")


def load(name):
    with (RAW_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mentions_of(row):
    m = row.get("movieMentions")
    return m if isinstance(m, dict) else {}


def questions_of(row, key):
    q = row.get(key)
    return q if isinstance(q, dict) else {}


def suggested_ids(row, key="initiatorQuestions"):
    return {
        mid
        for mid, lab in questions_of(row, key).items()
        if isinstance(lab, dict) and lab.get("suggested") == 1
    }


def text_of(msg):
    t = msg.get("text")
    return t if isinstance(t, str) else ""


def norm_title(title):
    if not isinstance(title, str):
        return None
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return re.sub(r"\s+", " ", title).strip().lower()


def year_of(title):
    m = re.search(r"\((\d{4})\)\s*$", title or "")
    return m.group(1) if m else None


def seeker_context(row, sugg):
    """Pesan seeker sebelum pesan pertama yang menyebut film saran."""
    seeker = row.get("initiatorWorkerId")
    titles = mentions_of(row)
    parts, cut = [], False
    for msg in row.get("messages", []):
        t = text_of(msg)
        if sugg and set(TOKEN.findall(t)) & sugg:
            cut = True
            break
        if msg.get("senderWorkerId") == seeker:
            parts.append(TOKEN.sub(lambda m: str(titles.get(m.group(1)) or ""), t))
    return " ".join(parts).strip(), cut


def check_labels(rows, label):
    print(f"\n=== LABEL suggested: {label} ({len(rows)} dialog) ===")
    sama = sum(
        1 for r in rows
        if suggested_ids(r, "initiatorQuestions") == suggested_ids(r, "respondentQuestions")
    )
    print(f"suggested=1 sama antara initiatorQuestions dan respondentQuestions: {sama} dari {len(rows)}")
    kunci_beda = sum(
        1 for r in rows
        if set(questions_of(r, "initiatorQuestions")) != set(mentions_of(r))
    )
    print("Kunci initiatorQuestions berbeda dari movieMentions:", kunci_beda)
    jumlah = pd.Series([len(suggested_ids(r)) for r in rows])
    print("Jumlah item suggested=1 per dialog (initiatorQuestions):")
    print(jumlah.describe().round(2).to_string())
    print("Dialog dengan 0 item suggested:", int((jumlah == 0).sum()))


def check_context(rows, label):
    print(f"\n=== KONTEKS seeker sebelum rekomendasi pertama: {label} ===")
    kosong = tanpa_cut = 0
    panjang = []
    for r in rows:
        sugg = suggested_ids(r)
        ctx, cut = seeker_context(r, sugg)
        if not ctx:
            kosong += 1
        if sugg and not cut:
            tanpa_cut += 1
        panjang.append(len(ctx.split()))
    print("Dialog dengan konteks kosong:", kosong)
    print("Dialog punya item suggested tapi tidak ada pesan yang menyebutnya:", tanpa_cut)
    print("Panjang konteks (jumlah kata):")
    print(pd.Series(panjang).describe().round(1).to_string())


def show_examples(rows, n=3):
    print("\n=== CONTOH konteks dan item (train) ===")
    shown = 0
    for r in rows:
        sugg = suggested_ids(r)
        if not sugg:
            continue
        ctx, _ = seeker_context(r, sugg)
        titles = [mentions_of(r).get(i) for i in sorted(sugg)]
        print(f"\nconv {r['conversationId']}\n  konteks: {ctx[:300]}\n  item suggested: {titles}")
        shown += 1
        if shown >= n:
            break


def dup_groups(row):
    peta = {}
    for mid, t in mentions_of(row).items():
        k = norm_title(t)
        if k:
            peta.setdefault(k, []).append(mid)
    return {k: ids for k, ids in peta.items() if len(ids) > 1}


def check_duplicates(train, test, nb):
    print("\n=== JUDUL KEMBAR (judul sama tanpa tahun, ID berbeda, dalam satu dialog) ===")
    kategori = Counter()
    contoh = {}
    for r in train + test:
        titles = mentions_of(r)
        for k, ids in dup_groups(r).items():
            years = {year_of(titles[i]) for i in ids}
            if len(years) == 1:
                kat = "tahun sama (atau keduanya tanpa tahun)"
            elif None in years:
                kat = "salah satu tanpa tahun"
            else:
                kat = "tahun berbeda"
            kategori[kat] += 1
            lst = contoh.setdefault(kat, [])
            if len(lst) < 3:
                detail = " | ".join(
                    f"{i}={titles[i]!r} (CSV nbMentions={nb.get(i, 'tidak ada')})" for i in ids
                )
                lst.append(f"  conv {r['conversationId']}: {detail}")
    print("Jumlah kelompok per kategori:", dict(kategori))
    for kat, lst in contoh.items():
        print(f"\nContoh kategori '{kat}':")
        print("\n".join(lst))


def show_msgs_with_token(r, token, limit=2):
    seeker = r.get("initiatorWorkerId")
    n = 0
    for msg in r.get("messages", []):
        t = text_of(msg)
        if token in TOKEN.findall(t):
            who = "SEEKER" if msg.get("senderWorkerId") == seeker else "RECOMMENDER"
            print(f"    [{who}] {t[:140]}")
            n += 1
            if n >= limit:
                break


def check_anomalies(train, df, nb):
    print("\n=== ANOMALI ===")
    by_id = {r["conversationId"]: r for r in train}
    r = by_id.get(1973)
    if r:
        print("conv 1973: movieMentions =", repr(r.get("movieMentions")),
              "| jumlah pesan:", len(r["messages"]),
              "| initiatorQuestions:", repr(r.get("initiatorQuestions"))[:200])
    print("movie_id 50 di CSV:")
    print(df[df["movieId"].astype(str) == "50"].to_string())
    for cid in (1382, 4186):
        r = by_id.get(cid)
        if not r:
            continue
        print(f"\nconv {cid}: judul movie_id 50 = {mentions_of(r).get('50')!r}")
        print("  label initiator:", questions_of(r, "initiatorQuestions").get("50"))
        print("  pesan yang menyebut @50:")
        show_msgs_with_token(r, "50")
    r = by_id.get(1382)
    if r:
        print("\nconv 1382, pesan yang menyebut @115908:")
        show_msgs_with_token(r, "115908")


def check_reach(train, test):
    print("\n=== JANGKAUAN item test terhadap data train ===")
    train_sugg = set().union(*[suggested_ids(r) for r in train])
    train_mentioned = {i for r in train for i in mentions_of(r)}
    pairs = tidak_di_train = bukan_saran = 0
    punya = bisa = 0
    for r in test:
        s = suggested_ids(r)
        if not s:
            continue
        punya += 1
        if s & train_sugg:
            bisa += 1
        for i in s:
            pairs += 1
            if i not in train_mentioned:
                tidak_di_train += 1
            elif i not in train_sugg:
                bukan_saran += 1
    print("Pasangan (dialog test, item suggested):", pairs)
    print("  item tidak pernah muncul di train sama sekali:", tidak_di_train)
    print("  item muncul di train tapi tidak pernah sebagai saran:", bukan_saran)
    print(f"Dialog test dengan >=1 item suggested: {punya}")
    print(f"  di antaranya punya >=1 item yang pernah disarankan di train: {bisa}")


sys.stdout = OUT_PATH.open("w", encoding="utf-8")

train = load("train_data.jsonl")
test = load("test_data.jsonl")
df = pd.read_csv(RAW_DIR / "movies_with_mentions.csv")
nb = dict(zip(df["movieId"].astype(str), df["nbMentions"]))

check_labels(train, "TRAIN")
check_labels(test, "TEST")
check_context(train, "TRAIN")
check_context(test, "TEST")
show_examples(train)
check_duplicates(train, test, nb)
check_anomalies(train, df, nb)
check_reach(train, test)

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Selesai. Hasil ada di", OUT_PATH)