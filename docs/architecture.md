# CineMatch: Metode dan Arsitektur

Dokumen ini menjelaskan cara kerja CineMatch dan memisahkan tiga hal: **fakta dari paper**, **keputusan prototype**, dan **hasil terukur**.

> CineMatch adalah *prototype implementasi konsep NBCRS*, bukan replikasi penuh eksperimen paper. Komponen model-based (MB) dan kombinasi NB+M tidak diimplementasikan.

## 1. Fakta dari paper

Sumber: Xie dkk., *Neighborhood-Based Collaborative Filtering for Conversational Recommendation*, RecSys '24.

- NBCRS terdiri dari komponen neighborhood-based (NB) dan komponen model-based (MB) yang opsional. MB dipakai untuk menyempurnakan retriever, dan hasil NB dan MB dapat digabung.
- NB mengambil sekumpulan konteks training yang mirip dengan konteks test, lalu merekomendasikan item yang sering muncul di neighborhood itu.
- Skor NB (Persamaan 1): `score_neighbor(v | q) = Σ w_i · 1(ṽ_i = v)`, dijumlahkan atas pasangan (konteks, item) di neighborhood.
- Implementasi paper memakai **cosine similarity** untuk retrieval dan bobot **w_i = 1**. Encoder kalimat bawaan adalah `all-MiniLM-L6-v2` (catatan kaki paper).
- Varian **Zero-Shot** memakai retriever pre-trained tanpa fine-tuning; inilah yang paling dekat dengan CineMatch.
- Jumlah tetangga k dituning pada {1, 5, 30, 60, 90, 120, 150} memakai validation set (20% dari data training). Metrik: Recall@{1, 5, 20}, dilaporkan dengan standard error. Pembanding antara lain popularitas (PopRec).
- Dataset paper: Inspired, Redial, dan Reddit-Movie. Untuk Redial, Tabel 1 mencatat 6.476 film, 8.929 sampel train, dan 4.288 sampel test.
- Angka referensi paper pada Redial (Tabel 2, persen, Recall@1 / @5 / @20): NBCRS Zero-Shot 1,46 / 6,64 / 16,34; Popularitas 0,0 / 1,35 / 6,01. Angka ini hanya referensi orde besar, **bukan target** (lihat bagian 6).

**Tidak dijelaskan di teks paper yang dilampirkan** (sehingga menjadi keputusan prototype): rumus Recall@K yang persis, cara sampel dibentuk dari dialog, cara PopRec dihitung, aturan seri pada voting, dan cara membentuk konteks.

## 2. Keputusan prototype

| # | Keputusan |
|---|---|
| K1 | Item = film dengan `suggested = 1` pada `initiatorQuestions` **dan** `respondentQuestions` (label dua peserta sepakat). Alasan: himpunan saran berbeda di sekitar 24% dialog. |
| K2 | Item per dialog: ID unik dengan judul valid. Dialog tanpa item dibuang. |
| K3 | Konteks = pesan *seeker* sebelum pesan pertama yang menyebut film saran, token `@ID` diganti judul. Dialog berkonteks kosong dibuang. |
| K4 | Dialog 1973 (tanpa film) dan film ID 50 (judul `null`) dibuang lewat aturan K2. Judul tidak dikarang. |
| K5 | Judul kembar dengan ID berbeda **tidak digabung** (bisa remake, bukan duplikat). |
| K6 | Format data olahan: satu baris per (`conversation_id`, `movie_id`). |
| K7 | Train = datastore tetangga; test hanya untuk evaluasi. |
| K8 | Embedding `all-MiniLM-L6-v2`, CPU, dinormalisasi, float32, satu embedding per percakapan. |
| K9 | Seri pada voting: jumlah similarity tetangga pendukung (lebih besar menang), lalu `movie_id`. |
| K10 | Query hanya dirapikan spasinya. |
| K11 | Query kosong: HTTP 200 dengan `status: "empty_query"`. `top_similarity` selalu dikembalikan. Tidak ada ambang similarity di MVP. |
| K12 | Evaluasi: satu dialog = satu query; item relevan = semua item dialog itu (cocok lewat `movie_id`). |
| K13 | Recall@K per dialog = |Top-K ∩ relevan| / |relevan|, dirata-rata (macro), K dalam {1, 5, 20}, dengan standard error. |
| K14 | Metrik pendamping: Hit@K dan plafon (batas atas Recall per dialog). |
| K15 | Daftar peringkat: film unik hasil voting, N = 20, tanpa padding. |
| K16 | Baseline popularitas: film diurutkan menurut jumlah dialog datastore yang memuatnya (seri: `movie_id`). |
| K17 | Tuning k: dialog train dibagi 80/20 (seed 42); grid k = {1, 5, 10, 30, 60, 90, 120, 150}; dipilih menurut Recall@20 validasi. |
| K18 | Evaluasi akhir: 1.229 dialog test, datastore = seluruh train, k dari K17, dijalankan sekali. |
| K19 | Standard error = simpangan baku sampel (ddof = 1) / akar n. |
| K20 | k bawaan aplikasi = 90 (dipilih dari validasi, bukan dari test). |
| K21 | Frontend React + Vite, JavaScript, CSS biasa. |
| K22 | UI memanggil `http://127.0.0.1:8000` (dapat diganti lewat `VITE_API_URL`). |
| K23 | `top_similarity` ditampilkan sebagai kemiripan teks, bukan keyakinan. |
| K24 | Halaman memuat keterbatasan dan atribusi ReDial. |
| K25 | Latensi dilaporkan apa adanya (satu komputer, CPU saja). |
| K26 | Aplikasi membutuhkan dua proses: backend (port 8000) dan frontend (port 5173). |
| K27 | README hanya memuat angka dari laporan tertulis. |

## 3. Alur data dan arsitektur

```text
OFFLINE (dijalankan sekali)

data/raw/redial/*.jsonl                       data mentah, tidak diubah
   |  scripts/preprocess_redial.py            K1-K6
   v
data/processed/pairs.csv                      split, conversation_id, context, movie_id, movie_title
   |  scripts/build_embeddings.py             K8
   v
data/processed/conversations.csv              pemetaan percakapan -> baris embedding
data/embeddings/train_emb.npy, test_emb.npy   satu vektor 384 angka per percakapan


ONLINE (setiap permintaan)

Browser (React, :5173)
   |  POST /recommend  {query, k, n, show_neighbors}
   v
FastAPI (backend/app/main.py, :8000)          validasi masukan (schemas.py)
   v
Recommender.recommend_text                    rapikan spasi, encode query menjadi vektor
   v
nb.find_neighbors                             embedding train @ query -> top-k (cosine)
   v
nb.vote                                       suara per film (w = 1), seri sesuai K9
   v
JSON: neighbors, recommendations (votes, explanation, supporters)
   v
Browser: kartu Top-N + panel percakapan tetangga
```

Model embedding dan datastore dimuat sekali saat server menyala (`lifespan`).

## 4. Algoritma NB (yang diimplementasikan)

```text
masukan : query teks, k, N
1. q  <- encode(rapikan_spasi(query))                 vektor ternormalisasi
2. s  <- E_train @ q                                  cosine similarity ke semua percakapan train
3. NB <- k indeks dengan s terbesar (seri: baris lebih awal)
4. untuk tiap film v yang muncul pada NB:
        votes(v) <- jumlah percakapan di NB yang memuat v     (w_i = 1)
        sim_sum(v) <- jumlah s_i dari percakapan tersebut
5. urutkan film menurut (votes turun, sim_sum turun, movie_id naik)
6. kembalikan N teratas beserta tetangga dan pendukungnya
```

Interpretasi Persamaan 1 paper: indikator `ṽ_i = v` kami tafsirkan sebagai "film v termuat di item percakapan tetangga ke-i". Tiap film dihitung sekali per percakapan karena item unik per dialog (K2).

## 5. Modul dan tanggung jawab

| Berkas | Tanggung jawab |
|---|---|
| `backend/app/services/nb.py` | Memuat data, retrieval top-k, voting, `recommend` (inti NB). |
| `backend/app/services/recommender.py` | Memuat model; query teks -> embedding -> `recommend`; penanganan query kosong. |
| `backend/app/services/evaluation.py` | Recall@K, Hit@K, plafon, standard error, popularitas, pembagian validasi, peringkat NB untuk evaluasi. |
| `backend/app/main.py`, `schemas.py` | API `GET /health`, `POST /recommend`, validasi masukan, CORS. |
| `frontend/src/App.jsx` | Form query, status memuat/kosong/error, kartu Top-N, panel tetangga. |
| `scripts/` | Preprocessing, embedding, evaluasi, demo, smoke test. |

Evaluasi memanggil `nb.recommend` yang sama dengan aplikasi, sehingga yang dievaluasi adalah kode yang didemokan.

## 6. Hasil terukur

Semua angka berasal dari laporan tertulis.

**Preprocessing** (`docs/preprocess_report.txt`)

| | Dialog mentah | Dibuang | Dipakai | Pasangan (dialog, item) | Film unik sebagai item |
|---|---|---|---|---|---|
| Train | 10.006 | 440 | 9.566 | 33.246 | 5.272 |
| Test | 1.342 | 113 | 1.229 | 4.001 | 1.482 |

Alasan pembuangan: tanpa item saran valid (272 train, 61 test), konteks kosong (167 train, 52 test), `movieMentions` tidak valid (1 train). Pada pasangan test, 3.636 dari 4.001 (90,9%) memuat film yang juga menjadi item di train (batas atas teoretis).

**Pemilihan k pada validasi** (`docs/evaluation_report.txt`; datastore sementara 7.653 dialog, 1.913 query validasi)

| k | 1 | 5 | 10 | 30 | 60 | 90 | 120 | 150 |
|---|---|---|---|---|---|---|---|---|
| Recall@20 validasi (%) | 5,04 | 15,93 | 18,70 | 23,78 | 25,45 | 26,28 | 26,20 | 25,93 |

Baseline popularitas pada validasi: Recall@20 = 10,49%. Untuk k antara 60 dan 150 selisihnya kecil dibanding standard error sekitar 0,67.

**Evaluasi akhir pada test** (k = 90, 1.229 query; persen, rata-rata ± standard error)

| Metode | Recall@1 | Recall@5 | Recall@20 | Hit@1 | Hit@5 | Hit@20 |
|---|---|---|---|---|---|---|
| NB zero-shot | 2,40 ± 0,28 | 8,64 ± 0,52 | 19,35 ± 0,77 | 7,40 ± 0,75 | 22,86 ± 1,20 | 42,72 ± 1,41 |
| Popularitas | 1,35 ± 0,21 | 3,93 ± 0,35 | 9,23 ± 0,54 | 4,15 ± 0,57 | 11,55 ± 0,91 | 24,33 ± 1,22 |
| Plafon | 36,64 | 99,10 | 100,00 | - | - | - |

Plafon realistis (film harus ada di train): 35,74 / 90,37 / 91,00 untuk Recall@1 / 5 / 20.

**Smoke test API** (`docs/smoke_test_report.txt`): 211 pemeriksaan, 0 gagal. Waktu respons satu query yang diulang 20 kali: median 17 ms (min 15, maks 20), diukur di satu komputer (CPU saja), termasuk HTTP lokal.

**Cara membaca dan membandingkan dengan paper**

- NB mengungguli popularitas pada semua K. Selisih Recall@5 dan Recall@20 jauh lebih besar daripada standard error.
- Recall@1 rendah sebagian karena struktur data: rata-rata sekitar 3,3 film relevan per dialog, sehingga sistem sempurna pun hanya mencapai 36,64% pada K = 1.
- **Hasil ini tidak sebanding langsung dengan Tabel 2 paper.** Definisi item dan konteks, jumlah dialog (1.229 dari 1.342), dan protokol berbeda. Jangan menyimpulkan CineMatch lebih baik atau lebih buruk dari paper.

## 7. Perbedaan dengan paper

- Hanya komponen NB tanpa training. MB dan N+M tidak diimplementasikan.
- Dataset olahan, pembentukan konteks dan item, serta protokol evaluasi adalah keputusan prototype (bagian 1 dan 2).
- Hanya satu dataset (ReDial). Paper juga memakai Inspired dan Reddit-Movie.
- Pemilihan k memakai satu pembagian validasi 80/20 dengan seed 42, dengan grid yang sedikit berbeda dari paper (tambahan k = 10).
- Evaluasi hanya pada 1.229 dari 1.342 dialog test.

## 8. Keterbatasan dan pekerjaan lanjutan

**Keterbatasan terkonfirmasi**

- Hanya bahasa Inggris. Pada uji manual, query bahasa Indonesia menghasilkan rekomendasi yang tidak masuk akal.
- Sistem tidak memahami syarat seperti negasi ("tanpa banyak kekerasan"). Ia hanya mencocokkan kemiripan teks lalu menghitung suara.
- `top_similarity` bukan ukuran keyakinan: query "hi" mendapat 1,000 karena ada konteks train berisi sapaan saja. Belum ada ambang keyakinan di MVP.
- Judul kembar tidak digabung, sehingga Top-N bisa memuat dua entri untuk film yang sama.

**Hipotesis yang belum diuji** (dugaan, bukan hasil)

- Tahun di dalam judul memengaruhi kemiripan teks (konteks tanpa tahun belum diuji).
- Kalimat pembuka yang seragam ("Hi I am looking for a movie like ...") mendominasi embedding.
- Konteks dipotong sebelum rekomendasi pertama, sedangkan item relevan mencakup semua saran, sehingga item yang muncul belakangan sulit diprediksi.
- Recall validasi lebih tinggi daripada test (26,28 vs 19,35 untuk Recall@20) karena pekerja validasi sama dengan pekerja datastore, sedangkan data test dikumpulkan belakangan oleh pekerja berbeda.
- Pada query tak bermakna, film populer (misalnya Black Panther atau It) sering muncul teratas karena tetangganya konteks pendek dan generik. Ini pengamatan pada sebelas query.

**Pekerjaan lanjutan (tidak termasuk MVP)**

- Sinyal keyakinan yang diuji pada data validasi.
- Alias judul berbasis bukti dalam dialog.
- Evaluasi dengan beberapa seed, model embedding multibahasa, komponen MB dan N+M.

## 9. Reproduksibilitas

- Lingkungan yang dipakai: Windows, Python 3.14.5, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0 (CPU), sentence-transformers 6.2.0. Versi lengkap tercatat di `requirements.txt`.
- Frontend: Node.js v24.16.0, npm 11.13.0, create-vite v9.3.4, Vite v8.3.8.
- Seed pembagian validasi: 42. Model embedding: `all-MiniLM-L6-v2`.
- Langkah menjalankan: lihat `README.md`.