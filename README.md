# CineMatch

Prototype aplikasi web rekomendasi film berbasis percakapan. Project ini mengimplementasikan **konsep NBCRS** (*Neighborhood-Based Collaborative Filtering for Conversational Recommendation*; Xie dkk., RecSys '24), yaitu komponen neighborhood-based (NB) tanpa training.

> **Penting:** ini *prototype implementasi konsep*, **bukan replikasi penuh eksperimen paper**. Dataset olahan, pembentukan konteks dan item, protokol evaluasi, serta jumlah dialog berbeda dari paper. Komponen model-based (MB) dan kombinasi NB+M tidak diimplementasikan. Angka hasil di sini tidak boleh dibaca sebagai angka paper.

## Cara kerja

1. Pengguna menulis preferensi film dalam **bahasa Inggris**.
2. Query diubah menjadi sentence embedding (`all-MiniLM-L6-v2`, CPU).
3. Cosine similarity dihitung terhadap konteks 9.566 percakapan ReDial (train).
4. Diambil **top-k percakapan tetangga** (bawaan k = 90).
5. **Voting:** skor film = jumlah tetangga yang menyarankan film itu (bobot 1 per kemunculan). Seri diurutkan dari jumlah similarity tetangga pendukung, lalu `movie_id`.
6. Ditampilkan **Top-N film** (bawaan N = 10) beserta panel percakapan tetangga dan penjelasan berbasis suara.

Embedding hanya dipakai untuk mencari percakapan yang mirip. Peringkat film berasal dari voting atas film yang disarankan di percakapan tetangga.

## Data

- **ReDial** (Li dkk., NeurIPS 2018), lisensi **CC BY 4.0**: <https://redialdata.github.io/website/>
- Data mentah: 10.006 dialog train dan 1.342 dialog test (`data/raw/`, tidak diubah).
- Preprocessing (`scripts/preprocess_redial.py`), keputusan prototype:
  - **Item** = film dengan `suggested = 1` pada `initiatorQuestions` **dan** `respondentQuestions` (kedua peserta sepakat), berjudul valid.
  - **Konteks** = pesan *seeker* sebelum pesan pertama yang menyebut film saran, dengan token `@ID` diganti judul film.
  - Dialog tanpa item saran valid atau berkonteks kosong dibuang.
  - Hasil: **9.566** dialog train (dari 10.006) dan **1.229** dialog test (dari 1.342); 33.246 dan 4.001 pasangan (dialog, item).
- Rincian pembuangan: `docs/preprocess_report.txt`.

## Hasil evaluasi

Protokol lengkap: `docs/evaluation_protocol.md`. Ringkas: query = konteks satu dialog test; item relevan = semua item dialog itu; Recall@K dirata-rata per dialog (macro); k dipilih pada validasi (20% dari train, seed 42) lalu test dievaluasi satu kali (datastore = seluruh train).

Hasil pada **1.229 dialog test**, persen, rata-rata ± standard error, k = 90:

| Metode | Recall@1 | Recall@5 | Recall@20 | Hit@20 |
|---|---|---|---|---|
| NB zero-shot (k = 90) | 2,40 ± 0,28 | 8,64 ± 0,52 | 19,35 ± 0,77 | 42,72 ± 1,41 |
| Popularitas (baseline) | 1,35 ± 0,21 | 3,93 ± 0,35 | 9,23 ± 0,54 | 24,33 ± 1,22 |
| Plafon (batas atas rata-rata) | 36,64 | 99,10 | 100,00 | - |

Catatan membaca hasil:

- **Plafon** menunjukkan Recall tertinggi yang mungkin per dialog (rata-rata ada sekitar 3,3 film relevan per dialog), sehingga Recall@1 rendah sebagian karena struktur data. Plafon realistis (hanya film yang ada di train yang bisa ditebak): 35,74 / 90,37 / 91,00 untuk Recall@1 / 5 / 20.
- **k = 90** dipilih dari validasi (Recall@20 validasi 26,28%). Untuk k antara 60 dan 150 selisihnya kecil dibanding standard error, jadi k = 90 bukan puncak yang tegas.
- Recall pada validasi lebih tinggi daripada test (26,28 vs 19,35 untuk Recall@20). Dugaan penyebabnya belum diuji; angka test adalah hasil resmi.
- **Tidak sebanding langsung dengan paper** (definisi item dan konteks, jumlah dialog, dan protokol berbeda). Jangan menyimpulkan hasil ini lebih baik atau lebih buruk dari paper.
- Detail lengkap (semua nilai k, Hit@1/5, diagnostik): `docs/evaluation_report.txt` dan `docs/evaluation_results.csv`.
- Waktu respons API diukur di satu komputer (CPU saja): median 17 ms untuk satu query yang diulang 20 kali, termasuk HTTP lokal. Bukan klaim umum (`docs/smoke_test_report.txt`).

## Persyaratan

Diuji di **Windows**, Python 3.14.5, Node.js v24.16.0, npm 11.13.0. Versi library Python terkunci di `requirements.txt`. Tidak membutuhkan GPU. Vite membutuhkan Node.js 20.19+ atau 22.12+ (panduan resmi Vite).

## Menjalankan dari nol

Semua perintah di PowerShell, dari folder proyek (`cinematch`).

**1. Lingkungan Python**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**2. Unduh data ReDial** (CC BY 4.0)

```powershell
New-Item -ItemType Directory -Force -Path data\raw\redial
Invoke-WebRequest -Uri "https://github.com/ReDialData/website/raw/data/redial_dataset.zip" -OutFile data\raw\redial_dataset.zip
Expand-Archive -Path data\raw\redial_dataset.zip -DestinationPath data\raw\redial -Force
```

**3. Preprocessing dan embedding**

```powershell
python scripts\preprocess_redial.py
python scripts\build_embeddings.py
```

Model embedding diunduh otomatis saat pertama kali dipakai.

**4. Tes**

```powershell
python -m pytest -v
```

Jalankan dari folder proyek (bukan dari `frontend`), karena pengaturan ada di `pytest.ini`. Hasil yang diharapkan: **69 passed**.

**5. Evaluasi (opsional, butuh beberapa menit)**

```powershell
python scripts\run_evaluation.py
```

Hasil: `docs/evaluation_report.txt` dan `docs/evaluation_results.csv`.

**6. Menjalankan aplikasi** (dua jendela PowerShell)

Jendela 1, backend (folder proyek, `.venv` aktif):

```powershell
python -m uvicorn backend.app.main:app --port 8000
```

Tunggu pesan `Application startup complete.`. Cek di <http://127.0.0.1:8000/health> atau dokumentasi API di <http://127.0.0.1:8000/docs>.

Jendela 2, frontend:

```powershell
cd frontend
npm install
npm run dev
```

Buka <http://localhost:5173> (harus persis alamat ini; backend hanya mengizinkan origin `localhost` dan `127.0.0.1` pada port 5173 dan 3000). Alamat backend dapat diganti dengan variabel lingkungan `VITE_API_URL`.

**7. Smoke test end-to-end (opsional; backend harus menyala)**

```powershell
python scripts\smoke_test_api.py
```

## Penggunaan API

`POST /recommend` dengan body JSON:

| Field | Bawaan | Batas |
|---|---|---|
| `query` | (wajib) | maksimal 1.000 karakter |
| `k` | 90 | 1 sampai 200 |
| `n` | 10 | 1 sampai 50 |
| `show_neighbors` | 10 | 0 sampai 50 |

Query kosong menghasilkan HTTP 200 dengan `status: "empty_query"`. Masukan di luar batas ditolak dengan HTTP 422.

## Struktur folder

```text
cinematch/
├── backend/
│   ├── app/
│   │   ├── main.py              # API FastAPI
│   │   ├── schemas.py           # bentuk permintaan/balasan
│   │   └── services/
│   │       ├── nb.py            # retrieval + voting (inti NB)
│   │       ├── recommender.py   # pembungkus query teks
│   │       └── evaluation.py    # Recall@K, baseline, pembagian validasi
│   └── tests/                   # test_nb, test_recommender, test_evaluation, test_api
├── data/
│   ├── raw/redial/              # data mentah (tidak diubah)
│   ├── processed/               # pairs.csv, conversations.csv
│   └── embeddings/              # train_emb.npy, test_emb.npy
├── docs/                        # decisions.md, evaluation_protocol.md, laporan-laporan
├── frontend/                    # React + Vite
├── scripts/                     # preprocessing, embedding, evaluasi, demo, smoke test
├── pytest.ini
├── requirements.txt
└── README.md
```

## Keterbatasan

- **Hanya bahasa Inggris** (data dan model berbahasa Inggris). Pada uji manual, query bahasa Indonesia menghasilkan rekomendasi yang tidak masuk akal.
- Sistem hanya mencocokkan **kemiripan teks lalu menghitung suara**, sehingga tidak memahami syarat seperti negasi ("tanpa banyak kekerasan").
- **`top_similarity` bukan ukuran keyakinan.** Query "hi" mendapat 1,000 karena ada konteks train berisi sapaan saja. Belum ada ambang keyakinan di MVP.
- Pada query yang tidak bermakna, film yang sama (misalnya Black Panther atau It) sering muncul teratas. Ini pengamatan pada sebelas query, dugaan penyebabnya belum diuji.
- Evaluasi hanya memakai **1.229 dari 1.342** dialog test, satu pembagian validasi dengan satu seed, dan datastore tuning (80%) lebih kecil daripada datastore akhir.
- Judul sama dengan ID berbeda **tidak digabung**, sehingga Top-N bisa memuat dua entri untuk film yang sama dan Recall bisa sedikit terpotong.
- Hanya satu dataset (ReDial). Tidak ada login, personalisasi lintas sesi, ataupun deployment produksi.

## Dokumentasi lain

- `docs/decisions.md`: seluruh keputusan (K1 sampai K27), hasil tiap tahap, dan pengamatan.
- `docs/evaluation_protocol.md`: protokol evaluasi.
- `docs/*_report.txt` dan `docs/*_output.txt`: keluaran skrip.

## Atribusi dan referensi

- Zhouhang Xie, Junda Wu, Hyunsik Jeon, Zhankui He, Harald Steck, Rahul Jha, Dawen Liang, Nathan Kallus, Julian McAuley. 2024. *Neighborhood-Based Collaborative Filtering for Conversational Recommendation.* RecSys '24. <https://doi.org/10.1145/3640457.3688191>
- Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, Chris Pal. 2018. *Towards Deep Conversational Recommendations.* NeurIPS 31. Dataset ReDial, lisensi CC BY 4.0.
- Model embedding: `sentence-transformers/all-MiniLM-L6-v2`.