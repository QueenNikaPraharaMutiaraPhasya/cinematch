# Catatan Keputusan CineMatch

Project ini adalah prototype implementasi konsep NBCRS, bukan replikasi penuh paper.

## Lingkungan
- OS: Windows
- Python: Python 3.14.5
- Hardware: CPU saja (GPU terintegrasi, tidak dipakai)

## Keputusan
- Bahasa query: Inggris
- Embedding: all-MiniLM-L6-v2 (juga encoder default paper, tercantum di catatan kaki paper)
- Metode: NB tanpa training (bobot w_i = 1, seperti implementasi paper)
- Backend: FastAPI | Frontend: React
- Dataset: ReDial (CC BY 4.0), file mentah di data/raw/redial/: train_data.jsonl (10006 dialog), test_data.jsonl (1342 dialog), plus file movies_*
- Format item: lihat "Keputusan data v1" di bawah.
- Nilai k: belum diputuskan (paper menguji {1, 5, 30, 60, 90, 120, 150})

## Selesai
- [x] Tugas 0.1: environment dan model embedding berjalan
- [x] Tugas 0.2: struktur folder dan requirements.txt
- [x] Tugas 1.1: ReDial terunduh dan terperiksa.
- [x] Tugas 1.1: ReDial terunduh dan terperiksa
- [x] Tugas 1.2: struktur data dieksplorasi
- [x] Tugas 1.3: keputusan item dan konteks (v1)
- [x] Fase 1 (dataset dan keputusan data)
- [x] Tugas 2.1: preprocessing
- [x] Tugas 3.1: embedding konteks
- [x] Tugas 4.1: modul NB
- [x] Tugas 4.2: query teks bebas
- [x] Fase 5: tes otomatis (24 lulus)
- [x] Fase 6: evaluasi (protokol, fungsi, hasil)
- [x] Tugas 7.1: backend FastAPI
- [x] Tugas 7.1 dan 7.2: backend dan tes API
- [x] Tugas 8.1: kerangka React
- [x] Tugas 8.2: tampilan CineMatch
- [x] Fase 9: tes end-to-end (pytest, smoke test 211 pemeriksaan, uji manual UI)

## Berikutnya
-  Fase 9: tes end-to-end (9.1), lalu 9.2 sinyal keyakinan (opsional)

## Temuan data ReDial (Tugas 1.2)
- Train 10006 dialog, test 1342 dialog; movieMentions berupa dict ID -> judul.
- Anomali train: dialog 1973 (movieMentions list kosong); movie_id 50 berjudul null (dialog 1382, 4186); token @115908 di dialog 1382 tidak ada di movieMentions.
- Judul kembar (ID berbeda): 559 dialog train, 99 dialog test (belum diperiksa).
- 414 dari 2007 ID film test tidak pernah muncul di train.
- Arti label (datasheet ReDial): suggested 0 = disebut seeker, 1 = saran recommender.
- Usulan (belum final): item = suggested==1; konteks = pesan seeker sebelum rekomendasi pertama.


## Keputusan data v1 (keputusan prototype, bukan konfigurasi paper)
- K1: film dihitung sebagai saran jika suggested=1 di initiatorQuestions DAN respondentQuestions. Alasan: himpunan saran berbeda di ~24% dialog (2371/10006 train, 329/1342 test); sebab perbedaan belum diketahui. Varian pembanding: initiator saja.
- K2: item per dialog = film hasil K1, ID unik, judul valid. Dialog tanpa item dibuang. (Aturan sementara "initiator saja": 142 train, 32 test tanpa item; angka final di Fase 2.)
- K3: konteks = pesan seeker sebelum pesan pertama yang menyebut film dari K1, @ID diganti judul, spasi dirapikan. Dialog berkonteks kosong dibuang. (Sementara: 210 train, 58 test; final di Fase 2.)
- K4: dialog 1973 dan film ID 50 (judul null) dibuang lewat aturan K2. Judul tidak dikarang.
- K5: judul kembar tidak digabung di MVP (578 "tahun berbeda", 97 "salah satu tanpa tahun", 4 "tahun sama"). Kandidat perbaikan: alias berbasis bukti dalam dialog.
- K6: data olahan satu baris per (conversation_id, movie_id): split, conversation_id, context, movie_id, movie_title.
- K7: train = datastore tetangga; test hanya evaluasi; nilai k ditentukan di Fase 6.
- Batas atas teoretis (aturan sementara): 4100 dari 4518 pasangan test (90,7%) bisa tercapai; hanya batas atas.
- Keterbatasan: Top-N bisa memuat dua entri untuk film yang sama; Recall bisa terpotong oleh ID ganda.


## Hasil preprocessing (Tugas 2.1, scripts/preprocess_redial.py)
- Train: 10006 dialog mentah -> 9566 dipakai (dibuang 440: 272 tanpa item saran valid, 167 konteks kosong, 1 movieMentions tidak valid). 33246 pasangan, 5272 film unik.
- Test: 1342 -> 1229 dipakai (dibuang 113: 61 tanpa item saran valid, 52 konteks kosong). 4001 pasangan, 1482 film unik.
- Validasi: 0 pasangan duplikat, 0 konteks kosong, 0 judul kosong, 0 conversation_id yang sama di train dan test.
- Batas atas teoretis: 3636/4001 (90,9%) pasangan test punya film yang juga item di train (menggantikan angka sementara 4100/4518). Bukan hasil model.
- Judul item sama dengan judul film di konteks (ID beda): 41 pasangan train, 8 pasangan test; belum tentu duplikat (bisa remake). K5 dipertahankan, tanpa alias di MVP.
- Bug lama (conversationId bertipe str, bukan angka) sudah diperbaiki di skrip preprocessing.
- Keterbatasan: evaluasi hanya pada 1229 dari 1342 dialog test (91,6%); konteks pendek (min 1 kata) dapat melemahkan retrieval.
- K8: embedding all-MiniLM-L6-v2 (CPU), ternormalisasi (norma 1), float32, satu embedding per percakapan.


## Hasil embedding (Tugas 3.1)
- 9566 konteks train dan 1229 konteks test dienkode: train (9566, 384), test (1229, 384), float32, norma 1, tanpa NaN/Inf; kesejajaran baris terverifikasi (enkode ulang cocok).
- Model max_seq_length = 256 token; konteks terpanjang 114 kata. Belum diverifikasi apakah ada konteks yang terpotong (tindak lanjut).
- Uji kasar (1 contoh, bukan evaluasi): tetangga #1 mirip (sim 0,869), #2 dan #3 kurang relevan topiknya.
- K9 (keputusan prototype): skor item = jumlah suara (w_i = 1, seperti paper); seri diurutkan dari jumlah similarity tetangga pendukung, lalu movie_id. Paper tidak menyebut aturan seri.
- Modul NB: backend/app/services/nb.py. k=10 untuk demo; nilai final di Fase 6.


## Hasil demo NB (Tugas 4.1)
- Mekanisme terverifikasi: sanity lolos; aritmetika voting dicek manual pada 4 contoh, cocok.
- Observasi 3 contoh (bukan evaluasi): item sebenarnya tidak ada di Top-5; suara tertinggi 2/10.
- Pembanding (paper, protokol berbeda, bukan target): NBCRS Zero-Shot di Redial Recall@1/5/20 = 1,46 / 6,64 / 16,34 (%).
- Hipotesis untuk diuji di Fase 6: (1) tahun dalam judul memengaruhi kemiripan, uji konteks tanpa tahun; (2) kalimat pembuka seragam mendominasi embedding; (3) konteks dipotong sebelum rekomendasi pertama, tetapi target mencakup semua saran, sehingga item belakangan sulit diprediksi (dugaan, belum diverifikasi).
- K10: query hanya dirapikan spasinya. K11: query kosong -> status empty_query; top_similarity selalu dikembalikan; ambang similarity rendah dan fallback ke popularitas diputuskan di Fase 6 dari distribusi data.
- Batas bahasa: hanya query Inggris.


## Hasil demo query teks (Tugas 4.2)
- Panjang token: maks 163 (train) dan 147 (test), batas model 256; 0 konteks terpotong.
- Query kosong tertangani (status empty_query, tanpa error).
- top_similarity: query bermakna 0,727-0,818; "asdfgh qwerty" 0,315; "hi" 1,000 (ada konteks train berisi sapaan saja). Jadi top_similarity SAJA bukan sinyal keyakinan yang andal.
- K11 (revisi): tidak ada ambang similarity di MVP. Kandidat sinyal yang diuji di Fase 6: suara tertinggi / k, panjang konteks, top_similarity. Jumlah konteks train yang sangat pendek belum diukur.
- Rekomendasi untuk "hi" dan teks acak praktis arbitrer (suara tertinggi 2-3 dari 10); UI harus menampilkan similarity dan suara, bukan klaim keyakinan.
- Keterbatasan (sifat rancangan): NB tidak memahami syarat/negasi seperti "tanpa banyak kekerasan"; hanya mencocokkan kemiripan teks lalu menghitung suara.

## Tes otomatis (Tugas 5.1)
- python -m pytest -v: 24 passed (15,34 s), tanpa skipped/failed; termasuk 4 tes data ReDial asli. Python 3.14.5, pytest 9.2.0.

## Protokol evaluasi (Tugas 6.1)
- Rancangan lengkap: docs/evaluation_protocol.md (K12-K18), disetujui pengguna pada 2026-11-06.
- Ringkas: query = konteks dialog; relevan = semua item K1; Recall@{1,5,20} macro + Hit@K + plafon; baseline popularitas; tuning k pada validasi 80/20 (seed 42); evaluasi akhir pada 1229 dialog test dengan datastore train penuh.


## Fungsi evaluasi (Tugas 6.2)
- K12-K18 disetujui; protokol terkunci.
- K19: standard error = simpangan baku sampel (ddof=1) / akar n.
- Modul: backend/app/services/evaluation.py (memakai nb.recommend yang sama dengan aplikasi).
- Konstanta protokol: KS = (1, 5, 20), N_LIST = 20, GRID_K = (1, 5, 10, 30, 60, 90, 120, 150), VAL_FRAC = 0.2, SEED = 42.

- python -m pytest -v: 46 passed (Tugas 6.2: 24 tes lama + 22 tes evaluasi).


## Tes fungsi evaluasi (Tugas 6.2)
- python -m pytest -v: 46 passed (12,86 s): 24 tes lama + 22 tes evaluasi.
- Tugas 6.3 (evaluasi sungguhan) menyusul; k belum dipilih.


## Hasil evaluasi (Tugas 6.3; scripts/run_evaluation.py; docs/evaluation_report.txt)
- Protokol K12-K19 dijalankan tanpa perubahan. Validasi: datastore sementara 7653, query 1913 (seed 42). Test: 1229 query, datastore 9566.
- k terpilih = 90 (Recall@20 validasi 26,28%). k=60..150 selisihnya kecil dibanding standard error (~0,67).
- Test, NB zero-shot k=90 (persen, rata-rata +- SE): Recall@1/5/20 = 2,40 +- 0,28 / 8,64 +- 0,52 / 19,35 +- 0,77; Hit@1/5/20 = 7,40 / 22,86 / 42,72.
- Test, Popularitas: Recall@1/5/20 = 1,35 +- 0,21 / 3,93 +- 0,35 / 9,23 +- 0,54; Hit@1/5/20 = 4,15 / 11,55 / 24,33.
- Plafon: Recall@1/5/20 = 36,64 / 99,10 / 100,00; plafon realistis (film harus ada di train): 35,74 / 90,37 / 91,00.
- Konsistensi: 3636 dari 4001 pasangan test punya item di train (cocok dengan preprocessing).
- Ilustrasi kebocoran: validasi tanpa mengeluarkan query: Recall@20 41,99 vs bersih 26,28 (tidak valid; hanya ilustrasi).
- Observasi: validasi > test (Recall@20 26,28 vs 19,35); popularitas juga turun (10,49 -> 9,23). Dugaan penyebab (belum diuji): pekerja validasi sama dengan datastore; test dari pekerja berbeda dan dikumpulkan belakangan (datasheet ReDial).
- k kecil: daftar peringkat pendek (rata-rata 3,5 film untuk k=1; 16,4 untuk k=5; 20 untuk k>=10), sehingga Recall@20 kecil (tanpa padding, K15).
- Pembanding paper (protokol berbeda, BUKAN target; tidak boleh disimpulkan lebih baik/buruk): Redial NBCRS Zero-Shot 1,46/6,64/16,34; PopRec 0,0/1,35/6,01.
- K20: k bawaan aplikasi = 90 (dipilih dari validasi, bukan dari test).
- Ditunda (opsional, eksploratif, hanya di validasi): konteks tanpa tahun; pengecekan tumpang tindih pekerja.


## Backend FastAPI (Tugas 7.1)
- Berkas: backend/app/main.py, backend/app/schemas.py. Jalankan dari folder cinematch: python -m uvicorn backend.app.main:app --port 8000
- Endpoint: GET /health; POST /recommend (query, k=90, n=10, show_neighbors=10).
- Batas masukan: k 1-200, n 1-50, show_neighbors 0-50, query maks 1000 karakter. Diamati: k=0 ditolak validasi (detail greater_than_equal, ge=1).
- Query kosong -> HTTP 200 dengan status "empty_query" (K11). Model dan datastore dimuat sekali saat server menyala (lifespan).
- Penjelasan rekomendasi hanya berasal dari data: jumlah suara dari k percakapan dan total similarity. Maks 5 pendukung per rekomendasi.
- CORS: localhost dan 127.0.0.1 pada port 5173 dan 3000.


## Tes API (Tugas 7.2)
- python -m pytest -v: 69 passed (diharapkan 69: 46 tes lama + 23 tes API). Tes API memakai model palsu dan data ilustrasi; 3 tes memakai server sungguhan.

## Frontend (Fase 8)
- Node.js v24.16.0 (Vite membutuhkan 20.19+ atau 22.12+ menurut panduan resmi).
- K21: React + Vite, template JavaScript (react), CSS biasa, tanpa pustaka tambahan sampai dibutuhkan. Folder: frontend/.
- Alamat dev server: http://localhost:5173 (CORS backend mengizinkan port 5173 dan 3000).


## Kerangka frontend (Tugas 8.1)
- create-vite v9.3.4, template react (JavaScript), linter Oxlint; npm 11.13.0; 24 paket terpasang, 0 vulnerabilities. Halaman bawaan terbuka di localhost:5173.
- Berkas src bawaan: assets, App.css, App.jsx, index.css, main.jsx.

## Tampilan CineMatch (Tugas 8.2)
- K22: UI memanggil http://127.0.0.1:8000 (dapat diganti lewat VITE_API_URL); bawaan k=90, N=10, 10 tetangga ditampilkan.
- K23: top_similarity ditampilkan sebagai kemiripan teks tetangga terdekat, bukan keyakinan (contoh "hi" = 1,000 di Tugas 4.2).
- K24: halaman memuat keterbatasan (hanya Inggris, tidak memahami negasi, prototype bukan replikasi paper) dan atribusi ReDial (CC BY 4.0).
- Status UI: idle, loading, empty_query (pesan dari backend), error (koneksi gagal / ditolak server), hasil kosong.


## Hasil tampilan CineMatch (Tugas 8.2)
- Enam skenario UI lolos: contoh Top-10 dengan panel tetangga; query kosong (pesan dari backend); backend mati (kotak merah); k=5/N=3 (Top-3, 5 tetangga); percakapan pendukung; N=1.
- npm run build: sukses (Vite v8.3.8, 17 modul, JS 225,77 kB / gzip 70,94 kB, 303 ms).
- Observasi manual (bukan evaluasi): query "I want a psychological thriller with a plot twist" -> kemiripan tetangga terdekat 0,859; Get Out (2017) 13/90 suara, Seven (1995) 11/90. Dengan k=5: Shutter Island 3/5, Seven 2/5.
- K25: latensi diukur dan dilaporkan apa adanya (komputer ini, CPU saja); tidak dibandingkan dengan angka paper.
- K26: aplikasi membutuhkan dua jendela: backend (uvicorn, port 8000) dan frontend (npm run dev, port 5173).


## Tes end-to-end (Tugas 9.1)
- Smoke test terhadap server sungguhan (scripts/smoke_test_api.py, docs/smoke_test_report.txt): 211 pemeriksaan, 0 gagal. Mencakup struktur hasil (11 query termasuk kasus tepi), query kosong, batas parameter, batas panjang 1000/1001, masukan tidak valid (422), JSON rusak, GET /recommend (405), rute tidak ada (404), hasil identik untuk query sama.
- Latensi (komputer ini, CPU saja; satu query diulang 20 kali): median 17 ms, min 15, maks 20; 11 query pertama 17-32 ms. Termasuk HTTP lokal dan pengkodean query; bukan klaim umum (K25).
- Pengamatan (bukan evaluasi): "Black Panther (2018)" teratas untuk query tak bermakna/asing ("hi" 11/90, "asdfgh qwerty" 14/90, bahasa Indonesia 6/90, teks dengan baris baru 10/90); "It (2017)" teratas untuk "???" (18/90) dan teks <script> (29/90). Dugaan (belum diuji): tetangga untuk query tak bermakna adalah konteks pendek/generik yang saran filmnya condong ke film populer.
- top_similarity: bermakna 0,73-0,86; tak bermakna 0,23-0,32; "hi" 1,0; Indonesia 0,544; Inggris samar + emoji 0,571; teks SQL 0,467. Satu ambang top_similarity tidak memisahkan query bermakna dan tidak bermakna (hanya 11 query; bukan uji sistematis). K11 tetap terbuka.
- Manual: klik cepat dua contoh (berjalan); query Indonesia (hasil tidak masuk akal, keterbatasan bahasa terkonfirmasi); tampilan mobile satu kolom tanpa scroll horizontal; batas 1000 karakter; Enter membuat baris baru (tidak mengirim); navigasi Tab berfungsi.
- pytest: 69 passed. 

## Dokumentasi (Fase 10)
- K27: README hanya memuat angka dari laporan tertulis (evaluation_report.txt, preprocess_report.txt, smoke_test_report.txt).
- Tugas 10.1: README.md dibuat. Berikutnya: dokumen metode dan arsitektur (10.2), panduan demo (10.3).
- Opsional, eksploratif: Tugas 9.2 (sinyal keyakinan pada data validasi).


## Dokumen arsitektur (Tugas 10.2)
- docs/architecture.md dibuat: membedakan fakta paper, keputusan prototype (K1-K27), dan hasil terukur (dengan berkas sumber).
- Berikutnya: Tugas 10.3 (skenario demo dan bahan presentasi). Opsional: Tugas 9.2.


## Verifikasi dokumentasi (Tugas 10.2)
- pytest: 69 passed (16,68 s). Daftar berkas scripts, backend/tests, dan docs sesuai README.
- Lokasi README.md: (isi: dipindahkan dari docs/ ke folder proyek / sudah di folder proyek).

## Panduan demo (Tugas 10.3)
- docs/demo_guide.md dibuat: skenario demo, query contoh (kekuatan dan keterbatasan), Q&A jujur, kerangka slide, daftar periksa.
- Perilaku backend tanpa internet belum diverifikasi; diuji saat gladi bersih (isi hasilnya).