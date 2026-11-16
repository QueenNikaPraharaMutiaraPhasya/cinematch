# Protokol Evaluasi CineMatch (prototype implementasi konsep NBCRS)

Status: rancangan v1, disepakati SEBELUM kode evaluasi ditulis.
Ini bukan replikasi eksperimen paper.

## Tambahan evaluasi ranking kelompok

NDCG@K dan MRR@K adalah metrik tambahan kelompok kami, **bukan metrik asli
paper NBCRS**. Tambahan ini disepakati setelah hasil test eksperimen lama
diketahui. Hasilnya merupakan evaluasi tambahan, bukan eksperimen test baru
yang belum pernah diamati. K12-K19 dan pemilihan k berdasarkan Recall@20
validasi tetap berlaku; NDCG/MRR maupun data test tidak dipakai untuk tuning.

- Ground truth R tetap himpunan movie_id K12, dapat berisi beberapa film.
  Item relevan yang tidak ada di datastore tetap termasuk R; tidak ada alias judul.
- Binary relevance: gain 1 untuk ID relevan pada kemunculan pertamanya, 0 selain itu.
  ID duplikat tidak mendapat gain tambahan dan tidak dihapus dari ranking:
  posisi berikutnya tetap posisi asli (posisi dimulai dari 1).
- DCG@K = jumlah gain_i / log2(i+1) pada posisi 1 sampai min(K, panjang ranking).
- IDCG@K = jumlah 1 / log2(i+1) pada posisi 1 sampai min(K, |R|).
  Penyebut tidak diperkecil mengikuti panjang ranking atau banyaknya target yang ditemukan.
- NDCG@K = DCG@K / IDCG@K. Contoh R={A,B}, ranking=[A,X,B], K=3:
  NDCG = 1.5 / (1 + 1/log2(3)) = sekitar 0.91972.
- RR@K = 1/r untuk posisi film relevan pertama r di Top-K, atau 0 jika tidak ada.
  MRR@K adalah rata-rata RR atas seluruh query, termasuk query dengan nilai nol.
- K tetap {1,5,20}. Kedua metrik dirata-rata per dialog (macro), dengan standard
  error sampel ddof=1 dibagi akar n melalui mean_se(), sama seperti Recall.
  Satu query memiliki SE=0. Nilai fungsi 0-1; laporan/CSV menyajikan persen.
- Ranking kosong bernilai 0 untuk R tidak kosong. Ranking pendek tidak ditambah
  item. R kosong atau K<1 menghasilkan ValueError, konsisten dengan Recall.
- Skrip yang sama mengevaluasi ranking NB dan popularitas yang sudah dihitung.
  Recall, Hit, ceiling, split 80/20, seed 42, grid k, dan panjang daftar 20 tetap.

Jalankan dari root: `.\.venv\Scripts\python.exe scripts\run_evaluation.py`.
Hasil tambahan ditulis ke `docs/evaluation_report_ranking.txt` dan
`docs/evaluation_results_ranking.csv`. Laporan lama `evaluation_report.txt`
dan `evaluation_results.csv` tidak ditimpa. CSV tetap memakai kolom lama,
dengan tambahan nilai metric `ndcg` dan `mrr`. Tabel ranking dipisahkan dari
tabel metrik lama; baris plafon tetap hanya untuk Recall.

### Verifikasi terhadap arsip eksperimen

Pada eksekusi tambahan dengan embedding lokal yang tersedia, semua nilai metrik
test lama cocok dengan CSV arsip (presisi empat desimal persen). Empat baris
validasi berbeda sedikit: Recall@20/Hit@20 untuk k=60 dan Recall@1/Hit@1 untuk
k=90. Nilai mean lama -> baru berturut-turut adalah 25.4501 -> 25.4631,
54.3649 -> 54.4171, 3.3590 -> 3.3655, dan 11.1866 -> 11.2389 persen.
Kode evaluasi lama dari Git, dijalankan pada ranking dari embedding lokal saat ini,
menghasilkan mean dan standard error yang sama dengan baris baru tersebut.
Dengan demikian, perbedaan ini bukan akibat rumus metrik tambahan; penyebab
perbedaan terhadap artefak eksperimen historis belum ditentukan. k terpilih tetap 90.

## Fakta dari paper (teks yang dilampirkan)
- Metrik: Recall@k, k dalam {1, 5, 20}; hasil dilaporkan dengan standard error.
- 20% data training disisihkan untuk validasi; split test mengikuti penelitian lain ([11] di paper).
- Jumlah tetangga k dituning pada {1, 5, 30, 60, 90, 120, 150} memakai validation set.
- Pembanding termasuk PopRec (popularitas).
- Rumus Recall@k, cara membentuk sampel, dan cara PopRec dihitung TIDAK dijelaskan di teks yang dilampirkan.

## Keputusan prototype (bukan konfigurasi paper)
- K12 Unit evaluasi: satu dialog = satu query. Query = konteks (K3). Item relevan R = semua item dialog itu (K1), dicocokkan lewat movie_id persis (tanpa alias, K5).
- K13 Metrik utama: Recall@K = |Top-K irisan R| / |R| per dialog, dirata-rata atas semua dialog (macro). K dalam {1, 5, 20}. Dilaporkan dengan standard error (simpangan baku / akar n).
- K14 Metrik pendamping: Hit@K (1 jika ada item relevan di Top-K) dan plafon (rata-rata min(K, |R|) / |R| per dialog; belum memperhitungkan film yang tidak ada di train).
- K15 Daftar peringkat: item unik hasil voting, N = 20. Tanpa padding: jika item unik kurang dari K, Recall dihitung pada daftar yang ada.
- K16 Baseline popularitas: peringkat item menurut jumlah dialog di datastore yang memuat item itu (seri: movie_id). Daftar sama untuk semua query. Dihitung hanya dari datastore.
- K17 Tuning k: dialog train dibagi acak 80/20 (seed 42). Query validasi = 20%; datastore = 80%. Grid k = {1, 5, 10, 30, 60, 90, 120, 150}. Pilih k dengan Recall@20 validasi tertinggi (seri: k lebih kecil). Semua k dilaporkan.
- K18 Evaluasi akhir: query = 1229 dialog test; datastore = seluruh 9566 dialog train; k dari K17. Test tidak dipakai untuk memilih apa pun.

## Pencegahan data leakage
1. conversation_id test tidak ada di train (terverifikasi: 0 tumpang tindih).
2. Saat tuning, dialog validasi dikeluarkan dari datastore.
3. Konteks dipotong sebelum rekomendasi pertama (K3), sehingga item jawaban tidak ada di teks query. Pengecualian yang diketahui: judul sama dengan ID berbeda (41 pasangan train, 8 pasangan test).
4. Voting dan popularitas hanya memakai datastore.
5. Tidak ada keputusan (k, ambang, varian) yang dipilih memakai data test.

## Pelaporan wajib
Dataset (ReDial: 10006 train dan 1342 test dialog mentah; 9566 dan 1229 setelah filter K1-K3), preprocessing K1-K6, model embedding all-MiniLM-L6-v2, k, N, protokol di atas, versi library.

## Keterbatasan yang diketahui
- Hanya 91,6% dialog test dipakai; hasil tidak sebanding langsung dengan angka paper.
- Batas atas teoretis: 90,9% pasangan test punya film yang juga item di train.
- Datastore saat tuning (80%) lebih kecil daripada datastore akhir (100%).
- Satu split validasi dengan satu seed; tidak ada interval kepercayaan antar-seed.
- Top-N dapat memuat ID ganda untuk film yang sama (K5).
