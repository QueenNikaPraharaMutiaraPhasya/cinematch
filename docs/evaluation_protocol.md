# Protokol Evaluasi CineMatch (prototype implementasi konsep NBCRS)

Status: rancangan v1, disepakati SEBELUM kode evaluasi ditulis.
Ini bukan replikasi eksperimen paper.

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