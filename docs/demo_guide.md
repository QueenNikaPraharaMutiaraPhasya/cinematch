# CineMatch: Panduan Demo dan Bahan Presentasi

## 1. Pesan utama (sekitar 30 detik)

- CineMatch adalah **prototype web rekomendasi film berbasis percakapan**. Ia mengimplementasikan konsep NBCRS (komponen neighborhood-based tanpa training). Ia **bukan replikasi penuh paper**.
- Cara kerja: query pengguna -> sentence embedding -> percakapan ReDial yang mirip (top-k) -> voting film yang disarankan di percakapan itu -> Top-N.
- Hasil terukur: pada 1.229 dialog test, NB mengungguli baseline popularitas di semua K. Angkanya tidak sebanding langsung dengan paper.
- Keterbatasan yang jujur: hanya bahasa Inggris, tidak memahami negasi, dan `top_similarity` bukan ukuran keyakinan.

## 2. Persiapan (sebelum hari presentasi)

Semua perintah di PowerShell, dari folder `cinematch`.

1. **Backend** (jendela 1, `.venv` aktif):

   ```powershell
   python -m uvicorn backend.app.main:app --port 8000
   ```

   Tunggu `Application startup complete.` (butuh beberapa detik untuk memuat model).
2. **Cek kesehatan:** buka <http://127.0.0.1:8000/health>. Harus tampil `"status":"ok"` dan `"datastore_conversations":9566`.
3. **Frontend** (jendela 2): `cd frontend`, lalu `npm run dev`. Buka **persis** <http://localhost:5173>.
4. **Smoke test** (jendela 3, `.venv` aktif): `python scripts\smoke_test_api.py`. Harus berakhir dengan `gagal 0`.
5. **Gladi:** jalankan tiga query pada bagian 4 dan cocokkan hasilnya dengan tabel.
6. **Tanpa internet:** model diunduh dari internet saat pertama kali dipakai. Perilaku saat benar-benar offline **belum diverifikasi**. Uji dulu (nyalakan backend dengan internet dimatikan) sebelum mengandalkannya di tempat presentasi.
7. **Cadangan:** simpan screenshot hasil tiap langkah demo, dan buka `docs/evaluation_report.txt` serta `docs/architecture.md` di tab terpisah.

## 3. Skenario demo (sekitar 7 menit)

| # | Langkah | Yang ditunjukkan | Yang dikatakan |
|---|---|---|---|
| 1 | Buka halaman, tunjuk catatan di bagian bawah | Keterangan "prototype bukan replikasi paper", keterbatasan, dan atribusi ReDial | "Sistem ini hanya untuk bahasa Inggris dan hanya mencocokkan kemiripan teks." |
| 2 | Klik contoh **psychological thriller** | Top-10, batang suara, panel tetangga | "Sistem mencari percakapan yang mirip, lalu menghitung film yang paling sering disarankan di sana." |
| 3 | Buka **Percakapan pendukung** pada film teratas | Daftar percakapan beserta similarity | "Setiap rekomendasi punya bukti: percakapan mana yang menyarankannya." |
| 4 | **Pengaturan lanjutan**: k = 5, N = 3, cari lagi | Top-3, "dihitung dari 5 percakapan" | "k menentukan seberapa banyak tetangga dipakai. Evaluasi memilih k = 90." |
| 5 | Kembalikan k = 90 dan N = 10, klik contoh **something like Inception** | Suara tertinggi hanya 5 dari 90 | "Suara rendah berarti konsensus rendah. Suara adalah bukti, bukan jaminan." |
| 6 | Klik contoh **romantic comedy** | Konsensus tinggi (15 dari 90) | "Permintaan umum menghasilkan konsensus yang lebih tinggi." |
| 7 | Ketik `hi`, lalu `film horor yang menakutkan` | Similarity 1,000 untuk "hi", hasil tidak masuk akal untuk bahasa Indonesia | "Kemiripan teks bukan keyakinan. 'hi' mirip sempurna dengan percakapan sapaan, bukan dengan selera film." |
| 8 | Ketik query dengan negasi (lihat bagian 4) | Sistem tetap mengembalikan thriller | "Sistem tidak memahami 'tanpa banyak kekerasan'. Ia hanya mencocokkan kemiripan teks." |
| 9 | Kosongkan kotak, klik tombol | Pesan "Query kosong ..." tanpa error | "Kasus tepi ditangani, dan ada 69 tes otomatis." |
| 10 | Tampilkan tabel evaluasi (bagian 5) | NB vs popularitas vs plafon | "Hasil terukur dengan protokol yang ditetapkan sebelum melihat hasil." |

## 4. Query demo dan hasil yang tercatat

Hasil di bawah tercatat pada komputer pengembang (`docs/smoke_test_report.txt`), dengan k = 90 dan N = 10. Angka dalam kurung adalah jumlah suara dari 90. Hasil seharusnya sama saat demo. Jika berbeda, kabari pengembang dan jangan menyebutnya "benar".

| Query | `top_similarity` | 3 film teratas |
|---|---|---|
| `I want a psychological thriller with a plot twist` | 0,859 | Get Out (2017) (13); Seven (1995) (11); Shutter Island (2010) (11) |
| `something like Inception` | 0,727 | The Matrix (1999) (5); Star Wars: Episode VIII - The Last Jedi (2017) (5); Taken (2008) (5) |
| `a feel-good romantic comedy for a rainy evening` | 0,818 | 50 First Dates (2004) (15); How to Lose a Guy in 10 Days (2003) (15); 10 Things I Hate About You (1999) (13) |
| `hi` | 1,000 | Black Panther (2018) (11); Thor: Ragnarok (2017) (4); Memento (2000) (4) |
| `film horor yang menakutkan` (Indonesia) | 0,544 | Black Panther (2018) (6); Thor: Ragnarok (2017) (4); Get Out (2017) (4) |

Query negasi dari brief proyek: `I want a psychological thriller with a plot twist, but without too much violence.` Hasilnya **belum dicatat pada k = 90**. Jalankan saat gladi dan amati sendiri. Sistem tidak punya mekanisme yang menyaring kekerasan, jadi **jangan mengklaim** hasilnya "tanpa kekerasan".

## 5. Tabel hasil untuk slide

Sumber: `docs/evaluation_report.txt`. Test: 1.229 dialog, k = 90; persen, rata-rata ± standard error.

| Metode | Recall@1 | Recall@5 | Recall@20 | Hit@20 |
|---|---|---|---|---|
| NB zero-shot | 2,40 ± 0,28 | 8,64 ± 0,52 | 19,35 ± 0,77 | 42,72 ± 1,41 |
| Popularitas | 1,35 ± 0,21 | 3,93 ± 0,35 | 9,23 ± 0,54 | 24,33 ± 1,22 |
| Plafon | 36,64 | 99,10 | 100,00 | - |

Cara membaca: rata-rata ada sekitar 3,3 film relevan per dialog, sehingga sistem sempurna pun hanya mencapai 36,64% pada Recall@1. Hit@20 = 42,72% berarti pada sekitar 43 dari 100 dialog test, minimal satu film relevan masuk 20 teratas.

## 6. Pertanyaan yang mungkin muncul, dan jawaban jujur

**Apakah ini replikasi paper?**
Bukan. Ini prototype implementasi konsep NBCRS. Hanya komponen NB tanpa training yang dibuat. Dataset olahan, pembentukan konteks dan item, serta protokol evaluasi adalah keputusan sendiri yang tercatat di `docs/decisions.md`.

**Apakah hasilnya lebih baik dari paper?**
Tidak bisa disimpulkan. Paper melaporkan Recall pada Redial dengan definisi sampel dan jumlah dialog yang berbeda. Angka kita tidak sebanding langsung, dan kami tidak mengklaim lebih baik ataupun lebih buruk.

**Mengapa Recall@1 rendah?**
Karena rata-rata sekitar 3,3 film relevan per dialog, plafon Recall@1 hanya 36,64%. Hit@K lebih mudah dibaca.

**Mengapa k = 90?**
Dipilih dari data validasi (20% dari train, seed 42), bukan dari test. Untuk k antara 60 dan 150 selisihnya kecil dibanding standard error, jadi k = 90 bukan puncak yang tegas.

**Mengapa hasil validasi lebih tinggi dari test?**
Recall@20 validasi 26,28% vs test 19,35%. Dugaan: data validasi berasal dari pekerja yang sama dengan datastore, sedangkan data test dikumpulkan belakangan oleh pekerja yang berbeda. Dugaan ini belum diuji. Angka test adalah hasil resmi.

**Mengapa film seperti Black Panther muncul untuk query tak bermakna?**
Pengamatan pada sebelas query. Dugaan: tetangga query seperti itu adalah konteks pendek dan generik yang saran filmnya condong ke film populer. Dugaan ini belum diuji.

**Apakah `top_similarity` menunjukkan keyakinan sistem?**
Tidak. "hi" mendapat 1,000 karena ada konteks train yang isinya hanya sapaan. Karena itu UI menyebutnya "kemiripan teks", bukan keyakinan.

**Apakah memakai LLM?**
Tidak. Hanya model embedding `all-MiniLM-L6-v2` (CPU) dan voting.

**Berapa lama waktu responsnya?**
Median 17 ms untuk satu query yang diulang 20 kali, diukur di satu komputer (CPU saja), termasuk HTTP lokal. Itu bukan klaim umum.

**Dari mana datanya, dan apa lisensinya?**
ReDial (Li dkk., NeurIPS 2018), lisensi CC BY 4.0. Atribusinya ada di aplikasi dan README.

## 7. Kerangka slide

1. **Judul dan tujuan:** CineMatch, prototype implementasi konsep NBCRS (bukan replikasi paper).
2. **Masalah dan ide:** permintaan bahasa alami sering sudah pernah dijawab orang lain pada percakapan serupa (paper).
3. **Alur sistem:** diagram dari `docs/architecture.md`, bagian 3.
4. **Data dan preprocessing:** ReDial; 10.006 -> 9.566 dialog train dan 1.342 -> 1.229 dialog test.
5. **Dua keputusan penting:** item = film yang disepakati dua peserta sebagai saran (K1); konteks = pesan seeker sebelum rekomendasi pertama (K3).
6. **Demo langsung.**
7. **Hasil evaluasi:** tabel bagian 5, plafon, serta catatan ketidaksebandingan dengan paper.
8. **Keterbatasan:** hanya Inggris, tanpa negasi, similarity bukan keyakinan, dan evaluasi pada 1.229 dari 1.342 dialog.
9. **Perbedaan dengan paper:** hanya NB, satu dataset, protokol sendiri.
10. **Pekerjaan lanjutan:** sinyal keyakinan, alias judul, beberapa seed, model multibahasa, MB dan N+M.

## 8. Daftar periksa 5 menit sebelum presentasi

- [ ] Backend menyala dan `/health` menampilkan `ok` dan 9566.
- [ ] Frontend dibuka di `http://localhost:5173` (persis).
- [ ] Satu query uji sudah dijalankan dan hasilnya tampil.
- [ ] Program berat lain ditutup.
- [ ] Screenshot cadangan siap.
- [ ] `docs/evaluation_report.txt` dan `docs/architecture.md` terbuka di tab terpisah.
- [ ] Pertanyaan "apakah ini replikasi paper?" sudah punya jawaban siap.