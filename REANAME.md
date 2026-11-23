# Prediksi Arah Asap Kebakaran Hutan dan Lahan (Karhutla) di Pulau Sumatra Menggunakan AutoGluon-TimeSeries

Project capstone ini membangun sistem peramalan deret waktu (*time-series forecasting*) untuk memprediksi **arah pergerakan asap kebakaran hutan dan lahan (karhutla)** selama **7 hari ke depan** di **120 kabupaten pada 10 provinsi di Pulau Sumatra** menggunakan **AutoGluon-TimeSeries**.

Dokumentasi ini ditulis secara bertahap agar mudah dipahami oleh pemula yang sedang belajar Machine Learning dan AutoGluon.

---

## 1. Masalah Apa yang Diselesaikan?

Saat terjadi kebakaran hutan dan lahan, asap tidak diam di tempat—asap bergerak mengikuti **arah dan kecepatan angin**. Untuk membantu upaya pemantauan dan mitigasi, kita perlu memperkirakan ke arah mana asap akan bergerak dalam beberapa hari ke depan (`prediction_length = 7` hari).

### Mengapa Target Dipecah Menjadi `u_wind` dan `v_wind`?
Sudut arah (`0°–360°`) tidak bisa langsung diprediksi sebagai angka regresi biasa karena sifatnya melingkar:
- Secara nyata, sudut **`359°`** dan **`1°`** hanya berselisih **`2°`**.
- Namun secara angka biasa, `359 - 1 = 358`, sehingga model akan mengira kesalahannya sangat besar.

Solusi standarnya dalam meteorologi adalah memecah arah & kecepatan angin menjadi dua komponen vektor yang kontinu:
1. **`u_wind`**: komponen kecepatan angin arah **Barat → Timur** (m/s).  
   - Positif (`+`) = bergerak ke **Timur**, Negatif (`-`) = bergerak ke **Barat**.
2. **`v_wind`**: komponen kecepatan angin arah **Selatan → Utara** (m/s).  
   - Positif (`+`) = bergerak ke **Utara**, Negatif (`-`) = bergerak ke **Selatan**.

**Alur kerja model:**
1. Model AutoGluon 1 memprediksi `u_wind` (`7` hari ke depan).
2. Model AutoGluon 2 memprediksi `v_wind` (`7` hari ke depan).
3. Keduanya digabungkan dengan rumus `arctan2(u_pred, v_pred)` untuk mendapatkan **arah tujuan pergerakan asap (`0°–360°`)** dan **8 arah mata angin** (`Utara`, `Timur Laut`, `Timur`, `Tenggara`, `Selatan`, `Barat Daya`, `Barat`, `Barat Laut`).

---

## 2. Struktur File di Folder Project

```text
c:\Faiq\capstone\5\
├── dataset_pm25_hotspot_sumatra_2020_2025.xlsx   # File input gabungan PM2.5 & hotspot (2020–2025)
├── prepare_master_dataset.py                     # Skrip penggabungan & pembersihan data
├── dataset_sumatra_master_2022_2025.parquet      # Dataset master bersih siap AutoGluon (format cepat)
├── dataset_sumatra_master_2022_2025.csv          # Dataset master bersih (format CSV)
├── prediksi_arah_asap_autogluon.ipynb            # Notebook utama project (12 tahap lengkap)
├── hasil_prediksi_arah_asap_7hari.csv            # Output prediksi 7 hari ke depan (120 kabupaten)
├── autogluon_models\                             # Folder model terlatih (u_wind & v_wind)
└── README.md                                     # Dokumentasi project ini
```
*(Catatan: file sumber cuaca dan angin `dataset_cuaca_sumatra_2020_2025_valid.xlsx` dibaca otomatis oleh `prepare_master_dataset.py` dari folder `c:\Faiq\capstone\`).*

---

## 3. Penjelasan Dataset Master (`dataset_sumatra_master_2022_2025`)

Dataset master memiliki **149.520 baris × 17 kolom**:
- **Cakupan wilayah:** `120` kabupaten pada `10` provinsi di Pulau Sumatra (Aceh, Sumatera Utara, Sumatera Barat, Riau, Jambi, Sumatera Selatan, Bengkulu, Lampung, Kepulauan Bangka Belitung, Kepulauan Riau).
- **Rentang waktu:** `2022-08-04` s.d. `2025-12-31` (`1.246` hari berturut-turut per kabupaten, `0` missing values, `0` duplikasi).
- **Alasan mulai `2022-08-04`:** Data `pm25` baru tersedia secara riil sejak `2022-08-04`. Sebelum tanggal tersebut (`2020-01-01` s.d. `2022-08-03`), nilai `pm25` sempat terisi nilai konstan hasil *backward fill*, sehingga dipotong agar model hanya belajar dari data asli tanpa *data leakage*.

### Kamus Data (17 Kolom + 3 Fitur Kalender)

| Kolom | Peran di AutoGluon | Penjelasan |
| :--- | :--- | :--- |
| `kode_wilayah` | **`item_id`** | Kode unik tiap kabupaten dalam format teks 2 desimal (contoh `"11.10"`, `"14.01"`). |
| `tanggal` | **`timestamp`** | Tanggal pengamatan harian (`freq="D"`). |
| `u_wind` | **`target` (Model 1)** | Komponen vektor gerak Barat→Timur (m/s). |
| `v_wind` | **`target` (Model 2)** | Komponen vektor gerak Selatan→Utara (m/s). |
| `kabupaten`, `provinsi`, `latitude`, `longitude` | **`static_features`** | Identitas wilayah & koordinat geografis yang tidak berubah terhadap waktu. |
| `curah_hujan_mm` | **`past_covariate`** | Curah hujan harian (mm) — hujan menekan kebakaran & mengendapkan asap. |
| `suhu_rata_rata_c` | **`past_covariate`** | Suhu rata-rata harian (°C). |
| `kelembapan_rata_rata_persen` | **`past_covariate`** | Kelembapan relatif rata-rata harian (%). |
| `jumlah_hotspot` | **`past_covariate`** | Jumlah titik panas kebakaran harian per kabupaten. |
| `total_frp` | **`past_covariate`** | Total *Fire Radiative Power* (MW) yang menggambarkan kekuatan panas api. |
| `pm25` | **`past_covariate`** | Konsentrasi partikel halus PM2.5 (µg/m³) sebagai indikator keberadaan asap. |
| `bulan`, `sin_doy`, `cos_doy` | **`known_covariates`** | Fitur kalender musiman yang dibuat di notebook (nilainya sudah pasti diketahui untuk masa depan sehingga membantu model memahami musim kemarau/muson). |
| `kecepatan_angin_maks_ms`, `arah_angin_dominan_derajat`, `arah_gerak_asap_derajat` | *Referensi / Tidak dipakai sebagai input* | Tidak dimasukkan sebagai input pelatihan karena merupakan bentuk turunan langsung dari target (`u_wind`, `v_wind`). |

---

## 4. Memahami Konsep AutoGluon-TimeSeries (Untuk Pemula)

1. **`TimeSeriesDataFrame`**:
   Format tabel khusus AutoGluon yang memiliki 2 indeks wajib: `(item_id, timestamp)`. Satu file ini memuat seluruh `120` kabupaten sekaligus secara berurutan per hari.
2. **`past_covariates` vs `known_covariates`**:
   - **`past_covariates`**: data yang hanya kita punya sampai hari ini (hujan, suhu, PM2.5, hotspot hari kemarin/hari ini). Saat memprediksi 7 hari ke depan, kita belum tahu cuaca atau hotspot minggu depan.
   - **`known_covariates`**: data yang sudah pasti kita ketahui untuk 7 hari ke depan (kalender `bulan`, `sin_doy`, `cos_doy`). AutoGluon memanfaatkan ini untuk menangkap pola musiman tahunan.
3. **`prediction_length = 7`**:
   Menentukan seberapa jauh ke depan model diminta meramal (7 langkah waktu = 7 hari ke depan).
4. **`train_test_split(prediction_length=7)`**:
   Memotong `7` hari terakhir (`2025-12-25` s.d. `2025-12-31`) sebagai data uji (`test_data`), sementara `1.239` hari sebelumnya dipakai untuk melatih model (`train_data`).
5. **`TimeSeriesPredictor.fit(..., hyperparameters=CUSTOM_HPS)`**:
   AutoGluon secara otomatis melatih berbagai keluarga model:
   - **Statistik:** `Naive`, `SeasonalNaive`, `NPTS`
   - **Tree-Based:** `PerStepTabular` berbasis **CatBoost (`CAT`)** dan **LightGBM (`GBM`)**
   - **Deep Learning:** `TemporalFusionTransformer` dan `DeepAR`
   Lalu AutoGluon membandingkan performanya dan menggabungkan model terbaik menjadi `WeightedEnsemble`.
   > *(Catatan teknis: `RecursiveTabular` & `DirectTabular` pada preset default dihindari karena memanggil `mlforecast` → `xgboost.dll` yang terblokir kebijakan Windows Application Control `[WinError 4551]` di mesin ini; sebagai gantinya dipakai `PerStepTabular` berbasis CatBoost & LightGBM yang tetap memanfaatkan seluruh kovariat dan fitur statis).*

---

## 5. Cara Menjalankan Project

### Prasyarat Environment
Gunakan **Python 3.12** (kernel **`Python 3.12` / `python312`**), karena AutoGluon mendukung Python `3.10`–`3.13` (bukan Python `3.14`). Library utama yang digunakan (`autogluon.timeseries 1.6.3`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `nbformat`) sudah terpasang pada environment Python 3.12 Anda.

### Langkah Menjalankan
1. *(Opsional, jika data mentah diperbarui)* Jalankan skrip penggabungan dataset:
   ```powershell
   py -3.12 prepare_master_dataset.py
   ```
2. Buka [prediksi_arah_asap_autogluon.ipynb](file:///c:/Faiq/capstone/5/prediksi_arah_asap_autogluon.ipynb) di VS Code atau Jupyter.
3. Pastikan memilih kernel **`Python 3.12`** di pojok kanan atas notebook.
4. Jalankan sel secara berurutan dari atas ke bawah (**Run All**).
5. Output akhir berupa tabel prediksi 7 hari ke depan tersimpan di `hasil_prediksi_arah_asap_7hari.csv`.

---

## 6. Cara Membaca Metrik Evaluasi

- **RMSE (`u_wind` & `v_wind`):** Mengukur rata-rata akar kuadrat kesalahan prediksi kecepatan angin (m/s). Semakin kecil semakin akurat. *(Catatan: di tabel leaderboard AutoGluon, nilai `score_test` ditampilkan negatif seperti `-0.85` karena AutoGluon memakai aturan "semakin besar semakin baik", jadi RMSE sebenarnya adalah `0.85` m/s).*
- **Mean Angular Error (MAE Sudut):** Rata-rata selisih sudut terpendek antara arah gerak asap aktual dan prediksi (rentang `0°`–`180°`). Contoh: jika aktual `355°` dan prediksi `5°`, error-nya hanya `10°`.
- **Akurasi Toleransi `<= 45°` & Akurasi 8 Mata Angin:** Persentase prediksi yang arahnya masih meleset tidak lebih dari `45°` atau jatuh pada sektor mata angin yang tepat sama (misalnya sama-sama `Tenggara`).

---

## 7. Ringkasan Hasil Eksekusi & Validasi Resmi AutoGluon

Berdasarkan eksekusi penuh [prediksi_arah_asap_autogluon.ipynb](file:///c:/Faiq/capstone/5/prediksi_arah_asap_autogluon.ipynb) pada jendela uji `2025-12-25` s.d. `2025-12-31` (`120` kabupaten × `7` hari = `840` titik evaluasi):

- **Model Terbaik `u_wind`:** `WeightedEnsemble` (validasi `-1.3174`, test RMSE `1.6355` m/s; `PerStepTabular_2` test RMSE `1.6044` m/s), mengalahkan baseline `SeasonalNaive` (`2.0264`) dan `Naive` (`2.1434`).
- **Model Terbaik `v_wind`:** `WeightedEnsemble` (validasi `-1.6946`, test RMSE `1.6781` m/s), mengalahkan baseline `SeasonalNaive` (`2.1429`) dan `Naive` (`2.2498`).
- **Akurasi Arah Gerak Asap Gabungan (Seluruh Sumatra):**
  - **Mean Angular Error (MAE Sudut):** `38,78°` | **Median Angular Error:** `19,17°`
  - **Akurasi dalam toleransi `<= 45°`:** `70,60%` | **Akurasi tepat 8 mata angin:** `45,83%`
- **Performa di Provinsi Episenter Utama Karhutla:**
  - **Jambi:** MAE Sudut `11,11°` (Akurasi `<= 45°`: `96,83%`)
  - **Sumatera Selatan:** MAE Sudut `16,88°` (Akurasi `<= 45°`: `91,21%`)
  - **Riau:** MAE Sudut `27,86°` (Akurasi `<= 45°`: `82,86%`)
- **Output Operasional (`2026-01-01` s.d. `2026-01-07`):** Tersimpan di [hasil_prediksi_arah_asap_7hari.csv](file:///c:/Faiq/capstone/5/hasil_prediksi_arah_asap_7hari.csv) (`840` baris, `0` null).

---

## 8. Keterbatasan & Saran Pengembangan Lanjutan

1. **Ketinggian Angin:** Dataset memakai angin permukaan (`10 m`), sedangkan kepulan asap karhutla jarak jauh juga dipengaruhi angin lapisan atas (`850 hPa` / `925 hPa`).
2. **Validitas Data Hotspot:** Pastikan sumber data `jumlah_hotspot` dan `total_frp` yang digunakan pada laporan akhir berasal dari arsip satelit riil (misalnya **NASA FIRMS VIIRS/MODIS** atau **SiPongi KLHK**) agar variasi lonjakan karhutla antar tahun (seperti tahun El Niño 2023) tercermin akurat.
3. **Ketiadaan Prakiraan Cuaca Masa Depan (*NWP Known Covariates*):** Jika ingin akurasi horizon 7 hari lebih tinggi lagi, dapat ditambahkan prakiraan tekanan udara atau angin regional (misal dari ECMWF/BMKG) sebagai `known_covariates`.
4. **Efek Spasial Antarwilayah:** Untuk pengembangan lanjutan, dapat ditambahkan fitur *spatial lag* (hotspot dan arah angin dari kabupaten tetangga di hulu) atau model graf spasial (*Spatio-Temporal GNN*).
