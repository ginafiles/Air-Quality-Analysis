# 📊 Analisis Kualitas Udara Beijing (2013–2017): Evaluasi Dampak CAAP dan Tantangan Diagnostik

Proyek ini adalah analisis data eksploratif dan eksplanatori yang mendalam untuk mengevaluasi efektivitas kebijakan **Beijing Clean Air Action Plan (CAAP)** selama periode 2013 hingga 2017. Analisis difokuskan pada tren polutan utama (PM2.5, SO2, NO2), tantangan kimia atmosfer (Ozone Paradox), dan peran kausal meteorologi (Stagnasi Udara). Dataset yang digunakan untuk proyek analisis ini adalah [Beijing Multi-Site Air-Quality Data Set](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set/data).

## 🌟 Fitur Utama & Temuan Kunci

* **Keberhasilan:** CAAP berhasil menekan **SO2** secara drastis (penurunan **44.7%** di 2016). Namun, lonjakan PM2.5 dan NO2 di 2017 menegaskan bahwa kontrol NO2 dan keberlanjutan adalah tantangan utama.
* **Ozone Paradox:** Terdapat indikasi kuat rezim **VOC-limited**, dibuktikan oleh korelasi negatif antara O3 dan NO2 serta O3 tertinggi di area **Rural**.
* **Kausalitas Meteorologi:** **Stagnasi Udara** adalah penyebab dominan akumulasi polutan (meningkatkan PM2.5 hingga **2.49** kali lipat). Jalur **Adveksi** (transportasi) dari **Timur Laut (N-E)** menuntut strategi regional.

---

## 📂 Struktur Direktori Proyek
Air-Quality-Analysis <br>
├───dashboard <br>
| ├───main_data.csv <br>
| └───dashboard.py <br>
├───data <br>
| ├───data_1.csv <br>
| └───data_2.csv <br>
├───notebook.ipynb <br>
├───README.md <br>
└───requirements.txt <br>
└───url.txt <br>

---

## 💻 Panduan Instalasi dan Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan *dashboard* interaktif di lingkungan lokal Anda.

### 1. Prasyarat

Pastikan Python (versi 3.8+) terinstal. Instal semua pustaka yang dibutuhkan menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Penyiapan File
Pastikan file dashboard.py dan main_data.csv berada dalam folder dashboard/.

### 3. Menjalankan Dashboard
Arahkan Terminal atau Command Prompt ke folder dashboard/ dan jalankan aplikasi:

```bash
cd dashboard
streamlit run dashboard.py
```

Aplikasi akan terbuka secara otomatis di web browser Anda
