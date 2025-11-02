import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes

# =========================================================
# 1. MEMBACA DATA DAN AGREGASI
# =========================================================

@st.cache_data
def load_data():
    try:
        df_full_clean = pd.read_csv(
            './main_data.csv',
            parse_dates=True,
            infer_datetime_format=True
        )
        if 'datetime' in df_full_clean.columns:
            df_full_clean['datetime'] = pd.to_datetime(df_full_clean['datetime'], errors='coerce')
            df_full_clean.set_index('datetime', inplace=True)
        else:
            try:
                df_full_clean.index = pd.to_datetime(df_full_clean.index, errors='coerce')
            except Exception as e:
                raise ValueError(f"Gagal mengubah index menjadi datetime: {e}")

        # Cek hasilnya
        if not isinstance(df_full_clean.index, pd.DatetimeIndex):
            raise TypeError("Index df_full_clean bukan DatetimeIndex setelah parsing.")

        print("✅ Data berhasil dimuat dan index bertipe DatetimeIndex.")
        
        pollutants_caap = ['PM2.5', 'NO2', 'SO2']
        
        # --- PERHITUNGAN PB 1: Tren Tahunan ---
        
        # Menghitung Rata-rata Pasca-CAAP Per Tahun
        df_post_caap_annual = df_full_clean[df_full_clean['Pre_CAAP'] == False].groupby('year')[pollutants_caap].mean()
        
        # Menghitung Rata-rata Pra-CAAP (Baseline 2013)
        df_pre_caap_baseline = df_full_clean[df_full_clean['Pre_CAAP'] == True][pollutants_caap].mean().to_frame().T
        df_pre_caap_baseline.index = [2013]
        
        # Menggabungkan untuk Line Plot (PB 1)
        df_viz_pb1_raw = pd.concat([df_pre_caap_baseline, df_post_caap_annual])
        df_viz_pb1_raw.index.name = 'year'

        # --- PERHITUNGAN PB 2: Persentase Perubahan Tahunan ---
        baseline_values = df_pre_caap_baseline.iloc[0]
        # Hitung Persentase Perubahan Tahunan
        df_annual_change_pct = ((df_post_caap_annual - baseline_values) / baseline_values) * 100
        
        return df_full_clean, df_viz_pb1_raw, df_annual_change_pct
        
    except FileNotFoundError:
        st.error("Error: Pastikan file data (data_full_dashboard.csv) ada di folder aplikasi Streamlit.")
        st.stop()

df_full, df_viz_pb1_raw, df_annual_change = load_data()


# --- Inisialisasi Data Visualisasi ---

# PB 1: Data untuk Line Plot
df_viz_pb1 = df_viz_pb1_raw.reset_index()

# PB 2: Data untuk Bar Chart
df_viz_pb2_long_pct = df_annual_change.reset_index().melt(
    id_vars='year', var_name='Pollutant', value_name='Percentage_Change'
)
df_viz_pb2_long_pct['year'] = df_viz_pb2_long_pct['year'].astype(str)
pollutants_caap = ['PM2.5', 'NO2', 'SO2']

# PB 3: Data Ozon
df_summer = df_full[df_full['Season'] == 'Summer'].copy()
df_viz_pb3_scatter = df_summer[['O3', 'NO2']].sample(frac=0.01, random_state=42) 
df_viz_pb3_boxplot = df_summer[['O3', 'Area_Type']]

# PB 4: Data Wind Rose (Filter data ekstrem)
pm25_q_cutoff = 200 
df_viz_pb4_windrose = df_full[['PM2.5', 'WSPM', 'wd']].dropna().copy()
wd_mapping = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 
              'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 
              'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
df_viz_pb4_windrose['wd_deg'] = df_viz_pb4_windrose['wd'].map(wd_mapping).fillna(0)
df_extreme_wind = df_viz_pb4_windrose[df_viz_pb4_windrose['PM2.5'] > pm25_q_cutoff].copy()


# =========================================================
# 2. FUNGSI PLOTTING UTAMA
# =========================================================

def plot_pb1(df):
    """Plot Tren Konsentrasi Polutan Tahunan (2013-2017)"""
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    plt.suptitle('PB 1: Tren Konsentrasi Polutan Tahunan (2013-2017)', fontsize=16, y=1.02)
    pollutant_colors = {'PM2.5': 'tab:red', 'NO2': 'darkorange', 'SO2': 'tab:blue'}
    
    for i, pol in enumerate(['PM2.5', 'NO2', 'SO2']):
        ax = axes[i]
        sns.lineplot(ax=ax, data=df, x='year', y=pol, marker='o', color=pollutant_colors[pol])
        ax.set_title(f'{pol} Trend', loc='left', fontsize=12)
        ax.set_ylabel('Concentration ($\mu g/m^3$)')
        ax.set_xlabel('Year')
        ax.set_xticks(df['year'])
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_pb2(df_long_pct):
    """Plot Persentase Perubahan Tahunan (vs. Pra-CAAP)"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=False)
    plt.suptitle('PB 2: Evaluasi Dampak CAAP - Persentase Perubahan Tahunan (vs. Pra-CAAP)', fontsize=16)

    for i, pol in enumerate(pollutants_caap):
        ax = axes[i]
        df_pol = df_long_pct[df_long_pct['Pollutant'] == pol]
        colors = ['r' if p > 0 else 'g' for p in df_pol['Percentage_Change']]

        sns.barplot(ax=ax, data=df_pol, x='year', y='Percentage_Change', palette=colors, legend=False)
        ax.set_title(f'{pol} Change', fontsize=12)
        ax.set_xlabel('Year')
        ax.set_ylabel('Change (%) vs. Pra-CAAP Baseline')
        ax.axhline(0, color='gray', linestyle='--')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Anotasi
        for container in ax.containers:
            for bar in container:
                yval = bar.get_height()
                if yval >= 0:
                    y_text_pos = yval + 0.5 
                    va_align = 'bottom'
                else:
                    y_text_pos = yval - 0.5 
                    va_align = 'top'

                ax.text(bar.get_x() + bar.get_width() / 2, y_text_pos,
                        f'{yval:+.1f}%', ha='center', va=va_align, 
                        fontsize=9, fontweight='bold', color='black')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_pb3(df_scatter, df_boxplot):
    """Plot Ozone Paradox: Scatter O3 vs NO2 & Boxplot O3 per Area"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle('PB 3: Analisis Dampak Domino Ozon (Ozone Paradox)', fontsize=16)

    # A. Scatter Plot (Korelasi Ozon vs. NO₂)
    sns.regplot(
        data=df_scatter, x='NO2', y='O3',
        scatter_kws={'alpha': 0.3, 's': 10, 'color': '#f58518'}, 
        line_kws={'color': 'red', 'linewidth': 2},
        ax=axes[0]
    )
    axes[0].set_title('Korelasi Ozon vs. NO₂ Musim Panas', fontsize=12)
    axes[0].set_xlabel('NO₂ Konsentrasi ($\mu g/m^3$)')
    axes[0].set_ylabel('O₃ Konsentrasi ($\mu g/m^3$)')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # B. Box Plot O3 per Area Type
    sns.boxplot(
        data=df_boxplot, x='Area_Type', y='O3',
        order=['Urban', 'Suburban', 'Rural'], palette='pastel', ax=axes[1]
    )
    axes[1].set_title('Distribusi Ozon Berdasarkan Tipe Area', fontsize=12)
    axes[1].set_xlabel('Tipe Area (NOₓ Tinggi ke Rendah)')
    axes[1].set_ylabel('O₃ Konsentrasi ($\mu g/m^3$)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_pb4_windrose(df_extreme):
    """Plot Wind Rose Frekuensi WSPM pada Kondisi PM2.5 Ekstrem"""
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    
    ax.bar(
        df_extreme['wd_deg'], 
        df_extreme['WSPM'],
        normed=True,                
        opening=0.8,
        edgecolor='white',
        bins=np.arange(0, 10, 1),
        cmap=plt.cm.viridis
    )
    
    ax.set_legend(title='Kecepatan Angin (m/s)', loc='lower left', bbox_to_anchor=(-0.1, 0))
    ax.set_title('PB 4: Wind Rose PM2.5 Ekstrem (> 200)')
    
    return fig


# =========================================================
# 3. STREAMLIT APP LAYOUT
# =========================================================

st.set_page_config(layout="wide", page_title="Analisis Kualitas Udara Beijing")

st.title("🏙️ Analisis Kualitas Udara Beijing (2013–2017): Evaluasi Dampak CAAP dan Tantangan Diagnostik")

st.header("I. Ringkasan Utama")
st.markdown("""
Analisis menunjukkan **CAAP** (Clean Air Action Plan) oleh Pemerintah Beijing sukses menekan SO2 tetapi menghadapi tantangan besar dari **Anomali 2017**, **Ozone Paradox**, dan faktor **Stagnasi Udara** meteorologi.
""")

st.subheader("Evaluasi Kebijakan CAAP")
# Mengambil data dari df_annual_change yang sudah dihitung
pm25_2017_change = df_annual_change.loc[2017, 'PM2.5']
so2_2016_change = df_annual_change.loc[2016, 'SO2']
pm25_2016_change = df_annual_change.loc[2016, 'PM2.5']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PM2.5 di 2016", f"{pm25_2016_change:.2f}%", "Keberhasilan Jangka Pendek")
with col2:
    st.metric("SO2 di 2016", f"{so2_2016_change:.2f}%", "Sukses Besar (Batubara)")
with col3:
    st.metric("PM2.5 di 2017", f"{pm25_2017_change:.2f}%", "Anomali (Menghapus Kemajuan)")
    
st.markdown("---")


# =========================================================
# TEMUAN VISUAL PER Pertanyaan Bisnis
# =========================================================

st.header("II. Visualisasi & Bukti Kunci")

# PB 1: Tren
st.subheader("1. Tren Polutan Utama Tahunan")
st.pyplot(plot_pb1(df_viz_pb1), use_container_width=True)

# PB 2: Evaluasi CAAP Per Tahun
st.subheader("2. Dampak CAAP: Persentase Perubahan vs. Pra-CAAP")
st.pyplot(plot_pb2(df_viz_pb2_long_pct), use_container_width=True)


# PB 3: Ozone Paradox
st.subheader("3. Analisis Dampak Domino Ozon")
st.pyplot(plot_pb3(df_viz_pb3_scatter, df_viz_pb3_boxplot), use_container_width=True)


# PB 4: Wind Rose & Stagnasi
st.subheader("4. Peran Kausal Meteorologi")
colA, colB = st.columns([1, 2])
stagnation_ratio = 2.49 
with colA:
    st.metric("Rasio Stagnasi PM2.5", f"{stagnation_ratio:.2f}x Lebih Tinggi")
    st.markdown("""
    * **Stagnasi (Akumulasi):** Rata-rata *PM2.5* saat stagnasi adalah **2.49 kali lebih tinggi** saat kondisi angin pelan.
    * **Adveksi (Transportasi):** Polusi ekstrem dibawa masuk dari jalur **Timur Laut** (N-E), menegaskan perlunya pengendalian emisi regional.
    """)
with colB:
    st.pyplot(plot_pb4_windrose(df_extreme_wind), use_container_width=True)


st.markdown("---")
st.header("III. Kesimpulan Analisis")
st.markdown("""
            

### 🌟 Implikasi Strategis

1.  CAAP sukses menekan emisi batubara (**SO2** turun **44.7%** di **2016**). Namun, lonjakan **PM2.5** dan **NO2** di **2017** menegaskan bahwa kontrol **NO2** dan keberlanjutan adalah tantangan utama.

2.  Beijing berada dalam rezim **VOC-limited**, dibuktikan dengan O3 tertinggi di area **Rural**. Pengurangan NO2 tanpa dibarengi pengurangan **VOC** berisiko **memperburuk masalah **O3** (*Ozone Paradox*).

3.  **Stagnasi Udara** adalah mekanisme akumulasi polutan yang paling signifikan. Rata-rata **PM2.5** saat stagnasi adalah **2.49 kali lebih tinggi**, sementara jalur **Adveksi** dari sektor Timur Laut (N-E) menuntut **strategi pengendalian emisi regional** yang komprehensif.
""")