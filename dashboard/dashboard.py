import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from windrose import WindroseAxes

# =========================================================
#                   HELPER FUNCTIONS (PB 4)
# =========================================================

def convert_cardinal_to_degree(wd_series):
    """
    PB 4 Helper: Mengkonversi arah angin (kardinal/string) menjadi derajat numerik (0-360).
    Digunakan di plot_pb4_windrose.
    """
    
    # 16 arah kardinal
    cardinal16 = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    # Mapping string kardinal ke nilai derajat tengah sektor
    cardinal_to_deg = {d: i*(360/16) for i,d in enumerate(cardinal16)}

    # map string kardinal
    wd_deg = wd_series.map(cardinal_to_deg)
    
    # Mengisi nilai yang tersisa dengan mencoba konversi numerik langsung
    wd_numeric = pd.to_numeric(wd_series, errors='coerce')
    wd_deg = wd_deg.fillna(wd_numeric)
    
    # Menghapus NaN dan normalisasi range 0-360
    wd_deg = wd_deg.dropna()
    if not wd_deg.empty:
        wd_deg = wd_deg.astype(float) % 360
        
    return wd_deg


# =========================================================
#                   VISUALIZATION FUNCTIONS
# =========================================================

# --- PB 1: Line Plot Gabungan ---
def plot_pb1_combined_dynamic(df_filtered):
    """PB 1: Membuat Line Plot tren polutan tahunan (PM2.5, NO2, SO2)."""
    df_plot = df_filtered.groupby('year')[['PM2.5', 'NO2', 'SO2']].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(
        kind='line',
        marker='o',
        linewidth=2,
        ax=ax,
        color={'PM2.5': 'tab:red', 'NO2': 'darkorange', 'SO2': 'tab:blue'},
    )

    ax.set_title('Tren Konsentrasi Polutan Utama Tahunan', fontsize=14, fontweight='bold')
    ax.set_ylabel('Konsentrasi Polutan (µg/m³)')
    ax.set_xlabel('Tahun')
    ax.set_xticks(df_plot.index.astype(int))
    ax.legend(title='Polutan', frameon=False, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    max_val = df_plot.max().max()
    # Anotasi Tahun 2017
    ax.axvline(x=2017, color='gray', linestyle='--', alpha=0.5)
    ax.text(2017, max_val * 1.05, 'Bias Data 2017', color='gray', fontsize=9, ha='center')

    plt.tight_layout()
    return fig

# --- PB 2: Bar Chart Persentase Perubahan ---
def plot_pb2(df_change_filtered):
    """PB 2: Membuat Bar Chart persentase perubahan polutan vs. Baseline Pra-CAAP."""
    # Mengubah dari wide ke long format untuk plotting Seaborn
    df_viz = df_change_filtered.reset_index().melt(
        id_vars='year', var_name='Pollutant', value_name='Percentage_Change'
    )
    
    df_viz = df_viz[df_viz['year'] >= 2014].copy()
    df_viz['year'] = df_viz['year'].astype(str)
    
    n_pollutants = len(df_viz['Pollutant'].unique())
    if n_pollutants == 0: return None, "Pilih minimal satu polutan untuk divisualisasikan."

    fig, axes = plt.subplots(nrows=1, ncols=n_pollutants, figsize=(5 * n_pollutants, 5), sharey=False)
    
    if n_pollutants == 1: axes = [axes]
    
    plt.suptitle('Evaluasi Dampak CAAP: Persentase Perubahan Tahunan (vs. Baseline Pra-CAAP)', 
                 fontsize=14, fontweight='bold')
    
    for i, pol in enumerate(df_viz['Pollutant'].unique()):
        ax = axes[i]
        df_pol = df_viz[df_viz['Pollutant'] == pol]
        
        # Penentuan warna: Hijau (perbaikan/penurunan), Merah (memburuk/kenaikan)
        colors = ['green' if p < 0 else 'red' for p in df_pol['Percentage_Change']]

        sns.barplot(ax=ax, data=df_pol, x='year', y='Percentage_Change', palette=colors, legend=False)
        
        ax.set_title(f'{pol} Change', fontsize=12)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Perubahan (%) vs. Pra-CAAP Baseline')
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Tambahkan data label
        for container in ax.containers:
            for bar in container:
                yval = bar.get_height()
                padding = 1.5 
                y_text_pos = yval + padding if yval >= 0 else yval - padding 
                va_align = 'bottom' if yval >= 0 else 'top'

                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    y_text_pos,
                    f'{yval:+.1f}%', 
                    ha='center', va=va_align, 
                    fontsize=9, fontweight='bold', color='black'
                )
                
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, None

# --- PB 3: Line Plot Korelasi ---
def plot_pb3_correlation_trend(corr_series):
    """PB 3: Membuat Line Plot tren koefisien korelasi tahunan O3 vs NO2 Musim Panas."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    corr_series.plot(kind='line', marker='o', ax=ax, linewidth=2, color='darkorange', markersize=8)

    ax.set_title('Tren Koefisien Korelasi O₃ vs. NO₂ Musim Panas (2013–2017)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Koefisien Korelasi Pearson (r)')
    ax.set_xticks(corr_series.index.astype(int))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.grid(axis='both', linestyle=':', alpha=0.6)
    
    # Anotasi hasil Korelasi
    for year, r_val in corr_series.items():
        ax.annotate(
            f'r={r_val:.3f}', 
            (year, r_val), 
            textcoords="offset points", 
            xytext=(0, 10 if r_val < 0 else -15), 
            ha='center', 
            fontsize=10, 
            fontweight='bold', 
            color='darkorange'
        )

    plt.tight_layout()
    return fig

# --- PB 3: Box Plot Ozon per Area Type ---
def plot_pb3_boxplot(df_full):
    """PB 3: Membuat Box Plot distribusi Ozon Musim Panas berdasarkan Tipe Area."""
    # Note: df_full di sini sudah difilter berdasarkan tahun, tetapi tidak berdasarkan Area Global
    df_viz = df_full[df_full['Season'] == 'Summer'].copy()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df_viz,
        x='Area_Type',
        y='O3',
        order=['Urban', 'Suburban', 'Rural'],
        palette='pastel',
        ax=ax
    )
    # Tambahkan informasi tahun ke judul
    years = df_viz['year'].unique()
    year_title = f"Tahun: {', '.join(map(str, sorted(years))) if len(years) < 5 else 'Overall'}"
    
    ax.set_title(f'Distribusi Ozon Musim Panas Berdasarkan Tipe Area\n({year_title})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tipe Area')
    ax.set_ylabel(r'O₃ Konsentrasi ($\mu g/m^3$)')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# --- PB 4: Box Plot Stagnasi PM2.5 ---
def plot_pb4_boxplot_stagnation(df_filtered):
    """PB 4: Membuat Box Plot perbandingan PM2.5 saat Stagnan vs. Normal."""

    if df_filtered.empty:
        return None, "Data tidak cukup untuk Box Plot."

    df_plot = df_filtered.copy()
    df_plot['Is_Stagnant_label'] = df_plot['Is_Stagnant'].apply(
        lambda x: 'Stagnant' if x is True else 'Normal' if x is False else str(x)
    )

    unique_labels = df_plot['Is_Stagnant_label'].unique().tolist()
    if not any(l in ('Stagnant', 'Normal') for l in unique_labels):
        return None, "Kolom 'Is_Stagnant' tidak berisi nilai yang dikenali (Stagnant/Normal)."

    order = ['Normal', 'Stagnant']
    palette = {'Stagnant': '#B71C1C', 'Normal': '#1565C0'}

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df_plot,
        x='Is_Stagnant_label',
        y='PM2.5',
        order=order,
        palette=palette,
        ax=ax
    )

    ax.set_title(
        r'Distribusi $\text{PM2.5}$ Saat Stagnasi Udara (WSPM < 3.2 m/s)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Kondisi Udara', fontsize=12)
    ax.set_xticklabels(['Angin Normal', 'Stagnan (WSPM < 3.2 m/s)'])
    ax.set_ylabel(r'$\text{PM2.5}$ Konsentrasi ($\mu g/m^3$)')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig, None


# --- PB 4: Wind Rose Plot ---
def plot_windrose_single_condition(df_filtered_raw, pm25_condition, filter_title):
    """
    PB 4: Membuat plot windrose untuk satu kondisi PM2.5 tertentu.
    pm25_condition: 'Normal' (PM2.5 < 75) atau 'Extreme' (PM2.5 > 200).
    """

    # Menerapkan Filter Kondisi PM2.5
    if pm25_condition == 'Normal':
        df_filtered = df_filtered_raw[df_filtered_raw['PM2.5'] < 75].copy()
        plot_title = f"Wind Rose: Kondisi Normal (< 75) | {filter_title}"
    elif pm25_condition == 'Extreme':
        df_filtered = df_filtered_raw[df_filtered_raw['PM2.5'] > 200].copy()
        plot_title = f"Wind Rose: Kondisi Ekstrem (> 200) | {filter_title}"
    else:
        return None, "Kondisi PM2.5 tidak valid."

    # Menghapus NA pada kolom krusial
    df_plot = df_filtered[['WSPM', 'wd']].dropna().copy()
    
    if df_plot.empty:
        return None, f"Data {pm25_condition} tidak cukup setelah pemfilteran."

    if df_plot.index.has_duplicates:
          df_plot = df_plot[~df_plot.index.duplicated(keep='first')]

    wd_deg = convert_cardinal_to_degree(df_plot['wd'])
    
    if wd_deg.index.has_duplicates:
          wd_deg = wd_deg[~wd_deg.index.duplicated(keep='first')]

    # Sinkronisasi data WSPM dan wd_deg menggunakan concat
    df_plot = pd.concat([df_plot[['WSPM']], wd_deg.rename('wd_deg')], axis=1, join='inner')
    df_plot = df_plot.dropna(subset=['WSPM', 'wd_deg']).copy()
    
    if df_plot.empty:
        return None, f"Data angin untuk kondisi {pm25_condition} tidak valid ({len(df_filtered)} baris)."
    
    # PLOTTING
    speed_bins = np.arange(0, 10, 1)

    fig = plt.figure(figsize=(6, 6))
    ax = WindroseAxes.from_ax(fig=fig)
    cmap_object = cm.get_cmap('viridis') 

    ax.bar(
        df_plot['wd_deg'], 
        df_plot['WSPM'], 
        normed=True,
        opening=0.8,
        edgecolor='white',
        bins=speed_bins,
        cmap=cmap_object
    )

    ax.set_title(plot_title, fontsize=10, fontweight='bold')
    ax.set_legend(title='WSPM (m/s)', loc='lower left', bbox_to_anchor=(-0.1, -0.1))

    plt.tight_layout()
    return fig, None

# --- PB 4: Perhitungan Dampak Stagnasi ---
def calculate_pb4_impact(df_filtered):
    """PB 4: Menghitung perbandingan PM2.5 rata-rata saat Stagnan vs. Normal (untuk Metrik)."""
    
    if df_filtered.empty:
        return None, "Data tidak cukup untuk analisis dampak stagnasi."

    stagnation_analysis = df_filtered.groupby('Is_Stagnant')[['PM2.5', 'WSPM']].mean()
    
    if True not in stagnation_analysis.index or False not in stagnation_analysis.index:
        return None, "Hanya ada data Stagnan atau data Normal. Perbandingan tidak mungkin."
        
    pm25_stagnant = stagnation_analysis.loc[True, 'PM2.5']
    pm25_normal = stagnation_analysis.loc[False, 'PM2.5']
    
    if pm25_normal == 0:
        return None, "Rata-rata PM2.5 saat Normal adalah nol, perbandingan rasio tidak valid."

    ratio_pm25 = (pm25_stagnant / pm25_normal)
    
    return ratio_pm25, None

# =========================================================
#           MEMBACA DATA DAN AGREGASI STATIS
# =========================================================

@st.cache_data
def load_data():
    """Memuat data, melakukan Feature Engineering, dan menghitung metrik statis yang di-cache."""
    try:
        # --- Path & Load CSV ---
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'main_data.csv')

        df_full = pd.read_csv(
            file_path,
            parse_dates=True,
            infer_datetime_format=True
        )

        # --- Parsing kolom datetime ---
        if 'datetime' in df_full.columns:
            df_full['datetime'] = pd.to_datetime(df_full['datetime'], errors='coerce')
            df_full.set_index('datetime', inplace=True)
        else:
            try:
                df_full.index = pd.to_datetime(df_full.index, errors='coerce')
            except Exception as e:
                raise ValueError(f"Gagal mengubah index menjadi datetime: {e}")

        if not isinstance(df_full.index, pd.DatetimeIndex):
            raise TypeError("Index df_full bukan DatetimeIndex setelah parsing.")

        # --- Feature Engineering Kritis ---
        if 'year' not in df_full.columns:
            df_full['year'] = df_full.index.year
        df_full['Is_Stagnant'] = (df_full['WSPM'] < 3.2) # Stagnasi untuk PB 4
        df_full['Area_Type'] = df_full['Area_Type'].astype('category')
        df_full['Season'] = df_full['Season'].astype('category')
        
        pollutants_caap = ['PM2.5', 'NO2', 'SO2']

        # PB 4: Stagnasi Udara (Overall) - HANYA METRIK GLOBAL AWAL
        stagnation_analysis_overall = df_full.groupby('Is_Stagnant')[['PM2.5', 'WSPM']].mean()
        if False in stagnation_analysis_overall.index and stagnation_analysis_overall.loc[False, 'PM2.5'] != 0:
             ratio_pm25_overall = stagnation_analysis_overall.loc[True, 'PM2.5'] / stagnation_analysis_overall.loc[False, 'PM2.5']
        else:
             ratio_pm25_overall = np.nan


        # --- Kompilasi Semua Hasil ---
        metrics = {
            'df_full': df_full, # Ini adalah df_full_raw
            'pb4_stagnation': stagnation_analysis_overall,
            'pb4_ratio': ratio_pm25_overall,
        }
        return metrics

    except FileNotFoundError:
        st.error("Error: Pastikan file data (main_data.csv) ada di folder yang benar.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memuat data: {e}")
        st.stop()


# =========================================================
#           EKSTRAK HASIL DAN KONFIGURASI APLIKASI
# =========================================================
metrics = load_data()
df_full_raw = metrics['df_full'] 
stagnation_analysis_overall = metrics['pb4_stagnation']
ratio_pm25_overall = metrics['pb4_ratio']

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Analisis Kualitas Udara Beijing (2013-2017)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analisis Kualitas Udara Beijing (2013-2017): Evaluasi Dampak Clean Air Action Plan (CAAP) dan Tantangan Diagnostik")
st.caption("Proyek Analisis Data oleh Gina Melinia")

# =========================================================
#                   GLOBAL FILTER (AREA)
# =========================================================
st.sidebar.header("Navigasi Analisis")
analysis_page = st.sidebar.radio(
    "Pilih Pertanyaan Bisnis (PB):",
    (
        "1. Tren Tahunan Polutan (PB 1)",
        "2. Evaluasi Dampak CAAP (PB 2)",
        "3. Dinamika Ozon vs. NO₂ (PB 3)",
        "4. Peran Stagnasi Udara (PB 4)",
    ),
)
st.sidebar.markdown("---")
st.sidebar.header("Filter Global")

#  Filter Global Area
area_options = ['Overall'] + list(df_full_raw['Area_Type'].unique())
selected_area_global = st.sidebar.selectbox(
    "Filter Berdasarkan Tipe Area:",
    options=area_options,
    key='filter_area_global',
    help="Memfilter semua visualisasi berdasarkan Urban, Suburban, Rural, atau Keseluruhan (Overall).",
)

# Menerapkan Filter Area Global
df_full_filtered = df_full_raw.copy()
if selected_area_global != 'Overall':
    df_full_filtered = df_full_filtered[df_full_filtered['Area_Type'] == selected_area_global]

# Mengganti nama variabel df_full di seluruh kode menjadi df_full_filtered untuk konsistensi
df_full = df_full_filtered 


# =========================================================
#   PERHITUNGAN DINAMIS PASCA-FILTER GLOBAL (PB 1, 2, 3)
# =========================================================

# PB 1 & 2: Tren & Perubahan Tahunan
pollutants_caap = ['PM2.5', 'NO2', 'SO2']

# Rata-rata Pasca-CAAP (Pasca-September 2013)
df_post_caap_annual = (
    df_full[df_full['Pre_CAAP'] == False].groupby('year')[pollutants_caap].mean()
)
# Baseline Pra-CAAP (Hingga September 2013)
df_pre_caap_baseline = (
    df_full[df_full['Pre_CAAP'] == True][pollutants_caap].mean().to_frame().T
)
df_pre_caap_baseline.index = [2013]
df_viz_pb1 = pd.concat([df_pre_caap_baseline, df_post_caap_annual])
df_viz_pb1.index.name = 'year'
baseline_values = df_pre_caap_baseline.iloc[0]

# Perubahan Persentase Tahunan (PB 2)
df_annual_change = ((df_post_caap_annual - baseline_values) / baseline_values) * 100
df_annual_change.index.name = 'year'

# PB 3: Korelasi O3 vs NO2 Musim Panas
df_summer = df_full[df_full['Season'] == 'Summer'].copy()
ozone_nitro_corr = df_summer.groupby('year')[['O3', 'NO2']].corr().unstack()
ozone_nitro_corr_summer = ozone_nitro_corr[('O3', 'NO2')]
ozone_nitro_corr_summer.name = 'O3_NO2_Correlation_Summer'


# =========================================================
#                  STRUKTUR HALAMAN KONTEN
# =========================================================

# ---------------------------------------------------------
# PB 1: Tren Tahunan Polutan
# ---------------------------------------------------------
if analysis_page == "1. Tren Tahunan Polutan (PB 1)":
    st.header(f"1. Tren Konsentrasi Polutan Utama di Area: {selected_area_global}")
    st.info("Visualisasi tren polutan utama dari tahun ke tahun. Interaktif berdasarkan musim.")

    # Interaktivitas Filter
    col_filter_1, _ = st.columns(2)
    with col_filter_1:
        selected_season = st.selectbox(
            "Filter Berdasarkan Musim:",
            options=['Overall'] + list(df_full['Season'].unique()),
            help="Memfilter tren hanya untuk musim tertentu (e.g., Winter).",
        )

    # Logika Pemfilteran Dinamis
    df_filtered_pb1 = df_full.copy()
    if selected_season != 'Overall':
        df_filtered_pb1 = df_filtered_pb1[df_filtered_pb1['Season'] == selected_season]

    # Visualisasi & Metrik
    col_viz_1, col_viz_2 = st.columns([3, 1])
    with col_viz_1:
        st.subheader("Tren Polutan Gabungan")
        fig_pb1 = plot_pb1_combined_dynamic(df_filtered_pb1)
        st.pyplot(fig_pb1)

    with col_viz_2:
        st.subheader("Rata-Rata Terfilter")
        pm25_mean = df_filtered_pb1['PM2.5'].mean()
        no2_mean = df_filtered_pb1['NO2'].mean() 
        so2_mean = df_filtered_pb1['SO2'].mean()
        
        # Metrik PM2.5
        st.metric(label=f"PM2.5 Rata-Rata ({selected_area_global})", value=f"{pm25_mean:.2f} µg/m³")
        # Metrik NO2 
        st.metric(label=f"NO₂ Rata-Rata ({selected_area_global})", value=f"{no2_mean:.2f} µg/m³") 
        # Metrik SO2
        st.metric(label=f"SO₂ Rata-Rata ({selected_area_global})", value=f"{so2_mean:.2f} µg/m³")
        
        st.markdown("""
        **Interpretasi Kunci:** Tren menunjukkan keberhasilan penekanan **SO2**. Namun, **PM2.5** menunjukkan lonjakan pada 2017, menandakan tantangan berkelanjutan.
        """)


# ---------------------------------------------------------
# PB 2: Evaluasi Dampak CAAP
# ---------------------------------------------------------
elif analysis_page == "2. Evaluasi Dampak CAAP (PB 2)":
    st.header(f"2. Evaluasi Dampak Pra- vs. Pasca-CAAP di Area: {selected_area_global}")
    st.info("Perbandingan persentase perubahan rata-rata polutan Pasca-CAAP terhadap Baseline Pra-CAAP (hingga September 2013). Hasil ini sensitif terhadap filter Area Global.")

    # Interaktivitas: Polutan Filter
    pollutants_list = ['PM2.5', 'NO2', 'SO2']
    selected_pollutants = st.multiselect(
        "Pilih Polutan yang Ditampilkan:",
        options=pollutants_list,
        default=pollutants_list,
        help="Pilih polutan untuk melihat persentase perubahan tahunan terhadap Baseline Pra-CAAP."
    )
    
    # Memfilter DataFrame yang sudah dihitung dinamis
    df_filtered_change = df_annual_change[selected_pollutants] if selected_pollutants else df_annual_change.iloc[:,0:0]

    # --- BARIS 1: VISUALISASI GRAFIK ---
    st.subheader("Bar Chart Perubahan Persentase Tahunan (vs. Baseline Pra-CAAP)")
    
    if selected_pollutants:
        fig_pb2, error_msg = plot_pb2(df_filtered_change)
        if fig_pb2:
            st.pyplot(fig_pb2)
        else:
            st.warning(error_msg)
    else:
        st.warning("Silakan pilih minimal satu polutan untuk melihat Bar Chart.")

    st.markdown("---")
    
    # --- BARIS 2: RINGKASAN METRIK (3 KOLOM) ---
    st.subheader("Dampak Kunci Pasca-CAAP (Persentase Perubahan vs. Baseline)")
    
    col_metric_no2, col_metric_so2, col_metric_pm25 = st.columns(3)

    # Metric NO2
    with col_metric_no2:
        try:
            # NO2 Turun Terbesar (Cari nilai paling negatif, abaikan 2017)
            no2_drop = df_annual_change[df_annual_change.index != 2017]['NO2'].min()
            year_no2_drop = df_annual_change[df_annual_change.index != 2017]['NO2'].idxmin()
            st.metric(
                label=f"NO₂ Turun Terbesar ({year_no2_drop})", 
                value=f"{no2_drop:.1f}%", 
                delta="Penekanan Emisi Transportasi", 
                delta_color="normal"
            )
        except Exception: st.metric(label="NO₂ Turun Terbesar", value="N/A")

    # Metric SO2
    with col_metric_so2:
        try:
            so2_2016 = df_annual_change.loc[2016, 'SO2']
            st.metric(label="SO₂ Turun Terbesar (2016)", value=f"{so2_2016:.1f}%", delta="Target Batubara", delta_color="normal")
        except KeyError: st.metric(label="SO₂ Turun Terbesar (2016)", value="N/A")

    # Metric PM2.5
    with col_metric_pm25:
        try:
            pm25_2017 = df_annual_change.loc[2017, 'PM2.5']
            st.metric(label="PM2.5 Lonjakan (2017)", value=f"{pm25_2017:.1f}%", delta="Kemunduran", delta_color="inverse")
        except KeyError: st.metric(label="PM2.5 Lonjakan (2017)", value="N/A")


# ---------------------------------------------------------
# PB 3: Dinamika Ozon vs. NO₂
# ---------------------------------------------------------
elif analysis_page == "3. Dinamika Ozon vs. NO₂ (PB 3)":
    st.header(f"3. Dinamika Hubungan Ozon vs. NO₂ di Musim Panas (Area: {selected_area_global})")
    st.info("Mendiagnosis potensi Ozone Paradox dengan menganalisis korelasi tahunan dan distribusi spasial.")
    
    
    # === BARIS 1: TREN KORELASI ===
    st.subheader("A. Tren Korelasi Tahunan (Dinamika)")
    
    # Plot Korelasi (menggunakan data yang sudah difilter Area Global)
    fig_corr = plot_pb3_correlation_trend(ozone_nitro_corr_summer)
    st.pyplot(fig_corr)
    
    # Metrik Korelasi Kunci
    r_2014 = ozone_nitro_corr_summer.get(2014, np.nan)
    st.metric(label="Korelasi Terkuat Negatif (2014)", value=f"r = {r_2014:.3f}", delta="Hubungan Titrasi NOₓ terkuat", delta_color="off")
    
    st.markdown("---") # Separator visual
    
    # === BARIS 2: BOX PLOT DISTRIBUSI O3 ===
    st.subheader("B. Distribusi $\text{O}_3$ Berdasarkan Tipe Area (Diagnosis)")
    
    # Menambahkan Filter Lokal Tahun untuk Box Plot
    available_years_raw = sorted(df_full_raw['year'].unique().astype(str).tolist())
    col_filter_bp, _ = st.columns([1, 3])
    with col_filter_bp:
        selected_year_pb3_boxplot = st.selectbox(
            "Filter Tahun untuk Box Plot O₃:",
            options=['Overall'] + available_years_raw,
            key='filter_year_pb3_boxplot',
            help="Memfilter Box Plot O₃ berdasarkan tahun. (Menggunakan data Area keseluruhan/RAW)."
        )
    
    # Memfilter data untuk Box Plot (Menggunakan data RAW/UNFILTERED Area, tetapi difilter Tahun)
    df_filtered_boxplot = df_full_raw.copy()
    if selected_year_pb3_boxplot != 'Overall':
        df_filtered_boxplot = df_filtered_boxplot[df_filtered_boxplot['year'].astype(str) == selected_year_pb3_boxplot]

    # Plot Box Plot
    fig_boxplot = plot_pb3_boxplot(df_filtered_boxplot)
    st.pyplot(fig_boxplot)
    
    # Menghitung Rata-rata O3 untuk Metrik (Menggunakan data box plot yang sudah difilter tahun)
    df_summer_mean = df_filtered_boxplot[df_filtered_boxplot['Season'] == 'Summer'].groupby('Area_Type')['O3'].mean()
    
    col_metric_bp_1, col_metric_bp_2, _ = st.columns(3)
    with col_metric_bp_1:
        st.metric(label="O₃ Rata-rata di Rural (Tertinggi)", value=f"{df_summer_mean.get('Rural', np.nan):.2f} µg/m³")
    with col_metric_bp_2:
        st.metric(label="O₃ Rata-rata di Urban (Tertekan)", value=f"{df_summer_mean.get('Urban', np.nan):.2f} µg/m³")

    st.markdown("---")
    with st.expander("Diagnosis: Apakah terjadi Ozone Paradox?", expanded=False):
        st.markdown(
            """
            - Korelasi O3 dan NO2 yang konsisten negatif menunjukkan bahwa mekanisme titrasi NOx masih dominan di musim panas (Summer) Beijing.
            - Namun, pergerakan koefisien korelasi menuju nol setelah 2014, ditambah dengan fakta bahwa konsentrasi median O3 tertinggi ditemukan di area Suburban/Rular (sebagai hasil transpor), memberikan indikasi kuat adanya pergeseran rezim kimia yang dapat memicu Ozone Paradox.
            - Untuk mengkonfirmasi adanya Ozone Paradox, analisis harus diperluas untuk mengukur rezim sensitifitas ozon (NOx-limited vs VOC-limited), yang berada di luar lingkup analisis ini (tidak ada data terkait VOC di dataset).
            """
        )

# ---------------------------------------------------------
# PB 4: Peran Stagnasi Udara
# ---------------------------------------------------------
elif analysis_page == "4. Peran Stagnasi Udara (PB 4)":
    st.header(f"4. Pengaruh Dinamika Atmosfer pada PM2.5 di Area: {selected_area_global}")
    st.info("Analisis peran Stagnasi Udara dan pola angin terhadap tingkat polusi ekstrem. Hasil ini sensitif terhadap filter Area Global.")
    
    # Interaktivitas: Filter TAHUN dan MUSIM 
    col_filter_4b, col_filter_4c = st.columns(2)
    
    with col_filter_4b:
        available_years = sorted(df_full['year'].unique().astype(str).tolist())
        selected_year_pb4 = st.selectbox(
            "Filter Berdasarkan Tahun:",
            options=['Overall'] + available_years,
            key='filter_year_pb4',
            help="Memfilter analisis pada tahun tertentu."
        )
    with col_filter_4c: 
        available_seasons = sorted(df_full['Season'].unique().tolist())
        selected_season_pb4 = st.selectbox(
            "Filter Berdasarkan Musim:",
            options=['Overall'] + available_seasons,
            key='filter_season_pb4',
            help="Memfilter Wind Rose pada musim tertentu."
        )

    # Logika Pemfilteran Dinamis
    df_filtered_pb4 = df_full.copy()
    
    if selected_year_pb4 != 'Overall':
        df_filtered_pb4 = df_filtered_pb4[df_filtered_pb4['year'].astype(str) == selected_year_pb4]
    if selected_season_pb4 != 'Overall':
        df_filtered_pb4 = df_filtered_pb4[df_filtered_pb4['Season'] == selected_season_pb4]

    # Perhitungan Dampak Stagnasi Dinamis
    ratio_pm25_dynamic, error_ratio = calculate_pb4_impact(df_filtered_pb4)

    # --- Layout Bagian 1: Box Plot Stagnasi ---
    st.subheader("A. Perbandingan **PM2.5** vs. Stagnasi")

    col_viz_1 = st.columns(1)[0] # Ambil 1 kolom penuh
    with col_viz_1:
        fig_boxplot_pb4, error_boxplot_pb4 = plot_pb4_boxplot_stagnation(df_filtered_pb4)
        
        if error_boxplot_pb4:
            st.warning(error_boxplot_pb4)
        elif fig_boxplot_pb4:
            st.pyplot(fig_boxplot_pb4)

    st.markdown("---") # Garis pemisah visual

    # --- Layout Bagian 2: Wind Rose Normal vs. Ekstrem ---
    st.subheader("B. Analisis Pola Angin Berdasarkan Kondisi **PM2.5**")
    col_viz_2, col_viz_3 = st.columns(2) # Dua kolom terpisah untuk Wind Rose

    filter_title = f"Area: {selected_area_global} | Tahun: {selected_year_pb4} | Musim: {selected_season_pb4}"

    # Wind Rose Kiri: Normal
    with col_viz_2:
        fig_normal, error_normal = plot_windrose_single_condition(
            df_filtered_pb4, 
            'Normal', # Kondisi Normal
            filter_title
        )
        if error_normal:
            st.warning(f"Normal Wind Rose Error: {error_normal}")
        elif fig_normal:
            st.pyplot(fig_normal)

    # Wind Rose Kanan: Ekstrem
    with col_viz_3:
        fig_extreme, error_extreme = plot_windrose_single_condition(
            df_filtered_pb4, 
            'Extreme', # Kondisi Ekstrem
            filter_title
        )
        if error_extreme:
            st.warning(f"Ekstrem Wind Rose Error: {error_extreme}")
        elif fig_extreme:
            st.pyplot(fig_extreme)
    
    st.markdown("---")
    
    # Metrik Kunci
    if ratio_pm25_dynamic and not np.isnan(ratio_pm25_dynamic):
        increase_percent = (ratio_pm25_dynamic - 1) * 100
        
        st.markdown(f"""
        ### 📊 Dampak Kunci Stagnasi di **{selected_area_global}**
        """)
        
        st.metric(
            label="Kenaikan Rata-Rata PM2.5 Saat Stagnan",
            value=f"{increase_percent:.1f}%",
            delta="Dibandingkan saat Angin Normal",
            delta_color="normal"
        )
        st.markdown(f"""
        <p>Ketika kondisi udara stagnan (Kecepatan Angin < 3.2 m/s), konsentrasi PM2.5 rata-rata meningkat {increase_percent:.1f}% dibandingkan saat angin bergerak normal. Angin yang berasal dari 
        Timur Laut (N-E) mendominasi pada saat kondisi ekstrem.</p>
        """, unsafe_allow_html=True)
    else:
        st.warning("Metrik Dampak Stagnasi tidak dapat dihitung karena data yang difilter terlalu sedikit atau tidak seimbang.")