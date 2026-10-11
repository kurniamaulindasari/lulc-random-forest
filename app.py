from pathlib import Path
import io
import base64
import tempfile
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import folium
import geopandas as gpd
from PIL import Image
from folium.raster_layers import ImageOverlay
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ============================================================
# GeoLULC Jawa Timur — menggunakan hasil notebook yang disertakan
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="GeoLULC | Tutupan Lahan Jawa Timur",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kode kelas pada raster hasil klasifikasi enam kelas
KELAS_RASTER = {
    0: "Sawah",
    1: "Bangunan",
    2: "Hutan",
    3: "Mangrove",
    4: "Danau",
    5: "Lautan",
}
WARNA = {
    "Sawah": "#90EE90",
    "Bangunan": "#FF0000",
    "Hutan": "#008000",
    "Mangrove": "#005000",
    "Danau": "#ADD8E6",
    "Lautan": "#00008B",
}
FITUR = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI"]

st.markdown("""
<style>
:root { --ink:#172B3A; --muted:#637789; --teal:#087E8B; }
.stApp { background:#F4F7FA; color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#15344B 0%,#116B69 100%); }
[data-testid="stSidebar"] * { color:#F4FBFC !important; }
.hero { padding:1.35rem 1.6rem; border-radius:16px; color:white;
        background:linear-gradient(110deg,#17324D,#087E8B 65%,#26A69A); margin-bottom:1rem; }
.hero h1 { color:white; margin:0; font-size:2rem; }
.hero p { color:#E5F8F5; margin:.4rem 0 0 0; }
.metric-card { background:white; border:1px solid #DCE6ED; border-top:4px solid #159A9C;
               border-radius:12px; padding:.85rem 1rem; min-height:110px; }
.metric-label { color:#647789; font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; font-weight:800; }
.metric-value { color:#183044; font-size:1.65rem; font-weight:850; margin:.2rem 0; }
.metric-help { color:#647789; font-size:.78rem; }
.note { padding:.8rem 1rem; border-left:4px solid #159A9C; border-radius:7px; background:#E8F8F5; color:#254B59; }
.section-card { background:white; border:1px solid #DCE6ED; border-radius:12px; padding:.8rem 1rem; margin:.35rem 0; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, help_text="", accent="#159A9C"):
    st.markdown(
        f'<div class="metric-card" style="border-top-color:{accent}">'
        f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div>'
        f'<div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )

def data_path(filename):
    p = DATA_DIR / filename
    return p if p.exists() else None

def load_raster_file(path_or_uploaded):
    if isinstance(path_or_uploaded, (str, Path)):
        with rasterio.open(path_or_uploaded) as src:
            arr = src.read()
            profile = src.profile.copy()
            bounds = src.bounds
            crs = src.crs
            nodata = src.nodata
        return arr, profile, bounds, crs, nodata
    raw = path_or_uploaded.getvalue()
    with MemoryFile(raw) as mem:
        with mem.open() as src:
            arr = src.read()
            profile = src.profile.copy()
            bounds = src.bounds
            crs = src.crs
            nodata = src.nodata
    return arr, profile, bounds, crs, nodata

def raster_png_bytes(classes, max_side=1400):
    arr = np.asarray(classes)
    step = max(1, int(np.ceil(max(arr.shape) / max_side)))
    arr = arr[::step, ::step]
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    for label, name in KELAS_RASTER.items():
        rgb = tuple(int(WARNA[name][i:i+2], 16) for i in (1, 3, 5))
        rgba[arr == label, :3] = rgb
        rgba[arr == label, 3] = 220
    rgba[~np.isin(arr, list(KELAS_RASTER)), 3] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()

def make_raster_map(path_or_uploaded, opacity=0.78):
    arr, profile, bounds, crs, nodata = load_raster_file(path_or_uploaded)
    classes = arr[0]
    if crs and str(crs) != "EPSG:4326":
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21
        )
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    png = raster_png_bytes(classes)
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    center = [(south+north)/2, (west+east)/2]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    ImageOverlay(
        image=uri, bounds=[[south, west], [north, east]], opacity=opacity,
        name="Hasil klasifikasi Random Forest", interactive=True, cross_origin=False, zindex=2
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;border:1px solid #bbb;border-radius:7px;padding:10px 12px;font-size:12px;box-shadow:0 1px 5px #999"><b>Kelas Tutupan Lahan</b>'
    for label, name in KELAS_RASTER.items():
        legend += f'<div style="margin-top:3px"><span style="display:inline-block;width:11px;height:11px;background:{WARNA[name]};margin-right:6px"></span>{label}: {name}</div>'
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))
    return m, classes, profile, str(crs) if crs else "CRS tidak tersedia"

def read_vector_upload(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / uploaded.name
        p.write_bytes(uploaded.getvalue())
        if suffix == ".zip":
            gdf = gpd.read_file("zip://" + str(p))
        else:
            gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")

def inferred_class(filename):
    name = filename.lower()
    for key, val in [("sawah","Sawah"),("bangunan","Bangunan"),("hutan","Hutan"),
                     ("lahanhijau","Hutan"),("mangrove","Mangrove"),("danau","Danau"),
                     ("lautan","Lautan"),("perairan","Lautan")]:
        if key in name:
            return val
    return None

def add_vector_layer(m, gdf, layer_name, class_name=None):
    if gdf.empty:
        return
    cols = [c for c in gdf.columns if c != "geometry"]
    class_col = next((c for c in ["kelas","Kelas","label_teks","Nama Kelas","nama_kelas","class"] if c in gdf.columns), None)
    if class_col:
        def style(feature):
            value = str(feature["properties"].get(class_col, ""))
            match = next((v for v in KELAS_RASTER.values() if v.lower() == value.lower()), None)
            color = WARNA.get(match, "#1976D2")
            return {"color": color, "fillColor": color, "weight": 1.5, "fillOpacity": .35}
        folium.GeoJson(gdf, name=layer_name, style_function=style,
                       tooltip=folium.GeoJsonTooltip(fields=[class_col], aliases=["Kelas:"])).add_to(m)
    else:
        color = WARNA.get(class_name, "#1976D2")
        tooltip = folium.GeoJsonTooltip(fields=cols[:3]) if cols else None
        folium.GeoJson(gdf, name=layer_name, style_function=lambda f: {
            "color":color,"fillColor":color,"weight":1.5,"fillOpacity":.35
        }, tooltip=tooltip).add_to(m)

def load_test_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        p = data_path("hasil_testing.csv")
        if p is None:
            return None
        df = pd.read_csv(p)
    if not {"Label Aktual", "Label Prediksi"}.issubset(df.columns):
        return None
    df = df.copy()
    df["Label Aktual"] = pd.to_numeric(df["Label Aktual"], errors="coerce")
    df["Label Prediksi"] = pd.to_numeric(df["Label Prediksi"], errors="coerce")
    df = df.dropna(subset=["Label Aktual", "Label Prediksi"])
    df["Label Aktual"] = df["Label Aktual"].astype(int)
    df["Label Prediksi"] = df["Label Prediksi"].astype(int)
    return df

# Sidebar
with st.sidebar:
    st.markdown("## 🛰️ GeoLULC")
    st.caption("Dashboard hasil notebook")
    st.markdown("---")
    st.markdown("**Algoritma**  \nRandom Forest")
    st.markdown("**Citra**  \nSentinel-2 L2A")
    st.markdown("**Wilayah**  \nProvinsi Jawa Timur")
    st.markdown("**Fitur**  \nB02, B03, B04, B08, B11, NDVI, NDWI")
    st.markdown("---")
    page = st.radio("Navigasi", ["Ringkasan", "Peta & Sampel", "Evaluasi Model", "Prediksi GeoTIFF"])
    st.markdown("---")
    st.caption("Data hasil notebook dibaca otomatis dari folder data/.")

st.markdown("""
<div class="hero">
  <div style="font-size:.72rem;letter-spacing:.12em;font-weight:800;opacity:.85">REMOTE SENSING • MACHINE LEARNING • JAWA TIMUR</div>
  <h1>🛰️ Dashboard Tutupan Lahan Jawa Timur</h1>
  <p>Klasifikasi tutupan lahan menggunakan Sentinel-2 L2A dan Random Forest.</p>
</div>
""", unsafe_allow_html=True)

# Load test results automatically
df_test = load_test_data()
if df_test is not None and len(df_test):
    y_true = df_test["Label Aktual"].to_numpy()
    y_pred = df_test["Label Prediksi"].to_numpy()
    accuracy = accuracy_score(y_true, y_pred)
    f1macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
else:
    y_true = y_pred = None
    accuracy = f1macro = None

if page == "Ringkasan":
    st.markdown("### Ringkasan hasil penelitian")
    cols = st.columns(4)
    with cols[0]: metric_card("Algoritma", "Random Forest", "Model klasifikasi")
    with cols[1]: metric_card("Jumlah kelas raster", "6", "Sawah hingga lautan")
    with cols[2]: metric_card("Fitur", "7", "5 band + NDVI + NDWI")
    with cols[3]:
        if accuracy is not None:
            metric_card("Akurasi data uji", f"{accuracy*100:.2f}%", f"{len(df_test)} baris pada CSV", "#E9A23B")
        else:
            metric_card("Akurasi data uji", "Belum tersedia", "CSV hasil pengujian belum ditemukan", "#E9A23B")
    st.markdown('<div class="note">Peta, gambar klasifikasi, statistik, dan evaluasi berikut diambil dari file hasil notebook yang disertakan pada folder data. Angka evaluasi mengikuti CSV uji; kelas pada CSV uji tidak otomatis dianggap sama dengan enam kelas pada raster.</div>', unsafe_allow_html=True)
    st.markdown("### Gambar hasil klasifikasi dari notebook")
    preview = data_path("peta_klasifikasi_lulc.png")
    if preview:
        st.image(str(preview), caption="Peta klasifikasi tutupan lahan — Random Forest (hasil notebook)", use_container_width=True)
    else:
        st.warning("File peta_klasifikasi_lulc.png tidak ditemukan di folder data.")
    st.markdown("### Statistik enam kelas pada raster")
    stats_path = data_path("statistik_klasifikasi.csv")
    if stats_path:
        stats = pd.read_csv(stats_path)
        if "Kelas" in stats.columns and "Jumlah Piksel" in stats.columns:
            a,b = st.columns([1.05,1])
            with a: st.dataframe(stats, use_container_width=True, hide_index=True)
            with b:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(stats["Kelas"], stats["Persentase (%)"], color=[WARNA.get(n,"#94A3B8") for n in stats["Kelas"]])
                ax.set_xlabel("Persentase piksel (%)")
                ax.set_title("Distribusi Kelas Tutupan Lahan")
                ax.invert_yaxis()
                ax.spines[["top","right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    st.markdown("### File data yang dimuat otomatis")
    for filename in ["hasil_klasifikasi_lulc.tif","peta_klasifikasi_lulc.png","statistik_klasifikasi.csv",
                     "hasil_testing.csv","classification_report.csv","confusion_matrix.csv","model_random_forest.pkl","mapping_kelas.pkl"]:
        p = data_path(filename)
        st.write(("✅ " if p else "⚠️ ") + filename)

elif page == "Peta & Sampel":
    st.markdown("### 1. Evaluasi Poligon Sampel & Prediksi Data Uji")
    st.caption("Peta dasar menampilkan Jawa Timur. Jika memiliki file GeoJSON/ZIP shapefile sampel, tambahkan di bawah ini.")
    vector_files = st.file_uploader("Unggah sampel vektor (GeoJSON/JSON/ZIP)", type=["geojson","json","zip"], accept_multiple_files=True)
    m_samples = folium.Map(location=[-7.5,112.5], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    total_features = 0
    if vector_files:
        for vf in vector_files:
            try:
                gdf = read_vector_upload(vf)
                add_vector_layer(m_samples, gdf, vf.name, inferred_class(vf.name))
                total_features += len(gdf)
            except Exception as e:
                st.warning(f"{vf.name}: {e}")
    folium.LayerControl(collapsed=False).add_to(m_samples)
    st_folium(m_samples, use_container_width=True, height=430, returned_objects=[])
    if vector_files:
        st.caption(f"Jumlah geometri vektor yang dimuat: {total_features:,}")
    else:
        st.info("Belum ada data sampel vektor di folder data. Peta dasar tetap ditampilkan; unggah GeoJSON untuk menampilkan poligon sampel.")

    st.markdown("---")
    st.markdown("### 2. Klasifikasi Tutupan Lahan Skala Regional Jawa Timur")
    default_tif = data_path("hasil_klasifikasi_lulc.tif")
    uploaded_tif = st.file_uploader("Ganti raster klasifikasi (opsional)", type=["tif","tiff"], key="replace_tif")
    tif_source = uploaded_tif if uploaded_tif is not None else default_tif
    opacity = st.slider("Opasitas overlay", 0.2, 1.0, 0.78, 0.05)
    if tif_source:
        try:
            m, classes, profile, crs = make_raster_map(tif_source, opacity)
            st_folium(m, use_container_width=True, height=500, returned_objects=[])
            st.caption(f"Raster: {profile.get('width')} × {profile.get('height')} piksel · CRS: {crs}")
            stat_path = data_path("statistik_klasifikasi.csv")
            if stat_path:
                st.markdown("#### Statistik dari notebook")
                st.dataframe(pd.read_csv(stat_path), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Raster tidak dapat ditampilkan: {e}")
    else:
        st.error("hasil_klasifikasi_lulc.tif tidak ditemukan di folder data.")

elif page == "Evaluasi Model":
    st.markdown("### Evaluasi Kinerja Model & Kontribusi Fitur Spektral")
    uploaded_csv = st.file_uploader("Unggah CSV lain untuk mengganti hasil (opsional)", type=["csv"], key="replace_test")
    df = load_test_data(uploaded_csv) if uploaded_csv is not None else df_test
    if df is None or df.empty:
        st.warning("File hasil_testing.csv tidak ditemukan atau kolom Label Aktual/Label Prediksi tidak tersedia.")
    else:
        y_true = df["Label Aktual"].to_numpy()
        y_pred = df["Label Prediksi"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        f1macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        correct = int((y_true == y_pred).sum())
        wrong = len(df) - correct
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Akurasi", f"{acc*100:.2f}%", f"{correct} benar dari {len(df)}")
        with c2: metric_card("F1-macro", f"{f1macro:.3f}", "Rata-rata F1 tiap label")
        with c3: metric_card("Data uji", f"{len(df):,}", "Baris valid dalam CSV")
        with c4: metric_card("Prediksi keliru", f"{wrong}", f"{wrong/max(1,len(df))*100:.1f}% dari data uji", "#E76F51")
        # Preserve class names as recorded in the test CSV when available.
        name_lookup = {}
        if "Nama Aktual" in df.columns:
            for label, name in zip(df["Label Aktual"], df["Nama Aktual"]):
                name_lookup.setdefault(int(label), str(name))
        if "Nama Prediksi" in df.columns:
            for label, name in zip(df["Label Prediksi"], df["Nama Prediksi"]):
                name_lookup.setdefault(int(label), str(name))
        labels = sorted(set(y_true) | set(y_pred))
        names = [name_lookup.get(int(i), str(i)) for i in labels]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        left,right = st.columns([1,1.15])
        with left:
            st.markdown("#### Confusion Matrix — Random Forest")
            fig, ax = plt.subplots(figsize=(6.5,5.2))
            im = ax.imshow(cm, cmap="GnBu")
            ax.set_xticks(range(len(labels)), names, rotation=35, ha="right")
            ax.set_yticks(range(len(labels)), names)
            ax.set_xlabel("Label Prediksi")
            ax.set_ylabel("Label Aktual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                            color="white" if cm[i,j] > cm.max()*.55 else "#172B3A", fontsize=10)
            ax.set_title(f"Confusion Matrix ({len(df)} data uji)", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("#### Kinerja klasifikasi per kelas")
            report = classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T
            per_class = report_df.reindex(names)[["precision","recall","f1-score"]].rename(columns={
                "precision":"Precision","recall":"Recall","f1-score":"F1-score"})
            fig, ax = plt.subplots(figsize=(6.8,5.2))
            y = np.arange(len(per_class))
            height = .23
            palette = ["#16B6A6","#65D9C9","#087E75"]
            for offset, col, color in [(-height,"Precision",palette[0]),(0,"Recall",palette[1]),(height,"F1-score",palette[2])]:
                bars = ax.barh(y+offset, per_class[col]*100, height, label=col, color=color)
                ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
            ax.set_yticks(y, per_class.index)
            ax.invert_yaxis()
            ax.set_xlim(0,115)
            ax.set_xlabel("Skor (%)")
            ax.grid(axis="x", linestyle="--", alpha=.25)
            ax.set_axisbelow(True)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.legend(ncol=3, loc="lower right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown("#### Classification report")
        st.dataframe(report_df.round(3), use_container_width=True)
        st.download_button("Unduh classification report CSV", report_df.to_csv().encode("utf-8-sig"),
                           "classification_report_streamlit.csv", "text/csv")

        model_path = data_path("model_random_forest.pkl")
        if model_path:
            st.markdown("#### Kontribusi fitur spektral")
            try:
                import joblib
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = joblib.load(model_path)
                if hasattr(model, "feature_importances_"):
                    feat_names = list(getattr(model, "feature_names_in_", FITUR))
                    imp = pd.DataFrame({"Fitur":feat_names,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
                    fig, ax = plt.subplots(figsize=(7,3.5))
                    ax.barh(imp["Fitur"], imp["Importance"], color="#159A9C")
                    ax.set_title("Feature Importance — Random Forest")
                    ax.set_xlabel("Importance")
                    ax.spines[["top","right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig,use_container_width=True)
                    plt.close(fig)
            except Exception as e:
                st.info(f"Feature importance tidak bisa dibaca oleh versi scikit-learn saat ini: {e}")

elif page == "Prediksi GeoTIFF":
    st.markdown("### Prediksi citra Sentinel-2")
    st.info("Bagian ini untuk prediksi raster mentah 5-band baru. Untuk melihat hasil klasifikasi yang sudah ada, buka menu Peta & Sampel.")
    model_upload = st.file_uploader("Model Random Forest (.pkl/.joblib)", type=["pkl","joblib"])
    raw_upload = st.file_uploader("Citra Sentinel-2 mentah 5-band (.tif/.tiff)", type=["tif","tiff"])
    if st.button("Jalankan prediksi", type="primary", disabled=(model_upload is None or raw_upload is None)):
        try:
            import joblib
            model = joblib.load(io.BytesIO(model_upload.getvalue()))
            arr, profile, bounds, crs, nodata = load_raster_file(raw_upload)
            if arr.shape[0] < 5:
                st.error("Raster mentah harus memiliki minimal 5 band: B02, B03, B04, B08, B11.")
                st.stop()
            max_side = st.slider("Maksimum sisi raster prediksi", 500, 5000, 1500, 250)
            step = max(1, int(np.ceil(max(arr.shape[1],arr.shape[2])/max_side)))
            sample = arr[:5,::step,::step].astype("float32")
            b02,b03,b04,b08,b11 = sample
            ndvi = (b08-b04)/(b08+b04+1e-6)
            ndwi = (b03-b08)/(b03+b08+1e-6)
            cube = np.stack([b02,b03,b04,b08,b11,ndvi,ndwi],axis=-1)
            X = pd.DataFrame(cube.reshape(-1,7),columns=FITUR)
            good = np.isfinite(X).all(axis=1).to_numpy()
            if nodata is not None:
                good &= (sample.reshape(5,-1).T != nodata).all(axis=1)
            pred = np.full(len(X),255,dtype=np.uint8)
            if good.any():
                pred[good] = model.predict(X.loc[good]).astype(np.uint8)
            pred_map = pred.reshape(sample.shape[1],sample.shape[2])
            fig, ax = plt.subplots(figsize=(11,6))
            cmap = plt.matplotlib.colors.ListedColormap([WARNA[KELAS_RASTER[i]] for i in KELAS_RASTER])
            masked = np.ma.masked_where(~np.isin(pred_map,list(KELAS_RASTER)),pred_map)
            ax.imshow(masked,cmap=cmap,norm=plt.matplotlib.colors.BoundaryNorm(np.arange(-.5,6.5,1),cmap.N))
            ax.set_title("Prediksi tutupan lahan — Random Forest")
            ax.set_xlabel("Kolom piksel"); ax.set_ylabel("Baris piksel")
            ax.legend(handles=[plt.matplotlib.patches.Patch(facecolor=WARNA[n],label=n) for n in KELAS_RASTER.values()],
                      loc="upper left",bbox_to_anchor=(1.01,1))
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")

st.markdown("---")
st.caption("GeoLULC • Sentinel-2 L2A • Random Forest • Provinsi Jawa Timur")
