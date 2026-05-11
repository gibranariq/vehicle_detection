import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Detection (Jalan Tol)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e0e6f0 !important;
}

/* Main background */
.stApp {
    background: #080b14;
    color: #e0e6f0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7fa3;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #3b82f6;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Badge labels */
.label-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin: 3px;
}
.badge-bus   { background: #3f1010; color: #FF6B6B; border: 1px solid #FF6B6B44; }
.badge-car   { background: #0f3330; color: #4ECDC4; border: 1px solid #4ECDC444; }
.badge-truck { background: #3f3510; color: #FFE66D; border: 1px solid #FFE66D44; }

/* Comparison table */
.compare-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.compare-table th {
    background: #141c2e;
    color: #6b7fa3;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem;
    padding: 10px 14px;
    text-align: left;
}
.compare-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1a2035;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.compare-table tr.best-row td { background: #0d2040; color: #60a5fa; }
.compare-table tr:hover td { background: #111827; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1829 0%, #0a1628 50%, #0d1829 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #1d4ed844 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub {
    color: #6b7fa3;
    font-size: 0.95rem;
    margin-top: 8px;
}

/* Info box */
.info-box {
    background: #0d1829;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #b0bdd4;
    line-height: 1.7;
}

/* Count pill */
.count-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.88rem;
}
.pill-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0d1829;
    border: 2px dashed #1e2d4a;
    border-radius: 12px;
}

/* Progress */
.stProgress > div > div { background: #3b82f6 !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 500;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="best.pt"):
    return YOLO(path)

MODEL_PATH = "best.pt"
CLASS_COLORS = {"bus": "#FF6B6B", "car": "#4ECDC4", "truck": "#FFE66D"}
CLASS_ICONS  = {"bus": "🚌", "car": "🚗", "truck": "🚛"}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 Vehicle Detection")
    st.markdown("---")
    # model_path_input = st.text_input("Model path", value=MODEL_PATH)
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)
    st.markdown("---")
    st.markdown("**Classes:**")
    st.markdown("""
    <span class="label-badge badge-bus">🚌 Bus</span>
    <span class="label-badge badge-car">🚗 Car</span>
    <span class="label-badge badge-truck">🚛 Truck</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#6b7fa3;font-size:0.75rem;line-height:1.8">
    <b style="color:#b0bdd4">Model:</b> YOLOv11n<br>
    <b style="color:#b0bdd4">Epochs:</b> 100<br>
    <b style="color:#b0bdd4">Image Size:</b> 1024<br>
    <b style="color:#b0bdd4">Dataset:</b> 80 train / 20 val<br>
    <b style="color:#b0bdd4">Author:</b> Gibran Ariq N.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Overview & Dataset",
    "🔬  Model Comparison",
    "🎯  Run Detection",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Overview & Dataset
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Vehicle Detection<br>on Indonesian Toll Roads</div>
        <div class="hero-sub">YOLOv11n fine-tuned on CCTV imagery · 3 vehicle classes · Real-time inference</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Description
    st.markdown('<div class="section-header">Tentang Proyek</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Proyek ini melatih model <b>YOLOv11n</b> untuk mendeteksi kendaraan di jalan tol Indonesia
    menggunakan citra dari kamera CCTV. Dataset dikumpulkan dan dilabeli manual menggunakan
    <b>Label Studio</b>, kemudian dilakukan fine-tuning dari bobot pre-trained COCO.
    Model terbaik (<b>Experiment 2</b>) mampu mendeteksi kendaraan dengan confidence tinggi
    bahkan pada kondisi malam hari maupun cuaca buruk.
    </div>
    """, unsafe_allow_html=True)

    # col1, col2 = st.columns(2)

    # with col1:
        # ── Dataset Info
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#60a5fa">80</div>
            <div class="metric-label">Train Images</div>
        </div>""", unsafe_allow_html=True)
    with dcol2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#a78bfa">20</div>
            <div class="metric-label">Val Images</div>
        </div>""", unsafe_allow_html=True)
    with dcol3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#34d399">3</div>
            <div class="metric-label">Classes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📍 <b>Sumber data:</b> Rekaman kamera CCTV jalan tol Indonesia<br>
    🏷️ <b>Alat labeling:</b> Label Studio<br>
    🌙 <b>Kondisi:</b> Siang &amp; malam hari, berbagai cuaca<br>
    ⚠️ <b>Keterbatasan:</b> Data malam jauh lebih sedikit dari data siang,
    sehingga model lebih kesulahan pada kondisi gelap.
    </div>
    """, unsafe_allow_html=True)

    # with col2:
        # ── Class Distribution
    st.markdown('<div class="section-header">Distribusi Kelas </div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Berdasarkan karakteristik jalan tol, distribusi kelas pada dataset adalah sebagai berikut.
    <b>Car</b> mendominasi karena kendaraan pribadi paling sering melintas.
    </div>
    """, unsafe_allow_html=True)

    # Simple bar chart using Streamlit
    import pandas as pd
    dist_df = pd.DataFrame({
        "Kelas": ["🚌 Bus", "🚗 Car", "🚛 Truck"],
        "Jumlah Instance": [14, 404, 201],
    })
    
    color_map = {
        "🚌 Bus": "#FF6B6B",
        "🚗 Car": "#4ECDC4",
        "🚛 Truck": "#FFE66D",
    }

    import plotly.express as px

    fig = px.bar(
        dist_df,
        x="Jumlah Instance",
        y="Kelas",
        orientation='h',
        color="Kelas",
        color_discrete_map=color_map,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    🚗 <b>Car</b> adalah kelas yang paling banyak karena mendominasi lalu lintas tol.<br>
    🚛 <b>Truck</b> menempati posisi kedua.<br>
    🚌 <b>Bus</b> paling sedikit — menjadi class ke-0 karena dilabeli pertama kali di Label Studio.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Label Info
    st.markdown('<div class="section-header">Label & Kelas</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)

    for col, (cls, icon, color, badge_cls, desc) in zip(
        [lc1, lc2, lc3],
        [
            ("Bus",   "🚌", "#FF6B6B", "badge-bus",
             "Class ID: 0 · Dilabeli pertama di Label Studio · Termasuk bus besar & bus mini yang melintas di tol"),
            ("Car",   "🚗", "#4ECDC4", "badge-car",
             "Class ID: 1 · Kelas paling dominan · Mencakup semua jenis kendaraan pribadi roda 4"),
            ("Truck", "🚛", "#FFE66D", "badge-truck",
             "Class ID: 2 · Kendaraan niaga besar · Kadang mirip bus dari sudut CCTV sehingga bisa salah deteksi"),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44">
                <div style="font-size:2.5rem">{icon}</div>
                <div style="font-size:1.2rem;font-weight:700;color:{color};margin:8px 0">{cls}</div>
                <div style="font-size:0.8rem;color:#6b7fa3;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Before/After Comparison images
    st.markdown('<div class="section-header">Perbandingan Visual: Sebelum vs Sesudah Training</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Gambar di bawah menunjukkan perbandingan hasil deteksi menggunakan model COCO pre-trained (SEBELUM training)
    versus model yang sudah di-fine-tune (SESUDAH training). Terlihat jelas peningkatan signifikan dalam
    jumlah kendaraan yang berhasil terdeteksi dan ketepatan klasifikasinya.
    </div>
    """, unsafe_allow_html=True)

    img1_path = "before.png"
    img2_path = "result_after.png"

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**🔴 SEBELUM Training (Pretrained COCO)**")
            st.image(img1_path, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            Model COCO hanya mendeteksi 2–3 kendaraan dengan confidence rendah (0.26–0.34)
            dan bahkan salah mengklasifikasi bus sebagai "bus" dengan confidence sangat rendah.
            Banyak kendaraan sama sekali tidak terdeteksi.
            </div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown("**🟢 SESUDAH Training (Model 2 — Fine-tuned)**")
            st.image(img2_path, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            Model 2 mendeteksi jauh lebih banyak kendaraan dengan confidence tinggi (0.84–0.97).
            Truck dan car berhasil dibedakan dengan baik. Model lebih berani mendeteksi
            kendaraan walau dalam kondisi pencahayaan rendah.
            </div>""", unsafe_allow_html=True)

    # Video Results
    st.markdown('<div class="section-header">Hasil Deteksi pada Video</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Berikut adalah contoh hasil deteksi model pada video rekaman CCTV jalan tol. Terlihat bahwa model mampu mendeteksi berbagai kendaraan dengan bounding box yang cukup akurat, bahkan pada kondisi malam hari. Namun, beberapa kendaraan kecil atau yang berada jauh dari kamera masih bisa terlewatkan.
    </div>""", unsafe_allow_html=True)


    video_path = "https://youtu.be/4NZURigNABs"

    st.video(video_path)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Ringkasan Eksperimen</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Tiga eksperimen dilakukan untuk menemukan konfigurasi terbaik YOLOv11n pada dataset kendaraan tol.
    Semua menggunakan arsitektur <b>YOLOv11n (nano)</b> dengan base pre-trained COCO.
    Variabel yang diubah adalah <b>jumlah epoch</b> dan <b>ukuran input gambar (imgsz)</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Experiment Cards
    ec1, ec2, ec3 = st.columns(3)
    for col, (name, color, epochs, imgsz, batch, opt, note) in zip(
        [ec1, ec2, ec3],
        [
            ("Baseline", "#6b7fa3", 50,  640,  16, "auto (AdamW)",
             "Parameter default YOLOv11. Epoch sedikit & resolusi standard."),
            ("Experiment 2 ⭐", "#60a5fa", 100, 1024, 16, "auto (AdamW)",
             "Epoch 2× lebih banyak, resolusi lebih tinggi. Model terbaik!"),
            ("Experiment 3", "#a78bfa", 100, 1280, 16, "auto (AdamW)",
             "Resolusi tertinggi, namun lebih 'pelit' dalam mendeteksi."),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}55;text-align:left">
                <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:12px">{name}</div>
                <div style="font-size:0.82rem;color:#6b7fa3;line-height:2">
                    <b style="color:#b0bdd4">Epochs:</b> {epochs}<br>
                    <b style="color:#b0bdd4">Image Size:</b> {imgsz}<br>
                    <b style="color:#b0bdd4">Batch:</b> {batch}<br>
                    <b style="color:#b0bdd4">Optimizer:</b> {opt}<br>
                </div>
                <div style="font-size:0.78rem;color:#6b7fa3;margin-top:10px;border-top:1px solid #1e2d4a;padding-top:10px">
                    {note}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Table
    st.markdown('<div class="section-header">Perbandingan Metrik Evaluasi</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="compare-table">
    <thead>
        <tr>
            <th>Metrik</th>
            <th>Baseline</th>
            <th>Experiment 2 ⭐</th>
            <th>Experiment 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="color:#b0bdd4;font-family:'Space Grotesk',sans-serif">mAP50</td>
            <td>0.7341</td>
            <td style="color:#60a5fa;font-weight:700">0.7892</td>
            <td>0.7156</td>
        </tr>
        <tr class="best-row">
            <td style="color:#b0bdd4;font-family:'Space Grotesk',sans-serif">mAP50-95</td>
            <td>0.4821</td>
            <td style="color:#60a5fa;font-weight:700">0.5234</td>
            <td>0.4698</td>
        </tr>
        <tr>
            <td style="color:#b0bdd4;font-family:'Space Grotesk',sans-serif">Precision</td>
            <td style="font-weight:700">0.8102</td>
            <td>0.7645</td>
            <td>0.8390</td>
        </tr>
        <tr>
            <td style="color:#b0bdd4;font-family:'Space Grotesk',sans-serif">Recall</td>
            <td>0.7234</td>
            <td style="color:#60a5fa;font-weight:700">0.8012</td>
            <td>0.6871</td>
        </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric Explanations
    st.markdown('<div class="section-header">Penjelasan Metrik</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div class="info-box">
        📐 <b>Precision (P)</b><br>
        Seberapa akurat deteksi yang dilakukan model — dari semua kotak yang diprediksi,
        berapa persen yang benar-benar kendaraan.<br><br>
        🔍 <b>Recall (R)</b><br>
        Kemampuan model menemukan semua kendaraan yang ada — dari semua kendaraan nyata,
        berapa persen yang berhasil ditemukan model.
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class="info-box">
        📊 <b>mAP50</b><br>
        Mean Average Precision pada threshold IoU 0.50 — mengukur deteksi yang "mudah"
        (overlap bounding box minimal 50%).<br><br>
        📈 <b>mAP50-95</b><br>
        Rata-rata mAP pada berbagai threshold IoU (0.50–0.95) — ukuran performa yang
        lebih komprehensif dan ketat.
        </div>
        """, unsafe_allow_html=True)

    # ── Why Experiment 2 Wins
    st.markdown('<div class="section-header">Mengapa Model 2 Terbaik?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Model 2 dipilih sebagai model terbaik karena memiliki keseimbangan terbaik antara <b>mAP</b> dan <b>Recall</b>:
    <ul style="margin-top:8px;padding-left:18px;line-height:2">
    <li><b>mAP50 & mAP50-95 tertinggi</b> — model paling akurat secara keseluruhan</li>
    <li><b>Recall tertinggi</b> — paling berani dan mampu mendeteksi lebih banyak kendaraan</li>
    <li>Precision lebih rendah dari baseline karena lebih banyak mendeteksi → wajar ada lebih banyak false positive</li>
    <li>Resolusi 1024 dan 100 epoch memberikan waktu belajar yang cukup tanpa overfitting</li>
    <li>Model 3 (imgsz=1280) justru terlalu "pelit" dan melewatkan banyak kendaraan (recall rendah)</li>
    </ul>
    💡 <em>Dalam konteks traffic monitoring, Recall yang tinggi lebih penting — lebih baik ada sedikit false positive
    daripada melewatkan kendaraan yang benar-benar ada.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left-color:#f59e0b">
    ⚠️ <b>Tantangan yang ditemui:</b><br>
    • Data malam hari jauh lebih sedikit → model kurang performa di malam hari<br>
    • Menambah augmentasi kompleks justru menurunkan akurasi (kemungkinan karena data terlalu sedikit: 80 gambar)<br>
    • Labeling manual membutuhkan kehati-hatian — objek yang tidak jelas sengaja tidak dilabeli agar model tidak bingung
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Run Detection
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Deteksi Kendaraan</div>', unsafe_allow_html=True)

    input_type = st.radio("Pilih jenis input:", ["🖼️ Image", "🎬 Video"], horizontal=True)

    # ── Load model
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan di `{MODEL_PATH}`. Pastikan file `best.pt` ada di direktori yang sama dengan `app.py`.")
        st.stop()

    model = load_model(MODEL_PATH)

    # ════════════════════════
    # IMAGE DETECTION
    # ════════════════════════
    if input_type == "🖼️ Image":
        uploaded_img = st.file_uploader(
            "Upload gambar (JPG/PNG/JPEG)",
            type=["jpg", "jpeg", "png"],
            key="img_upload"
        )

        if uploaded_img:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col_orig, col_res = st.columns(2)

            with col_orig:
                st.markdown("**Input Gambar**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("🔍 Menjalankan deteksi..."):
                results = model(img_bgr, conf=conf_thresh, iou=iou_thresh)[0]

            annotated = results.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col_res:
                st.markdown("**Hasil Deteksi**")
                st.image(annotated_rgb, use_container_width=True)

            # ── Detection Stats
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Hasil Deteksi</div>', unsafe_allow_html=True)

            boxes = results.boxes
            counts = {"bus": 0, "car": 0, "truck": 0}
            class_names_map = {0: "bus", 1: "car", 2: "truck"}

            if boxes is not None and len(boxes):
                for cls_id in boxes.cls.cpu().numpy():
                    cls_name = class_names_map.get(int(cls_id), "unknown")
                    if cls_name in counts:
                        counts[cls_name] += 1

            total = sum(counts.values())

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#60a5fa">{total}</div>
                    <div class="metric-label">Total Kendaraan</div></div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#FF6B6B">{counts['bus']}</div>
                    <div class="metric-label">🚌 Bus</div></div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#4ECDC4">{counts['car']}</div>
                    <div class="metric-label">🚗 Car</div></div>""", unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#FFE66D">{counts['truck']}</div>
                    <div class="metric-label">🚛 Truck</div></div>""", unsafe_allow_html=True)

            # ── Confidence scores
            if boxes is not None and len(boxes):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Detail Deteksi (per bounding box):**")
                det_data = []
                for i, (cls_id, conf) in enumerate(
                    zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())
                ):
                    cls_name = class_names_map.get(int(cls_id), "?")
                    det_data.append({
                        "#": i + 1,
                        "Kelas": f"{CLASS_ICONS.get(cls_name,'')} {cls_name}",
                        "Confidence": f"{conf:.3f}",
                    })

                import pandas as pd
                st.dataframe(
                    pd.DataFrame(det_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Download result
            _, buf = cv2.imencode(".jpg", annotated)
            st.download_button(
                "⬇️ Download Hasil Deteksi",
                data=buf.tobytes(),
                file_name="detection_result.jpg",
                mime="image/jpeg",
            )

        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:32px">
            📸 Upload gambar kamera CCTV atau foto kendaraan di jalan tol untuk memulai deteksi.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════
    # VIDEO DETECTION
    # ════════════════════════
    else:
        uploaded_vid = st.file_uploader(
            "Upload video (MP4/MOV/AVI)",
            type=["mp4", "mov", "avi"],
            key="vid_upload"
        )

        if uploaded_vid:
            st.info("🎬 Video diterima. Memproses frame demi frame...")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.markdown(f"""
            <div class="info-box">
            📹 <b>Info Video:</b> {fw}×{fh}px · {fps:.1f} FPS · {total_frames} frame total
            </div>""", unsafe_allow_html=True)

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

            progress_bar = st.progress(0, text="Memproses video...")
            frame_display = st.empty()

            all_counts = {"bus": 0, "car": 0, "truck": 0}
            class_names_map = {0: "bus", 1: "car", 2: "truck"}
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
                annotated = results.plot()
                writer.write(annotated)

                if results.boxes is not None:
                    for cls_id in results.boxes.cls.cpu().numpy():
                        cn = class_names_map.get(int(cls_id), "")
                        if cn in all_counts:
                            all_counts[cn] += 1

                # Show every 10th frame
                if frame_idx % 10 == 0:
                    frame_display.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}/{total_frames}",
                        use_container_width=True,
                    )
                    progress_bar.progress(
                        min(frame_idx / max(total_frames, 1), 1.0),
                        text=f"Frame {frame_idx}/{total_frames}"
                    )

                frame_idx += 1

            cap.release()
            writer.release()
            progress_bar.progress(1.0, text="✅ Selesai!")
            frame_display.empty()

            st.success("🎉 Deteksi selesai!")

            # ── Summary
            st.markdown('<div class="section-header">Ringkasan Deteksi Video</div>', unsafe_allow_html=True)
            vc1, vc2, vc3, vc4 = st.columns(4)
            total_det = sum(all_counts.values())
            with vc1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#60a5fa">{total_det}</div>
                    <div class="metric-label">Total Deteksi</div></div>""", unsafe_allow_html=True)
            with vc2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#FF6B6B">{all_counts['bus']}</div>
                    <div class="metric-label">🚌 Bus</div></div>""", unsafe_allow_html=True)
            with vc3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#4ECDC4">{all_counts['car']}</div>
                    <div class="metric-label">🚗 Car</div></div>""", unsafe_allow_html=True)
            with vc4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#FFE66D">{all_counts['truck']}</div>
                    <div class="metric-label">🚛 Truck</div></div>""", unsafe_allow_html=True)

            # Download
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Video Hasil Deteksi",
                    data=f.read(),
                    file_name="detection_result.mp4",
                    mime="video/mp4",
                )

            os.unlink(tfile.name)

        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:32px">
            🎬 Upload video dari kamera CCTV jalan tol untuk menjalankan deteksi kendaraan frame-by-frame.
            <br><br>
            <span style="color:#6b7fa3;font-size:0.82rem">Format didukung: MP4, MOV, AVI</span>
            </div>
            """, unsafe_allow_html=True)
