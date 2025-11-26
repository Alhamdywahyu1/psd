import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ============================================================================
# 1. KONFIGURASI HALAMAN (UI)
# ============================================================================
st.set_page_config(
    page_title="Reddit Mod AI Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. LOAD MODEL & ALAT (CACHED)
# ============================================================================
@st.cache_resource
def load_resources():
    """
    Fungsi ini mendownload model dari Hugging Face dan meload-nya.
    Cache resource agar tidak download ulang setiap kali klik tombol.
    """
    # GANTI INI DENGAN REPO ANDA!
    REPO_ID = "alhamdy/redditPSD" 
    
    status_text = st.empty()
    status_text.info("🔄 Sedang memuat model AI dari Cloud...")

    try:
        # 1. Download & Load Model Logistic Regression
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model_reddit.pkl")
        model = joblib.load(model_path)

        # 2. Download & Load Scaler
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler_reddit.pkl")
        scaler = joblib.load(scaler_path)

        # 3. Load SBERT (Otomatis download jika belum ada di cache sistem)
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        status_text.empty() # Hapus pesan loading
        return model, scaler, sbert
    
    except Exception as e:
        status_text.error(f"Gagal memuat model. Pastikan REPO_ID benar. Error: {e}")
        return None, None, None

model, scaler, sbert = load_resources()

# ============================================================================
# 3. FUNGSI PREPROCESSING (SAMA PERSIS DENGAN TRAINING)
# ============================================================================
COMMERCIAL_KEYWORDS = ['buy', 'sell', 'discount', 'free', 'click', 'visit', 'check out', 'sale', 'offer', 'deal', 'promo', 'code', 'limited', 'subscribe', 'join', 'sign up', 'price', 'money', 'bonus']
LEGAL_KEYWORDS = ['sue', 'lawsuit', 'lawyer', 'legal', 'court', 'judge', 'illegal', 'law', 'crime', 'police', 'contract', 'rights', 'should i', 'can i', 'is it legal']

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|httpsS+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features_single(text):
    # Ekstrak fitur manual (sama seperti saat training)
    features = {}
    text_clean = clean_text(text)
    
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['has_url'] = 1 if re.search(r'http\S+|www\S+', text) else 0
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['has_all_caps'] = 1 if re.search(r'\b[A-Z]{3,}\b', text) else 0
    
    text_lower = text.lower()
    features['commercial_keyword_count'] = sum(1 for kw in COMMERCIAL_KEYWORDS if kw in text_lower)
    features['legal_keyword_count'] = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    
    return features, text_clean

def prepare_input(text, sbert_model, scaler_obj):
    # 1. Ekstrak Fitur Manual
    manual_features, text_clean = extract_features_single(text)
    df_manual = pd.DataFrame([manual_features])
    
    # 2. Embedding SBERT
    embedding = sbert_model.encode([text_clean])
    df_emb = pd.DataFrame(embedding)
    df_emb.columns = [f'emb_{i}' for i in range(df_emb.shape[1])]
    
    # 3. Gabungkan
    df_final = pd.concat([df_manual, df_emb], axis=1)
    
    # 4. Samakan kolom dengan Scaler (Agar urutan tidak tertukar)
    # Mengisi 0 jika ada kolom hilang (safe guard)
    df_final = df_final.reindex(columns=scaler_obj.feature_names_in_, fill_value=0)
    
    # 5. Scaling
    features_scaled = scaler_obj.transform(df_final)
    
    return features_scaled, manual_features

# ============================================================================
# 4. TAMPILAN UTAMA WEB
# ============================================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/58/Reddit_logo_new.svg/2560px-Reddit_logo_new.svg.png", width=100)
    st.header("Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan **Machine Learning** (Logistic Regression + SBERT) untuk mendeteksi apakah komentar Reddit melanggar aturan atau tidak.
    
    **Tugas PSD - Kelompok X**
    1. Nama Mahasiswa 1
    2. Nama Mahasiswa 2
    """)
    st.markdown("---")
    st.write("Model hosted on Hugging Face 🤗")

# --- Main Content ---
st.title("🛡️ Reddit Comment Moderator AI")
st.markdown("##### Deteksi Pelanggaran Aturan Komunitas secara Real-Time")

# Area Input
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Masukkan Komentar Reddit:", height=150, placeholder="Contoh: Click this link to buy cheap iphone!...")
    
    analyze_btn = st.button("🔍 Analisis Komentar", type="primary")

# Area Hasil (Output)
if analyze_btn and user_input:
    if model is None:
        st.error("Model belum siap. Periksa koneksi atau Repo ID.")
    else:
        with st.spinner("Sedang menganalisis teks..."):
            # Proses Data
            X_input, debug_features = prepare_input(user_input, sbert, scaler)
            
            # Prediksi
            prediction_prob = model.predict_proba(X_input)[0][1] # Ambil probabilitas kelas 1 (Violation)
            prediction_class = 1 if prediction_prob > 0.5 else 0

        # Tampilkan Hasil dengan layout kolom
        st.markdown("### Hasil Analisis")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction_class == 1:
                st.error("🚨 MELANGGAR ATURAN")
                st.write("Komentar ini berpotensi melanggar aturan.")
            else:
                st.success("✅ AMAN / SAFE")
                st.write("Komentar ini terlihat aman.")

        with res_col2:
            st.metric(label="Probability Score", value=f"{prediction_prob:.2%}")
            # Progress bar warna warni
            if prediction_prob > 0.5:
                st.progress(prediction_prob, text="Tingkat Risiko Tinggi")
            else:
                st.progress(prediction_prob, text="Tingkat Risiko Rendah")

        # --- Detail Teknis (Explainability) ---
        with st.expander("📊 Lihat Detail Fitur (Untuk Dosen/Analisis)"):
            st.write("Fitur yang diekstrak dari teks:")
            st.json(debug_features)
            st.write("""
            *Catatan: Jika 'Probability Score' > 50%, model mengklasifikasikan sebagai pelanggaran.*
            """)

elif analyze_btn and not user_input:
    st.warning("⚠️ Mohon masukkan teks terlebih dahulu.")