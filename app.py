import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ============================================================================
# 1. KONFIGURASI HALAMAN MODERN
# ============================================================================
st.set_page_config(
    page_title="AI Comment Moderator",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS Modern
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .comment-box {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .safe {
        border-left: 5px solid #10B981;
    }
    .violation {
        border-left: 5px solid #EF4444;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. LOAD MODEL & ALAT (CACHED)
# ============================================================================
@st.cache_resource
def load_resources():
    """
    Fungsi untuk memuat model dari Hugging Face
    """
    try:
        # Ganti dengan repo ID Anda
        REPO_ID = "alhamdy/redditPSD" 
        
        # Download & Load Model Logistic Regression
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model_reddit.pkl")
        model = joblib.load(model_path)

        # Download & Load Scaler
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler_reddit.pkl")
        scaler = joblib.load(scaler_path)

        # Load SBERT
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        return model, scaler, sbert
    
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

# ============================================================================
# 3. FUNGSI PREPROCESSING
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
    manual_features, text_clean = extract_features_single(text)
    df_manual = pd.DataFrame([manual_features])
    
    embedding = sbert_model.encode([text_clean])
    df_emb = pd.DataFrame(embedding)
    df_emb.columns = [f'emb_{i}' for i in range(df_emb.shape[1])]
    
    df_final = pd.concat([df_manual, df_emb], axis=1)
    df_final = df_final.reindex(columns=scaler_obj.feature_names_in_, fill_value=0)
    
    features_scaled = scaler_obj.transform(df_final)
    
    return features_scaled, manual_features

# ============================================================================
# 4. TAMPILAN UTAMA
# ============================================================================

# Header Modern
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: white; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🤖 AI Comment Moderator</h1>
        <p style='font-size: 1.1rem; opacity: 0.9;'>Deteksi konten tidak pantas dalam komentar secara real-time</p>
    </div>
    """, unsafe_allow_html=True)

# Load resources
model, scaler, sbert = load_resources()

# Input Area
with st.container():
    st.markdown('<div class="comment-box">', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "**Masukkan komentar untuk dianalisis:**",
        height=120,
        placeholder="Contoh: Click this link to get FREE iPhone! Limited time offer...",
        key="comment_input"
    )
    
    analyze_btn = st.button("🔍 Analisis Komentar", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Processing dan Results
if analyze_btn and user_input:
    if model is None:
        st.error("Model belum siap. Silakan coba lagi dalam beberapa saat.")
    else:
        with st.spinner("🔄 Menganalisis komentar..."):
            X_input, debug_features = prepare_input(user_input, sbert, scaler)
            prediction_prob = model.predict_proba(X_input)[0][1]
            prediction_class = 1 if prediction_prob > 0.5 else 0

        # Tampilkan Hasil
        result_class = "violation" if prediction_class == 1 else "safe"
        result_icon = "🚨" if prediction_class == 1 else "✅"
        result_text = "MELANGGAR ATURAN" if prediction_class == 1 else "AMAN"
        result_color = "#EF4444" if prediction_class == 1 else "#10B981"
        
        st.markdown(f"""
        <div class='result-box {result_class}'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <h3 style='color: {result_color}; margin: 0;'>{result_icon} {result_text}</h3>
                    <p style='margin: 0.5rem 0 0 0; color: #6B7280;'>
                        Probabilitas: <strong>{prediction_prob:.2%}</strong>
                    </p>
                </div>
                <div style='font-size: 2rem;'>
                    {result_icon}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress Bar
        progress_value = prediction_prob if prediction_class == 1 else (1 - prediction_prob)
        st.progress(progress_value, text=f"Tingkat keyakinan: {progress_value:.1%}")

        # Penjelasan singkat
        if prediction_class == 1:
            st.info("""
            **Komentar ini terdeteksi melanggar aturan karena:**
            - Mengandung elemen komersial atau promosi
            - Memiliki karakteristik spam atau konten mencurigakan
            - Pola teks menyerupai konten terlarang
            """)
        else:
            st.success("""
            **Komentar ini terlihat aman dan sesuai dengan aturan komunitas.**
            Tidak terdeteksi pola pelanggaran yang signifikan.
            """)

elif analyze_btn and not user_input:
    st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")

# Footer minimal
st.markdown("""
<div style='text-align: center; color: white; margin-top: 3rem; opacity: 0.7;'>
    <p>Powered by Machine Learning • Real-time Content Moderation</p>
</div>
""", unsafe_allow_html=True)