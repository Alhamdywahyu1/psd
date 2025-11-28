import streamlit as st
from huggingface_hub import InferenceClient

# ============================================================================
# 1. KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Reddit Moderator AI (Qwen 2.5)",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS agar tampilan lebih bersih
st.markdown("""
<style>
    .stTextArea textarea {font-size: 16px;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. SETUP MODEL & API
# ============================================================================

# --- PENTING: GANTI INI DENGAN REPO ID ANDA ---
# Contoh: "budi_santoso/qwen-reddit-moderator"
REPO_ID = "USERNAME_HF_ANDA/NAMA_REPO_MODEL_ANDA" 

# Ambil token dari Streamlit Secrets (untuk keamanan)
# Nanti kita set di dashboard Streamlit Cloud
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    # Fallback untuk testing lokal (TIDAK DISARANKAN UNTUK UPLOAD GITHUB)
    # Jika run lokal, Anda bisa isi token di sini sementara
    hf_token = "" 
    st.warning("⚠️ Token belum diset di Secrets. Aplikasi mungkin error jika model Private.")

# Inisialisasi Client Hugging Face
client = InferenceClient(model=REPO_ID, token=hf_token)

# ============================================================================
# 3. FUNGSI PREDIKSI
# ============================================================================
def classify_comment(comment, rule):
    # Prompt System sesuai training Anda
    system_prompt = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."
    
    # Format pesan untuk Qwen (Chat Template)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Comment: {comment}\n\nrule: {rule}"}
    ]

    try:
        # Panggil API Hugging Face
        response = client.chat_completion(
            messages=messages, 
            max_tokens=10, # Kita cuma butuh jawaban singkat Yes/No
            temperature=0.1 # Agar jawaban konsisten
        )
        # Ambil teks jawaban
        result_text = response.choices[0].message.content.strip()
        return result_text
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# 4. TAMPILAN USER INTERFACE (UI)
# ============================================================================
st.title("🛡️ Reddit Auto-Moderator")
st.caption(f"Powered by Custom Qwen 2.5 Model | Repo: {REPO_ID}")

st.markdown("---")

# Input Form
col1, col2 = st.columns([2, 1])

with col1:
    comment_input = st.text_area("Isi Komentar Reddit:", height=150, placeholder="Contoh: You are stupid and ugly!")

with col2:
    rule_input = st.text_area("Aturan (Rule):", height=150, value="No harassment or bullying", placeholder="Masukkan aturan yang dilanggar...")

# Tombol Analisis
if st.button("🔍 Cek Pelanggaran", type="primary", use_container_width=True):
    if not comment_input or not rule_input:
        st.warning("Mohon isi komentar dan aturan terlebih dahulu.")
    else:
        with st.spinner("🤖 AI sedang membaca komentar..."):
            # Lakukan Prediksi
            prediction = classify_comment(comment_input, rule_input)
            
            # Tampilkan Hasil
            st.markdown("### Hasil Analisis:")
            
            # Logic tampilan warna
            # Karena Qwen kadang menjawab "Yes." atau "Yes" (tanpa titik), kita cek stringnya
            if "Yes" in prediction or "YES" in prediction:
                st.error("🚨 MELANGGAR ATURAN (VIOLATION)")
                st.write(f"**AI Decision:** {prediction}")
                st.info("Komentar ini sebaiknya dihapus.")
            
            elif "No" in prediction or "NO" in prediction:
                st.success("✅ AMAN (SAFE)")
                st.write(f"**AI Decision:** {prediction}")
                st.info("Komentar ini tidak melanggar aturan.")
            
            else:
                # Jika AI bingung atau error
                st.warning("⚠️ HASIL TIDAK PASTI")
                st.write(f"Output AI: {prediction}")

# Sidebar Info
with st.sidebar:
    st.header("Tentang Model")
    st.write("""
    Web ini menggunakan model **Qwen 2.5 - 7B** yang telah di-finetune menggunakan teknik **LoRA**.
    
    **Fitur:**
    - Input Komentar
    - Input Aturan Spesifik
    - Output Yes/No
    """)
    st.markdown("---")
    st.write("Tugas PSD")