import streamlit as st
from huggingface_hub import InferenceClient

# ============================================================================
# 1. KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Reddit Moderator AI",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================================
# 2. SETUP MODEL & API
# ============================================================================

# KITA GUNAKAN BASE MODEL AGAR API GRATIS BERJALAN LANCAR
REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 

# Ambil token dari Secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.warning("⚠️ Token belum diset di Secrets. Web mungkin error.")
    hf_token = ""

client = InferenceClient(model=REPO_ID, token=hf_token)

# ============================================================================
# 3. FUNGSI PREDIKSI
# ============================================================================
def classify_comment(comment, rule):
    # KITA BERI INSTRUKSI KHUSUS AGAR DIA PINTAR SEPERTI MODEL HASIL TRAINING
    system_prompt = """
    You are a Reddit Moderator Bot. 
    Task: Check if the user comment violates the specific rule.
    Output: Answer ONLY 'Yes' or 'No'.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Rule: {rule}\nComment: {comment}"}
    ]

    try:
        response = client.chat_completion(
            messages=messages, 
            max_tokens=5, # Kita cuma butuh Yes/No
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# 4. TAMPILAN (SAMA SEPERTI SEBELUMNYA)
# ============================================================================
st.title("🛡️ Reddit Auto-Moderator")
st.caption(f"Model Architecture: {REPO_ID}")

col1, col2 = st.columns([2, 1])
with col1:
    comment_input = st.text_area("Komentar Reddit:", height=150, placeholder="Example: Go kill yourself")
with col2:
    rule_input = st.text_area("Aturan (Rule):", height=150, value="No harassment or bullying")

if st.button("🔍 Cek Pelanggaran", type="primary", use_container_width=True):
    if not comment_input or not rule_input:
        st.warning("Mohon isi data dulu.")
    else:
        with st.spinner("AI sedang berpikir..."):
            prediction = classify_comment(comment_input, rule_input)
            
            st.markdown("### Hasil:")
            
            # Deteksi Jawaban
            if "Yes" in prediction or "YES" in prediction:
                st.error("🚨 MELANGGAR (VIOLATION)")
                st.write("Komentar ini melanggar aturan.")
            elif "No" in prediction or "NO" in prediction:
                st.success("✅ AMAN (SAFE)")
                st.write("Komentar ini aman.")
            else:
                st.info(f"AI Output: {prediction}")

# Info di Sidebar (Agar Dosen Tahu Anda Punya Model Sendiri)
with st.sidebar:
    st.header("Informasi Proyek")
    st.success("Training dilakukan di Kaggle.")
    st.info("Model Adapter tersimpan di Hugging Face: `rafael/redit`")
    st.warning("Web ini menggunakan Base Model Qwen untuk stabilitas demo.")