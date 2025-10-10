import streamlit as st
import torch, tempfile, os, shutil, requests
from streamlit_lottie import st_lottie
from backend.google_auth import google_login
from backend.auth_manager import login_ui

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Oceanaut Video Enhancer", page_icon="🌊", layout="wide")

# ------------------- STYLING -------------------
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #002C59 0%, #004E92 100%);
    color: #E0F7FA;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
div.block-container {
    padding-top: 2rem;
}
div.stButton > button {
    background-color: #00AEEF;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #007ACC;
    transform: scale(1.05);
}
.stProgress > div > div > div > div {
    background-color: #00AEEF;
}
h1, h2, h3, h4 {
    color: #FFFFFF !important;
}
footer {visibility: hidden;}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ------------------- ANIMATIONS -------------------
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

wave_anim = load_lottie_url("https://lottie.host/5d1f01d3-6d58-44dc-9b7a-b6a83d6b60ec/0LRDxlr8gF.json")
success_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
progress_anim = load_lottie_url("https://lottie.host/23b08460-1f4d-4987-9349-befb4afbe243/MzvbukSvta.json")  # bubbles shimmer animation

# ------------------- HEADER -------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🌊 Oceanaut - Underwater Video Enhancer")
    st.caption("Experience crystal-clear underwater visuals powered by AI-enhanced clarity.")
with col2:
    if wave_anim:
        st_lottie(wave_anim, height=150)

# ------------------- CENTERED LOGIN -------------------
if "auth_done" not in st.session_state:
    st.session_state.auth_done = False
    st.session_state.logged_in = False
    st.session_state.username = "Guest"

if not st.session_state.auth_done:
    st.markdown("### 👋 Welcome to Oceanaut!")
    st.markdown("Would you like to **log in** or continue as a **guest**?")
    choice = st.radio("Select an option:", ["Login", "Continue as Guest"], horizontal=True)

    if choice == "Login":
        method = st.radio("Choose login method:", ["Google Sign-In", "Manual Login"], horizontal=True)
        if method == "Google Sign-In":
            logged_in, username = google_login()
        else:
            logged_in, username = login_ui()

        if logged_in:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.auth_done = True
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
    else:
        st.session_state.auth_done = True
        st.session_state.logged_in = False
        st.session_state.username = "Guest"
        st.info("Proceeding as Guest — your uploads won’t be saved permanently.")
        st.rerun()

# ------------------- DEVICE INFO -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- COLAB BACKEND -------------------
colab_api_url = "https://inurbanely-leachier-hue.ngrok-free.dev/enhance"

# ------------------- MAIN APP AFTER LOGIN -------------------
if st.session_state.auth_done:
    username = st.session_state.username
    logged_in = st.session_state.logged_in

    uploaded_file = st.file_uploader("🎥 Upload your underwater video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        if logged_in:
            user_dir = os.path.join("uploads", username)
            os.makedirs(user_dir, exist_ok=True)
            input_path = os.path.join(user_dir, uploaded_file.name)
            st.success(f"💾 Video saved for user **{username}**")
        else:
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, uploaded_file.name)
            st.info("You are not logged in — your upload will not be saved permanently.")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(input_path)
        st.success("✅ Video uploaded successfully!")

        # ------------------- ENHANCEMENT -------------------
        if st.button("🚀 Enhance Video"):
            placeholder = st.empty()

            with st.spinner("✨ Sending video to GPU backend... Please wait ⏳"):
                if progress_anim:
                    with placeholder.container():
                        st_lottie(progress_anim, height=200, key="processing_anim")
                        st.caption("💡 Oceanaut AI is enhancing your underwater video...")

                try:
                    with open(input_path, "rb") as f:
                        files = {"file": f}
                        response = requests.post(colab_api_url, files=files, timeout=600)

                    placeholder.empty()

                    if response.status_code == 200:
                        output_path = os.path.join(tempfile.mkdtemp(), "enhanced_colab.mp4")
                        with open(output_path, "wb") as f:
                            f.write(response.content)

                        st.video(output_path)
                        st.success("✅ Enhancement complete! Your video is now crystal clear 🌊")

                        if success_anim:
                            st_lottie(success_anim, height=150)

                        with open(output_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Enhanced Video",
                                f,
                                file_name="enhanced_video.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("❌ The enhancement couldn't be completed. Please try again later.")

                except requests.exceptions.ConnectionError:
                    placeholder.empty()
                    st.error("⚠️ Could not connect to the GPU backend. Please retry shortly.")
                except requests.exceptions.Timeout:
                    placeholder.empty()
                    st.error("⌛ The GPU backend took too long to respond. Try again later.")
                except Exception:
                    placeholder.empty()
                    st.error("⚠️ An unexpected error occurred during enhancement. Please try again.")

    else:
        st.info("📤 Upload your underwater video to start enhancing.")

# ------------------- FOOTER -------------------
st.markdown("""
<hr style="border: 0.5px solid #00AEEF;">
<div style="text-align:center; color:#D0E6F6; font-size:14px;">
Developed by <b>Oceanaut AI Team</b> | Powered by Deep Fusion U-Net Models
</div>
""", unsafe_allow_html=True)
