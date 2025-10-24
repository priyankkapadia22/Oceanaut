import streamlit as st
import torch, tempfile, os, shutil, cv2, requests
from streamlit_lottie import st_lottie
from backend.utils.video_utils import extract_frames, combine_frames_to_video
from backend.utils.preprocess import preprocess_frame, postprocess_tensor
from backend.oceanaut_model.model import EnhancedUNet
from backend.oceanaut_model.fusion_cnn import FusionUNet
from backend.auth_manager import login_ui
from backend.google_auth import google_login

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Oceanaut Video Enhancer", page_icon="🌊", layout="wide")

# ------------------- CUSTOM STYLING -------------------
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
h1, h2, h3, h4, p {
    color: #FFFFFF !important;
}
footer {visibility: hidden;}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ------------------- LOTTIE ANIMATIONS -------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

wave_anim = load_lottie_url("https://lottie.host/5d1f01d3-6d58-44dc-9b7a-b6a83d6b60ec/0LRDxlr8gF.json")
success_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
processing_anim = load_lottie_url("https://lottie.host/23b08460-1f4d-4987-9349-befb4afbe243/MzvbukSvta.json")

# ------------------- HEADER -------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🌊 Oceanaut - Underwater Video Enhancer")
    st.caption("Experience crystal-clear underwater visuals powered by AI-enhanced clarity.")
with col2:
    if wave_anim:
        st_lottie(wave_anim, height=150, key="wave_anim")

# ------------------- LOGIN PROMPT (CENTERED) -------------------
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

# ------------------- MODEL LOADING -------------------
@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_55 = EnhancedUNet(use_transformer=True).to(device)
    model_85 = EnhancedUNet(use_transformer=True).to(device)
    fusion_model = FusionUNet().to(device)

    model_55.load_state_dict(torch.load("backend/model_weights/model_epoch_55.pth", map_location=device))
    model_85.load_state_dict(torch.load("backend/model_weights/model_epoch_85.pth", map_location=device))
    fusion_model.load_state_dict(torch.load("backend/model_weights/fusion_model_epoch_60.pth", map_location=device))

    model_55.eval(); model_85.eval(); fusion_model.eval()
    return model_55, model_85, fusion_model, device

# ------------------- MAIN APP -------------------
if st.session_state.auth_done:
    username = st.session_state.username
    logged_in = st.session_state.logged_in
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            st.info("You are not logged in — the uploaded video won't be saved permanently.")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(input_path)
        st.success("✅ Video uploaded successfully!")

        # ------------------- ENHANCE VIDEO -------------------
        if st.button("🚀 Enhance Video"):
            placeholder = st.empty()

            with st.spinner("✨ Enhancing your video with Oceanaut AI... Please wait ⏳"):
                if processing_anim:
                    with placeholder.container():
                        st_lottie(processing_anim, height=200, key="processing")
                        st.caption("💡 Oceanaut AI is enhancing your underwater video...")

                frames_dir = os.path.join(tempfile.mkdtemp(), "frames")
                frames, fps = extract_frames(input_path, frames_dir)

                model_55, model_85, fusion_model, device = load_models()

                try:
                    enh55, enh85 = [], []
                    for frame_path in frames:
                        tensor = preprocess_frame(frame_path).to(device)
                        with torch.no_grad():
                            o55 = model_55(tensor)
                            o85 = model_85(tensor)
                        enh55.append(o55)
                        enh85.append(o85)

                    output_dir = os.path.join(tempfile.mkdtemp(), "enhanced_frames")
                    os.makedirs(output_dir, exist_ok=True)
                    progress = st.progress(0)
                    total_frames = len(frames)

                    for i in range(1, total_frames - 1):
                        with torch.no_grad():
                            stacked = torch.cat([
                                enh55[i - 1], enh85[i - 1],
                                enh55[i], enh85[i],
                                enh55[i + 1], enh85[i + 1]
                            ], dim=1)
                            fused = fusion_model(stacked)
                            enhanced = postprocess_tensor(fused.squeeze(0))

                        cv2.imwrite(os.path.join(output_dir, f"enh_{i:05d}.jpg"), enhanced)
                        if i % 10 == 0:
                            progress.progress((i + 1) / total_frames)

                    output_video = os.path.join(tempfile.mkdtemp(), "enhanced_video.mp4")
                    combine_frames_to_video(output_dir, output_video, fps)

                    placeholder.empty()
                    st.video(output_video)
                    st.success("✅ Enhancement complete! Your underwater world is now crystal clear 🌊")

                    if success_anim:
                        st_lottie(success_anim, height=150, key="done_anim")

                    with open(output_video, "rb") as f:
                        st.download_button(
                            "⬇️ Download Enhanced Video",
                            f,
                            file_name="enhanced_video.mp4",
                            mime="video/mp4"
                        )

                except Exception as e:
                    placeholder.empty()
                    st.error(f"⚠️ Something went wrong during enhancement: {e}")

                finally:
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    shutil.rmtree(output_dir, ignore_errors=True)

    else:
        st.info("📤 Upload your underwater video to start enhancing.")

# ------------------- FOOTER -------------------
st.markdown("""
<hr style="border: 0.5px solid #00AEEF;">
<div style="text-align:center; color:#D0E6F6; font-size:14px;">
Developed by <b>Oceanaut AI Team</b> | Powered by Deep Fusion U-Net Models
</div>
""", unsafe_allow_html=True)
