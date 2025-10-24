import streamlit as st
import torch, tempfile, os, cv2, requests
from PIL import Image
from streamlit_lottie import st_lottie
from backend.utils.image_utils import preprocess_image, postprocess_tensor, tensor_to_download_bytes
from backend.oceanaut_model.model import EnhancedUNet
from backend.oceanaut_model.fusion_cnn import FusionUNet
from backend.auth_manager import login_ui
from backend.google_auth import google_login

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Oceanaut Image Enhancer", page_icon="🌊", layout="wide")

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
    st.title("🌊 Oceanaut - Underwater Image Enhancer")
    st.caption("Experience crystal-clear underwater imagery powered by AI-enhanced clarity.")
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

# ------------------- MODEL LOADING (silent) -------------------
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

    uploaded_file = st.file_uploader("📸 Upload your underwater image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if logged_in:
            user_dir = os.path.join("uploads", username)
            os.makedirs(user_dir, exist_ok=True)
            input_path = os.path.join(user_dir, uploaded_file.name)
            st.success(f"💾 Image saved for user **{username}**")
        else:
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        image = Image.open(input_path).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        # ------------------- ENHANCE IMAGE -------------------
        if st.button("🚀 Enhance Image"):
            placeholder = st.empty()

            with st.spinner("✨ Enhancing your image with Oceanaut AI... Please wait ⏳"):
                if processing_anim:
                    with placeholder.container():
                        st_lottie(processing_anim, height=180, key="processing")
                        st.caption("💡 Oceanaut AI is restoring clarity and colors...")

                        # Subtle shimmer bar (ocean wave effect)
                        shimmer_html = """
                        <div style="
                            margin-top: 20px;
                            height: 10px;
                            width: 80%;
                            max-width: 400px;
                            background: linear-gradient(90deg, #004E92 25%, #00AEEF 50%, #004E92 75%);
                            background-size: 200% 100%;
                            border-radius: 8px;
                            animation: shimmer 1.6s infinite linear;
                            margin-left: auto;
                            margin-right: auto;
                        "></div>

                        <style>
                        @keyframes shimmer {
                            0% { background-position: 200% 0; }
                            100% { background-position: -200% 0; }
                        }
                        </style>
                        """
                        st.markdown(shimmer_html, unsafe_allow_html=True)

                try:
                    model_55, model_85, fusion_model, device = load_models()
                    inp = preprocess_image(image).to(device)

                    with torch.no_grad():
                        o55 = model_55(inp)
                        o85 = model_85(inp)
                        stacked = torch.cat([o55, o85, o55, o85, o55, o85], dim=1)
                        enhanced = fusion_model(stacked)

                    enhanced_img = postprocess_tensor(enhanced)
                    placeholder.empty()

                    # Gentle completion effect (soft glow fade)
                    st.markdown("""
                    <div style="
                        text-align:center;
                        padding:10px;
                        animation: glowfade 2s ease-in-out;
                        color:#AEEFFF;
                        font-weight:600;
                    ">
                    🌊 Enhancement Complete — Dive into Clarity!
                    </div>

                    <style>
                    @keyframes glowfade {
                        0% { opacity: 0; text-shadow: 0 0 5px #00AEEF; }
                        50% { opacity: 1; text-shadow: 0 0 15px #00AEEF; }
                        100% { opacity: 1; text-shadow: 0 0 5px #00AEEF; }
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.image(enhanced_img, caption="✨ Enhanced Image", use_container_width=True)
                    st.download_button(
                        label="⬇️ Download Enhanced Image",
                        data=tensor_to_download_bytes(enhanced_img),
                        file_name="enhanced_image.png",
                        mime="image/png"
                    )

                except Exception as e:
                    placeholder.empty()
                    st.error(f"⚠️ Something went wrong during enhancement: {e}")


    else:
        st.info("📤 Upload a marine image to start enhancing.")

# ------------------- FOOTER -------------------
st.markdown("""
<hr style="border: 0.5px solid #00AEEF;">
<div style="text-align:center; color:#D0E6F6; font-size:14px;">
Developed by <b>Oceanaut AI Team</b> | Powered by Deep Fusion U-Net Models
</div>
""", unsafe_allow_html=True)
