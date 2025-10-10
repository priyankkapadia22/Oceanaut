import streamlit as st
import json
import os

# Unified Google Sign-In Handler (works both local + Streamlit Cloud)
def google_login():
    st.markdown("### 🔐 Google Sign-In")

    # Already logged in
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.success(f"Welcome, {st.session_state['username']} 🌊")
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()
        return True, st.session_state["username"]

    # Detect environment
    on_cloud = any(
        d in os.getenv("STREAMLIT_SERVER_HOST", "")
        or os.getenv("STREAMLIT_SERVER_HEADLESS", "")
        or st.get_option("browser.serverAddress", "").endswith("streamlit.app")
        for d in ["streamlit", ".app"]
    )

    if on_cloud:
        # 🔹 Streamlit Cloud → Real Google Sign-In via Firebase JS SDK
        firebase_config = {
            "apiKey": st.secrets["firebase"]["apiKey"],
            "authDomain": st.secrets["firebase"]["authDomain"],
            "projectId": st.secrets["firebase"]["projectId"],
            "storageBucket": st.secrets["firebase"]["storageBucket"],
            "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
            "appId": st.secrets["firebase"]["appId"],
        }

        st.markdown(
            f"""
            <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js"></script>
            <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-auth-compat.js"></script>
            <script>
            const firebaseConfig = {json.dumps(firebase_config)};
            firebase.initializeApp(firebaseConfig);
            const provider = new firebase.auth.GoogleAuthProvider();
            function signIn() {{
                firebase.auth().signInWithPopup(provider)
                .then((result) => {{
                    const user = result.user;
                    const username = user.displayName;
                    const email = user.email;
                    window.parent.postMessage({{ "logged_in": true, "username": username, "email": email }}, "*");
                }})
                .catch((error) => {{
                    alert("Sign-in failed: " + error.message);
                }});
            }}
            </script>
            <button onclick="signIn()" style="background-color:#00AEEF;color:white;padding:0.6em 1.2em;border:none;border-radius:6px;font-weight:bold;cursor:pointer;">Sign in with Google</button>
            """,
            unsafe_allow_html=True,
        )
    else:
        # 🔹 Localhost → show info message
        st.info("To sign in with Google:")
        st.markdown(
            """
            1️⃣ Deploy your app on [Streamlit Cloud](https://share.streamlit.io/)  
            2️⃣ Add Firebase config keys to **Secrets**  
            3️⃣ Google Sign-In will work automatically there 🌐  
            """
        )
        st.caption("For now, use manual login locally to continue testing.")

    return False, ""
