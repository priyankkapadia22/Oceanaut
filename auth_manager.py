import streamlit as st
import sqlite3
import hashlib
import os

DB_PATH = "users.db"

def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        saved_video TEXT
    )''')
    conn.commit()
    conn.close()

def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
              (username, hash_pass(password)))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_pass(password)))
    result = c.fetchone()
    conn.close()
    return result is not None

def login_ui():
    st.sidebar.subheader("🔑 Manual Login / Signup")
    init_user_db()
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.radio("Choose Option", menu)

    if choice == "Sign Up":
        username = st.sidebar.text_input("Create Username")
        password = st.sidebar.text_input("Create Password", type="password")
        if st.sidebar.button("Sign Up"):
            add_user(username, password)
            st.sidebar.success("✅ Account created successfully!")
            st.sidebar.info("Please login to continue.")
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if verify_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.sidebar.success(f"Welcome, {username}!")
            else:
                st.sidebar.error("❌ Invalid username or password.")

    if st.session_state.get("logged_in"):
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    return st.session_state.get("logged_in", False), st.session_state.get("username", "")
