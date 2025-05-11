import streamlit as st
from database import login_user
import os

st.set_page_config(page_title="Login", layout="centered")

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state for navigation
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login():
    st.title("Login to Sales Analysis")
    st.image("assets/sales_image1.jpg", caption="Unlock Sales Insights", use_container_width=True)
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login", key="login_btn"):
        if not email or not password:
            st.error("Please fill in all fields")
        elif login_user(email, password):
            st.session_state['logged_in'] = True
            st.session_state['user_email'] = email
            st.success("Login successful! Redirecting to dashboard...")
            st.rerun()
        else:
            st.error("Invalid email or password")

# Show login form if not logged in
if not st.session_state['logged_in']:
    login()
else:
    st.switch_page("pages/dashboard.py")