import streamlit as st
from database import signup_user

st.set_page_config(page_title="Sign Up", layout="centered")

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def signup():
    st.title("Sign Up for Sales Analysis")
    st.image("assets/sales_image2.jpg", caption="Join the Sales Revolution", use_column_width=True)
    
    email = st.text_input("Email")
    password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up", key="signup_btn"):
        if not email or not password or not confirm_password:
            st.error("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        elif signup_user(email, password):
            st.success("Sign up successful! Please log in.")
            st.switch_page("pages/login.py")
        else:
            st.error("Email already exists")
    
    st.markdown("[Login](#login)", unsafe_allow_html=True)

signup()