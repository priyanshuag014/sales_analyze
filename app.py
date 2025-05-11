import streamlit as st
from database import init_db

# Initialize database
init_db()

# Set page config
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Main app logic
if st.session_state['logged_in']:
    # If logged in, show dashboard
    st.switch_page("pages/dashboard.py")
else:
    # If not logged in, show login page
    st.switch_page("pages/login.py")

st.title("Welcome to Sales Analysis Platform")
st.markdown("Analyze sales data with advanced predictive models. Sign up or log in to get started.")