import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

def setup_page(page_title="Data Analytics Dashboard", page_icon="📈"):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide"
    )

@st.cache_resource
def configure_gemini_api():
    """Configure Gemini API with API key from .env file"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in environment variables!")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file in your project root
        2. Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
        3. Get your API key from: https://makersuite.google.com/app/apikey
        """)
        return None

    try:
        genai.configure(api_key=api_key)
        # Using a common model, adjust if you have access to specific versions like 'gemini-2.0-flash'
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {str(e)}")
        return None