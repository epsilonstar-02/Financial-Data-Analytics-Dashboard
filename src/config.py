# trading_dashboard/config.py
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()

def configure_streamlit_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="TSLA Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )

@st.cache_resource
def get_gemini_model():
    """Configure and return Gemini API model."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in environment variables!")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file in your project root (`trading_dashboard/.env`)
        2. Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
        3. Get your API key from: https://makersuite.google.com/app/apikey
        """)
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Consider making the model name configurable if you plan to switch often
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using latest flash model
        # Test with a simple generation to ensure connectivity
        model.generate_content("test") 
        return model
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {str(e)}")
        st.error("Please ensure your API key is valid and has access to the 'gemini-1.5-flash-latest' model.")
        return None

# Initial setup when this module is imported
load_environment_variables()