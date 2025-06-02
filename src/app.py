# trading_dashboard/app.py
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
from datetime import datetime

# Import from our modules
import config
import ui_components
import data_processing
import charting
import ai_services

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'data_df' not in st.session_state:
        st.session_state.data_df = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None

def main():
    """Main function to run the Streamlit dashboard."""
    config.configure_streamlit_page()
    ui_components.load_custom_css()
    initialize_session_state()

    ui_components.display_main_header()
    
    # Attempt to get Gemini model early to show warning if not configured
    gemini_model = config.get_gemini_model()

    uploaded_file = ui_components.display_sidebar_upload()

    if uploaded_file:
        if st.session_state.data_df is None: # Load only if not already loaded or file changes
            with st.spinner("🔄 Loading and processing data... This may take a moment."):
                df = data_processing.load_and_process_data(uploaded_file)
                if df is not None and not df.empty:
                    st.session_state.data_df = df
                    st.session_state.data_summary = data_processing.create_data_summary(df)
                    st.success(f"✅ Data loaded successfully! {st.session_state.data_summary.get('total_records', 0)} records processed.")
                elif df is None: # Error handled in load_and_process_data
                    st.session_state.data_df = None # Ensure it's reset
                    st.session_state.data_summary = None
                else: # df is empty
                    st.warning("⚠️ Data loaded, but the processed DataFrame is empty. Please check file content and filters.")
                    st.session_state.data_df = None 
                    st.session_state.data_summary = None


    elif st.session_state.data_df is None : # No file and no data in session
        ui_components.display_file_upload_instructions()
        return # Stop further execution if no data

    # Retrieve from session state
    df = st.session_state.data_df
    summary = st.session_state.data_summary

    if df is None or df.empty or summary is None:
        if uploaded_file: # If a file was uploaded but processing failed or resulted in empty df
             st.warning("Data could not be processed or resulted in an empty dataset. Please check the file.")
        # else: # No file uploaded yet, instructions already shown
        return


    # --- Tabs ---
    tab_titles = ["📊 Interactive Chart", "🤖 AI Financial Analyst", "📋 Data Overview"]
    chart_tab, ai_tab, data_tab = st.tabs(tab_titles)

    with chart_tab:
        st.markdown('<p class="section-header">TSLA Candlestick Chart with Signals & Bands</p>', unsafe_allow_html=True)
        ui_components.display_metric_cards(summary)
        
        chart_render_data = charting.assemble_chart_data(df)
        if chart_render_data:
            renderLightweightCharts(chart_render_data, 'tsla_trading_chart')
            ui_components.display_chart_legend()
        else:
            st.warning("Could not generate chart data.")

    with ai_tab:
        st.markdown('<p class="section-header">🤖 AI Financial Analyst</p>', unsafe_allow_html=True)

        if not gemini_model:
            st.warning("⚠️ Gemini AI Model is not configured. Please check your API key in the .env file.")
        else:
            template_questions = [
                "How many LONG days were there in 2023?",
                "What was the average close price in Q2 2024?",
                "What's the highest high price recorded overall?",
                "Compare average closing price for LONG vs SHORT days.",
                "Describe the price statistics for January 2024.",
                "Count days where the closing price was above $180."
            ]
            ui_components.display_ai_tab_intro_and_templates(template_questions)

            # Custom styled text area container
            st.markdown('<label for="question_input_area_streamlit" style="font-weight: 500; color: #1E88E5; display: block; margin-bottom: 0.5rem;">Ask your question about TSLA data:</label>', unsafe_allow_html=True)
            st.markdown('<div class="question-input-box">', unsafe_allow_html=True)
            user_question = st.text_area(
                "Enter your question here:",
                value=st.session_state.get('question_input', ''),
                key="question_input_main_area", # Unique key
                height=100,
                placeholder="e.g., What was the volatility in March 2024?",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.question_input = user_question # Update session state from text_area

            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
                current_question = st.session_state.get('question_input', '').strip()
                if current_question:
                    with st.spinner("🧠 AI is analyzing... Please wait."):
                        ai_response_text = ai_services.get_ai_assistant_response(gemini_model, df, summary, current_question)
                        
                        # Store in chat history
                        st.session_state.chat_history.append({
                            'question': current_question,
                            'response': ai_response_text,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        # Display latest response immediately without re-rendering history via button
                        ui_components.display_ai_response(ai_response_text)

                else:
                    st.warning("Please enter a question first!")
            
            if col_btn2.button("🗑️ Clear Question Input", use_container_width=True):
                st.session_state.question_input = ""
                # st.session_state.question_input_main_area = "" # Clear widget state too
                st.rerun()
            
            ui_components.display_chat_history()


    with data_tab:
        ui_components.display_data_overview_tab(df, summary)


if __name__ == "__main__":
    main()