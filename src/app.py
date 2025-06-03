import streamlit as st
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts
from datetime import datetime

from config import setup_page, configure_gemini_api
from data_processing import load_and_prepare_data, create_data_summary
from charting import create_chart_config, create_markers, create_floating_band_series_components
from ai_agent import get_ai_response # parse_question_with_gemini is used within get_ai_response

# --- Column Mapping Configuration ---
# Users can modify these if their CSV has different column names
# Variants are tried in order.
COLUMN_MAP = {
    'timestamp': ['timestamp', 'Date', 'TIME', 'Datetime', 'datetime'],
    'open': ['open', 'Open', 'OPEN'],
    'high': ['high', 'High', 'HIGH'],
    'low': ['low', 'Low', 'LOW'],
    'close': ['close', 'Close', 'CLOSE'],
    'volume': ['volume', 'Volume', 'VOLUME'], # Optional, handled if present
    'direction': ['direction', 'Direction', 'Signal'] # Optional
}
# Default names for raw support/resistance list columns (if S/R processing is enabled)
RAW_SUPPORT_COL_DEFAULT = 'Support'
RAW_RESISTANCE_COL_DEFAULT = 'Resistance'


def main():
    setup_page(page_title="Financial Data Dashboard", page_icon="📊")

    # Custom CSS (remains largely the same)
    st.markdown("""
    <style>
    /* ... (your existing CSS, ensure it's generic and not TSLA specific) ... */
    .main-header { font-size: 2.5rem !important; font-weight: 700 !important; color: #1E88E5 !important; margin-bottom: 0.3rem !important; }
    .sub-header { font-size: 1.5rem !important; font-weight: 400 !important; color: #78909C !important; margin-bottom: 2rem !important; font-style: italic; }
    .section-header { font-size: 1.8rem !important; font-weight: 600 !important; color: #26A69A !important; border-bottom: 1px solid #26A69A; padding-bottom: 0.5rem; margin-bottom: 1.5rem !important; }
    .ai-response { background-color: #f0f8ff; border-left: 5px solid #1E88E5; padding: 20px; border-radius: 0 10px 10px 0; margin: 15px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .ai-response p { font-size: 1.05rem; line-height: 1.6; margin-bottom: 12px; }
    .ai-response-header { display: flex; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #e1e4e8; padding-bottom: 10px; }
    .ai-response-header h3 { margin: 0; font-size: 1.4rem; color: #1E88E5; }
    .ai-response-header svg { width: 24px; height: 24px; margin-right: 10px; }
    .question-box { border: 1px solid #78909C; border-radius: 10px; padding: 1px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: box-shadow 0.3s ease; }
    .question-box:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .ai-analysis-intro { background: linear-gradient(135deg, rgba(38, 166, 154, 0.05), rgba(30, 136, 229, 0.05)); padding: 20px; border-radius: 10px; margin-bottom: 25px; border: 1px solid rgba(30, 136, 229, 0.1); }
    div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(90deg, #1E88E5, #26A69A); color: white; font-weight: bold; padding: 0.75rem 1.25rem; border-radius: 30px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; margin-top: 10px; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { box-shadow: 0 6px 8px rgba(0,0,0,0.15); transform: translateY(-2px); }
    .chart-legend { background-color: rgba(19, 23, 34, 0.7); border-radius: 5px; padding: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">📊 Financial Data Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Candlestick Analysis with AI-Powered Insights</p>', unsafe_allow_html=True)

    st.sidebar.header("📁 Configuration")
    csv_file = st.sidebar.file_uploader(
        "Upload OHLC CSV Data", type="csv",
        help="Upload your CSV file. Expected columns: Timestamp, Open, High, Low, Close. Optional: Volume, Direction, Support, Resistance."
    )

    enable_sr_processing = st.sidebar.checkbox("Enable Support/Resistance Analysis", value=True,
                                             help="If checked, will look for Support/Resistance columns and attempt to draw bands.")
    raw_support_col_name = RAW_SUPPORT_COL_DEFAULT
    raw_resistance_col_name = RAW_RESISTANCE_COL_DEFAULT

    if enable_sr_processing:
        st.sidebar.markdown("##### S/R Column Names (Case-Sensitive)")
        raw_support_col_name = st.sidebar.text_input("Raw Support List Column", value=RAW_SUPPORT_COL_DEFAULT,
                                                     help="Name of the column in your CSV containing lists of support levels, e.g., '[150.0, 150.5]'")
        raw_resistance_col_name = st.sidebar.text_input("Raw Resistance List Column", value=RAW_RESISTANCE_COL_DEFAULT,
                                                        help="Name of the column in your CSV containing lists of resistance levels, e.g., '[160.0, 160.5]'")


    if not csv_file:
        st.info("👆 Please upload your CSV data file to begin analysis.")
        st.markdown("""
        **Expected CSV Format:**
        - Timestamp column (e.g., 'timestamp', 'Date')
        - OHLC columns (e.g., 'Open', 'High', 'Low', 'Close')
        - **Optional:**
            - 'Volume' column
            - 'Direction' column (e.g., 'LONG', 'SHORT', 'None')
            - Support list column (e.g., named 'Support', containing text like "[150.0, 150.5]")
            - Resistance list column (e.g., named 'Resistance', containing text like "[160.0, 160.5]")
        """)
        return

    df = pd.DataFrame()
    summary = {}
    sr_data_available_for_charting = False

    with st.spinner("🔄 Loading and processing data..."):
        df, sr_data_available_for_charting = load_and_prepare_data(
            csv_file, COLUMN_MAP,
            process_sr_data=enable_sr_processing,
            raw_support_col=raw_support_col_name,
            raw_resistance_col=raw_resistance_col_name
        )
        if df.empty:
            # Error messages are shown in load_and_prepare_data
            return
        
        # Pass sr_data_available_for_charting to create_data_summary
        summary = create_data_summary(df, sr_data_available_for_charting)
        st.success(f"✅ Data loaded successfully! {summary.get('total_records', 0)} records processed.")

    model = configure_gemini_api()
    available_columns_for_ai = list(df.columns) if not df.empty else []


    chart_tab, ai_tab, data_tab = st.tabs(["📊 Interactive Chart", "🤖 AI Analysis", "📋 Data Overview"])

    with chart_tab:
        st.markdown('<p class="section-header">Data Candlestick Chart</p>', unsafe_allow_html=True)
        if df.empty:
            st.warning("No data to display in chart.")
        else:
            # Metric cards (generic)
            # ... (similar structure as before, using summary data) ...
            st.markdown('<div style="display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 20px;">', unsafe_allow_html=True)
            # Simplified metric cards for brevity
            st.markdown(f'''<div class="stat-card" style="flex:1; margin:5px; border-left: 5px solid #26A69A;">Latest Close: <b>${summary.get('price_stats',{}).get('latest_close',0):.2f}</b></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="stat-card" style="flex:1; margin:5px; border-left: 5px solid #64B5F6;">Price Range: <b>${summary.get('price_stats',{}).get('lowest_price',0):.2f} - ${summary.get('price_stats',{}).get('highest_price',0):.2f}</b></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="stat-card" style="flex:1; margin:5px; border-left: 5px solid #FFD54F;">Total Records: <b>{summary.get('total_records',0)}</b></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="stat-card" style="flex:1; margin:5px; border-left: 5px solid #BA68C8;">Date Range: <b>{summary.get('date_range',{}).get('start','N/A')} to {summary.get('date_range',{}).get('end','N/A')}</b></div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


            chart_opts = create_chart_config()
            chart_bg_color = chart_opts['layout']['background']['color']
            candle_data = df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
            
            # Create markers if 'direction' column exists
            markers = create_markers(df) if 'direction' in df.columns else []
            
            chart_series_components = []

            # Conditional S/R Bands
            if sr_data_available_for_charting and enable_sr_processing:
                # Ensure these columns were actually created and are not all NaN
                if all(col in df.columns for col in ['support_low', 'support_high', 'res_low', 'res_high']) and \
                   not df[['support_low', 'support_high', 'res_low', 'res_high']].isnull().all().all():
                    
                    support_upper, support_lower_mask = create_floating_band_series_components(
                        df, 'support_low', 'support_high', '#26A69A', chart_bg_color, 'Support'
                    )
                    resistance_upper, resistance_lower_mask = create_floating_band_series_components(
                        df, 'res_low', 'res_high', '#EF5350', chart_bg_color, 'Resistance'
                    )
                    if resistance_upper and resistance_lower_mask:
                        chart_series_components.extend([resistance_upper, resistance_lower_mask])
                    if support_upper and support_lower_mask:
                        chart_series_components.extend([support_upper, support_lower_mask])
                else:
                    st.warning("S/R processing was enabled, but valid band data could not be generated for the chart.")
                    sr_data_available_for_charting = False # Update flag if bands can't be drawn

            candle_series = {
                'type': 'Candlestick', 'data': candle_data,
                'options': {'upColor': '#26a69a', 'downColor': '#ef5350', 'wickUpColor': '#26a69a', 'wickDownColor': '#ef5350'},
                'markers': markers
            }
            chart_series_components.append(candle_series) # Candles always on top or after bands

            charts_data = [{'chart': chart_opts, 'series': chart_series_components}]
            renderLightweightCharts(charts_data, key='ohlc_chart_generic')

            # Conditional Legend
            legend_html = """<div class="chart-legend"><h4 style="margin-top: 0; color: #FFF; font-size: 1.2rem;">📊 Chart Legend</h4>"""
            if markers: # If direction data and markers exist
                legend_html += """
                    <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                        <div style="flex: 1; min-width: 180px; margin: 5px;">
                            <p style="margin: 5px 0;">🟢 <span style="color: #26A69A; font-weight: bold;">Green Arrow (↑)</span>: LONG signal</p>
                            <p style="margin: 5px 0;">🔴 <span style="color: #EF5350; font-weight: bold;">Red Arrow (↓)</span>: SHORT signal</p>
                            <p style="margin: 5px 0;">🟡 <span style="color: #FFD700; font-weight: bold;">Yellow Circle</span>: Neutral/Other signal</p>
                        </div>"""
            if sr_data_available_for_charting and enable_sr_processing: # If S/R bands are shown
                legend_html += """
                        <div style="flex: 1; min-width: 180px; margin: 5px;">
                            <p style="margin: 5px 0;">🟢 <span style="color: #26A69A; font-weight: bold;">Green Band</span>: Support zone</p>
                            <p style="margin: 5px 0;">🔴 <span style="color: #EF5350; font-weight: bold;">Red Band</span>: Resistance zone</p>
                        </div>"""
            if not markers and not (sr_data_available_for_charting and enable_sr_processing):
                 legend_html += "<p style='color:#CCC; margin:5px 0;'>Candlestick data shown.</p>"

            if markers or (sr_data_available_for_charting and enable_sr_processing) : # Close the flex div if it was opened
                 legend_html += "</div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

    with ai_tab:
        st.markdown('<p class="section-header">🤖 AI-Powered Data Analysis</p>', unsafe_allow_html=True)
        if not model:
            st.warning("⚠️ Please configure your Gemini API key to use the AI agent.")
        elif df.empty:
            st.warning("⚠️ No data loaded for AI analysis.")
        else:
            st.markdown("""
            <div class="ai-analysis-intro">
                <h3 style="margin-top: 0; color: #1E88E5; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
                    Ask Anything About Your Data
                </h3>
                <p style="margin-bottom: 0; font-size: 1.05rem;">This AI assistant can analyze your uploaded data and answer specific questions about trends, metrics, and patterns.</p>
            </div>""", unsafe_allow_html=True)

            st.markdown('<h3 style="color: #26A69A; font-size: 1.3rem; margin-bottom: 15px; display: flex; align-items: center;"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg> Example Questions:</h3>', unsafe_allow_html=True)
            template_questions = [ # Generic examples
                "How many data points are there in 2023?", "Average close price in Q 특징?",
                "What's the max high between Jan 1st and March 31st, 2024?", "Compare volume for LONG vs SHORT days (if direction/volume exist)",
                "Show total bullish candles in 2023 (if direction exists)", "What was the highest price in the dataset?",
                "Count how many days had a price above 200", "Calculate average close price by month in 2023"
            ]
            cols = st.columns(2)
            # ... (Template question button rendering - similar to before, ensure keys are unique) ...
            for i, question_text in enumerate(template_questions):
                col = cols[i % 2]
                button_key = f"template_btn_generic_{i}"
                if col.button(f"{question_text}", key=button_key, use_container_width=True):
                    st.session_state.question_input = question_text
                st.markdown(f"""
                <style>
                div[data-testid="stButton"] > button[data-testid="{button_key}"] {{
                    background: linear-gradient(135deg, #f8f9fa, #f1f3f5); border-left: 4px solid #1E88E5;
                    padding: 12px !important; margin: 8px 0 !important; border-radius: 6px !important;
                    transition: transform 0.2s ease, box-shadow 0.2s ease; font-size: 0.95rem !important;
                    text-align: left !important; font-weight: normal !important; color: #333 !important;
                    display: block; width: 100%; border: 1px solid #ddd;
                }}
                div[data-testid="stButton"] > button[data-testid="{button_key}"]:hover {{
                    transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                </style>""", unsafe_allow_html=True)


            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            # ... (User question input area - similar to before) ...
            st.markdown('''<div style="display: flex; align-items: center; margin-bottom: 8px; padding: 10px 10px 0 10px;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg><label style="font-weight: 500; color: #1E88E5;">Ask your question about the data:</label></div>''', unsafe_allow_html=True)
            user_question = st.text_area("Ask your question:", value=st.session_state.get('question_input', ''), height=100, placeholder="e.g., What was the average closing price last month?", key="question_input_area_generic", label_visibility="collapsed")
            st.session_state.question_input = user_question
            st.markdown('</div>', unsafe_allow_html=True)


            if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True, key="get_analysis_button_generic"):
                current_question = st.session_state.get('question_input', '').strip()
                if current_question:
                    with st.spinner("🧠 AI is analyzing..."):
                        response_text = get_ai_response(model, df, summary, current_question, available_columns_for_ai)
                        # ... (Display AI response - similar to before) ...
                        st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                        st.markdown('''<div class="ai-response-header"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg><h3>AI Analysis Result</h3></div>''', unsafe_allow_html=True)
                        paragraphs = response_text.split('\n\n')
                        for i, paragraph in enumerate(paragraphs):
                            if paragraph.strip():
                                style = "font-weight: 500; font-size: 1.1rem; color: #333;" if i == 0 else ""
                                st.markdown(f"<p style='{style}'>{paragraph}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        if 'chat_history' not in st.session_state: st.session_state.chat_history = []
                        st.session_state.chat_history.append({'question': current_question, 'response': response_text, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                else:
                    st.warning("Please enter a question first!")
            
            if st.button("🗑️ Clear Question Input", key="clear_question_button_generic"):
                st.session_state.question_input = ""
                st.rerun()

            if st.session_state.get('chat_history', []):
                st.subheader("📝 Recent Analysis History:")
                # ... (Chat history display - similar to before) ...
                for chat in reversed(st.session_state.chat_history[-5:]):
                    with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Response:**\n{chat['response']}")
                if st.button("🗑️ Clear History", key="clear_history_button_main_generic"):
                    st.session_state.chat_history = []
                    st.rerun()


    with data_tab:
        st.markdown('<p class="section-header">📋 Data Overview</p>', unsafe_allow_html=True)
        if df.empty:
            st.warning("No data to display.")
        else:
            # ... (Data summary and sample display - similar to before, using generic summary) ...
            st.subheader("Dataset Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Total Records:** {summary.get('total_records',0)}")
                st.markdown(f"**Date Range:** {summary.get('date_range',{}).get('start','N/A')} to {summary.get('date_range',{}).get('end','N/A')}")
            with col2:
                st.markdown(f"**Avg. Close Price:** ${summary.get('price_stats',{}).get('avg_close',0):.2f}")
                st.markdown(f"**Highest Price:** ${summary.get('price_stats',{}).get('highest_price',0):.2f}")
            
            if 'direction' in df.columns:
                st.subheader("Trading Signal Distribution (if 'direction' column exists)")
                direction_counts_dict = summary.get('direction_counts', {})
                if direction_counts_dict:
                    # ... (display direction_df as before) ...
                    direction_counts = pd.Series(direction_counts_dict)
                    if not direction_counts.empty :
                        direction_df = pd.DataFrame({
                            'Signal': direction_counts.index, 'Count': direction_counts.values,
                            'Percentage': (direction_counts.values / direction_counts.sum() * 100).round(1) if direction_counts.sum() > 0 else 0
                        })
                        st.dataframe(direction_df.style.format({'Percentage': '{:.1f}%'}), use_container_width=True)
                    else:
                        st.markdown("No signal data to display from summary.")
                else:
                    st.markdown("No signal data in summary.")


            st.subheader("Data Sample (First 10 Rows)")
            # Display all available columns in the sample, or a curated list
            display_cols = [col for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'direction'] if col in df.columns]
            st.dataframe(df[display_cols].head(10), use_container_width=True)
            
            csv_data = df.to_csv(index=False)
            st.download_button(label="⬇️ Download Processed Data", data=csv_data, file_name="processed_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()