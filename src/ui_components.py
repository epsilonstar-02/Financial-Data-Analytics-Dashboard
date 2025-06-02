# trading_dashboard/ui_components.py
import streamlit as st
import pandas as pd

def load_custom_css():
    """Load custom CSS styles for the dashboard."""
    st.markdown("""
    <style>
    /* General Enhancements */
    body {
        font-family: 'Roboto', sans-serif; /* More modern font */
    }
    .main .block-container {
        padding-top: 2rem; /* More space at the top */
    }

    /* Headers */
    .main-header {
        font-size: 2.8rem !important; /* Slightly larger */
        font-weight: 700 !important;
        color: #1E88E5 !important;
        margin-bottom: 0.2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle shadow */
    }
    .sub-header {
        font-size: 1.6rem !important; /* Slightly larger */
        font-weight: 300 !important; /* Lighter weight */
        color: #546E7A !important; /* Softer color */
        margin-bottom: 2.5rem !important;
        font-style: italic;
    }
    .section-header {
        font-size: 2rem !important; /* Consistent larger size */
        font-weight: 600 !important;
        color: #00796B !important; /* Teal accent */
        border-bottom: 2px solid #00796B;
        padding-bottom: 0.6rem;
        margin-top: 2rem !important;
        margin-bottom: 1.8rem !important;
    }

    /* Metric Cards */
    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); /* Responsive grid */
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2C3E50, #34495E); /* Darker, sophisticated gradient */
        color: #ECF0F1;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        border-left: 6px solid var(--card-accent-color, #3498DB); /* Accent color variable */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .metric-card p:first-child { /* Label */
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
        color: #BDC3C7;
        font-weight: 500;
    }
    .metric-card p:last-child { /* Value */
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: var(--card-accent-color-text, #FFFFFF);
    }
    
    /* AI Interaction Elements */
    .ai-analysis-intro {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.08), rgba(38, 166, 154, 0.08));
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(30, 136, 229, 0.2);
    }
    .ai-response-container {
        background-color: #f9f9f9; /* Lighter background for response */
        border-left: 6px solid #1E88E5;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        color: #333;
    }
    .ai-response-container h3 { /* AI Response title */
        margin-top: 0;
        color: #1E88E5;
        font-size: 1.5rem;
    }
    .ai-response-container p {
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    .ai-response-container p:last-child {
        margin-bottom: 0;
    }
    .question-input-box { /* For st.text_area parent */
        border: 1px solid #B0BEC5;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background-color: #FFFFFF;
    }
    .question-input-box:focus-within {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
    }
    
    /* Template Question Buttons (Improved) */
    .stButton>button.template-question-btn {
        background-color: #F5F5F5;
        color: #333;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #1E88E5;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 500;
        width: 100%;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button.template-question-btn:hover {
        background-color: #E0E0E0;
        border-color: #1E88E5;
        color: #1E88E5;
        transform: translateX(2px);
    }

    /* Action Buttons (Primary) */
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #1E88E5, #26A69A);
        color: white;
        font-weight: bold;
        padding: 0.7rem 1.5rem; /* Standardized padding */
        border-radius: 30px; /* Pill shape */
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .stButton>button[kind="secondary"] { /* For clear buttons */
        border-radius: 30px;
        padding: 0.6rem 1.2rem;
    }
    
    /* Chart Legend */
    .chart-legend-box {
        background-color: rgba(30, 40, 50, 0.85); /* Darker, slightly more opaque */
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .chart-legend-box h4 {
        margin-top: 0; color: #E0E0E0; font-size: 1.3rem; border-bottom: 1px solid #4F5B66; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .chart-legend-box p { color: #B0BEC5; margin: 0.4rem 0; font-size: 0.95rem; }
    .chart-legend-box span { font-weight: bold; }

    /* Styling for st.tabs */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px; /* Spacing between tabs */
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px; /* Tab height */
        white-space: pre-wrap; /* Wrap text if needed */
		background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 8px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00796B; /* Active tab color */
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def display_main_header():
    """Displays the main and sub-headers of the dashboard."""
    st.markdown('<p class="main-header">📈 TSLA Trading Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Candlestick Analysis with AI-Powered Insights</p>', unsafe_allow_html=True)

def display_sidebar_upload():
    """Displays the file uploader in the sidebar."""
    st.sidebar.header("📁 Data Configuration")
    csv_file = st.sidebar.file_uploader(
        "Upload TSLA CSV Data",
        type="csv",
        help="Upload your TSLA_data - Sheet1.csv file. Required columns: timestamp, Open, High, Low, Close, direction, Support, Resistance."
    )
    return csv_file

def display_file_upload_instructions():
    """Displays instructions when no file is uploaded."""
    st.info("👆 Please upload your TSLA CSV file to begin analysis.")
    st.markdown("""
    **Expected CSV Format:**
    - `timestamp`: Date/time of the data point (e.g., YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).
    - `Open`, `High`, `Low`, `Close`: Standard OHLC price data (numeric).
    - `direction`: Trading signal (e.g., 'LONG', 'SHORT', or 'None'/empty for neutral).
    - `Support`: String representation of a list of support prices (e.g., "[150.0, 150.5]").
    - `Resistance`: String representation of a list of resistance prices (e.g., "[160.0, 160.5]").
    """)

def display_metric_cards(summary_data):
    """Displays key metrics in styled cards."""
    if not summary_data:
        return

    cards_html = '<div class="metric-card-grid">'
    
    latest_close = summary_data['price_stats'].get('latest_close', 0)
    price_low = summary_data['price_stats'].get('lowest_price', 0)
    price_high = summary_data['price_stats'].get('highest_price', 0)
    total_records = summary_data.get('total_records', 0)
    start_date = summary_data['date_range'].get('start', 'N/A')
    end_date = summary_data['date_range'].get('end', 'N/A')

    cards_html += f"""
    <div class="metric-card" style="--card-accent-color: #2ECC71; --card-accent-color-text: #2ECC71;">
        <p>Latest Close</p><p>${latest_close:.2f}</p>
    </div>
    <div class="metric-card" style="--card-accent-color: #3498DB; --card-accent-color-text: #3498DB;">
        <p>Price Range (Low - High)</p><p>${price_low:.2f} - ${price_high:.2f}</p>
    </div>
    <div class="metric-card" style="--card-accent-color: #F39C12; --card-accent-color-text: #F39C12;">
        <p>Total Records</p><p>{total_records}</p>
    </div>
    <div class="metric-card" style="--card-accent-color: #9B59B6; --card-accent-color-text: #9B59B6;">
        <p>Date Range</p><p style="font-size: 1.3rem;">{start_date} to {end_date}</p>
    </div>
    """
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

def display_chart_legend():
    """Displays the legend for the candlestick chart."""
    st.markdown("""
    <div class="chart-legend-box">
        <h4>📊 Chart Legend</h4>
        <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
            <div style="min-width: 180px; margin: 5px;">
                <p><span style="color: #26A69A;">🟢 Arrow (↑) / Candle Up</span>: LONG signal / Price Increase</p>
                <p><span style="color: #EF5350;">🔴 Arrow (↓) / Candle Down</span>: SHORT signal / Price Decrease</p>
            </div>
            <div style="min-width: 180px; margin: 5px;">
                <p><span style="color: #FFD700;">🟡 Circle</span>: Neutral signal</p>
                <p><span style="color: #26A69A;">🟩 Green Band</span>: Support Zone</p>
                <p><span style="color: #EF5350;">🟥 Red Band</span>: Resistance Zone</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_ai_tab_intro_and_templates(template_questions):
    """Displays intro for AI tab and template question buttons."""
    st.markdown("""
    <div class="ai-analysis-intro">  
        <h3 style="margin-top: 0; color: #1E88E5; display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 12px;">
                <path d="M12 2a10 10 0 0 0-7.52 16.94l-1.38 1.38A1 1 0 0 0 4 22h16a1 1 0 0 0 .9-1.68l-1.38-1.38A10 10 0 0 0 12 2zM8.5 11.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0zm7 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0zM12 17c-2 0-3-1-3-1s1-1 3-1 3 1 3 1-1 1-3 1z"/>
            </svg>
            Ask Your AI Financial Analyst
        </h3>
        <p style="margin-bottom: 0; font-size: 1.05rem;">Leverage AI to dissect your TSLA stock data. Ask specific questions about trends, metrics, and patterns to gain deeper insights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="color: #00796B; font-size: 1.4rem; margin-bottom: 1rem; display: flex; align-items: center;"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M21.73 18.23l-4.5-4.5A7.5 7.5 0 109 16.5a7.38 7.38 0 005.73-2.53l4.5 4.5a1.5 1.5 0 002.12-.01 1.5 1.5 0 00.01-2.13zM10 16a6 6 0 110-12 6 6 0 010 12z"/><line x1="10" y1="13" x2="10" y2="7"></line><line x1="7" y1="10" x2="13" y2="10"></line></svg> Quick Prompts:</h3>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, question in enumerate(template_questions):
        col = cols[i % 2]
        # Apply class for CSS targeting
        if col.button(question, key=f"template_q_{i}", help=f"Click to use: '{question}'"):
            st.session_state.question_input = question # Ensure this key is used for the text_area
            # No need to rerun here, text_area will update. If you want immediate processing, add rerun.
            st.rerun() # Added to make template selection more interactive


def display_ai_response(response_text):
    """Displays the AI's response in a styled container."""
    st.markdown('<div class="ai-response-container">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="display: flex; align-items: center;"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 15c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm0-8c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1s1 .45 1 1v1c0 .55-.45 1-1 1z"></path></svg>AI Analysis Result</h3>', unsafe_allow_html=True)
    
    # Try to split response by common section headers used in the prompt.
    # This is a heuristic and depends on the LLM consistently using these.
    response_parts = response_text.split("DETAILED ANALYSIS:")
    direct_answer = response_parts[0].replace("DIRECT ANSWER:", "").strip()
    
    st.markdown(f"<p><strong>{direct_answer}</strong></p>", unsafe_allow_html=True)

    if len(response_parts) > 1:
        detailed_parts = response_parts[1].split("INTERPRETATION:")
        detailed_analysis = detailed_parts[0].strip()
        st.markdown(f"<p style='margin-top:1em;'><em>Detailed Analysis:</em><br>{detailed_analysis}</p>", unsafe_allow_html=True)
        
        if len(detailed_parts) > 1:
            interpretation = detailed_parts[1].strip()
            st.markdown(f"<p style='margin-top:1em;'><em>Interpretation:</em><br>{interpretation}</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_chat_history():
    """Displays recent AI analysis history."""
    if st.session_state.get('chat_history', []):
        st.markdown("---")
        st.subheader("📝 Recent Analysis History (Last 5)")
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"❓: {chat['question'][:60]}... ({chat['timestamp']})"):
                st.markdown(f"**Your Question:**\n\n{chat['question']}")
                st.markdown(f"**AI Response:**")
                # Use the same display logic for consistency
                display_ai_response(chat['response']) # This will nest the styled response
        
        if st.button("🗑️ Clear Full Chat History", key="clear_full_history_button"):
            st.session_state.chat_history = []
            st.rerun()


def display_data_overview_tab(df, summary):
    """Displays the content for the Data Overview tab."""
    st.markdown('<p class="section-header">📋 TSLA Data Overview</p>', unsafe_allow_html=True)
    if df.empty:
        st.warning("No data loaded to display.")
        return

    st.subheader("📊 Dataset Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Records", value=summary.get('total_records', 'N/A'))
        st.metric(label="Average Close Price", value=f"${summary['price_stats'].get('avg_close', 0):.2f}")
    with col2:
        st.metric(label="Highest Price", value=f"${summary['price_stats'].get('highest_price', 0):.2f}")
        st.metric(label="Lowest Price", value=f"${summary['price_stats'].get('lowest_price', 0):.2f}")
    
    st.metric(label="Date Range", value=f"{summary['date_range'].get('start', 'N/A')} to {summary['date_range'].get('end', 'N/A')}")
    
    st.subheader("🚦 Trading Signal Distribution")
    direction_counts = pd.Series(summary.get('direction_counts', {}))
    if not direction_counts.empty:
        direction_df = pd.DataFrame({
            'Signal': direction_counts.index,
            'Count': direction_counts.values
        }).set_index('Signal')
        
        # Add percentage column
        direction_df['Percentage'] = (direction_df['Count'] / direction_df['Count'].sum() * 100)
        
        st.dataframe(direction_df.style.format({'Count': '{:,}', 'Percentage': '{:.1f}%'}), use_container_width=True)
        
        # Optionally, a bar chart for visual appeal
        # st.bar_chart(direction_df['Count'])

    st.subheader("📜 Data Sample (First 10 Rows)")
    st.dataframe(
        df[['timestamp', 'open', 'high', 'low', 'close', 'direction']].head(10),
        use_container_width=True,
        hide_index=True
    )

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Processed Data (CSV)",
        data=csv_data,
        file_name="tsla_processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )