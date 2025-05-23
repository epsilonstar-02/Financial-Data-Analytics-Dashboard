import streamlit as st
import pandas as pd
import ast
import google.generativeai as genai
from streamlit_lightweight_charts import renderLightweightCharts
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="TSLA Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Configure Gemini API
@st.cache_resource
def configure_gemini():
    """Configure Gemini API with API key from .env file"""
    # Try to get API key from environment variables
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
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {str(e)}")
        return None

# --- Data Loading & Processing ---
@st.cache_data
def load_data(csv_file) -> pd.DataFrame:
    """Load and process TSLA data with enhanced error handling"""
    try:
        df = pd.read_csv(csv_file, parse_dates=["timestamp"])
        
        # Enhanced column detection
        ohlc_map = {
            'open': ['Open', 'OPEN', 'open'],
            'high': ['High', 'HIGH', 'high'],
            'low': ['Low', 'LOW', 'low'],
            'close': ['Close', 'CLOSE', 'close'],
            'timestamp': ['timestamp', 'Date', 'TIME', 'date']
        }
        
        # Map columns to standard names
        for standard_name, variants in ohlc_map.items():
            match = next((col for col in df.columns if col in variants), None)
            if not match:
                st.error(f"❌ Missing required column: {standard_name}. Found columns: {list(df.columns)}")
                st.stop()
            
            if standard_name != 'timestamp':
                df[standard_name] = pd.to_numeric(df[match], errors='coerce')
            else:
                df[standard_name] = pd.to_datetime(df[match], errors='coerce')
        
        # Parse Support/Resistance lists safely
        def safe_literal_eval(val):
            try:
                if pd.isna(val) or val == '':
                    return []
                if isinstance(val, str):
                    return ast.literal_eval(val)
                return val if isinstance(val, list) else []
            except:
                return []
        
        df['support_list'] = df['Support'].apply(safe_literal_eval)
        df['resistance_list'] = df['Resistance'].apply(safe_literal_eval)
        
        # Compute support/resistance bounds using enhanced calculation
        df = improved_support_resistance_calculation(df)
        
        # Debug output for data quality
        if not df.empty:
            debug_info = debug_support_resistance(df)
            if debug_info['overlapping_bands'] > 0:
                st.warning(f"⚠️ Found {debug_info['overlapping_bands']} rows with overlapping support/resistance bands")
            
            # Log data quality metrics
            st.sidebar.metric("Data Quality", f"{debug_info['valid_support']/len(df)*100:.1f}% valid support bands")
            st.sidebar.metric("", f"{debug_info['valid_resistance']/len(df)*100:.1f}% valid resistance bands")
        
        # Convert timestamp to UNIX seconds
        df['time'] = (df['timestamp'].astype('int64') // 10**9).astype(int)
        
        # Clean data
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'timestamp'])
        df = df.drop_duplicates('time', keep='last')
        df = df.sort_values('time')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def create_data_summary(df):
    """Create a comprehensive data summary for the AI agent"""
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min().strftime('%Y-%m-%d'),
            'end': df['timestamp'].max().strftime('%Y-%m-%d')
        },
        'price_stats': {
            'highest_price': df['high'].max(),
            'lowest_price': df['low'].min(),
            'avg_close': df['close'].mean(),
            'latest_close': df['close'].iloc[-1]
        },
        'direction_counts': df['direction'].value_counts().to_dict(),
        'support_resistance_stats': {
            'avg_support_levels': df['support_list'].apply(len).mean(),
            'avg_resistance_levels': df['resistance_list'].apply(len).mean()
        }
    }
    return summary

# --- Chart Configuration ---
def create_chart_config():
    """Create professional chart configuration"""
    return {
        "height": 720,
        "layout": {
            "background": {"type": "solid", "color": "#131722"},
            "textColor": "#D9D9D9",
            "fontSize": 12,
            "fontFamily": 'Consolas, monospace'
        },
        "grid": {
            "vertLines": {"color": "rgba(100, 100, 100, 0.2)", "style": 2},
            "horzLines": {"color": "rgba(100, 100, 100, 0.2)", "style": 2}
        },
        "rightPriceScale": {
            "borderVisible": False,
            "scaleMargins": {"top": 0.1, "bottom": 0.2},
            "entireTextOnly": True
        },
        "timeScale": {
            "borderVisible": False,
            "timeVisible": True,
            "rightOffset": 12
        },
    }

def create_markers(df):
    """Create trading direction markers"""
    markers = []
    for _, row in df.iterrows():
        if row['direction'] == 'LONG':
            markers.append({
                'time': row['time'],
                'position': 'belowBar',
                'color': '#26A69A',
                'shape': 'arrowUp',
                'size': 1.5,
                'text': 'L'
            })
        elif row['direction'] == 'SHORT':
            markers.append({
                'time': row['time'],
                'position': 'aboveBar',
                'color': '#EF5350',
                'shape': 'arrowDown',
                'size': 1.5,
                'text': 'S'
            })
        else:  # None direction
            markers.append({
                'time': row['time'],
                'position': 'inBar',
                'color': '#FFD700',
                'shape': 'circle',
                'size': 1,
                'text': 'N'
            })
    return markers

def create_band_series(df, low_col, high_col, color, name):
    """Create support/resistance band series with improved validation"""
    valid_data = []
    
    for _, row in df.iterrows():
        low_val = row[low_col]
        high_val = row[high_col]
        
        # Enhanced validation
        if (pd.notna(low_val) and pd.notna(high_val) and 
            low_val > 0 and high_val > 0 and  # Ensure positive values
            low_val <= high_val and  # Ensure logical order
            abs(high_val - low_val) > 0.01):  # Minimum meaningful range
            
            valid_data.append({
                'time': row['time'],
                'value': float(low_val),  # Ensure float type
                'highValue': float(high_val)
            })
    
    if not valid_data:
        # Return empty series if no valid data
        return {
            'type': 'Area',
            'data': [],
            'options': {
                'topColor': f'{color}30',
                'bottomColor': f'{color}50',
                'lineWidth': 1,
                'lineStyle': 0,
                'priceScaleId': '',
                'title': name,
                'visible': False  # Hide if no data
            }
        }
    
    # Return series with valid data
    return {
        'type': 'Area',
        'data': valid_data,
        'options': {
            'topColor': f'{color}20',  # More transparent
            'bottomColor': f'{color}40',
            'lineWidth': 1,
            'lineStyle': 2,  # Dashed line
            'priceScaleId': '',
            'title': name,
            'crosshairMarkerVisible': False,
            'lastValueVisible': False
        }
    }

def calculate_default_bounds(df, window=20):
    """Calculate default support/resistance based on recent price action"""
    # Use recent lows for support, highs for resistance
    df = df.copy()
    df['default_support'] = df['low'].rolling(window=window, min_periods=1).min()
    df['default_resistance'] = df['high'].rolling(window=window, min_periods=1).max()
    return df

def fill_missing_values(df):
    """Fill missing support/resistance values using forward fill and default values"""
    # Ensure we have the default bounds calculated
    if 'default_support' not in df.columns:
        df = calculate_default_bounds(df)
    
    # Forward fill support/resistance values
    support_cols = ['support_low', 'support_high']
    resistance_cols = ['res_low', 'res_high']
    
    # Forward fill with previous valid values
    for cols in [support_cols, resistance_cols]:
        df[cols] = df[cols].fillna(method='ffill')
    
    # Fill any remaining NAs with default values
    df['support_low'] = df['support_low'].fillna(df['default_support'])
    df['support_high'] = df['support_high'].fillna(df['default_support'])
    df['res_low'] = df['res_low'].fillna(df['default_resistance'])
    df['res_high'] = df['res_high'].fillna(df['default_resistance'])
    
    return df

# --- AI Agent Functions ---
def improved_support_resistance_calculation(df, window=20):
    """
    Enhanced support/resistance calculation with robust missing data handling.
    
    Implements a three-tier strategy:
    1. Forward fill missing values from previous valid data
    2. Use interpolation for gaps when possible
    3. Fall back to price-based default bands when needed
    
    Args:
        df: DataFrame with price data and support/resistance lists
        window: Lookback window for calculating default bounds (default: 20)
    
    Returns:
        DataFrame with support/resistance bounds and filled missing values
    """
    def safe_bounds(price_list):
        """Safely calculate min/max bounds from a price list with validation"""
        if not price_list or len(price_list) == 0:
            return None, None
        
        # Filter out invalid prices
        valid_prices = [p for p in price_list if isinstance(p, (int, float)) and p > 0]
        
        if not valid_prices:
            return None, None
            
        return min(valid_prices), max(valid_prices)
    
    # Calculate initial bounds
    bounds = df[['support_list', 'resistance_list']].apply(
        lambda x: pd.Series({
            'support_low': safe_bounds(x['support_list'])[0],
            'support_high': safe_bounds(x['support_list'])[1],
            'res_low': safe_bounds(x['resistance_list'])[0],
            'res_high': safe_bounds(x['resistance_list'])[1]
        }), 
        axis=1
    )
    
    # Add bounds to dataframe
    df = pd.concat([df, bounds], axis=1)
    
    # Calculate default bounds based on price action
    df = calculate_default_bounds(df, window=window)
    
    # Fill missing values using our strategy
    df = fill_missing_values(df)
    
    # Ensure no NaN values remain
    for col in ['support_low', 'support_high', 'res_low', 'res_high']:
        if df[col].isna().any():
            # If we still have NaNs, fall back to default values
            if 'support' in col:
                df[col] = df[col].fillna(df['default_support'])
            else:
                df[col] = df[col].fillna(df['default_resistance'])
    
    # Clean up temporary columns
    df = df.drop(columns=['default_support', 'default_resistance'], errors='ignore')
    
    return df

def debug_support_resistance(df):
    """Debug support/resistance data quality"""
    print("=== Support/Resistance Debug Info ===")
    
    # Check data quality
    total_rows = len(df)
    support_valid = df[['support_low', 'support_high']].notna().all(axis=1).sum()
    resistance_valid = df[['res_low', 'res_high']].notna().all(axis=1).sum()
    
    print(f"Total rows: {total_rows}")
    print(f"Valid support bands: {support_valid} ({support_valid/total_rows*100:.1f}%)")
    print(f"Valid resistance bands: {resistance_valid} ({resistance_valid/total_rows*100:.1f}%)")
    
    # Sample some support/resistance data
    print("\n=== Sample Support/Resistance Data ===")
    sample_data = df[['timestamp', 'close', 'support_list', 'resistance_list', 
                      'support_low', 'support_high', 'res_low', 'res_high']].head(10)
    
    for _, row in sample_data.iterrows():
        print(f"\nDate: {row['timestamp'].strftime('%Y-%m-%d')}")
        print(f"Close: ${row['close']:.2f}")
        print(f"Support List: {row['support_list']}")
        print(f"Support Range: ${row['support_low']:.2f} - ${row['support_high']:.2f}" 
              if pd.notna(row['support_low']) else "No support")
        print(f"Resistance List: {row['resistance_list']}")
        print(f"Resistance Range: ${row['res_low']:.2f} - ${row['res_high']:.2f}" 
              if pd.notna(row['res_low']) else "No resistance")
    
    # Check for overlapping bands
    overlap_check = df[(df['support_high'] > df['res_low']) & 
                       df['support_low'].notna() & df['res_high'].notna()]
    
    if len(overlap_check) > 0:
        print(f"\n⚠️  WARNING: {len(overlap_check)} rows have overlapping support/resistance bands")
    
    return {
        'total_rows': total_rows,
        'valid_support': support_valid,
        'valid_resistance': resistance_valid,
        'overlapping_bands': len(overlap_check)
    }

def analyze_data_for_question(df, question):
    """Analyze data based on the specific question and return relevant subset"""
    question_lower = question.lower()
    
    # Create a comprehensive data analysis based on question type
    analysis_data = {}
    
    # Year-based questions
    if '2023' in question_lower:
        df_2023 = df[df['timestamp'].dt.year == 2023].copy()
        analysis_data['2023_data'] = {
            'total_days': len(df_2023),
            'bullish_days': len(df_2023[df_2023['direction'] == 'LONG']),
            'bearish_days': len(df_2023[df_2023['direction'] == 'SHORT']),
            'neutral_days': len(df_2023[df_2023['direction'].isna() | (df_2023['direction'] == 'None')]),
            'price_range': f"${df_2023['low'].min():.2f} - ${df_2023['high'].max():.2f}",
            'avg_close': f"${df_2023['close'].mean():.2f}",
            'sample_bullish_dates': df_2023[df_2023['direction'] == 'LONG']['timestamp'].dt.strftime('%Y-%m-%d').head(5).tolist()
        }
    
    # Price analysis questions
    if 'price' in question_lower or 'highest' in question_lower or 'lowest' in question_lower:
        analysis_data['price_analysis'] = {
            'highest_price': f"${df['high'].max():.2f}",
            'lowest_price': f"${df['low'].min():.2f}",
            'highest_date': df.loc[df['high'].idxmax()]['timestamp'].strftime('%Y-%m-%d'),
            'lowest_date': df.loc[df['low'].idxmin()]['timestamp'].strftime('%Y-%m-%d'),
            'price_volatility': f"{((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%"
        }
    
    # Support/Resistance analysis
    if 'support' in question_lower or 'resistance' in question_lower:
        # Flatten all support/resistance levels
        all_support = [price for sublist in df['support_list'].dropna() for price in sublist if price]
        all_resistance = [price for sublist in df['resistance_list'].dropna() for price in sublist if price]
        
        analysis_data['support_resistance'] = {
            'strongest_support': f"${min(all_support):.2f}" if all_support else "N/A",
            'strongest_resistance': f"${max(all_resistance):.2f}" if all_resistance else "N/A",
            'avg_support_levels_per_day': f"{len(all_support) / len(df):.1f}",
            'avg_resistance_levels_per_day': f"{len(all_resistance) / len(df):.1f}",
            'support_range': f"${min(all_support):.2f} - ${max(all_support):.2f}" if all_support else "N/A",
            'resistance_range': f"${min(all_resistance):.2f} - ${max(all_resistance):.2f}" if all_resistance else "N/A"
        }
    
    # Signal frequency analysis
    if 'signal' in question_lower or 'long' in question_lower or 'short' in question_lower:
        analysis_data['signal_analysis'] = {
            'total_signals': len(df),
            'long_signals': len(df[df['direction'] == 'LONG']),
            'short_signals': len(df[df['direction'] == 'SHORT']),
            'neutral_signals': len(df[df['direction'].isna() | (df['direction'] == 'None')]),
            'long_percentage': f"{len(df[df['direction'] == 'LONG']) / len(df) * 100:.1f}%",
            'short_percentage': f"{len(df[df['direction'] == 'SHORT']) / len(df) * 100:.1f}%"
        }
    
    # Monthly/Quarterly analysis
    if 'quarter' in question_lower or 'month' in question_lower:
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        monthly_stats = df.groupby('month').agg({
            'close': 'mean',
            'direction': lambda x: (x == 'LONG').sum()
        }).round(2)
        
        analysis_data['temporal_analysis'] = {
            'monthly_avg_price': monthly_stats['close'].to_dict(),
            'monthly_bullish_days': monthly_stats['direction'].to_dict()
        }
    
    return analysis_data

def create_context_prompt(df, summary, user_question, specific_analysis):
    """Create comprehensive context for the AI agent with specific data analysis"""
    
    # Sample data points for context
    sample_data = df.head(5)[['timestamp', 'open', 'high', 'low', 'close', 'direction']].to_string()
    
    context = f"""
You are a professional financial data analyst specializing in TSLA stock analysis. You have access to detailed TSLA trading data and specific analysis results.

DATASET OVERVIEW:
- Total Records: {summary['total_records']}
- Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}
- Price Statistics: {summary['price_stats']}
- Direction Distribution: {summary['direction_counts']}

SAMPLE DATA:
{sample_data}

SPECIFIC ANALYSIS FOR YOUR QUESTION:
{specific_analysis}

COLUMN DEFINITIONS:
- 'direction': Trading signal (LONG=bullish, SHORT=bearish, None/NaN=neutral)
- 'Support': List of support price levels for each day
- 'Resistance': List of resistance price levels for each day
- OHLC: Standard Open, High, Low, Close prices

USER QUESTION: {user_question}

INSTRUCTIONS:
- Use the specific analysis data provided above to answer the question directly
- Provide concrete numbers and insights from the actual dataset
- Be specific and factual, not generic
- If asking about 2023 data, use the 2023_data section from the analysis
- Reference actual dates, prices, and counts from the data
- Provide actionable insights based on the real data patterns

Please provide a detailed, data-driven analysis that directly answers the user's question using the specific analysis results provided.
"""
    return context

def get_ai_response(model, df, summary, question):
    """Get response from Gemini AI with specific data analysis"""
    try:
        # First, analyze the data based on the specific question
        specific_analysis = analyze_data_for_question(df, question)
        
        # Create context with both general summary and specific analysis
        context = create_context_prompt(df, summary, question, specific_analysis)
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# --- Streamlit App ---
def main():
    st.title("📈 TSLA Trading Analytics Dashboard")
    st.markdown("### Advanced Candlestick Analysis with AI-Powered Insights")
    
    # Sidebar configuration
    st.sidebar.header("📁 Configuration")
    
    # File uploader
    csv_file = st.sidebar.file_uploader(
        "Upload TSLA CSV Data", 
        type="csv",
        help="Upload your TSLA_data - Sheet1.csv file"
    )
    
    if not csv_file:
        st.info("👆 Please upload your TSLA CSV file to begin analysis")
        st.markdown("""
        **Expected CSV Format:**
        - timestamp, Open, High, Low, Close columns
        - direction column (LONG/SHORT/None)
        - Support column (list of support prices)
        - Resistance column (list of resistance prices)
        """)
        return
    
    # Load and process data
    with st.spinner("🔄 Loading and processing data..."):
        df = load_data(csv_file)
        summary = create_data_summary(df)
    
    # Configure Gemini
    model = configure_gemini()
    
    # Create tabs
    chart_tab, ai_tab, data_tab = st.tabs(["📊 Interactive Chart", "🤖 AI Analysis", "📋 Data Overview"])
    
    with chart_tab:
        st.header("TSLA Candlestick Chart with Trading Signals")
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Price", f"${summary['price_stats']['latest_close']:.2f}")
        with col2:
            st.metric("Price Range", f"${summary['price_stats']['lowest_price']:.2f} - ${summary['price_stats']['highest_price']:.2f}")
        with col3:
            st.metric("Total Records", summary['total_records'])
        with col4:
            st.metric("Date Range", f"{summary['date_range']['start']} to {summary['date_range']['end']}")
        
        # Create chart
        chart_opts = create_chart_config()
        candle_data = df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
        markers = create_markers(df)
        
        # Create bands
        support_band = create_band_series(df, 'support_low', 'support_high', '#26A69A', 'Support')
        resistance_band = create_band_series(df, 'res_low', 'res_high', '#EF5350', 'Resistance')
        
        # Candlestick series
        candle_series = {
            'type': 'Candlestick',
            'data': candle_data,
            'options': {
                'upColor': '#26a69a',
                'downColor': '#ef5350',
                'wickUpColor': '#26a69a',
                'wickDownColor': '#ef5350',
            },
            'markers': markers,
        }
        
        # Render chart
        charts = [{
            'chart': chart_opts,
            'series': [candle_series, support_band, resistance_band]
        }]
        
        renderLightweightCharts(charts, key='tsla_enhanced')
        
        # Legend
        st.markdown("""
        **Chart Legend:**
        - 🟢 **Green Arrow (↑)**: LONG signal (below candle)
        - 🔴 **Red Arrow (↓)**: SHORT signal (above candle)
        - 🟡 **Yellow Circle**: Neutral/No signal
        - 🟢 **Green Band**: Support levels
        - 🔴 **Red Band**: Resistance levels
        """)
    
    with ai_tab:
        st.header("🤖 AI-Powered Data Analysis")
        
        if not model:
            st.warning("⚠️ Please configure your Gemini API key to use the AI agent")
            return
        
        # Template questions
        st.subheader("💡 Try These Questions:")
        template_questions = [
            "How many days in 2023 was TSLA bullish (LONG direction)?",
            "What was the highest and lowest price of TSLA in the dataset?",
            "Analyze the correlation between support/resistance levels and price movements",
            "What patterns do you see in the trading signals over time?",
            "Compare the performance in different quarters",
            "What were the strongest support and resistance levels?",
            "How often did price break through resistance levels?",
            "Analyze the frequency of LONG vs SHORT signals"
        ]
        
        # Create buttons for template questions
        cols = st.columns(2)
        
        for i, question in enumerate(template_questions):
            col = cols[i % 2]
            if col.button(f"📝 {question}", key=f"template_{i}"):
                st.session_state.selected_question = question
        
        # Question input
        user_question = st.text_area(
            "🔍 Ask your question about TSLA data:",
            value=st.session_state.get('selected_question', ''),
            height=100,
            placeholder="e.g., How many bullish days were there in 2023?",
            key="question_input"
        )
        
        if st.button("🚀 Get AI Analysis", type="primary"):
            current_question = st.session_state.get('question_input', '').strip()
            if current_question:
                with st.spinner("🧠 AI is analyzing the data..."):
                    response = get_ai_response(model, df, summary, current_question)
                    st.markdown("### 📊 AI Analysis Result:")
                    st.markdown(response)
                    
                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({
                        'question': current_question,
                        'response': response[:200] + "..." if len(response) > 200 else response,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            else:
                st.warning("Please enter a question first!")
        
        # Chat history display
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Clear question button
        if st.button("🗑️ Clear Question"):
            st.session_state.selected_question = ""
            st.rerun()
        
        if st.session_state.chat_history:
            st.subheader("📝 Recent Analysis History:")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Response:** {chat['response']}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with data_tab:
        st.header("📋 Data Overview & Statistics")
        
        # Data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Price Statistics")
            st.json(summary['price_stats'])
            
            st.subheader("📊 Direction Distribution")
            st.bar_chart(pd.Series(summary['direction_counts']))
        
        with col2:
            st.subheader("🎯 Support/Resistance Stats")
            st.json(summary['support_resistance_stats'])
            
            st.subheader("📅 Date Range")
            st.json(summary['date_range'])
        
        # Raw data preview
        st.subheader("🔍 Data Preview (First 10 Rows)")
        st.dataframe(
            df[['timestamp', 'open', 'high', 'low', 'close', 'direction', 'Support', 'Resistance']].head(10),
            use_container_width=True
        )
        
        # Download processed data
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Processed Data",
            data=csv_data,
            file_name="tsla_processed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()