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
        
        ohlc_map = {
            'open': ['Open', 'OPEN', 'open'],
            'high': ['High', 'HIGH', 'high'],
            'low': ['Low', 'LOW', 'low'],
            'close': ['Close', 'CLOSE', 'close'],
            'timestamp': ['timestamp', 'Date', 'TIME', 'date']
        }
        
        for standard_name, variants in ohlc_map.items():
            match = next((col for col in df.columns if col in variants), None)
            if not match:
                st.error(f"❌ Missing required column: {standard_name}. Found columns: {list(df.columns)}")
                st.stop()
            
            if standard_name != 'timestamp':
                df[standard_name] = pd.to_numeric(df[match], errors='coerce')
            else:
                df[standard_name] = pd.to_datetime(df[match], errors='coerce')
        
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
        
        df = improved_support_resistance_calculation(df)
        
        df['time'] = (df['timestamp'].astype('int64') // 10**9).astype(int)
        
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'timestamp', 'support_low', 'support_high', 'res_low', 'res_high'])
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
            'latest_close': df['close'].iloc[-1] if not df.empty else 0
        },
        'direction_counts': df['direction'].value_counts().to_dict(),
        'support_resistance_stats': {
            'avg_support_levels': df['support_list'].apply(len).mean() if not df.empty else 0,
            'avg_resistance_levels': df['resistance_list'].apply(len).mean() if not df.empty else 0
        }
    }
    return summary

# --- Chart Configuration ---
def create_chart_config():
    """Create professional chart configuration"""
    return {
        "height": 720,
        "layout": {
            "background": {"type": "solid", "color": "#131722"}, # This is the chart_bg_color
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

def create_floating_band_series_components(df, low_col_name, high_col_name, band_color_hex, chart_bg_color_hex, base_name):
    """
    Creates two AreaSeries components to simulate a floating band.
    Returns a tuple: (upper_fill_series_definition, lower_mask_series_definition)
    """
    upper_band_data = []
    lower_band_data = []

    for _, row in df.iterrows():
        low_val = row[low_col_name]
        high_val = row[high_col_name]
        time_val = row['time']

        if (pd.notna(low_val) and pd.notna(high_val) and
            low_val > 0 and high_val > 0 and
            low_val <= high_val and
            abs(high_val - low_val) > 0.01): # Ensure meaningful band height

            upper_band_data.append({'time': time_val, 'value': float(high_val)})
            lower_band_data.append({'time': time_val, 'value': float(low_val)})

    if not upper_band_data or not lower_band_data:
        return None, None

    # Series 1: Fills from the high_col_name down to baseline with band color
    upper_fill_series = {
        'type': 'Area',
        'data': upper_band_data,
        'options': {
            'topColor': f'{band_color_hex}20',  # Semi-transparent
            'bottomColor': f'{band_color_hex}40', # Semi-transparent
            'lineColor': 'rgba(0,0,0,0)', # Invisible line
            'lineWidth': 0,
            'priceScaleId': '', 
            'title': base_name, # This will appear in the legend
            'crosshairMarkerVisible': False,
            'lastValueVisible': False,
            'visible': True,
        }
    }

    # Series 2: Fills from the low_col_name down to baseline with background color (mask)
    lower_mask_series = {
        'type': 'Area',
        'data': lower_band_data,
        'options': {
            'topColor': chart_bg_color_hex,   # Opaque background color
            'bottomColor': chart_bg_color_hex, # Opaque background color
            'lineColor': 'rgba(0,0,0,0)', # Invisible line
            'lineWidth': 0,
            'priceScaleId': '',
            'title': '', # Empty title, hopefully hides from legend or is inconspicuous
            'crosshairMarkerVisible': False,
            'lastValueVisible': False,
            'visible': True,
        }
    }
    
    return upper_fill_series, lower_mask_series


def calculate_default_bounds(df, window=20):
    """Calculate default support/resistance based on recent price action"""
    df = df.copy()
    df['default_support'] = df['low'].rolling(window=window, min_periods=1).min()
    df['default_resistance'] = df['high'].rolling(window=window, min_periods=1).max()
    return df

def fill_missing_values(df):
    """Fill missing support/resistance values using forward fill and default values"""
    if 'default_support' not in df.columns: # Ensure default bounds are calculated if not present
        df = calculate_default_bounds(df)
    
    support_cols = ['support_low', 'support_high']
    resistance_cols = ['res_low', 'res_high']
    
    # Forward fill with previous valid values
    for cols_list in [support_cols, resistance_cols]:
        for col in cols_list:
            if col in df.columns: # Ensure column exists before trying to fill
                 df[col] = df[col].fillna(method='ffill')
    
    # Fill any remaining NAs with default values
    df['support_low'] = df['support_low'].fillna(df['default_support'])
    df['support_high'] = df['support_high'].fillna(df['default_support']) # Often support_high might be support_low + spread, or another logic. For now, using default_support.
                                                                        # A better default for support_high might be default_support + a small spread, or just ensure it's >= support_low.
                                                                        # For simplicity, using default_support for both if totally missing, but ensure high >= low later.
    df['res_low'] = df['res_low'].fillna(df['default_resistance']) # Similar logic for res_low
    df['res_high'] = df['res_high'].fillna(df['default_resistance'])

    # Ensure high >= low after filling
    df['support_high'] = df[['support_high', 'support_low']].max(axis=1)
    df['res_high'] = df[['res_high', 'res_low']].max(axis=1)
    
    return df

# --- AI Agent Functions ---
def improved_support_resistance_calculation(df, window=20):
    """
    Enhanced support/resistance calculation with robust missing data handling.
    """
    def safe_bounds(price_list):
        if not price_list or len(price_list) == 0:
            return None, None
        valid_prices = [p for p in price_list if isinstance(p, (int, float)) and p > 0]
        if not valid_prices:
            return None, None
        return min(valid_prices), max(valid_prices)
    
    bounds = df[['support_list', 'resistance_list']].apply(
        lambda x: pd.Series({
            'support_low': safe_bounds(x['support_list'])[0],
            'support_high': safe_bounds(x['support_list'])[1],
            'res_low': safe_bounds(x['resistance_list'])[0],
            'res_high': safe_bounds(x['resistance_list'])[1]
        }), 
        axis=1
    )
    
    df = pd.concat([df, bounds], axis=1)
    df = calculate_default_bounds(df, window=window) # Calculates 'default_support' and 'default_resistance'
    df = fill_missing_values(df) # Uses default bounds and ffill
    
    # Final check for any NaNs that might have slipped through complex fills, use default price levels
    for col in ['support_low', 'support_high', 'res_low', 'res_high']:
        if df[col].isna().any():
            if 'support' in col:
                df[col] = df[col].fillna(df['default_support'])
            else: # resistance
                df[col] = df[col].fillna(df['default_resistance'])
    
    # Ensure logical order of low/high for bands after all filling
    df['support_high'] = df[['support_high', 'support_low']].max(axis=1)
    df['res_high'] = df[['res_high', 'res_low']].max(axis=1)
    
    # Avoid issues if default_support/resistance were themselves NaN (e.g., very short df)
    # This is a last resort if previous fills left NaNs
    final_fallback_cols = ['support_low', 'support_high', 'res_low', 'res_high']
    for col in final_fallback_cols:
        if df[col].isna().any():
            # If 'close' is available, use it as a last resort. Otherwise, might need to drop row or use 0.
            df[col] = df[col].fillna(df['close']) 


    df = df.drop(columns=['default_support', 'default_resistance'], errors='ignore')
    return df

def analyze_data_for_question(df, question):
    """Analyze data based on the specific question and return relevant subset"""
    question_lower = question.lower()
    analysis_data = {}
    
    if df.empty: # Handle empty DataFrame early
        return {"error": "No data available for analysis."}

    if '2023' in question_lower:
        df_2023 = df[df['timestamp'].dt.year == 2023].copy()
        if not df_2023.empty:
            analysis_data['2023_data'] = {
                'total_days': len(df_2023),
                'bullish_days': len(df_2023[df_2023['direction'] == 'LONG']),
                'bearish_days': len(df_2023[df_2023['direction'] == 'SHORT']),
                'neutral_days': len(df_2023[df_2023['direction'].isna() | (df_2023['direction'] == 'None')]),
                'price_range': f"${df_2023['low'].min():.2f} - ${df_2023['high'].max():.2f}",
                'avg_close': f"${df_2023['close'].mean():.2f}",
                'sample_bullish_dates': df_2023[df_2023['direction'] == 'LONG']['timestamp'].dt.strftime('%Y-%m-%d').head(5).tolist()
            }
        else:
            analysis_data['2023_data'] = "No data available for 2023."

    if 'price' in question_lower or 'highest' in question_lower or 'lowest' in question_lower:
        analysis_data['price_analysis'] = {
            'highest_price': f"${df['high'].max():.2f}",
            'lowest_price': f"${df['low'].min():.2f}",
            'highest_date': df.loc[df['high'].idxmax()]['timestamp'].strftime('%Y-%m-%d'),
            'lowest_date': df.loc[df['low'].idxmin()]['timestamp'].strftime('%Y-%m-%d'),
            'price_volatility': f"{((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%"
        }
    
    if 'support' in question_lower or 'resistance' in question_lower:
        all_support = [price for sublist in df['support_list'].dropna() for price in sublist if isinstance(price, (int, float)) and price > 0]
        all_resistance = [price for sublist in df['resistance_list'].dropna() for price in sublist if isinstance(price, (int, float)) and price > 0]
        
        analysis_data['support_resistance'] = {
            'strongest_support': f"${min(all_support):.2f}" if all_support else "N/A",
            'strongest_resistance': f"${max(all_resistance):.2f}" if all_resistance else "N/A", # Min for resistance? Should be max. Max for support? Should be min for "strongest"
            'avg_support_levels_per_day': f"{df['support_list'].apply(len).mean():.1f}",
            'avg_resistance_levels_per_day': f"{df['resistance_list'].apply(len).mean():.1f}",
            'support_range_from_lists': f"${min(all_support):.2f} - ${max(all_support):.2f}" if all_support else "N/A",
            'resistance_range_from_lists': f"${min(all_resistance):.2f} - ${max(all_resistance):.2f}" if all_resistance else "N/A"
        }
    
    if 'signal' in question_lower or 'long' in question_lower or 'short' in question_lower:
        analysis_data['signal_analysis'] = {
            'total_signals': len(df), # This is total records, not just signals
            'long_signals': len(df[df['direction'] == 'LONG']),
            'short_signals': len(df[df['direction'] == 'SHORT']),
            'neutral_signals': len(df[df['direction'].isna() | (df['direction'] == 'None')]),
            'long_percentage': f"{len(df[df['direction'] == 'LONG']) / len(df) * 100:.1f}%" if len(df) > 0 else "N/A",
            'short_percentage': f"{len(df[df['direction'] == 'SHORT']) / len(df) * 100:.1f}%" if len(df) > 0 else "N/A"
        }
    
    if 'quarter' in question_lower or 'month' in question_lower:
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        monthly_stats = df.groupby('month').agg(
            avg_close_price=('close', 'mean'),
            num_long_signals=('direction', lambda x: (x == 'LONG').sum())
        ).reset_index()
        monthly_stats['avg_close_price'] = monthly_stats['avg_close_price'].round(2)
        
        analysis_data['temporal_analysis'] = {
            'monthly_avg_price': monthly_stats.set_index('month')['avg_close_price'].to_dict(),
            'monthly_bullish_days': monthly_stats.set_index('month')['num_long_signals'].to_dict()
        }
    
    return analysis_data

def create_context_prompt(df, summary, user_question, specific_analysis):
    """Create comprehensive context for the AI agent with specific data analysis"""
    sample_data_str = "No sample data available."
    if not df.empty:
        sample_data_str = df.head(5)[['timestamp', 'open', 'high', 'low', 'close', 'direction']].to_string()
    
    context = f"""
You are a professional financial data analyst specializing in TSLA stock analysis. You have access to detailed TSLA trading data and specific analysis results.

DATASET OVERVIEW:
- Total Records: {summary.get('total_records', 'N/A')}
- Date Range: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}
- Price Statistics: {summary.get('price_stats', 'N/A')}
- Direction Distribution: {summary.get('direction_counts', 'N/A')}

SAMPLE DATA:
{sample_data_str}

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
        specific_analysis = analyze_data_for_question(df, question)
        context = create_context_prompt(df, summary, question, specific_analysis)
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        st.error(f"AI Error: {e}")
        return f"❌ Error generating response: {str(e)}"

# --- Streamlit App ---
def main():
    st.title("📈 TSLA Trading Analytics Dashboard")
    st.markdown("### Advanced Candlestick Analysis with AI-Powered Insights")
    
    st.sidebar.header("📁 Configuration")
    
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
        - Support column (list of support prices, e.g., "[150.0, 150.5]")
        - Resistance column (list of resistance prices, e.g., "[160.0, 160.5]")
        """)
        return
    
    with st.spinner("🔄 Loading and processing data..."):
        df = load_data(csv_file)
        if df is None or df.empty:
            st.error("Failed to load or process data. Please check the file and its contents (especially Support/Resistance columns).")
            return
        summary = create_data_summary(df)
    
    model = configure_gemini()
    
    chart_tab, ai_tab, data_tab = st.tabs(["📊 Interactive Chart", "🤖 AI Analysis", "📋 Data Overview"])
    
    with chart_tab:
        st.header("TSLA Candlestick Chart with Trading Signals")
        
        if df.empty:
            st.warning("No data to display in chart.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Latest Price", f"${summary['price_stats']['latest_close']:.2f}")
            with col2:
                st.metric("Price Range", f"${summary['price_stats']['lowest_price']:.2f} - ${summary['price_stats']['highest_price']:.2f}")
            with col3:
                st.metric("Total Records", summary['total_records'])
            with col4:
                st.metric("Date Range", f"{summary['date_range']['start']} to {summary['date_range']['end']}")
            
            chart_opts = create_chart_config()
            chart_bg_color = chart_opts['layout']['background']['color'] # Get background color for mask

            candle_data = df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
            markers = create_markers(df)
            
            # Create band series components
            support_upper, support_lower_mask = create_floating_band_series_components(
                df, 'support_low', 'support_high', '#26A69A', chart_bg_color, 'Support'
            )
            resistance_upper, resistance_lower_mask = create_floating_band_series_components(
                df, 'res_low', 'res_high', '#EF5350', chart_bg_color, 'Resistance'
            )
            
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
            
            # Assemble series in correct draw order for layering
            # Bands are drawn first, then candles on top.
            # Resistance band -> Support band -> Candlesticks
            chart_series_components = []
            if resistance_upper and resistance_lower_mask:
                chart_series_components.extend([resistance_upper, resistance_lower_mask])
            if support_upper and support_lower_mask:
                chart_series_components.extend([support_upper, support_lower_mask])
            chart_series_components.append(candle_series)
            
            charts_data = [{
                'chart': chart_opts,
                'series': chart_series_components
            }]
            
            renderLightweightCharts(charts_data, key='tsla_enhanced_floating_bands')
            
            st.markdown("""
            **Chart Legend:**
            - 🟢 **Green Arrow (↑)**: LONG signal (below candle)
            - 🔴 **Red Arrow (↓)**: SHORT signal (above candle)
            - 🟡 **Yellow Circle**: Neutral/No signal
            - 🟢 **Green Band**: Support zone
            - 🔴 **Red Band**: Resistance zone
            """)
    
    with ai_tab:
        st.header("🤖 AI-Powered Data Analysis")
        
        if not model:
            st.warning("⚠️ Please configure your Gemini API key to use the AI agent")
        elif df.empty:
            st.warning("⚠️ No data loaded for AI analysis.")
        else:
            st.subheader("💡 Try These Questions:")
            template_questions = [
                "How many days in 2023 was TSLA bullish (LONG direction)?",
                "What was the highest and lowest price of TSLA in the dataset?",
                "Analyze the correlation between support/resistance levels and price movements",
                "What patterns do you see in the trading signals over time?",
                "Compare the performance in different quarters",
                "What were the strongest support and resistance levels based on the provided lists?",
                "How often did price break through resistance levels?", # This might be hard for current analyze_data_for_question
                "Analyze the frequency of LONG vs SHORT signals"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(template_questions):
                col = cols[i % 2]
                if col.button(f"📝 {question}", key=f"template_{i}"):
                    st.session_state.selected_question = question
                    st.session_state.question_input = question # Also update text_area
                    st.rerun()

            user_question = st.text_area(
                "🔍 Ask your question about TSLA data:",
                value=st.session_state.get('question_input', ''), # Use question_input for persistence
                height=100,
                placeholder="e.g., How many bullish days were there in 2023?",
                key="question_input_area" # Different key from session_state.question_input if needed, but can be same
            )
            st.session_state.question_input = user_question # Sync text_area back to session state

            if st.button("🚀 Get AI Analysis", type="primary"):
                current_question = st.session_state.get('question_input', '').strip()
                if current_question:
                    with st.spinner("🧠 AI is analyzing the data..."):
                        response = get_ai_response(model, df, summary, current_question)
                        st.markdown("### 📊 AI Analysis Result:")
                        st.markdown(response)
                        
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        st.session_state.chat_history.append({
                            'question': current_question,
                            'response': response, # Store full response
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                else:
                    st.warning("Please enter a question first!")
            
            if st.button("🗑️ Clear Question Input"):
                st.session_state.question_input = ""
                st.rerun()
            
            if st.session_state.get('chat_history', []):
                st.subheader("📝 Recent Analysis History:")
                # Display latest 5, newest first
                for chat in reversed(st.session_state.chat_history[-5:]):
                    with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Response:**\n{chat['response']}") # Display full response here
                
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    with data_tab:
        st.header("📋 Data Overview & Statistics")
        if df.empty:
            st.warning("No data to display.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Price Statistics")
                st.json(summary['price_stats'])
                st.subheader("📊 Direction Distribution")
                if summary['direction_counts']:
                    st.bar_chart(pd.Series(summary['direction_counts']))
                else:
                    st.write("No direction data available.")
            with col2:
                st.subheader("🎯 Support/Resistance Stats (from lists)")
                st.json(summary['support_resistance_stats'])
                st.subheader("📅 Date Range")
                st.json(summary['date_range'])
            
            st.subheader("🔍 Data Preview (First 10 Rows with calculated band values)")
            st.dataframe(
                df[['timestamp', 'open', 'high', 'low', 'close', 'direction', 
                    'Support', 'Resistance', 'support_low', 'support_high', 'res_low', 'res_high']].head(10),
                use_container_width=True
            )
            
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Processed Data",
                data=csv_data,
                file_name="tsla_processed_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()