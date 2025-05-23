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
        return genai.GenerativeModel('gemini-2.0-flash')
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

def parse_question_with_gemini(model, df, question):
    """Use Gemini to parse a natural language question into structured JSON format"""
    if not model:
        return None
    
    try:
        # Create a prompt that instructs Gemini to parse the question into structured JSON
        prompt = f"""
        You are a specialized AI financial analyst for Tesla stock data. 
        Parse the following question about Tesla stock data and convert it to a structured JSON format.
        
        QUESTION: {question}
        
        OUTPUT FORMAT: Return a valid JSON object with the following fields:
        - action: (string) The query action type ('aggregate', 'filter', 'describe', 'compare', 'count')
        - columns: (array) List of column names relevant to the query (e.g. ["volume"], ["close"])
        - filters: (object) Dictionary with filter conditions like time ranges or direction filters
          e.g. {{
            "timestamp": {{"start": "2024-03-01", "end": "2024-03-31"}},
            "direction": "LONG"
          }}
        - metric: (string) Statistical operation like 'mean', 'sum', 'max', 'min', 'count' (if applicable)
        - group_by: (string, optional) Field to group by (e.g. "direction" or "quarter")
        
        AVAILABLE COLUMNS IN DATASET:
        - timestamp: Date in format YYYY-MM-DD
        - open: Opening price
        - high: Highest price
        - low: Lowest price
        - close: Closing price
        - volume: Trading volume
        - direction: Trading signal (LONG = bullish, SHORT = bearish, None = neutral)
        - support_low: Lower support level
        - support_high: Higher support level
        - res_low: Lower resistance level
        - res_high: Higher resistance level
        
        EXAMPLE 1:
        Question: "How many LONG days were there in 2024?"
        Response: {{"action": "count", "columns": [], "filters": {{"timestamp": {{"start": "2024-01-01", "end": "2024-12-31"}}, "direction": "LONG"}}, "metric": "count", "group_by": null}}
        
        EXAMPLE 2:
        Question: "Average close price in Q3 2023?"
        Response: {{"action": "aggregate", "columns": ["close"], "filters": {{"timestamp": {{"start": "2023-07-01", "end": "2023-09-30"}}}}, "metric": "mean", "group_by": null}}
        
        IMPORTANT: 
        - Use ISO date format (YYYY-MM-DD) for all dates
        - Use proper Python data types (strings, numbers, booleans)
        - Ensure the JSON is valid and properly formatted
        - If a field is not applicable, use null
        - Do not add any explanations or text outside the JSON object
        
        RESPONSE (JSON only):
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        # Remove markdown code blocks if present
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        import json
        parsed_json = json.loads(result)
        return parsed_json
        
    except Exception as e:
        st.error(f"Error parsing question: {e}")
        return None

def execute_structured_query(df, query_json):
    """Execute a structured query based on parsed JSON"""
    if df.empty or not query_json:
        return "No data available or invalid query structure."
    
    try:
        # Make a copy of the dataframe to avoid modifying the original
        filtered_df = df.copy()
        
        # Apply filters
        if 'filters' in query_json and query_json['filters']:
            filters = query_json['filters']
            
            # Apply timestamp filters
            if 'timestamp' in filters:
                timestamp_filter = filters['timestamp']
                if 'start' in timestamp_filter and timestamp_filter['start']:
                    start_date = pd.to_datetime(timestamp_filter['start'])
                    filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
                if 'end' in timestamp_filter and timestamp_filter['end']:
                    end_date = pd.to_datetime(timestamp_filter['end'])
                    filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
            
            # Apply direction filter
            if 'direction' in filters and filters['direction']:
                direction = filters['direction']
                filtered_df = filtered_df[filtered_df['direction'] == direction]
        
        # If no data matches filters
        if filtered_df.empty:
            return "No data found for the specified filters."
        
        # Get the action and metric
        action = query_json.get('action', '')
        metric = query_json.get('metric', '')
        columns = query_json.get('columns', [])
        group_by = query_json.get('group_by')
        
        # Execute the appropriate action
        result = {}
        
        if action == 'count':
            result['count'] = len(filtered_df)
            if columns:
                for col in columns:
                    if col in filtered_df.columns:
                        result[f'{col}_count'] = filtered_df[col].count()
        
        elif action == 'aggregate':
            if not columns:
                columns = ['close']  # Default to close price if no columns specified
            
            for col in columns:
                if col in filtered_df.columns:
                    if metric == 'mean':
                        result[f'avg_{col}'] = filtered_df[col].mean()
                    elif metric == 'sum':
                        result[f'sum_{col}'] = filtered_df[col].sum()
                    elif metric == 'max':
                        result[f'max_{col}'] = filtered_df[col].max()
                    elif metric == 'min':
                        result[f'min_{col}'] = filtered_df[col].min()
                    else:  # Default to mean if metric not specified
                        result[f'avg_{col}'] = filtered_df[col].mean()
        
        elif action == 'compare':
            if group_by and group_by in filtered_df.columns:
                # Special case for temporal grouping
                if group_by == 'quarter' and 'timestamp' in filtered_df.columns:
                    filtered_df['quarter'] = filtered_df['timestamp'].dt.quarter
                    group_by = 'quarter'
                elif group_by == 'month' and 'timestamp' in filtered_df.columns:
                    filtered_df['month'] = filtered_df['timestamp'].dt.month
                    group_by = 'month'
                elif group_by == 'year' and 'timestamp' in filtered_df.columns:
                    filtered_df['year'] = filtered_df['timestamp'].dt.year
                    group_by = 'year'
                
                if not columns:
                    columns = ['close']  # Default
                
                for col in columns:
                    if col in filtered_df.columns:
                        grouped = filtered_df.groupby(group_by)[col]
                        if metric == 'mean':
                            result[f'{col}_by_{group_by}'] = grouped.mean().to_dict()
                        elif metric == 'sum':
                            result[f'{col}_by_{group_by}'] = grouped.sum().to_dict()
                        elif metric == 'max':
                            result[f'{col}_by_{group_by}'] = grouped.max().to_dict()
                        elif metric == 'min':
                            result[f'{col}_by_{group_by}'] = grouped.min().to_dict()
                        elif metric == 'count':
                            result[f'{col}_count_by_{group_by}'] = grouped.count().to_dict()
                        else:  # Default to mean
                            result[f'{col}_by_{group_by}'] = grouped.mean().to_dict()
            
            # Special case for direction comparison
            elif group_by == 'direction':
                direction_counts = filtered_df['direction'].value_counts().to_dict()
                result['direction_counts'] = direction_counts
                
                if columns:
                    for col in columns:
                        if col in filtered_df.columns:
                            direction_stats = filtered_df.groupby('direction')[col].agg(['mean', 'max', 'min']).to_dict()
                            result[f'{col}_by_direction'] = direction_stats
        
        elif action == 'describe':
            if not columns:
                # Default to price columns
                columns = ['open', 'high', 'low', 'close']
            
            for col in columns:
                if col in filtered_df.columns:
                    col_stats = filtered_df[col].describe().to_dict()
                    result[f'{col}_stats'] = col_stats
        
        # Return metadata along with results
        result['metadata'] = {
            'filtered_rows': len(filtered_df),
            'total_rows': len(df),
            'date_range': {
                'start': filtered_df['timestamp'].min().strftime('%Y-%m-%d') if not filtered_df.empty else None,
                'end': filtered_df['timestamp'].max().strftime('%Y-%m-%d') if not filtered_df.empty else None
            }
        }
        
        # Convert any NumPy or Pandas specific types to Python native types for JSON serialization
        import json
        result_json = json.loads(json.dumps(result, default=str))
        
        return result_json
        
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return {"error": str(e)}

def format_query_results(model, question, query_results):
    """Format query results into a natural language response using Gemini with improved formatting"""
    if not model or not query_results:
        return "I couldn't analyze that question. Please try a different question."
    
    try:
        # Create a prompt for Gemini to summarize the results with clear formatting instructions
        prompt = f"""
        You are a specialized financial analyst for Tesla stock data. 
        Analyze the following query results and provide a well-formatted response that answers the user's question.
        
        USER QUESTION: {question}
        
        QUERY RESULTS: {query_results}
        
        FORMAT YOUR RESPONSE IN THREE CLEAR SECTIONS:
        
        1. DIRECT ANSWER: Start with a concise, direct answer to the user's question including specific numbers and metrics.
        
        2. DETAILED ANALYSIS: Provide additional context and details about the findings, using short, clear sentences.
        
        3. INTERPRETATION: End with a brief interpretation of what the numbers mean for TSLA stock.
        
        CRITICAL FORMATTING RULES:
        - Format large numbers with commas and prices with $ symbol
        - Format percentages properly (e.g., 24.5%)
        - Round decimal numbers to 2 places for readability
        - Use proper spacing between all words, numbers, and punctuation
        - NEVER run words together - each word must be separated by spaces
        - Use short sentences to avoid run-on text
        - After each sentence, add a space before starting the next one
        
        EXAMPLE OF PROPER FORMATTING:
        "In March 2024, Tesla's average closing price was $176.34. The stock ranged from a low of $162.51 to a high of $202.57.
        
        The average opening price was $176.85, with an average high of $179.70 and an average low of $173.73. The stock showed moderate volatility during this period.
        
        These figures suggest Tesla experienced a relatively stable trading month with slight upward momentum, maintaining investor confidence despite market fluctuations."
        """
        
        # Generate content with the improved prompt
        response = model.generate_content(prompt)
        
        # Return the formatted text
        return response.text
        
    except Exception as e:
        st.error(f"Error formatting results: {e}")
        # Fallback to regular response if structured output fails
        try:
            simple_prompt = f"Summarize these query results to answer: {question}\n\nResults: {query_results}"
            response = model.generate_content(simple_prompt)
            return response.text
        except:
            return f"Error generating response: Unable to format results properly."
    
    except Exception as e:
        st.error(f"Error formatting results: {e}")
        return f"Error generating response: {str(e)}"

def get_ai_response(model, df, summary, question):
    """Process a natural language question using the new structured approach"""
    try:
        # Step 1: Parse the question into structured JSON
        parsed_query = parse_question_with_gemini(model, df, question)
        
        if not parsed_query:
            # Fallback to the old approach if parsing fails
            specific_analysis = analyze_data_for_question(df, question)
            context = create_context_prompt(df, summary, question, specific_analysis)
            response = model.generate_content(context)
            return response.text
        
        # Step 2: Execute the structured query
        query_results = execute_structured_query(df, parsed_query)
        
        # Step 3: Format the results as natural language
        formatted_response = format_query_results(model, question, query_results)
        
        return formatted_response
    
    except Exception as e:
        st.error(f"AI Error: {e}")
        return f"❌ Error generating response: {str(e)}"

# --- Streamlit App ---
def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        margin-bottom: 0.3rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        color: #78909C !important;
        margin-bottom: 2rem !important;
        font-style: italic;
    }
    .stat-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #26A69A !important;
        border-bottom: 1px solid #26A69A;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }
    .ai-response {
        background-color: #f0f8ff;
        border-left: 5px solid #1E88E5;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .ai-response p {
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .ai-response-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #e1e4e8;
        padding-bottom: 10px;
    }
    .ai-response-header h3 {
        margin: 0;
        font-size: 1.4rem;
        color: #1E88E5;
    }
    .ai-response-header img {
        width: 24px;
        height: 24px;
        margin-right: 10px;
    }
    .question-box {
        border: 1px solid #78909C;
        border-radius: 10px;
        padding: 1px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }
    .question-box:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .template-card {
        background: linear-gradient(135deg, #f8f9fa, #f1f3f5);
        border-left: 4px solid #1E88E5;
        padding: 12px;
        margin: 8px 0;
        border-radius: 6px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
        font-size: 0.95rem;
    }
    .template-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .ai-analysis-intro {
        background: linear-gradient(135deg, rgba(38, 166, 154, 0.05), rgba(30, 136, 229, 0.05));
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border: 1px solid rgba(30, 136, 229, 0.1);
    }
    .analysis-button {
        background: linear-gradient(90deg, #1E88E5, #26A69A);
        color: white;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    .analysis-button:hover {
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .template-btn {
        text-align: left !important;
        margin: 3px 0 !important;
    }
    .chart-legend {
        background-color: rgba(19, 23, 34, 0.7);
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Dashboard Header
    st.markdown('<p class="main-header">📈 TSLA Trading Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Candlestick Analysis with AI-Powered Insights</p>', unsafe_allow_html=True)
    
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
        
        # Success message with animation
        st.success("✅ Data loaded successfully! " + str(summary['total_records']) + " records processed.")
    
    model = configure_gemini()
    
    chart_tab, ai_tab, data_tab = st.tabs(["📊 Interactive Chart", "🤖 AI Analysis", "📋 Data Overview"])
    
    with chart_tab:
        st.markdown('<p class="section-header">TSLA Candlestick Chart with Trading Signals</p>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data to display in chart.")
        else:
            # Enhanced metric cards with CSS styling
            st.markdown('<div style="display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 20px;">', unsafe_allow_html=True)
            
            # Latest Price Card
            st.markdown(f'''
            <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1E1E1E, #2A2A2A); margin: 5px; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 5px solid #26A69A;">
                <p style="margin: 0; color: #CCC; font-size: 0.9rem;">Latest Price</p>
                <p style="margin: 0; color: #26A69A; font-size: 1.8rem; font-weight: bold;">${summary['price_stats']['latest_close']:.2f}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Price Range Card
            st.markdown(f'''
            <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1E1E1E, #2A2A2A); margin: 5px; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 5px solid #64B5F6;">
                <p style="margin: 0; color: #CCC; font-size: 0.9rem;">Price Range</p>
                <p style="margin: 0; color: #64B5F6; font-size: 1.8rem; font-weight: bold;">${summary['price_stats']['lowest_price']:.2f} - ${summary['price_stats']['highest_price']:.2f}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Total Records Card
            st.markdown(f'''
            <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1E1E1E, #2A2A2A); margin: 5px; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 5px solid #FFD54F;">
                <p style="margin: 0; color: #CCC; font-size: 0.9rem;">Total Records</p>
                <p style="margin: 0; color: #FFD54F; font-size: 1.8rem; font-weight: bold;">{summary['total_records']}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Date Range Card
            st.markdown(f'''
            <div style="flex: 1; min-width: 150px; background: linear-gradient(135deg, #1E1E1E, #2A2A2A); margin: 5px; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 5px solid #BA68C8;">
                <p style="margin: 0; color: #CCC; font-size: 0.9rem;">Date Range</p>
                <p style="margin: 0; color: #BA68C8; font-size: 1.3rem; font-weight: bold;">{summary['date_range']['start']} to {summary['date_range']['end']}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
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
            <div class="chart-legend">
                <h4 style="margin-top: 0; color: #FFF; font-size: 1.2rem;">📊 Chart Legend</h4>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="flex: 1; min-width: 180px; margin: 5px;">
                        <p style="margin: 5px 0;">🟢 <span style="color: #26A69A; font-weight: bold;">Green Arrow (↑)</span>: LONG signal</p>
                        <p style="margin: 5px 0;">🔴 <span style="color: #EF5350; font-weight: bold;">Red Arrow (↓)</span>: SHORT signal</p>
                    </div>
                    <div style="flex: 1; min-width: 180px; margin: 5px;">
                        <p style="margin: 5px 0;">🟡 <span style="color: #FFD700; font-weight: bold;">Yellow Circle</span>: Neutral signal</p>
                        <p style="margin: 5px 0;">🟢 <span style="color: #26A69A; font-weight: bold;">Green Band</span>: Support zone</p>
                        <p style="margin: 5px 0;">🔴 <span style="color: #EF5350; font-weight: bold;">Red Band</span>: Resistance zone</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with ai_tab:
        st.markdown('<p class="section-header">🤖 AI-Powered Data Analysis</p>', unsafe_allow_html=True)
        
        if not model:
            st.warning("⚠️ Please configure your Gemini API key to use the AI agent")
        elif df.empty:
            st.warning("⚠️ No data loaded for AI analysis.")
        else:
            # Introduction with enhanced animation
            st.markdown("""
            <div class="ai-analysis-intro">  
                <h3 style="margin-top: 0; color: #1E88E5; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">  
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M12 16v-4"></path>
                        <path d="M12 8h.01"></path>
                    </svg>
                    Ask Anything About Your TSLA Data
                </h3>
                <p style="margin-bottom: 0; font-size: 1.05rem;">This AI-powered assistant can analyze your Tesla stock data and answer specific questions about trends, metrics, and patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 style="color: #26A69A; font-size: 1.3rem; margin-bottom: 15px; display: flex; align-items: center;"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg> Example Questions:</h3>', unsafe_allow_html=True)
            
            template_questions = [
                "How many LONG days were there in 2024?",
                "Average close price in Q3 2023?",
                "What's the max high between Jan and March 2024?",
                "Compare volume for LONG vs SHORT days",
                "Show total bullish candles in 2023",
                "What was the highest price in the dataset?",
                "Count how many days had a price above $200",
                "Calculate average close price by month in 2023"
            ]
            
            # Improved template cards
            cols = st.columns(2)
            for i, question in enumerate(template_questions):
                col = cols[i % 2]
                
                # Use a regular Streamlit button but with custom styling
                button_key = f"template_{i}"
                if col.button(f"{question}", key=button_key, use_container_width=True):
                    st.session_state.selected_question = question
                    st.session_state.question_input = question
                    st.rerun()
                
                # Apply custom styling to the button using CSS and button key
                st.markdown(f"""
                <style>
                [data-testid="stButton"] > button[kind="secondary"][data-testid="{button_key}"] {{
                    background: linear-gradient(135deg, #f8f9fa, #f1f3f5);
                    border-left: 4px solid #1E88E5;
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: 6px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                    font-size: 0.95rem;
                    text-align: left;
                    font-weight: normal;
                }}
                [data-testid="stButton"] > button[kind="secondary"][data-testid="{button_key}"]:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                </style>
                """, unsafe_allow_html=True)

            # Enhanced question input with icon and better styling
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.markdown('''
            <div style="display: flex; align-items: center; margin-bottom: 8px; padding: 0 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;">  
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <label style="font-weight: 500; color: #1E88E5;">Ask your question about TSLA data:</label>
            </div>
            ''', unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask your question:",  # Providing a proper label
                value=st.session_state.get('question_input', ''),
                height=100,
                placeholder="e.g., How many bullish days were there in 2023?",
                key="question_input_area",
                help="Ask questions about price trends, trading signals, or statistical analysis of your TSLA data",
                label_visibility="collapsed"  # Hide the label since we're using a custom one
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.question_input = user_question

            # Custom styled analysis button
            st.markdown('''
            <style>
            div[data-testid="stButton"] button {
                background: linear-gradient(90deg, #1E88E5, #26A69A);
                color: white;
                font-weight: bold;
                padding: 0.5rem 1rem;
                border-radius: 30px;
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            div[data-testid="stButton"] button:hover {
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
                transform: translateY(-2px);
            }
            </style>
            ''', unsafe_allow_html=True)
            
            # Analysis button with animation
            if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
                current_question = st.session_state.get('question_input', '').strip()
                if current_question:
                    with st.spinner("🧠 AI is analyzing the data..."):
                        # First try to parse with the new structured approach
                        parsed_query = parse_question_with_gemini(model, df, current_question)
                        
                        # Get the final response
                        response = get_ai_response(model, df, summary, current_question)
                        
                        st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                        st.markdown('<h3 style="color: #1E88E5; margin-top: 0;">📊 AI Analysis Result:</h3>', unsafe_allow_html=True)
                        
                        # Enhanced AI response styling with custom header and formatted paragraphs
                        st.markdown('''
                        <div class="ai-response-header">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                                <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                                <line x1="12" y1="22.08" x2="12" y2="12"></line>
                            </svg>
                            <h3>AI Analysis Result</h3>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Format the response with enhanced styling for paragraphs
                        paragraphs = response.split('\n\n')
                        for i, paragraph in enumerate(paragraphs):
                            if paragraph.strip():
                                # Apply different styling to the first paragraph (direct answer)
                                if i == 0:
                                    st.markdown(f"<p style='font-weight: 500; font-size: 1.1rem; color: #333;'>{paragraph}</p>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<p>{paragraph}</p>", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
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
        st.markdown('<p class="section-header">📋 TSLA Data Overview</p>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data to display.")
        else:
            # Data summary
            st.subheader("Dataset Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Total Records:** {summary['total_records']}")
                st.markdown(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
                st.markdown(f"**Avg. Close Price:** ${summary['price_stats']['avg_close']:.2f}")
            
            with col2:
                st.markdown(f"**Highest Price:** ${summary['price_stats']['highest_price']:.2f}")
                st.markdown(f"**Lowest Price:** ${summary['price_stats']['lowest_price']:.2f}")
                st.markdown(f"**Latest Close:** ${summary['price_stats']['latest_close']:.2f}")
            
            # Direction counts
            st.subheader("Trading Signal Distribution")
            direction_counts = pd.Series(summary.get('direction_counts', {}))
            
            if not direction_counts.empty:
                # Convert to DataFrame for better display
                direction_df = pd.DataFrame({
                    'Signal': direction_counts.index,
                    'Count': direction_counts.values,
                    'Percentage': (direction_counts.values / direction_counts.sum() * 100).round(1)
                })
                
                st.dataframe(direction_df.style.format({'Percentage': '{:.1f}%'}), use_container_width=True)
            
            # Add a sample data view at the bottom of the data tab
            st.subheader("TSLA Data Sample")
            st.dataframe(
                df[['timestamp', 'open', 'high', 'low', 'close', 'direction']].head(10),
                use_container_width=True
            )
            
            # Add download button for the data
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Processed Data",
                data=csv_data,
                file_name="tsla_processed_data.csv",
                mime="text/csv"
            )
    
    # Add clear history button at the bottom of the AI tab
    with ai_tab:
        if not df.empty and 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", key="clear_history"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()