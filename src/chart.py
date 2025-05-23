import streamlit as st
import pandas as pd
import ast
from streamlit_lightweight_charts import renderLightweightCharts

st.title("TSLA Candlestick-Chart")

# --- 1) Load & Prep Data ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    # Enhanced column detection with error display
    ohlc_map = {
        'open': ['Open', 'OPEN', 'open'],
        'high': ['High', 'HIGH', 'high'],
        'low': ['Low', 'LOW', 'low'],
        'close': ['Close', 'CLOSE', 'close'],
        'timestamp': ['timestamp', 'Date', 'TIME']
    }
    
    for standard_name, variants in ohlc_map.items():
        match = next((col for col in df.columns if col in variants), None)
        if not match:
            st.error(f" Missing required column: {standard_name}. Found columns: {list(df.columns)}")
            st.stop()
        df[standard_name] = pd.to_numeric(df[match], errors='coerce') if standard_name != 'timestamp' else df[match]
    
    # parse Support/Resistance string lists into Python lists
    df['support_list']    = df['Support'].apply(ast.literal_eval)
    df['resistance_list'] = df['Resistance'].apply(ast.literal_eval)
    # compute min/max (None if empty)
    df['support_low']  = df['support_list'].apply(lambda x: min(x) if x else None)
    df['support_high'] = df['support_list'].apply(lambda x: max(x) if x else None)
    df['res_low']      = df['resistance_list'].apply(lambda x: min(x) if x else None)
    df['res_high']     = df['resistance_list'].apply(lambda x: max(x) if x else None)
    # convert timestamp to UNIX seconds for both Candlestick and Lines
    df['time'] = (df['timestamp'].astype('int64') // 10**9).astype(int)
    return df

# Sidebar: CSV uploader
csv_file = st.sidebar.file_uploader("Upload TSLA CSV", type="csv")
if not csv_file:
    st.info("Please upload your `TSLA_data - Sheet1.csv` file.")
    st.stop()

df = load_data(csv_file)
# Clean missing values
df = df.dropna(subset=['open','high','low','close'])
# Handle duplicates
df = df.drop_duplicates('time', keep='last')
# Sort chronologically
df = df.sort_values(['time', 'timestamp'])

# --- 2) Premium Chart Styling ---
chart_opts = {
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

# Candlestick series with markers
candle_data = df[['time','open','high','low','close']].to_dict('records')

# --- 3) Marker Configuration ---
markers = []
for r in df.itertuples():
    if r.direction == 'LONG':
        markers.append({
            'time': r.time,
            'position': 'belowBar',
            'color': '#26A69A',
            'shape': 'arrowUp',
            'size': 1.5
        })
    elif r.direction == 'SHORT':
        markers.append({
            'time': r.time,
            'position': 'aboveBar',
            'color': '#EF5350',
            'shape': 'arrowDown',
            'size': 1.5
        })
    else:  # Handle None direction
        markers.append({
            'time': r.time,
            'position': 'inBar',
            'color': '#FFD700',  # Gold/Yellow
            'shape': 'circle',
            'size': 1
        })

# --- 4) Precise Band Definitions ---
# Support band using min/max from Support column lists
df['support_low'] = df['support_list'].apply(lambda x: min(x) if x else None)
df['support_high'] = df['support_list'].apply(lambda x: max(x) if x else None)

# Resistance band using min/max from Resistance column lists
df['res_low'] = df['resistance_list'].apply(lambda x: min(x) if x else None)
df['res_high'] = df['resistance_list'].apply(lambda x: max(x) if x else None)

def create_band_series(low_col: str, high_col: str, color: str):
    valid_data = []
    for t, low, high in zip(df['time'], df[low_col], df[high_col]):
        if pd.notna(low) and pd.notna(high) and low <= high:
            valid_data.append({'time': t, 'value': low, 'highValue': high})
    
    return {
        'type': 'Area',
        'data': valid_data,
        'options': {
            'topColor': f'{color}20',  # Reduced opacity
            'bottomColor': f'{color}40',
            'lineWidth': 1,
            'lineStyle': 0,  # Solid line
            'priceScaleId': ''
        }
    }

support_band = create_band_series('support_low', 'support_high', '#26A69A')
resistance_band = create_band_series('res_low', 'res_high', '#EF5350')

candle_series = {
    'type': 'Candlestick',
    'data': candle_data,
    'options': {
        'upColor': '#26a69a', 'downColor': '#ef5350',
        'wickUpColor': '#26a69a', 'wickDownColor': '#ef5350',
    },
    'markers': markers,
}

# Assemble charts payload and render
charts = [{
    'chart': chart_opts,
    'series': [
        candle_series,
        support_band,
        resistance_band,
    ]
}]

renderLightweightCharts(charts, key='tsla_full')
