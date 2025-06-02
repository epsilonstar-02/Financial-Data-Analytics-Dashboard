# trading_dashboard/charting.py
import pandas as pd

CHART_BG_COLOR = "#131722" # Dark background for the chart

def get_lightweight_chart_options():
    """Return the configuration options for the Lightweight Chart."""
    return {
        "height": 720,
        "layout": {
            "background": {"type": "solid", "color": CHART_BG_COLOR},
            "textColor": "#D9D9D9",
            "fontSize": 12,
            "fontFamily": 'Consolas, "Courier New", monospace' # Monospace for better number alignment
        },
        "grid": {
            "vertLines": {"color": "rgba(70, 70, 80, 0.5)", "style": 2}, # Softer grid lines
            "horzLines": {"color": "rgba(70, 70, 80, 0.5)", "style": 2}
        },
        "rightPriceScale": {
            "borderVisible": False,
            "scaleMargins": {"top": 0.15, "bottom": 0.15}, # More vertical padding
            "entireTextOnly": True
        },
        "timeScale": {
            "borderVisible": False,
            "timeVisible": True,
            "secondsVisible": False, # Usually not needed for daily/hourly
            "rightOffset": 12,
            "fixLeftEdge": True, # Improves initial view
            "fixRightEdge": True,
        },
        "crosshair": { # Enhanced crosshair
            "mode": 0, # Magnet mode
            "vertLine": {"width": 1, "color": "#C3BCDB40", "style": 0},
            "horzLine": {"width": 1, "color": "#C3BCDB40", "style": 0}
        },
        "localization": {"priceFormatter": "price => '$' + price.toFixed(2)"} # Format price on scale
    }

def create_candlestick_series(df: pd.DataFrame, markers: list):
    """Create the candlestick series data for the chart."""
    candle_data = df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
    return {
        'type': 'Candlestick',
        'data': candle_data,
        'options': {
            'upColor': '#26A69A',      # Bullish candle
            'downColor': '#EF5350',    # Bearish candle
            'borderUpColor': '#26A69A',
            'borderDownColor': '#EF5350',
            'wickUpColor': '#26A69A',
            'wickDownColor': '#EF5350',
        },
        'markers': markers,
    }

def create_trading_signal_markers(df: pd.DataFrame):
    """Create markers for trading signals (LONG, SHORT, NEUTRAL)."""
    markers = []
    if 'direction' not in df.columns or 'time' not in df.columns:
        return markers # Return empty if essential columns are missing

    for _, row in df.iterrows():
        if pd.isna(row['time']): continue # Skip if time is NaN

        common_attrs = {'time': row['time'], 'size': 1.2} # Default size

        if row['direction'] == 'LONG':
            markers.append({
                **common_attrs,
                'position': 'belowBar', 'color': '#26A69A', 'shape': 'arrowUp', 'text': 'L'
            })
        elif row['direction'] == 'SHORT':
            markers.append({
                **common_attrs,
                'position': 'aboveBar', 'color': '#EF5350', 'shape': 'arrowDown', 'text': 'S'
            })
        elif pd.notna(row['direction']): # For 'None' or other neutral indicators
            markers.append({
                **common_attrs, 'size': 0.8, # Smaller for neutral
                'position': 'inBar', 'color': '#FFD700', 'shape': 'circle', 'text': 'N'
            })
    return markers

def _create_area_series_for_band(data: list, color_hex: str, line_visible: bool = False, title: str = ""):
    """Helper to create a single area series component for floating bands."""
    # Use more transparency for fills
    return {
        'type': 'Area',
        'data': data,
        'options': {
            'topColor': f'{color_hex}1A',  # Very transparent
            'bottomColor': f'{color_hex}0D', # Even more transparent
            'lineColor': f'{color_hex}FF' if line_visible else 'rgba(0,0,0,0)', # Line color or invisible
            'lineWidth': 1 if line_visible else 0,
            'priceScaleId': '', # Attach to main price scale
            'title': title, # For legend
            'crosshairMarkerVisible': False,
            'lastValueVisible': False,
            'visible': True,
        }
    }

def create_floating_band_series(df: pd.DataFrame, low_col: str, high_col: str, band_color_hex: str, band_name: str):
    """
    Creates two AreaSeries components to simulate a floating band.
    One for the upper boundary (colored fill down to 0)
    One for the lower boundary (background-colored fill down to 0, acting as a mask)
    Returns a tuple: (upper_fill_series, lower_mask_series)
    """
    upper_band_data = []
    lower_band_data = []

    if not all(col in df.columns for col in [low_col, high_col, 'time']):
        return None, None # Essential columns missing

    for _, row in df.iterrows():
        low_val = row[low_col]
        high_val = row[high_col]
        time_val = row['time']

        if pd.notna(low_val) and pd.notna(high_val) and pd.notna(time_val) and \
           isinstance(low_val, (int, float)) and isinstance(high_val, (int, float)) and \
           low_val > 0 and high_val > 0 and low_val <= high_val and \
           abs(high_val - low_val) > 0.001: # Ensure meaningful band

            upper_band_data.append({'time': time_val, 'value': float(high_val)})
            lower_band_data.append({'time': time_val, 'value': float(low_val)})

    if not upper_band_data or not lower_band_data:
        return None, None

    # Series 1: Fills from high_col down to baseline with band color (transparent)
    upper_fill_series = _create_area_series_for_band(upper_band_data, band_color_hex, title=f"{band_name} High")
    
    # Series 2: Fills from low_col down to baseline with background color (masking effect)
    # This creates the "bottom" of the band by "erasing" the fill from upper_fill_series.
    lower_mask_series = _create_area_series_for_band(lower_band_data, CHART_BG_COLOR, title="") # No title for mask

    # To make the band more distinct, we can add thin lines at the band edges
    # These are separate line series on top of the area fills
    upper_line_series = {
        'type': 'Line', 'data': upper_band_data,
        'options': {'color': f'{band_color_hex}99', 'lineWidth': 1, 'priceScaleId': '', 'title': band_name, 'lastValueVisible': False}
    }
    lower_line_series = {
        'type': 'Line', 'data': lower_band_data,
        'options': {'color': f'{band_color_hex}99', 'lineWidth': 1, 'priceScaleId': '', 'title': '', 'lastValueVisible': False} # No duplicate title
    }
    
    return upper_fill_series, lower_mask_series, upper_line_series, lower_line_series


def assemble_chart_data(df: pd.DataFrame):
    """Assembles all components for the Lightweight Chart."""
    if df is None or df.empty:
        return None

    chart_options = get_lightweight_chart_options()
    markers = create_trading_signal_markers(df)
    
    series_components = []

    # Support Band (Green)
    support_fill_upper, support_mask_lower, sup_upper_line, sup_lower_line = create_floating_band_series(
        df, 'support_low', 'support_high', '#26A69A', 'Support' # Greenish
    )
    if support_fill_upper and support_mask_lower:
        series_components.extend([support_fill_upper, support_mask_lower, sup_upper_line, sup_lower_line])

    # Resistance Band (Red)
    res_fill_upper, res_mask_lower, res_upper_line, res_lower_line = create_floating_band_series(
        df, 'res_low', 'res_high', '#EF5350', 'Resistance' # Reddish
    )
    if res_fill_upper and res_mask_lower:
         series_components.extend([res_fill_upper, res_mask_lower, res_upper_line, res_lower_line])
    
    # Candlestick series should be added last to be on top of area fills, but lines for bands on top of candles.
    # Order: Area Fills -> Candles -> Band Lines
    # The library handles draw order based on list order for series of same type.
    # However, lines will draw over areas and candles over areas.
    # For simplicity, the current library draws series in the order they are provided.
    # Let's try: Area Fills (Support, Res) -> Candles -> Lines (Support, Res)
    
    final_series_list = []
    # Area fills first
    if support_fill_upper: final_series_list.extend([support_fill_upper, support_mask_lower])
    if res_fill_upper: final_series_list.extend([res_fill_upper, res_mask_lower])
    
    # Candlesticks
    candlestick_series = create_candlestick_series(df, markers)
    final_series_list.append(candlestick_series)

    # Then lines for better visibility of band edges
    if sup_upper_line: final_series_list.extend([sup_upper_line, sup_lower_line])
    if res_upper_line: final_series_list.extend([res_upper_line, res_lower_line])


    return [{
        'chart': chart_options,
        'series': final_series_list
    }]