import pandas as pd

def create_chart_config():
    """Create professional chart configuration"""
    return {
        "height": 720,
        "layout": {
            "background": {"type": "solid", "color": "#131722"}, # Dark background for the chart
            "textColor": "#D9D9D9", # Light text color for axes, etc.
            "fontSize": 12,
            "fontFamily": 'Consolas, monospace'
        },
        "grid": {
            "vertLines": {"color": "rgba(70, 70, 70, 0.5)", "style": 2}, # Slightly more visible grid lines
            "horzLines": {"color": "rgba(70, 70, 70, 0.5)", "style": 2}
        },
        "rightPriceScale": {
            "borderVisible": False,
            "scaleMargins": {"top": 0.1, "bottom": 0.2}, # Adjust margins for better price scale visibility
            "entireTextOnly": True
        },
        "timeScale": {
            "borderVisible": False,
            "timeVisible": True, # Ensure time is visible on the time scale
            "rightOffset": 12 # Space on the right of the chart
        },
    }

def create_markers(df):
    """
    Create trading direction markers.
    Returns an empty list if 'direction' column is missing or df is empty.
    """
    markers = []
    # If 'direction' column does not exist, or 'time' is missing, or df is empty, return no markers.
    if df.empty or 'direction' not in df.columns or 'time' not in df.columns:
        return markers

    for _, row in df.iterrows():
        direction_value = row['direction']
        time_value = row['time'] # Assuming 'time' is the UNIX timestamp for the chart

        # Handle various forms of missing or neutral values
        if pd.isna(direction_value):
            direction_value = 'None' # Standardize NaN/NA to 'None' string for marker logic
        
        # Ensure direction_value is a string for consistent comparison
        direction_str = str(direction_value).upper()

        if direction_str == 'LONG':
            markers.append({
                'time': time_value,
                'position': 'belowBar',
                'color': '#26A69A', # Green for long
                'shape': 'arrowUp',
                'size': 1.5,
                'text': 'L' # Optional text on marker
            })
        elif direction_str == 'SHORT':
            markers.append({
                'time': time_value,
                'position': 'aboveBar',
                'color': '#EF5350', # Red for short
                'shape': 'arrowDown',
                'size': 1.5,
                'text': 'S' # Optional text on marker
            })
        # Only create neutral markers if the direction value is explicitly 'None' (or other neutral strings)
        # This avoids creating markers if the direction value is something unexpected.
        elif direction_str == 'NONE' or not direction_value: # Catches 'None', empty strings after str conversion
            markers.append({
                'time': time_value,
                'position': 'inBar', # Position inside the bar for neutral
                'color': '#FFD700', # Yellow for neutral
                'shape': 'circle',
                'size': 1,
                'text': 'N' # Optional text on marker
            })
        # Optionally, log or handle other unexpected direction values:
        # else:
        #     if direction_value: # If it's not an empty string or None that we already handled
        #         print(f"Note: Unknown or unhandled direction value '{row['direction']}' at time {time_value}")
    return markers


def create_floating_band_series_components(df, low_col_name, high_col_name, band_color_hex, chart_bg_color_hex, base_name):
    """
    Creates two AreaSeries components to simulate a floating band.
    Returns a tuple: (upper_fill_series_definition, lower_mask_series_definition)
    Returns None, None if essential columns are missing, df is empty, or no valid band data can be generated.
    """
    upper_band_data = []
    lower_band_data = []

    # Check for essential columns
    if df.empty or low_col_name not in df.columns or high_col_name not in df.columns or 'time' not in df.columns:
        return None, None

    for _, row in df.iterrows():
        low_val = row[low_col_name]
        high_val = row[high_col_name]
        time_val = row['time']

        # Ensure values are valid numbers and form a meaningful band
        if (pd.notna(low_val) and pd.notna(high_val) and
            isinstance(low_val, (int, float)) and isinstance(high_val, (int, float)) and
            low_val > 0 and high_val > 0 and # Assuming prices/levels are positive
            low_val <= high_val and
            abs(high_val - low_val) > 0.001): # Ensure meaningful band height (adjust threshold if needed)

            upper_band_data.append({'time': time_val, 'value': float(high_val)})
            lower_band_data.append({'time': time_val, 'value': float(low_val)})

    # If no valid data points were found to create bands
    if not upper_band_data or not lower_band_data:
        return None, None

    # Series 1: Fills from the high_col_name down to baseline with band color (semi-transparent)
    upper_fill_series = {
        'type': 'Area',
        'data': upper_band_data,
        'options': {
            'topColor': f'{band_color_hex}20',  # Semi-transparent band color (e.g., alpha 20)
            'bottomColor': f'{band_color_hex}40', # Slightly more opaque at the bottom (e.g., alpha 40)
            'lineColor': 'rgba(0,0,0,0)', # Invisible line for the area series
            'lineWidth': 0,
            'priceScaleId': '', # Attach to the main price scale if not specified (usually right)
            'title': base_name, # This will appear in the legend
            'crosshairMarkerVisible': False, # No crosshair marker for these band series
            'lastValueVisible': False, # Don't show last value label for bands
            'visible': True,
        }
    }

    # Series 2: Fills from the low_col_name down to baseline with background color (masking effect)
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