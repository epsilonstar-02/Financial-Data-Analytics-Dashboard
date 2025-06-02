# trading_dashboard/data_processing.py
import streamlit as st
import pandas as pd
import ast
from datetime import datetime

def _safe_literal_eval(val):
    """Safely evaluate a string literal to a list, returning empty list on failure or invalid input."""
    try:
        if pd.isna(val) or val == '' or not isinstance(val, str):
            return []
        evaluated = ast.literal_eval(val)
        return evaluated if isinstance(evaluated, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []

def _calculate_rolling_bounds(df, window=20):
    """Calculate rolling min/max for default support/resistance."""
    df_copy = df.copy()
    df_copy['default_support'] = df_copy['low'].rolling(window=window, min_periods=1).min()
    df_copy['default_resistance'] = df_copy['high'].rolling(window=window, min_periods=1).max()
    return df_copy

def _fill_missing_band_values(df):
    """Fill missing support/resistance band values using ffill and default bounds."""
    df_copy = df.copy()
    if 'default_support' not in df_copy.columns or 'default_resistance' not in df_copy.columns:
        st.warning("Default bounds not found for filling missing values. Recalculating.")
        df_copy = _calculate_rolling_bounds(df_copy) # Ensure defaults exist

    band_cols = ['support_low', 'support_high', 'res_low', 'res_high']
    for col in band_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(method='ffill')
    
    # Fill remaining NaNs with default values
    df_copy['support_low'] = df_copy['support_low'].fillna(df_copy['default_support'])
    df_copy['support_high'] = df_copy['support_high'].fillna(df_copy['default_support'])
    df_copy['res_low'] = df_copy['res_low'].fillna(df_copy['default_resistance'])
    df_copy['res_high'] = df_copy['res_high'].fillna(df_copy['default_resistance'])

    # Ensure logical order (high >= low)
    df_copy['support_high'] = df_copy[['support_high', 'support_low']].max(axis=1)
    df_copy['res_high'] = df_copy[['res_high', 'res_low']].max(axis=1)
    
    # Final fallback for any remaining NaNs (e.g., if default_support itself was NaN)
    for col in band_cols:
        if df_copy[col].isna().any():
            df_copy[col] = df_copy[col].fillna(df_copy['close']) # Use close as last resort

    return df_copy.drop(columns=['default_support', 'default_resistance'], errors='ignore')


def calculate_support_resistance_bands(df, window=20):
    """
    Enhanced support/resistance calculation with robust missing data handling.
    This function expects 'support_list' and 'resistance_list' columns to exist.
    """
    df_copy = df.copy()

    def get_min_max_from_list(price_list):
        if not price_list or not isinstance(price_list, list):
            return None, None
        valid_prices = [p for p in price_list if isinstance(p, (int, float)) and p > 0]
        if not valid_prices:
            return None, None
        return min(valid_prices), max(valid_prices)

    # Calculate low/high from lists
    bounds_from_lists = df_copy.apply(
        lambda row: pd.Series({
            'support_low': get_min_max_from_list(row['support_list'])[0],
            'support_high': get_min_max_from_list(row['support_list'])[1],
            'res_low': get_min_max_from_list(row['resistance_list'])[0],
            'res_high': get_min_max_from_list(row['resistance_list'])[1]
        }),
        axis=1
    )
    df_copy = pd.concat([df_copy, bounds_from_lists], axis=1)
    
    # Calculate default rolling bounds
    df_copy = _calculate_rolling_bounds(df_copy, window=window)
    
    # Fill missing values using ffill and defaults
    df_copy = _fill_missing_band_values(df_copy)
    
    return df_copy


@st.cache_data
def load_and_process_data(uploaded_file) -> pd.DataFrame | None:
    """Load, process, and clean TSLA data from a CSV file."""
    if uploaded_file is None:
        return None
    
    try:
        df = pd.read_csv(uploaded_file)

        # Standardize column names (case-insensitive matching)
        column_map = {
            'timestamp': ['timestamp', 'date', 'time'],
            'open': ['open', 'opn'],
            'high': ['high', 'hi'],
            'low': ['low', 'lo'],
            'close': ['close', 'cls'],
            'direction': ['direction', 'signal'],
            'Support': ['support', 'suprt'], # Original column name for list
            'Resistance': ['resistance', 'res'] # Original column name for list
        }
        
        standardized_columns = {}
        df_columns_lower = {col.lower(): col for col in df.columns}

        for std_name, variations in column_map.items():
            found_col = None
            for var in variations:
                if var in df_columns_lower:
                    found_col = df_columns_lower[var]
                    break
            if not found_col:
                # Support and Resistance list columns are crucial but might be missing.
                # OHLC and timestamp are more critical.
                if std_name not in ['Support', 'Resistance']: # Allow S/R lists to be optional initially
                    st.error(f"❌ Missing critical data column (or variation): '{std_name}'. Found columns: {list(df.columns)}")
                    return None
                else: # If S/R lists are missing, create empty ones for now
                    df[std_name] = pd.Series([[] for _ in range(len(df))]) 
                    standardized_columns[std_name] = std_name # Use standard name
                    continue # Skip renaming if column was created
            
            standardized_columns[std_name] = found_col

        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in standardized_columns.items() if v in df.columns})

        # Ensure all expected standard columns exist after renaming, even if created
        expected_std_cols = ['timestamp', 'open', 'high', 'low', 'close', 'direction', 'Support', 'Resistance']
        for col_name in expected_std_cols:
            if col_name not in df.columns:
                 # This case should ideally be handled by the mapping logic above
                if col_name in ['Support', 'Resistance']: # If S/R lists are missing, create empty ones
                    df[col_name] = pd.Series([[] for _ in range(len(df))])
                else: # Should not happen if initial check is robust
                    st.error(f"Column '{col_name}' is missing after standardization. Critical error.")
                    return None

        # Type conversions and parsing
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parse support/resistance lists
        df['support_list'] = df['Support'].apply(_safe_literal_eval)
        df['resistance_list'] = df['Resistance'].apply(_safe_literal_eval)
        
        # Calculate support/resistance bands from parsed lists and defaults
        df = calculate_support_resistance_bands(df)

        # Prepare 'time' column for charting library (Unix timestamp in seconds)
        df['time'] = (df['timestamp'].astype('int64') // 10**9).astype(int)
        
        # Final cleanup
        required_chart_cols = ['open', 'high', 'low', 'close', 'timestamp', 'time',
                               'support_low', 'support_high', 'res_low', 'res_high']
        df = df.dropna(subset=required_chart_cols) # Drop rows if essential chart data is missing
        df = df.drop_duplicates('time', keep='last') # Ensure unique timestamps for charting
        df = df.sort_values('time').reset_index(drop=True)
        
        if df.empty:
            st.warning("⚠️ No valid data rows remained after processing. Please check your CSV content and format.")
            return None
            
        return df

    except Exception as e:
        st.error(f"❌ Error loading or processing data: {str(e)}")
        st.error("Please ensure your CSV file is correctly formatted. Check for valid numbers in price columns and correctly formatted lists in Support/Resistance columns (e.g., '[100.5, 101.0]').")
        return None

def create_data_summary(df: pd.DataFrame) -> dict:
    """Create a comprehensive data summary for display and AI context."""
    if df is None or df.empty:
        return {
            'total_records': 0,
            'date_range': {'start': 'N/A', 'end': 'N/A'},
            'price_stats': {'highest_price': 0, 'lowest_price': 0, 'avg_close': 0, 'latest_close': 0},
            'direction_counts': {},
            'support_resistance_stats': {'avg_support_levels': 0, 'avg_resistance_levels': 0}
        }

    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
            'end': df['timestamp'].max().strftime('%Y-%m-%d') if not df.empty else 'N/A'
        },
        'price_stats': {
            'highest_price': df['high'].max() if not df.empty else 0,
            'lowest_price': df['low'].min() if not df.empty else 0,
            'avg_close': df['close'].mean() if not df.empty else 0,
            'latest_close': df['close'].iloc[-1] if not df.empty else 0
        },
        'direction_counts': df['direction'].value_counts().to_dict() if 'direction' in df.columns and not df.empty else {},
        'support_resistance_stats': {
            'avg_support_levels': df['support_list'].apply(len).mean() if 'support_list' in df.columns and not df.empty else 0,
            'avg_resistance_levels': df['resistance_list'].apply(len).mean() if 'resistance_list' in df.columns and not df.empty else 0
        }
    }
    return summary