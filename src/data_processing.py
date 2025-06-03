import streamlit as st
import pandas as pd
import ast

# --- Helper functions for S/R calculation (remain largely the same internally) ---
def calculate_default_bounds(df, window=20):
    df = df.copy()
    if 'low' in df.columns and 'high' in df.columns:
        df['default_support'] = df['low'].rolling(window=window, min_periods=1).min()
        df['default_resistance'] = df['high'].rolling(window=window, min_periods=1).max()
    else: # Fallback if low/high aren't present, though they should be for OHLC
        df['default_support'] = pd.Series([None] * len(df), dtype=float) # Ensure dtype for empty series
        df['default_resistance'] = pd.Series([None] * len(df), dtype=float)
    return df

def fill_missing_values(df):
    df = df.copy()
    # Ensure default bounds are calculated if not present, especially if S/R columns might not exist yet
    if 'default_support' not in df.columns or 'default_resistance' not in df.columns:
        df = calculate_default_bounds(df)

    support_cols = ['support_low', 'support_high']
    resistance_cols = ['res_low', 'res_high']

    # Forward fill with previous valid values
    for cols_list in [support_cols, resistance_cols]:
        for col in cols_list:
            if col in df.columns: # Ensure column exists before trying to fill
                 df[col] = df[col].fillna(method='ffill')

    # Fill any remaining NAs with default values only if default_support/resistance exist and are not all NaN
    if 'default_support' in df.columns and not df['default_support'].isnull().all():
        if 'support_low' in df.columns: df['support_low'] = df['support_low'].fillna(df['default_support'])
        if 'support_high' in df.columns: df['support_high'] = df['support_high'].fillna(df['default_support'])
    if 'default_resistance' in df.columns and not df['default_resistance'].isnull().all():
        if 'res_low' in df.columns: df['res_low'] = df['res_low'].fillna(df['default_resistance'])
        if 'res_high' in df.columns: df['res_high'] = df['res_high'].fillna(df['default_resistance'])

    # Ensure high >= low after filling, if columns exist
    if 'support_low' in df.columns and 'support_high' in df.columns:
        df['support_high'] = df[['support_high', 'support_low']].max(axis=1)
    if 'res_low' in df.columns and 'res_high' in df.columns:
        df['res_high'] = df[['res_high', 'res_low']].max(axis=1)
    return df

def improved_support_resistance_calculation(df, window=20):
    df = df.copy()
    def safe_bounds(price_list):
        if not price_list or len(price_list) == 0:
            return None, None
        valid_prices = [p for p in price_list if isinstance(p, (int, float)) and p > 0]
        if not valid_prices:
            return None, None
        return min(valid_prices), max(valid_prices)

    # These columns are only created if support_list/resistance_list exist from raw data
    if 'support_list' in df.columns and 'resistance_list' in df.columns:
        bounds = df[['support_list', 'resistance_list']].apply(
            lambda x: pd.Series({
                'support_low': safe_bounds(x['support_list'])[0],
                'support_high': safe_bounds(x['support_list'])[1],
                'res_low': safe_bounds(x['resistance_list'])[0],
                'res_high': safe_bounds(x['resistance_list'])[1]
            }), axis=1
        )
        df = pd.concat([df, bounds], axis=1)
    else: # Ensure columns exist with None (or float dtype) if lists weren't provided, so fill_missing_values doesn't break
        for col in ['support_low', 'support_high', 'res_low', 'res_high']:
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df), dtype=float)


    df = calculate_default_bounds(df, window=window) # Calculates 'default_support' and 'default_resistance'
    df = fill_missing_values(df) # Uses default bounds and ffill
    
    # Final check for any NaNs that might have slipped through complex fills, use default price levels
    sr_cols_to_check = ['support_low', 'support_high', 'res_low', 'res_high']
    for col in sr_cols_to_check:
        if col in df.columns and df[col].isna().any():
            if 'support' in col and 'default_support' in df.columns:
                df[col] = df[col].fillna(df['default_support'])
            elif 'res' in col and 'default_resistance' in df.columns:
                df[col] = df[col].fillna(df['default_resistance'])
    
    # Ensure logical order of low/high for bands after all filling
    if 'support_low' in df.columns and 'support_high' in df.columns:
        df['support_high'] = df[['support_high', 'support_low']].max(axis=1)
    if 'res_low' in df.columns and 'res_high' in df.columns:
        df['res_high'] = df[['res_high', 'res_low']].max(axis=1)
    
    # Avoid issues if default_support/resistance were themselves NaN (e.g., very short df)
    # This is a last resort if previous fills left NaNs
    if 'close' in df.columns: # Only if 'close' column exists
        for col in sr_cols_to_check:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df['close'])


    df = df.drop(columns=['default_support', 'default_resistance'], errors='ignore')
    return df
# --- End of S/R helper functions ---

@st.cache_data
def load_and_prepare_data(csv_file, column_map_config, process_sr_data=True, raw_support_col='Support', raw_resistance_col='Resistance'):
    """
    Loads data, standardizes OHLC, timestamp.
    Optionally processes S/R data if columns exist and process_sr_data is True.
    Does NOT create a 'direction' column if it's not found in the source.
    Attempts case-insensitive column matching.
    """
    try:
        df_original = pd.read_csv(csv_file)
        df = df_original.copy()

        # Timestamp processing (case-insensitive)
        ts_col_variants = column_map_config.get('timestamp', ['timestamp', 'Date', 'TIME', 'date'])
        found_ts_col = None
        for col_variant in ts_col_variants:
            matching_cols = [c for c in df.columns if c.lower() == col_variant.lower()]
            if matching_cols:
                actual_col_name = matching_cols[0] # Take the first match
                df['timestamp'] = pd.to_datetime(df[actual_col_name], errors='coerce')
                if actual_col_name.lower() != 'timestamp': # Avoid dropping if it's already named 'timestamp' (case-insensitively)
                    df = df.drop(columns=[actual_col_name])
                found_ts_col = 'timestamp'
                break
        if not found_ts_col:
            st.error(f"❌ Timestamp column not found (expected one of {ts_col_variants}). Found: {list(df.columns)}")
            st.stop()
            return pd.DataFrame(), False # Return flag indicating S/R status

        # OHLC processing (case-insensitive)
        ohlc_cols = ['open', 'high', 'low', 'close']
        for std_name in ohlc_cols:
            variants = column_map_config.get(std_name, [std_name.capitalize(), std_name.upper(), std_name])
            found_ohlc_col = None
            for col_variant in variants:
                matching_cols = [c for c in df.columns if c.lower() == col_variant.lower()]
                if matching_cols:
                    actual_col_name = matching_cols[0]
                    df[std_name] = pd.to_numeric(df[actual_col_name], errors='coerce')
                    if actual_col_name.lower() != std_name: # Avoid dropping if it's already the standard name (case-insensitively)
                         df = df.drop(columns=[actual_col_name])
                    found_ohlc_col = std_name
                    break
            if not found_ohlc_col:
                st.error(f"❌ Required OHLC column '{std_name}' not found (expected one of {variants}). Found: {list(df.columns)}")
                st.stop()
                return pd.DataFrame(), False

        # Optional 'direction' column (case-insensitive) - **NO DEFAULT CREATION**
        direction_variants = column_map_config.get('direction', ['direction', 'Direction', 'Signal'])
        # found_direction_col_name = None # Not strictly needed if we don't create default
        for col_variant in direction_variants:
            matching_cols = [c for c in df.columns if c.lower() == col_variant.lower()]
            if matching_cols:
                actual_col_name = matching_cols[0]
                if actual_col_name.lower() != 'direction': # Rename if not already 'direction' (case-insensitively)
                    df.rename(columns={actual_col_name: 'direction'}, inplace=True)
                # found_direction_col_name = 'direction' # Mark that 'direction' now exists
                break
        # If 'direction' column was not found, it will simply not exist in the DataFrame.

        # Optional 'volume' column (case-insensitive)
        volume_variants = column_map_config.get('volume', ['volume', 'Volume', 'VOLUME'])
        # found_volume_col = None # Not strictly needed
        for col_variant in volume_variants:
            matching_cols = [c for c in df.columns if c.lower() == col_variant.lower()]
            if matching_cols:
                actual_col_name = matching_cols[0]
                df['volume'] = pd.to_numeric(df[actual_col_name], errors='coerce')
                if actual_col_name.lower() != 'volume':
                    df = df.drop(columns=[actual_col_name])
                # found_volume_col = 'volume'
                break


        df['time'] = (df['timestamp'].astype('int64') // 10**9).astype(int)
        
        essential_cols_for_dropna = ['open', 'high', 'low', 'close', 'timestamp', 'time']
        df = df.dropna(subset=essential_cols_for_dropna) # Drop rows if core OHLC data is missing after conversion
        if df.empty: # If all rows were dropped
            st.warning("All rows were dropped after processing essential OHLC and timestamp columns. Please check data quality.")
            return pd.DataFrame(), False

        df = df.drop_duplicates('time', keep='last').sort_values('time')

        sr_data_processed_successfully = False
        if process_sr_data:
            # Case-insensitive check for raw S/R columns
            actual_raw_support_col = next((c for c in df.columns if c.lower() == raw_support_col.lower()), None)
            actual_raw_resistance_col = next((c for c in df.columns if c.lower() == raw_resistance_col.lower()), None)

            if actual_raw_support_col and actual_raw_resistance_col:
                def safe_literal_eval(val):
                    try:
                        if pd.isna(val) or val == '': return []
                        if isinstance(val, str): return ast.literal_eval(val)
                        return val if isinstance(val, list) else []
                    except: return [] # Return empty list on any parsing error
                
                df['support_list'] = df[actual_raw_support_col].apply(safe_literal_eval)
                df['resistance_list'] = df[actual_raw_resistance_col].apply(safe_literal_eval)
                
                df = improved_support_resistance_calculation(df) # This will add S/R band columns
                
                # Check if S/R band columns were successfully created and have some non-null data
                sr_band_cols = ['support_low', 'support_high', 'res_low', 'res_high']
                if all(col in df.columns for col in sr_band_cols):
                    # A more robust check for successful processing
                    sr_data_processed_successfully = not df[sr_band_cols].isnull().all().all()

                if not sr_data_processed_successfully and (df['support_list'].apply(len).sum() > 0 or df['resistance_list'].apply(len).sum() > 0) :
                     st.warning("S/R data was present in raw columns, but processing might not have yielded valid bands for all points. Check S/R column format (e.g., '[100, 101]').")
                
                # Clean up intermediate list columns if they are not the raw ones and not needed further
                # These columns are temporary for processing S/R bands
                if 'support_list' in df.columns: df = df.drop(columns=['support_list'], errors='ignore')
                if 'resistance_list' in df.columns: df = df.drop(columns=['resistance_list'], errors='ignore')
            
            elif process_sr_data: # process_sr_data is true, but one or both raw S/R columns are missing
                st.warning(f"Support/Resistance processing enabled, but columns ('{raw_support_col}', '{raw_resistance_col}') not found using case-insensitive search. S/R bands will not be shown.")
        
        return df, sr_data_processed_successfully

    except Exception as e:
        st.error(f"❌ Error in load_and_prepare_data: {str(e)}")
        st.exception(e) # Provides full traceback in console for debugging
        st.stop()
        return pd.DataFrame(), False


def create_data_summary(df, sr_data_available=False):
    """Create a comprehensive data summary. sr_data_available refers to if S/R bands were successfully processed."""
    if df is None or df.empty:
        return {
            'total_records': 0, 'date_range': {'start': 'N/A', 'end': 'N/A'},
            'price_stats': {'highest_price': 0, 'lowest_price': 0, 'avg_close': 0, 'latest_close': 0},
            'direction_counts': {}, 'volume_stats': {'avg_volume': 'N/A', 'total_volume': 'N/A'},
            'support_resistance_stats': {'status': 'N/A'} # Status of S/R band processing
        }
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else 'N/A',
            'end': df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else 'N/A'
        },
        'price_stats': {
            'highest_price': df['high'].max() if 'high' in df.columns and not df.empty else 0,
            'lowest_price': df['low'].min() if 'low' in df.columns and not df.empty else 0,
            'avg_close': df['close'].mean() if 'close' in df.columns and not df.empty else 0,
            'latest_close': df['close'].iloc[-1] if 'close' in df.columns and not df.empty else 0
        },
        # Only include direction_counts if the 'direction' column actually exists in the df
        'direction_counts': df['direction'].value_counts().to_dict() if 'direction' in df.columns and not df.empty else {},
        'volume_stats': {
            'avg_volume': f"{df['volume'].mean():,.0f}" if 'volume' in df.columns and not df.empty and df['volume'].notna().any() else 'N/A',
            'total_volume': f"{df['volume'].sum():,.0f}" if 'volume' in df.columns and not df.empty and df['volume'].notna().any() else 'N/A'
        }
    }

    # Update S/R stats based on whether bands were successfully processed and are available
    if sr_data_available: # This flag is True if S/R processing was enabled AND successful
        summary['support_resistance_stats'] = {'status': 'Enabled and band data available'}
    elif 'support_low' in df.columns: # S/R processing might have been attempted but not fully successful
        summary['support_resistance_stats'] = {'status': 'Enabled but band data might be incomplete/missing'}
    else: # S/R processing was disabled or no S/R columns were found to begin with
        summary['support_resistance_stats'] = {'status': 'Disabled or no S/R data found'}
        
    return summary