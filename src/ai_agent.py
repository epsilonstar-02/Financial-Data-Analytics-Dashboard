import streamlit as st
import pandas as pd
import json

def analyze_data_for_question(df, question, available_columns): # Pass available_columns
    """Analyze data based on the specific question and return relevant subset"""
    question_lower = question.lower()
    analysis_data = {}

    if df.empty:
        return {"error": "No data available for analysis."}

    # Year-specific analysis (e.g., 2023) - this is generic enough
    # but relies on 'timestamp' and 'direction'
    if any(year_str in question_lower for year_str in ["2020", "2021", "2022", "2023", "2024", "2025"]): # Example years
        year_to_analyze = next((int(s) for s in question_lower.split() if s.isdigit() and len(s) == 4), None)
        if year_to_analyze and 'timestamp' in available_columns:
            df_year = df[df['timestamp'].dt.year == year_to_analyze].copy()
            if not df_year.empty:
                year_data = {
                    'total_days': len(df_year),
                    'price_range': f"${df_year['low'].min():.2f} - ${df_year['high'].max():.2f}" if 'low' in df_year and 'high' in df_year else "N/A",
                    'avg_close': f"${df_year['close'].mean():.2f}" if 'close' in df_year else "N/A",
                }
                if 'direction' in available_columns:
                    year_data['bullish_days'] = len(df_year[df_year['direction'] == 'LONG'])
                    year_data['bearish_days'] = len(df_year[df_year['direction'] == 'SHORT'])
                    year_data['neutral_days'] = len(df_year[~df_year['direction'].isin(['LONG', 'SHORT'])])
                    year_data['sample_bullish_dates'] = df_year[df_year['direction'] == 'LONG']['timestamp'].dt.strftime('%Y-%m-%d').head(3).tolist()
                analysis_data[f'{year_to_analyze}_data'] = year_data
            else:
                analysis_data[f'{year_to_analyze}_data'] = f"No data available for {year_to_analyze}."


    if 'price' in question_lower or 'highest' in question_lower or 'lowest' in question_lower:
        if all(c in available_columns for c in ['high', 'low', 'close', 'timestamp']):
            analysis_data['price_analysis'] = {
                'highest_price': f"${df['high'].max():.2f}",
                'lowest_price': f"${df['low'].min():.2f}",
                'highest_date': df.loc[df['high'].idxmax()]['timestamp'].strftime('%Y-%m-%d'),
                'lowest_date': df.loc[df['low'].idxmin()]['timestamp'].strftime('%Y-%m-%d'),
                'price_volatility': f"{((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%"
            }

    # Support/Resistance analysis if relevant columns are present
    # This part used 'support_list', 'resistance_list'. If they are dropped after processing,
    # we need to rely on 'support_low', 'res_high' etc.
    # For simplicity, let's assume if S/R was processed, 'support_low' etc. exist.
    if ('support' in question_lower or 'resistance' in question_lower) and \
       all(c in available_columns for c in ['support_low', 'support_high', 'res_low', 'res_high']):
        analysis_data['support_resistance_bands_info'] = {
            'lowest_support_level': f"${df['support_low'].min():.2f}",
            'highest_resistance_level': f"${df['res_high'].max():.2f}",
            'avg_support_range': f"${(df['support_high'] - df['support_low']).mean():.2f}",
            'avg_resistance_range': f"${(df['res_high'] - df['res_low']).mean():.2f}",
        }

    if ('signal' in question_lower or 'long' in question_lower or 'short' in question_lower) and \
        'direction' in available_columns:
        analysis_data['signal_analysis'] = {
            'total_datapoints_with_signals': len(df),
            'long_signals': len(df[df['direction'] == 'LONG']),
            'short_signals': len(df[df['direction'] == 'SHORT']),
            'neutral_signals': len(df[~df['direction'].isin(['LONG', 'SHORT'])]), # Count 'None' or other values
            'long_percentage': f"{len(df[df['direction'] == 'LONG']) / len(df) * 100:.1f}%" if len(df) > 0 else "N/A",
            'short_percentage': f"{len(df[df['direction'] == 'SHORT']) / len(df) * 100:.1f}%" if len(df) > 0 else "N/A"
        }

    if ('quarter' in question_lower or 'month' in question_lower) and \
        all(c in available_columns for c in ['timestamp', 'close', 'direction']):
        df_temp = df.copy()
        df_temp['month'] = df_temp['timestamp'].dt.month
        df_temp['quarter'] = df_temp['timestamp'].dt.quarter
        
        monthly_stats = df_temp.groupby('month').agg(
            avg_close_price=('close', 'mean'),
            num_long_signals=('direction', lambda x: (x == 'LONG').sum())
        ).reset_index()
        monthly_stats['avg_close_price'] = monthly_stats['avg_close_price'].round(2)
        
        analysis_data['temporal_analysis'] = {
            'monthly_avg_price': monthly_stats.set_index('month')['avg_close_price'].to_dict(),
            'monthly_bullish_days': monthly_stats.set_index('month')['num_long_signals'].to_dict()
        }
    
    if not analysis_data: # If no specific analysis was triggered
        analysis_data['general_info'] = "Could not extract specific analysis for this question. Please try rephrasing or asking about available data aspects."
        
    return analysis_data


def create_context_prompt(df_summary, user_question, specific_analysis, available_columns):
    """Create comprehensive context for the AI agent with specific data analysis"""
    # Sample data string is removed as it's part of the summary now if needed
    
    context = f"""
You are a professional financial data analyst. You have access to detailed trading data and specific analysis results.

DATASET OVERVIEW:
- Total Records: {df_summary.get('total_records', 'N/A')}
- Date Range: {df_summary.get('date_range', {}).get('start', 'N/A')} to {df_summary.get('date_range', {}).get('end', 'N/A')}
- Price Statistics: {df_summary.get('price_stats', 'N/A')}
- Direction Distribution (if available): {df_summary.get('direction_counts', 'N/A')}
- Support/Resistance Data Status: {df_summary.get('support_resistance_stats', {}).get('status', 'N/A')}

AVAILABLE COLUMNS FOR QUERYING: {', '.join(available_columns)}
Common columns include: timestamp, open, high, low, close.
Optional columns might include: direction, volume, support_low, support_high, res_low, res_high.

SPECIFIC ANALYSIS FOR YOUR QUESTION (if applicable):
{specific_analysis}

USER QUESTION: {user_question}

INSTRUCTIONS:
- Use the specific analysis data provided above OR the general dataset overview to answer the question.
- If the question is about data from a specific year, use the relevant section from the analysis (e.g., YYYY_data).
- Provide concrete numbers and insights from the dataset. Be specific and factual.
- Reference actual dates, prices, and counts from the data if available in the specific analysis.
- If specific analysis is not directly relevant, use your general understanding of the dataset overview and available columns.
- If S/R data status is 'Disabled or no data', do not invent S/R information.

Please provide a detailed, data-driven analysis that directly answers the user's question.
"""
    return context

def parse_question_with_gemini(model, question, available_columns):
    """Use Gemini to parse a natural language question into structured JSON format"""
    if not model:
        st.error("Gemini model not available for parsing question.")
        return None

    cols_str = ", ".join(available_columns)
    prompt = f"""
        You are an AI assistant that converts natural language questions about financial data into structured JSON queries.
        
        QUESTION: "{question}"
        
        AVAILABLE COLUMNS IN THE DATASET:
        {cols_str}
        
        Notes on common columns:
        - timestamp: Date/time of the record (format YYYY-MM-DD or includes time)
        - open, high, low, close: Standard OHLC price values
        - volume: Trading volume (optional)
        - direction: Trading signal (e.g., LONG, SHORT, None) (optional)
        - support_low, support_high: Support band price levels (optional)
        - res_low, res_high: Resistance band price levels (optional)
        
        OUTPUT FORMAT: Return a valid JSON object with the following fields:
        - action: (string) Query action type (e.g., 'aggregate', 'filter', 'describe', 'compare', 'count')
        - columns: (array) List of column names relevant to the query (e.g., ["volume"], ["close"])
        - filters: (object) Dictionary with filter conditions (e.g., time ranges, value conditions, direction)
          Example: {{"timestamp": {{"start": "2023-01-01", "end": "2023-01-31"}}, "direction": "LONG", "close": {{">": 150}}}}
        - metric: (string) Statistical operation (e.g., 'mean', 'sum', 'max', 'min', 'count') if action is 'aggregate'.
        - group_by: (string, optional) Field to group by (e.g., "direction", "month", "quarter", "year").
        
        EXAMPLES:
        Question: "How many LONG days were there in 2023?"
        Response: {{"action": "count", "columns": [], "filters": {{"timestamp": {{"start": "2023-01-01", "end": "2023-12-31"}}, "direction": "LONG"}}, "metric": "count", "group_by": null}}
        
        Question: "What was the average closing price in January 2024?"
        Response: {{"action": "aggregate", "columns": ["close"], "filters": {{"timestamp": {{"start": "2024-01-01", "end": "2024-01-31"}}}}, "metric": "mean", "group_by": null}}

        Question: "Highest price last week?" (Assume current date is 2024-06-10 for this example)
        Response: {{"action": "aggregate", "columns": ["high"], "filters": {{"timestamp": {{"start": "2024-06-03", "end": "2024-06-09"}}}}, "metric": "max", "group_by": null}}

        Question: "Compare average volume for LONG vs SHORT signals"
        Response: {{"action": "compare", "columns": ["volume"], "filters": {{}}, "metric": "mean", "group_by": "direction"}}

        IMPORTANT:
        - Use ISO date format (YYYY-MM-DD) for all dates in filters.
        - Ensure the JSON is valid. If a field is not applicable, use null or omit it if appropriate for the structure.
        - Do not add any explanations or text outside the JSON object.
        - If the question implies a time range (e.g. "last month", "this year"), calculate the specific start and end dates.
        
        RESPONSE (JSON only):
        """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing question with Gemini: {e}")
        return None

# `execute_structured_query` remains largely the same, it operates on the df and query_json.
# It needs to be robust to missing columns specified in query_json['columns'] if AI hallucinates.
def execute_structured_query(df, query_json):
    if df.empty or not query_json:
        return "No data available or invalid query structure."
    try:
        filtered_df = df.copy()
        action = query_json.get('action', '')
        metric = query_json.get('metric', '')
        columns = [col for col in query_json.get('columns', []) if col in df.columns] # Ensure columns exist
        group_by = query_json.get('group_by')
        if group_by and group_by not in df.columns and group_by not in ['month', 'quarter', 'year']: # check if group_by col exists
            group_by = None # Invalidate group_by if column doesn't exist

        # Apply filters
        if 'filters' in query_json and query_json['filters']:
            filters = query_json['filters']
            for col, cond in filters.items():
                if col == 'timestamp' and isinstance(cond, dict): # Date range
                    if 'start' in cond and cond['start']:
                        filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(cond['start'])]
                    if 'end' in cond and cond['end']:
                        filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(cond['end'])]
                elif col in df.columns: # Other column filters
                    if isinstance(cond, dict): # e.g. "close": {">": 150}
                        for op, val in cond.items():
                            if op == '>': filtered_df = filtered_df[filtered_df[col] > val]
                            elif op == '<': filtered_df = filtered_df[filtered_df[col] < val]
                            elif op == '>=': filtered_df = filtered_df[filtered_df[col] >= val]
                            elif op == '<=': filtered_df = filtered_df[filtered_df[col] <= val]
                            elif op == '==': filtered_df = filtered_df[filtered_df[col] == val]
                            elif op == '!=': filtered_df = filtered_df[filtered_df[col] != val]
                    else: # e.g. "direction": "LONG"
                        filtered_df = filtered_df[filtered_df[col] == cond]
        
        if filtered_df.empty: return "No data found for the specified filters."
        
        result = {}
        if action == 'count':
            result['count'] = len(filtered_df)
            if columns: # Count non-nulls in specified columns
                 for col in columns: result[f'{col}_count_non_null'] = filtered_df[col].count()

        elif action == 'aggregate' and columns:
            for col in columns:
                if metric == 'mean': result[f'avg_{col}'] = filtered_df[col].mean()
                elif metric == 'sum': result[f'sum_{col}'] = filtered_df[col].sum()
                elif metric == 'max': result[f'max_{col}'] = filtered_df[col].max()
                elif metric == 'min': result[f'min_{col}'] = filtered_df[col].min()
                elif metric == 'median': result[f'median_{col}'] = filtered_df[col].median()
                elif metric == 'std': result[f'std_{col}'] = filtered_df[col].std()
                else: result[f'avg_{col}'] = filtered_df[col].mean() # Default to mean
        
        elif action == 'describe' and columns:
            for col in columns: result[f'{col}_stats'] = filtered_df[col].describe().to_dict()

        elif action == 'compare' and group_by and columns:
            # Temporal grouping
            df_temp_group = filtered_df.copy()
            if group_by in ['month', 'quarter', 'year'] and 'timestamp' in df_temp_group.columns:
                if group_by == 'month': df_temp_group['grouping_key'] = df_temp_group['timestamp'].dt.to_period('M')
                elif group_by == 'quarter': df_temp_group['grouping_key'] = df_temp_group['timestamp'].dt.to_period('Q')
                elif group_by == 'year': df_temp_group['grouping_key'] = df_temp_group['timestamp'].dt.to_period('Y')
                actual_group_by_col = 'grouping_key'
            elif group_by in df_temp_group.columns:
                 actual_group_by_col = group_by
            else: # group_by column not valid
                return {"error": f"Grouping column '{group_by}' not found or invalid."}


            for col in columns:
                grouped_data = df_temp_group.groupby(actual_group_by_col)[col]
                if metric == 'mean': result[f'{col}_avg_by_{group_by}'] = grouped_data.mean().to_dict()
                elif metric == 'sum': result[f'{col}_sum_by_{group_by}'] = grouped_data.sum().to_dict()
                elif metric == 'count': result[f'{col}_count_by_{group_by}'] = grouped_data.count().to_dict()
                elif metric == 'max': result[f'{col}_max_by_{group_by}'] = grouped_data.max().to_dict()
                elif metric == 'min': result[f'{col}_min_by_{group_by}'] = grouped_data.min().to_dict()
                else: result[f'{col}_avg_by_{group_by}'] = grouped_data.mean().to_dict()
        
        result['metadata'] = {
            'filtered_rows': len(filtered_df), 'total_rows_in_original_df': len(df),
            'date_range_of_filtered_data': {
                'start': filtered_df['timestamp'].min().strftime('%Y-%m-%d') if not filtered_df.empty else None,
                'end': filtered_df['timestamp'].max().strftime('%Y-%m-%d') if not filtered_df.empty else None
            }
        }
        return json.loads(json.dumps(result, default=str)) # Ensure serializable
    except Exception as e:
        st.error(f"Error executing structured query: {e}")
        st.exception(e)
        return {"error": str(e)}


# `format_query_results` remains largely the same, it's about presenting JSON nicely.

def get_ai_response(model, df, summary, question, available_columns):
    """Process a natural language question using the new structured approach or fallback"""
    try:
        parsed_query = parse_question_with_gemini(model, question, available_columns)

        if not parsed_query:
            st.warning("AI parsing failed. Using fallback analysis method based on keywords.")
            # Fallback to the old direct analysis method
            specific_analysis = analyze_data_for_question(df, question, available_columns)
            context = create_context_prompt(summary, question, specific_analysis, available_columns)
            response = model.generate_content(context)
            return response.text

        query_results = execute_structured_query(df, parsed_query)
        # Pass available_columns also to format_query_results if it needs to reference them for context
        formatted_response = format_query_results(model, question, query_results) # Definition of format_query_results not shown here but assumed to exist
        return formatted_response

    except Exception as e:
        st.error(f"AI Error in get_ai_response: {e}")
        st.exception(e)
        try:
            st.warning("Critical AI error, attempting simpler fallback.")
            specific_analysis = analyze_data_for_question(df, question, available_columns)
            context = create_context_prompt(summary, question, specific_analysis, available_columns)
            response = model.generate_content(context)
            return response.text
        except Exception as final_fallback_e:
            st.error(f"Complete AI fallback failed: {final_fallback_e}")
            return f"❌ Error generating response: {str(final_fallback_e)}"

# Definition for format_query_results (from original code, slightly adapted)
def format_query_results(model, question, query_results):
    """Format query results into a natural language response using Gemini with improved formatting"""
    if not model or not query_results:
        return "I couldn't analyze that question or the results are empty. Please try a different question."
    
    # Check if query_results is just an error message string
    if isinstance(query_results, str) and ("No data found" in query_results or "invalid query" in query_results):
        return query_results
    if isinstance(query_results, dict) and "error" in query_results:
        return f"An error occurred while processing your query: {query_results['error']}"


    try:
        prompt = f"""
        You are a specialized financial analyst.
        Analyze the following query results and provide a well-formatted response that answers the user's question.
        
        USER QUESTION: {question}
        
        QUERY RESULTS (JSON): {json.dumps(query_results, indent=2)}
        
        FORMAT YOUR RESPONSE IN CLEAR SECTIONS if applicable (e.g., Direct Answer, Details, Interpretation).
        
        CRITICAL FORMATTING RULES:
        - Format large numbers with commas and prices with a currency symbol (e.g., $) if appropriate.
        - Format percentages properly (e.g., 24.5%).
        - Round decimal numbers to 2-4 places for readability, as appropriate for the metric.
        - Use proper spacing. NEVER run words together.
        - Use short, clear sentences.
        
        EXAMPLE OF PROPER FORMATTING:
        "For January 2024, the average closing price was $176.34. The stock ranged from a low of $162.51 to a high of $202.57.
        This indicates a period of moderate volatility. The analysis covered 20 trading days within this month."
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error formatting results: {e}")
        return f"Error generating response: Unable to format results properly. Raw results: {query_results}"