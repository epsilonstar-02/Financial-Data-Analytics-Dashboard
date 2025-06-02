# trading_dashboard/ai_services.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime

# --- Fallback Analysis (Simplified version of original analyze_data_for_question) ---
def _analyze_data_for_fallback(df: pd.DataFrame, question: str) -> dict:
    """Simplified data analysis for fallback AI context."""
    if df.empty:
        return {"error": "No data available for analysis."}
    
    analysis_data = {}
    question_lower = question.lower()

    if '2023' in question_lower and 'timestamp' in df.columns and not df[df['timestamp'].dt.year == 2023].empty:
        df_2023 = df[df['timestamp'].dt.year == 2023].copy()
        analysis_data['2023_summary'] = {
            'count': len(df_2023),
            'avg_close': df_2023['close'].mean() if 'close' in df_2023 else 'N/A',
            'high': df_2023['high'].max() if 'high' in df_2023 else 'N/A',
        }
    
    if 'price' in question_lower and not df.empty:
        analysis_data['overall_price_info'] = {
            'max_high': df['high'].max(),
            'min_low': df['low'].min(),
            'latest_close': df['close'].iloc[-1]
        }
    return analysis_data


def _create_fallback_prompt(df: pd.DataFrame, summary: dict, user_question: str, specific_analysis: dict) -> str:
    """Creates a prompt for the AI when structured query fails or is not applicable."""
    sample_data_str = "No sample data available."
    if not df.empty:
        sample_data_str = df.head(3)[['timestamp', 'open', 'high', 'low', 'close', 'direction']].to_string()

    return f"""
You are a TSLA stock data analyst. Answer the user's question based on the provided data summary, a small data sample, and any specific analysis performed.

DATASET OVERVIEW:
{json.dumps(summary, indent=2)}

SAMPLE DATA (first 3 rows):
{sample_data_str}

SPECIFIC ANALYSIS FOR THIS QUESTION (if any):
{json.dumps(specific_analysis, indent=2)}

USER QUESTION: {user_question}

INSTRUCTIONS:
- Focus on the user's question.
- Use the data and analysis provided.
- Be concise and factual.
- If data for a specific query (e.g. a year not in the dataset) is not present, state that clearly.
- Provide a direct answer, then a brief explanation.
"""

# --- Structured Query Processing with Gemini ---

def parse_question_to_structured_query(model, question: str) -> dict | None:
    """Uses Gemini to parse a natural language question into a structured JSON query."""
    if not model:
        st.warning("AI Model not available for parsing question.")
        return None
    
    # Enhanced prompt with more examples and clearer instructions
    prompt = f"""
    You are an AI specializing in parsing natural language questions about Tesla (TSLA) stock data into structured JSON queries.
    
    QUESTION: "{question}"
    
    OUTPUT FORMAT: Return ONLY a valid JSON object. Do NOT include any explanatory text before or after the JSON.
    The JSON should have these fields:
    - "action": (string) Type of query: "aggregate", "filter_count", "describe", "compare_groups", "find_extreme".
    - "columns": (array of strings) Relevant column(s) for the action (e.g., ["close"], ["volume"]).
    - "filters": (object, optional) Conditions to filter data.
        - "timestamp": {{ "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }} (optional start/end)
        - "direction": "LONG" | "SHORT" | "NEUTRAL" (or null/None if not specified)
        - "price_condition": {{ "column": "high", "operator": ">", "value": 200 }} (e.g., high > 200)
    - "metric": (string, for "aggregate" or "compare_groups") "mean", "sum", "max", "min", "count", "volatility".
    - "group_by": (string, for "compare_groups", optional) Column to group by (e.g., "direction", "month", "quarter", "year").
    - "extreme_type": (string, for "find_extreme") "highest" or "lowest".
    
    AVAILABLE DATASET COLUMNS (all lowercase for parsing):
    - timestamp (datetime)
    - open, high, low, close (numeric)
    - volume (numeric, if available, assume not for now if not mentioned)
    - direction (string: "LONG", "SHORT", "None" or null for neutral)
    - support_low, support_high, res_low, res_high (numeric band boundaries)

    EXAMPLES:
    1. Question: "How many LONG days were there in 2024?"
       Response: {{"action": "filter_count", "columns": [], "filters": {{"timestamp": {{"start": "2024-01-01", "end": "2024-12-31"}}, "direction": "LONG"}}, "metric": "count"}}
    2. Question: "Average close price in Q3 2023?"
       Response: {{"action": "aggregate", "columns": ["close"], "filters": {{"timestamp": {{"start": "2023-07-01", "end": "2023-09-30"}}}}, "metric": "mean"}}
    3. Question: "What was the highest high price overall?"
       Response: {{"action": "find_extreme", "columns": ["high"], "extreme_type": "highest"}}
    4. Question: "Compare average volume for LONG vs SHORT days." (Assume 'volume' column exists)
       Response: {{"action": "compare_groups", "columns": ["volume"], "metric": "mean", "group_by": "direction", "filters": {{"direction": ["LONG", "SHORT"]}}}}
    5. Question: "Details about closing prices in January 2024."
       Response: {{"action": "describe", "columns": ["close"], "filters": {{"timestamp": {{"start": "2024-01-01", "end": "2024-01-31"}}}}}}
    6. Question: "Count days where high was above $250 last year." (Assuming current year is 2024 for 'last year' interpretation, or model should ask for clarification if ambiguous)
       Response: {{"action": "filter_count", "columns": [], "filters": {{"timestamp": {{"start": "2023-01-01", "end": "2023-12-31"}}, "price_condition": {{"column": "high", "operator": ">", "value": 250}}}}, "metric": "count"}}

    RESPONSE (JSON only):
    """
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean Markdown code block fences if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        parsed_json = json.loads(result_text)
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"AI Error: Failed to parse AI's structured query (JSON decode error): {e}")
        st.caption(f"AI Raw Output:\n```\n{result_text}\n```")
        return None
    except Exception as e:
        st.error(f"AI Error: Unexpected error parsing question: {str(e)}")
        st.caption(f"AI Raw Output (if available):\n```\n{response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}\n```")
        return None


def execute_structured_query(df: pd.DataFrame, query_json: dict) -> dict:
    """Executes a structured query (JSON) on the DataFrame."""
    if df.empty or not query_json:
        return {"error": "No data or invalid query."}

    filtered_df = df.copy()
    
    # Apply filters
    if 'filters' in query_json and query_json['filters']:
        filters = query_json['filters']
        if 'timestamp' in filters and isinstance(filters['timestamp'], dict):
            ts_filter = filters['timestamp']
            if 'start' in ts_filter and ts_filter['start']:
                filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(ts_filter['start'])]
            if 'end' in ts_filter and ts_filter['end']:
                filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(ts_filter['end'])]
        
        if 'direction' in filters and filters['direction']:
            # Handle single string or list of directions
            directions_to_filter = filters['direction']
            if isinstance(directions_to_filter, str):
                if directions_to_filter.upper() == "NEUTRAL": # Handle neutral explicitly
                     filtered_df = filtered_df[filtered_df['direction'].isna() | (filtered_df['direction'].str.upper() == 'NONE')]
                else:
                    filtered_df = filtered_df[filtered_df['direction'].str.upper() == directions_to_filter.upper()]
            elif isinstance(directions_to_filter, list):
                directions_to_filter = [d.upper() for d in directions_to_filter]
                if "NEUTRAL" in directions_to_filter:
                    is_neutral = filtered_df['direction'].isna() | (filtered_df['direction'].str.upper() == 'NONE')
                    other_directions = [d for d in directions_to_filter if d != "NEUTRAL"]
                    if other_directions:
                         filtered_df = filtered_df[is_neutral | filtered_df['direction'].str.upper().isin(other_directions)]
                    else: # Only neutral
                        filtered_df = filtered_df[is_neutral]
                else:
                    filtered_df = filtered_df[filtered_df['direction'].str.upper().isin(directions_to_filter)]


        if 'price_condition' in filters and isinstance(filters['price_condition'], dict):
            cond = filters['price_condition']
            col, op, val = cond.get('column'), cond.get('operator'), cond.get('value')
            if col in filtered_df.columns and op and pd.notna(val):
                if op == '>': filtered_df = filtered_df[filtered_df[col] > val]
                elif op == '<': filtered_df = filtered_df[filtered_df[col] < val]
                elif op == '>=': filtered_df = filtered_df[filtered_df[col] >= val]
                elif op == '<=': filtered_df = filtered_df[filtered_df[col] <= val]
                elif op == '==': filtered_df = filtered_df[filtered_df[col] == val]
    
    if filtered_df.empty:
        return {"message": "No data matches the specified criteria.", "count": 0}

    action = query_json.get('action')
    columns = query_json.get('columns', [])
    metric = query_json.get('metric')
    results = {'filtered_record_count': len(filtered_df)}

    try:
        if action == 'filter_count':
            results['count'] = len(filtered_df)
            if columns: # if specific columns were mentioned for counting non-nulls
                 for col in columns:
                    if col in filtered_df.columns:
                        results[f'{col}_non_null_count'] = filtered_df[col].count()


        elif action == 'aggregate' and columns and metric:
            for col in columns:
                if col in filtered_df.columns:
                    if metric == 'mean': results[f'average_{col}'] = filtered_df[col].mean()
                    elif metric == 'sum': results[f'total_{col}'] = filtered_df[col].sum()
                    elif metric == 'max': results[f'maximum_{col}'] = filtered_df[col].max()
                    elif metric == 'min': results[f'minimum_{col}'] = filtered_df[col].min()
                    elif metric == 'count': results[f'count_{col}'] = filtered_df[col].count()
                    elif metric == 'volatility' and col == 'close': # Example custom metric
                        results['price_volatility_std_dev'] = filtered_df['close'].std()
                        results['price_volatility_range'] = filtered_df['high'].max() - filtered_df['low'].min()


        elif action == 'find_extreme' and columns and query_json.get('extreme_type'):
            extreme_type = query_json['extreme_type']
            for col in columns:
                if col in filtered_df.columns:
                    if extreme_type == 'highest':
                        results[f'highest_{col}'] = filtered_df[col].max()
                        results[f'date_of_highest_{col}'] = filtered_df.loc[filtered_df[col].idxmax()]['timestamp'].strftime('%Y-%m-%d')
                    elif extreme_type == 'lowest':
                        results[f'lowest_{col}'] = filtered_df[col].min()
                        results[f'date_of_lowest_{col}'] = filtered_df.loc[filtered_df[col].idxmin()]['timestamp'].strftime('%Y-%m-%d')
        
        elif action == 'describe' and columns:
            for col in columns:
                if col in filtered_df.columns:
                    results[f'{col}_statistics'] = filtered_df[col].describe().to_dict()

        elif action == 'compare_groups' and columns and metric and query_json.get('group_by'):
            group_by_col = query_json['group_by']
            
            # Temporal grouping enhancements
            temp_df = filtered_df.copy() # Work on a copy for adding temp columns
            if group_by_col == 'month': temp_df['month'] = temp_df['timestamp'].dt.month_name()
            elif group_by_col == 'quarter': temp_df['quarter'] = 'Q' + temp_df['timestamp'].dt.quarter.astype(str)
            elif group_by_col == 'year': temp_df['year'] = temp_df['timestamp'].dt.year
            # Add day_of_week if needed: temp_df['day_of_week'] = temp_df['timestamp'].dt.day_name()
            
            if group_by_col not in temp_df.columns: # If not an existing or created column
                return {"error": f"Grouping column '{group_by_col}' not found."}

            for col_to_agg in columns:
                if col_to_agg in temp_df.columns:
                    grouped_data = temp_df.groupby(group_by_col)[col_to_agg]
                    if metric == 'mean': agg_result = grouped_data.mean()
                    elif metric == 'sum': agg_result = grouped_data.sum()
                    elif metric == 'max': agg_result = grouped_data.max()
                    elif metric == 'min': agg_result = grouped_data.min()
                    elif metric == 'count': agg_result = grouped_data.count()
                    else: agg_result = pd.Series() # Empty series for unknown metric
                    
                    results[f'{metric}_{col_to_agg}_by_{group_by_col}'] = agg_result.to_dict()
        
        # Clean results for JSON serialization (e.g., convert NaNs, Timestamps)
        return json.loads(json.dumps(results, default=str, allow_nan=True))

    except Exception as e:
        return {"error": f"Error during query execution: {str(e)}"}


def format_query_results_for_display(model, question: str, query_results: dict) -> str:
    """Uses Gemini to format the structured query results into a natural language response."""
    if not model:
        return "AI Model not available for formatting results."
    if "error" in query_results: # If query execution itself had an error
        return f"I encountered an issue processing your request: {query_results['error']}. Please try rephrasing."
    if query_results.get("message") and query_results.get("count") == 0: # No data matched
        return query_results["message"]

    prompt = f"""
    You are an AI financial analyst, tasked with explaining TSLA stock data query results to a user.
    
    USER'S ORIGINAL QUESTION: "{question}"
    
    STRUCTURED QUERY RESULTS (JSON):
    {json.dumps(query_results, indent=2, default=str)}
    
    INSTRUCTIONS FOR YOUR RESPONSE:
    1.  Start with a "DIRECT ANSWER" section that concisely answers the user's question using key figures from the results.
        - Example: "DIRECT ANSWER: In Q1 2023, the average closing price for TSLA was $185.50."
    2.  Follow with a "DETAILED ANALYSIS" section that provides more context or related data points from the results, if available and relevant.
        - Example: "DETAILED ANALYSIS: During this period, the highest closing price reached $210.20 and the lowest was $160.80. There were 60 trading days included in this calculation."
    3.  Conclude with a brief "INTERPRETATION" section, offering a simple takeaway or implication of the findings.
        - Example: "INTERPRETATION: This suggests a period of significant price fluctuation for TSLA in early 2023."
    
    CRITICAL FORMATTING RULES:
    -   Use "$" for prices, round to 2 decimal places (e.g., $176.34).
    -   Format large numbers with commas (e.g., 1,234,567).
    -   Round other decimals to 1 or 2 places as appropriate.
    -   Use clear, simple language. Avoid jargon where possible.
    -   If the results indicate "No data matches...", state that clearly.
    -   Be specific and refer to the data. For example, if a date range was used, mention it.

    YOUR RESPONSE (Formatted as described):
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: Failed to format results into natural language: {str(e)}")
        return "I found some data, but had trouble summarizing it. Here's the raw data: \n" + json.dumps(query_results, indent=2, default=str)


def get_ai_assistant_response(model, df: pd.DataFrame, summary: dict, question: str) -> str:
    """Orchestrates AI interaction: parse, execute query, format response, with fallback."""
    if not model:
        return "AI assistant is currently unavailable. Please check API configuration."
    if df.empty:
        return "No data loaded. Please upload a TSLA data file."

    # Attempt structured query path
    st.write("🤖 AI is thinking... (Attempting structured query)")
    parsed_query_json = parse_question_to_structured_query(model, question)

    if parsed_query_json:
        st.write("⚙️ AI understood query structure. Executing...")
        # For debugging, show the parsed query:
        # st.expander("Parsed AI Query (JSON)").json(parsed_query_json)
        
        query_execution_results = execute_structured_query(df, parsed_query_json)
        
        # For debugging, show execution results:
        # st.expander("Query Execution Results (JSON)").json(query_execution_results)

        st.write("📝 Formatting results...")
        formatted_response = format_query_results_for_display(model, question, query_execution_results)
        return formatted_response
    else:
        # Fallback to general contextual understanding
        st.write("⚠️ Structured query failed. Trying general understanding...")
        specific_analysis = _analyze_data_for_fallback(df, question)
        fallback_prompt = _create_fallback_prompt(df, summary, question, specific_analysis)
        
        try:
            response = model.generate_content(fallback_prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"AI Error during fallback: {str(e)}")
            return "I encountered an issue trying to understand your question through the fallback method. Please try rephrasing."