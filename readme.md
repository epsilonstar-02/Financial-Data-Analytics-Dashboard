# Financial Data Analysis Dashboard 📊🤖

## Overview

This project is a Streamlit-based web application designed for interactive analysis of financial time-series data, typically OHLC (Open, High, Low, Close) data. It provides an interactive candlestick chart with optional support/resistance bands and trading signal markers. A key feature is its AI-powered data analysis assistant, which allows users to ask natural language questions about their uploaded data and receive insightful answers.

The dashboard is built to be generic, meaning it can work with any CSV file containing the necessary OHLC and timestamp columns, with flexible column name mapping.

## Features

*   **Interactive Candlestick Charts:** Visualize OHLC data with zooming and panning capabilities.
*   **Custom CSV Upload:** Users can upload their own financial data in CSV format.
*   **Flexible Column Mapping:** The application attempts to automatically map common column names for timestamp and OHLC data.
*   **Optional Support & Resistance Bands:** If your data includes support and resistance levels (as lists of values in specified columns), these can be visualized as floating bands on the chart. This feature can be enabled/disabled.
*   **Optional Trading Signal Markers:** If your data includes a 'direction' column (e.g., LONG, SHORT, None), these signals can be displayed as markers on the chart.
*   **AI-Powered Data Analysis:**
    *   Leverages Google's Gemini API to understand and answer natural language questions about the uploaded dataset.
    *   Parses questions into structured queries to extract relevant information.
    *   Provides formatted, natural language responses.
*   **Data Summary & Overview:** Displays key statistics and a sample of the uploaded data.
*   **Modular Codebase:** Organized into separate Python files for configuration, data processing, charting, and AI logic for better maintainability.
*   **Dark Theme UI:** Aesthetically pleasing and modern user interface.

## File Structure

The project is organized into the following Python files:

*   **`app.py`**: The main Streamlit application script. It handles the UI layout, user interactions, and orchestrates calls to other modules.
*   **`config.py`**: Manages Streamlit page configuration and Google Gemini API setup (including loading API keys from a `.env` file).
*   **`data_processing.py`**: Contains all logic for loading, cleaning, transforming, and summarizing the CSV data. This includes processing OHLC, timestamp, and optional S/R and direction data.
*   **`charting.py`**: Includes functions for creating the candlestick chart configuration, markers for trading signals, and the floating support/resistance band components.
*   **`ai_agent.py`**: Houses all functionalities related to the AI assistant, such as parsing natural language questions into structured queries, executing these queries against the data, and formatting the results into user-friendly responses using the Gemini API.
*   **`.env` (User-created)**: Stores the Google Gemini API key. This file is not included in the repository and must be created by the user.

## Setup and Installation

### Prerequisites

*   Python 3.8 or higher.
*   Access to Google Gemini API and an API key.

### Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages using the provided `requirements.txt` file (if available) or install them manually:
    ```bash
    pip install streamlit pandas python-dotenv google-generativeai streamlit-lightweight-charts
    ```
    If a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google Gemini API Key:**
    *   Create a file named `.env` in the root directory of the project.
    *   Add your Gemini API key to this file in the following format:
        ```
        GEMINI_API_KEY=your_actual_api_key_here
        ```
    *   You can obtain a Gemini API key from [Google AI Studio (formerly MakerSuite)](https://makersuite.google.com/app/apikey).

## Usage

1.  **Run the Streamlit Application:**
    Navigate to the project's root directory in your terminal and run:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Upload CSV Data:**
    *   Use the sidebar "Upload OHLC CSV Data" button to select and upload your financial data file.
    *   Ensure your CSV file adheres to the expected format (see [CSV Data Format](#csv-data-format) below).

3.  **Configure Options (Sidebar):**
    *   **Enable Support/Resistance Analysis:** Check this box if your CSV contains support and resistance data and you want to visualize it.
    *   **S/R Column Names:** If S/R analysis is enabled, you can specify the exact (case-insensitive) names of the columns in your CSV that contain the raw support and resistance lists. Defaults are 'Support' and 'Resistance'.

4.  **Interact with Tabs:**
    *   **📊 Interactive Chart:** View the candlestick chart. If S/R or direction data is available and processed, it will be displayed. The chart is zoomable and pannable.
    *   **🤖 AI Analysis:**
        *   Ask natural language questions about your data in the text area (e.g., "What was the average closing price last month?", "How many bullish days were there in 2023?").
        *   Use the example questions as a starting point.
        *   Click "🚀 Get AI Analysis" to receive a response.
        *   View and clear chat history.
    *   **📋 Data Overview:** See a summary of your dataset, including total records, date range, price statistics, and (if applicable) trading signal distribution and volume stats. You can also view a sample of the processed data and download it.

## CSV Data Format

The application expects a CSV file with the following columns:

### Required Columns:

*   **Timestamp Column:** Contains date or datetime information for each data point.
    *   *Common Names (case-insensitive):* `timestamp`, `Date`, `TIME`, `Datetime`, `datetime`
*   **OHLC Columns:** Standard Open, High, Low, Close price data.
    *   *Common Names (case-insensitive):*
        *   Open: `open`, `Open`, `OPEN`
        *   High: `high`, `High`, `HIGH`
        *   Low: `low`, `Low`, `LOW`
        *   Close: `close`, `Close`, `CLOSE`

### Optional Columns:

*   **Volume Column:** Trading volume.
    *   *Common Names (case-insensitive):* `volume`, `Volume`, `VOLUME`
*   **Direction Column:** Trading signals.
    *   *Common Names (case-insensitive):* `direction`, `Direction`, `Signal`
    *   *Expected Values:* `LONG`, `SHORT`, `None` (or other strings treated as neutral/other). If this column is not present, no signal markers will be shown.
*   **Raw Support List Column:** (Used if "Enable Support/Resistance Analysis" is checked)
    *   *Default Name (case-insensitive):* `Support` (configurable in sidebar)
    *   *Format:* A string representation of a Python list of numerical support levels, e.g., `"[150.0, 150.5, 149.8]"` or `"[150]"`. Empty lists `[]` or empty strings are also handled.
*   **Raw Resistance List Column:** (Used if "Enable Support/Resistance Analysis" is checked)
    *   *Default Name (case-insensitive):* `Resistance` (configurable in sidebar)
    *   *Format:* Similar to the support list, e.g., `"[160.0, 160.5]"`.

The application will attempt to find these columns using the common names. If your column names differ significantly, you might need to adjust the `COLUMN_MAP` in `app.py` or rename your CSV columns.

## Customization

*   **Column Name Mapping:** The `COLUMN_MAP` dictionary at the beginning of `app.py` can be extended with more variants if your CSV files use different common names for standard columns.
*   **Support/Resistance Column Names:** These can be configured directly in the Streamlit application's sidebar when S/R processing is enabled.

## AI Agent Details

*   The AI assistant uses the Google Gemini API (specifically, the `gemini-1.5-flash` model by default, which can be changed in `config.py`).
*   It works by:
    1.  Receiving your natural language question.
    2.  Being provided with a list of available columns in your current dataset.
    3.  Using a prompt to ask the Gemini model to convert your question into a structured JSON query.
    4.  Executing this JSON query locally on your Pandas DataFrame.
    5.  Sending the query results back to the Gemini model with another prompt to format them into a human-readable answer.
*   **Example AI Questions:**
    *   "What was the highest closing price in March 2023?"
    *   "How many records are there between 2022-01-01 and 2022-06-30?"
    *   "Show me the average volume per month for last year." (if 'volume' column exists)
    *   "What is the percentage of LONG signals?" (if 'direction' column exists)

## Troubleshooting & Notes

*   **Gemini API Key:** Ensure your `.env` file is correctly set up with a valid API key. The application will show an error if the key is missing or invalid.
*   **CSV Format:** If data loading fails, double-check your CSV file for correct column names and data types, especially for support/resistance lists if enabled.
*   **Internet Connection:** An internet connection is required for the AI features to communicate with the Gemini API.
*   **Data Volume:** Extremely large CSV files might lead to performance issues in the browser or during processing.
*   **AI Accuracy:** While powerful, AI responses should always be cross-verified, especially for critical financial decisions. The AI's ability to parse questions and generate queries depends on the clarity of the question and the structure of the data.

## Future Enhancements (Ideas)

*   More advanced technical indicators (e.g., Moving Averages, RSI, MACD).
*   User-defined column mapping via the UI.
*   Saving/loading analysis sessions or chart configurations.
*   Direct database connections as a data source.
*   More sophisticated error handling and feedback for AI query generation.
*   Export chart as image.
*   Date range selector for the chart.

## License

This project can be considered under the MIT License (or specify your preferred license).