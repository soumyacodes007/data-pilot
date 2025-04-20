import streamlit as st
import pandas as pd
import sweetviz as sv
import dtale
import google.generativeai as genai
import os
import io
import webbrowser
from dotenv import load_dotenv
import logging
import time # To add slight delay for dtale if needed

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env file

# --- Constants ---
SWEETVIZ_REPORT_PATH = "streamlit_sweetviz_report.html"

# --- Helper Functions (Adapted from previous script) ---

# Use Streamlit's caching to avoid reloading data unnecessarily
@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded file object."""
    try:
        logging.info(f"Attempting to load data from uploaded file: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        if df.empty:
            logging.warning("The loaded DataFrame is empty.")
            st.warning("The uploaded CSV file is empty.")
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"Error: The file {uploaded_file.name} is empty.")
        st.error(f"Error: The uploaded file '{uploaded_file.name}' is empty.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data: {e}")
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Caching the report generation might be heavy, let's generate on demand
# @st.cache_resource # Use cache_resource if generation is very slow and df doesn't change often
def generate_sweetviz_report(df: pd.DataFrame, report_path: str):
    """Generates and saves a Sweetviz HTML report."""
    if df is None or df.empty:
        st.warning("Cannot generate Sweetviz report: DataFrame is empty or invalid.")
        return None
    try:
        logging.info(f"Generating Sweetviz report (saving as {report_path})...")
        report = sv.analyze(df)
        report.show_html(report_path, open_browser=False, layout='vertical') # Use vertical layout often better for embedding
        logging.info(f"Sweetviz report saved to: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"An error occurred during Sweetviz report generation: {e}")
        st.error(f"Error generating Sweetviz report: {e}")
        return None

# This should not be cached as it relies on external state (dtale server)
def launch_dtale(df: pd.DataFrame):
    """Launches a dtale instance in a new browser tab."""
    if df is None or df.empty:
        st.warning("Cannot launch dtale: DataFrame is empty or invalid.")
        return None
    try:
        logging.info("Launching dtale instance...")
        # Kill existing dtale instances for the same data to avoid port conflicts (optional, can be aggressive)
        # dtale.instances().shutdown() # Use with caution
        d = dtale.show(df, open_browser=False) # Don't automatically open, we will try
        logging.info(f"dtale instance should be running at: {d._main_url}")
        try:
            webbrowser.open(d._main_url, new=2) # new=2 tries to open in a new tab
            st.success(f"dtale dashboard launched in a new tab! URL: {d._main_url}")
            st.info("You might need to manually stop the underlying 'dtale' process later if it doesn't close automatically.")
        except Exception as web_err:
            logging.warning(f"Could not automatically open browser: {web_err}")
            st.warning(f"Could not automatically open browser. Please manually navigate to: {d._main_url}")
        return d._main_url
    except Exception as e:
        logging.error(f"An error occurred while launching dtale: {e}")
        st.error(f"Error launching dtale: {e}")
        return None

# Cache the summary generation based on the DataFrame content
@st.cache_data
def get_data_summary(df: pd.DataFrame) -> str | None:
    """Generates a text summary of basic DataFrame statistics."""
    if df is None or df.empty:
        logging.warning("Cannot generate summary for empty or invalid DataFrame.")
        return None

    logging.info("Generating basic data statistics for Gemini...")
    summary = []
    summary.append(f"Dataset Shape: Rows={df.shape[0]}, Columns={df.shape[1]}")
    summary.append("\n--- Column Names ---")
    summary.append(", ".join(df.columns.tolist()))

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    summary.append("\n--- Data Types & Non-Null Counts ---")
    summary.append(info_str)

    summary.append("\n--- Descriptive Statistics (Numerical) ---")
    try:
        desc_num = df.describe(include='number').to_string()
        summary.append(desc_num)
    except Exception:
         summary.append("No numerical columns found or error generating stats.")

    summary.append("\n--- Descriptive Statistics (Categorical/Object) ---")
    try:
        desc_obj = df.describe(include=['object', 'category']).to_string()
        summary.append(desc_obj)
    except Exception:
        summary.append("No object or category columns found or error generating stats.")

    summary.append("\n--- Missing Values (Top 5 Columns with Missing) ---")
    missing_values = df.isnull().sum()
    missing_filtered = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_filtered.empty:
         summary.append(missing_filtered.head().to_string()) # Show top 5
         if len(missing_filtered) > 5:
              summary.append(f"...and {len(missing_filtered)-5} more columns with missing values.")
         summary.append(f"\nTotal missing cells: {df.isnull().sum().sum()}")
    else:
        summary.append("No missing values found.")

    return "\n".join(summary)

# Cache the Gemini response based on the summary text and API key presence
# NOTE: Caching based on API key is tricky; if key changes, cache might be invalid.
# For simplicity here, we cache based on the summary only.
# A better approach might involve session state or more complex caching keys.
@st.cache_data # Cache based on the input summary text
def generate_gemini_insights(data_summary: str, api_key: str) -> str | None:
    """Uses Gemini API to generate insights based on the data summary."""
    if not api_key:
        st.error("Google API Key not provided or found. Cannot generate insights.")
        return "Error: Google API Key not configured."
    if not data_summary:
        st.warning("No data summary available to send to Gemini.")
        return "Error: No data summary available to analyze."

    try:
        logging.info("Configuring Gemini API...")
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048,
        }
        safety_settings = [
            {"category": f"HARM_CATEGORY_{cat}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for cat in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-flash", # Using Flash model - usually faster and cheaper
                                   generation_config=generation_config,
                                   safety_settings=safety_settings)

        prompt = f"""
        You are an expert data analyst assisting with Exploratory Data Analysis (EDA) in a Streamlit app.
        Based only on the following statistical summary of a dataset provided by the user, generate a concise analysis presented in Markdown format.

        Focus on:
        1.  *Data Overview:* Briefly describe the dataset size (rows, columns) and general data types mentioned.
        2.  *Potential Data Quality Issues:* Highlight missing values (mentioning count or specific columns if provided), potential outliers suggested by min/max/std dev, or unusual distributions indicated by the stats.
        3.  *Key Observations:* Mention any immediate insights suggested by the descriptive statistics (e.g., ranges, common categories, unique value counts if available).
        4.  *Suggestions for Next Steps:* Recommend 2-3 specific, actionable next steps for the user to perform within the context of typical EDA (e.g., "Visualize the distribution of column X," "Investigate the relationship between columns Y and Z," "Address missing values in column A," "Examine the unique values in categorical column B"). Keep suggestions general enough to apply to many datasets.

        *Statistical Summary Provided:*
        
        {data_summary}
        

        *Your Concise Markdown Analysis:*
        """

        logging.info("Sending request to Gemini API...")
        response = model.generate_content(prompt)
        logging.info("Received response from Gemini API.")

        # Accessing the text part correctly
        if response and response.parts:
             return response.text # Use .text attribute for gemini-pro and recent models
        elif hasattr(response, 'text'): # Fallback just in case
             return response.text
        else:
             # If the structure is different, you might need to inspect response
             logging.error(f"Unexpected Gemini response structure: {response}")
             return "Error: Could not parse the response from Gemini."


    except Exception as e:
        logging.error(f"An error occurred while interacting with the Gemini API: {e}")
        st.error(f"An error occurred connecting to Gemini: {e}")
        return f"Error: Could not generate insights using Gemini. Details: {e}"

# --- Streamlit App UI ---

# Adjust the function name to match the call in app.py
def render_eda_tab(st_module):
    """Renders the EDA components within a Streamlit tab.

    Args:
        st_module: The Streamlit module (passed as 'st').
    """
    # Use st_module instead of st directly within this function
    st_module.title("📊 Automated EDA Bot")
    st_module.markdown("Explore your data using Sweetviz, dtale, and get AI-powered insights from Google Gemini.")

    # --- Sidebar is handled in app.py, but we need API key and file --- 
    # --- We'll rely on app.py managing the data loading via session state --- 
    
    # Retrieve necessary items from session state (assuming they are set in app.py)
    df = st_module.session_state.get('df', None)
    api_key = st_module.session_state.get('google_api_key', None)
    
    if df is None:
        st_module.warning("No data loaded. Please upload a CSV file in the main app sidebar.")
        return # Exit if no data

    # --- Main Area within the Tab --- 
    st_module.success(f"Using loaded dataset ({df.shape[0]} rows, {df.shape[1]} columns)")

    # Initialize session state variables if they don't exist (prefix to avoid potential conflicts)
    if 'eda_gemini_insights' not in st_module.session_state:
        st_module.session_state.eda_gemini_insights = None
    if 'eda_sweetviz_report_path' not in st_module.session_state:
        st_module.session_state.eda_sweetviz_report_path = None
    if 'eda_dtale_url' not in st_module.session_state:
         st_module.session_state.eda_dtale_url = None


    tab1, tab2, tab3, tab4 = st_module.tabs([" Glimpse Data ", " Sweetviz Report ", " Gemini AI Analysis ", " dtale Interactive "])

    with tab1:
        st_module.subheader("Data Preview (First 10 Rows)")
        st_module.dataframe(df.head(10))
        st_module.subheader("Basic Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st_module.text(buffer.getvalue())


    with tab2:
        st_module.subheader("Sweetviz EDA Report")
        if st_module.button("Generate Sweetviz Report", key="gen_sweetviz_btn"):
            with st_module.spinner("Generating Sweetviz report... This might take a moment."):
                report_file = generate_sweetviz_report(df, SWEETVIZ_REPORT_PATH)
                st_module.session_state.eda_sweetviz_report_path = report_file

        if st_module.session_state.eda_sweetviz_report_path:
            try:
                with open(st_module.session_state.eda_sweetviz_report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st_module.components.v1.html(html_content, height=600, scrolling=True)
                st_module.success(f"Report generated: {st_module.session_state.eda_sweetviz_report_path}")
                # Add download button
                with open(st_module.session_state.eda_sweetviz_report_path, "rb") as fp:
                    st_module.download_button(
                        label="Download Report (HTML)",
                        data=fp,
                        file_name="sweetviz_report.html",
                        mime="text/html"
                    )
            except FileNotFoundError:
                st_module.error("Report file not found. Please generate it again.")
                st_module.session_state.eda_sweetviz_report_path = None # Reset state
            except Exception as e:
                st_module.error(f"Could not display report: {e}")
        else:
            st_module.info("Click the button above to generate the Sweetviz report.")

    with tab3:
        st_module.subheader("Gemini AI Analysis")
        if not api_key:
             st_module.warning("Google Gemini API Key not found. Please configure it in the main app sidebar to enable AI analysis.")
        else:
             if st_module.button("Generate Gemini Insights", key="gen_gemini_btn"):
                 with st_module.spinner("Querying Gemini AI for insights... Please wait."):
                     data_summary = get_data_summary(df)
                     if data_summary:
                         insights = generate_gemini_insights(data_summary, api_key)
                         st_module.session_state.eda_gemini_insights = insights
                     else:
                         st_module.error("Could not generate data summary for Gemini.")
                         st_module.session_state.eda_gemini_insights = None

             if st_module.session_state.eda_gemini_insights:
                 st_module.markdown(st_module.session_state.eda_gemini_insights)
                 # Add download button for insights
                 st_module.download_button(
                      label="Download Insights (TXT)",
                      data=st_module.session_state.eda_gemini_insights,
                      file_name="gemini_eda_insights.txt",
                      mime="text/plain"
                 )
             else:
                 st_module.info("Click the button above to get AI-powered insights based on your data's statistics.")

    with tab4:
        st_module.subheader("Interactive Exploration with dtale")
        st_module.info("Launch an interactive dashboard for deeper data exploration in a new browser tab.")
        if st_module.button("Launch dtale Dashboard", key="launch_dtale_btn"):
            with st_module.spinner("Launching dtale... check for a new browser tab."):
                 dtale_url = launch_dtale(df)
                 st_module.session_state.eda_dtale_url = dtale_url # Store URL even if browser fails to open
                 if dtale_url:
                     st_module.markdown(f"dtale is running at: [{dtale_url}]({dtale_url})")
                 else:
                     st_module.error("Failed to launch dtale.")

        # Display the URL if it was successfully launched previously in the session
        if st_module.session_state.get('eda_dtale_url'):
            st_module.markdown(f"*Previous dtale instance URL (might still be running):* [{st_module.session_state.eda_dtale_url}]({st_module.session_state.eda_dtale_url})")