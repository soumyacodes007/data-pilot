# code for data pilot \
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pickle
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error,
                             precision_score, recall_score, f1_score)

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- Import Custom Modules ---
import ml_agent # Added import
import model_tester # Import the new tester module
import eda # Add import for the new EDA module

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
CLASSIFICATION_MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'GradientBoostingClassifier': LogisticRegression(max_iter=1000, random_state=42)
}
REGRESSION_MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

# Basic Hyperparameter Grids (Keep small for speed)
PARAM_GRIDS = {
    'LogisticRegression': {'model__C': [0.1, 1.0, 10.0]},
    'RandomForestClassifier': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10, None]},
    'GradientBoostingClassifier': {'model__C': [0.1, 1.0, 10.0]},
    'LinearRegression': {}, # No typical hyperparameters to tune simply
    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
    'RandomForestRegressor': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10, None]},
    'GradientBoostingRegressor': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}
}

# --- Page Configuration ---
st.set_page_config(page_title="DataPilot: No-Code ML", layout="wide")
st.title("📊 DataPilot: No-Code ML Assistant")

# --- Session State Initialization ---
def initialize_session_state():
    keys_defaults = {
        'messages': [], 'df': None, 'target_variable': None, 'task_type': None,
        'numerical_cols': [], 'categorical_cols': [], '_last_target': None,
        'results': None, 'best_model': None, 'best_model_name': None,
        'llm_report': None, 'plots': None, 'api_key_provided': False,
        'agent_initialized': False, 'llm_initialized': False,
        'X_test': None, 'y_test': None, 'google_api_key': None,
        'feature_instructions': '', 'engineered_df': None, 
        'feature_code': None, 'feature_steps': None,
        'model_instructions': '', 'custom_model': None, 
        'custom_model_code': None, 'custom_model_results': None,
        'custom_model_name': None
    }
    for key, default in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session_state()

# --- Helper Functions ---

@st.cache_data # Cache the dataframe loading
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file, handling potential encoding issues."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error(f"Error reading CSV file with multiple encodings: {e}")
            return None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def get_column_types(df):
    """Identifies numerical and categorical columns."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols

def determine_task_type(df, target_variable):
    """Determines if the task is classification or regression based on target."""
    if not target_variable or target_variable not in df.columns:
        return None
    target_col = df[target_variable]
    target_dtype = target_col.dtype
    unique_values = target_col.nunique()

    # Heuristics for task type
    if pd.api.types.is_object_dtype(target_dtype) or pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_bool_dtype(target_dtype):
        return "Classification"
    elif pd.api.types.is_numeric_dtype(target_dtype):
        # If relatively few unique numeric values, could be classification (e.g., 0/1, labels 1-5)
        if unique_values <= min(20, len(target_col) * 0.1): # Adjust threshold as needed
             # Further check if values are integer-like
             if np.all(np.equal(np.mod(target_col.dropna(), 1), 0)):
                 return "Classification"
             else:
                 return "Regression" # Continuous numeric values
        else:
            return "Regression"
    return None

def create_preprocessing_pipeline(numerical_features, categorical_features, numeric_strategy='median', scale_numeric=True, numeric_scaler='standard'):
    """Creates a Scikit-learn preprocessing pipeline."""
    numeric_steps = [('imputer', SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        if numeric_scaler == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        else: # Default to standard scaler
             numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numerical_features:
        transformers.append(('num', numeric_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        # If no features, return a "passthrough" transformer or handle appropriately
        # For simplicity, we'll raise an error or return None if called with no features
        raise ValueError("No numerical or categorical features provided for preprocessing.")


    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # Pass through any columns not specified
    return preprocessor


def train_and_evaluate_models(df, target_variable, task_type, numerical_features, categorical_features, test_size=0.2, random_state=42):
    """Trains multiple models, tunes hyperparameters, evaluates, and compares."""
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Ensure target is numeric for classification models if needed (e.g., LabelEncoder)
    # Basic check: if classification and target is object/category, try converting
    if task_type == "Classification" and not pd.api.types.is_numeric_dtype(y.dtype):
         try:
             from sklearn.preprocessing import LabelEncoder
             le = LabelEncoder()
             y = le.fit_transform(y)
             st.info(f"Target variable '{target_variable}' encoded numerically for classification.")
             # Store encoder if needed for inverse transform later (optional for MVP)
             # st.session_state.label_encoder = le
         except Exception as e:
             st.error(f"Could not automatically encode target variable '{target_variable}'. Please ensure it's numeric or boolean for classification. Error: {e}")
             return None # Stop training if target encoding fails

    # Split data (stratify for classification)
    stratify_option = y if task_type == "Classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_option
        )
    except ValueError as e:
         if "stratify" in str(e):
             st.warning(f"Could not stratify split (target might have classes with only 1 member). Proceeding without stratification. Error: {e}")
             X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=test_size, random_state=random_state
             )
         else:
             raise e


    # Create preprocessor
    try:
        # Use RobustScaler for regression if outliers might be an issue
        numeric_scaler = 'robust' if task_type == 'Regression' else 'standard'
        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, numeric_scaler=numeric_scaler)
    except ValueError as e:
        st.error(f"Error creating preprocessing pipeline: {e}")
        return None

    models_to_run = CLASSIFICATION_MODELS if task_type == "Classification" else REGRESSION_MODELS
    results = {}
    primary_metric = 'accuracy' if task_type == "Classification" else 'r2' # Used for selecting best model
    scoring = 'accuracy' if task_type == "Classification" else 'r2' # Scoring for GridSearch

    for name, model in models_to_run.items():
        st.write(f"--- Training {name} ---")
        try:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            param_grid = PARAM_GRIDS.get(name, {})

            # Run GridSearchCV if a grid is defined
            if param_grid:
                grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=scoring, n_jobs=-1) # Use multiple cores if available
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
                best_score = grid_search.best_score_
                st.write(f"Best Params ({name}): {grid_search.best_params_}")
                st.write(f"Best CV Score ({scoring}, {name}): {best_score:.4f}")
            else:
                # If no grid, just fit the pipeline
                pipeline.fit(X_train, y_train)
                best_estimator = pipeline
                best_score = None # No CV score without GridSearch

            # Evaluate on test set
            y_pred = best_estimator.predict(X_test)

            model_results = {'model_object': best_estimator, 'best_cv_score': best_score}

            if task_type == "Classification":
                model_results['accuracy_test'] = accuracy_score(y_test, y_pred)
                model_results['precision_test'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                model_results['recall_test'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                model_results['f1_test'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                try: # Handle binary vs multiclass for report
                    model_results['classification_report_dict'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                except ValueError:
                    model_results['classification_report_dict'] = "Error generating report (check target format)"

            else: # Regression
                model_results['r2_test'] = r2_score(y_test, y_pred)
                model_results['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred))
                model_results['mae_test'] = mean_absolute_error(y_test, y_pred)

            results[name] = model_results

        except Exception as e:
            st.error(f"Error training model {name}: {e}")
            results[name] = {'error': str(e)} # Store error message


    # Find best model based on the primary test metric
    best_model_name = None
    best_metric_value = -np.inf # Initialize for maximization (like accuracy, R2)

    # Handle case where no models trained successfully
    valid_results = {name: res for name, res in results.items() if 'error' not in res and primary_metric+'_test' in res}

    if not valid_results:
        st.error("No models trained successfully.")
        return None

    for name, res in valid_results.items():
        metric_val = res.get(primary_metric + '_test', -np.inf)
        if metric_val > best_metric_value:
            best_metric_value = metric_val
            best_model_name = name

    if best_model_name:
         st.success(f"🏆 Best Model Identified: {best_model_name} ({primary_metric}: {best_metric_value:.4f})")
         best_model = results[best_model_name]['model_object']
    else:
         st.warning("Could not determine the best model.")
         best_model = None


    return {
        'all_model_results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'X_test': X_test,
        'y_test': y_test,
        'primary_metric': primary_metric,
        'task_type': task_type # Pass task type along
    }

def generate_evaluation_plots(best_model_results, best_model_name, X_test, y_test, task_type):
    """Generates evaluation plots for the best model."""
    plots = {}
    if not best_model_results or 'model_object' not in best_model_results:
        return plots # No model to plot

    model = best_model_results['model_object']
    y_pred = model.predict(X_test)

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    if task_type == "Classification":
        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            plots['confusion_matrix'] = buf
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate confusion matrix: {e}")

    else: # Regression
        # Actual vs Predicted Plot
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
            # Add diagonal line
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Actual vs. Predicted - {best_model_name}')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            plots['prediction_plot'] = buf
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate prediction plot: {e}")

    return plots

def get_llm(api_key):
    """Initializes the Gemini LLM via Langchain."""
    try:
        # Ensure genai is configured before using LangChain wrapper
        genai.configure(api_key=api_key)
        # Verify model access with a simple call directly to genai
        # model_check = genai.GenerativeModel('gemini-1.5-flash') # Use a reliable model for check
        # model_check.generate_content("test") # Simple check

        # Use the desired model for the LangChain wrapper
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", # Use flash for speed/cost balance
                                     temperature=0.3, # Slightly creative for reporting
                                     google_api_key=api_key,
                                     convert_system_message_to_human=True) # Good practice
        st.session_state.llm_initialized = True
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM. Check API Key and model access. Error: {e}")
        st.session_state.llm_initialized = False
        return None

# --- Function to generate Business Insights from Data ---
# Refactored function
def generate_llm_report(api_key: str, df: pd.DataFrame, target_variable: str, numerical_features: list, categorical_features: list):
    """Generates business-focused insights using an LLM based on data analysis."""
    if df is None or df.empty:
        return "No data available to generate insights."
    if not target_variable:
        return "Please select a target variable first."

    llm = get_llm(api_key)
    if not llm:
        return "LLM could not be initialized. Cannot generate insights."

    # --- Generate a concise data summary for the LLM ---
    data_summary_parts = []
    data_summary_parts.append(f"Dataset Overview: {df.shape[0]} customers, {df.shape[1]} data points per customer.")
    data_summary_parts.append(f"Business Goal: Understand factors influencing '{target_variable}'.")
    data_summary_parts.append(f"\nKey Data Points Tracked:")
    data_summary_parts.append(f"  - Numerical: {', '.join(numerical_features) if numerical_features else 'None'}")
    data_summary_parts.append(f"  - Categorical/Textual: {', '.join(categorical_features) if categorical_features else 'None'}")

    # Basic stats for numerical features
    if numerical_features:
        data_summary_parts.append("\nNumerical Data Summary (Averages):")
        try:
            # Show mean for numerical features, maybe grouped by target for comparison if target is suitable
            if df[target_variable].nunique() < 10 and not pd.api.types.is_object_dtype(df[target_variable]): # Basic check for suitable target grouping
                 grouped_means = df.groupby(target_variable)[numerical_features].mean().round(2)
                 data_summary_parts.append(grouped_means.to_string())
            else:
                 means = df[numerical_features].mean().round(2)
                 data_summary_parts.append(means.to_string())
        except Exception as e:
            data_summary_parts.append(f"(Could not generate numerical summary: {e})")

    # Value counts for key categorical features (limit to avoid overwhelming)
    if categorical_features:
        data_summary_parts.append("\nCategorical Data Summary (Most Common Categories):")
        cats_to_summarize = categorical_features[:5] # Limit to first 5 for brevity
        for cat in cats_to_summarize:
            try:
                top_cats = df[cat].value_counts().head(3) # Show top 3
                data_summary_parts.append(f"  - {cat}: {', '.join([f'{idx} ({val})' for idx, val in top_cats.items()])} ...")
            except Exception as e:
                 data_summary_parts.append(f"(Could not summarize {cat}: {e})")

    data_summary = "\n".join(data_summary_parts)

    # --- Build the NEW Prompt for Business Insights ---
    final_prompt = f"""
You are an experienced Business Analyst. You are given a summary of customer data for a business.
Your goal is to identify actionable insights directly from the data patterns to help the business achieve its goal related to '{target_variable}'.

*   Focus on identifying characteristics of different customer groups based on the data provided.
*   Highlight any patterns that seem strongly related to the target variable '{target_variable}'.
*   Suggest 2-3 specific, actionable business strategies based *only* on these data patterns. Examples: target specific customer segments for marketing, adjust pricing based on usage patterns, focus retention efforts on groups showing high risk, identify potential upselling opportunities.
*   Write clearly and concisely for a non-technical business audience. Avoid technical jargon like 'correlation', 'p-value', 'feature', 'model', etc. Use terms like 'customer characteristics', 'data points', 'patterns', 'trends'.
*   Base your analysis *only* on the summary provided below. Do not invent data or assume external knowledge.

**Data Summary:**
{data_summary}

**Your Business Analysis & Recommendations:**
"""

    # --- Invoke LLM ---
    try:
        system_message = SystemMessage(content="You are a Business Analyst providing actionable insights from data.")
        human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        chain = chat_prompt | llm
        response = chain.invoke({"user_prompt": final_prompt})
        report_text = response.content

        return report_text

    except Exception as e:
        st.error(f"Error generating report with LLM: {e}")
        return f"Error generating report: {e}"

# --- Main App Logic ---

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    # API Key Input
    api_key_input = st.text_input(
        "Enter your Google API Key:",
        type="password",
        key="api_key_input_field",
        help="Get your key from Google AI Studio. Required for AI features."
    )
    st.caption("Get key from [Google AI Studio](https://aistudio.google.com/)")

    # Use environment variable if available, otherwise use input
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY") or api_key_input
    st.session_state.api_key_provided = bool(st.session_state.google_api_key)

    if not st.session_state.api_key_provided:
        st.warning("⚠️ Google API Key needed for AI features.", icon="🔑")
    else:
        st.success("✅ API Key loaded.", icon="🔑")


    st.divider()

    # File Uploader
    uploaded_file = st.file_uploader("1. Upload your CSV file", type="csv", key="file_uploader")

    # Load Data and Update Session State
    if uploaded_file is not None:
        if st.session_state.df is None: # Only load if df is not already loaded
             st.session_state.df = load_data(uploaded_file)
             if st.session_state.df is not None:
                 st.success(f"CSV Loaded: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns.")
                 # Reset dependent states if new file uploaded
                 st.session_state.target_variable = None
                 st.session_state.task_type = None
                 st.session_state.results = None
                 st.session_state.llm_report = None
                 st.session_state.plots = None
             else:
                 st.error("Failed to load CSV.")
                 st.session_state.df = None # Ensure df is None if loading failed
    elif st.session_state.df is not None:
        # Allow user to clear the loaded data
        if st.button("Clear Loaded Data"):
             initialize_session_state() # Reset everything
             st.rerun()


    # Target Variable Selection
    if st.session_state.df is not None:
        columns = ["<Select>"] + list(st.session_state.df.columns)
        # Determine previous index if target exists
        prev_target = st.session_state.get('target_variable', None)
        prev_index = columns.index(prev_target) if prev_target in columns else 0

        target_variable_select = st.selectbox(
            "2. Select Target Variable (to predict)",
            options=columns,
            index=prev_index,
            key="target_variable_selector"
        )

        if target_variable_select != "<Select>":
            if target_variable_select != st.session_state.target_variable:
                 st.session_state.target_variable = target_variable_select
                 # Reset downstream states when target changes
                 st.session_state.task_type = None
                 st.session_state.results = None
                 st.session_state.llm_report = None
                 st.session_state.plots = None
                 st.rerun() # Rerun to update task type etc.
        else:
             st.session_state.target_variable = None

        if st.session_state.target_variable:
             st.write(f"Target: **{st.session_state.target_variable}**")
             # Determine Task Type after target selection
             if not st.session_state.task_type:
                  st.session_state.numerical_cols, st.session_state.categorical_cols = get_column_types(st.session_state.df)
                  st.session_state.task_type = determine_task_type(st.session_state.df, st.session_state.target_variable)
                  # Separate features from target
                  st.session_state.numerical_features = [c for c in st.session_state.numerical_cols if c != st.session_state.target_variable]
                  st.session_state.categorical_features = [c for c in st.session_state.categorical_cols if c != st.session_state.target_variable]

             if st.session_state.task_type:
                  st.success(f"Task Type: **{st.session_state.task_type}**")
             else:
                  st.error("Could not determine task type (Classification/Regression) for the selected target.")
        else:
            st.warning("Please select a target variable.")


# Main Area Logic
if st.session_state.df is None:
    st.info("👋 Welcome! Please upload a CSV file and select the target variable in the sidebar to get started.")
elif not st.session_state.target_variable:
    st.warning("🎯 Please select the target variable you want to predict from the sidebar.")
elif not st.session_state.task_type:
     st.error(f"Could not determine if '{st.session_state.target_variable}' implies a Classification or Regression task. Please check the data in this column.")
else:
    # Data and target are ready, display tabs
    tab_explore, tab_train, tab_report, tab_advanced_eda, tab_feature_eng, tab_model_agent, tab_model_tester = st.tabs([
        "🔎 Explore Data",
        "🚀 Train & Compare Models",
        "💡 View Report & Insights",
        "📊 Advanced EDA",
        "🧪 Feature Engineering",
        "🤖 ML Model Agent",
        "📊 Test Model"
    ])

    with tab_explore:
        st.header("Exploratory Data Analysis (EDA)")
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head())

        st.subheader("Dataset Information")
        buffer = io.StringIO()
        st.session_state.df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Descriptive Statistics (Numerical)")
        st.dataframe(st.session_state.df[st.session_state.numerical_cols].describe())

        st.subheader("Missing Values")
        missing_data = st.session_state.df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Count']
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
        if not missing_data.empty:
            st.dataframe(missing_data)
        else:
            st.success("✅ No missing values found!")

        
        st.subheader("Basic Distributions")
       
        num_to_plot = [c for c in st.session_state.numerical_features] 
        if num_to_plot:
            st.write("#### Numerical Feature Histograms")
            num_plots = len(num_to_plot)
            cols_per_row = 3
            num_rows = (num_plots + cols_per_row - 1) // cols_per_row
            plot_idx = 0
            for r in range(num_rows):
                 cols = st.columns(cols_per_row)
                 for c in range(cols_per_row):
                     if plot_idx < num_plots:
                         col_name = num_to_plot[plot_idx]
                         with cols[c]:
                             try:
                                 fig, ax = plt.subplots(figsize=(5,3))
                                 sns.histplot(st.session_state.df[col_name], kde=True, ax=ax, bins=20)
                                 ax.set_title(f'{col_name}')
                                 ax.tick_params(axis='x', rotation=45)
                                 plt.tight_layout()
                                 st.pyplot(fig)
                                 plt.close(fig)
                             except Exception as e:
                                 st.warning(f"Plot error for {col_name}: {e}")
                         plot_idx += 1


        
        cat_to_plot = [c for c in st.session_state.categorical_features] # Features only
        if cat_to_plot:
             st.write("#### Categorical Feature Counts")
             max_cats_to_show = 15
             num_plots = len(cat_to_plot)
             cols_per_row = 2
             num_rows = (num_plots + cols_per_row - 1) // cols_per_row
             plot_idx = 0
             for r in range(num_rows):
                 cols = st.columns(cols_per_row)
                 for c in range(cols_per_row):
                     if plot_idx < num_plots:
                         col_name = cat_to_plot[plot_idx]
                         with cols[c]:
                             try:
                                 n_unique = st.session_state.df[col_name].nunique()
                                 fig, ax = plt.subplots(figsize=(6, 4))
                                 if n_unique > max_cats_to_show:
                                     top_cats = st.session_state.df[col_name].value_counts().nlargest(max_cats_to_show).index
                                     # Assign y variable to hue and set legend=False
                                     sns.countplot(y=col_name, data=st.session_state.df[st.session_state.df[col_name].isin(top_cats)], order=top_cats, ax=ax, palette='viridis', hue=col_name, legend=False)
                                     ax.set_title(f'{col_name} (Top {max_cats_to_show})')
                                 else:
                                     # Assign y variable to hue and set legend=False
                                     sns.countplot(y=col_name, data=st.session_state.df, order=st.session_state.df[col_name].value_counts().index, ax=ax, palette='viridis', hue=col_name, legend=False)
                                     ax.set_title(f'{col_name}')
                                 plt.tight_layout()
                                 st.pyplot(fig)
                                 plt.close(fig)
                             except Exception as e:
                                 st.warning(f"Plot error for {col_name}: {e}")
                         plot_idx += 1


    with tab_train:
        st.header("🚀 Train, Tune, and Compare Models")
        st.markdown(f"""
        Click the button below to automatically train and evaluate several standard **{st.session_state.task_type}** models on your data for predicting **'{st.session_state.target_variable}'**.
        This includes:
        1.  Preprocessing data (imputation, scaling, encoding).
        2.  Training models like Logistic/Linear Regression, Random Forest, Gradient Boosting.
        3.  Basic hyperparameter tuning (using GridSearchCV).
        4.  Comparing models based on performance metrics on a hidden test set.
        """)

        if st.button("✨ Train Models Now", key="train_button", type="primary"):
            if not st.session_state.api_key_provided:
                st.error("Google API Key needed in the sidebar to proceed.")
            else:
                with st.spinner("⚙️ Running automated ML pipeline... This may take a few minutes."):
                    try:
                        results_data = train_and_evaluate_models(
                            st.session_state.df,
                            st.session_state.target_variable,
                            st.session_state.task_type,
                            st.session_state.numerical_features,
                            st.session_state.categorical_features
                        )
                        if results_data:
                             st.session_state.results = results_data['all_model_results']
                             st.session_state.best_model = results_data['best_model']
                             st.session_state.best_model_name = results_data['best_model_name']
                             st.session_state.X_test = results_data['X_test']
                             st.session_state.y_test = results_data['y_test']
                             st.session_state.results_meta = results_data # Store task type etc.

                             # Generate plots for the best model
                             if st.session_state.best_model_name:
                                st.session_state.plots = generate_evaluation_plots(
                                    st.session_state.results[st.session_state.best_model_name],
                                    st.session_state.best_model_name,
                                    st.session_state.X_test,
                                    st.session_state.y_test,
                                    st.session_state.task_type
                                )
                             st.success("✅ Automated ML pipeline finished!")
                             st.balloons()
                        else:
                            st.error("Automated ML pipeline failed to produce results.")
                            # Clear potentially inconsistent state
                            st.session_state.results = None
                            st.session_state.best_model = None


                    except Exception as e:
                         st.error(f"An unexpected error occurred during training: {e}")
                         st.session_state.results = None # Clear results on error

        # Display Results Table if available
        if st.session_state.get("results"):
            st.subheader("📊 Model Comparison Results")
            results_list = []
            for name, res in st.session_state.results.items():
                 if 'error' in res:
                     row = {'Model': name, 'Status': 'Error', 'Details': res['error']}
                 else:
                     row = {'Model': name, 'Status': 'Success'}
                     if st.session_state.task_type == "Classification":
                          row['Accuracy (Test)'] = res.get('accuracy_test')
                          row['F1 Score (Test)'] = res.get('f1_test')
                          row['Precision (Test)'] = res.get('precision_test')
                          row['Recall (Test)'] = res.get('recall_test')
                     else:
                          row['R² Score (Test)'] = res.get('r2_test')
                          row['RMSE (Test)'] = res.get('rmse_test')
                          row['MAE (Test)'] = res.get('mae_test')
                     row['Best CV Score'] = res.get('best_cv_score') # Display CV score from GridSearch
                 results_list.append(row)

            results_df = pd.DataFrame(results_list).set_index('Model')
            # Format numeric columns
            for col in results_df.select_dtypes(include=np.number).columns:
                 results_df[col] = results_df[col].map('{:.4f}'.format)

            st.dataframe(results_df)

            # Highlight best model
            if st.session_state.best_model_name:
                st.info(f"🏆 **Best Model:** {st.session_state.best_model_name} (based on test set {st.session_state.results_meta['primary_metric']})")

                # Display plots for the best model
                st.subheader("Best Model Performance Visualized")
                if st.session_state.get("plots"):
                    if st.session_state.task_type == "Classification" and 'confusion_matrix' in st.session_state.plots:
                        st.image(st.session_state.plots['confusion_matrix'], caption=f"Confusion Matrix ({st.session_state.best_model_name})")
                    elif st.session_state.task_type == "Regression" and 'prediction_plot' in st.session_state.plots:
                        st.image(st.session_state.plots['prediction_plot'], caption=f"Actual vs. Predicted ({st.session_state.best_model_name})")
                    else:
                         st.info("No performance plots generated for the best model.")
                else:
                     st.info("No performance plots generated for the best model.")


    with tab_report:
        st.header("💡 Business Insights from Data")
        if st.session_state.df is None or not st.session_state.target_variable:
            st.info("Load data and select a target variable in the sidebar first to generate business insights.")
        else:
            # Button to generate LLM report
            st.subheader("🤖 AI Generated Business Insights")
            if st.button("📝 Generate AI Insights Report", key="generate_report_button"):
                if not st.session_state.api_key_provided:
                    st.error("Google API Key needed in the sidebar to generate the AI report.")
                else:
                    with st.spinner("🧠 AI is analyzing data and writing the report..."):
                        try:
                            report_text = generate_llm_report(
                                st.session_state.google_api_key,
                                st.session_state.df,
                                st.session_state.target_variable,
                                st.session_state.numerical_features,
                                st.session_state.categorical_features
                            )
                            st.session_state.llm_report = report_text
                        except Exception as e:
                            st.error(f"Error generating AI report: {e}")
                            st.session_state.llm_report = f"Failed to generate report. Error: {e}"

            # Display LLM report if generated
            if st.session_state.get("llm_report"):
                st.markdown(st.session_state.llm_report)
                # Download button for the report
                st.download_button(
                    label="📥 Download AI Insights Report (.txt)",
                    data=st.session_state.llm_report,
                    file_name="datapilot_ai_business_insights.txt",
                    mime="text/plain"
                )
            else:
                 st.info("Click the button above to generate AI-powered business insights based on your data.")


            # Download options for the best model
            st.subheader("💾 Download Best Trained Model (Optional)")
            if st.session_state.get('best_model'):
                model_buffer = io.BytesIO()
                pickle.dump(st.session_state.best_model, model_buffer)
                model_buffer.seek(0)
                st.download_button(
                    label=f"📥 Download Model: {st.session_state.best_model_name}.pkl",
                    data=model_buffer,
                    file_name=f"{st.session_state.best_model_name.lower()}_best_model.pkl",
                    mime="application/octet-stream",
                    help="Downloads the trained scikit-learn pipeline object (including preprocessing)."
                )
            else:
                st.warning("No best model available to download.")


    with tab_advanced_eda:
        st.header("📊 Advanced Exploratory Data Analysis")
        if st.session_state.df is not None:
            # Call the function from the imported eda module
            try:
                eda.render_eda_tab(st)
            except AttributeError:
                 st.error("The `eda.py` file does not have a `render_eda_tab(st)` function. Please ensure it exists.")
            except ImportError:
                 st.error("Could not import the `eda` module. Make sure `eda.py` is in the same directory.")
            except Exception as e:
                 st.error(f"An error occurred while rendering the Advanced EDA tab: {e}")
        else:
            st.info("Upload data first to see advanced EDA.")


    # Add a new tab for Feature Engineering
    with tab_feature_eng:
        st.header("🧪 Automated Feature Engineering")
        st.markdown("""
        This feature uses an AI agent to analyze your dataset and automatically engineer features to improve model performance.
        Provide custom instructions or let the AI decide what feature engineering steps are needed.
        """)
        
        # Check if API key is provided
        if not st.session_state.api_key_provided:
            st.warning("Please provide your Google API Key in the sidebar to use the feature engineering functionality.", icon="🔑")
        elif st.session_state.df is None:
            st.info("Upload a CSV file to enable feature engineering.")
        else:
            # Create collapsible section for custom instructions
            with st.expander("Feature Engineering Instructions", expanded=True):
                # Initialize the session state for feature engineering if needed
                if 'feature_instructions' not in st.session_state:
                    st.session_state.feature_instructions = ""
                if 'engineered_df' not in st.session_state:
                    st.session_state.engineered_df = None
                if 'feature_code' not in st.session_state:
                    st.session_state.feature_code = None
                if 'feature_steps' not in st.session_state:
                    st.session_state.feature_steps = None
                
                # Text area for custom instructions
                feature_instructions = st.text_area(
                    "Custom Instructions (Optional)",
                    value=st.session_state.feature_instructions,
                    help="Provide specific instructions for feature engineering. Leave empty for automated analysis.",
                    placeholder="Examples:\n- Convert 'date_column' to day of week, month, and year features\n- Create interaction between 'price' and 'quantity'\n- Encode categorical variables with target encoding instead of one-hot encoding",
                    height=150
                )
                
                # Run Feature Engineering button
                if st.button("🚀 Run Feature Engineering", type="primary"):
                    if not st.session_state.api_key_provided:
                        st.error("Google API Key needed in the sidebar to proceed.")
                    else:
                        with st.spinner("⚙️ Running automated feature engineering... This may take a few minutes."):
                            try:
                                # Get the configured LLM
                                llm = get_llm(st.session_state.google_api_key)
                                if llm is None:
                                    st.error("Could not initialize LLM. Check your API key.")
                                    st.session_state.engineered_df = None
                                    st.session_state.feature_code = None
                                    st.session_state.feature_steps = None
                                else:
                                    # Create a chain that will:
                                    # 1. Analyze the data
                                    # 2. Recommend feature engineering steps
                                    # 3. Generate Python code to implement those steps
                                    # 4. Execute the code on the data
                                    
                                    # Use a prompt template
                                    system_prompt = """You are an expert in feature engineering for machine learning. 
                                    Analyze the provided dataset and create engineered features that would be useful for predictive modeling.
                                    
                                    Guidelines:
                                    - Convert features to appropriate data types
                                    - Remove features with unique values for each row (like IDs)
                                    - Remove constant features (same value in all rows)
                                    - Encode high-cardinality categoricals (using threshold <= 5% of dataset size)
                                    - One-hot encode remaining categorical variables (if not too many)
                                    - Create datetime-based features if datetime columns are present
                                    - Convert booleans to integer (1/0)
                                    - Handle the target variable appropriately (if provided)
                                    - Follow any user-provided instructions that override these defaults
                                    
                                    IMPORTANT: Return your response in a valid JSON format with these fields:
                                    {
                                      "recommended_steps": ["list of steps as strings"],
                                      "feature_engineering_code": "complete Python function that takes a dataframe and returns the engineered dataframe"
                                    }
                                    """
                                    
                                    user_prompt = f"""
                                    Analyze this dataset to determine appropriate feature engineering steps.
                                    
                                    Dataset columns and sample:
                                    {st.session_state.df.head(5).to_string()}
                                    
                                    Data types:
                                    {st.session_state.df.dtypes.to_string()}
                                    
                                    Basic statistics:
                                    {st.session_state.df.describe().to_string()}
                                    
                                    User instructions: {feature_instructions if feature_instructions else "No specific instructions - use your best judgment based on the data."}
                                    
                                    Target variable (if applicable): {st.session_state.target_variable if st.session_state.target_variable else "None provided"}
                                    
                                    Provide your recommendations and a complete Python function named 'engineer_features' that takes a dataframe parameter and returns the engineered dataframe.
                                    The function should implement all recommended steps, with appropriate handling and error checking.
                                    """
                                    
                                    # Fix import by using the same imports as at the top of the file
                                    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
                                    from langchain.schema import SystemMessage
                                    
                                    chat_prompt = ChatPromptTemplate.from_messages([
                                        SystemMessage(content=system_prompt),
                                        HumanMessagePromptTemplate.from_template("{user_prompt}")
                                    ])
                                    
                                    chain = chat_prompt | llm
                                    
                                    # Execute the chain
                                    response = chain.invoke({"user_prompt": user_prompt})
                                    
                                    try:
                                        # Try to parse the response as JSON
                                        import json
                                        import re
                                        
                                        # Extract JSON if it's embedded in markdown code blocks or text
                                        response_text = response.content
                                        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                                        if json_match:
                                            json_str = json_match.group(1)
                                        else:
                                            json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                                            if json_match:
                                                json_str = json_match.group(1)
                                            else:
                                                # Try to find JSON-like structure with braces
                                                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                                                if json_match:
                                                    json_str = json_match.group(1)
                                                else:
                                                    json_str = response_text
                                        
                                        result = json.loads(json_str)
                                        
                                        # Extract code and steps
                                        feature_code = result.get("feature_engineering_code", "")
                                        recommended_steps = result.get("recommended_steps", [])
                                        
                                        # If the code is wrapped in markdown code blocks, extract it
                                        code_match = re.search(r'```python\s*(.*?)\s*```', feature_code, re.DOTALL)
                                        if code_match:
                                            feature_code = code_match.group(1)
                                        
                                        # Initialize a flag to track if code extraction was successful
                                        code_extracted_successfully = False
                                        
                                        # If the code is wrapped in markdown code blocks, extract it
                                        if not feature_code.strip():
                                            code_match = re.search(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
                                            if code_match:
                                                feature_code = code_match.group(1)
                                        else:
                                            code_match = re.search(r'```python\s*(.*?)\s*```', feature_code, re.DOTALL)
                                            if code_match:
                                                feature_code = code_match.group(1)
                                        
                                        # Check if code was extracted and set the flag
                                        if feature_code.strip():
                                            code_extracted_successfully = True
                                        else:
                                            st.error("No feature engineering code could be extracted from the response")
                                            st.session_state.feature_code = response_text  # Save the raw response for debugging
                                            st.session_state.feature_steps = None
                                            st.session_state.engineered_df = None
                                        
                                        # Only execute the feature engineering code if it was extracted successfully
                                        if code_extracted_successfully:
                                            try:
                                                st.code(feature_code, language='python')
                                                import types
                                                
                                                # Create a module-like object to hold the function
                                                module = types.ModuleType('feature_engineering')
                                                
                                                # Execute the code string in the module's namespace
                                                exec(feature_code, module.__dict__)
                                                
                                                # Get the function from the module
                                                engineer_features = getattr(module, 'engineer_features')
                                                
                                                # Apply function to the data
                                                engineered_df = engineer_features(st.session_state.df.copy())
                                                
                                                # Save results to session state
                                                st.session_state.engineered_df = engineered_df
                                                st.session_state.feature_code = feature_code
                                                st.session_state.feature_steps = recommended_steps
                                                st.session_state.feature_instructions = feature_instructions  # Save the instructions
                                                
                                                st.success("✅ Feature engineering completed successfully!")
                                            
                                            except Exception as e:
                                                st.error(f"Error executing generated feature engineering code: {e}")
                                                # Keep code for debugging, but clear results
                                                st.session_state.engineered_df = None
                                                st.session_state.feature_code = response_text  # Save the raw response for debugging
                                                st.session_state.feature_steps = None
                                        
                                    except Exception as e:
                                        st.error(f"Error processing LLM response or parsing JSON: {e}")
                                        st.session_state.feature_code = response.content if 'response' in locals() and hasattr(response, 'content') else 'Error during LLM call'
                                        st.session_state.feature_steps = None
                                        st.session_state.engineered_df = None
                                        
                            except Exception as e:
                                st.error(f"An unexpected error occurred during feature engineering: {e}")
                                st.session_state.engineered_df = None
                                st.session_state.feature_code = None
                                st.session_state.feature_steps = None
            
            # Display results if available
            if st.session_state.get("engineered_df") is not None:
                # Show tabs for different views of the results
                result_tab1, result_tab2, result_tab3 = st.tabs([
                    "📊 Engineered Data",
                    "📝 Feature Engineering Steps",
                    "💻 Generated Code"
                ])
                
                with result_tab1:
                    st.subheader("Engineered Dataset")
                    st.dataframe(st.session_state.engineered_df.head(10))
                    
                    # Compare shape before and after
                    original_shape = st.session_state.df.shape
                    new_shape = st.session_state.engineered_df.shape
                    
                    st.info(f"""
                    **Data Transformation Summary:**
                    - Original data: {original_shape[0]} rows × {original_shape[1]} columns
                    - Engineered data: {new_shape[0]} rows × {new_shape[1]} columns
                    - Columns added/removed: {new_shape[1] - original_shape[1]}
                    """)
                    
                    # Option to download the engineered dataset
                    csv = st.session_state.engineered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Engineered Data (.csv)",
                        data=csv,
                        file_name="engineered_data.csv",
                        mime="text/csv"
                    )
                
                with result_tab2:
                    st.subheader("Recommended Feature Engineering Steps")
                    if st.session_state.feature_steps:
                        for i, step in enumerate(st.session_state.feature_steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.info("No specific steps were recommended.")
                
                with result_tab3:
                    st.subheader("Generated Python Code")
                    if st.session_state.feature_code:
                        st.code(st.session_state.feature_code, language="python")
                        
                        # Option to download the code
                        st.download_button(
                            label="📥 Download Code (.py)",
                            data=st.session_state.feature_code,
                            file_name="engineer_features.py",
                            mime="text/plain"
                        )
                    else:
                        st.info("No code was generated.")
                
                # Provide option to use engineered data for modeling
                st.subheader("Use Engineered Data")
                if st.button("🔄 Use for Modeling", help="Replace the current dataset with the engineered version for training models"):
                    st.session_state.df = st.session_state.engineered_df
                    # Reset dependent states when data changes
                    st.session_state.results = None
                    st.session_state.llm_report = None
                    st.session_state.plots = None
                    st.session_state.agent_initialized = False  # Reset chat agent with new data
                    # Keep target variable if it exists in the new dataframe
                    if st.session_state.target_variable not in st.session_state.engineered_df.columns:
                        st.session_state.target_variable = None
                        st.session_state.task_type = None
                    
                    st.success("✅ Dataset updated to use engineered features! Navigate to other tabs to continue your analysis.")
                    st.experimental_rerun()  # Rerun the app to reflect changes

  
    with tab_model_agent:
        ml_agent.render_ml_agent_tab(st) # Added call to module function

    
    with tab_model_tester: # Added new tab block
        model_tester.render_model_tester_tab(st) # Call function from model_tester.py

# --- Footer ---
st.sidebar.divider()
st.sidebar.info("DataPilot: AIgnite Project")