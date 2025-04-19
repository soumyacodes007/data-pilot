import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import types
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import traceback # Import traceback
import ml_agent

# Assuming get_llm is defined in app.py or another shared module
# If not, it needs to be defined or imported here as well.
# from app import get_llm # Example import

# Placeholder for get_llm if not imported - replace with actual import
def get_llm(api_key):
    # Simplified placeholder - replace with your actual get_llm implementation
    # This requires langchain and google-genai to be installed and configured
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key, convert_system_message_to_human=True)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def render_ml_agent_tab(st):
    """Renders the content for the ML Model Agent tab."""
    st.header("🤖 ML Model Agent")
    st.markdown("""
    Use natural language to train custom ML models on your dataset.
    Ask for specific models, tuning approaches, or custom architectures.
    The AI agent will generate and execute the code to train your models.
    """)

    # Check dependencies (API key, data, target variable)
    if not st.session_state.api_key_provided:
        st.warning("Please provide your Google API Key in the sidebar to use the ML Model Agent.", icon="🔑")
        return # Stop rendering if no API key
    if st.session_state.df is None:
        st.info("Upload a CSV file to enable ML model training.")
        return # Stop rendering if no data
    if not st.session_state.target_variable:
        st.warning("Please select a target variable in the sidebar to train models.")
        return # Stop rendering if no target

    # Initialize session state keys if they don't exist
    if 'model_instructions' not in st.session_state:
        st.session_state.model_instructions = ""
    if 'custom_model' not in st.session_state:
        st.session_state.custom_model = None
    if 'custom_model_code' not in st.session_state:
        st.session_state.custom_model_code = None
    if 'custom_model_results' not in st.session_state:
        st.session_state.custom_model_results = None
    if 'custom_model_name' not in st.session_state:
        st.session_state.custom_model_name = None

    # --- Model Training Instructions Expander ---
    with st.expander("Model Training Instructions", expanded=True):
        model_instructions = st.text_area(
            "Instructions for Model Training",
            value=st.session_state.model_instructions,
            help="Describe the model you want to train, specific parameters, and techniques.",
            placeholder="Examples:\n- Train an XGBoost model with early stopping\n- Create a stacked ensemble of random forest and logistic regression\n- Train a neural network with 2 hidden layers for my classification task\n- Use RandomizedSearchCV to tune hyperparameters",
            height=150
        )

        # --- Run Model Training Button ---
        if st.button("🚀 Generate and Train Model", type="primary", key="ml_agent_train_button"):
            with st.spinner("⚙️ Generating and training custom model... This may take several minutes."):
                # Outer try-except for the whole process
                try:
                    st.session_state.model_instructions = model_instructions
                    llm = get_llm(st.session_state.google_api_key)

                    if not llm:
                        st.error("LLM not available. Cannot generate model.")
                        # Clear relevant state if LLM fails
                        st.session_state.custom_model = None
                        st.session_state.custom_model_results = None
                        st.session_state.custom_model_code = None
                        st.session_state.custom_model_name = None
                        return # Exit if no LLM

                    # --- System and User Prompts (Keep as is) ---
                    system_prompt = """You are an expert in machine learning model development.
                    Your task is to create a custom ML model based on the user's requirements and dataset.

                    Guidelines:
                    - Follow the user's instructions precisely.
                    - If no specific model is requested, choose an appropriate one for the task (e.g., XGBoost, RandomForest).
                    - Include necessary preprocessing within the function: handle missing values (impute numerical with mean/median, categorical with mode), scale numerical features (StandardScaler), and encode categorical features (OneHotEncoder). Use ColumnTransformer.
                    - Split data into train/test sets (use test_size=0.2, random_state=42).
                    - Implement cross-validation (e.g., 5-fold) for robust evaluation, reporting the mean score.
                    - Evaluate the model on the test set using appropriate metrics (Accuracy, F1, Precision, Recall for classification; R2, RMSE, MAE for regression).
                    - Generate evaluation plots: Confusion Matrix (classification) or Actual vs. Predicted scatter plot (regression), and Residuals plot. Save plots to BytesIO objects.
                    - Use scikit-learn, XGBoost, or other mainstream Python ML libraries.
                    - Define a Python function named 'train_custom_model' that takes the DataFrame and target variable name as input.
                    - The function MUST return a dictionary containing at least: 'model' (the trained pipeline/model object), 'metrics' (a dictionary of evaluation metric names and values), 'plots' (a dictionary of plot names and BytesIO objects), 'test_predictions', 'y_test'. Include 'feature_importances' or 'hyperparameters' if applicable.

                    IMPORTANT: Return your response ONLY as a valid JSON object (no introductory text, no markdown formatting before or after the JSON) with these exact fields:
                    {
                      "model_name": "Descriptive name of the chosen model (e.g., 'XGBoost Classifier')",
                      "model_code": "Complete, executable Python code string for the 'train_custom_model' function, including all necessary imports WITHIN the function string.",
                      "explanation": "Brief explanation of the model choice and key steps."
                    }
                    Ensure the 'model_code' string is properly escaped for JSON.
                    """
                    user_prompt = f"""
                    Create a custom ML model function based on these instructions:
                    "{model_instructions if model_instructions else "Create the best model for my data with appropriate preprocessing and evaluation."}"

                    Task type: {st.session_state.task_type}
                    Target variable: {st.session_state.target_variable}
                    Dataset columns: {list(st.session_state.df.columns)}
                    Numerical feature columns (suggested): {st.session_state.numerical_features}
                    Categorical feature columns (suggested): {st.session_state.categorical_features}
                    Dataset sample (first 5 rows):
{st.session_state.df.head().to_string()}

                    Generate the 'train_custom_model' function code string as specified in the guidelines.
                    """

                    # --- LLM Interaction (Keep as is) ---
                    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
                    from langchain.schema import SystemMessage
                    chat_prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(content=system_prompt),
                        HumanMessagePromptTemplate.from_template("{user_prompt}")
                    ])
                    chain = chat_prompt | llm
                    response = chain.invoke({"user_prompt": user_prompt})
                    response_text = response.content

                    # --- JSON Parsing and Code Extraction ---
                    model_code = ""
                    model_name = "Custom Model"
                    explanation = ""
                    code_extracted_successfully = False

                    # Inner try-except for parsing/extraction
                    try:
                        try:
                            # Attempt direct JSON parsing first
                            result = json.loads(response_text)
                            model_code = result.get("model_code", "")
                            model_name = result.get("model_name", "Custom Model")
                            explanation = result.get("explanation", "")

                        except json.JSONDecodeError as json_err:
                            st.warning(f"JSON parsing failed: {json_err}. Attempting manual extraction...")
                            # Fallback: Regex extraction
                            # Nested try-except for manual extraction
                            try:
                                model_name_match = re.search(r'"model_name"\s*:\s*"([^"]*)"', response_text)
                                model_name = model_name_match.group(1) if model_name_match else "Custom Model"

                                code_match = re.search(r'"model_code"\s*:\s*"(.*)"(?=,\s*"explanation")', response_text, re.DOTALL)
                                if code_match:
                                    model_code = code_match.group(1)
                                    # Break down replace chain for clarity and robustness
                                    model_code = model_code.replace('\\n', '\n')
                                    model_code = model_code.replace('\\"', '"')
                                    model_code = model_code.replace('\\t', '\t')
                                    model_code = model_code.replace('\\\'', '\'') # Unescape single quotes
                                else:
                                    code_match = re.search(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
                                    if code_match:
                                        model_code = code_match.group(1)

                                explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text)
                                explanation = explanation_match.group(1) if explanation_match else ""

                            # Catch errors during manual extraction
                            except Exception as extract_err:
                                st.error(f"Manual extraction failed: {extract_err}")
                                st.session_state.custom_model_code = response_text
                                model_code = "" # Ensure model_code is empty if extraction fails

                        # --- Code Cleaning and Validation ---
                        if model_code:
                            model_code = model_code.strip()
                            if model_code.startswith("\"") and model_code.endswith("\""):
                                model_code = model_code[1:-1]
                            if model_code.startswith("\n"):
                                model_code = model_code.lstrip("\n")
                            if model_code.startswith("```python"):
                                model_code = model_code.lstrip("```python").rstrip("```").strip()
                                
                            if "def train_custom_model(" in model_code:
                                code_extracted_successfully = True
                            else:
                                st.error("Extracted code does not appear to be a valid function definition.")
                                st.session_state.custom_model_code = model_code
                        else:
                            st.error("No model code could be extracted.")
                            st.session_state.custom_model_code = response_text

                    # Catch errors specifically during parsing/extraction phase
                    except Exception as parse_extract_err:
                        st.error(f"Error during JSON parsing or code extraction: {parse_extract_err}")
                        st.session_state.custom_model_code = response_text
                        code_extracted_successfully = False # Ensure flag is false

                    # --- Code Execution ---
                    if code_extracted_successfully:
                        st.subheader("Generated Code (Executing...)")
                        st.code(model_code, language='python')
                        # Inner try-except for code execution
                        try:
                            module = types.ModuleType('model_training')
                            exec(model_code, module.__dict__)

                            if hasattr(module, 'train_custom_model'):
                                train_custom_model = getattr(module, 'train_custom_model')
                                model_results = train_custom_model(st.session_state.df.copy(), st.session_state.target_variable)

                                if isinstance(model_results, dict) and 'error' not in model_results:
                                    st.session_state.custom_model = model_results.get('model')
                                    st.session_state.custom_model_code = model_code
                                    st.session_state.custom_model_results = model_results
                                    st.session_state.custom_model_name = model_name
                                    st.success(f"✅ Successfully trained {model_name}!")
                                elif isinstance(model_results, dict) and 'error' in model_results:
                                    st.error(f"Error within generated function: {model_results['error']}")
                                    st.session_state.custom_model = None
                                    st.session_state.custom_model_results = None
                                else:
                                    st.error("Generated function returned unexpected format.")
                                    st.session_state.custom_model = None
                                    st.session_state.custom_model_results = None
                            else:
                                st.error("'train_custom_model' function not found in executed code.")
                                st.session_state.custom_model = None
                                st.session_state.custom_model_results = None

                        # Catch execution errors
                        except SyntaxError as se:
                            st.error(f"Syntax Error executing generated code: {se}")
                            st.text(f"Line: {se.lineno}, Offset: {se.offset}")
                            st.text(traceback.format_exc())
                            st.session_state.custom_model = None
                            st.session_state.custom_model_results = None
                        except Exception as exec_err:
                            st.error(f"Error executing generated code: {exec_err}")
                            st.text(traceback.format_exc())
                            st.session_state.custom_model = None
                            st.session_state.custom_model_results = None
                    else:
                        # Code extraction failed, clear state
                        st.session_state.custom_model = None
                        st.session_state.custom_model_results = None
                        # custom_model_code might contain the error source, so don't clear it necessarily

                # Catch any outer errors (e.g., during LLM call itself)
                except Exception as outer_err:
                    st.error(f"An unexpected error occurred: {outer_err}")
                    st.text(traceback.format_exc())
                    st.session_state.custom_model = None
                    st.session_state.custom_model_results = None
                    st.session_state.custom_model_code = None
                    st.session_state.custom_model_name = None

    # --- Display Results Section (Keep as is) ---
    if st.session_state.get("custom_model_results") is not None:
        st.subheader("Custom Model Results")
        results = st.session_state.custom_model_results
        model_name = st.session_state.custom_model_name

        tab_perf, tab_code, tab_details = st.tabs([
            "📊 Model Performance",
            "💻 Generated Code",
            "🔍 Detailed Results"
        ])

        with tab_perf:
            st.subheader(f"Model: {model_name}")
            metrics = results.get('metrics', {})
            if metrics:
                 st.write("Test Set Metrics:")
                 cols = st.columns(len(metrics))
                 i = 0
                 for name, value in metrics.items():
                     with cols[i]:
                         st.metric(name.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, (float, int)) else value)
                     i += 1
            else:
                 st.info("No metrics dictionary found.")
            plots = results.get('plots', {})
            if plots:
                 st.write("Evaluation Plots:")
                 for plot_name, plot_data in plots.items():
                     if isinstance(plot_data, BytesIO):
                         st.image(plot_data, caption=plot_name.replace('_', ' ').title())
            else:
                st.info("No plots found.")
            if st.session_state.custom_model is not None:
                model_buffer = io.BytesIO()
                try:
                    pickle.dump(st.session_state.custom_model, model_buffer)
                    model_buffer.seek(0)
                    st.download_button(
                        label=f"📥 Download Model ({model_name}.pkl)",
                        data=model_buffer,
                        file_name=f"custom_{model_name.replace(' ', '_').lower()}.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as pickle_err:
                    st.warning(f"Could not pickle model: {pickle_err}")

        with tab_code:
            st.subheader("Generated Python Code")
            if st.session_state.custom_model_code:
                st.code(st.session_state.custom_model_code, language="python")
                st.download_button(
                    label="📥 Download Code (.py)",
                    data=st.session_state.custom_model_code,
                    file_name="train_custom_model.py",
                    mime="text/plain"
                )
            else:
                st.info("No model code available.")

        with tab_details:
            st.subheader("Detailed Results")
            explanation = results.get('explanation', st.session_state.get('custom_model_explanation', ''))
            if explanation:
                 st.write("**Model Explanation:**")
                 st.markdown(f"> {explanation}")
            if 'feature_importances' in results:
                st.write("**Feature Importances:**")
                feature_imp = results['feature_importances']
                try:
                    if isinstance(feature_imp, (pd.DataFrame, pd.Series)):
                        st.dataframe(feature_imp)
                    elif isinstance(feature_imp, dict):
                        imp_df = pd.DataFrame(list(feature_imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
                        st.dataframe(imp_df)
                    else:
                         st.write("Importances in unrecognized format.")
                         st.write(feature_imp)
                except Exception as fe_err:
                    st.warning(f"Could not display feature importances: {fe_err}")
            if 'hyperparameters' in results:
                st.write("**Model Hyperparameters:**")
                st.json(results['hyperparameters'])
            if 'test_predictions' in results and 'y_test' in results:
                 st.write("**Sample Predictions (Test Set):**")
                 try:
                     preds_df = pd.DataFrame({
                         'Actual': results['y_test'],
                         'Predicted': results['test_predictions']
                     }).head(10)
                     st.dataframe(preds_df)
                 except Exception as pred_err:
                     st.warning(f"Could not display sample predictions: {pred_err}")

    elif st.session_state.get('custom_model_code'):
         st.subheader("Generated Code (Execution Failed)")
         st.code(st.session_state.custom_model_code, language='python')