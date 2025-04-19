import streamlit as st
import pandas as pd
import numpy as np

def get_pipeline_features(pipeline):
    """Extracts numerical and categorical features from a scikit-learn pipeline's preprocessor."""
    numerical_features = []
    categorical_features = []
    try:
        preprocessor = pipeline.named_steps.get('preprocessor')
        if preprocessor and hasattr(preprocessor, 'transformers_'):
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    numerical_features.extend(features)
                elif name == 'cat':
                    categorical_features.extend(features)
                # Handle cases where transformer is 'passthrough' or 'drop' if needed
        else:
            st.warning("Could not find or parse the preprocessor in the pipeline.")
            # Fallback to session state features if preprocessor parsing fails
            numerical_features = st.session_state.get('numerical_features', [])
            categorical_features = st.session_state.get('categorical_features', [])
            
    except Exception as e:
        st.error(f"Error extracting features from pipeline: {e}")
        # Fallback
        numerical_features = st.session_state.get('numerical_features', [])
        categorical_features = st.session_state.get('categorical_features', [])
        
    return numerical_features, categorical_features

def render_model_tester_tab(st):
    """Renders the content for the Model Tester tab."""
    st.header("🧪 Test Trained Model")
    st.markdown("Input custom values for the features below to get a prediction from the currently active model.")

    # Determine which model is active (custom ML agent or best automated)
    active_model = None
    model_name = "None"
    if st.session_state.get('custom_model') is not None:
        active_model = st.session_state.custom_model
        model_name = st.session_state.get('custom_model_name', "Custom Model")
        st.info(f"Using model: **{model_name}** (from ML Agent Tab)")
    elif st.session_state.get('best_model') is not None:
        active_model = st.session_state.best_model
        model_name = st.session_state.get('best_model_name', "Best Automated Model")
        st.info(f"Using model: **{model_name}** (from Train & Compare Tab)")
    else:
        st.warning("No trained model found. Please train a model using the '🚀 Train & Compare Models' or '🤖 ML Model Agent' tab first.")
        return

    # Extract features from the pipeline
    numerical_features, categorical_features = get_pipeline_features(active_model)
    all_features = numerical_features + categorical_features

    if not all_features:
        st.error("Could not determine the input features for the selected model.")
        return

    st.subheader("Input Features")
    input_data = {}

    # Create input fields dynamically
    cols = st.columns(2)
    col_idx = 0
    for feature in all_features:
        with cols[col_idx % 2]:
            if feature in numerical_features:
                # Attempt to get min/max/median from original data for better defaults/range
                default_val = float(st.session_state.df[feature].median()) if feature in st.session_state.df else 0.0
                min_val = float(st.session_state.df[feature].min()) if feature in st.session_state.df else None
                max_val = float(st.session_state.df[feature].max()) if feature in st.session_state.df else None
                input_data[feature] = st.number_input(
                    f"Enter {feature}", 
                    value=default_val, 
                    min_value=min_val, 
                    max_value=max_val, 
                    step=1.0 if st.session_state.df[feature].dtype in [np.int64, np.int32] else 0.1, # Heuristic step
                    key=f"input_{feature}"
                )
            elif feature in categorical_features:
                unique_values = list(st.session_state.df[feature].unique()) if feature in st.session_state.df else ["Value 1", "Value 2"]
                # Ensure unique_values are strings for selectbox
                unique_values = [str(val) for val in unique_values if pd.notna(val)]
                default_cat_val = unique_values[0] if unique_values else ""
                input_data[feature] = st.selectbox(
                    f"Select {feature}", 
                    options=unique_values, 
                    index=0, 
                    key=f"input_{feature}"
                )
            else:
                 st.warning(f"Feature '{feature}' type unknown, using text input.")
                 input_data[feature] = st.text_input(f"Enter {feature}", key=f"input_{feature}")
        col_idx += 1

    # Prediction Button
    if st.button("📊 Get Prediction", key="get_prediction_button"):
        try:
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure column order matches the order used during training (important for some models)
            # Extract the feature order from the first step (preprocessor) if possible
            try:
                ordered_features = numerical_features + categorical_features # Assuming this order was used
                input_df = input_df[ordered_features] # Reorder input df columns
            except Exception as reorder_err:
                 st.warning(f"Could not reliably reorder features, using original order: {reorder_err}")
                 # Continue with the current input_df order

            # Make prediction using the pipeline (handles preprocessing)
            prediction = active_model.predict(input_df)
            prediction_proba = None
            if st.session_state.task_type == "Classification" and hasattr(active_model, "predict_proba"):
                 prediction_proba = active_model.predict_proba(input_df)

            st.subheader("Prediction Result")
            st.success(f"Predicted {st.session_state.target_variable}: **{prediction[0]}**")

            if prediction_proba is not None:
                st.write("Prediction Probabilities:")
                # Try to get class labels if possible (e.g., from LabelEncoder if used and stored)
                # For now, just show probabilities per class index
                try:
                     classes = active_model.classes_ if hasattr(active_model, 'classes_') else [f"Class {i}" for i in range(prediction_proba.shape[1])]
                     proba_df = pd.DataFrame(prediction_proba, columns=classes)
                     st.dataframe(proba_df)
                except Exception as e:
                    st.warning(f"Could not display class labels for probabilities: {e}")
                    st.write(prediction_proba)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            import traceback
            st.text(traceback.format_exc()) 