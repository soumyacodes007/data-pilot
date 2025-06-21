  DataPilot offers a no-code platform that automates the end-to-end machine learning process from data ingestion to insight generation. Its unique capability is in its automation of data cleaning, model training, evaluation, and hyperparameter tuning powered by ai agents  allowing individuals without ML expertise to access advanced analytics. By doing so, it accelerates decision-making and innovation across different domains. 
 # DataPilot: No-Code ML Assistant

DataPilot is a Streamlit web application designed to empower users with limited coding experience to perform end-to-end machine learning tasks. It allows users to upload their datasets (CSV format), explore the data, automatically train and compare various ML models, generate business-focused insights using AI, perform automated feature engineering, and test trained models.

## ✨ Features

*   **📤 Data Upload:** Easily upload your dataset in CSV format.
*   **🔎 Basic Data Exploration:**
    *   View dataset preview, information (column types, non-null counts), and descriptive statistics.
    *   Identify missing values.
    *   Visualize basic distributions with histograms for numerical features and count plots for categorical features.
*   **🚀 Automated Model Training & Comparison:**
    *   Automatically detects the task type (Classification or Regression) based on the selected target variable.
    *   Preprocesses data (imputation, scaling, one-hot encoding).
    *   Trains multiple standard models (e.g., Logistic/Linear Regression, Random Forest).
    *   Performs basic hyperparameter tuning using GridSearchCV.
    *   Compares models based on relevant metrics (Accuracy, F1, Precision, Recall for classification; R², RMSE, MAE for regression) on a test set.
    *   Identifies and highlights the best-performing model.
    *   Visualizes performance of the best model (e.g., Confusion Matrix, Actual vs. Predicted plot).
*   **💡 AI-Powered Business Insights:**
    *   Leverages Google's Generative AI (Gemini) to analyze the dataset.
    *   Generates actionable business insights and recommendations based on data patterns, tailored for a non-technical audience.
*   **📊 Advanced EDA (External Module - `eda.py`):**
    *   **Sweetviz Integration:** Generate comprehensive, beautiful EDA reports viewable within the app.
    *   **dtale Integration:** Launch an interactive spreadsheet-like dashboard for deep data exploration in a separate browser tab.
    *   **Gemini Analysis:** Get AI-driven summaries and potential issues directly from data statistics.
*   **🧪 Automated Feature Engineering:**
    *   Uses an AI agent (Gemini) to analyze the dataset and suggest feature engineering steps.
    *   Optionally accepts custom user instructions.
    *   Generates and executes Python code to create new features.
    *   Allows users to use the engineered dataset for subsequent modeling.
*   **🤖 ML Model Agent:**
    *   Interact with an AI agent (Gemini) to generate custom ML model code based on natural language instructions.
    *   Train the custom model and view its results.
*   **📊 Test Model:**
    *   Upload a saved model (e.g., the `.pkl` file downloaded from the app).
    *   Provide new, unseen data (in CSV format).
    *   Generate predictions using the loaded model on the new data.
    *   Download the predictions.
*   **💾 Model Download:** Download the best-performing trained model pipeline (including preprocessing steps) as a `.pkl` file.
*   **📄 Report Download:** Download the AI-generated business insights report as a `.txt` file.

## 🛠️ Technology Stack

*   **Web Framework:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
*   **Numerical Computing:** [NumPy](https://numpy.org/)
*   **Machine Learning:** [Scikit-learn](https://scikit-learn.org/stable/)
*   **AI/LLM Integration:** [Google Generative AI SDK](https://github.com/google/generative-ai-python), [Langchain](https://www.langchain.com/) (for Agents)
*   **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Automated EDA:** [Sweetviz](https://github.com/fbdesignpro/sweetviz), [dtale](https://github.com/man-group/dtale)
*   **Environment Variables:** [python-dotenv](https://github.com/theskumar/python-dotenv)

## ⚙️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Copy the contents of `.env.example` into `.env`.
    *   Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual Google API Key obtained from [Google AI Studio](https://aistudio.google.com/).
    *   `.env` file content:
        ```
        GOOGLE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

## ▶️ Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
2.  **Open in Browser:** The application should automatically open in your web browser. If not, navigate to the local URL provided in the terminal (usually `http://localhost:8501`).
3.  **Upload Data:** Use the sidebar uploader to select and upload your CSV file.
4.  **Select Target:** Choose the target variable you want to predict or analyze from the dropdown in the sidebar.
5.  **Explore Tabs:** Navigate through the different tabs to explore data, train models, generate insights, perform feature engineering, etc.

## 📁 Project Structure

```
.
├── .env           # Local environment variables (contains API key - DO NOT COMMIT)
├── .env.example   # Example environment variable file
├── app.py         # Main Streamlit application script
├── eda.py         # Module for Advanced EDA functionalities (Sweetviz, dtale, Gemini summary)
├── ml_agent.py    # Module for the ML Model Agent tab
├── model_tester.py# Module for the Model Tester tab
├── requirements.txt # Python dependencies
└── README.md      # This file
```

## 📄 License

(Optional: Add your license information here, e.g., MIT License)

## 🙏 Contributing

(Optional: Add contribution guidelines if applicable) 
  
  **What does DataPilot actually do?**
- **Upload your data** - Just drag and drop your dataset (CSV or Excel). We’ll handle the rest.
    
- **Clean your data** -  Get one of the best eda compare , plot graphs ask question make insights 
    
- **Train multiple models** - X AI : no black box it will autometically explain everything in huaman understandable langauge
    DataPilot tries out different ML models and picks the best one for your data.
    
- **Tune things up** - We optimize your model behind the scenes, so you don’t need to worry about the technical stuff.
    
- **Show you the results** - Visuals, accuracy scores, and performance breakdowns all easy to understand.
    
- **Generate a smart report** - Get a downloadable summary of what we found and what it means for you which is understandable for normal people

**Target Audience**
- Data Analysts - to reduce the work force and deliver ML insights without code.
- Beginner ML user - Build and use models without coding or deep ML knowledge.
- Financial Analyst - Forecast trends and identify anomalies in transactions.
- Business Analyst - Transform raw data into predictive insights for smarter decision-making.
- Product Manager - Quickly validate product hypotheses and optimize features using data-driven insights.

**MVP Features**
- Data Cleaning Automation – Automatically prepares your data by handling missing values, outliers, encoding, and scaling.
- Model Selection & Comparison – Trains and compares multiple ML models to recommend the best fit.
- Hyperparameter Tuning – Boosts model accuracy using automated tuning techniques like Grid Search or Bayesian Optimization.
- Performance Evaluation – Visualizes and summarizes key metrics like accuracy and F1-score.
- Insightful Reporting – Generates easy-to-understand reports with actionable insights.
- Exploratory Data Analysis (EDA) – Helps visualize and understand your dataset through intuitive charts and comparisons.
## Here is the detailed video explaining the functionalities: 
https://youtu.be/Ycud4n3gxAY?si=jWDw6ZuufXkOfj01
