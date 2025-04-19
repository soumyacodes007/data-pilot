import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Chat with CSV", layout="wide")
st.title("💬 Chat with Your CSV Data")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Function to initialize the agent with the dataframe
def initialize_agent(df, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=api_key)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    return agent

# Sidebar for CSV upload and API key entry
with st.sidebar:
    st.header("Configuration")

    # Use a form for API key and initialization
    with st.form("api_key_form"):
        # API Key input - Use session state value
        entered_api_key = st.text_input(
            "Enter Google API Key:", 
            type="password", 
            value=st.session_state.api_key, # Use session state value
            help="Enter your Google API key to use the Gemini model. Press Enter or click Set Key."
        )

        submitted = st.form_submit_button("Set API Key & Initialize Agent")

        if submitted:
            st.session_state.api_key = entered_api_key
            st.success("API Key updated in session.")

            # Attempt to initialize agent immediately after setting the key
            if st.session_state.api_key and st.session_state.df is not None:
                # Re-initialize agent only if key changes or not initialized yet
                if st.session_state.get("_current_agent_key") != st.session_state.api_key: 
                    try:
                        with st.spinner("Initializing the chat agent..."):
                           st.session_state.agent = initialize_agent(st.session_state.df, st.session_state.api_key)
                           st.session_state._current_agent_key = st.session_state.api_key # Track key used for agent
                           st.success("Chat agent initialized successfully!")
                    except Exception as agent_init_e:
                        st.error(f"Failed to initialize agent: {agent_init_e}")
                        st.session_state.agent = None # Ensure agent is None if init fails
                        st.session_state._current_agent_key = None
            elif not st.session_state.api_key:
                st.warning("API Key is missing. Agent cannot be initialized.")
            elif st.session_state.df is None:
                 st.warning("Please upload a CSV file first.")
    
    # CSV file uploader (outside the API key form, but logic might depend on it)
    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the dataframe
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded successfully! Shape: {st.session_state.df.shape}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head(5))
            
            # Initialize agent if API key exists and file is loaded 
            # (This part might be redundant if using the form button, but good as fallback)
            if st.session_state.api_key and st.session_state.df is not None:
                # Check if agent needs initialization (e.g., if key was set before file upload)
                if "agent" not in st.session_state or st.session_state.agent is None or \
                   st.session_state.get("_current_agent_key") != st.session_state.api_key:
                   # Check if the button was *not* just pressed to avoid double init
                   # This simple check might need refinement depending on desired UX
                   # if not submitted: # This might not work as expected due to Streamlit rerun
                    try:
                        # Don't show spinner if already initialized via form
                        if st.session_state.get("_current_agent_key") != st.session_state.api_key:
                           with st.spinner("Initializing the chat agent (file upload context)..."):
                                st.session_state.agent = initialize_agent(st.session_state.df, st.session_state.api_key)
                                st.session_state._current_agent_key = st.session_state.api_key 
                           st.success("Chat agent ready!")
                    except Exception as agent_init_e:
                        st.error(f"Failed to initialize agent: {agent_init_e}")
                        st.session_state.agent = None
                        st.session_state._current_agent_key = None
            elif not st.session_state.api_key:
                 st.warning("Please enter your Google API key to initialize the chat agent.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main area for chat
st.header("Chat with your data")

# Display information messages if needed
if st.session_state.df is None:
    st.info("👈 Please upload a CSV file to begin chatting with your data.")
# Check session state for API key
elif not st.session_state.api_key: 
    st.warning("⚠️ Please enter your Google API key in the sidebar to enable chat.")
elif st.session_state.agent is None:
     st.warning("⚠️ Agent not initialized. Please ensure the API key is correct and the agent could be created.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask something about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.agent:
                        # Use the raw prompt directly
                        response = st.session_state.agent.run(prompt)
                        st.write(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Agent not initialized. Please check your API key.")
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer with instructions
with st.expander("How to use this chat"):
    st.markdown("""
    1. Upload a CSV file using the sidebar
    2. Enter your Google API key (required for Gemini model)
    3. Ask questions about your data in natural language
    
    **Example questions you can ask:**
    - What's the shape of the dataset?
    - Show me the correlation between column X and Y
    - What are the top 5 values in column Z?
    - Create a histogram of column X
    - Identify outliers in the dataset
    """)
