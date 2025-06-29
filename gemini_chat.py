import google.generativeai as genai
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import requests

# Define the API endpoint and model
GEMINI_MODEL = "gemini-pro"  # Use the stable gemini-pro model
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def configure_gemini(api_key):
    """Configure the Gemini API with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False

def analyze_data_with_gemini(df, query, model_name=GEMINI_MODEL):
    """Use Gemini to analyze data based on user query"""
    
    # Prepare data context
    data_context = _prepare_data_context(df)
    
    # Format the prompt
    prompt = f"""
You are an expert data analyst using Python. I'm going to provide you with a dataset and a question about it.
Please analyze the data and answer the question. If the question asks for a visualization, please provide Python code using Plotly to create the visualization.

DATASET INFORMATION:
{data_context}

USER QUESTION: {query}

If you need to return code for visualization, please format it as follows:
```python
# Visualization code
import plotly.express as px
...
fig = px.line(...)  # Create a plotly figure
```

If you're providing a direct answer without visualization code, please format it clearly as a concise response.
"""
    
    try:
        # Try using the google.generativeai library first
        try:
            # Get Gemini model
            model = genai.GenerativeModel(model_name)
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Process the response
            processed_response = _process_gemini_response(response, df)
            return processed_response
        
        except Exception as lib_error:
            # If the library approach fails, use direct API call
            st.warning(f"Primary model failed: {str(lib_error)}. Trying direct API call...")
            
            # Get API key from session state
            api_key = st.session_state.get("gemini_api_key", "")
            if not api_key:
                return {
                    "response_type": "error",
                    "content": "No API key provided. Please set the Gemini API key."
                }
                
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            data = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 0.95,
                    "maxOutputTokens": 8192
                }
            }
            
            try:
                response = requests.post(
                    GEMINI_API_ENDPOINT,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    # Extract text from the response
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text = candidate["content"]["parts"][0].get("text", "")
                            # Create a response object similar to what the library would return
                            class SimpleResponse:
                                def __init__(self, text):
                                    self.text = text
                            
                            simple_response = SimpleResponse(text)
                            processed_response = _process_gemini_response(simple_response, df)
                            return processed_response
                
                # If direct API call fails, try alternate models
                st.warning("Primary model endpoint failed. Trying alternate models...")
                return try_alternate_models(prompt, df, api_key)
                
            except requests.RequestException as req_err:
                # If the direct API call fails, try alternate models
                st.warning(f"Request error: {str(req_err)}. Trying alternate models...")
                return try_alternate_models(prompt, df, api_key)
    
    except Exception as e:
        return {
            "response_type": "error",
            "content": f"Error communicating with Gemini API: {str(e)}"
        }

def _prepare_data_context(df):
    """Prepare data context information for the Gemini prompt"""
    # Basic dataframe info
    num_rows, num_cols = df.shape
    column_names = df.columns.tolist()
    column_types = df.dtypes.to_dict()
    
    # Sample data (first 5 rows)
    sample_data = df.head(5).to_dict(orient='records')
    
    # Descriptive statistics for numeric columns
    numeric_stats = df.describe().to_dict()
    
    # Create a text representation of the dataset
    context = f"""
Dataset Shape: {num_rows} rows, {num_cols} columns

Column Names and Types:
{json.dumps({col: str(column_types[col]) for col in column_names}, indent=2)}

Sample Data (first 5 rows):
{json.dumps(sample_data, indent=2, default=str)}

Summary Statistics for Numerical Columns:
{json.dumps(numeric_stats, indent=2, default=str)}
"""
    
    return context

def _process_gemini_response(response, df):
    """Process the response from Gemini API"""
    if not response or not response.text:
        return {
            "response_type": "error",
            "content": "Received empty response from Gemini"
        }
    
    response_text = response.text
    
    # Check if the response contains Python code for visualization
    code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    
    if code_match:
        # Extract the Python code
        code = code_match.group(1).strip()
        
        try:
            # Create a namespace with necessary imports and dataframe
            namespace = {
                "pd": pd,
                "px": px,
                "go": go,
                "np": np,
                "df": df,
                "fig": None
            }
            
            # Execute the visualization code
            exec(code, namespace)
            
            # Get the figure from the namespace
            fig = namespace.get("fig")
            
            if fig:
                # The code successfully created a visualization
                return {
                    "response_type": "visualization",
                    "content": response_text.replace(code_match.group(0), "").strip(),
                    "figure": fig
                }
            else:
                # Code executed but no figure was created
                return {
                    "response_type": "error",
                    "content": "The code executed successfully but did not create a visualization figure."
                }
        except Exception as e:
            # Code execution failed
            error_message = f"Failed to generate visualization: {str(e)}"
            print(f"Code execution error: {error_message}")
            print(f"Code that failed: {code}")
            
            return {
                "response_type": "error",
                "content": error_message,
                "original_response": response_text,
                "failed_code": code
            }
    
    # No visualization code found, return the text response
    return {
        "response_type": "text",
        "content": response_text
    }

def try_alternate_models(prompt, df, api_key):
    """Try different Gemini model versions if the main one fails"""
    alternate_models = [
        "gemini-1.0-pro",
        "gemini-1.0-pro-latest",
        "gemini-1.5-flash-latest"
    ]
    
    for model in alternate_models:
        try:
            # Try using the library first
            try:
                genai_model = genai.GenerativeModel(model)
                response = genai_model.generate_content(prompt)
                return _process_gemini_response(response, df)
            except Exception:
                # Try direct API call
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
                
                data = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.4,
                        "topK": 32,
                        "topP": 0.95,
                        "maxOutputTokens": 8192
                    }
                }
                
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text = candidate["content"]["parts"][0].get("text", "")
                            class SimpleResponse:
                                def __init__(self, text):
                                    self.text = text
                            
                            simple_response = SimpleResponse(text)
                            return _process_gemini_response(simple_response, df)
        except Exception:
            # Try next model
            continue
    
    # If all models fail, return error
    return {
        "response_type": "error",
        "content": "All Gemini model attempts failed. Please check your API key and try again later."
    } 