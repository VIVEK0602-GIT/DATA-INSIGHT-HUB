# DataInsightHub

A comprehensive data analysis platform with AI-powered insights and visualization capabilities.

![DataInsightHub](https://img.shields.io/badge/DataInsightHub-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

DataInsightHub is an interactive data analysis platform built with Streamlit that combines powerful visualization tools, machine learning model evaluation, and AI-powered chat capabilities. It's designed to help data analysts and business users quickly gain insights from their data without writing code.

## Features

### Data Management
- Upload CSV and Excel files
- Save and retrieve datasets from a local database
- Load sample data for demo purposes

### Data Analysis
- Descriptive statistics for numerical and categorical data
- Correlation analysis with interactive heatmaps
- Principal Component Analysis (PCA)
- K-means clustering

### Visualization
- Create customizable charts:
  - Bar charts
  - Line charts
  - Scatter plots
  - Histograms
  - Box plots
  - Pie charts
  - Heatmaps
- Advanced chart customization options:
  - Custom color themes
  - Axis ranges
  - Layout settings
  - Interactive elements

### Performance Metrics
- Evaluate machine learning models:
  - Classification metrics (accuracy, precision, recall, F1)
  - Regression metrics (MSE, RMSE, MAE, R²)
  - Confusion matrices
  - ROC curves
  - Feature importance analysis
- Compare model performance (Logistic/Linear Regression vs Random Forest)

### AI-Powered Chat
- Ask questions about your data in natural language
- Generate visualizations through chat
- Powered by Google's Gemini AI

### Export Options
- Download data in CSV, Excel, or JSON formats
- Export visualizations as HTML, PNG, SVG, or JSON
- Save analyses and visualizations to the database

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataInsightHub.git
cd DataInsightHub
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at http://localhost:8501

3. Upload your data file or use the sample data

4. Explore the different tabs to analyze and visualize your data

5. Chat with the AI assistant to get insights about your data

## API Key Setup

To use the AI chat functionality, you need to set up a Google Gemini API key:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to create an API key
2. Add your API key to the `app.py` file:
```python
GEMINI_API_KEY = "your-api-key-here"
```

## Structure

- `app.py`: Main Streamlit application
- `data_utils.py`: Data processing and analysis utilities
- `database.py`: SQLite database interface for storing datasets and results
- `gemini_chat.py`: Integration with Google's Gemini AI
- `requirements.txt`: Python dependencies
- `data/`: Directory for database and uploaded files

## Known Issues and Solutions

1. When using datetime columns for model evaluation:
   - The system automatically converts datetime values to appropriate formats for machine learning
   - For classification tasks, datetime values are converted to strings
   - For regression tasks, datetime values are converted to ordinal values

2. Visualization rendering issues:
   - If a complex visualization fails to render, the system will attempt to create a simplified version
   - Advanced chart options are available in expandable sections

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Machine learning components from [scikit-learn](https://scikit-learn.org/)
- AI functionality from [Google Gemini](https://ai.google.dev/) 