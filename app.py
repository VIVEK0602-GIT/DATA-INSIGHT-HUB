import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import json
import base64

# Import our modules
import database as db
import data_utils as du
import gemini_chat as gemini

# Global visualization configuration for consistent appearance
def apply_standard_layout(fig, title, x_title=None, y_title=None):
    """Apply standard layout settings to ensure text visibility"""
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 22, 'color': 'white'}
        },
        paper_bgcolor='rgba(30, 30, 45, 1)',  # Dark blue-gray background
        plot_bgcolor='rgba(30, 30, 45, 1)',   # Match paper background
        font={'color': 'white', 'size': 14},  # White text for all elements
        margin=dict(t=100, b=80, l=80, r=40), # Increased margins for labels
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=14, color='white'),
            gridcolor='rgba(80, 80, 100, 0.3)'
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(size=16, color='white')
            ),
            tickfont=dict(size=14, color='white'),
            gridcolor='rgba(80, 80, 100, 0.3)'
        ),
        legend=dict(
            font=dict(size=14, color='white'),
            bgcolor='rgba(30, 30, 45, 0.5)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        )
    )
    return fig

# Helper function for safe visualization creation with improved visibility
def safe_create_visualization(df, chart_type, x_axis, y_axis, color=None, title=None, height=None, width=None, **kwargs):
    """Create visualization with better error handling and improved text visibility"""
    try:
        # Make sure height is at least 600px to avoid text_area rendering issues
        if height is None or height < 600:
            height = 600
        
        # Create visualization based on chart type
        fig = None
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=title, height=height, width=width, **kwargs)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, color=color, title=title, height=height, width=width, **kwargs)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=title, height=height, width=width, **kwargs)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color, title=title, height=height, width=width, **kwargs)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color, title=title, height=height, width=width, **kwargs)
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis, title=title, height=height, width=width, **kwargs)
        
        if fig:
            # Apply standard layout for improved visibility
            apply_standard_layout(fig, title, x_title=x_axis, y_title=y_axis)
            
            # For categorical charts, ensure text is readable
            if chart_type in ["Bar Chart", "Pie Chart"]:
                fig.update_traces(textfont=dict(size=14, color='white'))
            
            # For histograms, improve bin visibility
            if chart_type == "Histogram":
                fig.update_traces(marker=dict(
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ))
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Configure page
st.set_page_config(
    page_title="DataInsightHub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyDVOxhkHvtkb-E1dcrInXGlHoW5Yzw2nWE"  # Replace with your actual API key

try:
    success = gemini.configure_gemini(GEMINI_API_KEY)
    gemini_configured = success
    if success:
        st.session_state.gemini_api_key = GEMINI_API_KEY
except Exception as e:
    gemini_configured = False
    gemini_error = str(e)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_visualization' not in st.session_state:
    st.session_state.current_visualization = None

# Main heading
st.title("DataInsightHub")
st.write("A data analysis platform with AI-powered chat capabilities")

# Sidebar for data management
with st.sidebar:
    st.header("Data Management")
    
    st.markdown("---")
    
    # Data loading options
    st.subheader("📂 Load Data")
    
    # Option 1: Upload file
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
    
    # Option 2: Load from database
    st.markdown("### 💾 Saved Datasets")
    try:
        datasets = db.get_datasets()
        if not datasets.empty:
            dataset_options = {f"{row['name']} (ID: {row['id']})": row['id'] for _, row in datasets.iterrows()}
            selected_dataset = st.selectbox("Select a saved dataset", 
                                           options=list(dataset_options.keys()),
                                           index=None)
            if selected_dataset:
                selected_id = dataset_options[selected_dataset]
                dataset_info = db.get_dataset(selected_id)
                if dataset_info is not None:
                    # Get data from file path or recreate if needed
                    if os.path.exists(dataset_info['file_path']):
                        if dataset_info['file_path'].endswith('.csv'):
                            st.session_state.df = pd.read_csv(dataset_info['file_path'])
                        elif dataset_info['file_path'].endswith('.xlsx'):
                            st.session_state.df = pd.read_excel(dataset_info['file_path'])
                    else:
                        st.warning("Original file not found. Using sample data.")
                        st.session_state.df = du.create_sample_data()
                    
                    st.session_state.dataset_id = selected_id
                    st.session_state.file_info = {
                        'filename': dataset_info['name'],
                        'size': dataset_info['file_size'],
                        'shape': (dataset_info['row_count'], len(json.loads(dataset_info['column_info'])['columns']))
                    }
                    st.success(f"Loaded dataset: {dataset_info['name']}")
                    st.rerun()
        else:
            st.info("No saved datasets found.")
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
    
    # Option 3: Use sample data
    st.markdown("### 🧪 Sample Data")
    if st.button("Load Sample Data"):
        st.session_state.df = du.create_sample_data()
        st.session_state.file_info = {
            'filename': 'sample_data.csv',
            'size': len(st.session_state.df) * len(st.session_state.df.columns) * 8,
            'type': 'text/csv',
            'shape': st.session_state.df.shape
        }
        st.success("Sample data loaded successfully!")
        st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df, st.session_state.file_info = du.load_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.df, st.session_state.file_info = du.load_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                
            # Show file details
            if st.session_state.df is not None:
                st.success("File uploaded successfully!")
                
                # Save to database option
                st.markdown("### 💾 Save to Database")
                save_name = st.text_input("Dataset Name", value=uploaded_file.name)
                save_desc = st.text_area("Description", value="")
                
                if st.button("Save Dataset"):
                    try:
                        # Save file locally
                        os.makedirs("data/uploads", exist_ok=True)
                        file_path = f"data/uploads/{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Save to database
                        dataset_id = db.save_dataset(
                            name=save_name,
                            description=save_desc,
                            df=st.session_state.df,
                            file_path=file_path,
                            file_size=uploaded_file.size
                        )
                        st.session_state.dataset_id = dataset_id
                        st.success(f"Dataset saved with ID: {dataset_id}")
                    except Exception as e:
                        st.error(f"Error saving dataset: {str(e)}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # If data is loaded, show summary
    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📋 Data Summary")
        st.markdown(f"""
        **File**: {st.session_state.file_info['filename']}  
        **Rows**: {st.session_state.file_info['shape'][0]}  
        **Columns**: {st.session_state.file_info['shape'][1]}
        """)
        
        # Clear data button
        if st.button("Clear Data"):
            st.session_state.df = None
            st.session_state.file_info = None
            st.session_state.dataset_id = None
            st.session_state.current_visualization = None
            st.rerun()

# Main content area - only show if data is loaded
if st.session_state.df is not None:
    # Create tabs for different functionalities
    tabs = st.tabs(["Data Preview", "Analysis", "Visualization", "Performance Metrics", "Chat with Data", "Saved Items"])
    
    # Tab 1: Data Preview
    with tabs[0]:
        st.header("Data Preview")
        
        # Display the first few rows of the data
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Display data types and summary
        st.subheader("Column Information")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Data types
            dtypes_df = pd.DataFrame({
                'Column': st.session_state.df.columns.tolist(),
                'Data Type': [str(dtype) for dtype in st.session_state.df.dtypes.values]
            })
            # Convert all columns to string type to avoid Arrow conversion issues
            dtypes_df = dtypes_df.astype(str)
            st.dataframe(dtypes_df, use_container_width=True)
        
        with col2:
            # Missing values
            missing_df = pd.DataFrame({
                'Column': st.session_state.df.columns.tolist(),
                'Missing Values': st.session_state.df.isna().sum().values.tolist(),
                'Missing %': (st.session_state.df.isna().sum().values / len(st.session_state.df) * 100).round(2).tolist()
            })
            # Convert all columns to string type to avoid Arrow conversion issues
            missing_df = missing_df.astype(str)
            st.dataframe(missing_df, use_container_width=True)
        
        # Download options
        st.subheader("Download Data")
        download_format = st.selectbox("Select format", ["CSV", "Excel", "JSON"])
        
        if download_format == "CSV":
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{st.session_state.file_info['filename']}.csv",
                mime="text/csv"
            )
        elif download_format == "Excel":
            buffer = io.BytesIO()
            st.session_state.df.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name=f"{st.session_state.file_info['filename']}.xlsx",
                mime="application/vnd.ms-excel"
            )
        elif download_format == "JSON":
            json_str = st.session_state.df.to_json(orient="records")
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{st.session_state.file_info['filename']}.json",
                mime="application/json"
            )
    
    # Tab 2: Analysis
    with tabs[1]:
        st.header("Data Analysis")
        
        # Create subtabs for different analyses
        analysis_tabs = st.tabs(["Descriptive Statistics", "Correlation Analysis", "Advanced Analysis"])
        
        # Subtab 1: Descriptive Statistics
        with analysis_tabs[0]:
            st.subheader("Descriptive Statistics")
            
            # Get numeric columns only
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Select columns to analyze
                selected_cols = st.multiselect(
                    "Select columns for analysis",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]  # Default to first 5 numeric columns
                )
                
                if selected_cols:
                    # Calculate statistics
                    stats = du.get_descriptive_stats(st.session_state.df[selected_cols])
                    
                    # Display statistics
                    st.dataframe(stats, use_container_width=True)
                    
                    # Save statistics
                    if st.session_state.dataset_id and st.button("Save Statistics to Database"):
                        try:
                            result_id = db.save_analysis_result(
                                dataset_id=st.session_state.dataset_id,
                                analysis_type="descriptive_statistics",
                                result_data=stats.to_dict()
                            )
                            st.success(f"Statistics saved with ID: {result_id}")
                        except Exception as e:
                            st.error(f"Error saving statistics: {str(e)}")
                    
                    # Download option
                    csv = stats.to_csv()
                    st.download_button(
                        label="Download Statistics as CSV",
                        data=csv,
                        file_name="descriptive_statistics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Please select at least one column for analysis")
            else:
                st.warning("No numeric columns found in the dataset")
            
            # For categorical data
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.subheader("Categorical Data Analysis")
                
                selected_cat_cols = st.multiselect(
                    "Select categorical columns for analysis",
                    cat_cols,
                    default=cat_cols[:min(2, len(cat_cols))]
                )
                
                if selected_cat_cols:
                    # Get categorical statistics
                    cat_stats = du.get_categorical_stats(st.session_state.df[selected_cat_cols])
                    
                    for idx, col in enumerate(selected_cat_cols):
                        st.write(f"### Column: {col}")
                        
                        # Get value counts
                        value_counts = st.session_state.df[col].value_counts().reset_index()
                        value_counts.columns = ['Value', 'Count']  # Explicitly name columns
                        
                        # Display the top 10 values
                        st.dataframe(value_counts.head(10).astype(str), use_container_width=True)
                        
                        # Visualization of distribution
                        try:
                            # Convert all values to strings to avoid any typecasting issues
                            value_counts['Value'] = value_counts['Value'].astype(str)
                            
                            # Use simpler Streamlit bar chart for better stability
                            st.write(f"### Distribution of {col}")
                            
                            # Limit to top 10 values for better readability
                            chart_data = value_counts.head(10)
                            st.bar_chart(chart_data, x='Value', y='Count')
                            
                            # Display data table with value counts
                            st.write("Value Distribution:")
                            st.dataframe(value_counts.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                            st.code(str(e))
                            
                            # Try simpler method
                            try:
                                st.warning("Trying simplified data table...")
                                st.dataframe(value_counts.head(10))
                            except Exception as fallback_error:
                                st.error(f"Fallback display also failed: {str(fallback_error)}")
                                st.code(str(fallback_error))
        
        # Subtab 2: Correlation Analysis
        with analysis_tabs[1]:
            st.subheader("Correlation Analysis")
            
            # Only show for numeric data
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Select columns for correlation
                selected_cols = st.multiselect(
                    "Select columns for correlation analysis",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    # Select correlation method
                    method = st.selectbox(
                        "Correlation Method",
                        ["pearson", "spearman", "kendall"],
                        index=0
                    )
                    
                    # Calculate correlation
                    corr_matrix = st.session_state.df[selected_cols].corr(method=method)
                    
                    # Show correlation matrix
                    st.write("### Correlation Matrix")
                    try:
                        # Display correlation matrix as a styled dataframe
                        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                        
                        # Display top correlations
                        st.write("### Top Correlations")
                        
                        # Get upper triangle of correlation matrix (to avoid duplicates)
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        corr_df_masked = corr_matrix.mask(mask)
                        
                        # Unstack, sort and get top 5
                        top_corr = corr_df_masked.unstack().sort_values(kind="quicksort", ascending=False).dropna()
                        top_corr = top_corr[top_corr != 1].head(5)  # Exclude self-correlations (correlation=1)
                        
                        # Create and display a dataframe with top correlations
                        if not top_corr.empty:
                            top_corr_df = pd.DataFrame(top_corr).reset_index()
                            top_corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
                            st.dataframe(top_corr_df, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error creating correlation visualization: {str(e)}")
                        st.code(str(e))
                        # Fallback to simple display
                        st.write("Correlation matrix (fallback view):")
                        st.dataframe(corr_matrix.round(2), use_container_width=True)
                    
                    # Save to database
                    if st.session_state.dataset_id and st.button("Save Correlation to Database"):
                        try:
                            # Convert the correlation matrix to a valid JSON format
                            corr_data = corr_matrix.reset_index().to_dict(orient='records')
                            result_id = db.save_analysis_result(
                                dataset_id=st.session_state.dataset_id,
                                analysis_type="correlation_matrix",
                                result_data=json.dumps(corr_data)
                            )
                            st.success(f"Correlation matrix saved with ID: {result_id}")
                        except Exception as e:
                            st.error(f"Error saving correlation matrix: {str(e)}")
                            st.code(str(e))
                    
                    # Download option
                    csv = corr_matrix.to_csv()
                    st.download_button(
                        label="Download Correlation Matrix as CSV",
                        data=csv,
                        file_name="correlation_matrix.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Please select at least two columns for correlation analysis")
            else:
                st.warning("Need at least two numeric columns for correlation analysis")
        
        # Subtab 3: Advanced Analysis
        with analysis_tabs[2]:
            st.subheader("Advanced Analysis")
            
            # PCA analysis
            if st.checkbox("Principal Component Analysis (PCA)"):
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    # Select columns for PCA
                    selected_cols = st.multiselect(
                        "Select columns for PCA",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))],
                        key="pca_columns"
                    )
                    
                    if len(selected_cols) >= 2:
                        # Number of components
                        n_components = st.slider(
                            "Number of Components",
                            min_value=2,
                            max_value=min(len(selected_cols), 10),
                            value=2
                        )
                        
                        # Run PCA
                        if st.button("Run PCA"):
                            # Perform PCA analysis
                            pca_df, explained_variance, loadings = du.run_pca_analysis(
                                st.session_state.df[selected_cols],
                                n_components=n_components
                            )
                            
                            if pca_df is not None:
                                # Show explained variance
                                st.write("### Explained Variance")
                                explained_var_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(n_components)],
                                    'Explained Variance (%)': [v * 100 for v in explained_variance]
                                })
                                st.dataframe(explained_var_df, use_container_width=True)
                                
                                # Show loadings
                                st.write("### Component Loadings")
                                st.dataframe(loadings, use_container_width=True)
                                
                                # Visualization of first two components
                                st.write("### PCA Scatter Plot (First Two Components)")
                                fig = px.scatter(
                                    pca_df,
                                    x='PC1',
                                    y='PC2',
                                    title="PCA Projection"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least two columns for PCA")
                else:
                    st.warning("Need at least two numeric columns for PCA")
            
            # K-means clustering
            if st.checkbox("K-means Clustering"):
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    # Select columns for clustering
                    selected_cols = st.multiselect(
                        "Select columns for clustering",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))],
                        key="kmeans_columns"
                    )
                    
                    if len(selected_cols) >= 2:
                        # Number of clusters
                        n_clusters = st.slider(
                            "Number of Clusters",
                            min_value=2,
                            max_value=10,
                            value=3
                        )
                        
                        # Run clustering
                        if st.button("Run K-means Clustering"):
                            # Perform clustering
                            cluster_df, centers = du.run_kmeans_clustering(
                                st.session_state.df[selected_cols],
                                n_clusters=n_clusters
                            )
                            
                            if cluster_df is not None:
                                # Show cluster centers
                                st.write("### Cluster Centers")
                                st.dataframe(centers, use_container_width=True)
                                
                                # Show cluster distribution
                                st.write("### Cluster Distribution")
                                cluster_counts = cluster_df['cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                fig = px.bar(
                                    cluster_counts,
                                    x='Cluster',
                                    y='Count',
                                    title="Distribution of Clusters"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Visualization of clusters using first two features
                                if len(selected_cols) >= 2:
                                    st.write("### Cluster Visualization")
                                    fig = px.scatter(
                                        cluster_df,
                                        x=selected_cols[0],
                                        y=selected_cols[1],
                                        color='cluster',
                                        title=f"Clusters by {selected_cols[0]} and {selected_cols[1]}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least two columns for clustering")
                else:
                    st.warning("Need at least two numeric columns for clustering")
    
    # Tab 3: Visualization
    with tabs[2]:
        st.header("Data Visualization")
        
        # Create two columns: one for controls, one for the visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Visualization controls
            st.subheader("Chart Options")
            
            # Chart type selection
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap"],
                key="chart_type_select"
            )
            
            # Get column lists by type
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            temporal_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
            all_cols = st.session_state.df.columns.tolist()
            
            # Dynamic controls based on chart type
            x_axis = None
            y_axis = None
            
            if chart_type == "Bar Chart":
                # Only allow categorical or datetime for X, numeric for Y
                valid_x = categorical_cols + temporal_cols
                valid_y = numeric_cols
                x_axis = st.selectbox("X-axis (Categories)", valid_x, key="bar_x_axis")
                y_axis = st.selectbox("Y-axis (Values)", valid_y, key="bar_y_axis")
            elif chart_type == "Line Chart":
                # Only allow datetime or numeric for X, numeric for Y
                valid_x = temporal_cols + numeric_cols
                valid_y = numeric_cols
                x_axis = st.selectbox("X-axis (Time or Numeric)", valid_x, key="line_x_axis")
                y_axis = st.selectbox("Y-axis (Values)", valid_y, key="line_y_axis")
            elif chart_type == "Scatter Plot":
                # Only allow numeric for both axes
                x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x_axis")
                y_axis = st.selectbox("Y-axis", numeric_cols, key="scatter_y_axis")
            elif chart_type == "Histogram":
                x_axis = st.selectbox("Value", numeric_cols, key="hist_x_axis")
                bins = st.slider("Number of bins", 5, 50, 10, key="hist_bins")
            elif chart_type == "Box Plot":
                x_axis = st.selectbox("X-axis (Categories)", categorical_cols + temporal_cols, key="box_x_axis")
                y_axis = st.selectbox("Y-axis (Values)", numeric_cols, key="box_y_axis")
            elif chart_type == "Pie Chart":
                x_axis = st.selectbox("Labels (Categories)", categorical_cols, key="pie_x_axis")
                y_axis = st.selectbox("Values (Numeric)", numeric_cols, key="pie_y_axis")
            elif chart_type == "Heatmap":
                st.info("Heatmap will use the correlation matrix of numeric columns.")
            
            # Chart title
            title = st.text_input("Chart Title", f"{chart_type} of {x_axis if x_axis else 'data'}", key="chart_title")
            
            # Create visualization button
            create_viz = st.button("Create Visualization")
        
        with col2:
            # Display area for visualization
            if create_viz and not st.session_state.get('viz_created', False):
                st.session_state['viz_created'] = True
                st.write("Creating visualization...")
                try:
                    error_messages = []
                    # Validation for chart type and axes
                    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Pie Chart"] and (x_axis is None or y_axis is None):
                        error_messages.append("Please select both X-axis and Y-axis columns")
                    elif chart_type == "Histogram" and x_axis is None:
                        error_messages.append("Please select a Value column")
                    # Warn if too many unique values for X-axis
                    if x_axis is not None and st.session_state.df[x_axis].nunique() > 50:
                        error_messages.append(f"Too many unique values in {x_axis} for a meaningful chart. Please select a column with fewer unique values.")
                    if error_messages:
                        for msg in error_messages:
                            st.warning(msg)
                        st.info("Please check your selections and try again.")
                    else:
                        # Group datetime X-axis if needed
                        df_viz = st.session_state.df.copy()
                        if x_axis in temporal_cols:
                            df_viz[x_axis] = pd.to_datetime(df_viz[x_axis])
                            df_viz['__grouped_x__'] = df_viz[x_axis].dt.date
                            x_axis_grouped = '__grouped_x__'
                        else:
                            x_axis_grouped = x_axis
                        # Use the simplified visualization function
                        if chart_type == "Histogram":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis,
                                y_axis=None,
                                title=title
                            )
                        elif chart_type == "Bar Chart":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis_grouped,
                                y_axis=y_axis,
                                title=title
                            )
                        elif chart_type == "Line Chart":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis_grouped,
                                y_axis=y_axis,
                                title=title
                            )
                        elif chart_type == "Scatter Plot":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis,
                                y_axis=y_axis,
                                title=title
                            )
                        elif chart_type == "Box Plot":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis_grouped,
                                y_axis=y_axis,
                                title=title
                            )
                        elif chart_type == "Pie Chart":
                            result = safe_create_visualization(
                                df_viz,
                                chart_type,
                                x_axis=x_axis,
                                y_axis=y_axis,
                                title=title
                            )
                        elif chart_type == "Heatmap":
                            # Use correlation matrix
                            corr = df_viz[numeric_cols].corr()
                            result = px.imshow(corr, text_auto=True, title=title)
                        else:
                            result = None
                        if result:
                            st.plotly_chart(result, use_container_width=True)
                            st.write("### Data Summary")
                            if chart_type != "Histogram" and y_axis:
                                st.write(f"**Statistics for {y_axis}:**")
                                st.dataframe(st.session_state.df[y_axis].describe())
                            if x_axis:
                                st.write(f"**Distribution of {x_axis}:**")
                                value_counts = st.session_state.df[x_axis].value_counts().head(10).reset_index()
                                value_counts.columns = ['Value', 'Count']
                                st.dataframe(value_counts)
                        else:
                            st.error("Could not create visualization. Please try different selections.")
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.code(str(e))
            elif not create_viz:
                st.session_state['viz_created'] = False
                st.info("Configure your visualization options and click 'Create Visualization'")
                st.write("### Sample Visualization Preview")
                sample_df = pd.DataFrame({
                    'category': ['A', 'B', 'C', 'D'],
                    'value': [10, 20, 15, 25]
                })
                st.bar_chart(sample_df, x='category', y='value')
    
    # Tab 4: Performance Metrics
    with tabs[3]:
        st.header("Model Performance Metrics")
        st.write("Evaluate machine learning model performance on your data")
        
        # Select model type
        model_type = st.selectbox(
            "Model Type",
            ["Classification", "Regression"],
            key="model_type"
        )
        
        # Get column lists by type
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = st.session_state.df.columns.tolist()
        
        # Select target column
        if model_type == "Classification":
            # For classification, only allow categorical or integer columns with <30 unique values
            valid_targets = [col for col in all_cols if (str(st.session_state.df[col].dtype) in ["object", "category", "int64", "int32"] and st.session_state.df[col].nunique() <= 30)]
            if not valid_targets:
                st.warning("No suitable target columns for classification (must be categorical/integer with <=30 unique values).")
                target_col = None
            else:
                target_col = st.selectbox(
                    "Target Column (to predict)",
                    valid_targets,
                    key="classification_target"
                )
        else:
            # For regression, the target should be numeric
            if numeric_cols:
                target_col = st.selectbox(
                    "Target Column (to predict)",
                    numeric_cols,
                    key="regression_target"
                )
            else:
                st.warning("No numeric columns found for regression target.")
                target_col = None
        
        # Select feature columns
        if target_col:
            # Remove target column from possible features
            available_features = [col for col in all_cols if col != target_col]
            feature_cols = st.multiselect(
                "Feature Columns (predictors)",
                available_features,
                default=available_features[:min(5, len(available_features))],
                key="feature_columns"
            )
            # Test size slider
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="test_size"
            )
            # Button to run evaluation
            if st.button("Evaluate Model Performance", key="evaluate_model"):
                if feature_cols and target_col:
                    with st.spinner("Training and evaluating models..."):
                        try:
                            performance_data = du.evaluate_model_performance(
                                df=st.session_state.df,
                                target_col=target_col,
                                feature_cols=feature_cols,
                                model_type=model_type.lower(),
                                test_size=test_size
                            )
                            if performance_data['success']:
                                st.session_state.performance_data = performance_data
                                metric_tabs = st.tabs([
                                    "Model Comparison", 
                                    "Feature Importance", 
                                    "Detailed Metrics"
                                ])
                                with metric_tabs[0]:
                                    if model_type.lower() == 'classification':
                                        st.subheader("Classification Metrics")
                                        metrics_df = pd.DataFrame({
                                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                            'Logistic Regression': [
                                                performance_data['logistic_regression']['accuracy'],
                                                performance_data['logistic_regression']['precision'],
                                                performance_data['logistic_regression']['recall'],
                                                performance_data['logistic_regression']['f1']
                                            ],
                                            'Random Forest': [
                                                performance_data['random_forest']['accuracy'],
                                                performance_data['random_forest']['precision'],
                                                performance_data['random_forest']['recall'],
                                                performance_data['random_forest']['f1']
                                            ]
                                        })
                                        st.dataframe(metrics_df, use_container_width=True)
                                        chart_data = pd.DataFrame({
                                            'Logistic Regression': metrics_df['Logistic Regression'],
                                            'Random Forest': metrics_df['Random Forest']
                                        }, index=metrics_df['Metric'])
                                        st.subheader("Metrics Comparison")
                                        st.bar_chart(chart_data)
                                        st.subheader("Confusion Matrix (Random Forest)")
                                        cm = np.array(performance_data['confusion_matrix']['random_forest'])
                                        labels = list(performance_data.get('label_mapping', {}).keys())
                                        if not labels:
                                            labels = [str(i) for i in range(cm.shape[0])]
                                        if cm.shape[0] == len(labels):
                                            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                                            st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                                        else:
                                            st.error(f"Confusion matrix shape mismatch: {cm.shape[0]} vs {len(labels)} labels. Showing raw matrix.")
                                            st.dataframe(pd.DataFrame(cm), use_container_width=True)
                                    else:
                                        st.subheader("Regression Metrics")
                                        metrics_df = pd.DataFrame({
                                            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
                                            'Linear Regression': [
                                                performance_data['linear_regression']['mse'],
                                                performance_data['linear_regression']['rmse'],
                                                performance_data['linear_regression']['mae'],
                                                performance_data['linear_regression']['r2']
                                            ],
                                            'Random Forest': [
                                                performance_data['random_forest']['mse'],
                                                performance_data['random_forest']['rmse'],
                                                performance_data['random_forest']['mae'],
                                                performance_data['random_forest']['r2']
                                            ]
                                        })
                                        st.dataframe(metrics_df, use_container_width=True)
                                        error_metrics = ['MSE', 'RMSE', 'MAE']
                                        error_chart_data = pd.DataFrame({
                                            'Linear Regression': metrics_df.loc[metrics_df['Metric'].isin(error_metrics), 'Linear Regression'].values,
                                            'Random Forest': metrics_df.loc[metrics_df['Metric'].isin(error_metrics), 'Random Forest'].values
                                        }, index=error_metrics)
                                        st.subheader("Error Metrics Comparison")
                                        st.bar_chart(error_chart_data)
                                        st.subheader("R² Score Comparison")
                                        r2_data = pd.DataFrame({
                                            'Model': ['Linear Regression', 'Random Forest'],
                                            'R²': [
                                                performance_data['linear_regression']['r2'],
                                                performance_data['random_forest']['r2']
                                            ]
                                        })
                                        st.bar_chart(r2_data, x='Model', y='R²')
                                with metric_tabs[1]:
                                    st.subheader("Feature Importance")
                                    feature_data = pd.DataFrame(performance_data['feature_importance'])
                                    feature_data = feature_data.sort_values('importance', ascending=False)
                                    feature_data['importance'] = feature_data['importance'].round(4)
                                    st.dataframe(feature_data, use_container_width=True)
                                    st.bar_chart(feature_data, x='feature', y='importance')
                                with metric_tabs[2]:
                                    if model_type.lower() == 'classification':
                                        st.subheader("Classification Details")
                                        if 'roc_curve' in performance_data:
                                            st.write(f"AUC Score (Logistic Regression): {performance_data['roc_curve']['logistic_regression']['auc']:.4f}")
                                            st.write(f"AUC Score (Random Forest): {performance_data['roc_curve']['random_forest']['auc']:.4f}")
                                        st.subheader("Prediction vs Actual (Sample)")
                                        if 'predictions' in performance_data:
                                            pred_df = pd.DataFrame(performance_data['predictions'][:20])
                                            st.dataframe(pred_df, use_container_width=True)
                                    else:
                                        st.subheader("Prediction vs Actual (Sample)")
                                        if 'predictions' in performance_data:
                                            pred_df = pd.DataFrame(performance_data['predictions'][:20])
                                            pred_df = pred_df.round(2)
                                            st.dataframe(pred_df, use_container_width=True)
                                if st.session_state.dataset_id:
                                    st.subheader("Save Results")
                                    if st.button("Save Performance Metrics", key="save_metrics"):
                                        try:
                                            result_id = db.save_analysis_result(
                                                dataset_id=st.session_state.dataset_id,
                                                analysis_type="model_performance",
                                                result_data=json.dumps(performance_data, default=str)
                                            )
                                            st.success(f"Performance metrics saved with ID: {result_id}")
                                        except Exception as e:
                                            st.error(f"Error saving performance metrics: {str(e)}")
                                            st.code(str(e))
                            else:
                                st.error(f"Error evaluating model performance: {performance_data['error']}")
                        except Exception as e:
                            st.error(f"Error during model evaluation: {str(e)}")
                            st.code(str(e))
                else:
                    st.warning("Please select target column and at least one feature column.")
        else:
            st.info("Please select a target column to predict.")
    
    # Tab 5: Chat with Data
    with tabs[4]:
        st.header("Chat with Your Data")
        st.write("Ask questions about your data in natural language using Gemini AI.")
        
        # Check if Gemini API is configured correctly
        if not gemini_configured:
            st.error(f"Gemini API is not configured correctly. Error: {gemini_error}")
            st.info("Please update the GEMINI_API_KEY in app.py with your actual API key.")
            st.stop()  # Stop execution of this tab if Gemini is not configured
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
                
                # If the message has a figure, display it
                if "figure" in message:
                    try:
                        st.plotly_chart(message["figure"])
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
        
        # User input for new questions
        user_question = st.text_input("Ask a question about your data:")
        
        # Button to submit question
        if st.button("Ask", key="ask_button"):
            if user_question:
                # Add user question to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Use a spinner while getting the response
                with st.spinner("Thinking..."):
                    try:
                        # Get response from Gemini
                        response = gemini.analyze_data_with_gemini(st.session_state.df, user_question)
                        
                        # Process the response based on type
                        if response["response_type"] == "visualization":
                            # Response includes a visualization
                            try:
                                ai_message = {
                                    "role": "assistant",
                                    "content": response["content"],
                                    "figure": response["figure"]
                                }
                                st.session_state.chat_history.append(ai_message)
                                
                                # Display the response and visualization
                                st.markdown(f"**AI:** {response['content']}")
                                try:
                                    st.plotly_chart(response["figure"])
                                except Exception as viz_error:
                                    st.error(f"Error displaying visualization: {str(viz_error)}")
                                    st.code(str(viz_error))
                                
                                # Option to save the visualization
                                if st.session_state.dataset_id and st.button("Save This Visualization", key="save_viz_button"):
                                    try:
                                        # Convert figure to JSON-serializable format
                                        chart_config = response["figure"].to_dict()
                                        
                                        # Ensure all data is JSON serializable
                                        chart_config_json = json.dumps(chart_config, default=str)
                                        
                                        viz_id = db.save_visualization(
                                            dataset_id=st.session_state.dataset_id,
                                            title=f"Chat Visualization - {user_question[:30]}...",
                                            chart_type="chat_response",
                                            config=chart_config_json
                                        )
                                        st.success(f"Visualization saved with ID: {viz_id}")
                                    except Exception as e:
                                        st.error(f"Error saving visualization: {str(e)}")
                                        st.code(str(e))
                            except Exception as resp_error:
                                st.error(f"Error processing visualization response: {str(resp_error)}")
                                st.code(str(resp_error))
                        
                        elif response["response_type"] == "error":
                            # Error in processing
                            ai_message = {
                                "role": "assistant",
                                "content": f"Sorry, I encountered an error: {response['content']}"
                            }
                            st.session_state.chat_history.append(ai_message)
                            st.error(response["content"])
                        
                        else:
                            # Text response
                            ai_message = {
                                "role": "assistant",
                                "content": response["content"]
                            }
                            st.session_state.chat_history.append(ai_message)
                            st.markdown(f"**AI:** {response['content']}")
                    
                    except Exception as e:
                        # Handle any unexpected errors
                        error_message = f"Sorry, an error occurred: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                        st.error(error_message)
                        st.write("Detailed error information for debugging:")
                        st.code(str(e))
                
                # Force a rerun to update the UI with the new chat
                st.rerun()
                
        # Clear chat history button
        if st.session_state.chat_history and st.button("Clear Chat History", key="clear_chat_button"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Example questions
        st.subheader("Example Questions:")
        example_questions = [
            "What's the average sales by region?",
            "Show me the relationship between price and quantity",
            "Create a bar chart of total sales by product",
            "Which product has the highest average rating?",
            "What's the distribution of customer ages?"
        ]
        
        # Display example questions as buttons with unique keys
        cols = st.columns(len(example_questions))
        for i, (col, question) in enumerate(zip(cols, example_questions)):
            with col:
                if st.button(question, key=f"example_question_{i}"):
                    # Set the question and trigger a rerun to simulate clicking the Ask button
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    with st.spinner(f"Analyzing: {question}"):
                        try:
                            response = gemini.analyze_data_with_gemini(st.session_state.df, question)
                            
                            if response["response_type"] == "visualization":
                                try:
                                    ai_message = {
                                        "role": "assistant",
                                        "content": response["content"],
                                        "figure": response["figure"]
                                    }
                                    st.session_state.chat_history.append(ai_message)
                                except Exception as viz_error:
                                    st.error(f"Error with visualization: {str(viz_error)}")
                                    ai_message = {
                                        "role": "assistant",
                                        "content": f"Error creating visualization: {str(viz_error)}"
                                    }
                                    st.session_state.chat_history.append(ai_message)
                            elif response["response_type"] == "error":
                                ai_message = {
                                    "role": "assistant",
                                    "content": f"Sorry, I encountered an error: {response['content']}"
                                }
                                st.session_state.chat_history.append(ai_message)
                            else:
                                ai_message = {
                                    "role": "assistant",
                                    "content": response["content"]
                                }
                                st.session_state.chat_history.append(ai_message)
                        except Exception as e:
                            error_message = f"Sorry, an error occurred: {str(e)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    
                    st.rerun()
    
    # Tab 6: Saved Items
    with tabs[5]:
        st.header("Saved Items")
        st.write("View and manage saved datasets, analyses, and visualizations.")
        
        # Create subtabs for different item types
        saved_tabs = st.tabs(["Datasets", "Analyses", "Visualizations"])
        
        # Saved Datasets
        with saved_tabs[0]:
            st.subheader("Saved Datasets")
            
            # Get datasets from database
            try:
                datasets = db.get_datasets()
                if not datasets.empty:
                    for _, dataset in datasets.iterrows():
                        with st.expander(f"{dataset['name']} (ID: {dataset['id']})"):
                            st.write(f"**Description:** {dataset['description'] or 'No description'}")
                            st.write(f"**Rows:** {dataset['row_count']}")
                            st.write(f"**Created:** {dataset['created_at']}")
                            
                            # Load button
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Load Dataset", key=f"load_{dataset['id']}"):
                                    # Load dataset from file path or recreate
                                    try:
                                        if os.path.exists(dataset['file_path']):
                                            if dataset['file_path'].endswith('.csv'):
                                                st.session_state.df = pd.read_csv(dataset['file_path'])
                                            elif dataset['file_path'].endswith('.xlsx'):
                                                st.session_state.df = pd.read_excel(dataset['file_path'])
                                        else:
                                            st.warning("Original file not found. Using sample data.")
                                            st.session_state.df = du.create_sample_data()
                                        
                                        st.session_state.dataset_id = dataset['id']
                                        st.session_state.file_info = {
                                            'filename': dataset['name'],
                                            'size': dataset['file_size'],
                                            'shape': (dataset['row_count'], len(json.loads(dataset['column_info'])['columns']))
                                        }
                                        st.success(f"Loaded dataset: {dataset['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error loading dataset: {str(e)}")
                            
                            with col2:
                                # Delete button
                                if st.button(f"Delete Dataset", key=f"delete_{dataset['id']}"):
                                    confirm = st.checkbox(f"Confirm deletion of {dataset['name']}", key=f"confirm_{dataset['id']}")
                                    
                                    if confirm:
                                        try:
                                            # Delete file if it exists
                                            if os.path.exists(dataset['file_path']):
                                                try:
                                                    os.remove(dataset['file_path'])
                                                except:
                                                    st.warning("Could not delete the file, but database entry will be removed.")
                                            
                                            # Delete from database
                                            db.delete_dataset(dataset['id'])
                                            st.success(f"Dataset {dataset['name']} deleted.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting dataset: {str(e)}")
                else:
                    st.info("No saved datasets found.")
            except Exception as e:
                st.error(f"Error loading datasets: {str(e)}")
                st.code(str(e))
        
        # Saved Analyses
        with saved_tabs[1]:
            st.subheader("Saved Analyses")
            
            # Get analyses from database
            try:
                analyses = db.get_analysis_results()
                if not analyses.empty:
                    for _, analysis in analyses.iterrows():
                        with st.expander(f"{analysis['analysis_type']} (ID: {analysis['id']})"):
                            st.write(f"**Dataset ID:** {analysis['dataset_id']}")
                            st.write(f"**Created:** {analysis['created_at']}")
                            
                            # Display the analysis data
                            try:
                                result_data = json.loads(analysis['result_data'])
                                if isinstance(result_data, dict):
                                    # Convert to DataFrame for better display
                                    result_df = pd.DataFrame(result_data)
                                    st.dataframe(result_df, use_container_width=True)
                                else:
                                    st.json(result_data)
                            except Exception as e:
                                st.error(f"Could not parse result data: {str(e)}")
                            
                            # Delete button
                            if st.button(f"Delete Analysis", key=f"delete_analysis_{analysis['id']}"):
                                confirm = st.checkbox(f"Confirm deletion", key=f"confirm_analysis_{analysis['id']}")
                                
                                if confirm:
                                    try:
                                        # Delete from database
                                        db.delete_analysis_result(analysis['id'])
                                        st.success(f"Analysis deleted.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting analysis: {str(e)}")
                else:
                    st.info("No saved analyses found.")
            except Exception as e:
                st.error(f"Error loading analyses: {str(e)}")
                st.code(str(e))
        
        # Saved Visualizations
        with saved_tabs[2]:
            st.subheader("Saved Visualizations")
            
            # Get visualizations from database
            try:
                visualizations = db.get_visualizations()
                if not visualizations.empty:
                    for _, viz in visualizations.iterrows():
                        with st.expander(f"{viz['title']} (ID: {viz['id']})"):
                            st.write(f"**Dataset ID:** {viz['dataset_id']}")
                            st.write(f"**Chart Type:** {viz['chart_type']}")
                            st.write(f"**Created:** {viz['created_at']}")
                            
                            # Try to recreate the visualization
                            try:
                                # Parse the config
                                config = json.loads(viz['config'])
                                
                                # Create figure from config
                                try:
                                    fig = go.Figure(data=config.get('data', []), layout=config.get('layout', {}))
                                except Exception as fig_error:
                                    # Fallback approach for older or malformed configs
                                    st.warning(f"Using fallback visualization method: {str(fig_error)}")
                                    data_traces = []
                                    
                                    if 'data' in config:
                                        for trace in config['data']:
                                            trace_type = trace.get('type', 'scatter')
                                            
                                            if trace_type == 'bar':
                                                data_traces.append(go.Bar(
                                                    x=trace.get('x', []),
                                                    y=trace.get('y', []),
                                                    name=trace.get('name', ''),
                                                    marker=trace.get('marker', {})
                                                ))
                                            elif trace_type == 'scatter':
                                                data_traces.append(go.Scatter(
                                                    x=trace.get('x', []),
                                                    y=trace.get('y', []),
                                                    mode=trace.get('mode', 'markers'),
                                                    name=trace.get('name', ''),
                                                    marker=trace.get('marker', {})
                                                ))
                                            elif trace_type == 'pie':
                                                data_traces.append(go.Pie(
                                                    labels=trace.get('labels', []),
                                                    values=trace.get('values', []),
                                                    name=trace.get('name', '')
                                                ))
                                            else:
                                                st.warning(f"Unsupported trace type: {trace_type}")
                                        
                                        # Create figure with parsed traces
                                        fig = go.Figure(data=data_traces)
                                        
                                        # Add layout if available
                                        if 'layout' in config:
                                            fig.update_layout(**config['layout'])
                                    else:
                                        st.error("Invalid visualization config: No data found")
                                        fig = go.Figure()
                                
                                # Update layout with title
                                fig.update_layout(title=viz['title'], title_x=0.5)
                                
                                # Display the figure
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Export options
                                export_format = st.selectbox("Export As", ["HTML", "JSON"], key=f"export_{viz['id']}")
                                
                                if export_format == "HTML":
                                    try:
                                        buffer = io.StringIO()
                                        fig.write_html(buffer)
                                        html_bytes = buffer.getvalue().encode()
                                        st.download_button(
                                            label=f"Download HTML",
                                            data=html_bytes,
                                            file_name=f"{viz['title'].replace(' ', '_')}.html",
                                            mime="text/html",
                                            key=f"download_html_{viz['id']}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error creating HTML download: {str(e)}")
                                elif export_format == "JSON":
                                    try:
                                        # Create a clean JSON representation of the figure
                                        fig_json = json.dumps(fig.to_dict())
                                        st.download_button(
                                            label=f"Download JSON",
                                            data=fig_json,
                                            file_name=f"{viz['title'].replace(' ', '_')}.json",
                                            mime="application/json",
                                            key=f"download_json_{viz['id']}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error creating JSON download: {str(e)}")
                            except Exception as e:
                                st.error(f"Error recreating visualization: {str(e)}")
                            
                            # Delete button
                            if st.button(f"Delete Visualization", key=f"delete_viz_{viz['id']}"):
                                confirm = st.checkbox(f"Confirm deletion", key=f"confirm_viz_{viz['id']}")
                                
                                if confirm:
                                    try:
                                        # Delete from database
                                        db.delete_visualization(viz['id'])
                                        st.success(f"Visualization deleted.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting visualization: {str(e)}")
                else:
                    st.info("No saved visualizations found.")
            except Exception as e:
                st.error(f"Error loading visualizations: {str(e)}")
                st.code(str(e))
else:
    # Show message when no data is loaded
    st.info("👈 Please upload a dataset using the sidebar to get started or load the sample data.")
    
    # Show sample features
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Features")
        st.markdown("""
        - 📊 **Comprehensive Data Analysis**: Descriptive statistics, correlation analysis, and more
        - 📈 **Interactive Visualizations**: Create and customize various charts and graphs
        - 💬 **AI-Powered Chat**: Ask questions about your data in natural language using Gemini AI
        - 💾 **Data Management**: Save datasets, analyses, and visualizations for future reference
        - 📤 **Export and Share**: Export results in multiple formats
        """)
    
    with col2:
        st.header("How to Get Started")
        st.markdown("""
        1. Upload your CSV or Excel file using the sidebar
        2. Enter your Gemini API key to use the chat functionality
        3. Explore your data through the different tabs
        4. Save your results to the database for future reference
        
        Don't have data? Click "Load Sample Data" to try the platform with our sample dataset.
        """)
