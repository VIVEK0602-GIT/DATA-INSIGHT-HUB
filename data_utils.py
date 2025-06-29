import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json
import base64
import io

def load_csv(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        # Read the file
        df = pd.read_csv(uploaded_file)
        
        # Get file info
        file_info = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'shape': df.shape
        }
        
        return df, file_info
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def load_excel(uploaded_file):
    """Load data from uploaded Excel file"""
    try:
        # Read the file
        df = pd.read_excel(uploaded_file)
        
        # Get file info
        file_info = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'shape': df.shape
        }
        
        return df, file_info
    except Exception as e:
        raise Exception(f"Error loading Excel file: {str(e)}")

def get_data_summary(df):
    """Get summary information about the dataframe"""
    if df is None or df.empty:
        return {}
    
    # Basic info
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    summary['column_types'] = {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': date_cols
    }
    
    return summary

def get_descriptive_stats(df):
    """Get descriptive statistics for numeric columns"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    # Get descriptive statistics
    stats = numeric_df.describe().T
    
    # Add additional statistics
    stats['median'] = numeric_df.median()
    stats['skew'] = numeric_df.skew()
    stats['kurtosis'] = numeric_df.kurtosis()
    stats['missing'] = numeric_df.isnull().sum()
    stats['missing_pct'] = (numeric_df.isnull().sum() / len(numeric_df)) * 100
    
    return stats

def get_categorical_stats(df):
    """Get statistics for categorical columns"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Select categorical columns
    cat_df = df.select_dtypes(include=['object', 'category'])
    
    if cat_df.empty:
        return pd.DataFrame()
    
    # Initialize results
    results = []
    
    for col in cat_df.columns:
        # Value counts
        value_counts = df[col].value_counts()
        top_values = {str(k): v for k, v in value_counts.head(5).to_dict().items()}
        
        # Statistics
        stats = {
            'column': col,
            'unique_values': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'top_values': top_values
        }
        
        results.append(stats)
    
    return pd.DataFrame(results)

def get_correlation_matrix(df):
    """Calculate correlation matrix for numeric columns"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    return corr_matrix

def create_visualization(df, chart_type, x_col=None, y_col=None, color_col=None, size_col=None, 
                        title=None, height=600, width=800, **kwargs):
    """Create a visualization based on the specified parameters"""
    if df is None or df.empty:
        return None
    
    # Default title if not provided
    if title is None:
        title = f"{chart_type.title()} Chart"
    
    try:
        # Create different chart types
        if chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, 
                            title=title, height=height, width=width)
        
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                        title=title, height=height, width=width)
        
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                        title=title, height=height, width=width)
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, color=color_col, 
                            title=title, height=height, width=width)
        
        elif chart_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, color=color_col, 
                        title=title, height=height, width=width)
        
        elif chart_type == 'violin':
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, 
                            title=title, height=height, width=width)
        
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_col, values=y_col, title=title, height=height, width=width)
        
        elif chart_type == 'heatmap':
            # For heatmap, we need correlation matrix
            if x_col is None and y_col is None:
                corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='Viridis'
                ))
                fig.update_layout(title=title, height=height, width=width)
            else:
                # If columns are specified, use them to create a crosstab
                heatmap_data = pd.crosstab(df[x_col], df[y_col], normalize='columns')
                fig = px.imshow(heatmap_data, title=title, height=height, width=width)
        
        else:
            return None
        
        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title={'x': 0.5},
            legend_title_text=color_col if color_col else '',
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        raise Exception(f"Error creating visualization: {str(e)}")

def get_chart_config(fig):
    """Get the configuration of a chart for saving"""
    # Get chart data
    chart_data = json.loads(fig.to_json())
    
    return chart_data

def run_pca_analysis(df, n_components=2):
    """Run Principal Component Analysis on numeric data"""
    if df is None or df.empty:
        return None, None, None
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return None, None, None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    
    # Get loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_df.columns
    )
    
    return pca_df, explained_variance, loadings

def run_kmeans_clustering(df, n_clusters=3):
    """Run K-means clustering on numeric data"""
    if df is None or df.empty:
        return None, None
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return None, None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the original dataframe
    cluster_df = df.copy()
    cluster_df['cluster'] = clusters
    
    # Get cluster centers
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=numeric_df.columns
    )
    centers.index.name = 'cluster'
    
    return cluster_df, centers

def create_sample_data():
    """Create a sample dataset for demo purposes"""
    # Create sample data
    np.random.seed(42)
    
    # Sample size
    n = 100
    
    # Generate dates for the last 100 days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n)
    
    # Sample data for sales analysis
    data = {
        'date': dates,
        'product_id': np.random.randint(1, 6, size=n),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Food'], size=n),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=n),
        'price': np.random.uniform(10, 100, size=n).round(2),
        'quantity': np.random.randint(1, 10, size=n),
        'customer_age': np.random.randint(18, 70, size=n),
        'rating': np.random.uniform(1, 5, size=n).round(1),
    }
    
    # Calculate sales
    data['sales'] = (data['price'] * data['quantity']).round(2)
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Map product_id to product_name
    product_map = {
        1: 'Laptop',
        2: 'Smartphone',
        3: 'Headphones',
        4: 'Tablet',
        5: 'Smartwatch'
    }
    
    df['product_name'] = df['product_id'].map(product_map)
    
    return df

def export_dataframe(df, format='csv'):
    """Export dataframe to specified format and return as download link"""
    if df is None or df.empty:
        return None
    
    # Create a buffer to store the file
    buffer = io.BytesIO()
    
    # Export to specified format
    if format.lower() == 'csv':
        df.to_csv(buffer, index=False)
        mime_type = 'text/csv'
        file_ext = 'csv'
    elif format.lower() == 'excel':
        df.to_excel(buffer, index=False)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_ext = 'xlsx'
    elif format.lower() == 'json':
        df.to_json(buffer, orient='records')
        mime_type = 'application/json'
        file_ext = 'json'
    else:
        return None
    
    # Get the value from the buffer
    buffer.seek(0)
    data = buffer.getvalue()
    
    # Encode the data
    b64 = base64.b64encode(data).decode()
    
    # Create the download link
    href = f'data:{mime_type};base64,{b64}'
    
    return href

def evaluate_model_performance(df, target_col, feature_cols, model_type='classification', test_size=0.2, random_state=42):
    """
    Evaluate model performance using various metrics
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    target_col : str
        The name of the target column
    feature_cols : list
        List of feature column names
    model_type : str
        Type of model: 'classification' or 'regression'
    test_size : float
        Size of the test set (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics and visualization data
    """
    if df is None or df.empty or target_col not in df.columns:
        return {
            'success': False,
            'error': 'Invalid data or target column not found'
        }
    
    # Check if all feature columns exist in dataframe
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        return {
            'success': False,
            'error': f'Feature columns not found: {missing_cols}'
        }
    
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Handle datetime values in target column
        if pd.api.types.is_datetime64_any_dtype(df_copy[target_col]):
            if model_type == 'classification':
                df_copy[target_col] = df_copy[target_col].astype(str)
            else:
                df_copy[target_col] = df_copy[target_col].map(lambda x: x.toordinal() if hasattr(x, 'toordinal') else x)

        # Prepare data
        X = df_copy[feature_cols]
        y = df_copy[target_col]

        # Encode all categorical features at once
        X = pd.get_dummies(X, drop_first=True)

        # Handle datetime values in features
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].map(lambda x: x.toordinal() if hasattr(x, 'toordinal') else x)

        # Encode target if classification and not numeric
        label_mapping = None
        if model_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        result = {
            'success': True,
            'model_type': model_type,
            'target_column': target_col,
            'feature_columns': list(X.columns),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        if label_mapping:
            result['label_mapping'] = label_mapping

        # Classification models
        if model_type == 'classification':
            # Train logistic regression model
            log_reg = LogisticRegression(random_state=random_state, max_iter=1000)
            log_reg.fit(X_train, y_train)
            log_reg_pred = log_reg.predict(X_test)
            log_reg_prob = log_reg.predict_proba(X_test)

            # Train random forest model
            rf = RandomForestClassifier(random_state=random_state)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_prob = rf.predict_proba(X_test)

            # Calculate metrics
            log_reg_metrics = {
                'accuracy': accuracy_score(y_test, log_reg_pred),
                'precision': precision_score(y_test, log_reg_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, log_reg_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, log_reg_pred, average='weighted', zero_division=0)
            }
            rf_metrics = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0)
            }

            # Confusion matrix
            log_reg_cm = confusion_matrix(y_test, log_reg_pred)
            rf_cm = confusion_matrix(y_test, rf_pred)

            # Feature importance (for random forest)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            # Predictions vs actuals (for display)
            pred_vs_actual = pd.DataFrame({
                'actual': y_test,
                'logistic_regression': log_reg_pred,
                'random_forest': rf_pred
            })
            if label_mapping:
                inv_map = {v: k for k, v in label_mapping.items()}
                pred_vs_actual = pred_vs_actual.applymap(lambda x: inv_map.get(x, x))

            # Add results to dictionary
            result['logistic_regression'] = log_reg_metrics
            result['random_forest'] = rf_metrics
            result['confusion_matrix'] = {
                'logistic_regression': log_reg_cm.tolist(),
                'random_forest': rf_cm.tolist()
            }
            result['feature_importance'] = feature_importance.to_dict(orient='records')
            result['predictions'] = pred_vs_actual.head(100).to_dict(orient='records')

            # Try to compute ROC curve (only for binary classification)
            if len(np.unique(y_test)) == 2:
                fpr_lr, tpr_lr, _ = roc_curve(y_test, log_reg_prob[:, 1])
                fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob[:, 1])
                result['roc_curve'] = {
                    'logistic_regression': {
                        'fpr': fpr_lr.tolist(),
                        'tpr': tpr_lr.tolist(),
                        'auc': auc(fpr_lr, tpr_lr)
                    },
                    'random_forest': {
                        'fpr': fpr_rf.tolist(),
                        'tpr': tpr_rf.tolist(),
                        'auc': auc(fpr_rf, tpr_rf)
                    }
                }

        # Regression models
        elif model_type == 'regression':
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
            lin_reg_pred = lin_reg.predict(X_test)
            rf_reg = RandomForestRegressor(random_state=random_state)
            rf_reg.fit(X_train, y_train)
            rf_reg_pred = rf_reg.predict(X_test)

            lin_reg_metrics = {
                'mse': mean_squared_error(y_test, lin_reg_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lin_reg_pred)),
                'mae': mean_absolute_error(y_test, lin_reg_pred),
                'r2': r2_score(y_test, lin_reg_pred)
            }
            rf_reg_metrics = {
                'mse': mean_squared_error(y_test, rf_reg_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_reg_pred)),
                'mae': mean_absolute_error(y_test, rf_reg_pred),
                'r2': r2_score(y_test, rf_reg_pred)
            }

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_reg.feature_importances_
            }).sort_values('importance', ascending=False)

            prediction_data = pd.DataFrame({
                'actual': y_test,
                'linear_regression': lin_reg_pred,
                'random_forest': rf_reg_pred
            })

            result['linear_regression'] = lin_reg_metrics
            result['random_forest'] = rf_reg_metrics
            result['feature_importance'] = feature_importance.to_dict(orient='records')
            result['predictions'] = prediction_data.head(100).to_dict(orient='records')

        return result

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_performance_visualization(performance_data, chart_type='metrics_comparison'):
    """
    Create visualizations for model performance metrics
    
    Parameters:
    -----------
    performance_data : dict
        The performance metrics data returned by evaluate_model_performance()
    chart_type : str
        Type of chart to create: 'metrics_comparison', 'confusion_matrix', 
        'feature_importance', 'roc_curve', 'prediction_vs_actual'
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A plotly figure object
    """
    if not performance_data['success']:
        return None
    
    model_type = performance_data['model_type']
    
    try:
        # Metrics comparison chart
        if chart_type == 'metrics_comparison':
            # Different metrics based on model type
            if model_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                log_reg_metrics = [performance_data['logistic_regression'][m] for m in metrics]
                rf_metrics = [performance_data['random_forest'][m] for m in metrics]
                
                # Create bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=log_reg_metrics,
                    name='Logistic Regression',
                    marker_color='#2a9df4',
                    text=[f'{val:.3f}' for val in log_reg_metrics],
                    textposition='auto',
                    textfont=dict(color='white', size=14)
                ))
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=rf_metrics,
                    name='Random Forest',
                    marker_color='#ff6361',
                    text=[f'{val:.3f}' for val in rf_metrics],
                    textposition='auto',
                    textfont=dict(color='white', size=14)
                ))
                
                fig.update_layout(
                    title={
                        'text': 'Classification Metrics Comparison',
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 22, 'color': 'white'}
                    },
                    xaxis_title={
                        'text': 'Metric',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    yaxis_title={
                        'text': 'Value',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    yaxis=dict(range=[0, 1], tickfont=dict(size=14, color='white')),
                    xaxis=dict(tickfont=dict(size=14, color='white')),
                    legend_title={
                        'text': 'Model',
                        'font': {'size': 14, 'color': 'white'}
                    },
                    legend=dict(
                        font=dict(size=14, color='white'),
                        bgcolor='rgba(30, 30, 45, 0.5)',
                        bordercolor='rgba(255, 255, 255, 0.2)'
                    ),
                    paper_bgcolor='rgba(30, 30, 45, 1)',
                    plot_bgcolor='rgba(30, 30, 45, 1)',
                    template='plotly_dark',
                    margin=dict(t=100, b=80, l=80, r=40)
                )
            
            elif model_type == 'regression':
                # For regression, we'll create two separate charts for better scaling
                # Chart 1: MSE, RMSE, MAE
                error_metrics = ['mse', 'rmse', 'mae']
                lin_error_vals = [performance_data['linear_regression'][m] for m in error_metrics]
                rf_error_vals = [performance_data['random_forest'][m] for m in error_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=error_metrics,
                    y=lin_error_vals,
                    name='Linear Regression',
                    marker_color='#2a9df4',
                    text=[f'{val:.3f}' for val in lin_error_vals],
                    textposition='auto',
                    textfont=dict(color='white', size=14)
                ))
                fig.add_trace(go.Bar(
                    x=error_metrics,
                    y=rf_error_vals,
                    name='Random Forest',
                    marker_color='#ff6361',
                    text=[f'{val:.3f}' for val in rf_error_vals],
                    textposition='auto',
                    textfont=dict(color='white', size=14)
                ))
                
                fig.update_layout(
                    title={
                        'text': 'Regression Error Metrics Comparison',
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 22, 'color': 'white'}
                    },
                    xaxis_title={
                        'text': 'Metric',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    yaxis_title={
                        'text': 'Value',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    legend_title={
                        'text': 'Model',
                        'font': {'size': 14, 'color': 'white'}
                    },
                    legend=dict(
                        font=dict(size=14, color='white'),
                        bgcolor='rgba(30, 30, 45, 0.5)',
                        bordercolor='rgba(255, 255, 255, 0.2)'
                    ),
                    xaxis=dict(tickfont=dict(size=14, color='white')),
                    yaxis=dict(tickfont=dict(size=14, color='white')),
                    paper_bgcolor='rgba(30, 30, 45, 1)',
                    plot_bgcolor='rgba(30, 30, 45, 1)',
                    template='plotly_dark',
                    margin=dict(t=100, b=80, l=80, r=40)
                )
        
        # Confusion matrix
        elif chart_type == 'confusion_matrix' and model_type == 'classification':
            # Get confusion matrix data for random forest (better model usually)
            cm = np.array(performance_data['confusion_matrix']['random_forest'])
            
            # Create heatmap
            labels = list(performance_data.get('label_mapping', {}).keys())
            if not labels:
                labels = [str(i) for i in range(cm.shape[0])]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"}
            ))
            
            fig.update_layout(
                title={
                    'text': 'Confusion Matrix (Random Forest)',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 22, 'color': 'white'}
                },
                xaxis_title={
                    'text': "Predicted Label",
                    'font': {'size': 16, 'color': 'white'}
                },
                yaxis_title={
                    'text': "True Label",
                    'font': {'size': 16, 'color': 'white'}
                },
                xaxis=dict(tickfont=dict(size=14, color='white')),
                yaxis=dict(tickfont=dict(size=14, color='white')),
                paper_bgcolor='rgba(30, 30, 45, 1)',
                plot_bgcolor='rgba(30, 30, 45, 1)',
                template='plotly_dark',
                margin=dict(t=100, b=80, l=80, r=40)
            )
        
        # Feature importance
        elif chart_type == 'feature_importance':
            # Sort feature importance
            feature_importance = sorted(
                performance_data['feature_importance'], 
                key=lambda x: x['importance'],
                reverse=True
            )[:10]  # Top 10 features
            
            features = [item['feature'] for item in feature_importance]
            importances = [item['importance'] for item in feature_importance]
            
            fig = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='#3366CC',
                text=[f'{val:.4f}' for val in importances],
                textposition='auto',
                textfont=dict(color='white', size=14)
            ))
            
            fig.update_layout(
                title={
                    'text': 'Feature Importance (Random Forest)',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 22, 'color': 'white'}
                },
                xaxis_title={
                    'text': "Importance",
                    'font': {'size': 16, 'color': 'white'}
                },
                yaxis_title={
                    'text': "Feature",
                    'font': {'size': 16, 'color': 'white'}
                },
                yaxis={'categoryorder': 'total ascending', 'tickfont': dict(size=14, color='white')},
                xaxis={'tickfont': dict(size=14, color='white')},
                paper_bgcolor='rgba(30, 30, 45, 1)',
                plot_bgcolor='rgba(30, 30, 45, 1)',
                template='plotly_dark',
                margin=dict(t=100, b=80, l=120, r=40)  # Extra left margin for feature names
            )
        
        # ROC curve (only for binary classification)
        elif chart_type == 'roc_curve' and model_type == 'classification' and 'roc_curve' in performance_data:
            fig = go.Figure()
            
            # Add diagonal reference line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray', width=2),
                name='Random Classifier'
            ))
            
            # Add ROC curves
            lr_data = performance_data['roc_curve']['logistic_regression']
            rf_data = performance_data['roc_curve']['random_forest']
            
            fig.add_trace(go.Scatter(
                x=lr_data['fpr'],
                y=lr_data['tpr'],
                mode='lines',
                name=f'Logistic Regression (AUC = {lr_data["auc"]:.3f})',
                line=dict(color='#2a9df4', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=rf_data['fpr'],
                y=rf_data['tpr'],
                mode='lines',
                name=f'Random Forest (AUC = {rf_data["auc"]:.3f})',
                line=dict(color='#ff6361', width=3)
            ))
            
            fig.update_layout(
                title={
                    'text': 'Receiver Operating Characteristic (ROC) Curve',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 22, 'color': 'white'}
                },
                xaxis_title={
                    'text': "False Positive Rate",
                    'font': {'size': 16, 'color': 'white'}
                },
                yaxis_title={
                    'text': "True Positive Rate",
                    'font': {'size': 16, 'color': 'white'}
                },
                xaxis=dict(tickfont=dict(size=14, color='white')),
                yaxis=dict(tickfont=dict(size=14, color='white')),
                legend=dict(
                    font=dict(size=14, color='white'),
                    bgcolor='rgba(30, 30, 45, 0.5)',
                    bordercolor='rgba(255, 255, 255, 0.2)',
                    x=0.01, 
                    y=0.99
                ),
                paper_bgcolor='rgba(30, 30, 45, 1)',
                plot_bgcolor='rgba(30, 30, 45, 1)',
                template='plotly_dark',
                margin=dict(t=100, b=80, l=80, r=40)
            )
        
        # Predicted vs actual (for regression)
        elif chart_type == 'prediction_vs_actual' and model_type == 'regression':
            # Extract data
            actual = [item['actual'] for item in performance_data['predictions']]
            lr_pred = [item['linear_regression'] for item in performance_data['predictions']]
            rf_pred = [item['random_forest'] for item in performance_data['predictions']]
            
            fig = go.Figure()
            
            # Add ideal line (y=x)
            min_val = min(min(actual), min(lr_pred), min(rf_pred))
            max_val = max(max(actual), max(lr_pred), max(rf_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray', width=2),
                name='Ideal Prediction'
            ))
            
            # Add actual vs predicted points
            fig.add_trace(go.Scatter(
                x=actual,
                y=lr_pred,
                mode='markers',
                name='Linear Regression',
                marker=dict(color='#2a9df4', size=10, opacity=0.7, line=dict(color='white', width=1))
            ))
            
            fig.add_trace(go.Scatter(
                x=actual,
                y=rf_pred,
                mode='markers',
                name='Random Forest',
                marker=dict(color='#ff6361', size=10, opacity=0.7, line=dict(color='white', width=1))
            ))
            
            fig.update_layout(
                title={
                    'text': 'Predicted vs Actual Values',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 22, 'color': 'white'}
                },
                xaxis_title={
                    'text': "Actual Values",
                    'font': {'size': 16, 'color': 'white'}
                },
                yaxis_title={
                    'text': "Predicted Values",
                    'font': {'size': 16, 'color': 'white'}
                },
                xaxis=dict(tickfont=dict(size=14, color='white')),
                yaxis=dict(tickfont=dict(size=14, color='white')),
                legend=dict(
                    font=dict(size=14, color='white'),
                    bgcolor='rgba(30, 30, 45, 0.5)',
                    bordercolor='rgba(255, 255, 255, 0.2)'
                ),
                paper_bgcolor='rgba(30, 30, 45, 1)',
                plot_bgcolor='rgba(30, 30, 45, 1)',
                template='plotly_dark',
                margin=dict(t=100, b=80, l=80, r=40)
            )
        
        else:
            return None
        
        return fig
    
    except Exception as e:
        print(f"Error creating performance visualization: {str(e)}")
        return None 