import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
import streamlit as st

# Use SQLite for simplicity and portability
DB_PATH = "data/datainsighthub.db"

def ensure_db_exists():
    """Ensure the database directory and file exist"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_path TEXT,
        column_info TEXT,
        row_count INTEGER,
        file_size INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        analysis_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result_data TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS visualizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        title TEXT NOT NULL,
        chart_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_connection():
    """Get a connection to the SQLite database"""
    return sqlite3.connect(DB_PATH)

def save_dataset(name, description, df, file_path=None, file_size=None):
    """Save dataset metadata to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    column_info = json.dumps({
        'columns': df.columns.tolist(),
        'dtypes': {col: str(df[col].dtype) for col in df.columns}
    })
    
    cursor.execute('''
    INSERT INTO datasets (name, description, file_path, column_info, row_count, file_size)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, description, file_path, column_info, len(df), file_size))
    
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return dataset_id

def get_datasets():
    """Get list of all saved datasets"""
    conn = get_connection()
    df = pd.read_sql('SELECT * FROM datasets ORDER BY created_at DESC', conn)
    conn.close()
    return df

def get_dataset(dataset_id):
    """Get dataset by ID"""
    conn = get_connection()
    df = pd.read_sql('SELECT * FROM datasets WHERE id = ?', conn, params=(dataset_id,))
    conn.close()
    if len(df) > 0:
        return df.iloc[0]
    return None

def save_analysis_result(dataset_id, analysis_type, result_data):
    """Save analysis result to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if isinstance(result_data, (pd.DataFrame, dict, list)):
        result_data = json.dumps(result_data, default=str)
    
    cursor.execute('''
    INSERT INTO analysis_results (dataset_id, analysis_type, result_data)
    VALUES (?, ?, ?)
    ''', (dataset_id, analysis_type, result_data))
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return result_id

def get_analysis_results(dataset_id=None):
    """Get analysis results, optionally filtered by dataset_id"""
    conn = get_connection()
    
    if dataset_id:
        df = pd.read_sql(
            'SELECT * FROM analysis_results WHERE dataset_id = ? ORDER BY created_at DESC', 
            conn, params=(dataset_id,)
        )
    else:
        df = pd.read_sql('SELECT * FROM analysis_results ORDER BY created_at DESC', conn)
    
    conn.close()
    return df

def save_visualization(dataset_id, title, chart_type, config):
    """Save visualization to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if isinstance(config, dict):
        config = json.dumps(config)
    
    cursor.execute('''
    INSERT INTO visualizations (dataset_id, title, chart_type, config)
    VALUES (?, ?, ?, ?)
    ''', (dataset_id, title, chart_type, config))
    
    vis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return vis_id

def get_visualizations(dataset_id=None):
    """Get visualizations, optionally filtered by dataset_id"""
    conn = get_connection()
    
    if dataset_id:
        df = pd.read_sql(
            'SELECT * FROM visualizations WHERE dataset_id = ? ORDER BY created_at DESC', 
            conn, params=(dataset_id,)
        )
    else:
        df = pd.read_sql('SELECT * FROM visualizations ORDER BY created_at DESC', conn)
    
    conn.close()
    return df

# Initialize database when module is imported
ensure_db_exists() 