"""
Automated preprocessing pipelines - 100% offline
Handles missing values, scaling, and outlier removal automatically
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DataPipeline:
    """One-click data preparation and cleaning"""
    
    def auto_clean(self, df):
        """Full cleaning: Remove duplicates, fill missing, handle outliers"""
        if df is None: return None, "❌ No data provided"
        
        # 1. Remove Duplicates
        before = len(df)
        df = df.drop_duplicates()
        dup_removed = before - len(df)
        
        # 2. Fill Missing Values
        num_cols = df.select_dtypes('number').columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
            
        cat_cols = df.select_dtypes('object').columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
        # 3. Basic Outlier Removal (IQR Method)
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            
        return df, f"✅ Auto-Cleaned: {dup_removed} dups removed, missing filled, outliers handled ({len(df)} rows left)."

    def get_scaler_pipeline(self):
        """Returns a scaling pipeline for ML models"""
        return Pipeline([
            ('scaler', StandardScaler())
        ])
