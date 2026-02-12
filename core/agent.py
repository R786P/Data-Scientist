"""
Data Science Agent - Pure Python (No API Keys Required)
Modular core agent for EDA, visualization, and lightweight ML
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

class DataScienceAgent:
    """Offline data science agent - no internet/API required"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.last_loaded_file: Optional[str] = None
    
    # ============ DATA LOADING ============
    def load_data(self, file_path: str) -> str:
        """Load CSV/Excel file into memory"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                return f"❌ Unsupported format: {file_path}. Use CSV/Excel."
            
            self.last_loaded_file = file_path
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            return (f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\n"
                   f"Numeric columns: {num_cols[:3]}\n"
                   f"Categorical columns: {cat_cols[:3]}")
        except Exception as e:
            return f"❌ Error loading {file_path}: {str(e)}"
    
    # ============ BASIC STATISTICS ============
    def show_basic_info(self) -> str:
        """Show dataset shape, columns, and missing values"""
        if self.df is None:
            return "⚠️ No data loaded. Use 'load filename.csv' first."
        
        miss = self.df.isnull().sum()
        miss_str = "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()]) if miss.sum() > 0 else "None"
        
        return (f"📊 Shape: {self.df.shape}\n"
               f"Columns: {list(self.df.columns)}\n"
               f"Missing Values:\n{miss_str}")
    
    # ============ VISUALIZATION ============
    def generate_visualization(self, plot_type: str) -> str:
        """Create basic plots (histogram, bar, scatter)"""
        if self.df is None:
            return "⚠️ No data loaded. Use 'load filename.csv' first."
        
        try:
            plt.figure(figsize=(8, 5))
            
            if "hist" in plot_type.lower() or "dist" in plot_type.lower():
                num_cols = self.df.select_dtypes('number').columns
                if len(num_cols) == 0:
                    return "⚠️ No numeric columns for histogram."
                col = num_cols[0]
                self.df[col].hist(bins=20, edgecolor='black')
                plt.title(f'Histogram: {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            elif "bar" in plot_type.lower() or "count" in plot_type.lower():
                cat_cols = self.df.select_dtypes('object').columns
                if len(cat_cols) == 0:
                    return "⚠️ No categorical columns for bar chart."
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(5)
                counts.plot(kind='bar', color='#667eea')
                plt.title(f'Top 5: {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
            
            elif "scatter" in plot_type.lower() or "relation" in plot_type.lower():
                num_cols = self.df.select_dtypes('number').columns
                if len(num_cols) < 2:
                    return "⚠️ Need at least 2 numeric columns for scatter plot."
                sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], alpha=0.6)
                plt.title(f'{num_cols[0]} vs {num_cols[1]}')
                plt.xlabel(num_cols[0])
                plt.ylabel(num_cols[1])
            
            else:
                return "⚠️ Unsupported plot type. Try: 'histogram', 'bar', or 'scatter'"
            
            plt.tight_layout()
            plt.savefig('plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            return "✅ Plot saved as 'plot.png' (check Files tab)"
        
        except Exception as e:
            return f"❌ Plot error: {str(e)}"
    
    # ============ TOP N ANALYSIS ============
    def top_n_analysis(self, n: int = 5, metric: str = "revenue") -> str:
        """Find top N items by a metric (revenue/sales/price)"""
        if self.df is None:
            return "⚠️ No data loaded."
        
        # Auto-detect metric column
        metric_col = None
        for col in self.df.columns:
            if metric.lower() in col.lower():
                metric_col = col
                break
        
        if not metric_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns found."
            metric_col = num_cols[0]
        
        # Auto-detect grouping column (categorical with <20 unique values)
        group_col = None
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and 2 <= self.df[col].nunique() <= 20:
                group_col = col
                break
        
        if not group_col:
            # Fallback: just show top N rows
            result = self.df.nlargest(n, metric_col)[[metric_col]]
            output = f"🏆 Top {n} by {metric_col}:\n"
            for idx, (_, row) in enumerate(result.iterrows(), 1):
                output += f"{idx}. ₹{row[metric_col]:,.0f}\n"
            return output
        
        # Group by + sum
        grouped = self.df.groupby(group_col)[metric_col].sum().nlargest(n)
        total = self.df[metric_col].sum()
        
        output = f"🏆 Top {n} by {metric_col}:\n"
        for idx, (name, val) in enumerate(grouped.items(), 1):
            pct = (val / total) * 100
            output += f"{idx}. {name}: ₹{val:,.0f} ({pct:.1f}%)\n"
        return output
    
    # ============ TREND PREDICTION (No sklearn) ============
    def predict_trend(self, column: str = "revenue") -> str:
        """Simple trend
