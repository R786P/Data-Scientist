import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataScienceAgent:
    """100% offline agent - no API key, no card required"""
    
    def __init__(self):
        self.df = None
    
    def query(self, user_query: str) -> str:
        """Process query using pure Python logic (no LLM)"""
        query_lower = user_query.lower().strip()
        
        # ============ LOAD DATA ============
        if "load" in query_lower and ".csv" in query_lower:
            match = re.search(r'[\w\-.]+\.csv', user_query)
            if match:
                filename = match.group()
                try:
                    self.df = pd.read_csv(filename)
                    cols = list(self.df.columns)
                    num_cols = self.df.select_dtypes('number').columns.tolist()
                    cat_cols = self.df.select_dtypes('object').columns.tolist()
                    return (f"✅ Loaded {len(self.df)} rows × {len(cols)} columns\n"
                           f"Numeric: {num_cols[:3]}\n"
                           f"Categorical: {cat_cols[:3]}")
                except Exception as e:
                    return f"❌ Error loading {filename}: {str(e)}"
            return "❌ Please specify CSV filename (e.g., 'load sales.csv')"
        
        # ============ BASIC INFO ============
        if "info" in query_lower or "basic" in query_lower or "shape" in query_lower:
            if self.df is None:
                return "⚠️ Load data first using 'load filename.csv'"
            miss = self.df.isnull().sum()
            miss_str = "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()]) if miss.sum() > 0 else "None"
            return (f"📊 Shape: {self.df.shape}\n"
                   f"Columns: {list(self.df.columns)}\n"
                   f"Missing Values:\n{miss_str}")
        
        # ============ TOP N ANALYSIS ============
        if "top" in query_lower and ("by" in query_lower or "revenue" in query_lower or "sales" in query_lower):
            if self.df is None:
                return "⚠️ Load data first"
            
            # Auto-detect value column
            value_col = None
            for col in self.df.columns:
                if any(x in col.lower() for x in ['revenue', 'sales', 'amount', 'total', 'price']):
                    value_col = col
                    break
            
            if not value_col:
                num_cols = self.df.select_dtypes('number').columns.tolist()
                if not num_cols:
                    return "❌ No numeric columns found for analysis"
                value_col = num_cols[0]
            
            # Auto-detect groupby column
            group_col = None
            for col in self.df.columns:
                if self.df[col].nunique() < min(20, len(self.df)//5) and self.df[col].dtype == 'object':
                    group_col = col
                    break
            
            if not group_col:
                return f"💡 Top values in '{value_col}':\n{self.df.nlargest(5, value_col)[value_col].to_string()}"
            
            # Group by + sum
            result = self.df.groupby(group_col)[value_col].sum().nlargest(5)
            total = self.df[value_col].sum()
            output = f"🏆 Top 5 by {value_col}:\n"
            for idx, (name, val) in enumerate(result.items(), 1):
                pct = (val / total) * 100
                output += f"{idx}. {name}: ₹{val:,.0f} ({pct:.1f}%)\n"
            return output
        
        # ============ TREND PREDICTION (No sklearn) ============
        if "predict" in query_lower or "trend" in query_lower or "forecast" in query_lower:
            if self.df is None:
                return "⚠️ Load data first"
            
            # Find time/sequence column
            time_col = None
            for col in self.df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                    time_col = col
                    break
            
            # Find value column
            value_col = None
            for col in self.df.columns:
                if any(x in col.lower() for x in ['revenue', 'sales', 'amount', 'total']):
                    value_col = col
                    break
            
            if not value_col:
                num_cols = self.df.select_dtypes('number').columns.tolist()
                if not num_cols:
                    return "❌ No numeric columns for prediction"
                value_col = num_cols[0]
            
            series = self.df[value_col].dropna()
            if len(series) < 5:
                return f"⚠️ Need min 5 values in '{value_col}' for prediction"
            
            # Simple trend using last 3 values
            last3 = series.iloc[-3:].values
            trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
            next_val = series.iloc[-1] + (series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else series.iloc[-1]
            
            return (f"📈 Trend for '{value_col}': {trend}\n"
                   f"Last value: ₹{series.iloc[-1]:,.0f}\n"
                   f"Next prediction: ₹{next_val:,.0f}")
        
        # ============ CUSTOMER SEGMENTATION ============
        if "segment" in query_lower or "customer" in query_lower or "group" in query_lower:
            if self.df is None:
                return "⚠️ Load data first"
            
            rev_col = None
            for col in self.df.columns:
                if any(x in col.lower() for x in ['revenue', 'sales', 'amount', 'total']):
                    rev_col = col
                    break
            
            if not rev_col:
                return "❌ Need revenue/sales column for segmentation"
            
            q25 = self.df[rev_col].quantile(0.25)
            q75 = self.df[rev_col].quantile(0.75)
            high = self.df[self.df[rev_col] > q75]
            medium = self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)]
            low = self.df[self.df[rev_col] < q25]
            total = len(self.df)
            
            return (f"👥 Customer Segments:\n"
                   f"• High Value (>₹{q75:,.0f}): {len(high)} ({len(high)/total*100:.0f}%)\n"
                   f"• Medium Value: {len(medium)} ({len(medium)/total*100:.0f}%)\n"
                   f"• Low Value (<₹{q25:,.0f}): {len(low)} ({len(low)/total*100:.0f}%)")
        
        # ============ OUTLIER DETECTION ============
        if "outlier" in query_lower or "anomaly" in query_lower or "unusual" in query_lower:
            if self.df is None:
                return "⚠️ Load data first"
            
            col = None
            for c in self.df.columns:
                if any(x in c.lower() for x in ['price', 'amount', 'revenue', 'sales']):
                    col = c
                    break
            
            if not col:
                num_cols = self.df.select_dtypes('number').columns.tolist()
                if not num_cols:
                    return "❌ No numeric columns for outlier detection"
                col = num_cols[0]
            
            series = self.df[col].dropna()
            if len(series) < 4:
                return f"⚠️ Need min 4 values in '{col}'"
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = series[(series < lower) | (series > upper)]
            
            if len(outliers) == 0:
                return f"✅ No outliers detected in '{col}'"
            
            return (f"⚠️ Outliers in '{col}': {len(outliers)} values ({len(outliers)/len(series)*100:.1f}%)\n"
                   f"Range: ₹{outliers.min():,.0f} to ₹{outliers.max():,.0f}")
        
        # ============ VISUALIZATION ============
        if "plot" in query_lower or "chart" in query_lower or "graph" in query_lower or "visualize" in query_lower:
            if self.df is None:
                return "⚠️ Load data first"
            
            try:
                plt.figure(figsize=(8, 5))
                num_cols = self.df.select_dtypes('number').columns
                cat_cols = self.df.select_dtypes('object').columns
                
                if "bar" in query_lower and len(cat_cols) > 0:
                    col = cat_cols[0]
                    self.df[col].value_counts().head(5).plot(kind='bar')
                    plt.title(f'Top Categories: {col}')
                    plt.xticks(rotation=45, ha='right')
                elif "hist" in query_lower and len(num_cols) > 0:
                    col = num_cols[0]
                    self.df[col].hist(bins=20)
                    plt.title(f'Histogram: {col}')
                else:
                    if len(num_cols) >= 2:
                        self.df.plot.scatter(x=num_cols[0], y=num_cols[1], alpha=0.6)
                        plt.title(f'Scatter: {num_cols[0]} vs {num_cols[1]}')
                    elif len(num_cols) > 0:
                        self.df[num_cols[0]].hist(bins=20)
                        plt.title(f'Histogram: {num_cols[0]}')
                    else:
                        return "⚠️ No suitable columns for plotting"
                
                plt.tight_layout()
                plt.savefig('plot.png', dpi=150)
                plt.close()
                return "✅ Plot saved as 'plot.png' (check Files tab in Render)"
            except Exception as e:
                return f"❌ Plot error: {str(e)}"
        
        # ============ DEFAULT HELP ============
        return ("💡 Supported commands:\n"
               "• 'load sales.csv'\n"
               "• 'show basic info'\n"
               "• 'top 5 by revenue'\n"
               "• 'predict trend'\n"
               "• 'segment customers'\n"
               "• 'detect outliers'\n"
               "• 'create bar chart'")
