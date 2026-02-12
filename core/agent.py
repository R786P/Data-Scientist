import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.last_loaded_file = None
    
    def load_data(self, file_path):
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
            return f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\nNumeric: {num_cols[:3]}\nCategorical: {cat_cols[:3]}"
        except Exception as e:
            return f"❌ Error loading {file_path}: {str(e)}"
    
    def show_basic_info(self):
        if self.df is None:
            return "⚠️ No data loaded. Use 'load filename.csv' first."
        miss = self.df.isnull().sum()
        miss_str = "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()]) if miss.sum() > 0 else "None"
        return f"📊 Shape: {self.df.shape}\nColumns: {list(self.df.columns)}\nMissing Values:\n{miss_str}"
    
    def generate_visualization(self, plot_type):
        if self.df is None:
            return "⚠️ No data loaded. Use 'load filename.csv' first."
        try:
            plt.figure(figsize=(8, 5))
            if "hist" in plot_type.lower():
                num_cols = self.df.select_dtypes('number').columns
                if len(num_cols) == 0:
                    return "⚠️ No numeric columns for histogram."
                col = num_cols[0]
                self.df[col].hist(bins=20, edgecolor='black')
                plt.title(f'Histogram: {col}')
            elif "bar" in plot_type.lower():
                cat_cols = self.df.select_dtypes('object').columns
                if len(cat_cols) == 0:
                    return "⚠️ No categorical columns for bar chart."
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(5)
                counts.plot(kind='bar', color='#667eea')
                plt.title(f'Top 5: {col}')
                plt.xticks(rotation=45, ha='right')
            elif "scatter" in plot_type.lower():
                num_cols = self.df.select_dtypes('number').columns
                if len(num_cols) < 2:
                    return "⚠️ Need at least 2 numeric columns for scatter plot."
                sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], alpha=0.6)
                plt.title(f'{num_cols[0]} vs {num_cols[1]}')
            else:
                return "⚠️ Unsupported plot type. Try: 'histogram', 'bar', or 'scatter'"
            plt.tight_layout()
            plt.savefig('plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            return "✅ Plot saved as 'plot.png'"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"
    
    def top_n_analysis(self, n=5, metric="revenue"):
        if self.df is None:
            return "⚠️ No data loaded."
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
        group_col = None
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and 2 <= self.df[col].nunique() <= 20:
                group_col = col
                break
        if not group_col:
            result = self.df.nlargest(n, metric_col)[[metric_col]]
            output = f"🏆 Top {n} by {metric_col}:\n"
            for idx, (_, row) in enumerate(result.iterrows(), 1):
                output += f"{idx}. ₹{row[metric_col]:,.0f}\n"
            return output
        grouped = self.df.groupby(group_col)[metric_col].sum().nlargest(n)
        total = self.df[metric_col].sum()
        output = f"🏆 Top {n} by {metric_col}:\n"
        for idx, (name, val) in enumerate(grouped.items(), 1):
            pct = (val / total) * 100
            output += f"{idx}. {name}: ₹{val:,.0f} ({pct:.1f}%)\n"
        return output
    
    def predict_trend(self, column="revenue"):
        if self.df is None:
            return "⚠️ No data loaded."
        target_col = None
        for col in self.df.columns:
            if column.lower() in col.lower():
                target_col = col
                break
        if not target_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns for prediction."
            target_col = num_cols[0]
        series = self.df[target_col].dropna()
        if len(series) < 5:
            return f"⚠️ Need min 5 values in '{target_col}' for prediction."
        last3 = series.iloc[-3:].values
        trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
        next_val = series.iloc[-1] + (series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else series.iloc[-1]
        return f"📈 Trend for '{target_col}': {trend}\nLast: ₹{series.iloc[-1]:,.0f}\nNext: ₹{next_val:,.0f}"
    
    def segment_customers(self):
        if self.df is None:
            return "⚠️ No data loaded."
        rev_col = None
        for col in self.df.columns:
            if any(x in col.lower() for x in ['revenue', 'sales', 'amount', 'total', 'price']):
                rev_col = col
                break
        if not rev_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ Need a numeric column for segmentation."
            rev_col = num_cols[0]
        q25 = self.df[rev_col].quantile(0.25)
        q75 = self.df[rev_col].quantile(0.75)
        high = self.df[self.df[rev_col] > q75]
        medium = self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)]
        low = self.df[self.df[rev_col] < q25]
        total = len(self.df)
        return f"👥 Segments:\nHigh (>₹{q75:,.0f}): {len(high)} ({len(high)/total*100:.0f}%)\nMedium: {len(medium)} ({len(medium)/total*100:.0f}%)\nLow (<₹{q25:,.0f}): {len(low)} ({len(low)/total*100:.0f}%)"
    
    def detect_outliers(self, column="price"):
        if self.df is None:
            return "⚠️ No data loaded."
        target_col = None
        for col in self.df.columns:
            if column.lower() in col.lower():
                target_col = col
                break
        if not target_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns for outlier detection."
            target_col = num_cols[0]
        series = self.df[target_col].dropna()
        if len(series) < 4:
            return f"⚠️ Need min 4 values in '{target_col}'."
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            return f"✅ No outliers in '{target_col}'."
        return f"⚠️ Outliers in '{target_col}': {len(outliers)} values\nRange: ₹{outliers.min():,.0f} to ₹{outliers.max():,.0f}"
    
    def query(self, user_query):
        query_lower = user_query.lower().strip()
        if "load" in query_lower and ".csv" in query_lower:
            match = re.search(r'[\w\-.]+\.csv', user_query)
            if match:
                return self.load_data(match.group())
            return "❌ Please specify CSV filename (e.g., 'load sales.csv')"
        if "info" in query_lower or "basic" in query_lower or "shape" in query_lower:
            return self.show_basic_info()
        if "top" in query_lower and ("by" in query_lower or "revenue" in query_lower or "sales" in query_lower):
            n_match = re.search(r'top\s+(\d+)', query_lower)
            n = int(n_match.group(1)) if n_match else 5
            metric = "revenue"
            for word in ["revenue", "sales", "price", "amount", "quantity"]:
                if word in query_lower:
                    metric = word
                    break
            return self.top_n_analysis(n=n, metric=metric)
        if "predict" in query_lower or "trend" in query_lower:
            col = "revenue"
            for word in ["revenue", "sales", "price", "amount"]:
                if word in query_lower:
                    col = word
                    break
            return self.predict_trend(column=col)
        if "segment" in query_lower or "customer" in query_lower:
            return self.segment_customers()
        if "outlier" in query_lower or "anomaly" in query_lower:
            col = "price"
            for word in ["price", "revenue", "sales", "amount"]:
                if word in query_lower:
                    col = word
                    break
            return self.detect_outliers(column=col)
        if "plot" in query_lower or "chart" in query_lower or "graph" in query_lower:
            return self.generate_visualization(plot_type=query_lower)
        return "💡 Commands:\n• 'load sales.csv'\n• 'top 5 by revenue'\n• 'predict trend'\n• 'segment customers'\n• 'detect outliers'\n• 'create bar chart'"
