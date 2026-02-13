import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.last_file = None
    
    def load_data(self, fp):
        try:
            self.df = pd.read_csv(fp)
            self.last_file = fp
            num = self.df.select_dtypes('number').columns.tolist()
            cat = self.df.select_dtypes('object').columns.tolist()
            return f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\nNumeric: {num[:3]}\nCategorical: {cat[:3]}"
        except Exception as e:
            return f"❌ Error loading {fp}: {str(e)}"
    
    def show_info(self):
        if self.df is None: return "⚠️ Load data first (upload CSV)"
        miss = self.df.isnull().sum()
        miss_str = "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()]) if miss.sum() > 0 else "None"
        return f"📊 Shape: {self.df.shape}\nColumns: {list(self.df.columns)}\nMissing:\n{miss_str}"
    
    def show_missing(self):
        if self.df is None: return "⚠️ Load data first"
        miss = self.df.isnull().sum()
        if miss.sum() == 0: return "✅ No missing values"
        return "⚠️ Missing values:\n" + "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()])
    
    def clean_data(self):
        if self.df is None: return "⚠️ Load data first"
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        return f"🧹 Cleaned: {before-after} rows removed ({after} remaining)"
    
    def fill_missing(self):
        if self.df is None: return "⚠️ Load data first"
        num_cols = self.df.select_dtypes('number').columns
        for col in num_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        return f"✅ Filled missing values with mean for {len(num_cols)} numeric columns"
    
    def remove_duplicates(self):
        if self.df is None: return "⚠️ Load data first"
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        return f"✅ Removed {before-after} duplicates ({after} rows remaining)"
    
    def top_n(self, n=5, metric="revenue"):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if metric.lower() in c.lower()), None)
        if not col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ No numeric columns"
            col = num_cols[0]
        if col not in self.df.columns: return f"❌ Column '{col}' not found"
        result = self.df.nlargest(n, col)[[col]]
        total = self.df[col].sum()
        out = f"🏆 Top {n} by {col}:\n"
        for i, (_, row) in enumerate(result.iterrows(), 1):
            pct = (row[col] / total) * 100 if total != 0 else 0
            out += f"{i}. ₹{row[col]:,.0f} ({pct:.1f}%)\n"
        return out
    
    def group_by(self, col_name):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if col_name.lower() in c.lower()), None)
        if not col or col not in self.df.columns:
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            if not cat_cols: return "❌ No categorical columns for grouping"
            col = cat_cols[0]
        num_col = next((c for c in self.df.select_dtypes('number').columns if 'revenue' in c.lower() or 'sales' in c.lower() or 'amount' in c.lower()), self.df.select_dtypes('number').columns[0])
        grouped = self.df.groupby(col)[num_col].sum().nlargest(10)
        total = self.df[num_col].sum()
        out = f"📊 {col} vs {num_col}:\n"
        for name, val in grouped.items():
            pct = (val / total) * 100
            out += f"• {name}: ₹{val:,.0f} ({pct:.1f}%)\n"
        return out
    
    def predict_trend(self, col_name="revenue"):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if col_name.lower() in c.lower()), None)
        if not col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ No numeric columns"
            col = num_cols[0]
        series = self.df[col].dropna()
        if len(series) < 5: return f"⚠️ Need min 5 values in '{col}'"
        last3 = series.iloc[-3:].values
        trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
        next_val = series.iloc[-1] + (series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else series.iloc[-1]
        return f"📈 '{col}': {trend}\nLast: ₹{series.iloc[-1]:,.0f}\nNext: ₹{next_val:,.0f}"
    
    def segment_customers(self):
        if self.df is None: return "⚠️ Load data first"
        rev_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total','price'])), None)
        if not rev_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ Need numeric column"
            rev_col = num_cols[0]
        q25 = self.df[rev_col].quantile(0.25)
        q75 = self.df[rev_col].quantile(0.75)
        high = self.df[self.df[rev_col] > q75]
        medium = self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)]
        low = self.df[self.df[rev_col] < q25]
        total = len(self.df)
        return f"👥 Segments:\nHigh (>₹{q75:,.0f}): {len(high)} ({len(high)/total*100:.0f}%)\nMedium: {len(medium)} ({len(medium)/total*100:.0f}%)\nLow (<₹{q25:,.0f}): {len(low)} ({len(low)/total*100:.0f}%)"
    
    def detect_outliers(self, col_name="revenue"):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if col_name.lower() in c.lower()), None)
        if not col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ No numeric columns"
            col = num_cols[0]
        series = self.df[col].dropna()
        if len(series) < 4: return f"⚠️ Need min 4 values in '{col}'"
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0: return f"✅ No outliers in '{col}'"
        return f"⚠️ Outliers in '{col}': {len(outliers)} values ({len(outliers)/len(series)*100:.1f}%)\nRange: ₹{outliers.min():,.0f} to ₹{outliers.max():,.0f}"
    
    def generate_plot(self, plot_type="bar"):
        if self.df is None: return "⚠️ Load data first"
        try:
            plt.figure(figsize=(8,5))
            num_cols = self.df.select_dtypes('number').columns
            cat_cols = self.df.select_dtypes('object').columns
            
            if "bar" in plot_type.lower() and len(cat_cols) > 0:
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(10)
                counts.plot(kind='bar', color='#667eea')
                plt.title(f'Top: {col}')
                plt.xticks(rotation=45, ha='right')
            elif "hist" in plot_type.lower() and len(num_cols) > 0:
                col = num_cols[0]
                self.df[col].hist(bins=20, edgecolor='black', color='#4ECDC4')
                plt.title(f'Histogram: {col}')
            elif "scatter" in plot_type.lower() and len(num_cols) >= 2:
                sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], alpha=0.6, color='#FF6B6B')
                plt.title(f'{num_cols[0]} vs {num_cols[1]}')
            else:
                if len(num_cols) >= 2:
                    sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], alpha=0.6)
                    plt.title(f'Scatter: {num_cols[0]} vs {num_cols[1]}')
                elif len(num_cols) > 0:
                    self.df[num_cols[0]].hist(bins=20)
                    plt.title(f'Histogram: {num_cols[0]}')
                else:
                    return "⚠️ No suitable columns for plotting"
            
            plt.tight_layout()
            os.makedirs('static', exist_ok=True)
            plt.savefig('static/plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            return "✅ Plot saved → Refresh page to see below"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"
    
    def show_correlations(self):
        if self.df is None: return "⚠️ Load data first"
        num_df = self.df.select_dtypes('number')
        if num_df.shape[1] < 2: return "⚠️ Need min 2 numeric columns for correlation"
        corr = num_df.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i,j] > 0.5:
                    pairs.append(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i,j]:.2f}")
        if not pairs: return "⚠️ No strong correlations (>0.5) found"
        return "🔗 Strong correlations:\n" + "\n".join(pairs[:5])
    
    def filter_high(self, threshold=100000):
        if self.df is None: return "⚠️ Load data first"
        rev_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total'])), self.df.select_dtypes('number').columns[0])
        filtered = self.df[self.df[rev_col] > threshold]
        return f"✅ Filtered: {len(filtered)} rows with {rev_col} > ₹{threshold:,}\nTop 3:\n" + "\n".join([f"• ₹{row[rev_col]:,.0f}" for _, row in filtered.head(3).iterrows()])
    
    def filter_low(self, threshold=5):
        if self.df is None: return "⚠️ Load data first"
        qty_col = next((c for c in self.df.columns if 'quantity' in c.lower() or 'qty' in c.lower()), self.df.select_dtypes('number').columns[0])
        filtered = self.df[self.df[qty_col] < threshold]
        return f"✅ Filtered: {len(filtered)} rows with {qty_col} < {threshold}\nSample:\n" + "\n".join([f"• {row[qty_col]}" for _, row in filtered.head(3).iterrows()])
    
    def agg_stats(self, stat_type="total"):
        if self.df is None: return "⚠️ Load data first"
        rev_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total'])), self.df.select_dtypes('number').columns[0])
        if stat_type == "total" or "sum" in stat_type:
            return f"💰 Total {rev_col}: ₹{self.df[rev_col].sum():,.0f}"
        elif stat_type == "average" or "mean" in stat_type:
            return f"📊 Avg {rev_col}: ₹{self.df[rev_col].mean():,.0f}"
        elif stat_type == "max":
            return f"📈 Max {rev_col}: ₹{self.df[rev_col].max():,.0f}"
        elif stat_type == "min":
            return f"📉 Min {rev_col}: ₹{self.df[rev_col].min():,.0f}"
        else:
            return f"⚠️ Unknown stat: {stat_type}"
    
    def returns_analysis(self):
        if self.df is None: return "⚠️ Load data first"
        ret_col = next((c for c in self.df.columns if 'return' in c.lower() or 'refund' in c.lower()), None)
        if not ret_col:
            return "⚠️ No returns column found"
        total_returns = self.df[ret_col].sum()
        total_orders = len(self.df)
        avg_returns = total_returns / total_orders
        return f"📦 Returns Analysis:\nTotal returns: {total_returns}\nAvg per order: {avg_returns:.1f}\nReturn rate: {(total_returns/total_orders)*100:.1f}%"
    
    def profit_analysis(self):
        if self.df is None: return "⚠️ Load data first"
        margin_col = next((c for c in self.df.columns if 'margin' in c.lower() or 'profit' in c.lower()), None)
        if not margin_col:
            return "⚠️ No profit margin column found"
        avg_margin = self.df[margin_col].mean() * 100
        high_margin = len(self.df[df[margin_col] > 0.4])
        return f"💡 Profit Analysis:\nAvg margin: {avg_margin:.1f}%\nHigh margin (>40%): {high_margin} items ({high_margin/len(self.df)*100:.0f}%)"
    
    def query(self, q):
        q = q.lower().strip()
        
        # Auto-load if CSV mentioned (but usually auto-loaded via upload endpoint)
        if "load" in q and ".csv" in q:
            m = re.search(r'[\w\-.]+\.csv', q)
            return self.load_data(m.group()) if m else "❌ Specify filename like 'sales.csv'"
        
        # Info commands
        if "info" in q or "basic" in q or "shape" in q:
            return self.show_info()
        if "missing" in q or "null" in q:
            return self.show_missing()
        
        # Cleaning commands
        if "clean" in q and ("data" in q or "missing" in q):
            return self.clean_data()
        if "fill" in q and "missing" in q:
            return self.fill_missing()
        if "remove" in q and "duplicate" in q:
            return self.remove_duplicates()
        
        # Top N commands
        if "top" in q:
            n_match = re.search(r'top\s+(\d+)', q)
            n = int(n_match.group(1)) if n_match else 5
            metric = "revenue"
            for word in ["revenue", "sales", "price", "quantity", "amount"]:
                if word in q:
                    metric = word
                    break
            return self.top_n(n=n, metric=metric)
        
        # Group by commands
        if "group" in q or "by" in q:
            col = "product"
            for word in ["product", "region", "category", "customer", "date"]:
                if word in q:
                    col = word
                    break
            return self.group_by(col)
        
        # Prediction commands
        if "predict" in q or "trend" in q or "forecast" in q:
            col = "revenue"
            for word in ["revenue", "sales", "quantity", "price"]:
                if word in q:
                    col = word
                    break
            return self.predict_trend(col)
        
        # Segmentation
        if "segment" in q or "customer" in q:
            return self.segment_customers()
        
        # Outliers
        if "outlier" in q or "anomaly" in q:
            col = "revenue"
            for word in ["revenue", "sales", "price", "quantity"]:
                if word in q:
                    col = word
                    break
            return self.detect_outliers(col)
        
        # Visualization
        if "plot" in q or "chart" in q or "graph" in q or "visualize" in q:
            pt = "bar"
            if "hist" in q: pt = "histogram"
            elif "scatter" in q: pt = "scatter"
            return self.generate_plot(pt)
        
        # Correlations
        if "correlation" in q or "correlate" in q:
            return self.show_correlations()
        
        # Filters
        if "filter" in q and ("high" in q or ">100000" in q or "large" in q):
            return self.filter_high()
        if "filter" in q and ("low" in q or "<5" in q or "small" in q):
            return self.filter_low()
        
        # Aggregations
        if "total revenue" in q or "sum revenue" in q:
            return self.agg_stats("total")
        if "average revenue" in q or "mean revenue" in q:
            return self.agg_stats("average")
        if "max revenue" in q:
            return self.agg_stats("max")
        if "min revenue" in q:
            return self.agg_stats("min")
        
        # Business analysis
        if "return" in q or "refund" in q:
            return self.returns_analysis()
        if "profit" in q or "margin" in q:
            return self.profit_analysis()
        
        # Help
        return ("💡 Supported commands:\n"
               "• 'top 5 by revenue'\n"
               "• 'group by region'\n"
               "• 'predict trend'\n"
               "• 'segment customers'\n"
               "• 'detect outliers price'\n"
               "• 'create bar chart'\n"
               "• 'show missing values'\n"
               "• 'clean data'\n"
               "• 'filter high revenue'\n"
               "• 'total revenue'")
