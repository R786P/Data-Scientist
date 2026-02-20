import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.database import SessionLocal, UserQuery
from core.ml import MLModels

# ✅ LangChain Imports (Optional)
try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents import create_pandas_dataframe_agent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.last_file = None
        self.ml = MLModels()
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # ✅ Initialize AI if available
        if LLM_AVAILABLE and self.api_key:
            try:
                llm = ChatGroq(
                    temperature=0.5,
                    groq_api_key=self.api_key,
                    model_name="llama-3.1-8b-instant",
                    max_tokens=500,
                    timeout=15
                )
                print("✅ AI Mode: ON")
            except Exception as e:
                print(f"⚠️ AI Mode: OFF (API Key issue: {e})")
    
    def load_data(self, fp):
        try:
            self.df = pd.read_csv(fp, encoding='latin1')
            self.last_file = fp
            num = self.df.select_dtypes('number').columns.tolist()
            cat = self.df.select_dtypes('object').columns.tolist()
            
            # ✅ Initialize AI Agent with data
            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0.5,
                        groq_api_key=self.api_key,
                        model_name="llama-3.1-8b-instant",
                        max_tokens=500,
                        timeout=15
                    )
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm, self.df, verbose=False, allow_dangerous_code=True
                    )
                except Exception as e:
                    print(f"⚠️ AI Agent init failed: {e}")
            
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
        if self.df is None:
            return "⚠️ Load data first (upload CSV)"
        target_col = None
        for col in self.df.columns:
            if col_name.lower() in col.lower():
                target_col = col
                break
        if not target_col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns for prediction"
            target_col = num_cols[0]
        series = self.df[target_col].dropna().tolist()
        if len(series) >= 3:
            result = self.ml.forecast_time_series(series, periods=3)
            if 'forecast' in result:
                forecast_str = " → ".join([f"₹{v:,.0f}" for v in result['forecast']])
                return f"📈 ML Forecast ({result.get('trend', 'N/A')}):\nNext 3: {forecast_str}"
        last3 = series[-3:] if len(series) >= 3 else series
        trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
        next_val = series[-1] + (series[-1] - series[-2]) if len(series) > 1 else series[-1]
        return f"📈 Trend: {trend}\nNext: ₹{next_val:,.0f}"
    
    def segment_customers(self):
        if self.df is None:
            return "⚠️ Load data first (upload CSV)"
        spend_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total','spend'])), None)
        freq_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['freq','purchase','order','quantity'])), None)
        sample = {
            'annual_spend': self.df[spend_col].mean() * 12 if spend_col else 150000,
            'purchase_frequency': self.df[freq_col].mean() if freq_col else 12
        }
        result = self.ml.segment_customer(sample)
        if 'error' in result:
            rev_col = spend_col or self.df.select_dtypes('number').columns[0]
            q25 = self.df[rev_col].quantile(0.25)
            q75 = self.df[rev_col].quantile(0.75)
            high = len(self.df[self.df[rev_col] > q75])
            medium = len(self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)])
            low = len(self.df[self.df[rev_col] < q25])
            total = len(self.df)
            return (f"👥 Segments (Rule-based):\n"
                   f"High: {high} ({high/total*100:.0f}%)\n"
                   f"Medium: {medium} ({medium/total*100:.0f}%)\n"
                   f"Low: {low} ({low/total*100:.0f}%)")
        return (f"🏷️ {result['segment']} Segment\n"
               f"Discount: {result['discount_eligible']}\n"
               f"💡 {result['recommendation']}")
    
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
            return "✅ Plot saved! View at: /plot.png"
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
        high_margin = len(self.df[self.df[margin_col] > 0.4])
        return f"💡 Profit Analysis:\nAvg margin: {avg_margin:.1f}%\nHigh margin (>40%): {high_margin} items ({high_margin/len(self.df)*100:.0f}%)"
    
    # ✅ HYBRID QUERY METHOD (Fast Rule-Based + AI Fallback + DB Logging)
    def query(self, q, user_id=None):
        q = q.lower().strip()
        
        # ✅ GREETINGS (Instant - 0 sec)
        if q in ['hi', 'hello', 'hey', 'hii', 'namaste']:
            return "👋 Hello! Main aapka Data Scientist Agent hoon. CSV upload karo aur kuch bhi pucho!"
        
        # ✅ HELP (Instant - 0 sec)
        if 'help' in q or 'kya kar' in q or 'what can' in q:
            return ("💡 Try these commands:\n"
                   "• 'top 5 by revenue'\n"
                   "• 'group by region'\n"
                   "• 'predict trend'\n"
                   "• 'segment customers'\n"
                   "• 'detect outliers'\n"
                   "• 'create bar chart'\n"
                   "• 'show missing values'\n"
                   "• 'clean data'\n"
                   "• 'total revenue'")
        
        # FLEXIBLE PARSING (not exact match)
        if "load" in q and ".csv" in q:
            m = re.search(r'[\w\-.]+\.csv', q)
            return self.load_data(m.group()) if m else "❌ Specify filename"
        
        # Info (multiple variations)
        if any(x in q for x in ["info", "basic", "shape", "columns", "structure"]):
            return self.show_info()
        
        if any(x in q for x in ["missing", "null", "empty", "nan"]):
            return self.show_missing()
        
        # Cleaning (multiple variations)
        if "clean" in q and any(x in q for x in ["data", "dataset", "missing", "remove"]):
            return self.clean_data()
        
        if "fill" in q and "missing" in q:
            return self.fill_missing()
        
        if "remove" in q and any(x in q for x in ["duplicate", "dup", "copy"]):
            return self.remove_duplicates()
        
        # Top N (flexible parsing)
        if "top" in q:
            n_match = re.search(r'top\s+(\d+)', q)
            n = int(n_match.group(1)) if n_match else 5
            metric = "revenue"
            for word in ["revenue", "sales", "price", "quantity", "amount", "profit", "margin"]:
                if word in q:
                    metric = word
                    break
            return self.top_n(n=n, metric=metric)
        
        # Group by (flexible)
        if "group" in q or "by" in q or "breakdown" in q:
            col = "product"
            for word in ["product", "region", "category", "customer", "date", "month", "year"]:
                if word in q:
                    col = word
                    break
            return self.group_by(col)
        
        # Prediction (flexible)
        if any(x in q for x in ["predict", "trend", "forecast", "next", "future"]):
            col = "revenue"
            for word in ["revenue", "sales", "quantity", "price", "demand"]:
                if word in q:
                    col = word
                    break
            return self.predict_trend(col)
        
        # Segmentation (flexible)
        if any(x in q for x in ["segment", "customer", "group", "cluster", "tier"]):
            return self.segment_customers()
        
        # Outliers (flexible)
        if any(x in q for x in ["outlier", "anomaly", "unusual", "weird", "strange"]):
            col = "revenue"
            for word in ["revenue", "sales", "price", "quantity", "amount"]:
                if word in q:
                    col = word
                    break
            return self.detect_outliers(col)
        
        # Visualization (flexible)
        if any(x in q for x in ["plot", "chart", "graph", "visualize", "show", "display"]):
            if "bar" in q or "count" in q:
                return self.generate_plot("bar")
            elif "hist" in q or "distribut" in q:
                return self.generate_plot("histogram")
            elif "scatter" in q or "relation" in q:
                return self.generate_plot("scatter")
            else:
                return self.generate_plot("bar")  # Default
        
        # Correlations
        if any(x in q for x in ["correlat", "relationship", "link", "connect"]):
            return self.show_correlations()
        
        # Filters
        if "filter" in q or "where" in q:
            if any(x in q for x in ["high", "large", "big", ">100000"]):
                return self.filter_high()
            elif any(x in q for x in ["low", "small", "tiny", "<5"]):
                return self.filter_low()
        
        # Aggregations
        if "total" in q or "sum" in q:
            return self.agg_stats("total")
        if "average" in q or "mean" in q or "avg" in q:
            return self.agg_stats("average")
        if "max" in q or "maximum" in q or "highest" in q:
            return self.agg_stats("max")
        if "min" in q or "minimum" in q or "lowest" in q:
            return self.agg_stats("min")
        
        # Business analysis
        if any(x in q for x in ["return", "refund", "cancel"]):
            return self.returns_analysis()
        if any(x in q for x in ["profit", "margin", "earnings"]):
            return self.profit_analysis()
        
        # ✅ AI FALLBACK (For complex queries - 15-30 sec)
        if LLM_AVAILABLE and self.api_key and self.agent_executor and self.df is not None:
            try:
                prompt = f"Answer in simple Hindi/English mix. Max 2 sentences. Question: {q}"
                res = self.agent_executor.invoke({"input": prompt})
                final_response = str(res.get('output', 'Could not process.'))
            except Exception as e:
                final_response = "💡 AI busy hai. Simple pucho jaise 'top 5', 'average', 'summary'"
        else:
            final_response = ("💡 Try these commands:\n"
                   "• 'top 5 by revenue'\n"
                   "• 'group by region'\n"
                   "• 'predict trend'\n"
                   "• 'segment customers'\n"
                   "• 'detect outliers'\n"
                   "• 'create bar chart'\n"
                   "• 'show missing values'\n"
                   "• 'clean data'\n"
                   "• 'total revenue'")
        
        # ✅ Database Logging (For current version with login)
        try:
            db = SessionLocal()
            new_log = UserQuery(query_text=q, response_text=final_response, user_id=user_id)
            db.add(new_log)
            db.commit()
            db.close()
        except Exception as e:
            print(f"⚠️ DB logging failed: {e}")
        
        return final_response
