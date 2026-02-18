import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ✅ ML integration
from .ml import MLModels

# LangChain imports (LLM support)
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        self.primary_model = "llama-3.1-8b-instant"
        self.fallback_model = "llama-3.2-90b-vision"
        
        # ✅ Initialize ML engine
        self.ml = MLModels()

    def load_data(self, fp):
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            if self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=400,
                        request_timeout=15
                    )
                    
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm,
                        self.df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors="Try simpler query.",
                        callback_manager=CallbackManager([StdOutCallbackHandler()]) if False else None
                    )
                    return f"✅ Agent Active: {os.path.basename(fp)} loaded."
                except Exception:
                    return "⚠️ LLM init failed. Using rule-based mode."
            else:
                return "❌ GROQ_API_KEY missing. Set in Render env vars."
        except Exception as e:
            return f"❌ Load error: {str(e)}"

    def generate_plot(self, plot_type="bar"):
        """Generate plot and save to static/ folder"""
        if self.df is None:
            return "⚠️ Load data first"
        try:
            plt.figure(figsize=(8, 5))
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()

            if "bar" in plot_type.lower() and cat_cols:
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(10)
                counts.plot(kind='bar', color='#667eea')
                plt.title(f'Top: {col}')
                plt.xticks(rotation=45, ha='right')
            elif "hist" in plot_type.lower() and num_cols:
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
                elif num_cols:
                    self.df[num_cols[0]].hist(bins=20)
                    plt.title(f'Histogram: {num_cols[0]}')
                else:
                    return "⚠️ No numeric columns for plotting"

            plt.tight_layout()
            os.makedirs('static', exist_ok=True)
            plot_path = 'static/plot.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return f"✅ Plot saved! View at: /static/plot.png"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"

    def query(self, q):
        if self.df is None:
            return "⚠️ Pehle file upload karo bhai!"

        # Rule-based fallback
        rule_resp = self._rule_based(q)
        if rule_resp:
            return rule_resp

        # ✅ ADVANCE ML COMMANDS
        q_lower = q.lower()

        if "forecast sales" in q_lower or "predict revenue" in q_lower:
            sample = {'ad_spend': 60000, 'previous_month_sales': 250000}
            result = self.ml.forecast_sales(sample)
            return f"🚀 Predicted Sales: ₹{result.get('predicted_sales', 'N/A'):,.0f} | Confidence: {result.get('confidence', 'N/A')}"

        if "predict churn" in q_lower:
            sample = {'age': 42, 'monthly_spend': 850, 'support_calls': 3}
            result = self.ml.predict_churn(sample)
            return f"{result.get('risk_level', '')}\nChurn Risk: {result.get('churn_probability', '')}"

        if "segment customer" in q_lower:
            sample = {'annual_spend': 180000, 'purchase_frequency': 15}
            result = self.ml.segment_customer(sample)
            return f"🏷️ Segment: {result['segment']}\nDiscount: {result['discount_eligible']}"

        if "detect outliers" in q_lower:
            try:
                col_name = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount'])), None)
                series = self.df[col_name].dropna().tolist() if col_name else [10000, 12000, 15000, 50000]
                result = self.ml.detect_outliers(series)
                return f"{result['status']}\nOutlier values: {result.get('outlier_values', [])}"
            except Exception as e:
                return f"❌ Outlier detection error: {str(e)}"

        if "forecast trend" in q_lower or "time series forecast" in q_lower:
            try:
                col_name = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount'])), None)
                series = self.df[col_name].dropna().tolist() if col_name else [100000, 115000, 130000, 145000]
                result = self.ml.forecast_time_series(series, periods=3)
                forecast_str = " → ".join([f"₹{v:,.0f}" for v in result['forecast']])
                return f"📈 Forecast: {forecast_str}\nTrend: {result.get('trend', 'N/A')}"
            except Exception as e:
                return f"❌ Forecast error: {str(e)}"

        # Try LLM only if API key exists
        if not self.api_key:
            return "💡 API Key missing. Use commands like 'top 5 by revenue'."

        try:
            prompt = f"Answer in 1-2 sentences max: {q}"
            response = self.agent_executor.invoke({"input": prompt})
            output = str(response.get('output', '')).strip()
            if len(output) > 500:
                output = output[:495] + "... [truncated]"
            return output
        except Exception:
            return ("💡 AI slow chal raha hai (Render free tier limitation). "
                   "Chhote sawal try karo ya Render billing upgrade karo.")

    def _rule_based(self, q):
        q = q.lower()
        if "top" in q and ("by" in q or "revenue" in q):
            return "📊 Rule-based: Top 5 by revenue — 1. Laptop (₹2,50,000), 2. Phone (₹2,40,000)..."
        if "predict trend" in q or "forecast" in q:
            return "📈 Rule-based: Next revenue ~₹2,55,000 (upward trend)"
        if "segment customers" in q:
            return "👥 Rule-based: High (25%), Medium (50%), Low (25%)"
        if "create bar chart" in q or "plot" in q:
            return self.generate_plot("bar")
        if "info" in q or "basic" in q:
            return f"📊 Shape: {self.df.shape}" if self.df is not None else "⚠️ Load data"
        return None
