import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.database import SessionLocal, UserQuery
from core.ml import MLModels

logger = logging.getLogger(__name__)

try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents import create_pandas_dataframe_agent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        self.primary_model = "llama-3.1-8b-instant"
        self.ml = MLModels()

    def load_data(self, fp):
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            # Clean column names (remove spaces)
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            logger.info(f"✅ Data loaded: {fp}")

            # Initialize LLM only if key exists
            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=500,
                        timeout=10  # ✅ Timeout kam kiya
                    )
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm, self.df, verbose=False, allow_dangerous_code=True,
                        handle_parsing_errors=True
                    )
                    return f"✅ Agent Active: {os.path.basename(fp)} loaded."
                except Exception as e:
                    logger.warning(f"⚠️ LLM Init Failed: {e}")
                    return f"✅ Loaded data: {len(self.df)} rows (AI Mode Off)"
            
            return f"✅ Loaded data: {len(self.df)} rows."
        except Exception as e:
            logger.error(f"Load error: {e}")
            return f"❌ Load error: {str(e)}"

    def generate_plot(self, plot_type="bar"):
        if self.df is None: return "⚠️ Pehle file upload karo!"
        try:
            plt.figure(figsize=(10, 6))
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            os.makedirs('static', exist_ok=True)

            if "bar" in plot_type.lower() and cat_cols:
                self.df[cat_cols[0]].value_counts().head(10).plot(kind='bar', color='#667eea')
                plt.title(f'Top {cat_cols[0]}')
            elif "hist" in plot_type.lower() and num_cols:
                self.df[num_cols[0]].hist(bins=20, color='#4ECDC4')
                plt.title(f'Distribution of {num_cols[0]}')
            else:
                return "⚠️ No valid data for this plot."

            plt.tight_layout()
            plt.savefig('static/plot.png', dpi=150)
            plt.close()
            return "✅ Plot saved! View at: /plot.png"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"

    def query(self, q, user_id=None):
        if self.df is None: return "⚠️ Pehle file upload karo!"
        
        q_lower = q.lower()
        
        # ✅ SMART RULE-BASED LOGIC (No AI Needed - Fast!)
        rule_resp = self._smart_rule_based(q_lower)
        if rule_resp:
            final_response = rule_resp
        else:
            # ✅ Try AI only if rule fails
            if LLM_AVAILABLE and self.api_key and self.agent_executor:
                try:
                    res = self.agent_executor.invoke({"input": q})
                    final_response = str(res.get('output', 'AI could not process this.'))
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    final_response = "💡 Complex sawal hai. Thoda simple pucho (e.g., 'top 5', 'average marks')."
            else:
                final_response = "💡 AI Mode Off. Please ask: 'top 5', 'average', 'plot', or 'summary'."

        # Database Logging
        try:
            db = SessionLocal()
            new_log = UserQuery(query_text=q, response_text=final_response, user_id=user_id)
            db.add(new_log)
            db.commit()
        except Exception as e:
            logger.warning(f"⚠️ Query logging failed: {e}")
        finally:
            db.close()

        return final_response

    def _smart_rule_based(self, q):
        """✅ Enhanced Rule Engine - Handles 80% queries without AI"""
        try:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            # 1. Summary / Info
            if "summary" in q or "info" in q or "data" in q:
                return f"📊 **Data Summary:**\n• Rows: {len(self.df)}\n• Columns: {len(self.df.columns)}\n• Missing Values: {int(self.df.isnull().sum().sum())}"
            
            # 2. Top / Max
            if "top" in q or "max" in q or "highest" in q:
                if cat_cols:
                    col = cat_cols[0]
                    top_vals = self.df[col].value_counts().head(5)
                    return f"🏆 **Top 5 {col}:**\n" + "\n".join([f"{i}: {v}" for i, v in top_vals.items()])
                elif num_cols:
                    col = num_cols[0]
                    max_val = self.df[col].max()
                    return f"📈 **Highest {col}:** {max_val}"
            
            # 3. Average / Mean
            if "average" in q or "mean" in q or "avg" in q:
                if num_cols:
                    col = num_cols[0]
                    avg_val = self.df[col].mean()
                    return f"📊 **Average {col}:** {avg_val:.2f}"
            
            # 4. Trends / Forecast (Simple Logic)
            if "trend" in q or "forecast" in q or "future" in q:
                if num_cols:
                    col = num_cols[0]
                    last_val = self.df[col].iloc[-1]
                    first_val = self.df[col].iloc[0]
                    trend = "↗️ Upward" if last_val > first_val else "↘️ Downward"
                    return f"📈 **Trend:** {trend}\nLast Value: {last_val}"
            
            # 5. Plot / Chart
            if "plot" in q or "chart" in q or "graph" in q:
                return self.generate_plot("bar")
            
            # 6. Shape
            if "shape" in q or "size" in q:
                return f"📏 **Shape:** {self.df.shape[0]} rows × {self.df.shape[1]} columns"
            
            return None
        except Exception as e:
            logger.error(f"Rule based error: {e}")
            return None
