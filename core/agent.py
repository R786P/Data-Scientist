import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Master Imports: DB aur Models
from core.database import SessionLocal
from core.models import UserQuery

# Setup Logger
logger = logging.getLogger(__name__)

# ✅ ML integration with error safety
try:
    from .ml import MLModels
except Exception as e:
    logger.warning(f"⚠️ MLModels not loaded: {str(e)}")
    MLModels = None 

# LangChain imports (LLM support)
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
        
        # ✅ Initialize ML engine
        self.ml = MLModels() if MLModels is not None else None

    def load_data(self, fp):
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            logger.info(f"✅ Data loaded: {fp}")

            if LLM_AVAILABLE and self.api_key:
                llm = ChatGroq(
                    temperature=0,
                    groq_api_key=self.api_key,
                    model_name=self.primary_model,
                    max_tokens=400
                )
                self.agent_executor = create_pandas_dataframe_agent(
                    llm, self.df, verbose=False, allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                return f"✅ Agent Active: {os.path.basename(fp)} loaded."
            return f"✅ Loaded data: {len(self.df)} rows."
        except Exception as e:
            logger.error(f"Load error: {e}")
            return f"❌ Load error: {str(e)}"

    # --- 📊 MASTER FEATURE: MULTIPLE PLOTS FOR DASHBOARD ---
    def generate_multi_plots(self):
        """Dashboard ke liye ek saath multiple plots generate karta hai"""
        if self.df is None: return "⚠️ Load data first"
        try:
            os.makedirs('static', exist_ok=True)
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()

            # 1. Bar Chart (Top Categories)
            if cat_cols:
                plt.figure(figsize=(8, 5))
                self.df[cat_cols[0]].value_counts().head(10).plot(kind='bar', color='#667eea')
                plt.title(f'Top {cat_cols[0]}')
                plt.savefig('static/plot_bar.png')
                plt.close()

            # 2. Histogram (Distribution)
            if num_cols:
                plt.figure(figsize=(8, 5))
                sns.histplot(self.df[num_cols[0]], kde=True, color='#4ECDC4')
                plt.title(f'Distribution of {num_cols[0]}')
                plt.savefig('static/plot_dist.png')
                plt.close()

            # 3. Correlation Heatmap
            if len(num_cols) > 1:
                plt.figure(figsize=(8, 5))
                sns.heatmap(self.df[num_cols].corr(), annot=True, cmap='coolwarm')
                plt.title('Feature Correlation')
                plt.savefig('static/plot_heatmap.png')
                plt.close()

            return "✅ Multi-plots generated for Dashboard."
        except Exception as e:
            return f"❌ Multi-plot error: {str(e)}"

    def generate_plot(self, plot_type="bar"):
        if self.df is None: return "⚠️ Load data first"
        try:
            plt.figure(figsize=(10, 6))
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            os.makedirs('static', exist_ok=True)

            if "bar" in plot_type.lower() and cat_cols:
                self.df[cat_cols[0]].value_counts().head(10).plot(kind='bar', color='#667eea')
            elif "hist" in plot_type.lower() and num_cols:
                self.df[num_cols[0]].hist(bins=20, color='#4ECDC4')
            else:
                return "⚠️ No valid data for this plot."

            plt.tight_layout()
            plt.savefig('static/plot.png', dpi=150)
            plt.close()
            return "✅ Plot saved! View at: /plot.png"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"

    def query(self, q):
        if self.df is None: return "⚠️ Pehle file upload karo!"
        
        q_lower = q.lower()
        
        # Check for Dashboard/Multi-plot command
        if "dashboard" in q_lower or "multi plot" in q_lower:
            return self.generate_multi_plots()

        # Rule-based / Plotting
        rule_resp = self._rule_based(q)
        if rule_resp: 
            final_response = rule_resp
        else:
            # LLM Logic
            try:
                res = self.agent_executor.invoke({"input": q})
                final_response = str(res.get('output', 'AI could not process this.'))
            except:
                final_response = "💡 AI slow hai. Chhote sawal pucho."

        # Database Logging
        db = SessionLocal()
        try:
            new_log = UserQuery(query_text=q, response_text=final_response)
            db.add(new_log)
            db.commit()
        finally:
            db.close()

        return final_response

    def _rule_based(self, q):
        q = q.lower()
        if "plot" in q or "chart" in q: return self.generate_plot("bar")
        if "summary" in q or "eda" in q: 
            missing = int(self.df.isnull().sum().sum())
            return f"📊 EDA: {len(self.df)} rows | Missing: {missing}"
        return None
