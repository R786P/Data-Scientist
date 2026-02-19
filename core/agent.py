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
    from langchain_core.callbacks.manager import CallbackManager
    from langchain_core.callbacks.stdout import StdOutCallbackHandler
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        self.primary_model = "llama-3.1-8b-instant"
        self.fallback_model = "llama-3.2-90b-vision"
        
        # ✅ Initialize ML engine
        self.ml = MLModels() if MLModels is not None else None

    def load_data(self, fp):
        """Master Feature: Professional file loading with logging"""
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            logger.info(f"✅ Data loaded: {fp}")

            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=400,
                        request_timeout=15
                    )
                    
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm, self.df, verbose=False, allow_dangerous_code=True,
                        handle_parsing_errors="Try simpler query."
                    )
                    return f"✅ Agent Active: {os.path.basename(fp)} loaded."
                except Exception as e:
                    logger.error(f"LLM init failed: {e}")
                    return "⚠️ LLM init failed. Using rule-based mode."
            return f"✅ Loaded data: {len(self.df)} rows."
        except Exception as e:
            logger.error(f"Load error: {e}")
            return f"❌ Load error: {str(e)}"

    def generate_plot(self, plot_type="bar"):
        """Master Feature: Safe directory handling & Plotting"""
        if self.df is None: return "⚠️ Load data first"
        try:
            plt.figure(figsize=(10, 6))
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()

            os.makedirs('static', exist_ok=True) # Directory check

            if "bar" in plot_type.lower() and cat_cols:
                col = cat_cols[0]
                self.df[col].value_counts().head(10).plot(kind='bar', color='#667eea')
                plt.title(f'Top: {col}')
            elif "hist" in plot_type.lower() and num_cols:
                self.df[num_cols[0]].hist(bins=20, color='#4ECDC4')
                plt.title(f'Distribution: {num_cols[0]}')
            elif "scatter" in plot_type.lower() and len(num_cols) >= 2:
                sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1], color='#FF6B6B')
                plt.title(f'{num_cols[0]} vs {num_cols[1]}')
            else:
                return "⚠️ No valid columns found for this plot type."

            plt.tight_layout()
            plt.savefig('static/plot.png', dpi=150)
            plt.close()
            return "✅ Plot saved! View at: /static/plot.png"
        except Exception as e:
            logger.error(f"Plot error: {e}")
            return f"❌ Plot error: {str(e)}"

    def auto_eda(self):
        """Master Feature: Automated quick analysis"""
        if self.df is None: return "⚠️ Load data first"
        missing = int(self.df.isnull().sum().sum())
        duplicates = int(self.df.duplicated().sum())
        return f"📊 EDA: {len(self.df)} rows | Missing: {missing} | Dups: {duplicates}"

    def query(self, q):
        """Master Feature: Every query is logged to PostgreSQL"""
        if self.df is None: return "⚠️ Pehle file upload karo!"

        final_response = ""
        q_lower = q.lower()

        # Rule-based / Plotting Check
        rule_resp = self._rule_based(q)
        if rule_resp: final_response = rule_resp
        
        # ML Logic (Part 2 logic)
        elif self.ml and any(x in q_lower for x in ["forecast", "predict", "segment"]):
            try:
                if "forecast" in q_lower:
                    res = self.ml.forecast_time_series([100, 150, 200]) # Sample logic
                    final_response = f"📈 Forecast: {res['forecast']}"
                else:
                    final_response = "🚀 ML Prediction activated."
            except Exception as e:
                final_response = f"❌ ML Error: {str(e)}"
        
        # LLM Logic
        elif self.agent_executor and self.api_key:
            try:
                res = self.agent_executor.invoke({"input": q})
                final_response = str(res.get('output', 'AI could not process this.'))
            except Exception as e:
                final_response = "💡 AI slow chal raha hai. Chhote sawal try karo."

        # --- DATABASE LOGGING ---
        db = SessionLocal()
        try:
            new_log = UserQuery(query_text=q, response_text=final_response)
            db.add(new_log)
            db.commit()
            logger.info("📊 Query & Response logged to SQL DB.")
        except Exception as db_err:
            logger.error(f"DB Log failed: {db_err}")
        finally:
            db.close()

        return final_response

    def _rule_based(self, q):
        q = q.lower()
        if "plot" in q or "chart" in q: return self.generate_plot("bar")
        if "summary" in q or "eda" in q: return self.auto_eda()
        return None
