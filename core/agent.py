import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

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
            
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            logger.info(f"✅ Data loaded: {fp}")

            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0.7,  # ✅ Conversational (not 0)
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=800,
                        timeout=30
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
        
        # ✅ Rule-Based for Speed (Plot, Summary)
        if "plot" in q_lower or "chart" in q_lower:
            final_response = self.generate_plot("bar")
        else:
            # ✅ Generative AI for Conversation
            if LLM_AVAILABLE and self.api_key and self.agent_executor:
                try:
                    # ✅ Conversational Prompt
                    prompt = f"Answer in friendly human language (Hindi/English mix). Keep it short (2-3 sentences). Question: {q}"
                    res = self.agent_executor.invoke({"input": prompt})
                    final_response = str(res.get('output', 'AI could not process this.'))
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    final_response = "💡 AI busy hai. Thoda simple pucho (e.g., 'top 5', 'average')."
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
