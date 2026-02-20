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
except ImportError as e:
    LLM_AVAILABLE = False
    logger.error(f"❌ LangChain import failed: {e}")

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
            logger.info(f"✅ Data loaded: {fp} ({len(self.df)} rows)")

            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0.5,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=500,
                        timeout=15  # ✅ Fast timeout
             self.agent_executor = create_pandas_dataframe_agent(
             llm, self.df, verbose=False, allow_dangerous_code=True
                    ) 
                    logger.info("✅ LLM Agent initialized")
                    return f"✅ Agent Active: {os.path.basename(fp)} loaded."
                except Exception as e:
                    logger.warning(f"⚠️ LLM Init Failed: {e}")
                    return f"✅ Loaded data: {len(self.df)} rows (AI Mode: OFF)"
            
            return f"✅ Loaded data: {len(self.df)} rows."
        except Exception as e:
            logger.error(f"❌ Load error: {e}")
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
        
        # ✅ FAST RULE-BASED (Instant Response - 90% queries)
        rule_response = self._fast_rule_based(q_lower)
        if rule_response:
            final_response = rule_response
        else:
            # ✅ SLOW AI (Only for complex queries - 10% queries)
            if LLM_AVAILABLE and self.api_key and self.agent_executor:
                try:
                    prompt = f"Answer in simple Hindi/English mix like a friend. Max 2 sentences. Question: {q}"
                    res = self.agent_executor.invoke({"input": prompt})
                    final_response = str(res.get('output', 'Could not process.'))
                except Exception as e:
                    logger.error(f"❌ LLM Error: {e}")
                    final_response = "💡 AI busy hai. Simple pucho jaise 'top 5', 'average', 'summary'"
            else:
                final_response = "💡 AI Mode OFF. Try: 'top 5', 'average', 'summary', 'plot'"

        # Database Logging (Optional)
        try:
            db = SessionLocal()
            new_log = UserQuery(query_text=q, response_text=final_response, user_id=user_id)
            db.add(new_log)
            db.commit()
            db.close()
        except:
            pass

        return final_response

    def _fast_rule_based(self, q):
        """✅ Instant Responses (No AI needed)"""
        try:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            # Greetings
            if q in ['hi', 'hello', 'hey', 'hii', 'namaste']:
                return "👋 Hello! Main aapka Data Scientist Agent hoon. File upload ho gayi hai — pucho kuch bhi!"
            
            # How are you
            if 'kaise ho' in q or 'how are you' in q:
                return "😊 Main badhiya hoon! Aap batao, kya analyze karna hai aaj?"
            
            # What can you do
            if 'kya kar' in q or 'what can' in q or 'help' in q:
                return "📊 Main ye kar sakta hoon:\n• Top 5 categories\n• Average/Mean values\n• Charts/Plots\n• Summary/Info\n• Trends analysis"
            
            # Summary/Info
            if 'summary' in q or 'info' in q or 'data' in q or 'kya hai' in q:
                return f"📊 **Data Summary:**\n• Total Rows: {len(self.df)}\n• Columns: {len(self.df.columns)}\n• Column Names: {', '.join(self.df.columns.tolist()[:5])}..."
            
            # Top/Max
            if 'top' in q or 'max' in q or 'highest' in q or 'sabse zyada' in q:
                if cat_cols:
                    col = cat_cols[0]
                    top_vals = self.df[col].value_counts().head(5)
                    result = f"🏆 **Top 5 {col}:**\n"
                    for i, (name, count) in enumerate(top_vals.items(), 1):
                        result += f"{i}. {name}: {count}\n"
                    return result
                elif num_cols:
                    col = num_cols[0]
                    max_val = self.df[col].max()
                    return f"📈 **Highest {col}:** {max_val}"
            
            # Average/Mean
            if 'avg' in q or 'average' in q or 'mean' in q or 'ausat' in q:
                if num_cols:
                    col = num_cols[0]
                    avg_val = self.df[col].mean()
                    return f"📊 **Average {col}:** {avg_val:.2f}"
            
            # Count/Rows
            if 'count' in q or 'rows' in q or 'lines' in q or 'kitne' in q:
                return f"📏 **Total Records:** {len(self.df)} rows × {len(self.df.columns)} columns"
            
            # Plot/Chart
            if 'plot' in q or 'chart' in q or 'graph' in q or 'dikha' in q:
                return self.generate_plot("bar")
            
            # Trend
            if 'trend' in q or 'forecast' in q or 'bhavishya' in q:
                if num_cols:
                    col = num_cols[0]
                    last_val = self.df[col].iloc[-1] if len(self.df) > 0 else 0
                    first_val = self.df[col].iloc[0] if len(self.df) > 0 else 0
                    trend = "↗️ Badh raha hai" if last_val > first_val else "↘️ Ghat raha hai"
                    return f"📈 **Trend:** {trend}\nCurrent Value: {last_val}"
            
            # Missing values
            if 'missing' in q or 'null' in q or 'khali' in q:
                missing = int(self.df.isnull().sum().sum())
                return f"⚠️ **Missing Values:** {missing} total"
            
            return None  # Let AI handle complex queries
        except Exception as e:
            logger.error(f"Rule based error: {e}")
            return None
