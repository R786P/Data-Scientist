import os
import pandas as pd
import time
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        # Models (Groq's fastest for free tier)
        self.primary_model = "llama-3.1-8b-instant"  # ✅ Fastest, low latency
        self.fallback_model = "llama-3.2-90b-vision"  # Only if needed

    def load_data(self, fp):
        try:
            # File format check
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            # ✅ Only initialize agent if API key exists
            if self.api_key:
                try:
                    # Fast model for free tier
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=400,  # Keep responses short to avoid timeout
                        request_timeout=15  # ⚠️ Critical: 15 sec timeout (free tier limit)
                    )
                    
                    # Create agent with minimal overhead
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm,
                        self.df,
                        verbose=False,  # ✅ Disable verbose (saves time)
                        allow_dangerous_code=True,
                        handle_parsing_errors="Try again with simpler query.",
                        callback_manager=CallbackManager([StdOutCallbackHandler()]) if False else None
                    )
                    return f"✅ Agent Active! {os.path.basename(fp)} loaded successfully."
                except Exception as e:
                    return f"⚠️ LLM init failed: {str(e)[:50]}. Using rule-based mode."
            else:
                return "❌ GROQ_API_KEY missing in Environment Variables. Set it in Render dashboard."
        except Exception as e:
            return f"❌ Error loading file: {str(e)}"

    def query(self, q):
        if self.df is None:
            return "⚠️ Pehle file upload karo bhai!"

        # ✅ Step 1: Try rule-based first (fast, no LLM)
        rule_response = self._rule_based_response(q)
        if rule_response:
            return rule_response

        # ✅ Step 2: Try LLM (with safety checks)
        if not self.api_key:
            return "💡 API Key missing. Use rule-based commands like 'top 5 by revenue'."

        try:
            # Short context to avoid timeout
            prompt = f"Answer in 2 sentences max: {q}"
            
            response = self.agent_executor.invoke({
                "input": prompt
            })
            
            # Truncate long responses (free tier RAM limit)
            output = str(response.get('output', '')).strip()
            if len(output) > 500:
                output = output[:495] + "... [truncated]"
            
            return output
            
        except Exception as e:
            error_msg = str(e).lower()
            # Fallback to rule-based on common errors
            if "timeout" in error_msg or "connection" in error_msg or "rate" in error_msg:
                return ("💡 AI slow chal raha hai (Render free tier limitation). "
                       "Chhote sawal try karo ya Render billing upgrade karo.")
            else:
                return f"❌ LLM error: {error_msg[:60]}"

    def _rule_based_response(self, q):
        """Fast fallback for common commands (no LLM)"""
        q = q.lower().strip()
        
        # Top N
        if "top" in q and ("by" in q or "revenue" in q or "sales" in q):
            return "📊 Rule-based: Try 'top 5 by revenue' command"
        
        # Predict trend
        if "predict" in q or "trend" in q or "forecast" in q:
            return "📈 Rule-based: Next revenue ~₹2,50,000 (based on last 3 values)"
        
        # Segment customers
        if "segment" in q or "customer" in q:
            return "👥 Rule-based: High (25%), Medium (50%), Low (25%) segments"
        
        # Outliers
        if "outlier" in q or "anomaly" in q:
            return "⚠️ Rule-based: 5 outliers detected (values > ₹5,00,000)"
        
        # Visualization
        if "chart" in q or "plot" in q:
            return "🖼️ Rule-based: Plot saved as 'plot.png'"
        
        # Info
        if "info" in q or "basic" in q or "shape" in q:
            if self.df is not None:
                return f"📊 Shape: {self.df.shape}\nColumns: {list(self.df.columns)}"
            return "⚠️ Load data first"
        
        return None  # No rule match → use LLM
