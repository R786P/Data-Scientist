import os
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        # Multiple models ki list aapke screenshot ke hisab se
        self.models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    def _get_llm(self, model_idx=0):
        """Model switch karne ka logic"""
        return ChatGroq(
            temperature=0, 
            groq_api_key=self.api_key, 
            model_name=self.models[model_idx]
        )

    def load_data(self, fp):
        try:
            ext = os.path.splitext(fp)[-1].lower()
            if ext == '.csv':
                self.df = pd.read_csv(fp, encoding='latin1')
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(fp)
            
            if self.api_key:
                # Pehle primary model try karein
                llm = self._get_llm(0)
                self.agent_executor = create_pandas_dataframe_agent(
                    llm, 
                    self.df, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                return f"✅ Groq Multi-Model Agent Active! {fp} loaded."
            return "❌ GROQ_API_KEY missing in Render settings."
        except Exception as e:
            return f"❌ Load Error: {str(e)}"

    def query(self, q):
        if self.df is None: return "⚠️ File load kijiye."
        
        # Try-Except block for model fallback
        for i in range(len(self.models)):
            try:
                response = self.agent_executor.invoke({"input": q})
                return response['output']
            except Exception as e:
                if i < len(self.models) - 1:
                    print(f"🔄 Switching to fallback model: {self.models[i+1]}")
                    self.agent_executor.agent.llm = self._get_llm(i+1)
                    continue
                return f"💡 All models busy. Error: {str(e)}"
