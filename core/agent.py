import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        # Screenshot ke hisab se 'llama-3.3-70b-versatile' best hai, 
        # par speed ke liye 'llama-3.1-8b-instant' fallback rakhenge.
        self.primary_model = "llama-3.3-70b-versatile"
        self.fast_model = "llama-3.1-8b-instant"

    def load_data(self, fp):
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            if self.api_key:
                # Optimized for Speed
                llm = ChatGroq(
                    temperature=0, 
                    groq_api_key=self.api_key, 
                    model_name=self.primary_model,
                    max_tokens=1000 # Takki response lamba na khinche
                )
                
                self.agent_executor = create_pandas_dataframe_agent(
                    llm, 
                    self.df, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                return f"✅ Groq Agent Ready! Model: {self.primary_model}"
            return "❌ API Key missing."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def query(self, q):
        if self.df is None: return "⚠️ File load karein."
        try:
            # Response timeout se bachne ke liye invoke use karein
            response = self.agent_executor.invoke({"input": q})
            return response['output']
        except Exception as e:
            return "💡 AI is taking too long. Try a simpler question."
