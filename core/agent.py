import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        # Aapke screenshot ke best models
        self.primary_model = "llama-3.3-70b-versatile" 
        self.fallback_model = "llama-3.1-8b-instant"

    def load_data(self, fp):
        try:
            # File format check
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            if self.api_key:
                # Agent initialization with Groq
                llm = ChatGroq(
                    temperature=0, 
                    groq_api_key=self.api_key, 
                    model_name=self.primary_model,
                    max_tokens=500 # Response chhota rakhein taaki timeout na ho
                )
                
                self.agent_executor = create_pandas_dataframe_agent(
                    llm, 
                    self.df, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                return f"✅ Agent Active! {os.path.basename(fp)} loaded successfully."
            return "❌ API Key missing in Environment Variables."
        except Exception as e:
            return f"❌ Error loading file: {str(e)}"

    def query(self, q):
        if self.df is None: 
            return "⚠️ Pehle file upload karo bhai!"
        
        try:
            # Direct execution
            response = self.agent_executor.invoke({"input": q})
            return response['output']
        except Exception as e:
            # Agar primary model fail ho toh switch karein
            print(f"🔄 Switching to fast model due to error: {e}")
            try:
                llm_fast = ChatGroq(
                    temperature=0, 
                    groq_api_key=self.api_key, 
                    model_name=self.fallback_model
                )
                # Re-run with lighter model
                self.agent_executor.agent.llm = llm_fast
                response = self.agent_executor.invoke({"input": q})
                return response['output']
            except:
                return "💡 AI bohot slow chal raha hai. Chhota sawal poochiye ya Render billing check karein."
