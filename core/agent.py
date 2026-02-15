import os
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        # Render se API Key uthana
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
    def _init_brain(self):
        """AI Brain ko initialize karne ka internal tarika"""
        if not self.api_key:
            return None
        
        # Flash model sabse fast hai, iska stable version use kar rahe hain
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=self.api_key,
            temperature=0,
            safety_settings={}, # Restrictions hatane ke liye
            convert_system_message_to_human=True
        )

    def load_data(self, fp):
        try:
            ext = os.path.splitext(fp)[-1].lower()
            if ext == '.csv':
                self.df = pd.read_csv(fp, encoding='latin1')
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(fp)
            
            llm = self._init_brain()
            if llm:
                # Master Executor jo aapka har sawal samjhega
                self.agent_executor = create_pandas_dataframe_agent(
                    llm, 
                    self.df, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                return f"✅ Master Agent Ready: {fp} load ho gayi. Ab jo mann mein aaye poocho!"
            return "❌ API Key ka issue hai, Render settings check karein."
        except Exception as e:
            return f"❌ Data Load Error: {str(e)}"

    def query(self, q):
        q_lower = q.lower().strip()
        
        # Load command backup
        if "load" in q_lower:
            m = re.search(r'[\w\-.]+\.(csv|xlsx|xls)', q_lower)
            if m: return self.load_data(m.group())

        if self.df is None:
            return "⚠️ Pehle file load kijiye (e.g. load test.xlsx)"

        try:
            if self.agent_executor:
                # Unlimited Query Execution
                response = self.agent_executor.invoke(q)
                return response['output']
            return "❌ Agent brain is offline."
        except Exception as e:
            # Agar Gemini-1.5-Flash na chale, toh error message thoda helpful rakha hai
            return f"💡 AI thinking: API connectivity issue ho sakta hai. Error: {str(e)}"
