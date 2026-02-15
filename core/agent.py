import os
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        
        # Render ke 'Environment Variables' se API Key uthana
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            # Gemini 1.5 Flash: Fast and Smart
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0
            )
        else:
            self.llm = None
            print("⚠️ Error: GOOGLE_API_KEY nahi mili!")

    def load_data(self, fp):
        try:
            ext = os.path.splitext(fp)[-1].lower()
            
            # Excel aur CSV dono ke liye support
            if ext == '.csv':
                try:
                    self.df = pd.read_csv(fp, encoding='utf-8')
                except:
                    self.df = pd.read_csv(fp, encoding='latin1')
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(fp)
            else:
                return f"❌ Format {ext} support nahi hai. Sirf CSV/Excel chahiyen."

            if self.llm is not None:
                # Master Brain: Jo user ki query se khud code likhega
                self.agent_executor = create_pandas_dataframe_agent(
                    self.llm, 
                    self.df, 
                    verbose=True, 
                    allow_dangerous_code=True # Python logic execution ke liye
                )
                return f"✅ Master Agent Live: {fp} load ho gayi hai. Ab kuch bhi poocho!"
            else:
                return "❌ AI Model ready nahi hai. API Key check karein."
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def query(self, q):
        q_lower = q.lower().strip()
        
        # 1. Load command logic (Bina data ke bhi chalna chahiye)
        if "load" in q_lower:
            m = re.search(r'[\w\-.]+\.(csv|xlsx|xls)', q_lower)
            if m:
                return self.load_data(m.group())
            return "❌ Filename correct likhein (e.g. load data.xlsx)"

        # 2. Data check
        if self.df is None:
            return "⚠️ Pehle file load karein (Type: load filename.xlsx)"

        # 3. AI Execution (Unlimited Queries)
        try:
            if self.agent_executor:
                # Gemini aapki query ko solve karega
                response = self.agent_executor.invoke(q)
                return response['output']
            else:
                return "❌ Agent Brain initialized nahi hai."
        except Exception as e:
            return f"💡 AI is trying to understand... Error: {str(e)}"
