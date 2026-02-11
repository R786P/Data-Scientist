hereimport os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

class DataScienceAgent:
    def __init__(self, api_key=None):
        if api_key: os.environ["GOOGLE_API_KEY"] = api_key
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Set GOOGLE_API_KEY in Render Environment Variables")
        self.df = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        @tool
        def load_data(fp): 
            try:
                self.df = pd.read_csv(fp) if fp.endswith('.csv') else pd.read_excel(fp)
                return f"✓ Loaded {self.df.shape[0]} rows"
            except Exception as e: return f"✗ Error: {str(e)}"
        
        @tool
        def show_info(_): 
            if self.df is None: return "⚠ Load data first"
            return f"Shape: {self.df.shape}"
        
        @tool
        def visualize(pt): 
            if self.df is None: return "⚠ Load data first"
            try:
                plt.figure(figsize=(8,5))
                if pt=="histogram" and (nc:=self.df.select_dtypes('number').columns).size>0:
                    self.df[nc[0]].hist(); plt.title(f'Histogram: {nc[0]}')
                elif pt=="bar" and (cc:=self.df.select_dtypes('object').columns).size>0:
                    self.df[cc[0]].value_counts().head(5).plot(kind='bar'); plt.title(f'Top: {cc[0]}')
                else: return "⚠ Can't create plot"
                plt.tight_layout(); plt.savefig('plot.png'); plt.close()
                return "✓ Plot saved"
            except Exception as e: return f"✗ Plot error: {str(e)}"
        
        @tool
        def predict_trend(col): 
            if self.df is None: return "⚠ Load data first"
            try:
                cols = [c for c in self.df.columns if col.lower() in c.lower()]
                if not cols: 
                    num_cols = self.df.select_dtypes('number').columns.tolist()
                    if not num_cols: return "⚠ No numeric columns"
                    target = num_cols[0]
                else:
                    target = cols[0]
                series = self.df[target].dropna()
                if len(series) < 5: return f"⚠ Need min 5 values"
                last3 = series.iloc[-3:].values
                trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
                next_val = series.iloc[-1] + (series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else series.iloc[-1]
                return f"📈 '{target}': {trend}\nNext: ₹{next_val:,.0f}"
            except Exception as e: return f"⚠ Prediction error: {str(e)}"
        
        @tool
        def segment_customers(_): 
            if self.df is None: return "⚠ Load data first"
            try:
                rev_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total'])), None)
                if not rev_col: return "⚠ Need revenue column"
                q25 = self.df[rev_col].quantile(0.25)
                q75 = self.df[rev_col].quantile(0.75)
                high = self.df[self.df[rev_col] > q75]
                medium = self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)]
                low = self.df[self.df[rev_col] < q25]
                total = len(self.df)
                return (f"👥 Segments:\nHigh (>₹{q75:,.0f}): {len(high)} ({len(high)/total*100:.0f}%)\n"
                       f"Medium: {len(medium)} ({len(medium)/total*100:.0f}%)\n"
                       f"Low (<₹{q25:,.0f}): {len(low)} ({len(low)/total*100:.0f}%)")
            except Exception as e: return f"⚠ Segmentation error: {str(e)}"
        
        @tool
        def detect_outliers(col): 
            if self.df is None: return "⚠ Load data first"
            try:
                cols = [c for c in self.df.columns if col.lower() in c.lower()]
                if not cols: 
                    num_cols = self.df.select_dtypes('number').columns.tolist()
                    if not num_cols: return "⚠ No numeric columns"
                    target = num_cols[0]
                else:
                    target = cols[0]
                series = self.df[target].dropna()
                if len(series) < 4: return f"⚠ Need min 4 values"
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = series[(series < lower) | (series > upper)]
                if len(outliers) == 0: return f"✅ No outliers"
                return f"⚠️ Outliers: {len(outliers)} values outside ₹{lower:,.0f}-₹{upper:,.0f}"
            except Exception as e: return f"⚠ Outlier error: {str(e)}"
        
        @tool
        def custom_code(code): 
            if self.df is None: return "⚠ Load data first"
            try:
                loc = {'df': self.df, 'pd': pd, 'np': np}; exec(f"result = {code}", {}, loc)
                return f"✓ {str(loc['result'])[:500]}"
            except Exception as e: return f"✗ Error: {str(e)}"
        
        return [
            Tool(name="Load Data", func=load_data, description="Load CSV. Input: filename"),
            Tool(name="Show Info", func=show_info, description="Dataset stats"),
            Tool(name="Visualize", func=visualize, description="Plot: 'histogram' or 'bar'"),
            Tool(name="Predict Trend", func=predict_trend, description="Forecast trend. Input: column name"),
            Tool(name="Segment Customers", func=segment_customers, description="3-tier segmentation. Input: 'any'"),
            Tool(name="Detect Outliers", func=detect_outliers, description="Find anomalies. Input: column name"),
            Tool(name="Custom Code", func=custom_code, description="Pandas/numpy code")
        ]
    
    def _create_agent(self):
        prompt = PromptTemplate.from_template("You are expert data scientist. Tools:\n{tools}\nQuestion: {input}\n{agent_scratchpad}")
        return AgentExecutor(agent=create_react_agent(self.llm, self.tools, prompt), tools=self.tools, verbose=False, handle_parsing_errors=True)
    
    def query(self, q):
        if "load" in q.lower() and ".csv" in q.lower():
            if m:=re.search(r'[\w\-.]+\.csv', q): 
                print(f"📥 Loading: {m.group()}"); print(self.tools[0].func(m.group()))
        try:
            return self.agent.invoke({"input": q})['output']
        except Exception as e:
            return f"⚠ Error: {str(e)}"
