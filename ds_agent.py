import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import warnings
warnings.filterwarnings('ignore')

# ===== SETUP: Add your Gemini API key here =====
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"  # https://aistudio.google.com/app/apikey

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=2048
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        @tool
        def load_data(file_path: str) -> str:
            """Load CSV/Excel file into memory"""
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)
                return f"✓ Loaded {self.df.shape[0]} rows, {self.df.shape[1]} columns\nColumns: {list(self.df.columns)}"
            except Exception as e:
                return f"✗ Error: {str(e)}"
        
        @tool
        def show_basic_info(dummy: str) -> str:
            """Show dataset shape, columns, dtypes, missing values"""
            if self.df is None:
                return "⚠ No data loaded. Use 'load_data' first."
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info = buffer.getvalue()
            missing = self.df.isnull().sum()
            return f"Shape: {self.df.shape}\n\nColumns & Types:\n{self.df.dtypes}\n\nMissing Values:\n{missing[missing>0]}"
        
        @tool
        def generate_visualization(plot_type: str) -> str:
            """Create plot: 'histogram', 'scatter', 'bar', 'box'"""
            if self.df is None:
                return "⚠ No data loaded."
            try:
                plt.figure(figsize=(10,6))
                if plot_type == "histogram" and self.df.select_dtypes(include='number').shape[1] > 0:
                    num_col = self.df.select_dtypes(include='number').columns[0]
                    sns.histplot(self.df[num_col], kde=True)
                    plt.title(f'Histogram of {num_col}')
                elif plot_type == "scatter" and self.df.select_dtypes(include='number').shape[1] >= 2:
                    cols = self.df.select_dtypes(include='number').columns[:2]
                    sns.scatterplot(data=self.df, x=cols[0], y=cols[1])
                    plt.title(f'Scatter: {cols[0]} vs {cols[1]}')
                elif plot_type == "bar" and self.df.select_dtypes(exclude='number').shape[1] > 0:
                    cat_col = self.df.select_dtypes(exclude='number').columns[0]
                    self.df[cat_col].value_counts().head(10).plot(kind='bar')
                    plt.title(f'Top categories in {cat_col}')
                else:
                    return "⚠ Cannot generate this plot type with current data."
                plt.tight_layout()
                plt.savefig('plot.png')
                plt.close()
                return "✓ Plot saved as 'plot.png' – check Files tab in Colab"
            except Exception as e:
                return f"✗ Plot error: {str(e)}"
        
        @tool
        def run_custom_code(python_code: str) -> str:
            """Execute safe pandas operations (e.g., 'df.groupby("category").sum()')"""
            if self.df is None:
                return "⚠ No data loaded."
            try:
                # Security note: In production, use proper sandboxing
                local_vars = {'df': self.df, 'pd': pd}
                exec(f"result = {python_code}", globals(), local_vars)
                output = str(local_vars['result'])
                return f"✓ Result:\n{output[:1000]}"  # Limit output size
            except Exception as e:
                return f"✗ Code error: {str(e)}"
        
        return [
            Tool(name="Load Data", func=load_data, description="Load CSV/Excel file. Input: file path"),
            Tool(name="Show Info", func=show_basic_info, description="Show dataset stats. Input: 'any'"),
            Tool(name="Visualize", func=generate_visualization, description="Create plot. Input: 'histogram', 'scatter', 'bar', or 'box'"),
            Tool(name="Custom Code", func=run_custom_code, description="Run pandas code. Input: pandas expression like 'df.head()' or 'df.groupby(\"col\").mean()'")
        ]
    
    def _create_agent(self):
        template = """You are an expert data scientist assistant. Help the user analyze datasets.
Available tools:
{tools}

Use this format:
Question: the input question
Thought: your reasoning
Action: tool name
Action Input: input for tool
Observation: tool result
... (repeat if needed)
Final Answer: concise summary with insights

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
    
    def query(self, user_query: str):
        """Main interface – user gives query, agent responds"""
        if "load" in user_query.lower() and ".csv" in user_query.lower():
            # Auto-extract filename
            import re
            match = re.search(r'[\w\-.]+\.csv', user_query)
            if match:
                filename = match.group()
                print(f"📥 Auto-loading: {filename}")
                print(self.tools[0].func(filename))
                print("\n" + "="*50)
        
        response = self.agent.invoke({"input": user_query})
        print("\n💡 FINAL ANSWER:")
        print(response['output'])

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    agent = DataScienceAgent()
    
    print("="*60)
    print("🤖 DATA SCIENCE AI AGENT READY")
    print("Commands you can try:")
    print("  • 'load sales.csv'")
    print("  • 'show me dataset info'")
    print("  • 'create histogram'")
    print("  • 'what are top 3 categories by sales?'")
    print("="*60 + "\n")
    
    # Example queries (uncomment to test)
    # agent.query("load sample_data.csv")
    # agent.query("show basic statistics")
    # agent.query("create a bar chart of categories")
