"""
Data Science AI Agent
An autonomous agent that performs data analysis via natural language queries.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
from typing import Optional, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

warnings.filterwarnings('ignore')

class DataScienceAgent:
    """AI Agent that performs data science tasks via natural language."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            api_key: Gemini API key (optional - will read from GOOGLE_API_KEY env var)
        """
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.df: Optional[pd.DataFrame] = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=2048
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Create tools for the agent."""
        
        @tool
        def load_data(file_path: str) -> str:
            """Load CSV/Excel file into memory."""
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    return f"✗ Unsupported file format: {file_path}"
                
                return (f"✓ Successfully loaded {self.df.shape[0]} rows × {self.df.shape[1]} columns\n"
                       f"Columns: {list(self.df.columns)}")
            except Exception as e:
                return f"✗ Error loading file: {str(e)}"
        
        @tool
        def show_basic_info(dummy: str) -> str:
            """Show dataset statistics."""
            if self.df is None:
                return "⚠ No data loaded. Use 'load_data' first."
            
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_str = buffer.getvalue()
            
            missing = self.df.isnull().sum()
            missing_str = missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values"
            
            return (f"Shape: {self.df.shape}\n\n"
                   f"Column Types:\n{self.df.dtypes.to_string()}\n\n"
                   f"Missing Values:\n{missing_str}")
        
        @tool
        def generate_visualization(plot_type: str) -> str:
            """Generate visualization based on plot type."""
            if self.df is None:
                return "⚠ No data loaded. Use 'load_data' first."
            
            try:
                plt.figure(figsize=(10, 6))
                
                if plot_type.lower() == "histogram":
                    num_cols = self.df.select_dtypes(include='number').columns
                    if len(num_cols) == 0:
                        return "⚠ No numeric columns found for histogram."
                    col = num_cols[0]
                    sns.histplot(self.df[col], kde=True)
                    plt.title(f'Histogram of {col}')
                    plt.xlabel(col)
                
                elif plot_type.lower() == "bar":
                    cat_cols = self.df.select_dtypes(include='object').columns
                    if len(cat_cols) == 0:
                        return "⚠ No categorical columns found for bar chart."
                    col = cat_cols[0]
                    counts = self.df[col].value_counts().head(10)
                    counts.plot(kind='bar')
                    plt.title(f'Top 10 Categories in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                
                elif plot_type.lower() == "scatter":
                    num_cols = self.df.select_dtypes(include='number').columns
                    if len(num_cols) < 2:
                        return "⚠ Need at least 2 numeric columns for scatter plot."
                    sns.scatterplot(data=self.df, x=num_cols[0], y=num_cols[1])
                    plt.title(f'Scatter Plot: {num_cols[0]} vs {num_cols[1]}')
                    plt.xlabel(num_cols[0])
                    plt.ylabel(num_cols[1])
                
                elif plot_type.lower() == "box":
                    num_cols = self.df.select_dtypes(include='number').columns
                    if len(num_cols) == 0:
                        return "⚠ No numeric columns found for box plot."
                    col = num_cols[0]
                    sns.boxplot(data=self.df, y=col)
                    plt.title(f'Box Plot of {col}')
                    plt.ylabel(col)
                
                else:
                    return f"⚠ Unsupported plot type: {plot_type}. Supported: histogram, bar, scatter, box"
                
                plt.tight_layout()
                plt.savefig('plot.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                return "✓ Plot saved as 'plot.png' in current directory"
            
            except Exception as e:
                return f"✗ Error generating plot: {str(e)}"
        
        @tool
        def run_custom_code(python_code: str) -> str:
            """Execute safe pandas operations."""
            if self.df is None:
                return "⚠ No data loaded. Use 'load_data' first."
            
            try:
                # Security note: For production, use proper sandboxing
                local_vars = {'df': self.df, 'pd': pd}
                exec(f"result = {python_code}", {"__builtins__": {}}, local_vars)
                output = str(local_vars.get('result', 'No result returned'))
                return f"✓ Result:\n{output[:1000]}"  # Limit output size
            except Exception as e:
                return f"✗ Error executing code: {str(e)}"
        
        return [
            Tool(
                name="Load Data",
                func=load_data,
                description="Load CSV/Excel file. Input: file path (e.g., 'sales.csv')"
            ),
            Tool(
                name="Show Info",
                func=show_basic_info,
                description="Show dataset statistics (shape, dtypes, missing values). Input: any string"
            ),
            Tool(
                name="Visualize",
                func=generate_visualization,
                description="Create plot. Input: 'histogram', 'bar', 'scatter', or 'box'"
            ),
            Tool(
                name="Custom Code",
                func=run_custom_code,
                description="Run pandas operation. Input: pandas expression like 'df.head()' or 'df.groupby(\"category\").sum()'"
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent."""
        template = """You are an expert data scientist assistant. Help users analyze datasets using the available tools.

Available tools:
{tools}

Use this format:
Question: the input question
Thought: your reasoning
Action: tool name (exactly as shown above)
Action Input: input for the tool
Observation: tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: concise, actionable insight

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=7
        )
    
    def query(self, user_query: str) -> str:
        """
        Process user query and return result.
        
        Args:
            user_query: Natural language query
            
        Returns:
            Agent response
        """
        # Auto-detect and load CSV if mentioned in query
        if "load" in user_query.lower() and ".csv" in user_query.lower():
            match = re.search(r'[\w\-.]+\.csv', user_query)
            if match:
                filename = match.group()
                print(f"📥 Auto-loading: {filename}")
                result = self.tools[0].func(filename)
                print(result)
                print("-" * 50)
        
        try:
            response = self.agent.invoke({"input": user_query})
            return response['output']
        except Exception as e:
            return f"⚠ Agent error: {str(e)}"


# ======================
# CLI INTERFACE
# ======================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 DATA SCIENCE AI AGENT")
    print("=" * 70)
    print("\nCommands you can try:")
    print("  • 'load sales.csv'")
    print("  • 'show basic info'")
    print("  • 'top 3 products by revenue'")
    print("  • 'create bar chart'")
    print("  • 'average price by region'")
    print("\n💡 Tip: Place your CSV files in the same directory as this script")
    print("=" * 70 + "\n")
    
    # Initialize agent
    try:
        agent = DataScienceAgent()
        print("✅ Agent initialized successfully!\n")
    except ValueError as e:
        print(f"❌ {e}")
        print("\n🔧 Fix: Set your Gemini API key:")
        print("   Option 1: export GOOGLE_API_KEY='your_key_here' (Linux/Mac)")
        print("   Option 2: set GOOGLE_API_KEY='your_key_here' (Windows)")
        print("   Option 3: Pass key when creating agent: DataScienceAgent(api_key='your_key')")
        exit(1)
    
    # Interactive loop
    while True:
        try:
            query = input("\n❓ Your query (or 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Happy analyzing!")
                break
            
            if not query:
                continue
            
            print("\n⏳ Processing...")
            result = agent.query(query)
            print(f"\n💡 {result}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
