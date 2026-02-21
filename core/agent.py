import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.database import SessionLocal, UserQuery
from core.ml import MLModels

# ✅ LangChain Imports (Optional)
try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents import create_pandas_dataframe_agent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.last_file = None
        self.ml = MLModels()
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # ✅ Tableau-style color palettes
        self.color_palettes = {
            'default': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            'business': ['#2c3e50', '#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
            'modern': ['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#95E1D3']
        }
        
        # ✅ Initialize AI if available
        if LLM_AVAILABLE and self.api_key:
            try:
                llm = ChatGroq(
                    temperature=0.5,
                    groq_api_key=self.api_key,
                    model_name="llama-3.1-8b-instant",
                    max_tokens=500,
                    timeout=15
                )
                print("✅ AI Mode: ON")
            except Exception as e:
                print(f"⚠️ AI Mode: OFF (API Key issue: {e})")
    
    def load_data(self, fp):
        try:
            self.df = pd.read_csv(fp, encoding='latin1')
            self.last_file = fp
            num = self.df.select_dtypes('number').columns.tolist()
            cat = self.df.select_dtypes('object').columns.tolist()
            
            # ✅ Initialize AI Agent with data
            if LLM_AVAILABLE and self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0.5,
                        groq_api_key=self.api_key,
                        model_name="llama-3.1-8b-instant",
                        max_tokens=500,
                        timeout=15
                    )
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm, self.df, verbose=False, allow_dangerous_code=True
                    )
                except Exception as e:
                    print(f"⚠️ AI Agent init failed: {e}")
            
            return f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\nNumeric: {num[:3]}\nCategorical: {cat[:3]}"
        except Exception as e:
            return f"❌ Error loading {fp}: {str(e)}"
    
    # ✅ HEADMASTER: Auto-Chart Recommendation
    def recommend_chart(self, query=""):
        if self.df is None:
            return "⚠️ Load data first"
        
        num_cols = self.df.select_dtypes('number').columns.tolist()
        cat_cols = self.df.select_dtypes('object').columns.tolist()
        
        recommendations = []
        
        # Time series detection
        if any('date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() for col in self.df.columns):
            recommendations.append("📈 Line Chart (Time Series)")
        
        # Comparison
        if len(cat_cols) > 0 and len(num_cols) > 0:
            recommendations.append("📊 Bar Chart (Comparison)")
        
        # Distribution
        if len(num_cols) > 0:
            recommendations.append("📉 Histogram (Distribution)")
            recommendations.append("📦 Box Plot (Outliers)")
        
        # Relationship
        if len(num_cols) >= 2:
            recommendations.append("🔗 Scatter Plot (Relationship)")
            recommendations.append("🔥 Heatmap (Correlation)")
        
        # Composition
        if len(cat_cols) > 0:
            recommendations.append("🥧 Pie Chart (Composition)")
        
        return "💡 **Recommended Charts:**\n" + "\n".join(recommendations[:5])
    
    # ✅ HEADMASTER: Advanced Chart Types
    def generate_plot(self, plot_type="bar", column=None, groupby=None, color_scheme="default"):
        if self.df is None: 
            return "⚠️ Pehle file upload karo!"
        
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            colors = self.color_palettes.get(color_scheme, self.color_palettes['default'])
            
            # Column selection
            if column and column in self.df.columns:
                target_col = column
            elif num_cols:
                target_col = num_cols[0]
            else:
                return "⚠️ No numeric columns for plotting"
            
            os.makedirs('static', exist_ok=True)
            
            # ✅ BAR CHART
            if "bar" in plot_type.lower():
                if cat_cols:
                    col = cat_cols[0]
                    counts = self.df[col].value_counts().head(10)
                    ax.bar(range(len(counts)), counts.values, color=colors[0])
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha='right')
                    ax.set_title(f'Top 10 {col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Count')
                elif groupby and groupby in self.df.columns:
                    grouped = self.df.groupby(groupby)[target_col].sum().nlargest(10)
                    ax.bar(range(len(grouped)), grouped.values, color=colors)
                    ax.set_xticks(range(len(grouped)))
                    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                    ax.set_title(f'{target_col} by {groupby}', fontsize=14, fontweight='bold')
            
            # ✅ LINE CHART (Time Series)
            elif "line" in plot_type.lower():
                date_col = next((c for c in self.df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                if date_col:
                    self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                    self.df = self.df.sort_values(date_col)
                    ax.plot(self.df[date_col], self.df[target_col], color=colors[0], linewidth=2, marker='o')
                    ax.set_title(f'{target_col} Over Time', fontsize=14, fontweight='bold')
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(target_col)
                    plt.xticks(rotation=45)
                else:
                    ax.plot(self.df[target_col].values, color=colors[0], linewidth=2, marker='o')
                    ax.set_title(f'{target_col} Trend', fontsize=14, fontweight='bold')
            
            # ✅ PIE CHART
            elif "pie" in plot_type.lower():
                if cat_cols:
                    col = cat_cols[0]
                    counts = self.df[col].value_counts().head(8)
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            
            # ✅ HISTOGRAM
            elif "hist" in plot_type.lower() or "distribution" in plot_type.lower():
                ax.hist(self.df[target_col].dropna(), bins=20, color=colors[0], edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
            
            # ✅ BOX PLOT
            elif "box" in plot_type.lower():
                if cat_cols and len(num_cols) > 0:
                    col = cat_cols[0]
                    top_categories = self.df[col].value_counts().head(5).index
                    filtered_df = self.df[self.df[col].isin(top_categories)]
                    data_to_plot = [filtered_df[filtered_df[col] == cat][target_col].dropna().values for cat in top_categories]
                    ax.boxplot(data_to_plot, labels=top_categories, patch_artist=True,
                              boxprops=dict(facecolor=colors[0], color=colors[0]),
                              medianprops=dict(color='white'))
                    ax.set_title(f'{target_col} by {col}', fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45)
                else:
                    ax.boxplot(self.df[target_col].dropna(), patch_artist=True,
                              boxprops=dict(facecolor=colors[0], color=colors[0]))
                    ax.set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
            
            # ✅ SCATTER PLOT
            elif "scatter" in plot_type.lower():
                if len(num_cols) >= 2:
                    ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], 
                              alpha=0.6, color=colors[0], s=50)
                    ax.set_xlabel(num_cols[0])
                    ax.set_ylabel(num_cols[1])
                    ax.set_title(f'{num_cols[0]} vs {num_cols[1]}', fontsize=14, fontweight='bold')
                    
                    # Add correlation coefficient
                    corr = self.df[num_cols[0]].corr(self.df[num_cols[1]])
                    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # ✅ AREA CHART
            elif "area" in plot_type.lower():
                if len(num_cols) >= 2:
                    self.df[num_cols[:5]].plot.area(ax=ax, alpha=0.7, color=colors[:5])
                    ax.set_title('Area Chart - Multiple Variables', fontsize=14, fontweight='bold')
                else:
                    ax.fill_between(range(len(self.df)), self.df[target_col].values, alpha=0.5, color=colors[0])
                    ax.set_title(f'{target_col} Area Chart', fontsize=14, fontweight='bold')
            
            # ✅ HEATMAP (Correlation)
            elif "heatmap" in plot_type.lower() or "correlation" in plot_type.lower():
                if len(num_cols) >= 2:
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_matrix = self.df[num_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, linewidths=1, ax=ax, fmt='.2f')
                    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('static/plot.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"✅ {plot_type.title()} Chart saved! View at: /plot.png"
        except Exception as e:
            return f"❌ Plot error: {str(e)}"
    
    # ✅ HEADMASTER: Dashboard (Multiple Plots)
    def generate_dashboard(self):
        if self.df is None:
            return "⚠️ Load data first"
        
        try:
            os.makedirs('static', exist_ok=True)
            
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            # Create dashboard with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Data Science Dashboard', fontsize=16, fontweight='bold')
            
            # Top-Left: Bar Chart
            if cat_cols:
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(8)
                axes[0, 0].bar(range(len(counts)), counts.values, color='#667eea')
                axes[0, 0].set_xticks(range(len(counts)))
                axes[0, 0].set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
                axes[0, 0].set_title(f'Top {col}', fontsize=12, fontweight='bold')
            
            # Top-Right: Histogram
            if num_cols:
                col = num_cols[0]
                axes[0, 1].hist(self.df[col].dropna(), bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
                axes[0, 1].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            
            # Bottom-Left: Box Plot
            if num_cols and len(num_cols) > 1:
                data = [self.df[col].dropna().values for col in num_cols[:4]]
                axes[1, 0].boxplot(data, labels=num_cols[:4], patch_artist=True,
                                  boxprops=dict(facecolor='#FF6B6B', color='#FF6B6B'))
                axes[1, 0].set_title('Variable Comparison', fontsize=12, fontweight='bold')
                plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Bottom-Right: Correlation Heatmap
            if len(num_cols) >= 2:
                corr_matrix = self.df[num_cols[:6]].corr()
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig('static/dashboard.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return "✅ Dashboard saved! View at: /dashboard.png"
        except Exception as e:
            return f"❌ Dashboard error: {str(e)}"
    
    # ✅ HEADMASTER: KPI Cards
    def get_kpi_cards(self):
        if self.df is None:
            return "⚠️ Load data first"
        
        num_cols = self.df.select_dtypes('number').columns.tolist()
        cat_cols = self.df.select_dtypes('object').columns.tolist()
        
        kpis = []
        
        # Total Records
        kpis.append(f"📊 **Total Records:** {len(self.df):,}")
        
        # Total Columns
        kpis.append(f"📋 **Total Columns:** {len(self.df.columns)}")
        
        # Key Metric (First numeric column)
        if num_cols:
            col = num_cols[0]
            kpis.append(f"💰 **Total {col}:** ₹{self.df[col].sum():,.2f}")
            kpis.append(f"📈 **Avg {col}:** ₹{self.df[col].mean():,.2f}")
        
        # Missing Values
        missing = int(self.df.isnull().sum().sum())
        kpis.append(f"⚠️ **Missing Values:** {missing} ({missing/len(self.df)*100:.1f}%)")
        
        return "\n".join(kpis)
    
    # ✅ HEADMASTER: Export to Excel
    def export_to_excel(self, filename="analysis_report.xlsx"):
        if self.df is None:
            return "⚠️ Load data first"
        
        try:
            os.makedirs('static', exist_ok=True)
            filepath = f"static/{filename}"
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Raw Data
                self.df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Summary Statistics
                summary = self.df.describe()
                summary.to_excel(writer, sheet_name='Summary Stats')
                
                # Missing Values
                missing = self.df.isnull().sum()
                missing_df = pd.DataFrame({'Column': missing.index, 'Missing Count': missing.values})
                missing_df.to_excel(writer, sheet_name='Missing Values', index=False)
                
                # Correlation
                num_cols = self.df.select_dtypes('number').columns.tolist()
                if len(num_cols) >= 2:
                    corr = self.df[num_cols].corr()
                    corr.to_excel(writer, sheet_name='Correlation')
            
            return f"✅ Excel report saved: {filepath}"
        except Exception as e:
            return f"❌ Export error: {str(e)}"
    
    # ... (rest of the methods remain same: show_info, top_n, group_by, etc.)
    # ... (query method remains same with Python code execution support)
    
    def query(self, q, user_id=None):
        q_original = q
        q = q.lower().strip()
        
        # ✅ DETECT PYTHON CODE
        python_patterns = [
            r'df\.', r'pd\.', r'np\.', r'plt\.',
            r'\.head\(', r'\.tail\(', r'\.mean\(', r'\.sum\(',
            r'=', r'import',
        ]
        
        is_python_code = any(re.search(pattern, q) for pattern in python_patterns)
        
        if is_python_code and self.df is not None:
            return self.execute_python_code(q_original, user_id)
        
        # ✅ GREETINGS
        if q in ['hi', 'hello', 'hey', 'hii', 'namaste']:
            return "👋 Hello! Main aapka Data Scientist Agent hoon. CSV upload karo aur kuch bhi pucho!"
        
        # ✅ HELP
        if 'help' in q or 'kya kar' in q or 'what can' in q:
            return ("💡 **Natural Language Commands:**\n"
                   "• 'top 5 by revenue'\n"
                   "• 'group by region'\n"
                   "• 'predict trend'\n"
                   "• 'create bar chart'\n"
                   "• 'show dashboard'\n"
                   "• 'export to excel'\n"
                   "• 'recommend chart'\n\n"
                   "💻 **Python Code (Advanced):**\n"
                   "• `df.head()`\n"
                   "• `df['column'].mean()`\n"
                   "• `df.nlargest(5, 'revenue')`\n")
        
        # ✅ HEADMASTER: Dashboard Command
        if 'dashboard' in q or 'multiple plots' in q or 'all charts' in q:
            return self.generate_dashboard()
        
        # ✅ HEADMASTER: KPI Cards
        if 'kpi' in q or 'metrics' in q or 'summary cards' in q:
            return self.get_kpi_cards()
        
        # ✅ HEADMASTER: Export to Excel
        if 'export' in q or 'excel' in q or 'download data' in q:
            return self.export_to_excel()
        
        # ✅ HEADMASTER: Chart Recommendation
        if 'recommend' in q or 'suggest chart' in q or 'best chart' in q:
            return self.recommend_chart(q_original)
        
        # ✅ HEADMASTER: Advanced Chart Types
        if 'line chart' in q or 'time series' in q or 'trend line' in q:
            return self.generate_plot('line')
        
        if 'pie chart' in q or 'composition' in q:
            return self.generate_plot('pie')
        
        if 'box plot' in q or 'boxplot' in q or 'outliers' in q:
            return self.generate_plot('box')
        
        if 'area chart' in q:
            return self.generate_plot('area')
        
        # ... (rest of query method remains same)
        
        return "💡 Try: 'dashboard', 'kpi', 'export excel', 'line chart', 'pie chart', 'box plot'"
    
    def execute_python_code(self, code, user_id=None):
        """Execute raw Python code safely"""
        try:
            safe_globals = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'df': self.df,
                '__builtins__': {}
            }
            
            result = eval(code, safe_globals, {})
            output = f"✅ Python Code Executed:\n```python\n{code}\n```\n\n**Output:**\n{result}"
            
            if user_id:
                try:
                    db = SessionLocal()
                    db.add(UserQuery(query_text=f"CODE: {code}", response_text=output, user_id=user_id))
                    db.commit()
                    db.close()
                except:
                    pass
            
            return output
        except Exception as e:
            error_msg = f"❌ Code Error: {str(e)}"
            if self.df is not None:
                error_msg += f"\n\n💡 **Available Columns:** {list(self.df.columns)}"
            return error_msg
