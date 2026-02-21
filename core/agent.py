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
            
            # ✅ Store column info for better suggestions
            self.available_columns = list(self.df.columns)
            self.numeric_columns = num
            self.categorical_columns = cat
            
            return f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\nNumeric: {num[:3]}\nCategorical: {cat[:3]}"
        except Exception as e:
            return f"❌ Error loading {fp}: {str(e)}"
    
    # ✅ ENHANCED: Better column matching
    def find_column(self, search_term):
        """Find best matching column name"""
        if self.df is None:
            return None
        
        search_lower = search_term.lower()
        
        # Exact match
        for col in self.df.columns:
            if search_lower in col.lower() or col.lower() in search_lower:
                return col
        
        # Partial match
        for col in self.df.columns:
            if any(word in col.lower() for word in search_lower.split()):
                return col
        
        return None
    
    # ✅ ENHANCED: Better error messages with suggestions
    def get_column_suggestions(self, search_term):
        """Suggest available columns"""
        if self.df is None:
            return "⚠️ No data loaded"
        
        suggestions = f"💡 **Available Columns:** {', '.join(self.available_columns)}"
        
        if self.numeric_columns:
            suggestions += f"\n💰 **Numeric Columns:** {', '.join(self.numeric_columns)}"
        
        if self.categorical_columns:
            suggestions += f"\n📋 **Text Columns:** {', '.join(self.categorical_columns)}"
        
        return suggestions
    
    def recommend_chart(self, query=""):
        if self.df is None:
            return "⚠️ Load data first"
        
        num_cols = self.df.select_dtypes('number').columns.tolist()
        cat_cols = self.df.select_dtypes('object').columns.tolist()
        
        recommendations = []
        
        if any('date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() for col in self.df.columns):
            recommendations.append("📈 Line Chart (Time Series)")
        
        if len(cat_cols) > 0 and len(num_cols) > 0:
            recommendations.append("📊 Bar Chart (Comparison)")
        
        if len(num_cols) > 0:
            recommendations.append("📉 Histogram (Distribution)")
            recommendations.append("📦 Box Plot (Outliers)")
        
        if len(num_cols) >= 2:
            recommendations.append("🔗 Scatter Plot (Relationship)")
            recommendations.append("🔥 Heatmap (Correlation)")
        
        if len(cat_cols) > 0:
            recommendations.append("🥧 Pie Chart (Composition)")
        
        return "💡 **Recommended Charts:**\n" + "\n".join(recommendations[:5])
    
    def generate_plot(self, plot_type="bar", column=None, groupby=None, color_scheme="default"):
        if self.df is None: 
            return "⚠️ Pehle file upload karo!"
        
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            colors = self.color_palettes.get(color_scheme, self.color_palettes['default'])
            
            if column and column in self.df.columns:
                target_col = column
            elif num_cols:
                target_col = num_cols[0]
            else:
                return "⚠️ No numeric columns for plotting"
            
            os.makedirs('static', exist_ok=True)
            
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
            
            elif "pie" in plot_type.lower():
                if cat_cols:
                    col = cat_cols[0]
                    counts = self.df[col].value_counts().head(8)
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            
            elif "hist" in plot_type.lower() or "distribution" in plot_type.lower():
                ax.hist(self.df[target_col].dropna(), bins=20, color=colors[0], edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
            
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
            
            elif "scatter" in plot_type.lower():
                if len(num_cols) >= 2:
                    ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], 
                              alpha=0.6, color=colors[0], s=50)
                    ax.set_xlabel(num_cols[0])
                    ax.set_ylabel(num_cols[1])
                    ax.set_title(f'{num_cols[0]} vs {num_cols[1]}', fontsize=14, fontweight='bold')
                    
                    corr = self.df[num_cols[0]].corr(self.df[num_cols[1]])
                    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            elif "area" in plot_type.lower():
                if len(num_cols) >= 2:
                    self.df[num_cols[:5]].plot.area(ax=ax, alpha=0.7, color=colors[:5])
                    ax.set_title('Area Chart - Multiple Variables', fontsize=14, fontweight='bold')
                else:
                    ax.fill_between(range(len(self.df)), self.df[target_col].values, alpha=0.5, color=colors[0])
                    ax.set_title(f'{target_col} Area Chart', fontsize=14, fontweight='bold')
            
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
    
    def generate_dashboard(self):
        if self.df is None:
            return "⚠️ Load data first"
        
        try:
            os.makedirs('static', exist_ok=True)
            
            num_cols = self.df.select_dtypes('number').columns.tolist()
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Data Science Dashboard', fontsize=16, fontweight='bold')
            
            if cat_cols:
                col = cat_cols[0]
                counts = self.df[col].value_counts().head(8)
                axes[0, 0].bar(range(len(counts)), counts.values, color='#667eea')
                axes[0, 0].set_xticks(range(len(counts)))
                axes[0, 0].set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
                axes[0, 0].set_title(f'Top {col}', fontsize=12, fontweight='bold')
            
            if num_cols:
                col = num_cols[0]
                axes[0, 1].hist(self.df[col].dropna(), bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
                axes[0, 1].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            
            if num_cols and len(num_cols) > 1:
                data = [self.df[col].dropna().values for col in num_cols[:4]]
                axes[1, 0].boxplot(data, labels=num_cols[:4], patch_artist=True,
                                  boxprops=dict(facecolor='#FF6B6B', color='#FF6B6B'))
                axes[1, 0].set_title('Variable Comparison', fontsize=12, fontweight='bold')
                plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
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
    
    def get_kpi_cards(self):
        if self.df is None:
            return "⚠️ Load data first"
        
        num_cols = self.df.select_dtypes('number').columns.tolist()
        cat_cols = self.df.select_dtypes('object').columns.tolist()
        
        kpis = []
        
        kpis.append(f"📊 **Total Records:** {len(self.df):,}")
        kpis.append(f"📋 **Total Columns:** {len(self.df.columns)}")
        
        if num_cols:
            col = num_cols[0]
            kpis.append(f"💰 **Total {col}:** {self.df[col].sum():,.2f}")
            kpis.append(f"📈 **Avg {col}:** {self.df[col].mean():,.2f}")
        
        missing = int(self.df.isnull().sum().sum())
        kpis.append(f"⚠️ **Missing Values:** {missing} ({missing/len(self.df)*100:.1f}%)")
        
        return "\n".join(kpis)
    
    def export_to_excel(self, filename="analysis_report.xlsx"):
        if self.df is None:
            return "⚠️ Load data first"
        
        try:
            os.makedirs('static', exist_ok=True)
            filepath = f"static/{filename}"
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                summary = self.df.describe()
                summary.to_excel(writer, sheet_name='Summary Stats')
                
                missing = self.df.isnull().sum()
                missing_df = pd.DataFrame({'Column': missing.index, 'Missing Count': missing.values})
                missing_df.to_excel(writer, sheet_name='Missing Values', index=False)
                
                num_cols = self.df.select_dtypes('number').columns.tolist()
                if len(num_cols) >= 2:
                    corr = self.df[num_cols].corr()
                    corr.to_excel(writer, sheet_name='Correlation')
            
            return f"✅ Excel report saved: {filepath}"
        except Exception as e:
            return f"❌ Export error: {str(e)}"
    
    # ✅ ENHANCED: Better Natural Language Processing
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
                   "• `df.nlargest(5, 'revenue')`\n\n"
                   + self.get_column_suggestions(""))
        
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
        
        # ✅ ENHANCED: Better Column Matching for Aggregations
        # Total/Sum
        if 'total' in q or 'sum' in q:
            # Try to find column name from query
            col = self.find_column(q.replace('total', '').replace('sum', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"💰 **Total {col}:** {self.df[col].sum():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        
        # Average/Mean
        if 'average' in q or 'mean' in q or 'avg' in q:
            col = self.find_column(q.replace('average', '').replace('mean', '').replace('avg', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📊 **Average {col}:** {self.df[col].mean():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        
        # Max/Highest
        if 'max' in q or 'maximum' in q or 'highest' in q:
            col = self.find_column(q.replace('max', '').replace('maximum', '').replace('highest', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📈 **Max {col}:** {self.df[col].max():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        
        # Min/Lowest
        if 'min' in q or 'minimum' in q or 'lowest' in q:
            col = self.find_column(q.replace('min', '').replace('minimum', '').replace('lowest', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📉 **Min {col}:** {self.df[col].min():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        
        # ✅ ENHANCED: Top N with Better Column Matching
        if 'top' in q:
            n_match = re.search(r'top\s+(\d+)', q)
            n = int(n_match.group(1)) if n_match else 5
            
            # Try to find column from query
            col = self.find_column(q.replace(f'top {n}', '').replace('top 5', '').replace('top', '').strip())
            
            if not col:
                # Try categorical first for "top" queries
                if self.categorical_columns:
                    col = self.categorical_columns[0]
                    top_vals = self.df[col].value_counts().head(n)
                    result = f"🏆 **Top {n} {col}:**\n"
                    for i, (name, count) in enumerate(top_vals.items(), 1):
                        result += f"{i}. {name}: {count}\n"
                    return result
                elif self.numeric_columns:
                    col = self.numeric_columns[0]
                    top_vals = self.df.nlargest(n, col)[[col]]
                    result = f"🏆 **Top {n} by {col}:**\n"
                    for i, (_, row) in enumerate(top_vals.iterrows(), 1):
                        result += f"{i}. {row[col]:,.2f}\n"
                    return result
            
            if col and col in self.categorical_columns:
                top_vals = self.df[col].value_counts().head(n)
                result = f"🏆 **Top {n} {col}:**\n"
                for i, (name, count) in enumerate(top_vals.items(), 1):
                    result += f"{i}. {name}: {count}\n"
                return result
            elif col and col in self.numeric_columns:
                top_vals = self.df.nlargest(n, col)[[col]]
                result = f"🏆 **Top {n} by {col}:**\n"
                for i, (_, row) in enumerate(top_vals.iterrows(), 1):
                    result += f"{i}. {row[col]:,.2f}\n"
                return result
            else:
                return "⚠️ Column not found. " + self.get_column_suggestions(q)
        
        # Group by
        if 'group' in q or 'by' in q or 'breakdown' in q:
            col = "product"
            for word in ["product", "region", "category", "customer", "date", "month", "year", "name"]:
                if word in q:
                    col = word
                    break
            
            found_col = self.find_column(col)
            if found_col and found_col in self.categorical_columns:
                num_col = self.numeric_columns[0] if self.numeric_columns else None
                if num_col:
                    grouped = self.df.groupby(found_col)[num_col].sum().nlargest(10)
                    total = self.df[num_col].sum()
                    out = f"📊 {found_col} vs {num_col}:\n"
                    for name, val in grouped.items():
                        pct = (val / total) * 100
                        out += f"• {name}: {val:,.2f} ({pct:.1f}%)\n"
                    return out
            
            return "⚠️ Group by column not found. " + self.get_column_suggestions(q)
        
        # Prediction/Trend
        if any(x in q for x in ["predict", "trend", "forecast", "next", "future"]):
            col = "revenue"
            for word in ["revenue", "sales", "quantity", "price", "demand"]:
                if word in q:
                    col = word
                    break
            
            found_col = self.find_column(col)
            if found_col and found_col in self.numeric_columns:
                return self.predict_trend(found_col)
            elif self.numeric_columns:
                return self.predict_trend(self.numeric_columns[0])
            else:
                return "⚠️ No numeric column for prediction. " + self.get_column_suggestions(q)
        
        # Segmentation
        if any(x in q for x in ["segment", "customer", "group", "cluster", "tier"]):
            return self.segment_customers()
        
        # Outliers
        if any(x in q for x in ["outlier", "anomaly", "unusual", "weird", "strange"]):
            col = "revenue"
            for word in ["revenue", "sales", "price", "quantity", "amount"]:
                if word in q:
                    col = word
                    break
            
            found_col = self.find_column(col)
            if found_col and found_col in self.numeric_columns:
                return self.detect_outliers(found_col)
            elif self.numeric_columns:
                return self.detect_outliers(self.numeric_columns[0])
            else:
                return "⚠️ No numeric column for outlier detection. " + self.get_column_suggestions(q)
        
        # Visualization
        if any(x in q for x in ["plot", "chart", "graph", "visualize", "show", "display"]):
            if "bar" in q or "count" in q:
                return self.generate_plot("bar")
            elif "hist" in q or "distribut" in q:
                return self.generate_plot("histogram")
            elif "scatter" in q or "relation" in q:
                return self.generate_plot("scatter")
            else:
                return self.generate_plot("bar")
        
        # Correlations
        if any(x in q for x in ["correlat", "relationship", "link", "connect"]):
            return self.show_correlations()
        
        # Filters
        if "filter" in q or "where" in q:
            if any(x in q for x in ["high", "large", "big", ">100000"]):
                return self.filter_high()
            elif any(x in q for x in ["low", "small", "tiny", "<5"]):
                return self.filter_low()
        
        # Business analysis
        if any(x in q for x in ["return", "refund", "cancel"]):
            return self.returns_analysis()
        if any(x in q for x in ["profit", "margin", "earnings"]):
            return self.profit_analysis()
        
        # ✅ ENHANCED: Default Response with Column Suggestions
        return ("💡 Command not recognized. Try these:\n"
               "• 'top 5'\n"
               "• 'average'\n"
               "• 'total'\n"
               "• 'dashboard'\n"
               "• 'bar chart'\n\n"
               + self.get_column_suggestions(q))
    
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
    
    # ... (rest of methods remain same: show_info, top_n, group_by, predict_trend, etc.)
    
    def show_info(self):
        if self.df is None: return "⚠️ Load data first (upload CSV)"
        miss = self.df.isnull().sum()
        miss_str = "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()]) if miss.sum() > 0 else "None"
        return f"📊 Shape: {self.df.shape}\nColumns: {list(self.df.columns)}\nMissing:\n{miss_str}"
    
    def show_missing(self):
        if self.df is None: return "⚠️ Load data first"
        miss = self.df.isnull().sum()
        if miss.sum() == 0: return "✅ No missing values"
        return "⚠️ Missing values:\n" + "\n".join([f"  • {col}: {cnt}" for col, cnt in miss[miss>0].items()])
    
    def clean_data(self):
        if self.df is None: return "⚠️ Load data first"
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        return f"🧹 Cleaned: {before-after} rows removed ({after} remaining)"
    
    def fill_missing(self):
        if self.df is None: return "⚠️ Load data first"
        num_cols = self.df.select_dtypes('number').columns
        for col in num_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        return f"✅ Filled missing values with mean for {len(num_cols)} numeric columns"
    
    def remove_duplicates(self):
        if self.df is None: return "⚠️ Load data first"
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        return f"✅ Removed {before-after} duplicates ({after} rows remaining)"
    
    def show_correlations(self):
        if self.df is None: return "⚠️ Load data first"
        num_df = self.df.select_dtypes('number')
        if num_df.shape[1] < 2: return "⚠️ Need min 2 numeric columns for correlation"
        corr = num_df.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i,j] > 0.5:
                    pairs.append(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i,j]:.2f}")
        if not pairs: return "⚠️ No strong correlations (>0.5) found"
        return "🔗 Strong correlations:\n" + "\n".join(pairs[:5])
    
    def filter_high(self, threshold=100000):
        if self.df is None: return "⚠️ Load data first"
        rev_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total'])), self.df.select_dtypes('number').columns[0])
        filtered = self.df[self.df[rev_col] > threshold]
        return f"✅ Filtered: {len(filtered)} rows with {rev_col} > ₹{threshold:,}\nTop 3:\n" + "\n".join([f"• {row[rev_col]:,.2f}" for _, row in filtered.head(3).iterrows()])
    
    def filter_low(self, threshold=5):
        if self.df is None: return "⚠️ Load data first"
        qty_col = next((c for c in self.df.columns if 'quantity' in c.lower() or 'qty' in c.lower()), self.df.select_dtypes('number').columns[0])
        filtered = self.df[self.df[qty_col] < threshold]
        return f"✅ Filtered: {len(filtered)} rows with {qty_col} < {threshold}\nSample:\n" + "\n".join([f"• {row[qty_col]}" for _, row in filtered.head(3).iterrows()])
    
    def predict_trend(self, col_name=None):
        if self.df is None:
            return "⚠️ Load data first (upload CSV)"
        if not col_name:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns for prediction"
            col_name = num_cols[0]
        series = self.df[col_name].dropna().tolist()
        if len(series) >= 3:
            result = self.ml.forecast_time_series(series, periods=3)
            if 'forecast' in result:
                forecast_str = " → ".join([f"{v:,.2f}" for v in result['forecast']])
                return f"📈 ML Forecast ({result.get('trend', 'N/A')}):\nNext 3: {forecast_str}"
        last3 = series[-3:] if len(series) >= 3 else series
        trend = "↗️ Upward" if last3[-1] > last3[0] else "↘️ Downward" if last3[-1] < last3[0] else "➡️ Stable"
        next_val = series[-1] + (series[-1] - series[-2]) if len(series) > 1 else series[-1]
        return f"📈 Trend: {trend}\nNext: {next_val:,.2f}"
    
    def segment_customers(self):
        if self.df is None:
            return "⚠️ Load data first (upload CSV)"
        spend_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total','spend'])), None)
        freq_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['freq','purchase','order','quantity'])), None)
        sample = {
            'annual_spend': self.df[spend_col].mean() * 12 if spend_col else 150000,
            'purchase_frequency': self.df[freq_col].mean() if freq_col else 12
        }
        result = self.ml.segment_customer(sample)
        if 'error' in result:
            rev_col = spend_col or self.df.select_dtypes('number').columns[0]
            q25 = self.df[rev_col].quantile(0.25)
            q75 = self.df[rev_col].quantile(0.75)
            high = len(self.df[self.df[rev_col] > q75])
            medium = len(self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)])
            low = len(self.df[self.df[rev_col] < q25])
            total = len(self.df)
            return (f"👥 Segments (Rule-based):\n"
                   f"High: {high} ({high/total*100:.0f}%)\n"
                   f"Medium: {medium} ({medium/total*100:.0f}%)\n"
                   f"Low: {low} ({low/total*100:.0f}%)")
        return (f"🏷️ {result['segment']} Segment\n"
               f"Discount: {result['discount_eligible']}\n"
               f"💡 {result['recommendation']}")
    
    def detect_outliers(self, col_name=None):
        if self.df is None: return "⚠️ Load data first"
        if not col_name:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ No numeric columns"
            col_name = num_cols[0]
        series = self.df[col_name].dropna()
        if len(series) < 4: return f"⚠️ Need min 4 values in '{col_name}'"
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0: return f"✅ No outliers in '{col_name}'"
        return f"⚠️ Outliers in '{col_name}': {len(outliers)} values ({len(outliers)/len(series)*100:.1f}%)\nRange: {outliers.min():,.2f} to {outliers.max():,.2f}"
    
    def returns_analysis(self):
        if self.df is None: return "⚠️ Load data first"
        ret_col = next((c for c in self.df.columns if 'return' in c.lower() or 'refund' in c.lower()), None)
        if not ret_col:
            return "⚠️ No returns column found"
        total_returns = self.df[ret_col].sum()
        total_orders = len(self.df)
        avg_returns = total_returns / total_orders
        return f"📦 Returns Analysis:\nTotal returns: {total_returns}\nAvg per order: {avg_returns:.1f}\nReturn rate: {(total_returns/total_orders)*100:.1f}%"
    
    def profit_analysis(self):
        if self.df is None: return "⚠️ Load data first"
        margin_col = next((c for c in self.df.columns if 'margin' in c.lower() or 'profit' in c.lower()), None)
        if not margin_col:
            return "⚠️ No profit margin column found"
        avg_margin = self.df[margin_col].mean() * 100
        high_margin = len(self.df[self.df[margin_col] > 0.4])
        return f"💡 Profit Analysis:\nAvg margin: {avg_margin:.1f}%\nHigh margin (>40%): {high_margin} items ({high_margin/len(self.df)*100:.0f}%)"
    
    def top_n(self, n=5, metric="revenue"):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if metric.lower() in c.lower()), None)
        if not col:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols: return "❌ No numeric columns"
            col = num_cols[0]
        if col not in self.df.columns: return f"❌ Column '{col}' not found"
        result = self.df.nlargest(n, col)[[col]]
        total = self.df[col].sum()
        out = f"🏆 Top {n} by {col}:\n"
        for i, (_, row) in enumerate(result.iterrows(), 1):
            pct = (row[col] / total) * 100 if total != 0 else 0
            out += f"{i}. {row[col]:,.2f} ({pct:.1f}%)\n"
        return out
    
    def group_by(self, col_name):
        if self.df is None: return "⚠️ Load data first"
        col = next((c for c in self.df.columns if col_name.lower() in c.lower()), None)
        if not col or col not in self.df.columns:
            cat_cols = self.df.select_dtypes('object').columns.tolist()
            if not cat_cols: return "❌ No categorical columns for grouping"
            col = cat_cols[0]
        num_col = next((c for c in self.df.select_dtypes('number').columns if 'revenue' in c.lower() or 'sales' in c.lower() or 'amount' in c.lower()), self.df.select_dtypes('number').columns[0])
        grouped = self.df.groupby(col)[num_col].sum().nlargest(10)
        total = self.df[num_col].sum()
        out = f"📊 {col} vs {num_col}:\n"
        for name, val in grouped.items():
            pct = (val / total) * 100
            out += f"• {name}: {val:,.2f} ({pct:.1f}%)\n"
        return out
