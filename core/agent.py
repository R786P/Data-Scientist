import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from core.database import SessionLocal, UserQuery
from core.ml import MLModels

try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_core.messages import HumanMessage, SystemMessage
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
        self.available_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.chat_history = []  # ✅ Conversation memory for AI mode

        self.color_palettes = {
            'default': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            'business': ['#2c3e50', '#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
            'modern': ['#667eea', '#764ba2', '#4ECDC4', '#FF6B6B', '#95E1D3']
        }

        if LLM_AVAILABLE and self.api_key:
            try:
                self.llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=self.api_key,
                    model_name="openai/gpt-oss-120b",
                    max_tokens=500,
                    timeout=15
                )
                print("✅ AI Mode: ON")
            except Exception as e:
                self.llm = None
                print(f"⚠️ AI Mode: OFF (API Key issue: {e})")
        else:
            self.llm = None

    def load_data(self, fp):
        try:
            self.df = pd.read_csv(fp, encoding='latin1')
            self.last_file = fp
            self.numeric_columns = self.df.select_dtypes('number').columns.tolist()
            self.categorical_columns = self.df.select_dtypes('object').columns.tolist()
            self.available_columns = list(self.df.columns)
            self.chat_history = []  # Reset chat history on new file

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

            eda_report = self._generate_auto_eda()
            return f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns\n\n{eda_report}"
        except Exception as e:
            return f"❌ Error loading {fp}: {str(e)}"

    def _generate_auto_eda(self):
        report = []
        report.append("📊 **AUTO EDA REPORT**\n")
        report.append(f"📋 **Total Records:** {len(self.df):,}")
        report.append(f"📋 **Total Columns:** {len(self.df.columns)}\n")
        report.append("📁 **Column Types:**")
        report.append(f"   • Numeric: {', '.join(self.numeric_columns) if self.numeric_columns else 'None'}")
        report.append(f"   • Text: {', '.join(self.categorical_columns) if self.categorical_columns else 'None'}\n")
        missing = self.df.isnull().sum().sum()
        report.append(f"⚠️ **Missing Values:** {missing} ({missing/len(self.df)*100:.1f}%)\n")
        report.append("💡 **WORKING Natural Language Commands:**")
        if self.numeric_columns:
            first_num = self.numeric_columns[0]
            report.append(f"   ✅ `Total {first_num}`")
            report.append(f"   ✅ `Average {first_num}`")
            report.append(f"   ✅ `Max {first_num}`")
            report.append(f"   ✅ `Min {first_num}`")
        if self.categorical_columns:
            first_cat = self.categorical_columns[0]
            report.append(f"   ✅ `Top 5 {first_cat}`")
            report.append(f"   ✅ `Group by {first_cat}`")
        report.append(f"   ✅ `Create bar chart`")
        report.append(f"   ✅ `Show dashboard`")
        report.append(f"   ✅ `Export Excel`")
        report.append(f"   ✅ `Export CSV`")
        report.append(f"   ✅ `Export HTML`\n")
        report.append("💻 **WORKING Python Code:**")
        if self.numeric_columns:
            first_num = self.numeric_columns[0]
            report.append(f"   ✅ `df['{first_num}'].mean()`")
            report.append(f"   ✅ `df['{first_num}'].sum()`")
            report.append(f"   ✅ `df.nlargest(5, '{first_num}')`")
        if self.categorical_columns:
            first_cat = self.categorical_columns[0]
            report.append(f"   ✅ `df['{first_cat}'].value_counts()`")
        report.append(f"   ✅ `df.head()`")
        report.append(f"   ✅ `df.describe()`\n")
        report.append("⚠️ **Commands That WON'T Work:**")
        report.append("   ❌ `Total revenue` (column doesn't exist)")
        report.append("   ❌ `Average salary` (column doesn't exist)")
        report.append("   💡 Use actual column names from above!\n")
        return "\n".join(report)

    def find_column(self, search_term):
        if self.df is None:
            return None
        search_lower = search_term.lower()
        for col in self.df.columns:
            if search_lower in col.lower() or col.lower() in search_lower:
                return col
        for col in self.df.columns:
            if any(word in col.lower() for word in search_lower.split()):
                return col
        return None

    def get_column_suggestions(self, search_term=""):
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
        if len(cat_cols) > 0 and len(num_cols) > 0:
            recommendations.append("📊 Bar Chart (Comparison)")
        if len(num_cols) > 0:
            recommendations.append("📉 Histogram (Distribution)")
            recommendations.append("📦 Box Plot (Outliers)")
        if len(num_cols) >= 2:
            recommendations.append("🔗 Scatter Plot (Relationship)")
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
            elif "line" in plot_type.lower():
                ax.plot(self.df[target_col].values, color=colors[0], linewidth=2, marker='o')
                ax.set_title(f'{target_col} Trend', fontsize=14, fontweight='bold')
            elif "pie" in plot_type.lower():
                if cat_cols:
                    col = cat_cols[0]
                    counts = self.df[col].value_counts().head(8)
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
            elif "hist" in plot_type.lower():
                ax.hist(self.df[target_col].dropna(), bins=20, color=colors[0], edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
            elif "scatter" in plot_type.lower():
                if len(num_cols) >= 2:
                    ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], alpha=0.6, color=colors[0], s=50)
                    ax.set_xlabel(num_cols[0])
                    ax.set_ylabel(num_cols[1])
                    ax.set_title(f'{num_cols[0]} vs {num_cols[1]}', fontsize=14, fontweight='bold')
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
                axes[1, 0].boxplot(data, labels=num_cols[:4], patch_artist=True, boxprops=dict(facecolor='#FF6B6B', color='#FF6B6B'))
                axes[1, 0].set_title('Variable Comparison', fontsize=12, fontweight='bold')
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
            return f"✅ Excel report saved: {filepath}\n\n📱 **Opens in Microsoft 365 App!**"
        except Exception as e:
            return f"❌ Export error: {str(e)}"

    def export_to_csv(self, filename="analysis_report.csv"):
        if self.df is None:
            return "⚠️ Load data first"
        try:
            os.makedirs('static', exist_ok=True)
            filepath = f"static/{filename}"
            self.df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return f"✅ CSV report saved: {filepath}\n\n📱 **Opens in Excel/Google Sheets!**"
        except Exception as e:
            return f"❌ Export error: {str(e)}"

    def export_to_html(self, filename="analysis_report.html"):
        if self.df is None:
            return "⚠️ Load data first"
        try:
            os.makedirs('static', exist_ok=True)
            filepath = f"static/{filename}"
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #667eea; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Data Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <h3>📄 Data Preview (First 10 Rows)</h3>
        {self.df.head(10).to_html(index=False)}
        <h3>📈 Summary Statistics</h3>
        {self.df.describe().to_html()}
        <p style="text-align: center; margin-top: 30px; color: #888;">Generated by Data Scientist Agent v1.8</p>
    </div>
</body>
</html>"""
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return f"✅ HTML report saved: {filepath}\n\n📱 **Opens in ANY browser!**"
        except Exception as e:
            return f"❌ Export error: {str(e)}"

    # ═══════════════════════════════════════════════════
    # ✅ NEW: CONVERSATIONAL AI MODE
    # ═══════════════════════════════════════════════════
    def conversational_query(self, user_message, user_id=None):
        """
        AI Chat Mode — Groq se natural conversational response leta hai.
        Data context automatically inject hota hai system prompt mein.
        """
        if self.llm is None:
            return "⚠️ AI Chat Mode ke liye GROQ_API_KEY required hai."

        # Dataset ka context banao
        if self.df is not None:
            data_context = f"""
Dataset loaded: {self.last_file}
Shape: {len(self.df)} rows × {len(self.df.columns)} columns
Numeric Columns: {', '.join(self.numeric_columns) if self.numeric_columns else 'None'}
Text Columns: {', '.join(self.categorical_columns) if self.categorical_columns else 'None'}
Missing Values: {int(self.df.isnull().sum().sum())}

Quick Stats:
{self.df.describe().to_string() if self.numeric_columns else 'No numeric data'}
"""
        else:
            data_context = "Abhi koi dataset load nahi hua hai."

        system_prompt = f"""Tu ek friendly aur expert Data Scientist AI Agent hai.
Tu Hinglish mein baat karta hai — matlab Hindi aur English mix — bilkul natural tarike se, jaise dost baat karta hai.
Tu data analysis, machine learning, statistics, aur visualization mein expert hai.

Current Dataset Info:
{data_context}

Rules:
- Friendly aur conversational reh, formal mat ban
- Emojis use kar expressions ke liye
- Short aur clear answers de
- Agar koi chart ya export karna ho toh bata: "bar chart banao", "export csv" etc commands use kar sakte hain
- Agar dataset ke baare mein kuch specific pucha jaye toh upar diye stats se answer de
- Hinglish mein baat kar — pure Hindi ya pure English nahi"""

        try:
            # Chat history add karo (last 6 messages — memory)
            messages = [SystemMessage(content=system_prompt)]
            for hist in self.chat_history[-6:]:
                messages.append(hist)
            messages.append(HumanMessage(content=user_message))

            response = self.llm.invoke(messages)
            ai_reply = response.content.strip()

            # History update karo
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(response)

            # DB mein save karo
            if user_id:
                try:
                    db = SessionLocal()
                    db.add(UserQuery(
                        query_text=f"[AI MODE] {user_message}",
                        response_text=ai_reply,
                        user_id=user_id
                    ))
                    db.commit()
                    db.close()
                except:
                    pass

            return ai_reply

        except Exception as e:
            return f"⚠️ AI Mode error: {str(e)}\n\nNormal mode mein switch karo."

    def query(self, q, user_id=None):
        q_original = q
        q = q.lower().strip()
        python_patterns = [r'df\.', r'pd\.', r'np\.', r'plt\.', r'\.head\(', r'\.tail\(', r'\.mean\(', r'\.sum\(', r'=', r'import']
        is_python_code = any(re.search(pattern, q) for pattern in python_patterns)
        if is_python_code and self.df is not None:
            return self.execute_python_code(q_original, user_id)
        if q in ['hi', 'hello', 'hey', 'hii', 'namaste']:
            return "👋 Hello! Main aapka Data Scientist Agent hoon. CSV upload karo aur kuch bhi pucho!"
        if 'help' in q or 'kya kar' in q or 'what can' in q or 'columns' in q or 'suggest' in q:
            return ("💡 **Natural Language Commands:**\n"
                   "• 'top 5 by revenue'\n"
                   "• 'group by region'\n"
                   "• 'create bar chart'\n"
                   "• 'show dashboard'\n"
                   "• 'export excel'\n"
                   "• 'export csv'\n"
                   "• 'export html'\n\n"
                   "💻 **Python Code:**\n"
                   "• `df.head()`\n"
                   "• `df['column'].mean()`\n"
                   "• `df.nlargest(5, 'column')`\n\n"
                   + self.get_column_suggestions(q))
        if 'dashboard' in q or 'multiple plots' in q or 'all charts' in q:
            return self.generate_dashboard()
        if 'kpi' in q or 'metrics' in q or 'summary cards' in q:
            return self.get_kpi_cards()
        if 'excel' in q:
            return self.export_to_excel()
        if 'csv' in q:
            return self.export_to_csv()
        if 'html' in q or 'export' in q:
            return self.export_to_html()
        if 'recommend' in q or 'suggest chart' in q or 'best chart' in q:
            return self.recommend_chart(q_original)
        if 'line chart' in q or 'time series' in q:
            return self.generate_plot('line')
        if 'pie chart' in q or 'composition' in q:
            return self.generate_plot('pie')
        if 'box plot' in q or 'boxplot' in q:
            return self.generate_plot('box')
        if 'total' in q or 'sum' in q:
            col = self.find_column(q.replace('total', '').replace('sum', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"💰 **Total {col}:** {self.df[col].sum():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        if 'average' in q or 'mean' in q or 'avg' in q:
            col = self.find_column(q.replace('average', '').replace('mean', '').replace('avg', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📊 **Average {col}:** {self.df[col].mean():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        if 'max' in q or 'maximum' in q or 'highest' in q:
            col = self.find_column(q.replace('max', '').replace('maximum', '').replace('highest', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📈 **Max {col}:** {self.df[col].max():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        if 'min' in q or 'minimum' in q or 'lowest' in q:
            col = self.find_column(q.replace('min', '').replace('minimum', '').replace('lowest', '').strip())
            if not col and self.numeric_columns:
                col = self.numeric_columns[0]
            if col:
                return f"📉 **Min {col}:** {self.df[col].min():,.2f}"
            else:
                return "⚠️ No numeric column found. " + self.get_column_suggestions(q)
        if 'top' in q:
            n_match = re.search(r'top\s+(\d+)', q)
            n = int(n_match.group(1)) if n_match else 5
            col = self.find_column(q.replace(f'top {n}', '').replace('top 5', '').replace('top', '').strip())
            if not col:
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
        if any(x in q for x in ["plot", "chart", "graph", "visualize", "show", "display"]):
            if "bar" in q or "count" in q:
                return self.generate_plot("bar")
            elif "hist" in q or "distribut" in q:
                return self.generate_plot("histogram")
            elif "scatter" in q or "relation" in q:
                return self.generate_plot("scatter")
            else:
                return self.generate_plot("bar")
        return ("💡 Command not recognized. Try these:\n"
               "• 'top 5'\n"
               "• 'average'\n"
               "• 'total'\n"
               "• 'dashboard'\n"
               "• 'bar chart'\n"
               "• 'export excel'\n"
               "• 'export csv'\n"
               "• 'export html'\n\n"
               + self.get_column_suggestions(q))

    def execute_python_code(self, code, user_id=None):
        try:
            safe_globals = {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'df': self.df, '__builtins__': {}}
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

    def show_correlations(self):
        if self.df is None:
            return "⚠️ Load data first"
        num_df = self.df.select_dtypes('number')
        if num_df.shape[1] < 2:
            return "⚠️ Need min 2 numeric columns"
        corr = num_df.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i,j] > 0.5:
                    pairs.append(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i,j]:.2f}")
        if not pairs:
            return "⚠️ No strong correlations (>0.5) found"
        return "🔗 Strong correlations:\n" + "\n".join(pairs[:5])

    def predict_trend(self, col_name=None):
        if self.df is None:
            return "⚠️ Load data first"
        if not col_name:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns"
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
            return "⚠️ Load data first"
        spend_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total','spend'])), None)
        freq_col = next((c for c in self.df.columns if any(x in c.lower() for x in ['freq','purchase','order','quantity'])), None)
        sample = {'annual_spend': self.df[spend_col].mean() * 12 if spend_col else 150000, 'purchase_frequency': self.df[freq_col].mean() if freq_col else 12}
        result = self.ml.segment_customer(sample)
        if 'error' in result:
            rev_col = spend_col or self.df.select_dtypes('number').columns[0]
            q25 = self.df[rev_col].quantile(0.25)
            q75 = self.df[rev_col].quantile(0.75)
            high = len(self.df[self.df[rev_col] > q75])
            medium = len(self.df[(self.df[rev_col] >= q25) & (self.df[rev_col] <= q75)])
            low = len(self.df[self.df[rev_col] < q25])
            total = len(self.df)
            return f"👥 Segments:\nHigh: {high} ({high/total*100:.0f}%)\nMedium: {medium} ({medium/total*100:.0f}%)\nLow: {low} ({low/total*100:.0f}%)"
        return f"🏷️ {result['segment']} Segment\nDiscount: {result['discount_eligible']}"

    def detect_outliers(self, col_name=None):
        if self.df is None:
            return "⚠️ Load data first"
        if not col_name:
            num_cols = self.df.select_dtypes('number').columns.tolist()
            if not num_cols:
                return "❌ No numeric columns"
            col_name = num_cols[0]
        series = self.df[col_name].dropna()
        if len(series) < 4:
            return f"⚠️ Need min 4 values"
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
        if len(outliers) == 0:
            return f"✅ No outliers in '{col_name}'"
        return f"⚠️ Outliers: {len(outliers)} values ({len(outliers)/len(series)*100:.1f}%)"
