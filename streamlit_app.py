"""
Data Science Agent - Streamlit Version
100% free deployment on Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from core.agent import DataScienceAgent

# Page config
st.set_page_config(
    page_title="🤖 Data Scientist Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = DataScienceAgent()

agent = st.session_state.agent

# Header
st.title("🤖 Data Scientist Agent")
st.markdown("Upload CSV file and get instant insights!")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        result = agent.load_data(uploaded_file.name)
        st.success(result)
        
        # Delete temp file
        os.remove(uploaded_file.name)
    
    st.markdown("---")
    st.header("💡 Commands")
    st.markdown("""
    - `show basic info`
    - `top 5 by revenue`
    - `predict trend`
    - `segment customers`
    - `detect outliers`
    - `create bar chart`
    """)

# Main content
if agent.df is None:
    st.info("👆 Upload a CSV file to get started!")
else:
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Basic Info", "📈 Top N", "🔮 Prediction", "👥 Segmentation", "🎨 Visualization"])
    
    # Tab 1: Basic Info
    with tab1:
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {agent.df.shape[0]} rows × {agent.df.shape[1]} columns")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Columns:**")
            st.write(list(agent.df.columns))
        with col2:
            st.write("**Data Types:**")
            st.write(agent.df.dtypes.to_dict())
        
        st.write("**Missing Values:**")
        missing = agent.df.isnull().sum()
        st.write(missing[missing > 0] if missing.sum() > 0 else "None")
    
    # Tab 2: Top N Analysis
    with tab2:
        st.subheader("Top Products/Customers")
        n = st.slider("Number of top items", 1, 20, 5)
        metric = st.selectbox("Metric", ["revenue", "sales", "price", "quantity"])
        
        if st.button("Analyze"):
            result = agent.top_n_analysis(n=n, metric=metric)
            st.code(result, language="text")
    
    # Tab 3: Trend Prediction
    with tab3:
        st.subheader("Trend Prediction")
        col = st.selectbox("Column to predict", agent.df.select_dtypes('number').columns.tolist())
        
        if st.button("Predict"):
            result = agent.predict_trend(column=col)
            st.code(result, language="text")
    
    # Tab 4: Customer Segmentation
    with tab4:
        st.subheader("Customer Segments")
        
        if st.button("Segment"):
            result = agent.segment_customers()
            st.code(result, language="text")
            
            # Visualize segments
            rev_col = next((c for c in agent.df.columns if any(x in c.lower() for x in ['revenue','sales','amount','total'])), None)
            if rev_col:
                fig, ax = plt.subplots(figsize=(8, 5))
                q25 = agent.df[rev_col].quantile(0.25)
                q75 = agent.df[rev_col].quantile(0.75)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                labels = ['Low Value', 'Medium Value', 'High Value']
                sizes = [
                    len(agent.df[agent.df[rev_col] < q25]),
                    len(agent.df[(agent.df[rev_col] >= q25) & (agent.df[rev_col] <= q75)]),
                    len(agent.df[agent.df[rev_col] > q75])
                ]
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Customer Segmentation by {rev_col}')
                st.pyplot(fig)
    
    # Tab 5: Visualization
    with tab5:
        st.subheader("Create Visualization")
        plot_type = st.selectbox("Plot Type", ["histogram", "bar", "scatter"])
        
        if st.button("Generate Plot"):
            result = agent.generate_visualization(plot_type)
            st.success(result)
            
            if os.path.exists('plot.png'):
                st.image('plot.png', use_column_width=True)
                with open('plot.png', 'rb') as f:
                    st.download_button("Download Plot", f, file_name='plot.png')
    
    # Chat interface
    st.markdown("---")
    st.subheader("💬 Ask Anything")
    user_query = st.text_input("Type your query (e.g., 'top 3 by revenue')")
    
    if st.button("Submit"):
        if user_query:
            result = agent.query(user_query)
            st.code(result, language="text")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ | Offline Data Scientist Agent | 100% Free**")
