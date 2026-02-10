import streamlit as st
import os

st.set_page_config(page_title="🤖 Data Science AI Agent", layout="wide")
st.title("🤖 Data Science AI Agent")
st.write("Your AI agent is ready to analyze data!")

# Optional: Import your agent logic
try:
    from agents import DataScienceAgent
    st.success("✅ Agent loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Agent not loaded: {str(e)}")

st.write("Upload CSV file to start analysis:")
uploaded_file = st.file_uploader("Choose CSV", type="csv")

if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head(5))
    st.write("Shape:", df.shape)
