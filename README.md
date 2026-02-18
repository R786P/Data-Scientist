
# 🤖 Data Science AI Agent

An autonomous AI agent that performs real data scientist tasks — just give it a query in natural language and let it handle analysis, visualization, and insights generation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)
![Google%20Colab](https://img.shields.io/badge/Google_Colab-Free-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ✨ Features

- ✅ Load CSV/Excel files automatically
- ✅ Exploratory Data Analysis (shape, dtypes, missing values)
- ✅ Generate visualizations (histograms, bar charts)
- ✅ Execute pandas operations via natural language (`groupby`, `filter`, `sort`)
- ✅ Business insights in English/Hindi
- ✅ Zero-setup: Runs directly in Google Colab (mobile-friendly)
- ✅ Production-ready modular code

## 🚀 Quick Start

###  Local Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/data-science-ai-agent.git
cd data-science-ai-agent

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export GOOGLE_API_KEY="your_gemini_api_key_here"

# 5. Run agent
python ds_agent.py
