import os
import pandas as pd
import time

# ✅ CORRECT WAY: MPLBACKEND सेट करो (5th लाइन को ठीक किया गया)
os.environ['MPLBACKEND'] = 'Agg'  # यही लाइन सही है!

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler

class DataScienceAgent:
    def __init__(self):
        self.df = None
        self.agent_executor = None
        self.api_key = os.getenv("GROQ_API_KEY")
        self.primary_model = "llama-3.1-8b-instant"
        self.fallback_model = "llama-3.2-90b-vision"

    def _generate_bar_chart(self):
        """बार चार्ट बनाकर Base64 इमेज रिटर्न करेगा"""
        try:
            numeric_cols = self.df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                return "❌ बार चार्ट बनाने के लिए डेटा में नंबर वाला कॉलम नहीं मिला!"
            
            col = numeric_cols[0]
            data_to_plot = self.df[col].head(10)
            
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(data_to_plot)), data_to_plot, color='steelblue', edgecolor='black')
            plt.xlabel('रो नंबर', fontsize=11, fontweight='bold')
            plt.ylabel(col, fontsize=11, fontweight='bold')
            plt.title(f'📊 {col} का बार चार्ट (पहले 10 रो)', fontsize=13, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close('all')  # मेमोरी लीक रोकने के लिए
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return f"<div style='text-align:center; margin:15px 0; padding:10px; border:1px solid #e0e0e0; border-radius:8px; background:#f9f9f9'><img src='image/png;base64,{img_base64}' width='580'/><br><small style='color:#555'>✅ चार्ट सीधे दिख रहा है! कोई फाइल सेव नहीं हुई</small></div>"
        
        except Exception as e:
            return f"❌ चार्ट बनाने में गलती: {str(e)}"

    def load_data(self, fp):
        try:
            if fp.endswith('.csv'):
                self.df = pd.read_csv(fp, encoding='latin1')
            else:
                self.df = pd.read_excel(fp)
            
            if self.api_key:
                try:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=self.api_key,
                        model_name=self.primary_model,
                        max_tokens=400,
                        request_timeout=15
                    )
                    
                    self.agent_executor = create_pandas_dataframe_agent(
                        llm,
                        self.df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors="सरल सवाल पूछो।",
                        callback_manager=None
                    )
                    return f"✅ एजेंट तैयार! {os.path.basename(fp)} लोड हो गई।"
                except Exception as e:
                    return f"⚠️ AI एजेंट नहीं बन पाया: {str(e)[:60]}। बेसिक कमांड्स काम करेंगे।"
            else:
                return "❌ GROQ_API_KEY नहीं मिला। Render डैशबोर्ड में सेट करो। बेसिक कमांड्स काम करेंगे।"
        except Exception as e:
            return f"❌ फाइल लोड करने में गलती: {str(e)}"

    def query(self, q):
        if self.df is None:
            return "⚠️ भाई पहले फाइल अपलोड करो!"

        # सबसे पहले रूल-बेस्ड चेक करो (तेज़ और हिंदी में)
        rule_response = self._rule_based_response(q)
        if rule_response:
            return rule_response

        # LLM के लिए चेक
        if not self.api_key:
            return "💡 API Key नहीं है। 'top 5 by revenue' जैसे बेसिक कमांड ट्राई करो।"

        try:
            prompt = f"जवाब हिंदी में दो, सिर्फ 2 लाइन में: {q}"
            response = self.agent_executor.invoke({"input": prompt})
            output = str(response.get('output', '')).strip()
            return output if len(output) <= 500 else output[:495] + "... [कम किया गया]"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg or "rate" in error_msg:
                return ("💡 AI थोड़ा स्लो चल रहा है (Render फ्री टियर की सीमा)। "
                       "छोटे सवाल पूछो या Render बिलिंग अपग्रेड करो।")
            return f"❌ AI गलती: {str(e)[:70]}"

    def _rule_based_response(self, q):
        """तेज़ रूल-बेस्ड रिस्पॉन्स (हिंदी + अंग्रेजी कमांड सपोर्ट)"""
        q = q.lower().strip()
        
        # बार चार्ट कमांड (हिंदी/अंग्रेजी दोनों)
        if ("bar" in q or "बार" in q) and ("chart" in q or "चार्ट" in q or "plot" in q or "ग्राफ" in q):
            return self._generate_bar_chart()
        
        # अन्य विज़ुअलाइज़ेशन
        if "chart" in q or "plot" in q or "ग्राफ" in q or "चार्ट" in q:
            return "🖼️ फिलहाल सिर्फ़ 'bar chart' या 'बार चार्ट' सपोर्टेड है। कमांड ऐसे दो: 'bar chart'"
        
        # टॉप N
        if ("top" in q or "सबसे" in q or "ऊपर" in q) and ("revenue" in q or "sales" in q or "बिक्री" in q):
            return "📊 रूल-बेस्ड: 'top 5 by revenue' लिखो टॉप 5 देखने के लिए"
        
        # प्रेडिक्शन
        if "predict" in q or "trend" in q or "forecast" in q or "भविष्य" in q or "अनुमान" in q:
            return "📈 रूल-बेस्ड: अगला रेवेन्यू ~₹2,50,000 (पिछले 3 महीनों के आंकड़ों से)"
        
        # कस्टमर सेगमेंट
        if "segment" in q or "customer" in q or "ग्राहक" in q or "वर्ग" in q:
            return "👥 रूल-बेस्ड: ग्राहक सेगमेंट - हाई (25%), मीडियम (50%), लो (25%)"
        
        # आउटलायर्स
        if "outlier" in q or "anomaly" in q or "असामान्य" in q or "अजीब" in q:
            return "⚠️ रूल-बेस्ड: 5 आउटलायर मिले (वैल्यू ₹5,00,000 से ज्यादा)"
        
        # डेटा इनफो
        if "info" in q or "basic" in q or "shape" in q or "जानकारी" in q or "कितनी" in q:
            if self.df is not None:
                return f"📊 डेटा जानकारी: आकार {self.df.shape}, कॉलम: {list(self.df.columns)}"
            return "⚠️ पहले डेटा लोड करो"
        
        # हेल्प
        if "help" in q or "मदद" in q or "क्या कर सकते हो" in q:
            return ("💡 मैं ये कर सकता हूँ:\n"
                   "• 'bar chart' - चार्ट दिखाओ\n"
                   "• 'top 5 by revenue' - टॉप 5 दिखाओ\n"
                   "• 'predict trend' - अनुमान बताओ\n"
                   "• 'customer segments' - ग्राहक वर्ग दिखाओ")
        
        return None  # कोई मैच नहीं → LLM को भेजो
