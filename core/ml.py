"""
Advanced ML Models - 100% Offline (No API Keys, No Card Required)
Trained models must be saved as .pkl files in models/ folder
Workflow: Train on Colab → Save .pkl → Load here for inference
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

class MLModels:
    """All-in-one ML engine for inference (no training on server)"""
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from models/ folder"""
        try:
            # Sales forecasting model (Linear Regression)
            if os.path.exists(os.path.join(self.models_dir, 'sales_model.pkl')):
                self.models['sales_forecast'] = joblib.load(os.path.join(self.models_dir, 'sales_model.pkl'))
                print("✅ Sales forecast model loaded")
            
            # Churn prediction model (Random Forest)
            if os.path.exists(os.path.join(self.models_dir, 'churn_model.pkl')):
                self.models['churn_predictor'] = joblib.load(os.path.join(self.models_dir, 'churn_model.pkl'))
                print("✅ Churn predictor model loaded")
            
            # Customer segmentation model (K-Means)
            if os.path.exists(os.path.join(self.models_dir, 'segmentation_model.pkl')):
                self.models['segmentation'] = joblib.load(os.path.join(self.models_dir, 'segmentation_model.pkl'))
                print("✅ Customer segmentation model loaded")
            
            # Outlier detection model (Isolation Forest)
            if os.path.exists(os.path.join(self.models_dir, 'outlier_detector.pkl')):
                self.models['outlier_detector'] = joblib.load(os.path.join(self.models_dir, 'outlier_detector.pkl'))
                print("✅ Outlier detector model loaded")
            
            if not self.models:
                print("⚠️ No pre-trained models found in models/ folder")
                print("💡 Train models on Colab → Save as .pkl → Upload to GitHub")
        except Exception as e:
            print(f"⚠️ Error loading models: {str(e)}")
    
    # ============ 1. SALES FORECASTING (Regression) ============
    def forecast_sales(self, features: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """
        Predict future sales using linear regression
        Input: {'ad_spend': 50000, 'previous_month_sales': 200000}
        """
        if 'sales_forecast' not in self.models:
            return {"error": "Sales forecast model not trained. Train on Colab first."}
        
        try:
            # Expected features (must match training)
            required = ['ad_spend', 'previous_month_sales']
            X = pd.DataFrame([{k: features.get(k, 0) for k in required}])
            
            prediction = self.models['sales_forecast'].predict(X)[0]
            confidence = "High" if prediction > 100000 else "Medium" if prediction > 50000 else "Low"
            
            return {
                "predicted_sales": round(prediction, 2),
                "confidence": confidence,
                "recommendation": f"Increase ad spend to boost sales to ₹{prediction*1.2:,.0f}" if prediction < 300000 else "Maintain current strategy"
            }
        except Exception as e:
            return {"error": f"Forecast error: {str(e)}"}
    
    # ============ 2. CHURN PREDICTION (Classification) ============
    def predict_churn(self, features: Dict[str, Union[int, float]]) -> Dict[str, Union[float, str]]:
        """
        Predict customer churn probability
        Input: {'age': 35, 'monthly_spend': 1200, 'support_calls': 2}
        """
        if 'churn_predictor' not in self.models:
            return {"error": "Churn model not trained. Train on Colab first."}
        
        try:
            required = ['age', 'monthly_spend', 'support_calls']
            X = pd.DataFrame([{k: features.get(k, 0) for k in required}])
            
            # Probability of churn (class 1)
            proba = self.models['churn_predictor'].predict_proba(X)[0][1]
            risk_level = "⚠️ HIGH RISK" if proba > 0.6 else "🟡 MEDIUM RISK" if proba > 0.3 else "✅ LOW RISK"
            
            return {
                "churn_probability": f"{proba*100:.1f}%",
                "risk_level": risk_level,
                "recommendation": "Offer 20% discount to retain customer" if proba > 0.6 else "Continue normal service"
            }
        except Exception as e:
            return {"error": f"Churn prediction error: {str(e)}"}
    
    # ============ 3. CUSTOMER SEGMENTATION (Clustering) ============
    def segment_customer(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Segment customer into High/Medium/Low value
        Input: {'annual_spend': 150000, 'purchase_frequency': 12}
        """
        if 'segmentation' not in self.models:
            # Fallback: Rule-based segmentation (no model needed)
            spend = features.get('annual_spend', 0)
            freq = features.get('purchase_frequency', 0)
            
            if spend > 200000 and freq > 10:
                segment = "Platinum"
                discount = "15%"
            elif spend > 100000 and freq > 5:
                segment = "Gold"
                discount = "10%"
            elif spend > 50000:
                segment = "Silver"
                discount = "5%"
            else:
                segment = "Bronze"
                discount = "0%"
            
            return {
                "segment": segment,
                "discount_eligible": discount,
                "recommendation": f"Target with {discount} loyalty discount"
            }
        
        try:
            required = ['annual_spend', 'purchase_frequency']
            X = pd.DataFrame([{k: features.get(k, 0) for k in required}])
            
            cluster = self.models['segmentation'].predict(X)[0]
            segments = {0: "Bronze", 1: "Silver", 2: "Gold", 3: "Platinum"}
            
            return {
                "segment": segments.get(cluster, "Unknown"),
                "discount_eligible": "15%" if cluster == 3 else "10%" if cluster == 2 else "5%" if cluster == 1 else "0%",
                "recommendation": f"Target {segments.get(cluster, 'customer')} segment with personalized offers"
            }
        except Exception as e:
            return {"error": f"Segmentation error: {str(e)}"}
    
    # ============ 4. OUTLIER DETECTION (Anomaly Detection) ============
    def detect_outliers(self, data: List[float]) -> Dict[str, Union[int, float, str]]:
        """
        Detect anomalies in numerical data using IQR method (no model needed)
        Input: [10000, 12000, 15000, 50000, 11000]  # 50000 is outlier
        """
        try:
            series = pd.Series(data)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) == 0:
                return {"status": "✅ Clean data", "outlier_count": 0}
            
            return {
                "status": f"⚠️ {len(outliers)} outliers detected",
                "outlier_count": len(outliers),
                "outlier_values": outliers.tolist(),
                "normal_range": f"₹{lower_bound:,.0f} to ₹{upper_bound:,.0f}",
                "recommendation": "Investigate transactions outside normal range"
            }
        except Exception as e:
            return {"error": f"Outlier detection error: {str(e)}"}
    
    # ============ 5. TIME SERIES FORECAST (Simple Exponential Smoothing) ============
    def forecast_time_series(self, historical_values: List[float], periods: int = 3) -> Dict[str, Union[List[float], str]]:
        """
        Forecast next N periods using simple exponential smoothing
        Input: [100000, 120000, 135000, 150000], periods=3
        """
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            
            if len(historical_values) < 3:
                return {"error": "Need at least 3 historical values for forecasting"}
            
            # Simple exponential smoothing
            model = SimpleExpSmoothing(historical_values)
            fit = model.fit(smoothing_level=0.6, optimized=False)
            forecast = fit.forecast(periods)
            
            # Trend analysis
            last_3 = historical_values[-3:]
            trend = "↗️ Upward" if last_3[-1] > last_3[0] else "↘️ Downward" if last_3[-1] < last_3[0] else "➡️ Stable"
            
            return {
                "historical": historical_values,
                "forecast": [round(float(x), 2) for x in forecast],
                "trend": trend,
                "recommendation": f"Expect {trend} trend in next {periods} periods"
            }
        except ImportError:
            return {
                "error": "statsmodels not installed",
                "recommendation": "Add 'statsmodels==0.14.1' to requirements.txt for time series forecasting"
            }
        except Exception as e:
            return {"error": f"Forecasting error: {str(e)}"}
    
    # ============ 6. FEATURE IMPORTANCE (Explainability) ============
    def explain_prediction(self, model_name: str, features: Dict[str, float]) -> Dict[str, Union[List, str]]:
        """
        Explain model prediction using feature importance (if available)
        """
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        try:
            model = self.models[model_name]
            
            # Check if model has feature_importances_ (tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = getattr(model, 'feature_names_in_', [f"feature_{i}" for i in range(len(importance))])
                
                # Sort by importance
                sorted_idx = importance.argsort()[::-1]
                top_features = [
                    {"feature": feature_names[i], "importance": round(importance[i]*100, 1)}
                    for i in sorted_idx[:3]
                ]
                
                return {
                    "top_drivers": top_features,
                    "explanation": f"{top_features[0]['feature']} is the strongest predictor ({top_features[0]['importance']}% influence)"
                }
            else:
                return {"explanation": "Model doesn't support feature importance (linear models use coefficients)"}
        except Exception as e:
            return {"error": f"Explanation error: {str(e)}"}


# ============ USAGE EXAMPLES (For Testing) ============
if __name__ == "__main__":
    print("="*60)
    print("🤖 CARD-FREE ML ENGINE - Offline Inference Only")
    print("="*60)
    
    ml = MLModels()
    
    print("\n✅ Available Models:")
    for name in ml.models.keys():
        print(f"  • {name}")
    
    print("\n📊 Example 1: Sales Forecast")
    result = ml.forecast_sales({'ad_spend': 60000, 'previous_month_sales': 250000})
    print(f"  Prediction: ₹{result.get('predicted_sales', 'N/A'):,.0f}")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    
    print("\n👥 Example 2: Churn Prediction")
    result = ml.predict_churn({'age': 42, 'monthly_spend': 850, 'support_calls': 3})
    print(f"  Risk: {result.get('risk_level', 'N/A')}")
    print(f"  Probability: {result.get('churn_probability', 'N/A')}")
    
    print("\n🏷️ Example 3: Customer Segmentation")
    result = ml.segment_customer({'annual_spend': 180000, 'purchase_frequency': 15})
    print(f"  Segment: {result.get('segment', 'N/A')}")
    print(f"  Discount: {result.get('discount_eligible', 'N/A')}")
    
    print("\n⚠️ Example 4: Outlier Detection")
    result = ml.detect_outliers([12000, 13500, 11800, 45000, 12200])
    print(f"  Status: {result.get('status', 'N/A')}")
    if 'outlier_values' in result:
        print(f"  Outliers: {result['outlier_values']}")
    
    print("\n📈 Example 5: Time Series Forecast")
    result = ml.forecast_time_series([100000, 115000, 130000, 145000], periods=3)
    if 'forecast' in result:
        print(f"  Next 3 months: {result['forecast']}")
        print(f"  Trend: {result.get('trend', 'N/A')}")
    
    print("\n" + "="*60)
    print("💡 To train models:")
    print("   1. Use Google Colab (free GPU)")
    print("   2. Train model → joblib.dump(model, 'model.pkl')")
    print("   3. Download → Upload to GitHub models/ folder")
    print("   4. Deploy → Inference works offline!")
    print("="*60)
