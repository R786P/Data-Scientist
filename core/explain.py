"""
Explainable AI (XAI) - 100% offline
Simple Feature Importance and Prediction Drivers
"""

import numpy as np

class ExplainableAI:
    """Explain predictions in business terms"""
    
    def explain_drivers(self, coefficients, feature_names, target_name="Target"):
        """Identify which factors are pushing the prediction up or down"""
        # Sort features by absolute importance
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[::-1]
        
        explanations = []
        for idx in sorted_idx[:3]:  # Top 3 drivers
            coef = coefficients[idx]
            feature = feature_names[idx]
            direction = "increases" if coef > 0 else "decreases"
            explanations.append(f"• {feature}: It {direction} {target_name} significantly.")
            
        return "💡 Key Drivers identified:\n" + "\n".join(explanations)

    def why_this_prediction(self, current, predicted):
        """Explain the gap between current and forecast"""
        diff = predicted - current
        impact = "growth" if diff > 0 else "decline"
        return f"📈 Explanation: We expect a {impact} of ₹{abs(diff):,.0f} based on recent historical trends."
