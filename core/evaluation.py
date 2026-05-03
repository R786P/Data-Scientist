"""
Model evaluation metrics - 100% offline
Adds R², MAE, RMSE and Cross-validation metrics
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

class ModelEvaluation:
    """Evaluate prediction quality and model robustness"""
    
    def evaluate_regression(self, y_true, y_pred):
        """Calculate R², MAE, and RMSE (Error Analysis)"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # R² Score calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        status = "✅ High" if r2 > 0.8 else "🟡 Moderate" if r2 > 0.5 else "⚠️ Low"
        
        return {
            "r2": round(r2, 3),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "accuracy_status": status,
            "interpretation": f"Model Accuracy ({status}): R²={r2:.2f}, Error=₹{mae:,.0f}"
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform 5-fold cross-validation to check stability"""
        model = LinearRegression()
        # Ensure X is 2D
        X_reshaped = X.reshape(-1, 1) if len(X.shape) == 1 else X
        scores = cross_val_score(model, X_reshaped, y, cv=cv, scoring='r2')
        
        return {
            "mean_r2": round(scores.mean(), 3),
            "std_r2": round(scores.std(), 3),
            "interpretation": f"Stability Check: {scores.mean():.2f} (±{scores.std():.2f})"
      }
