"""
Domain Specific Logic - Month 5
Adds Retail and Finance intelligence templates
"""

class DomainIntelligence:
    def analyze_retail(self, df):
        """Retail specific patterns: Seasonality and Stock alerts"""
        # Logic to find high sales months
        num_cols = df.select_dtypes('number').columns
        target = num_cols[0]
        avg_val = df[target].mean()
        peak_days = len(df[df[target] > avg_val * 1.5])
        
        return {
            "insight": f"🛍️ Retail Insight: Found {peak_days} high-performance periods.",
            "action": "Suggestion: Increase inventory for these peak cycles."
        }

    def analyze_finance(self, df):
        """Finance specific: Volatility and Risk"""
        num_cols = df.select_dtypes('number').columns
        target = num_cols[0]
        volatility = df[target].std() / df[target].mean()
        risk = "High" if volatility > 0.5 else "Stable"
        
        return {
            "insight": f"💰 Finance Insight: Data is {risk} (Volatility: {volatility:.2f})",
            "action": "Suggestion: Set tighter stop-loss or cash reserves."
        }
