"""
Statistical tests - 100% offline (no card/API needed)
Adds p-values, confidence intervals, hypothesis testing
"""

import numpy as np
from scipy import stats

class StatisticalTests:
    """Business-ready statistical analysis"""
    
    def t_test(self, group1, group2, alpha=0.05):
        """Compare two groups (e.g., North vs South region sales)"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        significant = p_value < alpha
        return {
            "p_value": round(p_value, 4),
            "significant": significant,
            "interpretation": f"{'✅ Significant difference' if significant else '⚠️ No significant difference'} (p={p_value:.4f})"
        }
    
    def confidence_interval(self, series, confidence=0.95):
        """Calculate confidence interval for mean"""
        n = len(series)
        mean = series.mean()
        std_err = series.std() / np.sqrt(n)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return {
            "mean": round(mean, 2),
            "lower": round(mean - margin, 2),
            "upper": round(mean + margin, 2),
            "interpretation": f"Mean: ₹{mean:,.0f} (95% CI: ₹{mean-margin:,.0f} to ₹{mean+margin:,.0f})"
        }
    
    def correlation_test(self, series1, series2):
        """Test if correlation is statistically significant"""
        corr, p_value = stats.pearsonr(series1, series2)
        significant = p_value < 0.05
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        return {
            "correlation": round(corr, 3),
            "p_value": round(p_value, 4),
            "significant": significant,
            "interpretation": f"{strength} {'positive' if corr > 0 else 'negative'} correlation (r={corr:.2f}, p={p_value:.4f})"
        }
