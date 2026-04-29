"""
Lightweight ML helper with forecasting and A/B testing (standalone for testing / integration).
Place in fixes/ml_with_abtest.py and import from other scripts.
"""

import os
from typing import List, Dict, Union
import numpy as np

class MLModels:
    """Minimal ML engine with forecasting and A/B testing helpers."""

    def __init__(self):
        pass

    def forecast_time_series(self, historical_values: List[float], periods: int = 3) -> Dict[str, Union[List[float], str]]:
        """
        Forecast next N periods using simple exponential smoothing fallback (pure numpy).
        Input: [100000, 120000, 135000, 150000], periods=3
        """
        try:
            if len(historical_values) < 3:
                return {"error": "Need at least 3 historical values for forecasting"}

            # Pure NumPy fallback (no external dependencies)
            avg_growth = (historical_values[-1] - historical_values[0]) / max(1, len(historical_values) - 1)
            last_val = historical_values[-1]
            forecast = [last_val + avg_growth * (i + 1) for i in range(periods)]

            # Trend analysis
            last_3 = historical_values[-3:]
            trend = "↗️ Upward" if last_3[-1] > last_3[0] else "↘️ Downward" if last_3[-1] < last_3[0] else "➡️ Stable"

            return {
                "historical": historical_values,
                "forecast": [round(float(x), 2) for x in forecast],
                "trend": trend,
                "recommendation": f"Expect {trend} trend in next {periods} periods"
            }
        except Exception as e:
            return {"error": f"Forecasting error: {str(e)}"}

    def ab_test(self, group_a: List[float], group_b: List[float], metric: str = 'continuous', n_boot: int = 2000) -> Dict[str, Union[float, str, bool]]:
        """
        Simple A/B testing helper.
        - For continuous metrics: two-sample t-test (Welch). Falls back to bootstrap if scipy not present.
        - For binary metrics (0/1): chi-square if scipy present, else simple proportions bootstrap.
        Input:
          group_a, group_b: lists of numbers (or 0/1 for binary)
          metric: 'continuous' or 'binary'
        Returns: statistic, p_value, significant (alpha=0.05), and short conclusion.
        """
        try:
            try:
                import scipy.stats as _ss
                has_scipy = True
            except Exception:
                has_scipy = False

            a = np.array(group_a, dtype=float)
            b = np.array(group_b, dtype=float)

            # Basic validation
            if len(a) < 2 or len(b) < 2:
                return {"error": "Need at least 2 samples in each group"}

            if metric == 'continuous':
                if has_scipy:
                    stat, p = _ss.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                    significant = bool(p < 0.05)
                    return {"statistic": float(stat), "p_value": float(p), "significant": significant,
                            "conclusion": ("Significant difference" if significant else "No significant difference")}
                # bootstrap fallback (difference of means)
                obs_diff = a.mean() - b.mean()
                pooled = np.concatenate([a, b])
                rng = np.random.default_rng()
                diffs = []
                for _ in range(n_boot):
                    samp_a = rng.choice(pooled, size=len(a), replace=True)
                    samp_b = rng.choice(pooled, size=len(b), replace=True)
                    diffs.append(samp_a.mean() - samp_b.mean())
                diffs = np.array(diffs)
                p = float((abs(diffs) >= abs(obs_diff)).sum() / len(diffs))
                significant = bool(p < 0.05)
                return {"statistic": float(obs_diff), "p_value": p, "significant": significant,
                        "conclusion": ("Significant difference" if significant else "No significant difference (bootstrap)")}
            else:
                # binary: treat as counts (1 = success)
                a1 = int((a == 1).sum())
                a0 = int(len(a) - a1)
                b1 = int((b == 1).sum())
                b0 = int(len(b) - b1)
                if has_scipy:
                    import numpy as _np2
                    obs = _np2.array([[a1, a0], [b1, b0]])
                    chi2, p, dof, exp = _ss.chi2_contingency(obs)
                    significant = bool(p < 0.05)
                    return {"chi2": float(chi2), "p_value": float(p), "significant": significant,
                            "counts": {"a": {"1": a1, "0": a0}, "b": {"1": b1, "0": b0}},
                            "conclusion": ("Significant difference in proportions" if significant else "No significant difference in proportions")}
                # bootstrap proportions fallback
                pooled = np.concatenate([a, b])
                rng = np.random.default_rng()
                obs_diff = (a1 / len(a)) - (b1 / len(b))
                diffs = []
                for _ in range(n_boot):
                    samp_a = rng.choice(pooled, size=len(a), replace=True)
                    samp_b = rng.choice(pooled, size=len(b), replace=True)
                    diffs.append(samp_a.mean() - samp_b.mean())
                diffs = np.array(diffs)
                p = float((abs(diffs) >= abs(obs_diff)).sum() / len(diffs))
                significant = bool(p < 0.05)
                return {"statistic": float(obs_diff), "p_value": p, "significant": significant,
                        "counts": {"a": {"1": a1, "0": a0}, "b": {"1": b1, "0": b0}},
                        "conclusion": ("Significant difference in proportions (bootstrap)" if significant else "No significant difference in proportions (bootstrap)")}
        except Exception as e:
            return {"error": f"A/B test error: {str(e)}"}
