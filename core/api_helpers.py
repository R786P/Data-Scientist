import time
from flask import Blueprint, request, jsonify, current_app
import os

api_bp = Blueprint('api_helpers', __name__)


@api_bp.route('/api/forecast', methods=['POST'])
def api_forecast():
    data = request.get_json() or {}
    series = data.get('series') or data.get('historical') or []
    try:
        periods = int(data.get('periods', 3))
    except:
        periods = 3
    if not isinstance(series, list) or len(series) < 3:
        return jsonify({"error": "Provide 'series' as a list of at least 3 numeric historical values."}), 400
    try:
        series_clean = [float(x) for x in series]
    except Exception:
        return jsonify({"error": "Series must contain numeric values only."}), 400
    # Prefer existing agent.ml if present
    try:
        agent = current_app.config.get('agent') or globals().get('agent')
        if agent and hasattr(agent, 'ml') and hasattr(agent.ml, 'forecast_time_series'):
            result = agent.ml.forecast_time_series(series_clean, periods=periods)
        else:
            # simple fallback
            avg_growth = (series_clean[-1] - series_clean[0]) / max(1, len(series_clean) - 1)
            last_val = series_clean[-1]
            forecast = [last_val + avg_growth * (i + 1) for i in range(periods)]
            last_3 = series_clean[-3:]
            trend = "↗️ Upward" if last_3[-1] > last_3[0] else "↘️ Downward" if last_3[-1] < last_3[0] else "➡️ Stable"
            result = {"historical": series_clean, "forecast": [round(float(x), 2) for x in forecast], "trend": trend}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Forecast error: {str(e)}"}), 500


@api_bp.route('/api/ab_test', methods=['POST'])
def api_ab_test():
    data = request.get_json() or {}
    a = data.get('groupA')
    b = data.get('groupB')
    metric = data.get('metric', 'continuous')
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
        return jsonify({"error": "Provide 'groupA' and 'groupB' arrays with >=2 samples each."}), 400
    try:
        agent = current_app.config.get('agent') or globals().get('agent')
        if agent and hasattr(agent, 'ml') and hasattr(agent.ml, 'ab_test'):
            res = agent.ml.ab_test(a, b, metric=metric)
            return jsonify(res)
        import numpy as _np
        a_np = _np.array(a, dtype=float)
        b_np = _np.array(b, dtype=float)
        if metric == 'continuous':
            obs_diff = float(a_np.mean() - b_np.mean())
            pooled = _np.concatenate([a_np, b_np])
            rng = _np.random.default_rng()
            diffs = [_np.mean(rng.choice(pooled, size=len(a_np), replace=True)) - _np.mean(rng.choice(pooled, size=len(b_np), replace=True)) for _ in range(1000)]
            diffs = _np.array(diffs)
            p = float((abs(diffs) >= abs(obs_diff)).sum() / len(diffs))
            return jsonify({"statistic": obs_diff, "p_value": p, "significant": bool(p < 0.05)})
        else:
            obs_diff = float((a_np == 1).mean() - (b_np == 1).mean())
            pooled = _np.concatenate([a_np, b_np])
            rng = _np.random.default_rng()
            diffs = [(rng.choice(pooled, size=len(a_np), replace=True) == 1).mean() - (rng.choice(pooled, size=len(b_np), replace=True) == 1).mean() for _ in range(1000)]
            diffs = _np.array(diffs)
            p = float((abs(diffs) >= abs(obs_diff)).sum() / len(diffs))
            return jsonify({"statistic": obs_diff, "p_value": p, "significant": bool(p < 0.05)})
    except Exception as e:
        return jsonify({"error": f"AB test failed: {str(e)}"}), 500


@api_bp.route('/api/hf_status', methods=['GET'])
def api_hf_status():
    import requests
    hf_space = os.getenv('HF_SPACE_URL', '').strip()
    hf_api = os.getenv('HF_API_URL', '').strip()
    hf_token = os.getenv('HF_API_TOKEN', '').strip()
    start = time.time()
    if hf_space:
        try:
            r = requests.get(hf_space.rstrip('/') + '/ping', timeout=6)
            latency = int((time.time() - start) * 1000)
            if r.status_code == 200:
                return jsonify({"status": "online", "source": "space", "latency_ms": latency, "details": r.json()})
            else:
                return jsonify({"status": "offline", "source": "space", "latency_ms": latency, "details": f"HTTP {r.status_code}"})
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return jsonify({"status": "offline", "source": "space", "latency_ms": latency, "details": str(e)})
    elif hf_api and hf_token:
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            r = requests.get(hf_api, headers=headers, timeout=6)
            latency = int((time.time() - start) * 1000)
            if r.status_code in (200, 201, 202):
                return jsonify({"status": "online", "source": "api", "latency_ms": latency, "details": {"status_code": r.status_code}})
            else:
                return jsonify({"status": "offline", "source": "api", "latency_ms": latency, "details": f"HTTP {r.status_code}"})
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return jsonify({"status": "offline", "source": "api", "latency_ms": latency, "details": str(e)})
    else:
        return jsonify({"status": "none", "source": "none", "latency_ms": 0, "details": "No HF_SPACE_URL or HF_API_URL configured."})
