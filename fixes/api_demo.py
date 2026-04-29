"""
Small Flask demo exposing 3 endpoints: /api/forecast, /api/ab_test, /api/hf_status
This is a standalone app so you can test the functionality quickly.
Run: python3 fixes/api_demo.py
Set env HF_SPACE_URL or HF_API_URL+HF_API_TOKEN to test hf_status.
"""

import os
import time
from flask import Flask, request, jsonify, render_template_string
import requests

# Import the ML helper bundled above
from fixes.ml_with_abtest import MLModels

app = Flask(__name__)
ml = MLModels()

# Simple index with DL status dot and polling script
INDEX_HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>API Demo — Forecast & A/B Test</title>
  <style>
    body{font-family:Inter,system-ui,Arial;margin:28px}
    .topbar{display:flex;align-items:center;gap:12px}
    #dlStatus{width:12px;height:12px;border-radius:50%;background:#16a34a;box-shadow:0 0 8px rgba(16,185,129,0.28);opacity:0.25;transition:opacity .2s, transform .2s}
    #dlStatus.blink{animation:dlBlink 1s linear infinite;opacity:1;transform:scale(1.05)}
    @keyframes dlBlink{0%,60%{opacity:1}61%,100%{opacity:0.15}}
    pre{background:#f6f8fa;padding:12px;border-radius:6px}
  </style>
</head>
<body>
  <div class="topbar">
    <h2>DS Agent — Demo APIs</h2>
    <div id="dlStatus" title="DL: checking..."></div>
  </div>
  <p>Use the endpoints <code>/api/forecast</code>, <code>/api/ab_test</code>, <code>/api/hf_status</code> (POST/GET with JSON).</p>
  <h4>Example forecast result (POST /api/forecast)</h4>
  <pre id="res">—</pre>
  <script>
    async function pollHFStatus(){
      try{
        const r = await fetch('/api/hf_status', {cache:'no-store'});
        if(!r.ok){document.getElementById('dlStatus').classList.remove('blink');document.getElementById('dlStatus').title='DL: unreachable';return}
        const j = await r.json();
        const el = document.getElementById('dlStatus');
        if(j.status && j.status==='online'){el.classList.add('blink');el.title=`DL Live (${j.source||'hf'}) • ${j.latency_ms||0}ms`}else{el.classList.remove('blink');el.title=`DL Offline • ${j.details||''}`}
      }catch(e){const el=document.getElementById('dlStatus');if(el){el.classList.remove('blink');el.title='DL: error'}}
    }
    setInterval(pollHFStatus,5000);pollHFStatus();

    // Demo call to forecast on load
    (async ()=>{
      const resp = await fetch('/api/forecast', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({series:[100000,115000,130000,145000],periods:3})});
      const j = await resp.json();
      document.getElementById('res').textContent = JSON.stringify(j, null, 2);
    })();
  </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/forecast', methods=['POST'])
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
    result = ml.forecast_time_series(series_clean, periods=periods)
    return jsonify(result)

@app.route('/api/ab_test', methods=['POST'])
def api_ab_test():
    data = request.get_json() or {}
    a = data.get('groupA')
    b = data.get('groupB')
    metric = data.get('metric', 'continuous')
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
        return jsonify({"error": "Provide 'groupA' and 'groupB' arrays with >=2 samples each."}), 400
    res = ml.ab_test(a, b, metric=metric)
    return jsonify(res)

@app.route('/api/hf_status', methods=['GET'])
def api_hf_status():
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
        return jsonify({"status": "none", "source": "none", "latency_ms": 0, "details": "No HF_SPACE_URL or HF_API_URL configured in environment."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting demo API on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
