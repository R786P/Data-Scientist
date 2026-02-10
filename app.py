from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "✅ Data Scientist Agent Running",
        "python_version": os.sys.version,
        "pandas_version": pd.__version__,
        "message": "Send POST /analyze with CSV data"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['rows'], columns=data['columns'])
        return jsonify({
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": df.head().to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
