import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ds_agent import DataScienceAgent

app = Flask(__name__, template_folder='templates')
# Files root directory me save hongi taaki agent unhe asani se dhund sake
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize Agent
agent = None
try:
    # API Key environment variable se aayegi (Render Dashboard me set karein)
    agent = DataScienceAgent()
    print("✅ Agent initialized successfully")
except Exception as e:
    print(f"⚠️ Agent initialization failed: {e}")

@app.route('/')
def home():
    return render_template('index.html')  # ✅ Frontend page

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "running",
        "agent_status": "active" if agent else "inactive (Check GOOGLE_API_KEY in Render)",
        "usage": {
            "upload_file": "POST /upload (form-data: file)",
            "chat": "POST /chat (json: { 'message': 'load data.csv' })"
        }
    })

# ✅ DEBUG ENDPOINT ADDED (For checking environment variables)
@app.route('/debug/env')
def debug_env():
    import os
    key = os.environ.get("GOOGLE_API_KEY", "NOT SET")
    return jsonify({
        "GOOGLE_API_KEY_present": "YES" if key != "NOT SET" else "NO",
        "key_preview": key[:8] + "..." if key != "NOT SET" else "N/A",
        "agent_status": "active" if agent else "inactive",
        "billing_setup_required": "Check Google Cloud Billing Console"
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            "message": f"File uploaded: {filename}. Ab aap chat me bol sakte hain: 'load {filename}'",
            "filename": filename
        })

@app.route('/chat', methods=['POST'])
def chat():
    if not agent:
        return jsonify({"error": "Agent not active. GOOGLE_API_KEY set karein."}), 500
    
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Agent se response lein
        response = agent.query(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
