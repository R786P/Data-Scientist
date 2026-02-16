import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from core.agent import DataScienceAgent

# Initialize app with static folder support
app = Flask(__name__, 
           template_folder='templates', 
           static_folder='static')
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Initialize agent
agent = DataScienceAgent()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = agent.load_data(filename)
        return jsonify({"message": f"✅ Uploaded: {filename}\n{result}"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    response = agent.query(user_message)
    return jsonify({"response": response})

# ✅ Critical: Serve static files (plot.png)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Optional: Direct plot access
@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
