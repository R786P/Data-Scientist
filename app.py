import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Nayi Files se imports
from core.agent import DataScienceAgent
from core.database import engine, Base
from utils.logging_config import setup_logging

# 1. Setup Professional Logging
setup_logging()
logger = logging.getLogger(__name__)

# 2. Initialize Database Tables
# Ye line aapke DATABASE_URL ka use karke tables apne aap bana degi
try:
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized successfully.")
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")

# Flask App Initialization
app = Flask(__name__, 
           template_folder='templates', 
           static_folder='static')

app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Increased to 10MB for Master Level

# Initialize agent
agent = DataScienceAgent()

# --- MASTER LEVEL: HEALTH CHECK ---
@app.route('/health', methods=['GET'])
def health():
    """Render check karne ke liye ki app alive hai ya nahi"""
    return jsonify({"status": "healthy", "version": "1.1.0"}), 200

# --- ROUTES ---

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
        
        logger.info(f"File uploaded: {filename}")
        result = agent.load_data(filename)
        return jsonify({"message": f"✅ Uploaded: {filename}\n{result}"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    logger.info(f"User Query: {user_message}")
    response = agent.query(user_message)
    return jsonify({"response": response})

# Serve static files (plot.png)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

if __name__ == '__main__':
    # Render dynamic port binding
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
