import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# --- NEW: Auth & Security Imports ---
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Original imports
from core.agent import DataScienceAgent
from core.database import engine, Base, SessionLocal
from utils.logging_config import setup_logging
# from utils.email import send_report # Future use ke liye ready rakhein

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-123") # Security ke liye zaroori
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

# --- 🔐 LOGIN MANAGER SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Agar user logged in nahi hai toh yahan bhejega

# Simple User Class (Inhe Database se link karenge agle step mein)
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id, "Admin")

# Initialize DB & Agent
try:
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized successfully.")
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")

agent = DataScienceAgent()

# --- 🚪 AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Abhi ke liye simple check, baad mein DB se verify karenge
        username = request.form.get('username')
        if username == "admin":
            user = User(1, "Admin")
            login_user(user)
            return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- 📊 DASHBOARD ROUTE ---

@app.route('/dashboard')
@login_required
def dashboard():
    """Naya Dashboard jahan multiple charts dikhenge"""
    return render_template('dashboard.html')

# --- 📧 EMAIL REPORT ROUTE ---

@app.route('/send_email_report', methods=['POST'])
@login_required
def email_report():
    data = request.get_json()
    email = data.get('email')
    plot_path = "static/plot.png"
    # result = send_report(email, plot_path) # utils/email.py se call hoga
    return jsonify({"message": "✅ Report dispatch system ready!"})

# --- EXISTING ROUTES (Protected with login_required) ---

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = agent.load_data(filename)
        return jsonify({"message": f"✅ Uploaded: {filename}\n{result}"}), 200
    return jsonify({"error": "No file selected"}), 400

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    logger.info(f"User Query: {user_message}")
    response = agent.query(user_message)
    return jsonify({"response": response})

# Health Check (Keep public)
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.2.0"}), 200

@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
