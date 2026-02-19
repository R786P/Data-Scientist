import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# --- 🔐 Auth & Security Imports ---
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash

# --- Core Imports ---
from core.agent import DataScienceAgent
from core.database import engine, Base, SessionLocal
from core.auth import User, create_default_admin  # ✅ DB User Model & Admin Creator
from utils.logging_config import setup_logging

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- App Configuration ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-123")  # ✅ Fallback included
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB Max Upload

# --- 🔐 Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- User Loader (Session ke liye zaroori) ---
@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        return user
    except Exception as e:
        logger.error(f"Error loading user: {e}")
        return None
    finally:
        db.close()

# --- 🗄️ Database Initialization ---
try:
    logger.info("🔄 Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized successfully.")
    
    # ✅ Auto-create admin user agar nahi hai
    create_default_admin()
    
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")
    logger.warning("⚠️ App will run but login may fail without DB.")

# --- Initialize Agent ---
agent = DataScienceAgent()

# --- 🚪 AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            
            # ✅ Password Hash Check
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                logger.info(f"✅ User logged in: {username}")
                next_page = request.args.get('next')
                return redirect(next_page if next_page else url_for('home'))
            else:
                logger.warning(f"⚠️ Failed login attempt for: {username}")
                return render_template('login.html', error="Invalid username or password")
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template('login.html', error="System error. Try again.")
        finally:
            db.close()
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    logger.info("👋 User logged out")
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
    # result = send_report(email, plot_path) # Future implementation
    logger.info(f"📧 Report requested for: {email}")
    return jsonify({"message": "✅ Report dispatch system ready!"})

# --- 🏠 EXISTING ROUTES (Protected) ---

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
        try:
            file.save(filepath)
            result = agent.load_data(filename)
            logger.info(f"📁 File uploaded: {filename}")
            return jsonify({"message": f"✅ Uploaded: {filename}\n{result}"}), 200
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"error": "Failed to save file"}), 500
    return jsonify({"error": "No file selected"}), 400

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    logger.info(f"💬 User Query: {user_message}")
    
    try:
        response = agent.query(user_message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"response": "⚠️ Error processing query. Check logs."}), 500

# --- 🏥 Health Check (Public) ---

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.2.0"}), 200

@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

# --- 🚀 Production Entry Point ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
