import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash

from core.database import engine, Base, SessionLocal
from core.auth import User, create_default_admin
from core.agent import DataScienceAgent

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-123")
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

# Database Init
try:
    logger.info("🔄 Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized successfully.")
    create_default_admin()
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")

agent = DataScienceAgent()

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                logger.info(f"✅ User logged in: {username}")
                return redirect(url_for('home'))
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
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

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
        return jsonify({"response": "⚠️ Error processing query."}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.2.0"}), 200

@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
