import os
import io
import logging
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, Response, send_file, session
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash

from core.database import engine, Base, SessionLocal, UserQuery
from core.auth import User, create_default_admin
from core.agent import DataScienceAgent
from utils.email import send_report_email
from utils.pdf_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-123")
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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

try:
    logger.info("🔄 Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized.")
    create_default_admin()
except Exception as e:
    logger.error(f"❌ Database init failed: {e}")

agent = DataScienceAgent()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('register.html', error="Username and password required")
        if len(username) < 3:
            return render_template('register.html', error="Username must be at least 3 characters")
        if len(password) < 6:
            return render_template('register.html', error="Password must be at least 6 characters")
        db = SessionLocal()
        try:
            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                return render_template('register.html', error="Username already exists")
            new_user = User(username=username, password_hash=generate_password_hash(password))
            db.add(new_user)
            db.commit()
            logger.info(f"✅ New user registered: {username}")
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Registration error: {e}")
            db.rollback()
            return render_template('register.html', error="System error. Try again.")
        finally:
            db.close()
    return render_template('register.html')

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
                logger.warning(f"⚠️ Failed login: {username}")
                return render_template('login.html', error="Invalid username or password")
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template('login.html', error="System error.")
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
    ai_mode = data.get('ai_mode', False)  # ✅ Frontend se AI mode flag aayega
    logger.info(f"💬 User Query (AI Mode: {ai_mode}): {user_message}")
    try:
        if ai_mode:
            # ✅ AI Conversational Mode
            response = agent.conversational_query(user_message, user_id=current_user.id)
        else:
            # ✅ Normal Command Mode (same as before)
            response = agent.query(user_message, user_id=current_user.id)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"response": "⚠️ Error processing query."}), 500

@app.route('/send_report', methods=['POST'])
@login_required
def send_report():
    try:
        data = request.get_json()
        client_email = data.get('email')
        if not client_email:
            return jsonify({"error": "Email required"}), 400
        pdf_filename = f"static/report_{current_user.username}_{int(datetime.now().timestamp())}.pdf"
        db = SessionLocal()
        last_query = db.query(UserQuery).filter_by(user_id=current_user.id).order_by(UserQuery.timestamp.desc()).first()
        insights = last_query.response_text if last_query else "No analysis data available"
        db.close()
        success = generate_pdf_report(pdf_filename, client_email, insights, 'static/plot.png')
        if success:
            email_sent = send_report_email(
                to_email=client_email,
                subject="📊 Your Data Analysis Report",
                body=f"Hi,\n\nPlease find attached your data analysis report.\n\nRegards,\nData Scientist Agent",
                attachment_path=pdf_filename
            )
            if email_sent:
                logger.info(f"✅ Report sent to {client_email}")
                return jsonify({"message": "✅ Report sent successfully!"}), 200
            else:
                logger.error("❌ Email failed")
                return jsonify({"error": "PDF created but email failed."}), 500
        else:
            logger.error("❌ PDF generation failed")
            return jsonify({"error": "PDF generation failed"}), 500
    except Exception as e:
        logger.error(f"Report error: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.9.0"}), 200

# ✅ PNG Routes
@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

@app.route('/dashboard.png')
def dashboard_png():
    return send_from_directory('static', 'dashboard.png')

# ✅ CSV — Memory se serve
@app.route('/download/csv')
@login_required
def download_csv():
    if agent.df is None:
        return jsonify({"error": "Pehle file upload karo!"}), 400
    csv_buffer = io.StringIO()
    agent.df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8-sig')
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={
            "Content-Disposition": "attachment; filename=data_export.csv",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )

# ✅ Excel — Memory se serve
@app.route('/download/excel')
@login_required
def download_excel():
    if agent.df is None:
        return jsonify({"error": "Pehle file upload karo!"}), 400
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        agent.df.to_excel(writer, sheet_name='Raw Data', index=False)
        agent.df.describe().to_excel(writer, sheet_name='Summary Stats')
        missing_df = pd.DataFrame({
            'Column': agent.df.isnull().sum().index,
            'Missing Count': agent.df.isnull().sum().values
        })
        missing_df.to_excel(writer, sheet_name='Missing Values', index=False)
    excel_buffer.seek(0)
    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='analysis_report.xlsx'
    )

# ✅ HTML — Memory se serve
@app.route('/download/html')
@login_required
def download_html():
    if agent.df is None:
        return jsonify({"error": "Pehle file upload karo!"}), 400
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #667eea; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Data Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <h3>📄 Data Preview (First 10 Rows)</h3>
        {agent.df.head(10).to_html(index=False)}
        <h3>📈 Summary Statistics</h3>
        {agent.df.describe().to_html()}
        <p style="text-align:center; margin-top:30px; color:#888;">Generated by Data Scientist Agent v1.9</p>
    </div>
</body>
</html>"""
    return Response(
        html_content,
        mimetype='text/html',
        headers={"Content-Disposition": "attachment; filename=analysis_report.html"}
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
