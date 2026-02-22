import os
import io
import uuid
import logging
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, redirect, url_for,
                   Response, send_file, session)
from werkzeug.utils import secure_filename
from flask_login import (LoginManager, login_user, login_required,
                         logout_user, current_user)
from werkzeug.security import check_password_hash, generate_password_hash

import razorpay

from core.database import (engine, Base, SessionLocal, UserQuery,
                            AffiliateLink, QueryCount)
from core.auth import (User, create_default_admin,
                       create_session_token, validate_session_token,
                       invalidate_session, get_user_plan, set_user_plan)
from core.agent import DataScienceAgent
from utils.email import send_report_email
from utils.pdf_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-change-this!")
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ── Razorpay ──────────────────────────────────────────────────────
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
rzp_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        rzp_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        logger.info("✅ Razorpay initialized")
    except Exception as e:
        logger.warning(f"⚠️ Razorpay init failed: {e}")

# ── Flask-Login ───────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

FREE_DAILY_LIMIT = 10  # queries per day for free users

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == int(user_id)).first()
    except:
        return None
    finally:
        db.close()

# ── DB Init ───────────────────────────────────────────────────────
try:
    Base.metadata.create_all(bind=engine)
    create_default_admin()
    logger.info("✅ Database ready")
except Exception as e:
    logger.error(f"❌ DB init: {e}")

agent = DataScienceAgent()

# ── Decorators ────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({"error": "Admin only!"}), 403
        return f(*args, **kwargs)
    return decorated

def single_session_check(f):
    """Blocks request if session token doesn't match DB (another device logged in)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if current_user.is_authenticated:
            token = session.get('session_token')
            if token and not validate_session_token(current_user.id, token):
                logout_user()
                session.clear()
                return jsonify({"error": "SESSION_EXPIRED",
                                "message": "Aapka account kisi aur device pe login hua hai!"}), 401
        return f(*args, **kwargs)
    return decorated

def pro_required(feature_name="This feature"):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            plan = get_user_plan(current_user.id)
            if plan != "pro":
                return jsonify({
                    "error": "PRO_REQUIRED",
                    "message": f"⭐ {feature_name} sirf Pro plan mein available hai! Upgrade karo.",
                    "upgrade_url": url_for('pricing')
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

def check_query_limit(f):
    """Limits free users to FREE_DAILY_LIMIT queries/day."""
    @wraps(f)
    def decorated(*args, **kwargs):
        plan = get_user_plan(current_user.id)
        if plan == "free":
            today = datetime.utcnow().strftime('%Y-%m-%d')
            db = SessionLocal()
            try:
                qc = db.query(QueryCount).filter_by(
                    user_id=current_user.id, date=today).first()
                count = qc.count if qc else 0
                if count >= FREE_DAILY_LIMIT:
                    return jsonify({
                        "response": f"⚠️ Aapki aaj ki {FREE_DAILY_LIMIT} free queries khatam ho gayi hain!\n\n⭐ Pro plan upgrade karo unlimited queries ke liye.",
                        "limit_reached": True
                    })
                # Increment
                if qc:
                    qc.count += 1
                else:
                    db.add(QueryCount(user_id=current_user.id, date=today, count=1))
                db.commit()
            except Exception as e:
                logger.error(f"Query count error: {e}")
            finally:
                db.close()
        return f(*args, **kwargs)
    return decorated

# ── Auth Routes ───────────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('register.html', error="Username and password required")
        if len(username) < 3:
            return render_template('register.html', error="Username min 3 characters")
        if len(password) < 6:
            return render_template('register.html', error="Password min 6 characters")
        db = SessionLocal()
        try:
            if db.query(User).filter(User.username == username).first():
                return render_template('register.html', error="Username already exists")
            user = User(username=username,
                        password_hash=generate_password_hash(password))
            db.add(user)
            db.commit()
            return redirect(url_for('login'))
        except Exception as e:
            db.rollback()
            return render_template('register.html', error="System error")
        finally:
            db.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                # ✅ Create new session token — kicks out any other device
                token = create_session_token(user.id)
                session['session_token'] = token
                return redirect(url_for('home'))
            return render_template('login.html', error="Invalid credentials")
        except Exception as e:
            return render_template('login.html', error="System error")
        finally:
            db.close()
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    invalidate_session(current_user.id)
    session.clear()
    logout_user()
    return redirect(url_for('login'))

# ── Main Pages ────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/pricing')
@login_required
def pricing():
    return render_template('pricing.html',
                           plan=get_user_plan(current_user.id),
                           razorpay_key=RAZORPAY_KEY_ID)

# ── User Info ─────────────────────────────────────────────────────
@app.route('/get_username')
@login_required
def get_username():
    plan = get_user_plan(current_user.id)
    today = datetime.utcnow().strftime('%Y-%m-%d')
    db = SessionLocal()
    try:
        qc = db.query(QueryCount).filter_by(
            user_id=current_user.id, date=today).first()
        queries_used = qc.count if qc else 0
    finally:
        db.close()
    return jsonify({
        "username": current_user.username,
        "is_admin": current_user.is_admin,
        "plan": plan,
        "queries_used": queries_used,
        "queries_limit": FREE_DAILY_LIMIT if plan == "free" else 99999
    })

# ── Affiliate Links (public GET, admin POST/DELETE) ───────────────
@app.route('/affiliate_links', methods=['GET'])
@login_required
def get_affiliate_links():
    db = SessionLocal()
    try:
        links = db.query(AffiliateLink).filter_by(is_active=True).all()
        return jsonify([{
            "id": l.id, "title": l.title,
            "url": l.url, "description": l.description
        } for l in links])
    finally:
        db.close()

@app.route('/admin/affiliate', methods=['POST'])
@login_required
@admin_required
def add_affiliate():
    data = request.get_json()
    db = SessionLocal()
    try:
        link = AffiliateLink(
            title=data.get('title','Sponsored'),
            url=data.get('url',''),
            description=data.get('description',''),
            is_active=True
        )
        db.add(link)
        db.commit()
        return jsonify({"message": "✅ Affiliate link added!", "id": link.id})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/admin/affiliate/<int:link_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_affiliate(link_id):
    db = SessionLocal()
    try:
        link = db.query(AffiliateLink).filter_by(id=link_id).first()
        if link:
            link.is_active = False
            db.commit()
            return jsonify({"message": "✅ Link removed!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/admin/affiliate/<int:link_id>', methods=['PUT'])
@login_required
@admin_required
def update_affiliate(link_id):
    data = request.get_json()
    db = SessionLocal()
    try:
        link = db.query(AffiliateLink).filter_by(id=link_id).first()
        if link:
            link.title = data.get('title', link.title)
            link.url = data.get('url', link.url)
            link.description = data.get('description', link.description)
            db.commit()
            return jsonify({"message": "✅ Link updated!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Admin Panel ───────────────────────────────────────────────────
@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    return render_template('admin.html')

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    from core.database import Subscription
    db = SessionLocal()
    try:
        users = db.query(User).all()
        result = []
        for u in users:
            sub = db.query(Subscription).filter_by(user_id=u.id, is_active=True).first()
            result.append({
                "id": u.id, "username": u.username,
                "is_admin": u.is_admin,
                "plan": sub.plan if sub else "free"
            })
        return jsonify(result)
    finally:
        db.close()

@app.route('/admin/set_plan', methods=['POST'])
@login_required
@admin_required
def admin_set_plan():
    data = request.get_json()
    user_id = data.get('user_id')
    plan = data.get('plan', 'free')
    expires_days = data.get('expires_days')
    expires_at = (datetime.utcnow() + timedelta(days=int(expires_days))
                  if expires_days else None)
    set_user_plan(user_id, plan, expires_at=expires_at)
    return jsonify({"message": f"✅ User {user_id} plan set to {plan}"})

# ── Razorpay Payment ──────────────────────────────────────────────
@app.route('/create_order', methods=['POST'])
@login_required
def create_order():
    if not rzp_client:
        return jsonify({"error": "Payment gateway not configured"}), 500
    data = request.get_json()
    plan = data.get('plan', 'pro')
    # Amount in paise (₹499 = 49900)
    amount = 49900 if plan == 'pro' else 49900
    try:
        order = rzp_client.order.create({
            "amount": amount,
            "currency": "INR",
            "receipt": f"order_{current_user.id}_{int(datetime.utcnow().timestamp())}",
            "notes": {"user_id": str(current_user.id), "plan": plan}
        })
        return jsonify({
            "order_id": order['id'],
            "amount": amount,
            "currency": "INR",
            "key": RAZORPAY_KEY_ID
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/verify_payment', methods=['POST'])
@login_required
def verify_payment():
    data = request.get_json()
    try:
        params = {
            'razorpay_order_id':   data['razorpay_order_id'],
            'razorpay_payment_id': data['razorpay_payment_id'],
            'razorpay_signature':  data['razorpay_signature']
        }
        rzp_client.utility.verify_payment_signature(params)
        # Payment verified — upgrade to pro for 30 days
        set_user_plan(
            current_user.id, "pro",
            razorpay_id=data['razorpay_payment_id'],
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        logger.info(f"✅ Payment verified for user {current_user.id}")
        return jsonify({"message": "✅ Payment successful! Pro plan activated for 30 days."})
    except Exception as e:
        logger.error(f"Payment verify failed: {e}")
        return jsonify({"error": "Payment verification failed"}), 400

# ── Upload ────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
@login_required
@single_session_check
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file and file.filename:
        filename = secure_filename(file.filename)
        try:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = agent.load_data(filename)
            return jsonify({"message": f"✅ Uploaded: {filename}\n{result}"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No file selected"}), 400

# ── Chat ──────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
@login_required
@single_session_check
@check_query_limit
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    ai_mode = data.get('ai_mode', False)

    # Pro-only: AI mode
    if ai_mode:
        plan = get_user_plan(current_user.id)
        if plan != "pro":
            return jsonify({
                "response": "⭐ AI Chat Mode sirf Pro plan mein available hai!\n\nUpgrade karo unlimited AI conversations ke liye.",
                "upgrade_required": True
            })

    try:
        if ai_mode:
            response = agent.conversational_query(user_message, user_id=current_user.id)
        else:
            response = agent.query(user_message, user_id=current_user.id)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "⚠️ Error processing query."}), 500

# ── Downloads (with affiliate popup data) ────────────────────────
@app.route('/download/csv')
@login_required
@single_session_check
def download_csv():
    if agent.df is None:
        return jsonify({"error": "Pehle file upload karo!"}), 400
    csv_buffer = io.StringIO()
    agent.df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8-sig')
    return Response(csv_data, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment; filename=data_export.csv"})

@app.route('/download/excel')
@login_required
@single_session_check
def download_excel():
    # Pro only
    plan = get_user_plan(current_user.id)
    if plan != "pro":
        return jsonify({"error": "PRO_REQUIRED",
                        "message": "⭐ Excel export sirf Pro plan mein!"}), 403
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
    return send_file(excel_buffer,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name='analysis_report.xlsx')

@app.route('/download/html')
@login_required
@single_session_check
def download_html():
    if agent.df is None:
        return jsonify({"error": "Pehle file upload karo!"}), 400
    html_content = f"""<!DOCTYPE html>
<html><head><title>Report</title>
<style>body{{font-family:Arial;padding:20px;background:#f5f5f5}}
.container{{max-width:800px;margin:0 auto;background:white;padding:30px;border-radius:10px}}
h1{{color:#667eea}}table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{border:1px solid #ddd;padding:12px;text-align:left}}
th{{background:#667eea;color:white}}tr:nth-child(even){{background:#f9f9f9}}</style>
</head><body><div class="container">
<h1>🤖 Data Analysis Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h3>Data Preview</h3>{agent.df.head(10).to_html(index=False)}
<h3>Summary</h3>{agent.df.describe().to_html()}
</div></body></html>"""
    return Response(html_content, mimetype='text/html',
                    headers={"Content-Disposition": "attachment; filename=report.html"})

# ── PNG Routes ────────────────────────────────────────────────────
@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')

@app.route('/dashboard.png')
def dashboard_png():
    # Pro only for dashboard
    if current_user.is_authenticated:
        plan = get_user_plan(current_user.id)
        if plan != "pro":
            return jsonify({"error": "PRO_REQUIRED"}), 403
    return send_from_directory('static', 'dashboard.png')

# ── Send Report ───────────────────────────────────────────────────
@app.route('/send_report', methods=['POST'])
@login_required
@single_session_check
def send_report():
    try:
        data = request.get_json()
        client_email = data.get('email')
        if not client_email:
            return jsonify({"error": "Email required"}), 400
        pdf_filename = f"static/report_{current_user.username}_{int(datetime.now().timestamp())}.pdf"
        db = SessionLocal()
        last_query = db.query(UserQuery).filter_by(
            user_id=current_user.id).order_by(UserQuery.timestamp.desc()).first()
        insights = last_query.response_text if last_query else "No data"
        db.close()
        success = generate_pdf_report(pdf_filename, client_email, insights, 'static/plot.png')
        if success:
            email_sent = send_report_email(
                to_email=client_email,
                subject="📊 Your Data Analysis Report",
                body="Hi,\n\nPlease find attached your report.\n\nRegards,\nDS Agent",
                attachment_path=pdf_filename
            )
            if email_sent:
                return jsonify({"message": "✅ Report sent!"})
            return jsonify({"error": "Email failed"}), 500
        return jsonify({"error": "PDF generation failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "3.0.0"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
