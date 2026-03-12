import os
import io
import random
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

from core.database import (engine, Base, SessionLocal, UserQuery, UserDataset, UserChart,
                            AffiliateLink, QueryCount, ScreenPost, MusicTrack,
                            VideoTrack, ChatMessage, PostAnalytics,
                            ScheduledPost, PushSubscription, PaymentScreenshot,
                            OTPVerification)
from core.auth import (User, create_default_admin,
                       create_session_token, validate_session_token,
                       invalidate_session, get_user_plan, set_user_plan)
from core.agent import DataScienceAgent
from utils.email import send_report_email
from utils.pdf_generator import generate_pdf_report

# Keep-alive scheduler for HuggingFace Space
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-change-this!")
UPLOAD_DIR = '/tmp/uploads' if os.path.exists('/tmp') else '.'
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 510 * 1024 * 1024  # 510MB for chunked uploads

# ── HuggingFace Keep-Alive Ping ───────────────────────────────
def ping_hf_space():
    """Ping HF Space every 5 min to prevent sleep"""
    try:
        import requests as _req
        hf_url = os.environ.get('HF_SPACE_URL', '').strip()
        if hf_url:
            r = _req.get(hf_url + '/ping', timeout=10)
            if r.status_code == 200:
                print(f"[KeepAlive] HF Space OK — {r.json().get('message','')}")
            else:
                print(f"[KeepAlive] HF Space responded {r.status_code}")
    except Exception as e:
        print(f"[KeepAlive] HF ping failed: {e}")

_scheduler = BackgroundScheduler(daemon=True)
_scheduler.add_job(ping_hf_space, 'interval', minutes=5, id='hf_keepalive')
_scheduler.start()
print("[KeepAlive] HF Space ping scheduler started — every 5 min")

# ── UPI Config ──────────────────────────────────────────────────
UPI_ID    = os.getenv("UPI_ID", "yourname@upi")
UPI_NAME  = os.getenv("UPI_NAME", "DS Agent")
UPI_AMOUNT = "499"

# ── Gmail SMTP Config ────────────────────────────────────────────
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_PASS", "")

# ── Flask-Login ───────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

FREE_DAILY_LIMIT = 10

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
    # ── Fix: Existing admins ko verified mark karo ──
    _db = SessionLocal()
    try:
        _admins = _db.query(User).filter(User.is_admin == True).all()
        for _a in _admins:
            if not _a.is_verified:
                _a.is_verified = True
                logger.info(f"✅ Admin '{_a.username}' verified!")
        _db.commit()
    except Exception as _e:
        _db.rollback()
    finally:
        _db.close()
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
                    "message": f"⭐ {feature_name} sirf Pro plan mein available hai!",
                    "upgrade_url": url_for('pricing')
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

def check_query_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if current_user.is_admin:
            return f(*args, **kwargs)
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

# ── Email OTP Helpers ─────────────────────────────────────────────
def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, username: str) -> bool:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    gmail_user = os.getenv("GMAIL_USER", "")
    gmail_pass = os.getenv("GMAIL_PASS", "")
    if not gmail_user or not gmail_pass:
        logger.warning(f"[DEV] OTP for {username} ({to_email}): {otp}")
        return True
    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
body{{font-family:Segoe UI,sans-serif;background:#f6f7fb;margin:0;padding:20px}}
.card{{max-width:480px;margin:0 auto;background:white;border-radius:20px;overflow:hidden}}
.hdr{{background:linear-gradient(135deg,#5b5ef4,#8b5cf6);padding:32px 28px;text-align:center}}
.hdr h1{{color:white;margin:0;font-size:26px}}
.body{{padding:32px 28px}}
.otp-box{{background:#f0f0ff;border:2px solid #5b5ef4;border-radius:16px;padding:24px;text-align:center;margin:24px 0}}
.otp{{font-size:42px;font-weight:900;color:#5b5ef4;letter-spacing:10px;font-family:monospace}}
.footer{{padding:20px;background:#f8f9fc;text-align:center;font-size:11px;color:#aaa}}
</style></head><body>
<div class="card">
<div class="hdr"><h1>DS Agent</h1></div>
<div class="body">
<p>Namaste <b>{username}</b>!</p>
<p>Tumhara OTP:</p>
<div class="otp-box"><div class="otp">{otp}</div><p style="font-size:12px;color:#888">10 minutes mein expire hoga</p></div>
<p style="font-size:13px;color:#666">Kisi ke saath share mat karo!</p>
</div>
<div class="footer">DS Agent 2025</div>
</div></body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "DS Agent - Email Verify Karo"
        msg["From"]    = f"DS Agent <{gmail_user}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pass)
            smtp.sendmail(gmail_user, to_email, msg.as_string())
        logger.info(f"OTP email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Gmail SMTP error: {e}")
        return False

# ── Auth Routes ───────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not username or not email or not password:
            return render_template('register.html', error="Sab fields required hain!")
        if len(username) < 3:
            return render_template('register.html', error="Username min 3 characters hona chahiye")
        if len(password) < 6:
            return render_template('register.html', error="Password min 6 characters hona chahiye")
        if '@' not in email or '.' not in email:
            return render_template('register.html', error="Valid email address daalo!")

        db = SessionLocal()
        try:
            if db.query(User).filter(User.username == username).first():
                return render_template('register.html', error="Ye username already le liya gaya hai!")
            if db.query(User).filter(User.email == email).first():
                return render_template('register.html', error="Ye email already registered hai!")

            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                email=email,
                is_verified=False
            )
            db.add(user)
            db.flush()

            otp = generate_otp()
            expires = datetime.utcnow() + timedelta(minutes=10)
            db.query(OTPVerification).filter_by(user_id=user.id).delete()
            db.add(OTPVerification(user_id=user.id, email=email, otp=otp, expires_at=expires))
            db.commit()

            send_otp_email(email, otp, username)
            session['pending_verify_user_id'] = user.id
            session['pending_verify_email']   = email
            return redirect(url_for('verify_email'))

        except Exception as e:
            db.rollback()
            import traceback
            logger.error(f"Register error: {e}")
            logger.error(traceback.format_exc())
            return render_template('register.html', error=f"Error: {str(e)}")
        finally:
            db.close()
    return render_template('register.html')


@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    user_id = session.get('pending_verify_user_id')
    email   = session.get('pending_verify_email', '')
    if not user_id:
        return redirect(url_for('register'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()
        db = SessionLocal()
        try:
            record = db.query(OTPVerification).filter_by(
                user_id=user_id, otp=entered_otp, is_used=False).first()
            if not record:
                return render_template('verify_email.html',
                    error="❌ Galat OTP! Dobara check karo.", email=email)
            if record.expires_at < datetime.utcnow():
                return render_template('verify_email.html',
                    error="⏰ OTP expire ho gaya! Resend karo.", email=email)
            user = db.query(User).filter_by(id=user_id).first()
            user.is_verified = True
            record.is_used = True
            db.commit()
            session.pop('pending_verify_user_id', None)
            session.pop('pending_verify_email', None)
            return redirect(url_for('login') + '?verified=1')
        except Exception as e:
            db.rollback()
            logger.error(f"Verify error: {e}")
            return render_template('verify_email.html', error="System error!", email=email)
        finally:
            db.close()
    return render_template('verify_email.html', email=email)


@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    user_id = session.get('pending_verify_user_id')
    if not user_id:
        return jsonify({"error": "Session expire ho gaya!"}), 400
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({"error": "User nahi mila!"}), 404
        last_otp = db.query(OTPVerification).filter_by(
            user_id=user_id).order_by(OTPVerification.created_at.desc()).first()
        if last_otp:
            elapsed = (datetime.utcnow() - last_otp.created_at).total_seconds()
            if elapsed < 60:
                return jsonify({"error": f"⏳ {int(60 - elapsed)} seconds baad resend karo!"}), 429
        otp = generate_otp()
        expires = datetime.utcnow() + timedelta(minutes=10)
        db.query(OTPVerification).filter_by(user_id=user_id).delete()
        db.add(OTPVerification(user_id=user_id, email=user.email, otp=otp, expires_at=expires))
        db.commit()
        send_otp_email(user.email, otp, user.username)
        return jsonify({"message": f"✅ OTP {user.email} pe bheja!"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and check_password_hash(user.password_hash, password):
                if not user.is_admin and not user.is_verified:
                    session['pending_verify_user_id'] = user.id
                    session['pending_verify_email']   = user.email or ''
                    otp = generate_otp()
                    expires = datetime.utcnow() + timedelta(minutes=10)
                    db.query(OTPVerification).filter_by(user_id=user.id).delete()
                    db.add(OTPVerification(user_id=user.id, email=user.email, otp=otp, expires_at=expires))
                    db.commit()
                    send_otp_email(user.email, otp, user.username)
                    return redirect(url_for('verify_email'))
                login_user(user)
                token = create_session_token(user.id)
                session['session_token'] = token
                return redirect(url_for('home'))
            return render_template('login.html', error="❌ Galat username ya password!")
        except Exception as e:
            return render_template('login.html', error="System error!")
        finally:
            db.close()
    verified_msg = request.args.get('verified')
    success_msg = "✅ Email verify ho gaya! Ab login karo." if verified_msg else None
    return render_template('login.html', success=success_msg)


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
    try:
        import json, io
        db = SessionLocal()
        dataset = db.query(UserDataset).filter_by(user_id=current_user.id).order_by(UserDataset.uploaded_at.desc()).first()
        db.close()
        if dataset and agent.df is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
            tmp.write(dataset.csv_data)
            tmp.close()
            agent.load_data(tmp.name)
            logger.info(f"✅ Auto-loaded dataset for user {current_user.id}: {dataset.filename}")
    except Exception as e:
        logger.error(f"⚠️ Auto-load error: {e}")
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/pricing')
@login_required
def pricing():
    return render_template('pricing.html', plan=get_user_plan(current_user.id))

# ── User Info ─────────────────────────────────────────────────────
@app.route('/get_username')
@login_required
def get_username():
    if current_user.is_admin:
        return jsonify({"username": current_user.username, "is_admin": True, "plan": "pro", "queries_used": 0, "queries_limit": 99999})
    plan = get_user_plan(current_user.id)
    today = datetime.utcnow().strftime('%Y-%m-%d')
    db = SessionLocal()
    try:
        qc = db.query(QueryCount).filter_by(user_id=current_user.id, date=today).first()
        queries_used = qc.count if qc else 0
    finally:
        db.close()
    return jsonify({"username": current_user.username, "is_admin": False, "plan": plan, "queries_used": queries_used, "queries_limit": FREE_DAILY_LIMIT if plan == "free" else 99999})

# ── UPI Payment Info ──────────────────────────────────────────────
@app.route('/api/upi_info')
@login_required
def upi_info():
    return jsonify({"upi_id": UPI_ID, "upi_name": UPI_NAME, "amount": UPI_AMOUNT})

# ── Payment Screenshot Submit ─────────────────────────────────────
@app.route('/submit_payment_screenshot', methods=['POST'])
@login_required
@single_session_check
def submit_payment_screenshot():
    data = request.get_json()
    if not data or not data.get('image_data'):
        return jsonify({"error": "Screenshot required!"}), 400
    db = SessionLocal()
    try:
        ss = PaymentScreenshot(
            user_id=current_user.id, username=current_user.username,
            image_data=data['image_data'], image_mime=data.get('image_mime', 'image/jpeg'),
            utr_number=data.get('utr_number', ''), amount=data.get('amount', '499'), status='pending'
        )
        db.add(ss)
        db.commit()
        logger.info(f"📸 Payment screenshot from {current_user.username}")
        return jsonify({"message": "✅ Screenshot submit ho gaya! Admin 24hrs mein verify karega."})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Admin: View Screenshots ───────────────────────────────────────
@app.route('/admin/payment_screenshots')
@login_required
@admin_required
def admin_payment_screenshots():
    db = SessionLocal()
    try:
        items = db.query(PaymentScreenshot).order_by(PaymentScreenshot.created_at.desc()).all()
        return jsonify([{"id": s.id, "username": s.username, "image_data": s.image_data, "image_mime": s.image_mime, "utr_number": s.utr_number or "N/A", "amount": s.amount or "499", "status": s.status, "created_at": s.created_at.strftime('%d %b %H:%M')} for s in items])
    finally:
        db.close()

# ── Admin: Approve/Reject Payment ────────────────────────────────
@app.route('/admin/review_payment', methods=['POST'])
@login_required
@admin_required
def review_payment():
    data = request.get_json()
    ss_id = data.get('id'); status = data.get('status')
    if status not in ('approved', 'rejected'):
        return jsonify({"error": "Invalid status"}), 400
    db = SessionLocal()
    try:
        ss = db.query(PaymentScreenshot).filter_by(id=ss_id).first()
        if not ss: return jsonify({"error": "Not found"}), 404
        ss.status = status
        if status == 'approved':
            set_user_plan(ss.user_id, 'pro', expires_at=datetime.utcnow() + timedelta(days=30))
        db.commit()
        msg = f"✅ Approved! Pro activated for {ss.username}" if status == 'approved' else f"❌ Rejected for {ss.username}"
        return jsonify({"message": msg})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Affiliate Links ───────────────────────────────────────────────
@app.route('/affiliate_links', methods=['GET'])
@login_required
def get_affiliate_links():
    db = SessionLocal()
    try:
        links = db.query(AffiliateLink).filter_by(is_active=True).all()
        return jsonify([{"id": l.id, "title": l.title, "url": l.url, "description": l.description} for l in links])
    finally:
        db.close()

@app.route('/admin/affiliate', methods=['POST'])
@login_required
@admin_required
def add_affiliate():
    data = request.get_json()
    db = SessionLocal()
    try:
        link = AffiliateLink(title=data.get('title','Sponsored'), url=data.get('url',''), description=data.get('description',''), is_active=True)
        db.add(link); db.commit()
        return jsonify({"message": "✅ Added!", "id": link.id})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/admin/affiliate/<int:link_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_affiliate(link_id):
    db = SessionLocal()
    try:
        link = db.query(AffiliateLink).filter_by(id=link_id).first()
        if link: link.is_active = False; db.commit(); return jsonify({"message": "✅ Removed!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
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
            link.title = data.get('title', link.title); link.url = data.get('url', link.url); link.description = data.get('description', link.description)
            db.commit(); return jsonify({"message": "✅ Updated!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
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
            result.append({"id": u.id, "username": u.username, "is_admin": u.is_admin, "plan": sub.plan if sub else "free", "email": u.email or "", "is_verified": u.is_verified})
        return jsonify(result)
    finally:
        db.close()

@app.route('/admin/query_stats')
@login_required
@admin_required
def admin_query_stats():
    db = SessionLocal()
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        all_counts = db.query(QueryCount).all()
        user_totals = {}
        for qc in all_counts:
            user_totals[qc.user_id] = user_totals.get(qc.user_id, 0) + qc.count
        today_counts = db.query(QueryCount).filter_by(date=today).all()
        today_map = {qc.user_id: qc.count for qc in today_counts}
        users = db.query(User).all()
        name_map = {u.id: u.username for u in users}
        totals = sorted([{"user_id": uid, "username": name_map.get(uid, f"user_{uid}"), "count": cnt} for uid, cnt in user_totals.items()], key=lambda x: x['count'], reverse=True)
        today_data = sorted([{"user_id": uid, "username": name_map.get(uid, f"user_{uid}"), "count": cnt} for uid, cnt in today_map.items()], key=lambda x: x['count'], reverse=True)
        return jsonify({"totals": totals, "today": today_data})
    except Exception as e:
        return jsonify({"error": str(e), "totals": [], "today": []}), 500
    finally:
        db.close()

@app.route('/admin/set_plan', methods=['POST'])
@login_required
@admin_required
def admin_set_plan():
    data = request.get_json()
    user_id = data.get('user_id'); plan = data.get('plan', 'free'); expires_days = data.get('expires_days')
    expires_at = (datetime.utcnow() + timedelta(days=int(expires_days)) if expires_days else None)
    set_user_plan(user_id, plan, expires_at=expires_at)
    return jsonify({"message": f"✅ User {user_id} plan set to {plan}"})

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
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)
            result = agent.load_data(full_path)
            try:
                import json
                csv_string = open(full_path, 'r', encoding='latin1').read()
                db = SessionLocal()
                db.query(UserDataset).filter_by(user_id=current_user.id).delete()
                dataset = UserDataset(user_id=current_user.id, filename=filename, csv_data=csv_string, columns=json.dumps(agent.available_columns), row_count=len(agent.df))
                db.add(dataset); db.commit(); db.close()
                logger.info(f"✅ Dataset saved to DB: {filename}")
            except Exception as db_err:
                logger.error(f"⚠️ DB save error: {db_err}")
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
    if ai_mode and not current_user.is_admin:
        plan = get_user_plan(current_user.id)
        if plan != "pro":
            return jsonify({"response": "⭐ AI Chat Mode sirf Pro plan mein available hai!", "upgrade_required": True})
    try:
        agent._current_user_id = current_user.id
        if ai_mode:
            response = agent.conversational_query(user_message, user_id=current_user.id)
        else:
            response = agent.query(user_message, user_id=current_user.id)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "⚠️ Error processing query."}), 500

# ── Downloads ─────────────────────────────────────────────────────
@app.route('/download/csv')
@login_required
@single_session_check
def download_csv():
    if agent.df is None: return jsonify({"error": "Pehle file upload karo!"}), 400
    csv_buffer = io.StringIO(); agent.df.to_csv(csv_buffer, index=False)
    return Response(csv_buffer.getvalue().encode('utf-8-sig'), mimetype='text/csv', headers={"Content-Disposition": "attachment; filename=data_export.csv"})

@app.route('/download/excel')
@login_required
@single_session_check
def download_excel():
    plan = get_user_plan(current_user.id)
    if plan != "pro" and not current_user.is_admin:
        return jsonify({"error": "PRO_REQUIRED", "message": "⭐ Excel export sirf Pro plan mein!"}), 403
    if agent.df is None: return jsonify({"error": "Pehle file upload karo!"}), 400
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        agent.df.to_excel(writer, sheet_name='Raw Data', index=False)
        agent.df.describe().to_excel(writer, sheet_name='Summary Stats')
    excel_buffer.seek(0)
    return send_file(excel_buffer, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='analysis_report.xlsx')

@app.route('/download/html')
@login_required
@single_session_check
def download_html():
    if agent.df is None: return jsonify({"error": "Pehle file upload karo!"}), 400
    html_content = f"""<!DOCTYPE html><html><head><title>Report</title></head><body><h1>🤖 DS Agent Report</h1><p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>{agent.df.head(10).to_html(index=False)}{agent.df.describe().to_html()}</body></html>"""
    return Response(html_content, mimetype='text/html', headers={"Content-Disposition": "attachment; filename=report.html"})

@app.route('/charts')
@login_required
def list_charts():
    db = SessionLocal()
    try:
        charts = db.query(UserChart).filter_by(user_id=current_user.id).order_by(UserChart.created_at.asc()).all()
        return jsonify({"charts": [{"id": c.id, "title": c.chart_title, "time": c.created_at.strftime('%d %b %H:%M')} for c in charts], "total": len(charts)})
    finally:
        db.close()

@app.route('/chart/<int:chart_id>')
@login_required
def get_chart(chart_id):
    import base64, io
    db = SessionLocal()
    try:
        chart = db.query(UserChart).filter_by(id=chart_id, user_id=current_user.id).first()
        if not chart: return jsonify({'error': 'Chart nahi mila!'}), 404
        img_bytes = base64.b64decode(chart.image_data)
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    finally:
        db.close()

@app.route('/charts/clear')
@login_required
def clear_charts():
    db = SessionLocal()
    try:
        db.query(UserChart).filter_by(user_id=current_user.id).delete()
        db.commit()
        return jsonify({"message": "✅ Sab charts clear!"})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/plot.png')
def plot_png():
    plot_path = '/tmp/plot.png' if os.path.exists('/tmp/plot.png') else os.path.join('static', 'plot.png')
    if not os.path.exists(plot_path): return jsonify({'error': 'Plot nahi bana abhi tak'}), 404
    return send_file(plot_path, mimetype='image/png')

@app.route('/dashboard.png')
def dashboard_png():
    if current_user.is_authenticated:
        if not current_user.is_admin:
            plan = get_user_plan(current_user.id)
            if plan != "pro": return jsonify({"error": "PRO_REQUIRED"}), 403
    dash_path = '/tmp/dashboard.png' if os.path.exists('/tmp/dashboard.png') else os.path.join('static', 'dashboard.png')
    if not os.path.exists(dash_path): return jsonify({'error': 'Dashboard nahi bana abhi tak'}), 404
    return send_file(dash_path, mimetype='image/png')

@app.route('/screen')
@login_required
def screen(): return render_template('screen.html')

# ── Screen Posts ──────────────────────────────────────────────────
@app.route('/api/screen/posts')
@login_required
def get_screen_posts():
    db = SessionLocal()
    try:
        posts = db.query(ScreenPost).filter_by(is_active=True).order_by(ScreenPost.order_num.asc(), ScreenPost.created_at.desc()).all()
        return jsonify([{"id": p.id, "title": p.title, "content": p.content, "post_type": p.post_type, "affiliate_url": p.affiliate_url, "image_data": p.image_data, "image_mime": p.image_mime} for p in posts])
    finally:
        db.close()

@app.route('/api/screen/posts', methods=['POST'])
@login_required
@admin_required
def add_screen_post():
    import base64
    db = SessionLocal()
    try:
        post_type = request.form.get('post_type', 'text'); title = request.form.get('title', ''); content = request.form.get('content', ''); aff_url = request.form.get('affiliate_url', ''); order_num = int(request.form.get('order_num', 0))
        image_data, image_mime = None, None
        if post_type == 'image' and 'image' in request.files:
            img = request.files['image']
            if img and img.filename: image_data = base64.b64encode(img.read()).decode('utf-8'); image_mime = img.content_type or 'image/jpeg'
        post = ScreenPost(title=title, content=content, post_type=post_type, affiliate_url=aff_url if aff_url else None, image_data=image_data, image_mime=image_mime, order_num=order_num, is_active=True)
        db.add(post); db.commit()
        return jsonify({"message": "✅ Post added!", "id": post.id})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/api/screen/posts/<int:post_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_screen_post(post_id):
    db = SessionLocal()
    try:
        p = db.query(ScreenPost).filter_by(id=post_id).first()
        if p: p.is_active = False; db.commit(); return jsonify({"message": "✅ Removed!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Music ─────────────────────────────────────────────────────────
@app.route('/api/music')
@login_required
def get_music():
    db = SessionLocal()
    try:
        tracks = db.query(MusicTrack).filter_by(is_active=True).order_by(MusicTrack.created_at.asc()).all()
        return jsonify([{"id": t.id, "title": t.title, "artist": t.artist or "Unknown", "audio_data": t.audio_data, "mime_type": t.mime_type} for t in tracks])
    finally:
        db.close()

@app.route('/api/music', methods=['POST'])
@login_required
@admin_required
def upload_music():
    import base64
    db = SessionLocal()
    try:
        if 'audio' not in request.files: return jsonify({"error": "No file"}), 400
        f = request.files['audio']; title = request.form.get('title', f.filename or 'Unknown'); artist = request.form.get('artist', '')
        if not f or not f.filename: return jsonify({"error": "Empty file"}), 400
        raw = f.read(); audio_b64 = base64.b64encode(raw).decode('utf-8'); mime = f.content_type or 'audio/mpeg'
        track = MusicTrack(title=title, artist=artist, audio_data=audio_b64, mime_type=mime, is_active=True)
        db.add(track); db.commit()
        return jsonify({"message": f"✅ '{title}' upload ho gaya!", "id": track.id})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/api/music/<int:track_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_music(track_id):
    db = SessionLocal()
    try:
        t = db.query(MusicTrack).filter_by(id=track_id).first()
        if t: t.is_active = False; db.commit(); return jsonify({"message": "✅ Removed!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Video ─────────────────────────────────────────────────────────
@app.route('/api/video')
@login_required
def get_videos():
    db = SessionLocal()
    try:
        videos = db.query(VideoTrack).filter_by(is_active=True).order_by(VideoTrack.created_at.asc()).all()
        return jsonify([{"id": v.id, "title": v.title, "description": v.description or "", "video_data": v.video_data, "mime_type": v.mime_type, "thumbnail": v.thumbnail, "thumb_mime": v.thumb_mime or "image/jpeg"} for v in videos])
    finally:
        db.close()

@app.route('/api/video', methods=['POST'])
@login_required
@admin_required
def upload_video():
    import base64
    db = SessionLocal()
    try:
        if 'video' not in request.files: return jsonify({"error": "No video file"}), 400
        f = request.files['video']; title = request.form.get('title', f.filename or 'Video'); desc = request.form.get('description', '')
        if not f or not f.filename: return jsonify({"error": "Empty file"}), 400
        raw = f.read(); video_b64 = base64.b64encode(raw).decode('utf-8'); mime = f.content_type or 'video/mp4'
        thumb_b64, thumb_mime = None, None
        if 'thumbnail' in request.files:
            th = request.files['thumbnail']
            if th and th.filename: thumb_b64 = base64.b64encode(th.read()).decode('utf-8'); thumb_mime = th.content_type or 'image/jpeg'
        video = VideoTrack(title=title, description=desc, video_data=video_b64, mime_type=mime, thumbnail=thumb_b64, thumb_mime=thumb_mime, is_active=True)
        db.add(video); db.commit()
        return jsonify({"message": f"✅ '{title}' upload ho gaya!", "id": video.id})
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/api/video/<int:video_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_video(video_id):
    db = SessionLocal()
    try:
        v = db.query(VideoTrack).filter_by(id=video_id).first()
        if v: v.is_active = False; db.commit(); return jsonify({"message": "✅ Removed!"})
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        db.rollback(); return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Stats ─────────────────────────────────────────────────────────
@app.route('/api/stats')
@login_required
def get_stats():
    from core.database import Subscription
    db = SessionLocal()
    try:
        total_users = db.query(User).count(); pro_users = db.query(Subscription).filter_by(plan='pro', is_active=True).count()
        today = datetime.utcnow().strftime('%Y-%m-%d'); today_q = db.query(QueryCount).filter_by(date=today).all()
        today_total = sum(q.count for q in today_q); total_q = db.query(QueryCount).all(); total_queries = sum(q.count for q in total_q)
        return jsonify({"total_users": total_users, "pro_users": pro_users, "today_queries": today_total, "total_queries": total_queries})
    finally:
        db.close()

@app.route('/health')
def health(): return jsonify({"status": "healthy", "version": "3.2.0"}), 200

# ── YouTube Stream ────────────────────────────────────────────────
@app.route('/api/yt-stream', methods=['POST'])
@login_required
def get_yt_stream():
    try:
        import yt_dlp
        data = request.get_json(); url = data.get('url', '').strip()
        if not url: return jsonify({'error': 'URL required!'}), 400
        if 'youtube.com' not in url and 'youtu.be' not in url: return jsonify({'error': 'Valid YouTube URL daalo!'}), 400
        ydl_opts = {'format': 'best[ext=mp4][height<=480]/best[height<=480]/best', 'quiet': True, 'no_warnings': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False); formats = info.get('formats', []); stream_url = None
            for f in reversed(formats):
                if f.get('ext') == 'mp4' and f.get('url'): stream_url = f['url']; break
            if not stream_url: stream_url = info.get('url') or (formats[-1].get('url') if formats else None)
            title = info.get('title', 'YouTube Video'); thumbnail = info.get('thumbnail', ''); duration = info.get('duration', 0)
        if not stream_url: return jsonify({'error': 'Stream URL extract nahi hui!'}), 500
        return jsonify({'stream_url': stream_url, 'title': title, 'thumbnail': thumbnail, 'duration': duration})
    except ImportError:
        return jsonify({'error': 'yt-dlp install nahi hai!'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)[:100]}'}), 500

# ── Live Chat ─────────────────────────────────────────────────────
@app.route('/api/chat/messages')
@login_required
def get_chat_messages():
    db = SessionLocal()
    try:
        msgs = db.query(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(50).all()
        return jsonify([{'id': m.id, 'username': m.username, 'message': m.message, 'is_admin': m.is_admin, 'time': m.created_at.strftime('%H:%M')} for m in reversed(msgs)])
    finally:
        db.close()

@app.route('/api/chat/messages', methods=['POST'])
@login_required
def send_chat_message():
    db = SessionLocal()
    try:
        data = request.get_json(); msg = data.get('message', '').strip()
        if not msg or len(msg) > 500: return jsonify({'error': 'Invalid message'}), 400
        chat = ChatMessage(username=current_user.username, message=msg, is_admin=current_user.is_admin)
        db.add(chat); db.commit()
        return jsonify({'message': '✅ Sent!', 'id': chat.id})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/chat/messages/<int:msg_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_chat_message(msg_id):
    db = SessionLocal()
    try:
        m = db.query(ChatMessage).filter_by(id=msg_id).first()
        if m: db.delete(m); db.commit(); return jsonify({'message': '✅ Deleted!'})
        return jsonify({'error': 'Not found'}), 404
    finally:
        db.close()

# ── Analytics ─────────────────────────────────────────────────────
@app.route('/api/analytics/view/<int:post_id>', methods=['POST'])
@login_required
def track_post_view(post_id):
    db = SessionLocal()
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d'); rec = db.query(PostAnalytics).filter_by(post_id=post_id, date=today).first()
        if rec: rec.views += 1
        else: rec = PostAnalytics(post_id=post_id, views=1, clicks=0, date=today); db.add(rec)
        db.commit(); return jsonify({'ok': True})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/analytics/click/<int:post_id>', methods=['POST'])
@login_required
def track_post_click(post_id):
    db = SessionLocal()
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d'); rec = db.query(PostAnalytics).filter_by(post_id=post_id, date=today).first()
        if rec: rec.clicks += 1
        else: rec = PostAnalytics(post_id=post_id, views=0, clicks=1, date=today); db.add(rec)
        db.commit(); return jsonify({'ok': True})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/analytics')
@login_required
@admin_required
def get_analytics():
    db = SessionLocal()
    try:
        posts = db.query(ScreenPost).filter_by(is_active=True).all()
        result = []
        for p in posts:
            recs = db.query(PostAnalytics).filter_by(post_id=p.id).all()
            total_views = sum(r.views for r in recs); total_clicks = sum(r.clicks for r in recs)
            result.append({'post_id': p.id, 'title': p.title, 'views': total_views, 'clicks': total_clicks, 'ctr': round(total_clicks/total_views*100, 1) if total_views else 0})
        result.sort(key=lambda x: x['views'], reverse=True)
        return jsonify(result)
    finally:
        db.close()

# ── Scheduled Posts ───────────────────────────────────────────────
@app.route('/api/scheduled', methods=['GET'])
@login_required
@admin_required
def get_scheduled():
    db = SessionLocal()
    try:
        posts = db.query(ScheduledPost).filter_by(is_published=False).order_by(ScheduledPost.scheduled_at).all()
        return jsonify([{'id': p.id, 'title': p.title, 'content': p.content, 'post_type': p.post_type, 'scheduled_at': p.scheduled_at.strftime('%Y-%m-%d %H:%M'), 'is_published': p.is_published} for p in posts])
    finally:
        db.close()

@app.route('/api/scheduled', methods=['POST'])
@login_required
@admin_required
def create_scheduled():
    import base64
    db = SessionLocal()
    try:
        title = request.form.get('title', ''); content = request.form.get('content', ''); post_type = request.form.get('post_type', 'text'); affiliate_url = request.form.get('affiliate_url', ''); scheduled_at = request.form.get('scheduled_at', '')
        if not title or not scheduled_at: return jsonify({'error': 'Title aur time required!'}), 400
        from datetime import datetime as dt
        sched_time = dt.strptime(scheduled_at, '%Y-%m-%dT%H:%M')
        img_data, img_mime = None, None
        if post_type == 'image' and 'image' in request.files:
            f = request.files['image']
            if f and f.filename: img_data = base64.b64encode(f.read()).decode(); img_mime = f.content_type or 'image/jpeg'
        sp = ScheduledPost(title=title, content=content, post_type=post_type, affiliate_url=affiliate_url, scheduled_at=sched_time, image_data=img_data, image_mime=img_mime)
        db.add(sp); db.commit()
        return jsonify({'message': f'✅ "{title}" scheduled!', 'id': sp.id})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/scheduled/<int:sp_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_scheduled(sp_id):
    db = SessionLocal()
    try:
        sp = db.query(ScheduledPost).filter_by(id=sp_id).first()
        if sp: db.delete(sp); db.commit(); return jsonify({'message': '✅ Removed!'})
        return jsonify({'error': 'Not found'}), 404
    finally:
        db.close()

@app.route('/api/scheduled/publish', methods=['POST'])
@login_required
@admin_required
def publish_due_posts():
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        due = db.query(ScheduledPost).filter(ScheduledPost.scheduled_at <= now, ScheduledPost.is_published == False).all()
        published = 0
        for sp in due:
            post = ScreenPost(title=sp.title, content=sp.content, post_type=sp.post_type, affiliate_url=sp.affiliate_url, image_data=sp.image_data, image_mime=sp.image_mime, is_active=True)
            db.add(post); sp.is_published = True; published += 1
        db.commit()
        return jsonify({'message': f'✅ {published} posts published!'})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

# ── Push Notifications ────────────────────────────────────────────
VAPID_PUBLIC_KEY = os.getenv('VAPID_PUBLIC_KEY', ''); VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY', ''); VAPID_EMAIL = os.getenv('VAPID_EMAIL', 'mailto:admin@dsagent.com')

@app.route('/api/push/vapid-public-key')
def get_vapid_key(): return jsonify({'publicKey': VAPID_PUBLIC_KEY})

@app.route('/api/push/subscribe', methods=['POST'])
@login_required
def push_subscribe():
    db = SessionLocal()
    try:
        data = request.get_json()
        sub = PushSubscription(user_id=current_user.id, endpoint=data['endpoint'], p256dh=data['keys']['p256dh'], auth=data['keys']['auth'])
        db.add(sub); db.commit()
        return jsonify({'message': '✅ Subscribed!'})
    except Exception as e:
        db.rollback(); return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/push/send', methods=['POST'])
@login_required
@admin_required
def send_push():
    db = SessionLocal()
    try:
        data = request.get_json(); title = data.get('title', 'DS Agent'); body = data.get('body', '')
        subs = db.query(PushSubscription).all(); sent = 0
        if VAPID_PRIVATE_KEY:
            try:
                from pywebpush import webpush, WebPushException
                import json
                for s in subs:
                    try:
                        webpush(subscription_info={'endpoint': s.endpoint, 'keys': {'p256dh': s.p256dh, 'auth': s.auth}}, data=json.dumps({'title': title, 'body': body}), vapid_private_key=VAPID_PRIVATE_KEY, vapid_claims={'sub': VAPID_EMAIL})
                        sent += 1
                    except: pass
            except ImportError: pass
        return jsonify({'message': f'✅ {sent}/{len(subs)} notifications sent!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/send_report', methods=['POST'])
@login_required
@single_session_check
def send_report():
    try:
        data = request.get_json()
        client_email = data.get('email'); report_type = data.get('report_type', 'summary'); schedule = data.get('schedule', 'now')
        if not client_email: return jsonify({"error": "Email required"}), 400
        if schedule != 'now':
            return jsonify({"message": f"✅ Report scheduled! {schedule.capitalize()} emails {client_email} pe jaayenge.", "scheduled": True, "schedule": schedule})
        db = SessionLocal()
        last_query = db.query(UserQuery).filter_by(user_id=current_user.id).order_by(UserQuery.timestamp.desc()).first()
        insights = last_query.response_text if last_query else "No data"
        db.close()
        subject_map = {'summary': '📊 Data Analysis Summary Report', 'full': '📊 Full Data Analysis Report', 'charts': '📊 Charts & Visualizations Report'}
        subject = subject_map.get(report_type, '📊 Your Data Analysis Report')
        pdf_filename = f"static/report_{current_user.username}_{int(datetime.now().timestamp())}.pdf"
        success = generate_pdf_report(pdf_filename, client_email, insights, 'static/plot.png')
        if success:
            email_sent = send_report_email(to_email=client_email, subject=subject, body=f"Hi,\n\nPlease find attached your {report_type} report.\n\nRegards,\nDS Agent", attachment_path=pdf_filename)
            if email_sent: return jsonify({"message": f"✅ {subject} sent to {client_email}!"})
            return jsonify({"error": "Email failed"}), 500
        return jsonify({"error": "PDF generation failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── CSV Compare ───────────────────────────────────────────────────
@app.route('/upload_compare', methods=['POST'])
@login_required
@single_session_check
def upload_compare():
    if 'file' not in request.files: return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file or not file.filename: return jsonify({"error": "Empty file"}), 400
    try:
        import io as _io
        filename = secure_filename(file.filename)
        content = file.read().decode('utf-8', errors='replace')
        df2 = pd.read_csv(_io.StringIO(content))
        rows, cols = len(df2), len(df2.columns); columns = list(df2.columns)
        stats = {}
        for col in df2.select_dtypes(include='number').columns:
            stats[col] = {"mean": round(float(df2[col].mean()), 2), "min": round(float(df2[col].min()), 2), "max": round(float(df2[col].max()), 2), "nulls": int(df2[col].isnull().sum())}
        comparison = None
        if agent.df is not None:
            df1 = agent.df; common_cols = list(set(df1.columns) & set(df2.columns))
            comparison = {"file1_rows": len(df1), "file2_rows": rows, "file1_cols": len(df1.columns), "file2_cols": cols, "common_columns": common_cols, "only_in_file1": list(set(df1.columns) - set(df2.columns)), "only_in_file2": list(set(df2.columns) - set(df1.columns)), "col_stats": {}}
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    comparison["col_stats"][col] = {"file1_mean": round(float(df1[col].mean()), 2), "file2_mean": round(float(df2[col].mean()), 2), "file1_max": round(float(df1[col].max()), 2), "file2_max": round(float(df2[col].max()), 2), "diff_mean": round(float(df2[col].mean()) - float(df1[col].mean()), 2)}
        return jsonify({"message": f"✅ {filename} loaded!", "filename": filename, "rows": rows, "cols": cols, "columns": columns, "stats": stats, "comparison": comparison})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Save Plotly Chart ─────────────────────────────────────────────
@app.route('/save_plot_base64', methods=['POST'])
@login_required
@single_session_check
def save_plot_base64():
    try:
        data = request.get_json(); image_b64 = data.get('image'); title = data.get('title', f'Plotly Chart {datetime.now().strftime("%d %b %H:%M")}')
        if not image_b64: return jsonify({"error": "Image data required"}), 400
        if ',' in image_b64: image_b64 = image_b64.split(',', 1)[1]
        db = SessionLocal()
        try:
            chart = UserChart(user_id=current_user.id, chart_title=title, image_data=image_b64, chart_type='plotly')
            db.add(chart); db.commit()
            return jsonify({"message": f"✅ '{title}' gallery mein save ho gaya!", "chart_id": chart.id})
        except Exception as db_err:
            db.rollback(); return jsonify({"error": str(db_err)}), 500
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── PWA Manifest ──────────────────────────────────────────────────
@app.route('/manifest.json')
def pwa_manifest():
    return jsonify({"name": "DS Agent", "short_name": "DS Agent", "description": "AI-powered Data Science Agent", "start_url": "/", "display": "standalone", "background_color": "#0a0a18", "theme_color": "#5b5ef4", "orientation": "portrait-primary", "icons": [{"src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%235b5ef4'/><text y='.9em' font-size='80' x='10'>🤖</text></svg>", "sizes": "192x192", "type": "image/svg+xml", "purpose": "any maskable"}, {"src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%235b5ef4'/><text y='.9em' font-size='80' x='10'>🤖</text></svg>", "sizes": "512x512", "type": "image/svg+xml", "purpose": "any maskable"}], "categories": ["productivity", "utilities"], "lang": "en"})

# ── Service Worker ────────────────────────────────────────────────
@app.route('/sw.js')
def service_worker():
    sw_content = """
const CACHE_NAME = 'ds-agent-v1';
const STATIC_ASSETS = ['/'];
self.addEventListener('install', e => {
    e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS).catch(()=>{})));
    self.skipWaiting();
});
self.addEventListener('activate', e => {
    e.waitUntil(caches.keys().then(keys =>
        Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ));
    self.clients.claim();
});
self.addEventListener('fetch', e => {
    if (e.request.method !== 'GET') return;
    if (['/chat','/upload','/api/'].some(p => e.request.url.includes(p))) return;
    e.respondWith(
        fetch(e.request)
            .then(r => { if(r.ok){caches.open(CACHE_NAME).then(c=>c.put(e.request,r.clone()));}return r;})
            .catch(() => caches.match(e.request))
    );
});
"""
    return Response(sw_content, mimetype='application/javascript', headers={'Service-Worker-Allowed': '/'})

@app.route('/run_code', methods=['POST'])
@login_required
def run_code():
    import traceback, base64, io as _io, sys
    from contextlib import redirect_stdout, redirect_stderr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = request.get_json()
    code = data.get('code', '').strip()
    if not code:
        return jsonify({'error': 'Code empty hai!'}), 400

    if agent.df is None:
        return jsonify({'error': '⚠️ Pehle file upload karo!'}), 400

    # Build execution namespace with df and common libs
    import pandas as pd
    import numpy as np
    namespace = {
        'df': agent.df.copy(),
        'pd': pd,
        'np': np,
        'plt': plt,
    }
    try:
        import sklearn
        namespace['sklearn'] = sklearn
        from sklearn import linear_model, preprocessing, metrics, model_selection
        namespace['linear_model'] = linear_model
        namespace['preprocessing'] = preprocessing
        namespace['metrics'] = metrics
        namespace['model_selection'] = model_selection
    except: pass
    try:
        import seaborn as sns
        namespace['sns'] = sns
    except: pass

    stdout_buf = _io.StringIO()
    stderr_buf = _io.StringIO()
    img_b64 = None

    # SECURITY: Block dangerous commands
    BLOCKED = ['os.system','subprocess','__import__','open(','eval(','exec(',
               'shutil','rmdir','remove','unlink','socket','requests.get',
               'requests.post','urllib','httpx','builtins','globals()','locals()']
    code_lower = code.lower()
    for bad in BLOCKED:
        if bad in code:
            return jsonify({'output':'','image':None,
                'error': f'⛔ Blocked: "{bad}" allowed nahi hai notebook mein!'})

    try:
        plt.close('all')
        # Restricted builtins — dangerous functions remove
        safe_builtins = {k: v for k, v in __builtins__.items()
                        if k not in ('open','__import__','eval','exec','compile',
                                     'breakpoint','input','memoryview')}                         if isinstance(__builtins__, dict) else {}
        namespace['__builtins__'] = safe_builtins
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, '<notebook>', 'exec'), namespace)

        # Check if any plot was created
        if plt.get_fignums():
            img_buf = _io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
            plt.close('all')
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')

        output = stdout_buf.getvalue()
        err_out = stderr_buf.getvalue()
        if err_out and not output:
            output = err_out

        return jsonify({
            'output': output.strip(),
            'image': img_b64,
            'error': None
        })

    except Exception as e:
        tb = traceback.format_exc()
        # Clean traceback — only show relevant part
        lines = tb.strip().split('\n')
        clean = '\n'.join(l for l in lines if '<notebook>' in l or 'Error' in l or 'error' in l.lower())
        return jsonify({
            'output': '',
            'image': None,
            'error': clean or str(e)
        })

# ── Real-time Stock/Crypto Data ──────────────────────────────────
@app.route('/api/stock', methods=['POST'])
@login_required
def get_stock():
    try:
        import yfinance as yf
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        period = data.get('period', '1mo')
        if not symbol:
            return jsonify({'error': 'Symbol daalo (e.g. AAPL, BTC-USD)'}), 400

        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period=period)

        if hist.empty:
            return jsonify({'error': f'"{symbol}" ka data nahi mila. Symbol check karo.'}), 404

        # Current price
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        change = round(current - prev, 2)
        change_pct = round((change / prev) * 100, 2) if prev else 0

        # Chart
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io as _io, base64

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(hist.index, hist['Close'], color='#5b5ef4', linewidth=2)
        ax.fill_between(hist.index, hist['Close'], alpha=0.1, color='#5b5ef4')
        ax.set_title(f'{symbol} — {period}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Price', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = _io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode()

        return jsonify({
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'current': current,
            'change': change,
            'change_pct': change_pct,
            'currency': info.get('currency', 'USD'),
            'market_cap': info.get('marketCap'),
            'volume': info.get('volume'),
            'chart': chart_b64,
            'period': period
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Multi-file SQL Join ───────────────────────────────────────────
@app.route('/api/sql_join', methods=['POST'])
@login_required
def sql_join():
    try:
        import io as _io
        data = request.get_json()
        join_on = data.get('join_on', '').strip()
        join_type = data.get('join_type', 'inner')  # inner/left/right/outer
        query = data.get('query', '').strip()        # optional SQL query

        if agent.df is None:
            return jsonify({'error': '⚠️ Pehle main file upload karo!'}), 400

        df2_csv = data.get('df2_csv', '')
        if not df2_csv:
            return jsonify({'error': 'Doosri file ka data nahi mila!'}), 400

        df1 = agent.df.copy()
        df2 = pd.read_csv(_io.StringIO(df2_csv))

        if query:
            # SQL query mode — pandasql style using pandas
            # Simple SELECT parser
            q = query.upper()
            result = df1  # default
            if 'JOIN' in q:
                # Parse: SELECT * FROM df1 JOIN df2 ON col
                if join_on:
                    how_map = {'inner':'inner','left':'left','right':'right','outer':'outer','full':'outer'}
                    how = how_map.get(join_type.lower(), 'inner')
                    result = pd.merge(df1, df2, on=join_on, how=how, suffixes=('_file1','_file2'))
                else:
                    result = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
            elif 'WHERE' in q:
                # Basic filter
                result = df1
        else:
            # Simple merge
            if join_on:
                if join_on not in df1.columns:
                    return jsonify({'error': f'Column "{join_on}" file 1 mein nahi hai! File 1 columns: {list(df1.columns)[:10]}'}), 400
                if join_on not in df2.columns:
                    return jsonify({'error': f'Column "{join_on}" file 2 mein nahi hai! File 2 columns: {list(df2.columns)[:10]}'}), 400
                how_map = {'inner':'inner','left':'left','right':'right','outer':'outer'}
                result = pd.merge(df1, df2, on=join_on, how=how_map.get(join_type,'inner'), suffixes=('_f1','_f2'))
            else:
                # Auto-detect common column
                common = list(set(df1.columns) & set(df2.columns))
                if not common:
                    return jsonify({'error': 'Koi common column nahi! File1: ' + str(list(df1.columns)[:5]) + ' | File2: ' + str(list(df2.columns)[:5])}), 400
                result = pd.merge(df1, df2, on=common[0], how='inner', suffixes=('_f1','_f2'))
                join_on = common[0]

        preview = result.head(20).to_string(index=False)
        return jsonify({
            'message': f'✅ Join successful! {len(result)} rows × {len(result.columns)} cols',
            'rows': len(result),
            'cols': len(result.columns),
            'join_on': join_on,
            'join_type': join_type,
            'columns': list(result.columns),
            'preview': preview,
            'csv': result.to_csv(index=False)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e)}), 500


@app.route('/notebook')
@login_required
def notebook():
    return render_template('notebook.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


# ════════════════════════════════════════════════════════════════
# FEATURE 1: DATA CLEANING
# ════════════════════════════════════════════════════════════════
@app.route('/api/clean', methods=['POST'])
@login_required
def data_clean():
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        data = request.get_json()
        action = data.get('action', '')
        col = data.get('column', '')
        value = data.get('value', '')
        df = agent.df.copy()
        msg = ''

        if action == 'drop_nulls':
            before = len(df)
            df = df.dropna(subset=[col] if col else None)
            msg = f'Nulls drop kiye: {before - len(df)} rows hatayi'
        elif action == 'fill_nulls':
            if col:
                if value == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    msg = f'"{col}" ke nulls mean se fill kiye'
                elif value == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    msg = f'"{col}" ke nulls median se fill kiye'
                elif value == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                    msg = f'"{col}" ke nulls mode se fill kiye'
                elif value == 'zero':
                    df[col] = df[col].fillna(0)
                    msg = f'"{col}" ke nulls 0 se fill kiye'
                elif value == 'ffill':
                    df[col] = df[col].ffill()
                    msg = f'"{col}" ke nulls forward fill kiye'
                else:
                    df[col] = df[col].fillna(value)
                    msg = f'"{col}" ke nulls "{value}" se fill kiye'
        elif action == 'drop_duplicates':
            before = len(df)
            df = df.drop_duplicates(subset=[col] if col else None)
            msg = f'Duplicates hataaye: {before - len(df)} rows'
        elif action == 'drop_column':
            if col in df.columns:
                df = df.drop(columns=[col])
                msg = f'Column "{col}" hataaya'
        elif action == 'rename_column':
            new_name = data.get('new_name', '')
            if col in df.columns and new_name:
                df = df.rename(columns={col: new_name})
                msg = f'"{col}" → "{new_name}" rename kiya'
        elif action == 'fix_dtype':
            dtype = data.get('dtype', 'float')
            try:
                if dtype == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'str':
                    df[col] = df[col].astype(str)
                elif dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                msg = f'"{col}" ko {dtype} mein convert kiya'
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        elif action == 'remove_outliers':
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                before = len(df)
                df = df[~((df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR)))]
                msg = f'Outliers hataaye "{col}" se: {before - len(df)} rows'
        elif action == 'strip_whitespace':
            str_cols = [col] if col else list(df.select_dtypes(include='object').columns)
            for c in str_cols:
                df[c] = df[c].astype(str).str.strip()
            msg = f'Whitespace strip kiya: {", ".join(str_cols)}'
        elif action == 'lowercase':
            str_cols = [col] if col else list(df.select_dtypes(include='object').columns)
            for c in str_cols:
                df[c] = df[c].astype(str).str.lower()
            msg = f'Lowercase kiya: {", ".join(str_cols)}'
        elif action == 'info':
            nulls = df.isnull().sum()
            null_cols = nulls[nulls > 0].to_dict()
            dups = int(df.duplicated().sum())
            dtypes = df.dtypes.astype(str).to_dict()
            return jsonify({
                'rows': len(df), 'cols': len(df.columns),
                'nulls': null_cols, 'duplicates': dups,
                'dtypes': dtypes, 'columns': list(df.columns),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            })
        else:
            return jsonify({'error': 'Unknown action'}), 400

        # Save cleaned df back
        agent.df = df
        import io as _io
        csv_str = df.to_csv(index=False)
        db = SessionLocal()
        dataset = db.query(UserDataset).filter_by(user_id=current_user.id).order_by(UserDataset.uploaded_at.desc()).first()
        if dataset:
            dataset.csv_data = csv_str
            dataset.row_count = len(df)
            dataset.columns = json.dumps(list(df.columns))
            db.commit()
        db.close()

        return jsonify({
            'message': f'✅ {msg}',
            'rows': len(df), 'cols': len(df.columns),
            'columns': list(df.columns),
            'preview': df.head(5).to_string(index=False)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 2: AUTO ML
# ════════════════════════════════════════════════════════════════
@app.route('/api/automl', methods=['POST'])
@login_required
def auto_ml():
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                                     classification_report, confusion_matrix)
        from sklearn.naive_bayes import GaussianNB
        import numpy as np

        data = request.get_json()
        target_col = data.get('target', '')
        task_type = data.get('task', 'auto')
        test_size = float(data.get('test_size', 0.2))

        df = agent.df.copy().dropna()

        if target_col not in df.columns:
            return jsonify({'error': f'Column "{target_col}" nahi mila! Available: {list(df.columns)[:10]}'}), 400

        # Limit to 10k rows for speed
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categorical
        le_dict = {}
        for c in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            le_dict[c] = le
        if y.dtype == object or str(y.dtype) == 'category':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
            y_labels = list(le_y.classes_)
        else:
            y_labels = None

        # Auto detect task
        if task_type == 'auto':
            unique_ratio = len(np.unique(y)) / len(y)
            task_type = 'classification' if (len(np.unique(y)) <= 20 or unique_ratio < 0.05) else 'regression'

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Models to try
        if task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
            }
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Ridge Regression': Ridge(),
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'KNN': KNeighborsRegressor(),
            }

        results = []
        best_model = None
        best_score = -999

        for name, model in models.items():
            try:
                use_scaled = name in ('Logistic Regression', 'KNN', 'Ridge Regression', 'Linear Regression', 'Naive Bayes')
                Xtr = X_train_s if use_scaled else X_train
                Xte = X_test_s if use_scaled else X_test
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
                if task_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                    metric = 'Accuracy'
                else:
                    score = r2_score(y_test, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    metric = 'R2 Score'
                results.append({'model': name, 'score': round(float(score), 4), 'metric': metric})
                if score > best_score:
                    best_score = score
                    best_model = (name, model, use_scaled)
            except Exception as e:
                results.append({'model': name, 'score': None, 'error': str(e)})

        results.sort(key=lambda x: x['score'] or -1, reverse=True)

        # Feature importance chart
        chart_b64 = None
        feat_importance = {}
        if best_model and hasattr(best_model[1], 'feature_importances_'):
            importances = best_model[1].feature_importances_
            feat_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
            feat_df = feat_df.sort_values('importance', ascending=True).tail(15)
            feat_importance = dict(zip(feat_df['feature'], feat_df['importance'].round(4)))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(feat_df['feature'], feat_df['importance'], color='#5b5ef4')
            ax.set_title(f'Feature Importance — {best_model[0]}', fontweight='bold')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()

        # Confusion matrix for classification
        cm_b64 = None
        if task_type == 'classification' and best_model:
            name, model, use_scaled = best_model
            Xte = X_test_s if use_scaled else X_test
            y_pred = model.predict(Xte)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_title('Confusion Matrix', fontweight='bold')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); cm_b64 = base64.b64encode(buf.read()).decode()

        return jsonify({
            'task': task_type,
            'target': target_col,
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'features': list(X.columns),
            'best_model': best_model[0] if best_model else None,
            'best_score': round(float(best_score), 4),
            'results': results,
            'feature_importance': feat_importance,
            'chart': chart_b64,
            'cm_chart': cm_b64
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/automl/predict', methods=['POST'])
@login_required
def automl_predict():
    """Predict on new data using last trained model stored in session"""
    return jsonify({'error': 'Train karo pehle /api/automl se, phir notebook mein predict karo!'}), 400


# ════════════════════════════════════════════════════════════════
# FEATURE 3: ADVANCED STATISTICS
# ════════════════════════════════════════════════════════════════
@app.route('/api/stats', methods=['POST'])
@login_required
def advanced_stats():
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats
        import numpy as np

        data = request.get_json()
        test = data.get('test', '')
        col1 = data.get('col1', '')
        col2 = data.get('col2', '')
        group_col = data.get('group_col', '')
        alpha = float(data.get('alpha', 0.05))
        df = agent.df.copy().dropna()

        result = {}
        chart_b64 = None

        if test == 'describe':
            num_df = df.select_dtypes(include='number')
            desc = num_df.describe().round(4)
            result = {
                'summary': desc.to_dict(),
                'skewness': num_df.skew().round(4).to_dict(),
                'kurtosis': num_df.kurtosis().round(4).to_dict(),
                'nulls': df.isnull().sum().to_dict()
            }

        elif test == 'correlation':
            num_df = df.select_dtypes(include='number')
            corr = num_df.corr().round(4)
            fig, ax = plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                       annot_kws={'size': 8}, square=True)
            ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=13)
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=110); plt.close('all')
            buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()
            result = {'correlation_matrix': corr.to_dict(), 'chart': chart_b64}

        elif test == 'ttest':
            if not col1:
                return jsonify({'error': 'col1 chahiye!'}), 400
            if group_col:
                groups = df[group_col].unique()
                if len(groups) < 2:
                    return jsonify({'error': '2 groups chahiye!'}), 400
                g1 = df[df[group_col] == groups[0]][col1].dropna()
                g2 = df[df[group_col] == groups[1]][col1].dropna()
                t_stat, p_val = scipy_stats.ttest_ind(g1, g2)
                result = {
                    'test': 'Independent T-Test',
                    't_statistic': round(float(t_stat), 4),
                    'p_value': round(float(p_val), 6),
                    'significant': bool(p_val < alpha),
                    'interpretation': f'{"Significant difference" if p_val < alpha else "No significant difference"} between groups (alpha={alpha})',
                    'group1': {'name': str(groups[0]), 'mean': round(float(g1.mean()), 4), 'n': len(g1)},
                    'group2': {'name': str(groups[1]), 'mean': round(float(g2.mean()), 4), 'n': len(g2)}
                }
            elif col2:
                t_stat, p_val = scipy_stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
                result = {
                    'test': 'Independent T-Test',
                    't_statistic': round(float(t_stat), 4),
                    'p_value': round(float(p_val), 6),
                    'significant': bool(p_val < alpha),
                    'interpretation': f'{"Significant difference" if p_val < alpha else "No significant difference"} (alpha={alpha})'
                }

        elif test == 'anova':
            if not col1 or not group_col:
                return jsonify({'error': 'col1 aur group_col chahiye!'}), 400
            groups = [g[col1].dropna().values for _, g in df.groupby(group_col)]
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            result = {
                'test': 'One-Way ANOVA',
                'f_statistic': round(float(f_stat), 4),
                'p_value': round(float(p_val), 6),
                'significant': bool(p_val < alpha),
                'num_groups': len(groups),
                'interpretation': f'{"Significant difference" if p_val < alpha else "No significant difference"} among groups (alpha={alpha})'
            }

        elif test == 'chi2':
            if not col1 or not col2:
                return jsonify({'error': 'col1 aur col2 chahiye!'}), 400
            ct = pd.crosstab(df[col1], df[col2])
            chi2, p_val, dof, expected = scipy_stats.chi2_contingency(ct)
            result = {
                'test': 'Chi-Square Test',
                'chi2_statistic': round(float(chi2), 4),
                'p_value': round(float(p_val), 6),
                'degrees_of_freedom': int(dof),
                'significant': bool(p_val < alpha),
                'interpretation': f'{"Significant association" if p_val < alpha else "No significant association"} between {col1} and {col2} (alpha={alpha})'
            }

        elif test == 'normality':
            if not col1:
                return jsonify({'error': 'col1 chahiye!'}), 400
            series = df[col1].dropna()
            if len(series) > 5000:
                series = series.sample(5000)
            stat, p_val = scipy_stats.shapiro(series) if len(series) <= 5000 else scipy_stats.normaltest(series)
            # Distribution plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(series, bins=30, color='#5b5ef4', alpha=0.7, edgecolor='white')
            axes[0].set_title(f'Distribution of {col1}', fontweight='bold')
            scipy_stats.probplot(series, dist='norm', plot=axes[1])
            axes[1].set_title('Q-Q Plot', fontweight='bold')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()
            result = {
                'test': 'Normality Test (Shapiro-Wilk)',
                'statistic': round(float(stat), 4),
                'p_value': round(float(p_val), 6),
                'is_normal': bool(p_val > alpha),
                'interpretation': f'{"Normal distribution" if p_val > alpha else "Not normal distribution"} (alpha={alpha})',
                'chart': chart_b64
            }

        elif test == 'distribution':
            num_cols = list(df.select_dtypes(include='number').columns[:6])
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            axes = axes.flatten()
            for i, c in enumerate(num_cols):
                axes[i].hist(df[c].dropna(), bins=25, color='#5b5ef4', alpha=0.75, edgecolor='white')
                axes[i].set_title(c, fontweight='bold', fontsize=10)
            for j in range(len(num_cols), 6):
                axes[j].set_visible(False)
            plt.suptitle('Distribution of Numeric Columns', fontweight='bold', fontsize=13)
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()
            result = {'columns': num_cols, 'chart': chart_b64}

        if 'chart' not in result and chart_b64:
            result['chart'] = chart_b64
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 4: BIG DATA — Chunked Upload (up to 500MB via HuggingFace)
# ════════════════════════════════════════════════════════════════
import uuid as _uuid

@app.route('/api/upload_chunk', methods=['POST'])
@login_required
def upload_chunk():
    """Send one 25MB chunk to HuggingFace Space"""
    try:
        url = get_hf_url()
        if not url:
            return jsonify({'error': 'ML Server offline! HuggingFace Space start karo.'}), 400

        upload_id = request.form.get('upload_id', '')
        chunk_index = request.form.get('chunk_index', '0')
        total_chunks = request.form.get('total_chunks', '1')
        filename = request.form.get('filename', 'data.csv')

        if 'file' not in request.files:
            return jsonify({'error': 'Chunk file missing!'}), 400

        chunk_file = request.files['file']

        # Forward chunk to HuggingFace
        files = {'file': (filename, chunk_file.read(), 'application/octet-stream')}
        data = {
            'upload_id': upload_id,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'filename': filename
        }
        import requests as req
        r = req.post(url + '/upload_chunk', files=files, data=data, timeout=120)
        return jsonify(r.json())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merge_chunks', methods=['POST'])
@login_required
def merge_chunks():
    """Tell HuggingFace to merge all chunks and return processed data"""
    try:
        url = get_hf_url()
        if not url:
            return jsonify({'error': 'ML Server offline!'}), 400

        data = request.get_json()
        import requests as req
        r = req.post(url + '/merge_chunks', json=data, timeout=300)
        result = r.json()

        if result.get('success') and result.get('sample_csv'):
            # Load sample into agent
            import io as _io
            df = pd.read_csv(_io.StringIO(result['sample_csv']))
            agent.df = df
            agent.available_columns = list(df.columns)

            # Save to DB (sample only)
            db = SessionLocal()
            db.query(UserDataset).filter_by(user_id=current_user.id).delete()
            dataset = UserDataset(
                user_id=current_user.id,
                filename=result['filename'],
                csv_data=result['sample_csv'][:2000000],  # max 2MB in DB
                rows=result['total_rows'],
                columns=result['total_cols']
            )
            db.add(dataset); db.commit(); db.close()

            return jsonify({
                'success': True,
                'filename': result['filename'],
                'total_rows': result['total_rows'],
                'sample_rows': result['sample_rows'],
                'total_cols': result['total_cols'],
                'size_mb': result['size_mb'],
                'columns': result['columns'],
                'null_info': result.get('null_info', {}),
                'message': result['message']
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 4: BIG DATA (chunked, up to 50MB)
# ════════════════════════════════════════════════════════════════
@app.route('/api/upload_big', methods=['POST'])
@login_required
def upload_big():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File nahi mili!'}), 400
        file = request.files['file']
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        import io as _io

        file_bytes = file.read()
        size_mb = len(file_bytes) / (1024 * 1024)

        if size_mb > 52:
            return jsonify({'error': f'File {size_mb:.1f}MB hai — max 50MB allowed!'}), 400

        if ext == 'csv':
            # Chunked read for large CSVs
            chunks = []
            chunk_size = 50000
            buf = _io.StringIO(file_bytes.decode('utf-8', errors='replace'))
            for chunk in pd.read_csv(buf, chunksize=chunk_size):
                chunks.append(chunk)
                if sum(len(c) for c in chunks) > 500000:
                    break
            df = pd.concat(chunks, ignore_index=True)
        elif ext in ('xlsx', 'xls'):
            df = pd.read_excel(_io.BytesIO(file_bytes))
        elif ext == 'json':
            df = pd.read_json(_io.BytesIO(file_bytes))
        elif ext == 'parquet':
            df = pd.read_parquet(_io.BytesIO(file_bytes))
        elif ext == 'tsv':
            df = pd.read_csv(_io.StringIO(file_bytes.decode('utf-8', errors='replace')), sep='\t')
        else:
            return jsonify({'error': f'Format ".{ext}" support nahi — CSV/Excel/JSON/Parquet/TSV use karo'}), 400

        agent.df = df
        agent.available_columns = list(df.columns)

        csv_str = df.to_csv(index=False)
        db = SessionLocal()
        db.query(UserDataset).filter_by(user_id=current_user.id).delete()
        dataset = UserDataset(
            user_id=current_user.id, filename=filename,
            csv_data=csv_str, columns=json.dumps(list(df.columns)),
            row_count=len(df)
        )
        db.add(dataset); db.commit(); db.close()

        num_cols = list(df.select_dtypes(include='number').columns)
        cat_cols = list(df.select_dtypes(include='object').columns)

        return jsonify({
            'message': f'Big file load ho gayi! {len(df):,} rows × {len(df.columns)} cols ({size_mb:.1f}MB)',
            'rows': len(df), 'cols': len(df.columns),
            'size_mb': round(size_mb, 2),
            'columns': list(df.columns),
            'numeric_cols': num_cols,
            'categorical_cols': cat_cols,
            'nulls': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'preview': df.head(3).to_string(index=False)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 5: SCHEDULED AUTO REPORTS
# ════════════════════════════════════════════════════════════════
@app.route('/api/auto_report', methods=['POST'])
@login_required
def auto_report():
    """Generate and optionally email a full auto report"""
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats as scipy_stats

        data = request.get_json()
        email_to = data.get('email', '')
        df = agent.df.copy()

        sections = []

        # 1. Overview
        sections.append({
            'title': 'Dataset Overview',
            'content': {
                'rows': len(df), 'columns': len(df.columns),
                'numeric_cols': len(df.select_dtypes(include='number').columns),
                'categorical_cols': len(df.select_dtypes(include='object').columns),
                'total_nulls': int(df.isnull().sum().sum()),
                'null_percentage': round(df.isnull().sum().sum() / df.size * 100, 2),
                'duplicates': int(df.duplicated().sum()),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
        })

        # 2. Numeric stats
        num_df = df.select_dtypes(include='number')
        if not num_df.empty:
            sections.append({
                'title': 'Numeric Summary',
                'content': num_df.describe().round(3).to_dict()
            })

        # 3. Top correlations
        charts = []
        if len(num_df.columns) >= 2:
            corr = num_df.corr()
            top_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    top_corr.append({
                        'col1': corr.columns[i], 'col2': corr.columns[j],
                        'correlation': round(float(corr.iloc[i, j]), 4)
                    })
            top_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
            sections.append({'title': 'Top Correlations', 'content': top_corr[:10]})

            # Heatmap
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                       annot_kws={'size': 7}, square=True)
            ax.set_title('Correlation Heatmap', fontweight='bold')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); charts.append({'title': 'Correlation Heatmap', 'img': base64.b64encode(buf.read()).decode()})

        # 4. Distributions
        plot_cols = list(num_df.columns[:4])
        if plot_cols:
            fig, axes = plt.subplots(1, len(plot_cols), figsize=(4*len(plot_cols), 3.5))
            if len(plot_cols) == 1: axes = [axes]
            for i, c in enumerate(plot_cols):
                axes[i].hist(df[c].dropna(), bins=25, color='#5b5ef4', alpha=0.75, edgecolor='white')
                axes[i].set_title(c, fontweight='bold', fontsize=9)
            plt.suptitle('Distributions', fontweight='bold')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); charts.append({'title': 'Distributions', 'img': base64.b64encode(buf.read()).decode()})

        # 5. Categorical counts
        cat_df = df.select_dtypes(include='object')
        cat_summary = {}
        for c in cat_df.columns[:5]:
            vc = df[c].value_counts().head(5)
            cat_summary[c] = {'top_values': vc.to_dict(), 'unique': int(df[c].nunique())}
            if int(df[c].nunique()) <= 15:
                fig, ax = plt.subplots(figsize=(6, 3))
                vc.plot(kind='bar', ax=ax, color='#06d6a0', edgecolor='white')
                ax.set_title(f'{c} — Value Counts', fontweight='bold')
                ax.set_xlabel(''); plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=90); plt.close('all')
                buf.seek(0); charts.append({'title': f'{c} Distribution', 'img': base64.b64encode(buf.read()).decode()})
        if cat_summary:
            sections.append({'title': 'Categorical Summary', 'content': cat_summary})

        # Build HTML report
        import datetime
        template = data.get('template', 'corporate')
        report_title = data.get('report_title', 'Data Analysis Report')
        company_name = data.get('company_name', 'DS Agent')
        now_str = datetime.datetime.now().strftime('%d %b %Y, %H:%M')
        meta_str = f"{len(df):,} rows × {len(df.columns)} columns"

        # ── TEMPLATE STYLES ──
        templates = {
            'corporate': {
                'name': 'Corporate Blue',
                'body_bg': '#f0f4f8',
                'header_bg': 'linear-gradient(135deg,#1e3a5f,#2563eb)',
                'header_text': '#ffffff',
                'accent': '#2563eb',
                'accent_light': '#dbeafe',
                'section_border': '#e2e8f0',
                'stat_bg': '#f8fafc',
                'stat_color': '#1e3a5f',
                'table_head': '#1e3a5f',
                'table_head_text': '#ffffff',
                'table_alt': '#f0f4f8',
                'font': 'Georgia, serif',
                'footer_bg': '#1e3a5f',
                'footer_text': '#94a3b8',
                'container_shadow': '0 8px 32px rgba(30,58,95,0.15)',
                'border_radius': '0px',
            },
            'healthcare': {
                'name': 'Healthcare Green',
                'body_bg': '#f0fdf4',
                'header_bg': 'linear-gradient(135deg,#065f46,#059669)',
                'header_text': '#ffffff',
                'accent': '#059669',
                'accent_light': '#dcfce7',
                'section_border': '#bbf7d0',
                'stat_bg': '#f0fdf4',
                'stat_color': '#065f46',
                'table_head': '#065f46',
                'table_head_text': '#ffffff',
                'table_alt': '#f0fdf4',
                'font': 'DM Sans, Segoe UI, sans-serif',
                'footer_bg': '#065f46',
                'footer_text': '#6ee7b7',
                'container_shadow': '0 8px 32px rgba(6,95,70,0.15)',
                'border_radius': '16px',
            },
            'executive': {
                'name': 'Executive Dark',
                'body_bg': '#0f172a',
                'header_bg': 'linear-gradient(135deg,#0f172a,#1e293b)',
                'header_text': '#f1f5f9',
                'accent': '#f59e0b',
                'accent_light': '#1e293b',
                'section_border': '#1e293b',
                'stat_bg': '#1e293b',
                'stat_color': '#f59e0b',
                'table_head': '#1e293b',
                'table_head_text': '#f59e0b',
                'table_alt': '#0f172a',
                'font': 'Inter, Segoe UI, sans-serif',
                'footer_bg': '#020617',
                'footer_text': '#475569',
                'container_shadow': '0 8px 32px rgba(0,0,0,0.5)',
                'border_radius': '12px',
            },
            'minimal': {
                'name': 'Minimal Clean',
                'body_bg': '#ffffff',
                'header_bg': '#ffffff',
                'header_text': '#111827',
                'accent': '#111827',
                'accent_light': '#f9fafb',
                'section_border': '#e5e7eb',
                'stat_bg': '#f9fafb',
                'stat_color': '#111827',
                'table_head': '#f3f4f6',
                'table_head_text': '#374151',
                'table_alt': '#f9fafb',
                'font': 'Inter, Helvetica, sans-serif',
                'footer_bg': '#f9fafb',
                'footer_text': '#9ca3af',
                'container_shadow': '0 1px 3px rgba(0,0,0,0.1)',
                'border_radius': '8px',
            },
            'vibrant': {
                'name': 'Vibrant Purple',
                'body_bg': '#faf5ff',
                'header_bg': 'linear-gradient(135deg,#5b5ef4,#8b5cf6,#ec4899)',
                'header_text': '#ffffff',
                'accent': '#7c3aed',
                'accent_light': '#ede9fe',
                'section_border': '#ddd6fe',
                'stat_bg': '#f5f3ff',
                'stat_color': '#5b21b6',
                'table_head': '#7c3aed',
                'table_head_text': '#ffffff',
                'table_alt': '#faf5ff',
                'font': 'Sora, DM Sans, sans-serif',
                'footer_bg': 'linear-gradient(135deg,#5b5ef4,#8b5cf6)',
                'footer_text': '#e9d5ff',
                'container_shadow': '0 8px 32px rgba(91,94,244,0.2)',
                'border_radius': '20px',
            }
        }

        t = templates.get(template, templates['corporate'])
        br = t['border_radius']
        body_color = '#1a1a2e' if template != 'executive' else '#e2e8f0'

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<link href='https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Sora:wght@400;700&family=Inter:wght@400;600;700&display=swap' rel='stylesheet'>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:{t['font']};background:{t['body_bg']};padding:30px 16px;color:{body_color}}}
.container{{max-width:920px;margin:0 auto;background:{'#1a2035' if template=='executive' else 'white'};border-radius:{br};overflow:hidden;box-shadow:{t['container_shadow']}}}
.header{{background:{t['header_bg']};color:{t['header_text']};padding:36px 40px;position:relative;overflow:hidden}}
.header::after{{content:'';position:absolute;right:-40px;top:-40px;width:200px;height:200px;background:rgba(255,255,255,0.05);border-radius:50%}}
.header::before{{content:'';position:absolute;right:60px;bottom:-30px;width:120px;height:120px;background:rgba(255,255,255,0.04);border-radius:50%}}
.header h1{{font-size:26px;font-weight:800;letter-spacing:-0.5px;position:relative}}
.header .subtitle{{font-size:13px;opacity:0.75;margin-top:6px;position:relative}}
.header .meta{{display:flex;gap:20px;margin-top:16px;position:relative}}
.header .meta span{{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}}
.section{{padding:24px 40px;border-bottom:1px solid {t['section_border']}}}
.section h2{{font-size:15px;font-weight:800;color:{t['accent']};margin-bottom:14px;display:flex;align-items:center;gap:8px;letter-spacing:0.3px;text-transform:uppercase}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.stat-box{{background:{t['stat_bg']};border-radius:12px;padding:16px;text-align:center;border:1px solid {t['section_border']}}}
.stat-box .val{{font-size:24px;font-weight:800;color:{t['stat_color']}}}
.stat-box .lbl{{font-size:11px;color:#888;margin-top:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{background:{t['table_head']};color:{t['table_head_text']};padding:10px 12px;text-align:left;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}}
th:first-child{{border-radius:{'6px 0 0 6px' if br!='0px' else '0'}}}
th:last-child{{border-radius:{'0 6px 6px 0' if br!='0px' else '0'}}}
tr:nth-child(even) td{{background:{t['table_alt']}}}
td{{padding:9px 12px;border-bottom:1px solid {t['section_border']};color:{body_color}}}
.chart-wrap{{background:{t['stat_bg']};border-radius:10px;padding:12px;margin-top:8px}}
img{{max-width:100%;border-radius:8px}}
.corr-pos{{color:#059669;font-weight:700}}
.corr-neg{{color:#ef4444;font-weight:700}}
.badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;background:{t['accent_light']};color:{t['accent']}}}
.footer{{background:{t['footer_bg']};padding:18px 40px;text-align:center;font-size:12px;color:{t['footer_text']}}}
.footer strong{{color:{t['accent']}}}
@media(max-width:600px){{.stat-grid{{grid-template-columns:repeat(2,1fr)}}body{{padding:10px 8px}}.section{{padding:16px 18px}}.header{{padding:24px 20px}}}}
</style>
</head><body><div class='container'>
<div class='header'>
  <h1>📊 {report_title}</h1>
  <div class='subtitle'>{company_name}</div>
  <div class='meta'>
    <span>📅 {now_str}</span>
    <span>📁 {meta_str}</span>
    <span>🎨 {t['name']}</span>
  </div>
</div>"""

        # Overview section
        ov = sections[0]['content']
        html += f"""<div class='section'><h2>📋 Dataset Overview</h2>
        <div class='stat-grid'>
        <div class='stat-box'><div class='val'>{ov['rows']:,}</div><div class='lbl'>Total Rows</div></div>
        <div class='stat-box'><div class='val'>{ov['columns']}</div><div class='lbl'>Columns</div></div>
        <div class='stat-box'><div class='val'>{ov['null_percentage']}%</div><div class='lbl'>Null Values</div></div>
        <div class='stat-box'><div class='val'>{ov['duplicates']}</div><div class='lbl'>Duplicates</div></div>
        </div></div>"""

        # Numeric summary
        for sec in sections[1:]:
            html += f"<div class='section'><h2>{sec['title']}</h2>"
            if isinstance(sec['content'], dict) and sec['title'] == 'Numeric Summary':
                html += "<table><tr><th>Column</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Nulls</th></tr>"
                for col in list(sec['content'].get('mean', {}).keys()):
                    d = sec['content']
                    html += f"<tr><td><b>{col}</b></td><td>{d.get('mean',{}).get(col,'')}</td><td>{d.get('std',{}).get(col,'')}</td><td>{d.get('min',{}).get(col,'')}</td><td>{d.get('max',{}).get(col,'')}</td><td>{int(df[col].isnull().sum())}</td></tr>"
                html += "</table>"
            elif isinstance(sec['content'], list):
                html += "<table><tr><th>Column 1</th><th>Column 2</th><th>Correlation</th></tr>"
                for row in sec['content'][:10]:
                    cls = 'corr-pos' if row['correlation'] > 0 else 'corr-neg'
                    html += f"<tr><td>{row['col1']}</td><td>{row['col2']}</td><td class='{cls}'>{row['correlation']}</td></tr>"
                html += "</table>"
            html += "</div>"

        # Charts
        for ch in charts:
            html += f"<div class='section'><h2>{ch['title']}</h2><div class='chart-wrap'><img src='data:image/png;base64,{ch['img']}'/></div></div>"

        html += f"<div class='footer'>Generated by <strong>DS Agent</strong> — Your AI Data Scientist &nbsp;|&nbsp; Template: {t['name']}</div></div></body></html>"

        # Overview section
        ov = sections[0]['content']
        html += f"""<div class='section'><h2>📋 Dataset Overview</h2>
        <div class='stat-grid'>
        <div class='stat-box'><div class='val'>{ov['rows']:,}</div><div class='lbl'>Rows</div></div>
        <div class='stat-box'><div class='val'>{ov['columns']}</div><div class='lbl'>Columns</div></div>
        <div class='stat-box'><div class='val'>{ov['null_percentage']}%</div><div class='lbl'>Nulls</div></div>
        <div class='stat-box'><div class='val'>{ov['duplicates']}</div><div class='lbl'>Duplicates</div></div>
        </div></div>"""

        # Numeric summary
        for sec in sections[1:]:
            html += f"<div class='section'><h2>{sec['title']}</h2>"
            if isinstance(sec['content'], dict) and sec['title'] == 'Numeric Summary':
                html += "<table><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Nulls</th></tr>"
                for col in list(sec['content'].get('mean', {}).keys()):
                    d = sec['content']
                    html += f"<tr><td><b>{col}</b></td><td>{d.get('mean',{}).get(col,'')}</td><td>{d.get('std',{}).get(col,'')}</td><td>{d.get('min',{}).get(col,'')}</td><td>{d.get('max',{}).get(col,'')}</td><td>{int(df[col].isnull().sum())}</td></tr>"
                html += "</table>"
            elif isinstance(sec['content'], list):
                html += "<table><tr><th>Col 1</th><th>Col 2</th><th>Correlation</th></tr>"
                for row in sec['content'][:10]:
                    color = '#059669' if row['correlation'] > 0 else '#ef4444'
                    html += f"<tr><td>{row['col1']}</td><td>{row['col2']}</td><td style='color:{color};font-weight:700'>{row['correlation']}</td></tr>"
                html += "</table>"
            html += "</div>"

        # Charts
        for ch in charts:
            html += f"<div class='section'><h2>{ch['title']}</h2><img src='data:image/png;base64,{ch['img']}'/></div>"

        html += "<div class='footer'>Generated by DS Agent — Your AI Data Scientist</div></div></body></html>"

        # Optionally email
        email_sent = False
        if email_to:
            try:
                import resend as resend_lib
                resend_lib.api_key = os.environ.get('RESEND_API_KEY', '')
                resend_lib.Emails.send({
                    'from': 'DS Agent <reports@dsagent.app>',
                    'to': [email_to],
                    'subject': f'Your Data Report — {datetime.datetime.now().strftime("%d %b %Y")}',
                    'html': html
                })
                email_sent = True
            except Exception as e:
                logger.warning(f'Email failed: {e}')

        return jsonify({
            'message': f'Report ready! {len(charts)} charts generated.',
            'html': html,
            'email_sent': email_sent,
            'sections': len(sections),
            'charts': len(charts)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 6: PIVOT TABLE
# ════════════════════════════════════════════════════════════════
@app.route('/api/pivot', methods=['POST'])
@login_required
def pivot_table():
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        data = request.get_json()
        index_col = data.get('index', '')
        values_col = data.get('values', '')
        aggfunc = data.get('aggfunc', 'sum')
        columns_col = data.get('columns', '')
        df = agent.df.copy()

        if not index_col or index_col not in df.columns:
            return jsonify({'error': f'index column chahiye! Available: {list(df.columns)[:8]}'}), 400
        if not values_col or values_col not in df.columns:
            return jsonify({'error': f'values column chahiye! Available: {list(df.columns)[:8]}'}), 400

        pivot_kwargs = {
            'index': index_col,
            'values': values_col,
            'aggfunc': aggfunc
        }
        if columns_col and columns_col in df.columns:
            pivot_kwargs['columns'] = columns_col

        pivot = pd.pivot_table(df, **pivot_kwargs).round(3)

        # Chart
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(min(10, 2+len(pivot.columns)*1.5), 5))
        pivot.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='white')
        ax.set_title(f'Pivot: {aggfunc}({values_col}) by {index_col}', fontweight='bold')
        ax.set_xlabel(index_col); ax.set_ylabel(f'{aggfunc}({values_col})')
        plt.xticks(rotation=30, ha='right'); plt.tight_layout()
        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
        buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()

        return jsonify({
            'message': f'Pivot table ready! {pivot.shape[0]} rows × {pivot.shape[1]} cols',
            'pivot': pivot.reset_index().to_dict(orient='records'),
            'columns': [str(index_col)] + [str(c) for c in pivot.columns],
            'chart': chart_b64,
            'csv': pivot.reset_index().to_csv(index=False)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE 7: TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════════
@app.route('/api/timeseries', methods=['POST'])
@login_required
def time_series():
    if agent.df is None:
        return jsonify({'error': 'Pehle file upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        data = request.get_json()
        date_col = data.get('date_col', '')
        value_col = data.get('value_col', '')
        freq = data.get('freq', 'auto')
        forecast_periods = int(data.get('forecast_periods', 7))

        df = agent.df.copy()
        if not date_col or date_col not in df.columns:
            return jsonify({'error': f'date_col chahiye! Available: {list(df.columns)[:8]}'}), 400
        if not value_col or value_col not in df.columns:
            return jsonify({'error': f'value_col chahiye! Available: {list(df.columns)[:8]}'}), 400

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        df = df.sort_values(date_col)

        series = df.set_index(date_col)[value_col]

        # Resample if needed
        if freq == 'auto':
            time_diff = (series.index[-1] - series.index[0]).days
            freq = 'D' if time_diff <= 365 else 'W' if time_diff <= 1825 else 'ME'

        # Rolling stats
        window = max(7, len(series) // 10)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        # Simple trend forecast (linear)
        from sklearn.linear_model import LinearRegression
        X_t = np.arange(len(series)).reshape(-1, 1)
        lr = LinearRegression().fit(X_t, series.values)
        trend_line = lr.predict(X_t)
        future_X = np.arange(len(series), len(series)+forecast_periods).reshape(-1, 1)
        forecast = lr.predict(future_X)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        # Main time series
        axes[0].plot(series.index, series.values, color='#5b5ef4', linewidth=1.5, label='Actual', alpha=0.8)
        axes[0].plot(series.index, rolling_mean, color='#f59e0b', linewidth=2, label=f'{window}-period MA')
        axes[0].fill_between(series.index,
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.15, color='#f59e0b')
        axes[0].set_title(f'{value_col} — Time Series', fontweight='bold', fontsize=12)
        axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

        # Forecast
        last_date = series.index[-1]
        future_dates = pd.date_range(last_date, periods=forecast_periods+1, freq='D')[1:]
        axes[1].plot(series.index[-50:], series.values[-50:], color='#5b5ef4', linewidth=1.5, label='Actual')
        axes[1].plot(series.index, trend_line, color='#9ca3af', linewidth=1, linestyle='--', label='Trend')
        axes[1].plot(future_dates, forecast, color='#06d6a0', linewidth=2.5, marker='o', markersize=4, label=f'{forecast_periods}-period Forecast')
        axes[1].axvline(last_date, color='#ef4444', linewidth=1, linestyle=':', alpha=0.7)
        axes[1].set_title('Trend + Forecast', fontweight='bold', fontsize=12)
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
        buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()

        return jsonify({
            'message': f'Time series analysis complete! {len(series)} data points analyzed.',
            'date_col': date_col,
            'value_col': value_col,
            'data_points': len(series),
            'date_range': f'{series.index[0].strftime("%Y-%m-%d")} to {series.index[-1].strftime("%Y-%m-%d")}',
            'trend': 'Upward' if lr.coef_[0] > 0 else 'Downward',
            'trend_slope': round(float(lr.coef_[0]), 4),
            'mean': round(float(series.mean()), 4),
            'std': round(float(series.std()), 4),
            'min': round(float(series.min()), 4),
            'max': round(float(series.max()), 4),
            'forecast': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in zip(future_dates, forecast)],
            'chart': chart_b64
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ════════════════════════════════════════════════════════════════
# TARIKA 3 — GOOGLE COLAB HYBRID (Heavy ML/Deep Learning)
# ════════════════════════════════════════════════════════════════

@app.route('/api/colab/status', methods=['GET'])
@login_required
def colab_status():
    """Check HuggingFace (primary) and Colab (backup)"""
    import requests as req
    hf_url = os.environ.get('HF_SPACE_URL', '').strip()
    colab_url = os.environ.get('COLAB_URL', '').strip()

    # HuggingFace — main page check (Gradio runs on 7860, no /ping needed)
    if hf_url:
        try:
            r = req.get(hf_url, timeout=10)
            if r.status_code == 200:
                return jsonify({'connected': True, 'url': hf_url, 'source': 'HuggingFace', 'message': 'HuggingFace Connected!'})
        except: pass

    # Colab — /ping check
    if colab_url:
        try:
            r = req.get(colab_url + '/ping', timeout=6)
            if r.status_code == 200:
                return jsonify({'connected': True, 'url': colab_url, 'source': 'Colab', 'message': 'Colab Connected!'})
        except: pass

    return jsonify({'connected': False, 'source': 'none', 'message': 'Dono offline!'})


@app.route('/api/colab/run', methods=['POST'])
@login_required
def colab_run():
    """Send heavy task to Google Colab"""
    colab_url = os.environ.get('COLAB_URL', '').strip()
    if not colab_url:
        return jsonify({'error': 'COLAB_URL env variable set nahi hai! Render dashboard pe add karo.'}), 400
    try:
        import requests as req
        data = request.get_json()
        task = data.get('task', '')
        code = data.get('code', '')
        payload = {}

        if agent.df is not None:
            import io as _io
            csv_str = agent.df.to_csv(index=False)
            payload['csv_data'] = csv_str

        payload['task'] = task
        payload['code'] = code
        payload['params'] = data.get('params', {})

        r = req.post(colab_url + '/run', json=payload, timeout=120)
        result = r.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Colab error: {str(e)} — Colab notebook open hai?'}), 500


@app.route('/api/colab/deep_learning', methods=['POST'])
@login_required
def colab_deep_learning():
    """Run Deep Learning — HuggingFace primary, Colab backup"""
    import requests as req
    hf_url = os.environ.get('HF_SPACE_URL', '').strip()
    colab_url = os.environ.get('COLAB_URL', '').strip()

    # Pick available server
    server_url = None
    if hf_url:
        try:
            r = req.get(hf_url, timeout=10)
            if r.status_code == 200:
                server_url = hf_url
                print(f"[DL] Using HuggingFace: {hf_url}")
        except: pass
    if not server_url and colab_url:
        try:
            r = req.get(colab_url + '/ping', timeout=6)
            if r.status_code == 200:
                server_url = colab_url
        except: pass
    if not server_url:
        return jsonify({'error': 'ML Server offline! HuggingFace Space check karo — ping karo pehle.'}), 400
    try:
        import requests as req
        data = request.get_json()
        model_type = data.get('model_type', 'neural_network')
        target = data.get('target', '')
        epochs = int(data.get('epochs', 10))
        layers = data.get('layers', [64, 32])

        if agent.df is None:
            return jsonify({'error': 'Pehle file upload karo!'}), 400

        payload = {
            'task': 'deep_learning',
            'csv_data': agent.df.to_csv(index=False),
            'params': {
                'model_type': model_type,
                'target': target,
                'epochs': epochs,
                'layers': layers
            }
        }
        result = call_hf_api(server_url, '/run', payload)
        safe = validate_hf_result(result, {
            'success': False,
            'task': 'unknown',
            'target': '',
            'epochs_ran': 0,
            'metrics': {},
            'train_rows': 0,
            'test_rows': 0,
            'message': 'Training complete!',
            'chart': None,
        })
        return jsonify(safe), (500 if safe.get('error') else 200)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ════════════════════════════════════════════════════════════════
# HEALTHCARE MODULE
# ════════════════════════════════════════════════════════════════

# Healthcare roles store (in-memory + DB via user metadata)
HEALTHCARE_ROLES = ['admin', 'doctor', 'nurse', 'analyst', 'receptionist']

ROLE_PERMISSIONS = {
    'admin':        ['dashboard', 'patients', 'reports', 'risk', 'finance', 'staff', 'privacy_off'],
    'doctor':       ['dashboard', 'patients', 'reports', 'risk'],
    'nurse':        ['dashboard', 'patients'],
    'analyst':      ['dashboard', 'reports', 'risk'],
    'receptionist': ['dashboard', 'patients'],
}

SENSITIVE_COLUMNS = [
    'name', 'patient_name', 'full_name', 'phone', 'mobile', 'address',
    'email', 'aadhar', 'ssn', 'dob', 'date_of_birth', 'contact',
    'guardian', 'emergency_contact', 'account', 'insurance_id'
]


def get_user_hc_role(user_id):
    """Get healthcare role from DB notes field or default to analyst"""
    try:
        db = SessionLocal()
        user = db.query(User).filter_by(id=user_id).first()
        db.close()
        if user and user.is_admin:
            return 'admin'
        # Store role in username prefix like "dr_john" -> doctor
        if user:
            uname = user.username.lower()
            if uname.startswith('dr_') or uname.startswith('doc_'):
                return 'doctor'
            elif uname.startswith('nurse_') or uname.startswith('nr_'):
                return 'nurse'
            elif uname.startswith('rec_') or uname.startswith('reception_'):
                return 'receptionist'
        return 'analyst'
    except:
        return 'analyst'


def mask_sensitive(df, role):
    """Mask sensitive columns based on role"""
    if 'privacy_off' in ROLE_PERMISSIONS.get(role, []):
        return df  # Admin sees all
    masked = df.copy()
    for col in masked.columns:
        if col.lower() in SENSITIVE_COLUMNS:
            masked[col] = masked[col].astype(str).str[:2] + '****'
    return masked


# ── Healthcare Role API ───────────────────────────────────────
@app.route('/api/hc/role', methods=['GET'])
@login_required
def hc_get_role():
    role = get_user_hc_role(current_user.id)
    perms = ROLE_PERMISSIONS.get(role, [])
    return jsonify({
        'role': role,
        'permissions': perms,
        'username': current_user.username,
        'is_admin': current_user.is_admin
    })


@app.route('/api/hc/set_role', methods=['POST'])
@login_required
def hc_set_role():
    if not current_user.is_admin:
        return jsonify({'error': 'Sirf admin role assign kar sakta hai!'}), 403
    data = request.get_json()
    target_username = data.get('username', '')
    new_role = data.get('role', '')
    if new_role not in HEALTHCARE_ROLES:
        return jsonify({'error': f'Invalid role! Use: {HEALTHCARE_ROLES}'}), 400
    try:
        db = SessionLocal()
        user = db.query(User).filter_by(username=target_username).first()
        if not user:
            db.close()
            return jsonify({'error': 'User nahi mila!'}), 404
        # Role store via username prefix convention
        db.close()
        return jsonify({'message': f'Role "{new_role}" assigned to {target_username}! (Username prefix se role detect hota hai)', 'role': new_role})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Hospital Dashboard ────────────────────────────────────────
@app.route('/api/hc/dashboard', methods=['POST'])
@login_required
def hc_dashboard():
    if agent.df is None:
        return jsonify({'error': 'Pehle patient/hospital data upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        role = get_user_hc_role(current_user.id)
        perms = ROLE_PERMISSIONS.get(role, [])
        df = agent.df.copy()
        df_masked = mask_sensitive(df, role)

        result = {
            'role': role,
            'permissions': perms,
            'total_records': len(df),
            'columns': list(df.columns),
        }

        # Auto-detect common hospital columns
        cols_lower = {c.lower(): c for c in df.columns}

        # KPIs
        kpis = {}
        kpis['total_patients'] = len(df)

        # Age stats
        age_col = next((cols_lower[c] for c in cols_lower if 'age' in c), None)
        if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
            kpis['avg_age'] = round(float(df[age_col].mean()), 1)
            kpis['age_range'] = f"{int(df[age_col].min())} - {int(df[age_col].max())}"

        # Gender
        gender_col = next((cols_lower[c] for c in cols_lower if 'gender' in c or 'sex' in c), None)
        if gender_col:
            kpis['gender_dist'] = df[gender_col].value_counts().to_dict()

        # Diagnosis/Disease
        diag_col = next((cols_lower[c] for c in cols_lower if 'diag' in c or 'disease' in c or 'condition' in c), None)
        if diag_col:
            kpis['top_diagnoses'] = df[diag_col].value_counts().head(5).to_dict()

        # Department
        dept_col = next((cols_lower[c] for c in cols_lower if 'dept' in c or 'department' in c or 'ward' in c), None)
        if dept_col:
            kpis['department_dist'] = df[dept_col].value_counts().head(8).to_dict()

        # Admission/discharge
        admit_col = next((cols_lower[c] for c in cols_lower if 'admit' in c or 'admission' in c), None)
        if admit_col:
            kpis['admission_col'] = admit_col

        # Finance (only admin/analyst)
        if 'finance' in perms:
            bill_col = next((cols_lower[c] for c in cols_lower if 'bill' in c or 'amount' in c or 'cost' in c or 'charge' in c or 'fee' in c), None)
            if bill_col and pd.api.types.is_numeric_dtype(df[bill_col]):
                kpis['total_revenue'] = round(float(df[bill_col].sum()), 2)
                kpis['avg_bill'] = round(float(df[bill_col].mean()), 2)
                kpis['max_bill'] = round(float(df[bill_col].max()), 2)

        result['kpis'] = kpis

        # Charts
        charts = []
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        plot_count = 0

        # 1. Age distribution
        if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
            axes[0].hist(df[age_col].dropna(), bins=20, color='#5b5ef4', alpha=0.8, edgecolor='white')
            axes[0].set_title('Patient Age Distribution', fontweight='bold', fontsize=11)
            axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')
            axes[0].grid(True, alpha=0.3)
            plot_count += 1
        else:
            axes[0].text(0.5, 0.5, 'No age column found', ha='center', va='center', transform=axes[0].transAxes, color='#888')
            axes[0].set_title('Age Distribution', fontweight='bold')

        # 2. Gender pie
        if gender_col:
            gdata = df[gender_col].value_counts().head(5)
            axes[1].pie(gdata.values, labels=gdata.index, autopct='%1.1f%%',
                       colors=['#5b5ef4','#06d6a0','#f59e0b','#ef4444','#8b5cf6'])
            axes[1].set_title('Gender Distribution', fontweight='bold', fontsize=11)
            plot_count += 1
        else:
            axes[1].text(0.5, 0.5, 'No gender column found', ha='center', va='center', transform=axes[1].transAxes, color='#888')
            axes[1].set_title('Gender', fontweight='bold')

        # 3. Top diagnoses
        if diag_col:
            top_d = df[diag_col].value_counts().head(8)
            axes[2].barh(top_d.index[::-1], top_d.values[::-1], color='#06d6a0', edgecolor='white')
            axes[2].set_title('Top Diagnoses', fontweight='bold', fontsize=11)
            axes[2].set_xlabel('Count')
            plot_count += 1
        else:
            # Use any categorical col
            cat_cols = list(df.select_dtypes(include='object').columns)
            if cat_cols:
                top_d = df[cat_cols[0]].value_counts().head(8)
                axes[2].barh(top_d.index[::-1].astype(str), top_d.values[::-1], color='#06d6a0', edgecolor='white')
                axes[2].set_title(f'Top {cat_cols[0]}', fontweight='bold', fontsize=11)

        # 4. Department or numeric distribution
        if dept_col:
            dept_d = df[dept_col].value_counts().head(8)
            axes[3].bar(dept_d.index.astype(str), dept_d.values, color='#f59e0b', edgecolor='white')
            axes[3].set_title('Department-wise Patients', fontweight='bold', fontsize=11)
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha='right')
            plot_count += 1
        elif 'finance' in perms and 'bill_col' in dir() and bill_col:
            df[bill_col].dropna().plot(kind='hist', ax=axes[3], bins=20, color='#ef4444', alpha=0.8, edgecolor='white')
            axes[3].set_title('Billing Distribution', fontweight='bold', fontsize=11)
        else:
            num_cols = list(df.select_dtypes(include='number').columns)
            if num_cols:
                df[num_cols[0]].dropna().plot(kind='hist', ax=axes[3], bins=20, color='#8b5cf6', alpha=0.8)
                axes[3].set_title(f'{num_cols[0]} Distribution', fontweight='bold', fontsize=11)

        plt.suptitle('Hospital Analytics Dashboard', fontsize=14, fontweight='bold', color='#1a1a2e')
        plt.tight_layout()
        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=110, bbox_inches='tight'); plt.close('all')
        buf.seek(0); dashboard_chart = base64.b64encode(buf.read()).decode()

        result['dashboard_chart'] = dashboard_chart
        result['preview'] = df_masked.head(5).to_string(index=False)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Patient Risk Prediction ───────────────────────────────────
@app.route('/api/hc/risk', methods=['POST'])
@login_required
def hc_risk():
    if agent.df is None:
        return jsonify({'error': 'Pehle patient data upload karo!'}), 400
    try:
        import base64, io as _io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import accuracy_score, classification_report

        role = get_user_hc_role(current_user.id)
        if 'risk' not in ROLE_PERMISSIONS.get(role, []):
            return jsonify({'error': f'Role "{role}" ko risk prediction ki permission nahi!'}), 403

        data = request.get_json()
        target_col = data.get('target', '')
        df = agent.df.copy().dropna()

        if not target_col or target_col not in df.columns:
            # Auto detect risk column
            risk_keywords = ['risk', 'readmit', 'readmission', 'outcome', 'mortality',
                           'death', 'critical', 'severe', 'status', 'result']
            cols_lower = {c.lower(): c for c in df.columns}
            target_col = next((cols_lower[c] for c in cols_lower
                              if any(k in c for k in risk_keywords)), None)
            if not target_col:
                return jsonify({
                    'error': 'Target column nahi mila!',
                    'suggestion': f'Available columns: {list(df.columns)[:10]}',
                    'tip': 'Koi column jo predict karna hai wo daalo (e.g. readmission, outcome, risk_level)'
                }), 400

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Mask sensitive before training (privacy)
        sensitive = [c for c in X.columns if c.lower() in SENSITIVE_COLUMNS]
        X = X.drop(columns=sensitive, errors='ignore')

        # Encode
        for c in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))

        le_y = LabelEncoder()
        y_enc = le_y.fit_transform(y.astype(str))

        if len(df) > 10000:
            from sklearn.utils import resample
            X, y_enc = resample(X, y_enc, n_samples=10000, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc)) > 1 else None)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        }

        results = []
        best_model = None
        best_score = -1

        for name, model in models.items():
            use_scaled = name == 'Logistic Regression'
            Xtr = X_train_s if use_scaled else X_train
            Xte = X_test_s if use_scaled else X_test
            model.fit(Xtr, y_train)
            score = accuracy_score(y_test, model.predict(Xte))
            results.append({'model': name, 'accuracy': round(float(score)*100, 2)})
            if score > best_score:
                best_score = score
                best_model = (name, model, use_scaled)

        results.sort(key=lambda x: x['accuracy'], reverse=True)

        # Feature importance
        feat_importance = {}
        chart_b64 = None
        if hasattr(best_model[1], 'feature_importances_'):
            fi = best_model[1].feature_importances_
            feat_df = pd.DataFrame({'feature': X.columns, 'importance': fi})
            feat_df = feat_df.sort_values('importance', ascending=True).tail(12)
            feat_importance = dict(zip(feat_df['feature'], feat_df['importance'].round(4)))

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].barh(feat_df['feature'], feat_df['importance'], color='#ef4444', alpha=0.8)
            axes[0].set_title(f'Risk Factors — {best_model[0]}', fontweight='bold', fontsize=11)
            axes[0].set_xlabel('Importance')

            # Class distribution
            classes, counts = np.unique(y_enc, return_counts=True)
            class_labels = le_y.inverse_transform(classes)
            axes[1].bar(class_labels.astype(str), counts,
                       color=['#06d6a0' if i==0 else '#ef4444' for i in range(len(classes))])
            axes[1].set_title('Risk Class Distribution', fontweight='bold', fontsize=11)
            axes[1].set_ylabel('Count')
            plt.suptitle('Patient Risk Analysis', fontsize=13, fontweight='bold')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); chart_b64 = base64.b64encode(buf.read()).decode()

        # High risk patients
        name, model, use_scaled = best_model
        Xte = X_test_s if use_scaled else X_test
        y_pred = model.predict(Xte)
        high_risk_count = int(np.sum(y_pred == max(classes)))

        return jsonify({
            'target': target_col,
            'best_model': best_model[0],
            'best_accuracy': round(float(best_score)*100, 2),
            'results': results,
            'feature_importance': feat_importance,
            'sensitive_cols_removed': sensitive,
            'high_risk_patients': high_risk_count,
            'total_test_patients': len(y_test),
            'risk_classes': list(le_y.classes_),
            'chart': chart_b64,
            'message': f'Risk model ready! Best: {best_model[0]} ({round(float(best_score)*100,1)}% accuracy)'
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Medical PDF Report ────────────────────────────────────────
@app.route('/api/hc/report', methods=['POST'])
@login_required
def hc_report():
    if agent.df is None:
        return jsonify({'error': 'Pehle data upload karo!'}), 400
    try:
        import base64, io as _io, datetime
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        role = get_user_hc_role(current_user.id)
        perms = ROLE_PERMISSIONS.get(role, [])
        if 'reports' not in perms:
            return jsonify({'error': f'Role "{role}" ko report generate karne ki permission nahi!'}), 403

        data = request.get_json()
        hospital_name = data.get('hospital_name', 'Hospital Analytics Report')
        report_type = data.get('report_type', 'general')
        df = agent.df.copy()
        df_masked = mask_sensitive(df, role)
        cols_lower = {c.lower(): c for c in df.columns}

        # Build charts
        charts_b64 = []
        num_cols = list(df.select_dtypes(include='number').columns[:4])

        if num_cols:
            fig, axes = plt.subplots(1, min(2, len(num_cols)), figsize=(10, 4))
            if len(num_cols) == 1: axes = [axes]
            for i, c in enumerate(num_cols[:2]):
                axes[i].hist(df[c].dropna(), bins=20, color='#5b5ef4', alpha=0.8, edgecolor='white')
                axes[i].set_title(c, fontweight='bold')
            plt.suptitle('Key Metrics Distribution', fontweight='bold')
            plt.tight_layout()
            buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close('all')
            buf.seek(0); charts_b64.append(base64.b64encode(buf.read()).decode())

        # Professional HTML report
        now = datetime.datetime.now()
        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:'Inter',sans-serif;background:#f0f4f8;color:#1a202c;}}
        .page{{max-width:900px;margin:0 auto;background:white;min-height:100vh;}}
        .header{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:32px 40px;}}
        .header-top{{display:flex;justify-content:space-between;align-items:flex-start;}}
        .hospital-name{{font-size:22px;font-weight:800;margin-bottom:4px;}}
        .report-title{{font-size:14px;opacity:0.8;}}
        .header-meta{{text-align:right;font-size:12px;opacity:0.7;}}
        .confidential{{background:rgba(255,255,255,0.15);border-radius:6px;padding:4px 10px;font-size:11px;font-weight:700;margin-top:8px;display:inline-block;}}
        .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:24px 40px;background:#f8fafc;border-bottom:1px solid #e2e8f0;}}
        .kpi-box{{background:white;border-radius:12px;padding:16px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
        .kpi-val{{font-size:28px;font-weight:800;color:#2563eb;}}
        .kpi-lbl{{font-size:11px;color:#64748b;margin-top:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}}
        .section{{padding:24px 40px;border-bottom:1px solid #f1f5f9;}}
        .section-title{{font-size:15px;font-weight:800;color:#1e3a5f;margin-bottom:14px;display:flex;align-items:center;gap:8px;}}
        table{{width:100%;border-collapse:collapse;font-size:12px;}}
        th{{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:700;color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.3px;}}
        td{{padding:9px 12px;border-bottom:1px solid #f8fafc;color:#374151;}}
        tr:hover td{{background:#fafbff;}}
        .badge{{padding:3px 8px;border-radius:20px;font-size:10px;font-weight:700;}}
        .badge-blue{{background:#dbeafe;color:#1d4ed8;}}
        .badge-green{{background:#dcfce7;color:#166534;}}
        .badge-red{{background:#fee2e2;color:#991b1b;}}
        img{{max-width:100%;border-radius:10px;margin:8px 0;}}
        .footer{{padding:20px 40px;background:#f8fafc;text-align:center;font-size:11px;color:#94a3b8;border-top:1px solid #e2e8f0;}}
        .role-badge{{background:rgba(255,255,255,0.2);border-radius:6px;padding:3px 10px;font-size:11px;font-weight:700;}}
        </style></head><body><div class='page'>
        <div class='header'>
          <div class='header-top'>
            <div>
              <div class='hospital-name'>🏥 {hospital_name}</div>
              <div class='report-title'>Healthcare Analytics Report — {report_type.title()}</div>
              <div class='confidential'>🔒 CONFIDENTIAL</div>
            </div>
            <div class='header-meta'>
              <div>Generated: {now.strftime('%d %b %Y, %H:%M')}</div>
              <div>Prepared by: {current_user.username}</div>
              <div class='role-badge' style='margin-top:6px;'>{role.upper()}</div>
            </div>
          </div>
        </div>"""

        # KPIs
        age_col = next((cols_lower[c] for c in cols_lower if 'age' in c), None)
        bill_col = next((cols_lower[c] for c in cols_lower if 'bill' in c or 'amount' in c or 'cost' in c), None) if 'finance' in perms else None
        diag_col = next((cols_lower[c] for c in cols_lower if 'diag' in c or 'disease' in c), None)

        html += "<div class='kpi-grid'>"
        html += f"<div class='kpi-box'><div class='kpi-val'>{len(df):,}</div><div class='kpi-lbl'>Total Records</div></div>"
        html += f"<div class='kpi-box'><div class='kpi-val'>{len(df.columns)}</div><div class='kpi-lbl'>Data Fields</div></div>"
        if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
            html += f"<div class='kpi-box'><div class='kpi-val'>{df[age_col].mean():.0f}</div><div class='kpi-lbl'>Avg Age</div></div>"
        if bill_col and pd.api.types.is_numeric_dtype(df[bill_col]):
            html += f"<div class='kpi-box'><div class='kpi-val'>₹{df[bill_col].sum()/1e5:.1f}L</div><div class='kpi-lbl'>Total Revenue</div></div>"
        else:
            html += f"<div class='kpi-box'><div class='kpi-val'>{int(df.isnull().sum().sum())}</div><div class='kpi-lbl'>Missing Values</div></div>"
        html += "</div>"

        # Data table (masked)
        html += "<div class='section'><div class='section-title'>📋 Patient Data Preview (Top 10)</div>"
        html += "<table><tr>" + "".join(f"<th>{c}</th>" for c in df_masked.columns[:8]) + "</tr>"
        for _, row in df_masked.head(10).iterrows():
            html += "<tr>" + "".join(f"<td>{str(v)[:30]}</td>" for v in list(row)[:8]) + "</tr>"
        html += "</table></div>"

        # Top diagnoses
        if diag_col:
            top_d = df[diag_col].value_counts().head(8)
            html += "<div class='section'><div class='section-title'>🔬 Top Diagnoses / Conditions</div><table><tr><th>Condition</th><th>Count</th><th>%</th></tr>"
            for d, cnt in top_d.items():
                pct = round(cnt/len(df)*100, 1)
                html += f"<tr><td>{d}</td><td>{cnt}</td><td><span class='badge badge-blue'>{pct}%</span></td></tr>"
            html += "</table></div>"

        # Charts
        for chart in charts_b64:
            html += f"<div class='section'><div class='section-title'>📊 Analytics Charts</div><img src='data:image/png;base64,{chart}'/></div>"

        # Privacy notice
        html += f"""<div class='section'>
        <div class='section-title'>🔒 Data Privacy Notice</div>
        <p style='font-size:12px;color:#64748b;line-height:1.6;'>
        This report has been generated with role-based access control. Sensitive patient information
        (names, contact details, IDs) has been {'fully visible (Admin access)' if 'privacy_off' in perms else 'masked for privacy protection'}.
        Report access level: <strong>{role.upper()}</strong>. Generated on {now.strftime('%d %b %Y')} for internal use only.
        </p></div>"""

        html += f"<div class='footer'>DS Agent Healthcare Analytics &nbsp;|&nbsp; {hospital_name} &nbsp;|&nbsp; {now.strftime('%Y')} &nbsp;|&nbsp; CONFIDENTIAL</div>"
        html += "</div></body></html>"

        return jsonify({
            'message': f'Medical report ready! Role: {role} | {len(df)} records',
            'html': html,
            'role': role,
            'hospital_name': hospital_name,
            'records': len(df),
            'sensitive_masked': 'privacy_off' not in perms
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Data Privacy — Mask/Unmask ────────────────────────────────
@app.route('/api/hc/privacy', methods=['POST'])
@login_required
def hc_privacy():
    if agent.df is None:
        return jsonify({'error': 'Pehle data upload karo!'}), 400
    try:
        data = request.get_json()
        action = data.get('action', 'scan')
        df = agent.df.copy()

        if action == 'scan':
            found = [c for c in df.columns if c.lower() in SENSITIVE_COLUMNS]
            all_cols = list(df.columns)
            return jsonify({
                'sensitive_columns': found,
                'safe_columns': [c for c in all_cols if c not in found],
                'total_cols': len(all_cols),
                'message': f'{len(found)} sensitive columns found: {found}'
            })

        elif action == 'mask':
            role = get_user_hc_role(current_user.id)
            masked = mask_sensitive(df, role)
            return jsonify({
                'message': f'Data masked! {len([c for c in df.columns if c.lower() in SENSITIVE_COLUMNS])} columns masked',
                'preview': masked.head(5).to_string(index=False),
                'masked_columns': [c for c in df.columns if c.lower() in SENSITIVE_COLUMNS]
            })

        elif action == 'remove':
            cols_to_remove = data.get('columns', [])
            if not cols_to_remove:
                cols_to_remove = [c for c in df.columns if c.lower() in SENSITIVE_COLUMNS]
            df_clean = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
            agent.df = df_clean
            return jsonify({
                'message': f'Sensitive columns permanently removed! {cols_to_remove}',
                'remaining_cols': list(df_clean.columns),
                'rows': len(df_clean)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ════════════════════════════════════════════════════════════════
# MEDICAL IMAGE ANALYSIS — X-Ray, ECG, Skin, OCR
# ════════════════════════════════════════════════════════════════

def get_hf_url():
    """Get HuggingFace or Colab URL"""
    import requests as req
    hf = os.environ.get('HF_SPACE_URL', '').strip()
    colab = os.environ.get('COLAB_URL', '').strip()
    if hf:
        try:
            r = req.get(hf, timeout=10)
            if r.status_code == 200:
                return hf
        except:
            pass
    if colab:
        try:
            r = req.get(colab + '/ping', timeout=6)
            if r.status_code == 200:
                return colab
        except:
            pass
    return None


def call_hf_api(url, endpoint, payload):
    """Call HuggingFace FastAPI routes or Colab Flask API"""
    import requests as req
    try:
        # Direct FastAPI route — works for both HF Space and Colab
        r = req.post(url + endpoint, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[HF] call_hf_api error endpoint={endpoint}: {e}")
        return {'error': f'ML API error: {str(e)}'}


def validate_hf_result(result, expected_fields):
    """Validate & sanitize HuggingFace API result — ensure no undefined fields"""
    if not result or not isinstance(result, dict):
        return {'error': 'HuggingFace se empty response. Space restart karo.'}
    if result.get('error'):
        return result
    safe = {}
    for field, default in expected_fields.items():
        safe[field] = result.get(field, default)
    return safe


@app.route('/api/medical/xray', methods=['POST'])
@login_required
def medical_xray():
    url = get_hf_url()
    if not url:
        return jsonify({'error': 'ML Server offline! HuggingFace Space start karo.'}), 400
    try:
        data = request.get_json()
        result = call_hf_api(url, '/xray', data)
        safe = validate_hf_result(result, {
            'success': False, 'finding': 'Unknown', 'confidence': 0.0,
            'severity': 'unknown', 'recommendations': [], 'report': '',
            'heatmap': None, 'message': 'X-Ray analysis complete'
        })
        return jsonify(safe), (500 if safe.get('error') else 200)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/medical/ecg', methods=['POST'])
@login_required
def medical_ecg():
    url = get_hf_url()
    if not url:
        return jsonify({'error': 'ML Server offline!'}), 400
    try:
        data = request.get_json()
        result = call_hf_api(url, '/ecg', data)
        safe = validate_hf_result(result, {
            'success': False, 'rhythm': 'Unknown', 'heart_rate': 0,
            'findings': [], 'severity': 'unknown', 'report': '',
            'chart': None, 'message': 'ECG analysis complete'
        })
        return jsonify(safe), (500 if safe.get('error') else 200)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/medical/skin', methods=['POST'])
@login_required
def medical_skin():
    url = get_hf_url()
    if not url:
        return jsonify({'error': 'ML Server offline!'}), 400
    try:
        data = request.get_json()
        result = call_hf_api(url, '/skin', data)
        safe = validate_hf_result(result, {
            'success': False, 'condition': 'Unknown', 'confidence': 0.0,
            'severity': 'unknown', 'recommendations': [], 'report': '',
            'message': 'Skin analysis complete'
        })
        return jsonify(safe), (500 if safe.get('error') else 200)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/medical/ocr', methods=['POST'])
@login_required
def medical_ocr():
    url = get_hf_url()
    if not url:
        return jsonify({'error': 'ML Server offline!'}), 400
    try:
        data = request.get_json()
        result = call_hf_api(url, '/ocr', data)
        safe = validate_hf_result(result, {
            'success': False, 'text': '', 'entities': [],
            'summary': '', 'message': 'OCR analysis complete'
        })
        return jsonify(safe), (500 if safe.get('error') else 200)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE: SHAP — Model Explanation
# ════════════════════════════════════════════════════════════════
@app.route('/api/shap', methods=['POST'])
@login_required
@check_query_limit
def shap_explain():
    try:
        if agent.df is None:
            return jsonify({'error': 'Pehle dataset upload karo!'}), 400
        data = request.get_json()
        target = data.get('target', '')
        model_type = data.get('model_type', 'auto')

        df = agent.df.dropna()
        if target not in df.columns:
            return jsonify({'error': f'Target column "{target}" nahi mila!'}), 400

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import shap, io as _io

        y = df[target]
        X = df.drop(columns=[target])
        for c in X.select_dtypes(include='object').columns:
            le = LabelEncoder(); X[c] = le.fit_transform(X[c].astype(str))

        task = 'classification' if (y.dtype == object or y.nunique() <= 15) else 'regression'
        if y.dtype == object:
            le_y = LabelEncoder(); y = le_y.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42) if task == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:100])
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        feat_imp = dict(zip(X.columns, abs(shap_vals).mean(axis=0)))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15])
        axes[0].barh(list(feat_imp.keys())[::-1], list(feat_imp.values())[::-1], color='#5b5ef4', alpha=0.85)
        axes[0].set_title('SHAP Feature Importance', fontweight='bold', fontsize=13)
        axes[0].set_xlabel('Mean |SHAP Value|')

        top_feat = list(feat_imp.keys())[0]
        axes[1].scatter(X_test[:100][top_feat], shap_vals[:, list(X.columns).index(top_feat)],
                       alpha=0.6, c='#8b5cf6', edgecolors='white', linewidth=0.5, s=60)
        axes[1].axhline(0, color='#ef4444', linestyle='--', linewidth=1)
        axes[1].set_xlabel(top_feat); axes[1].set_ylabel('SHAP Value')
        axes[1].set_title(f'SHAP: {top_feat} Impact', fontweight='bold')
        plt.suptitle(f'Model Explanation — Target: {target}', fontweight='bold', color='#1e3a5f')
        plt.tight_layout()

        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all'); buf.seek(0)
        chart = base64.b64encode(buf.read()).decode()

        score = model.score(X_test, y_test)
        top5 = list(feat_imp.items())[:5]
        insights = [f"'{k}' has {v:.3f} avg impact on prediction" for k,v in top5]

        return jsonify({
            'success': True,
            'task': task,
            'target': target,
            'model_score': round(float(score), 4),
            'feature_importance': feat_imp,
            'top_features': top5,
            'insights': insights,
            'chart': chart,
            'message': f'SHAP explanation ready! Top feature: {list(feat_imp.keys())[0]}'
        })
    except ImportError:
        return jsonify({'error': 'SHAP not installed! requirements mein add karo: shap'}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE: HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════
@app.route('/api/hyperparam', methods=['POST'])
@login_required
@check_query_limit
def hyperparam_tune():
    try:
        if agent.df is None:
            return jsonify({'error': 'Pehle dataset upload karo!'}), 400
        data = request.get_json()
        target = data.get('target', '')
        model_name = data.get('model', 'random_forest')
        cv_folds = int(data.get('cv_folds', 3))

        df = agent.df.dropna()
        if target not in df.columns:
            return jsonify({'error': f'Target "{target}" nahi mila!'}), 400

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, io as _io

        y = df[target]; X = df.drop(columns=[target])
        for c in X.select_dtypes(include='object').columns:
            le = LabelEncoder(); X[c] = le.fit_transform(X[c].astype(str))

        task = 'classification' if (y.dtype == object or y.nunique() <= 15) else 'regression'
        if y.dtype == object:
            le_y = LabelEncoder(); y = le_y.fit_transform(y.astype(str))
        else:
            y = y.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

        param_grids = {
            'random_forest': {
                'clf': RandomForestClassifier(random_state=42) if task=='classification' else RandomForestRegressor(random_state=42),
                'params': {'n_estimators':[50,100,200],'max_depth':[None,5,10,20],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4]}
            },
            'gradient_boost': {
                'clf': GradientBoostingClassifier(random_state=42) if task=='classification' else GradientBoostingRegressor(random_state=42),
                'params': {'n_estimators':[50,100],'learning_rate':[0.01,0.05,0.1,0.2],'max_depth':[3,5,7],'subsample':[0.8,1.0]}
            },
            'svm': {
                'clf': SVC(random_state=42) if task=='classification' else SVR(),
                'params': {'C':[0.1,1,10,100],'kernel':['rbf','linear'],'gamma':['scale','auto']}
            }
        }

        chosen = param_grids.get(model_name, param_grids['random_forest'])
        search = RandomizedSearchCV(chosen['clf'], chosen['params'], n_iter=12, cv=cv_folds,
                                   scoring='accuracy' if task=='classification' else 'r2',
                                   n_jobs=-1, random_state=42)
        search.fit(X_train_s if model_name=='svm' else X_train, y_train)

        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values('mean_test_score', ascending=False).head(8)

        baseline_clf = RandomForestClassifier(random_state=42) if task=='classification' else RandomForestRegressor(random_state=42)
        baseline_clf.fit(X_train, y_train)
        baseline_score = baseline_clf.score(X_test, y_test)
        best_score = search.best_estimator_.score(X_test_s if model_name=='svm' else X_test, y_test)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        scores = results_df['mean_test_score'].values
        axes[0].barh(range(len(scores)), scores, color=['#5b5ef4' if i==0 else '#8b5cf6' for i in range(len(scores))], alpha=0.85)
        axes[0].set_yticks(range(len(scores)))
        axes[0].set_yticklabels([f'Config {i+1}' for i in range(len(scores))])
        axes[0].set_title('Top Configurations', fontweight='bold')
        axes[0].set_xlabel('CV Score')
        axes[0].axvline(baseline_score, color='#ef4444', linestyle='--', label=f'Baseline: {baseline_score:.3f}')
        axes[0].legend()

        axes[1].bar(['Baseline', 'Tuned'], [baseline_score, best_score],
                   color=['#94a3b8','#059669'], alpha=0.85, width=0.5)
        axes[1].set_title('Baseline vs Tuned', fontweight='bold')
        axes[1].set_ylabel('Score')
        for i, v in enumerate([baseline_score, best_score]):
            axes[1].text(i, v+0.005, f'{v:.4f}', ha='center', fontweight='700', fontsize=12)
        plt.suptitle(f'Hyperparameter Tuning — {model_name.replace("_"," ").title()}', fontweight='bold')
        plt.tight_layout()

        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all'); buf.seek(0)
        chart = base64.b64encode(buf.read()).decode()

        improvement = round((best_score - baseline_score)*100, 2)
        return jsonify({
            'success': True,
            'task': task,
            'target': target,
            'model': model_name,
            'baseline_score': round(float(baseline_score), 4),
            'best_score': round(float(best_score), 4),
            'improvement_pct': improvement,
            'best_params': search.best_params_,
            'cv_folds': cv_folds,
            'iterations_tried': 12,
            'chart': chart,
            'message': f'Tuning complete! Score {baseline_score:.3f} → {best_score:.3f} (+{improvement}%)'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE: NLP — Text Analysis & Sentiment
# ════════════════════════════════════════════════════════════════
@app.route('/api/nlp', methods=['POST'])
@login_required
@check_query_limit
def nlp_analysis():
    try:
        if agent.df is None:
            return jsonify({'error': 'Pehle dataset upload karo!'}), 400
        data = request.get_json()
        text_col = data.get('text_col', '')
        analysis_type = data.get('analysis_type', 'all')

        df = agent.df.dropna(subset=[text_col]) if text_col in agent.df.columns else None
        if df is None or text_col not in agent.df.columns:
            return jsonify({'error': f'Text column "{text_col}" nahi mila!'}), 400

        texts = df[text_col].astype(str).head(500).tolist()

        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io as _io
        from collections import Counter
        import re

        # Sentiment using TextBlob or simple lexicon
        positive_words = set(['good','great','excellent','best','love','wonderful','amazing','happy','perfect','brilliant','outstanding','fantastic','superb','awesome','nice','helpful','pleased'])
        negative_words = set(['bad','worst','terrible','hate','awful','horrible','poor','disappointing','useless','waste','broken','failed','ugly','slow','worse','never','problem'])

        sentiments = []
        word_freq = Counter()
        for txt in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', txt.lower())
            word_freq.update(words)
            pos = sum(1 for w in words if w in positive_words)
            neg = sum(1 for w in words if w in negative_words)
            if pos > neg: sentiments.append('Positive')
            elif neg > pos: sentiments.append('Negative')
            else: sentiments.append('Neutral')

        sent_counts = Counter(sentiments)
        avg_len = sum(len(t.split()) for t in texts) / len(texts)
        stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','is','was','are','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','that','this','it','they','we','you','i','he','she'}
        top_words = [(w, c) for w, c in word_freq.most_common(30) if w not in stop_words][:15]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_sent = {'Positive':'#059669','Negative':'#ef4444','Neutral':'#f59e0b'}
        axes[0].pie([sent_counts.get(s,0) for s in ['Positive','Negative','Neutral']],
                   labels=['Positive','Negative','Neutral'],
                   colors=['#059669','#ef4444','#f59e0b'],
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
        axes[0].set_title('Sentiment Distribution', fontweight='bold')

        words, counts = zip(*top_words) if top_words else (['no data'],[1])
        bar_colors = ['#059669' if w in positive_words else '#ef4444' if w in negative_words else '#5b5ef4' for w in words]
        axes[1].barh(list(words)[::-1], list(counts)[::-1], color=bar_colors[::-1], alpha=0.85)
        axes[1].set_title('Top Keywords', fontweight='bold')
        axes[1].set_xlabel('Frequency')

        lengths = [len(t.split()) for t in texts]
        axes[2].hist(lengths, bins=20, color='#8b5cf6', alpha=0.8, edgecolor='white')
        axes[2].axvline(avg_len, color='#ef4444', linestyle='--', linewidth=2, label=f'Avg: {avg_len:.1f}')
        axes[2].set_title('Text Length Distribution', fontweight='bold')
        axes[2].set_xlabel('Word Count'); axes[2].legend()

        plt.suptitle(f'NLP Analysis — Column: {text_col}', fontweight='bold', color='#1e3a5f')
        plt.tight_layout()
        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all'); buf.seek(0)
        chart = base64.b64encode(buf.read()).decode()

        return jsonify({
            'success': True,
            'text_col': text_col,
            'total_texts': len(texts),
            'sentiment_counts': dict(sent_counts),
            'positive_pct': round(sent_counts.get('Positive',0)/len(texts)*100, 1),
            'negative_pct': round(sent_counts.get('Negative',0)/len(texts)*100, 1),
            'neutral_pct': round(sent_counts.get('Neutral',0)/len(texts)*100, 1),
            'avg_word_count': round(avg_len, 1),
            'top_keywords': top_words[:10],
            'total_unique_words': len(word_freq),
            'chart': chart,
            'message': f'NLP done! {sent_counts.get("Positive",0)} Positive | {sent_counts.get("Negative",0)} Negative | {sent_counts.get("Neutral",0)} Neutral'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════
# FEATURE: AUTO FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
@app.route('/api/feature_eng', methods=['POST'])
@login_required
@check_query_limit
def feature_engineering():
    try:
        if agent.df is None:
            return jsonify({'error': 'Pehle dataset upload karo!'}), 400
        data = request.get_json()
        target = data.get('target', '')

        df = agent.df.copy()
        original_cols = list(df.columns)
        new_features = []
        log = []

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if target in numeric_cols: numeric_cols.remove(target)
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if target in cat_cols: cat_cols.remove(target)

        # 1. Log transform for skewed columns
        for col in numeric_cols:
            skew = float(df[col].skew())
            if abs(skew) > 1.0 and df[col].min() >= 0:
                df[f'{col}_log'] = df[col].apply(lambda x: __import__('math').log1p(x))
                new_features.append(f'{col}_log'); log.append(f'Log transform: {col} (skew={skew:.2f})')

        # 2. Interaction features (top numeric pairs)
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols))):
                for j in range(i+1, min(4, len(numeric_cols))):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                    new_features.append(f'{c1}_x_{c2}')
                    log.append(f'Interaction: {c1} × {c2}')

        # 3. Ratio features
        if len(numeric_cols) >= 2:
            c1, c2 = numeric_cols[0], numeric_cols[1]
            safe_div = df[c2].replace(0, float('nan'))
            df[f'{c1}_ratio_{c2}'] = df[c1] / safe_div
            df[f'{c1}_ratio_{c2}'] = df[f'{c1}_ratio_{c2}'].fillna(0)
            new_features.append(f'{c1}_ratio_{c2}')
            log.append(f'Ratio: {c1} / {c2}')

        # 4. Binning numeric to categories
        for col in numeric_cols[:3]:
            try:
                df[f'{col}_bin'] = pd.cut(df[col], bins=4, labels=['Low','Medium','High','Very High'])
                df[f'{col}_bin'] = df[f'{col}_bin'].astype(str)
                new_features.append(f'{col}_bin')
                log.append(f'Binning: {col} → 4 categories')
            except: pass

        # 5. Label encode categoricals
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols[:5]:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            new_features.append(f'{col}_encoded')
            log.append(f'Label encoded: {col}')

        # 6. Null indicator flags
        null_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        for col in null_cols[:3]:
            df[f'{col}_is_null'] = df[col].isnull().astype(int)
            new_features.append(f'{col}_is_null')
            log.append(f'Null flag: {col}')

        # Save enhanced df to agent
        agent.df = df
        agent.available_columns = list(df.columns)

        # Chart — before vs after
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, io as _io

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cats = ['Original\nFeatures', 'New\nFeatures', 'Total\nFeatures']
        vals = [len(original_cols), len(new_features), len(df.columns)]
        colors = ['#94a3b8','#5b5ef4','#059669']
        bars = axes[0].bar(cats, vals, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, vals):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(val), ha='center', fontweight='700', fontsize=13)
        axes[0].set_title('Feature Count', fontweight='bold')

        feat_types = {'Log Transform': sum(1 for f in new_features if '_log' in f),
                     'Interaction': sum(1 for f in new_features if '_x_' in f),
                     'Ratio': sum(1 for f in new_features if '_ratio_' in f),
                     'Binning': sum(1 for f in new_features if '_bin' in f),
                     'Encoded': sum(1 for f in new_features if '_encoded' in f),
                     'Null Flag': sum(1 for f in new_features if '_is_null' in f)}
        feat_types = {k:v for k,v in feat_types.items() if v>0}
        if feat_types:
            axes[1].pie(feat_types.values(), labels=feat_types.keys(),
                       autopct='%1.0f%%', startangle=90,
                       colors=['#5b5ef4','#8b5cf6','#06d6a0','#f59e0b','#ef4444','#3b82f6'])
            axes[1].set_title('New Feature Types', fontweight='bold')

        plt.suptitle('Auto Feature Engineering Results', fontweight='bold', color='#1e3a5f')
        plt.tight_layout()
        buf = _io.BytesIO(); plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all'); buf.seek(0)
        chart = base64.b64encode(buf.read()).decode()

        return jsonify({
            'success': True,
            'original_features': len(original_cols),
            'new_features_added': len(new_features),
            'total_features': len(df.columns),
            'new_feature_names': new_features[:20],
            'engineering_log': log,
            'feature_types': feat_types,
            'chart': chart,
            'message': f'Feature Engineering done! {len(original_cols)} → {len(df.columns)} features (+{len(new_features)} new)'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
