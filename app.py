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

from core.database import (engine, Base, SessionLocal, UserQuery,
                            AffiliateLink, QueryCount, ScreenPost, MusicTrack,
                            VideoTrack, ChatMessage, PostAnalytics,
                            ScheduledPost, PushSubscription, PaymentScreenshot)
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
UPLOAD_DIR = '/tmp/uploads' if os.path.exists('/tmp') else '.'
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ── UPI Config ──────────────────────────────────────────────────
UPI_ID    = os.getenv("UPI_ID", "yourname@upi")        # Set in env
UPI_NAME  = os.getenv("UPI_NAME", "DS Agent")
UPI_AMOUNT = "499"

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
        # Admin ke liye koi limit nahi
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
    return render_template('pricing.html', plan=get_user_plan(current_user.id))

# ── User Info ─────────────────────────────────────────────────────
@app.route('/get_username')
@login_required
def get_username():
    # Admin hamesha pro plan pe hota hai
    if current_user.is_admin:
        return jsonify({
            "username": current_user.username,
            "is_admin": True,
            "plan": "pro",
            "queries_used": 0,
            "queries_limit": 99999
        })
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
        "is_admin": False,
        "plan": plan,
        "queries_used": queries_used,
        "queries_limit": FREE_DAILY_LIMIT if plan == "free" else 99999
    })

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
            user_id    = current_user.id,
            username   = current_user.username,
            image_data = data['image_data'],
            image_mime = data.get('image_mime', 'image/jpeg'),
            utr_number = data.get('utr_number', ''),
            amount     = data.get('amount', '499'),
            status     = 'pending'
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
        items = db.query(PaymentScreenshot).order_by(
            PaymentScreenshot.created_at.desc()).all()
        return jsonify([{
            "id": s.id,
            "username": s.username,
            "image_data": s.image_data,
            "image_mime": s.image_mime,
            "utr_number": s.utr_number or "N/A",
            "amount": s.amount or "499",
            "status": s.status,
            "created_at": s.created_at.strftime('%d %b %H:%M')
        } for s in items])
    finally:
        db.close()

# ── Admin: Approve/Reject Payment ────────────────────────────────
@app.route('/admin/review_payment', methods=['POST'])
@login_required
@admin_required
def review_payment():
    data = request.get_json()
    ss_id  = data.get('id')
    status = data.get('status')
    if status not in ('approved', 'rejected'):
        return jsonify({"error": "Invalid status"}), 400
    db = SessionLocal()
    try:
        ss = db.query(PaymentScreenshot).filter_by(id=ss_id).first()
        if not ss:
            return jsonify({"error": "Not found"}), 404
        ss.status = status
        if status == 'approved':
            set_user_plan(ss.user_id, 'pro',
                          expires_at=datetime.utcnow() + timedelta(days=30))
            logger.info(f"✅ Pro activated for user {ss.user_id} ({ss.username})")
        db.commit()
        msg = f"✅ Approved! Pro activated for {ss.username}" if status == 'approved' else f"❌ Rejected for {ss.username}"
        return jsonify({"message": msg})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ── Affiliate Links ───────────────────────────────────────────────
@app.route('/affiliate_links', methods=['GET'])
@login_required
def get_affiliate_links():
    db = SessionLocal()
    try:
        links = db.query(AffiliateLink).filter_by(is_active=True).all()
        return jsonify([{"id":l.id,"title":l.title,"url":l.url,"description":l.description} for l in links])
    finally:
        db.close()

@app.route('/admin/affiliate', methods=['POST'])
@login_required
@admin_required
def add_affiliate():
    data = request.get_json()
    db = SessionLocal()
    try:
        link = AffiliateLink(title=data.get('title','Sponsored'),url=data.get('url',''),description=data.get('description',''),is_active=True)
        db.add(link);db.commit()
        return jsonify({"message":"✅ Added!","id":link.id})
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

@app.route('/admin/affiliate/<int:link_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_affiliate(link_id):
    db = SessionLocal()
    try:
        link = db.query(AffiliateLink).filter_by(id=link_id).first()
        if link:link.is_active=False;db.commit();return jsonify({"message":"✅ Removed!"})
        return jsonify({"error":"Not found"}),404
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
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
            link.title=data.get('title',link.title);link.url=data.get('url',link.url);link.description=data.get('description',link.description)
            db.commit();return jsonify({"message":"✅ Updated!"})
        return jsonify({"error":"Not found"}),404
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
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
            sub = db.query(Subscription).filter_by(user_id=u.id,is_active=True).first()
            result.append({"id":u.id,"username":u.username,"is_admin":u.is_admin,"plan":sub.plan if sub else "free"})
        return jsonify(result)
    finally:
        db.close()

@app.route('/admin/query_stats')
@login_required
@admin_required
def admin_query_stats():
    from core.database import Subscription
    db = SessionLocal()
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        all_counts = db.query(QueryCount).all()
        user_totals = {}
        for qc in all_counts:
            user_totals[qc.user_id] = user_totals.get(qc.user_id,0)+qc.count
        today_counts = db.query(QueryCount).filter_by(date=today).all()
        today_map = {qc.user_id:qc.count for qc in today_counts}
        users = db.query(User).all()
        name_map = {u.id:u.username for u in users}
        totals = sorted([{"user_id":uid,"username":name_map.get(uid,f"user_{uid}"),"count":cnt} for uid,cnt in user_totals.items()],key=lambda x:x['count'],reverse=True)
        today_data = sorted([{"user_id":uid,"username":name_map.get(uid,f"user_{uid}"),"count":cnt} for uid,cnt in today_map.items()],key=lambda x:x['count'],reverse=True)
        return jsonify({"totals":totals,"today":today_data})
    except Exception as e:
        return jsonify({"error":str(e),"totals":[],"today":[]}),500
    finally:
        db.close()

@app.route('/admin/set_plan', methods=['POST'])
@login_required
@admin_required
def admin_set_plan():
    data = request.get_json()
    user_id=data.get('user_id');plan=data.get('plan','free');expires_days=data.get('expires_days')
    expires_at=(datetime.utcnow()+timedelta(days=int(expires_days)) if expires_days else None)
    set_user_plan(user_id,plan,expires_at=expires_at)
    return jsonify({"message":f"✅ User {user_id} plan set to {plan}"})

# ── Upload ────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
@login_required
@single_session_check
def upload():
    if 'file' not in request.files:
        return jsonify({"error":"No file"}),400
    file = request.files['file']
    if file and file.filename:
        filename = secure_filename(file.filename)
        try:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            result = agent.load_data(filename)
            return jsonify({"message":f"✅ Uploaded: {filename}\n{result}"}),200
        except Exception as e:
            return jsonify({"error":str(e)}),500
    return jsonify({"error":"No file selected"}),400

# ── Chat ──────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
@login_required
@single_session_check
@check_query_limit
def chat():
    data = request.get_json()
    user_message = data.get('message','').strip()
    ai_mode = data.get('ai_mode',False)
    if ai_mode and not current_user.is_admin:
        plan = get_user_plan(current_user.id)
        if plan != "pro":
            return jsonify({"response":"⭐ AI Chat Mode sirf Pro plan mein available hai!","upgrade_required":True})
    try:
        if ai_mode:
            response = agent.conversational_query(user_message,user_id=current_user.id)
        else:
            response = agent.query(user_message,user_id=current_user.id)
        return jsonify({"response":response})
    except Exception as e:
        return jsonify({"response":"⚠️ Error processing query."}),500

# ── Downloads ─────────────────────────────────────────────────────
@app.route('/download/csv')
@login_required
@single_session_check
def download_csv():
    if agent.df is None:return jsonify({"error":"Pehle file upload karo!"}),400
    csv_buffer=io.StringIO();agent.df.to_csv(csv_buffer,index=False)
    return Response(csv_buffer.getvalue().encode('utf-8-sig'),mimetype='text/csv',headers={"Content-Disposition":"attachment; filename=data_export.csv"})

@app.route('/download/excel')
@login_required
@single_session_check
def download_excel():
    plan=get_user_plan(current_user.id)
    if plan!="pro" and not current_user.is_admin:
        return jsonify({"error":"PRO_REQUIRED","message":"⭐ Excel export sirf Pro plan mein!"}),403
    if agent.df is None:return jsonify({"error":"Pehle file upload karo!"}),400
    excel_buffer=io.BytesIO()
    with pd.ExcelWriter(excel_buffer,engine='openpyxl') as writer:
        agent.df.to_excel(writer,sheet_name='Raw Data',index=False)
        agent.df.describe().to_excel(writer,sheet_name='Summary Stats')
    excel_buffer.seek(0)
    return send_file(excel_buffer,mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',as_attachment=True,download_name='analysis_report.xlsx')

@app.route('/download/html')
@login_required
@single_session_check
def download_html():
    if agent.df is None:return jsonify({"error":"Pehle file upload karo!"}),400
    html_content=f"""<!DOCTYPE html><html><head><title>Report</title></head><body><h1>🤖 DS Agent Report</h1><p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>{agent.df.head(10).to_html(index=False)}{agent.df.describe().to_html()}</body></html>"""
    return Response(html_content,mimetype='text/html',headers={"Content-Disposition":"attachment; filename=report.html"})

@app.route('/plot.png')
def plot_png():
    plot_path = '/tmp/plot.png' if os.path.exists('/tmp/plot.png') else os.path.join('static','plot.png')
    if not os.path.exists(plot_path):
        return jsonify({'error':'Plot nahi bana abhi tak'}),404
    return send_file(plot_path, mimetype='image/png')

@app.route('/dashboard.png')
def dashboard_png():
    if current_user.is_authenticated:
        if not current_user.is_admin:
            plan=get_user_plan(current_user.id)
            if plan!="pro":return jsonify({"error":"PRO_REQUIRED"}),403
    dash_path = '/tmp/dashboard.png' if os.path.exists('/tmp/dashboard.png') else os.path.join('static','dashboard.png')
    if not os.path.exists(dash_path):
        return jsonify({'error':'Dashboard nahi bana abhi tak'}),404
    return send_file(dash_path, mimetype='image/png')

@app.route('/send_report', methods=['POST'])
@login_required
@single_session_check
def send_report():
    try:
        data=request.get_json();client_email=data.get('email')
        if not client_email:return jsonify({"error":"Email required"}),400
        pdf_filename=f"static/report_{current_user.username}_{int(datetime.now().timestamp())}.pdf"
        db=SessionLocal()
        last_query=db.query(UserQuery).filter_by(user_id=current_user.id).order_by(UserQuery.timestamp.desc()).first()
        insights=last_query.response_text if last_query else "No data"
        db.close()
        success=generate_pdf_report(pdf_filename,client_email,insights,'static/plot.png')
        if success:
            email_sent=send_report_email(to_email=client_email,subject="📊 Your Data Analysis Report",body="Hi,\n\nPlease find attached your report.\n\nRegards,\nDS Agent",attachment_path=pdf_filename)
            if email_sent:return jsonify({"message":"✅ Report sent!"})
            return jsonify({"error":"Email failed"}),500
        return jsonify({"error":"PDF generation failed"}),500
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/screen')
@login_required
def screen():return render_template('screen.html')

# ── Screen Posts ──────────────────────────────────────────────────
@app.route('/api/screen/posts')
@login_required
def get_screen_posts():
    db=SessionLocal()
    try:
        posts=db.query(ScreenPost).filter_by(is_active=True).order_by(ScreenPost.order_num.asc(),ScreenPost.created_at.desc()).all()
        return jsonify([{"id":p.id,"title":p.title,"content":p.content,"post_type":p.post_type,"affiliate_url":p.affiliate_url,"image_data":p.image_data,"image_mime":p.image_mime} for p in posts])
    finally:
        db.close()

@app.route('/api/screen/posts', methods=['POST'])
@login_required
@admin_required
def add_screen_post():
    import base64
    db=SessionLocal()
    try:
        post_type=request.form.get('post_type','text');title=request.form.get('title','');content=request.form.get('content','');aff_url=request.form.get('affiliate_url','');order_num=int(request.form.get('order_num',0))
        image_data,image_mime=None,None
        if post_type=='image' and 'image' in request.files:
            img=request.files['image']
            if img and img.filename:image_data=base64.b64encode(img.read()).decode('utf-8');image_mime=img.content_type or 'image/jpeg'
        post=ScreenPost(title=title,content=content,post_type=post_type,affiliate_url=aff_url if aff_url else None,image_data=image_data,image_mime=image_mime,order_num=order_num,is_active=True)
        db.add(post);db.commit()
        return jsonify({"message":"✅ Post added!","id":post.id})
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

@app.route('/api/screen/posts/<int:post_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_screen_post(post_id):
    db=SessionLocal()
    try:
        p=db.query(ScreenPost).filter_by(id=post_id).first()
        if p:p.is_active=False;db.commit();return jsonify({"message":"✅ Removed!"})
        return jsonify({"error":"Not found"}),404
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

# ── Music ─────────────────────────────────────────────────────────
@app.route('/api/music')
@login_required
def get_music():
    db=SessionLocal()
    try:
        tracks=db.query(MusicTrack).filter_by(is_active=True).order_by(MusicTrack.created_at.asc()).all()
        return jsonify([{"id":t.id,"title":t.title,"artist":t.artist or "Unknown","audio_data":t.audio_data,"mime_type":t.mime_type} for t in tracks])
    finally:
        db.close()

@app.route('/api/music', methods=['POST'])
@login_required
@admin_required
def upload_music():
    import base64
    db=SessionLocal()
    try:
        if 'audio' not in request.files:return jsonify({"error":"No file"}),400
        f=request.files['audio'];title=request.form.get('title',f.filename or 'Unknown');artist=request.form.get('artist','')
        if not f or not f.filename:return jsonify({"error":"Empty file"}),400
        raw=f.read();audio_b64=base64.b64encode(raw).decode('utf-8');mime=f.content_type or 'audio/mpeg'
        track=MusicTrack(title=title,artist=artist,audio_data=audio_b64,mime_type=mime,is_active=True)
        db.add(track);db.commit()
        return jsonify({"message":f"✅ '{title}' upload ho gaya!","id":track.id})
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

@app.route('/api/music/<int:track_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_music(track_id):
    db=SessionLocal()
    try:
        t=db.query(MusicTrack).filter_by(id=track_id).first()
        if t:t.is_active=False;db.commit();return jsonify({"message":"✅ Removed!"})
        return jsonify({"error":"Not found"}),404
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

# ── Video ─────────────────────────────────────────────────────────
@app.route('/api/video')
@login_required
def get_videos():
    db=SessionLocal()
    try:
        videos=db.query(VideoTrack).filter_by(is_active=True).order_by(VideoTrack.created_at.asc()).all()
        return jsonify([{"id":v.id,"title":v.title,"description":v.description or "","video_data":v.video_data,"mime_type":v.mime_type,"thumbnail":v.thumbnail,"thumb_mime":v.thumb_mime or "image/jpeg"} for v in videos])
    finally:
        db.close()

@app.route('/api/video', methods=['POST'])
@login_required
@admin_required
def upload_video():
    import base64
    db=SessionLocal()
    try:
        if 'video' not in request.files:return jsonify({"error":"No video file"}),400
        f=request.files['video'];title=request.form.get('title',f.filename or 'Video');desc=request.form.get('description','')
        if not f or not f.filename:return jsonify({"error":"Empty file"}),400
        raw=f.read();video_b64=base64.b64encode(raw).decode('utf-8');mime=f.content_type or 'video/mp4'
        thumb_b64,thumb_mime=None,None
        if 'thumbnail' in request.files:
            th=request.files['thumbnail']
            if th and th.filename:thumb_b64=base64.b64encode(th.read()).decode('utf-8');thumb_mime=th.content_type or 'image/jpeg'
        video=VideoTrack(title=title,description=desc,video_data=video_b64,mime_type=mime,thumbnail=thumb_b64,thumb_mime=thumb_mime,is_active=True)
        db.add(video);db.commit()
        return jsonify({"message":f"✅ '{title}' upload ho gaya!","id":video.id})
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

@app.route('/api/video/<int:video_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_video(video_id):
    db=SessionLocal()
    try:
        v=db.query(VideoTrack).filter_by(id=video_id).first()
        if v:v.is_active=False;db.commit();return jsonify({"message":"✅ Removed!"})
        return jsonify({"error":"Not found"}),404
    except Exception as e:
        db.rollback();return jsonify({"error":str(e)}),500
    finally:
        db.close()

# ── Stats ─────────────────────────────────────────────────────────
@app.route('/api/stats')
@login_required
def get_stats():
    from core.database import Subscription
    db=SessionLocal()
    try:
        total_users=db.query(User).count();pro_users=db.query(Subscription).filter_by(plan='pro',is_active=True).count()
        today=datetime.utcnow().strftime('%Y-%m-%d');today_q=db.query(QueryCount).filter_by(date=today).all()
        today_total=sum(q.count for q in today_q);total_q=db.query(QueryCount).all();total_queries=sum(q.count for q in total_q)
        return jsonify({"total_users":total_users,"pro_users":pro_users,"today_queries":today_total,"total_queries":total_queries})
    finally:
        db.close()

@app.route('/health')
def health():return jsonify({"status":"healthy","version":"3.1.0"}),200

# ── YouTube Stream ────────────────────────────────────────────────
@app.route('/api/yt-stream', methods=['POST'])
@login_required
def get_yt_stream():
    try:
        import yt_dlp
        data=request.get_json();url=data.get('url','').strip()
        if not url:return jsonify({'error':'URL required!'}),400
        if 'youtube.com' not in url and 'youtu.be' not in url:return jsonify({'error':'Valid YouTube URL daalo!'}),400
        ydl_opts={'format':'best[ext=mp4][height<=480]/best[height<=480]/best','quiet':True,'no_warnings':True,'noplaylist':True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info=ydl.extract_info(url,download=False);formats=info.get('formats',[]);stream_url=None
            for f in reversed(formats):
                if f.get('ext')=='mp4' and f.get('url'):stream_url=f['url'];break
            if not stream_url:stream_url=info.get('url') or (formats[-1].get('url') if formats else None)
            title=info.get('title','YouTube Video');thumbnail=info.get('thumbnail','');duration=info.get('duration',0)
        if not stream_url:return jsonify({'error':'Stream URL extract nahi hui!'}),500
        return jsonify({'stream_url':stream_url,'title':title,'thumbnail':thumbnail,'duration':duration})
    except ImportError:
        return jsonify({'error':'yt-dlp install nahi hai!'}),500
    except Exception as e:
        return jsonify({'error':f'Error: {str(e)[:100]}'}),500

# ── Live Chat ─────────────────────────────────────────────────────
@app.route('/api/chat/messages')
@login_required
def get_chat_messages():
    db=SessionLocal()
    try:
        msgs=db.query(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(50).all()
        return jsonify([{'id':m.id,'username':m.username,'message':m.message,'is_admin':m.is_admin,'time':m.created_at.strftime('%H:%M')} for m in reversed(msgs)])
    finally:
        db.close()

@app.route('/api/chat/messages', methods=['POST'])
@login_required
def send_chat_message():
    db=SessionLocal()
    try:
        data=request.get_json();msg=data.get('message','').strip()
        if not msg or len(msg)>500:return jsonify({'error':'Invalid message'}),400
        chat=ChatMessage(username=current_user.username,message=msg,is_admin=current_user.is_admin)
        db.add(chat);db.commit()
        return jsonify({'message':'✅ Sent!','id':chat.id})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

@app.route('/api/chat/messages/<int:msg_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_chat_message(msg_id):
    db=SessionLocal()
    try:
        m=db.query(ChatMessage).filter_by(id=msg_id).first()
        if m:db.delete(m);db.commit();return jsonify({'message':'✅ Deleted!'})
        return jsonify({'error':'Not found'}),404
    finally:
        db.close()

# ── Analytics ─────────────────────────────────────────────────────
@app.route('/api/analytics/view/<int:post_id>', methods=['POST'])
@login_required
def track_post_view(post_id):
    db=SessionLocal()
    try:
        today=datetime.utcnow().strftime('%Y-%m-%d');rec=db.query(PostAnalytics).filter_by(post_id=post_id,date=today).first()
        if rec:rec.views+=1
        else:rec=PostAnalytics(post_id=post_id,views=1,clicks=0,date=today);db.add(rec)
        db.commit();return jsonify({'ok':True})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

@app.route('/api/analytics/click/<int:post_id>', methods=['POST'])
@login_required
def track_post_click(post_id):
    db=SessionLocal()
    try:
        today=datetime.utcnow().strftime('%Y-%m-%d');rec=db.query(PostAnalytics).filter_by(post_id=post_id,date=today).first()
        if rec:rec.clicks+=1
        else:rec=PostAnalytics(post_id=post_id,views=0,clicks=1,date=today);db.add(rec)
        db.commit();return jsonify({'ok':True})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

@app.route('/api/analytics')
@login_required
@admin_required
def get_analytics():
    db=SessionLocal()
    try:
        posts=db.query(ScreenPost).filter_by(is_active=True).all()
        result=[]
        for p in posts:
            recs=db.query(PostAnalytics).filter_by(post_id=p.id).all()
            total_views=sum(r.views for r in recs);total_clicks=sum(r.clicks for r in recs)
            result.append({'post_id':p.id,'title':p.title,'views':total_views,'clicks':total_clicks,'ctr':round(total_clicks/total_views*100,1) if total_views else 0})
        result.sort(key=lambda x:x['views'],reverse=True)
        return jsonify(result)
    finally:
        db.close()

# ── Scheduled Posts ───────────────────────────────────────────────
@app.route('/api/scheduled', methods=['GET'])
@login_required
@admin_required
def get_scheduled():
    db=SessionLocal()
    try:
        posts=db.query(ScheduledPost).filter_by(is_published=False).order_by(ScheduledPost.scheduled_at).all()
        return jsonify([{'id':p.id,'title':p.title,'content':p.content,'post_type':p.post_type,'scheduled_at':p.scheduled_at.strftime('%Y-%m-%d %H:%M'),'is_published':p.is_published} for p in posts])
    finally:
        db.close()

@app.route('/api/scheduled', methods=['POST'])
@login_required
@admin_required
def create_scheduled():
    import base64
    db=SessionLocal()
    try:
        title=request.form.get('title','');content=request.form.get('content','');post_type=request.form.get('post_type','text');affiliate_url=request.form.get('affiliate_url','');scheduled_at=request.form.get('scheduled_at','')
        if not title or not scheduled_at:return jsonify({'error':'Title aur time required!'}),400
        from datetime import datetime as dt
        sched_time=dt.strptime(scheduled_at,'%Y-%m-%dT%H:%M')
        img_data,img_mime=None,None
        if post_type=='image' and 'image' in request.files:
            f=request.files['image']
            if f and f.filename:img_data=base64.b64encode(f.read()).decode();img_mime=f.content_type or 'image/jpeg'
        sp=ScheduledPost(title=title,content=content,post_type=post_type,affiliate_url=affiliate_url,scheduled_at=sched_time,image_data=img_data,image_mime=img_mime)
        db.add(sp);db.commit()
        return jsonify({'message':f'✅ "{title}" scheduled!','id':sp.id})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

@app.route('/api/scheduled/<int:sp_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_scheduled(sp_id):
    db=SessionLocal()
    try:
        sp=db.query(ScheduledPost).filter_by(id=sp_id).first()
        if sp:db.delete(sp);db.commit();return jsonify({'message':'✅ Removed!'})
        return jsonify({'error':'Not found'}),404
    finally:
        db.close()

@app.route('/api/scheduled/publish', methods=['POST'])
@login_required
@admin_required
def publish_due_posts():
    db=SessionLocal()
    try:
        now=datetime.utcnow()
        due=db.query(ScheduledPost).filter(ScheduledPost.scheduled_at<=now,ScheduledPost.is_published==False).all()
        published=0
        for sp in due:
            post=ScreenPost(title=sp.title,content=sp.content,post_type=sp.post_type,affiliate_url=sp.affiliate_url,image_data=sp.image_data,image_mime=sp.image_mime,is_active=True)
            db.add(post);sp.is_published=True;published+=1
        db.commit()
        return jsonify({'message':f'✅ {published} posts published!'})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

# ── Push Notifications ────────────────────────────────────────────
VAPID_PUBLIC_KEY=os.getenv('VAPID_PUBLIC_KEY','');VAPID_PRIVATE_KEY=os.getenv('VAPID_PRIVATE_KEY','');VAPID_EMAIL=os.getenv('VAPID_EMAIL','mailto:admin@dsagent.com')

@app.route('/api/push/vapid-public-key')
def get_vapid_key():return jsonify({'publicKey':VAPID_PUBLIC_KEY})

@app.route('/api/push/subscribe', methods=['POST'])
@login_required
def push_subscribe():
    db=SessionLocal()
    try:
        data=request.get_json()
        sub=PushSubscription(user_id=current_user.id,endpoint=data['endpoint'],p256dh=data['keys']['p256dh'],auth=data['keys']['auth'])
        db.add(sub);db.commit()
        return jsonify({'message':'✅ Subscribed!'})
    except Exception as e:
        db.rollback();return jsonify({'error':str(e)}),500
    finally:
        db.close()

@app.route('/api/push/send', methods=['POST'])
@login_required
@admin_required
def send_push():
    db=SessionLocal()
    try:
        data=request.get_json();title=data.get('title','DS Agent');body=data.get('body','')
        subs=db.query(PushSubscription).all();sent=0
        if VAPID_PRIVATE_KEY:
            try:
                from pywebpush import webpush,WebPushException
                import json
                for s in subs:
                    try:
                        webpush(subscription_info={'endpoint':s.endpoint,'keys':{'p256dh':s.p256dh,'auth':s.auth}},data=json.dumps({'title':title,'body':body}),vapid_private_key=VAPID_PRIVATE_KEY,vapid_claims={'sub':VAPID_EMAIL})
                        sent+=1
                    except:pass
            except ImportError:pass
        return jsonify({'message':f'✅ {sent}/{len(subs)} notifications sent!'})
    except Exception as e:
        return jsonify({'error':str(e)}),500
    finally:
        db.close()

if __name__ == '__main__':
    port=int(os.environ.get('PORT',10000))
    app.run(host='0.0.0.0',port=port)
