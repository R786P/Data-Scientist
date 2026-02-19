import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_report(receiver_email, plot_path, summary_text="Please find the attached analysis report."):
    sender_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS") # Use App Password for Gmail
    
    if not sender_email or not password:
        return "❌ Email credentials missing in Environment Variables."

    msg = MIMEMultipart()
    msg['From'] = f"DS Agent <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = "📊 Your Data Analysis Report"

    msg.attach(MIMEText(summary_text, 'plain'))

    # Attachment logic
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=report.png")
            msg.attach(part)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return "✅ Email sent successfully!"
    except Exception as e:
        return f"❌ Email failed: {str(e)}"
