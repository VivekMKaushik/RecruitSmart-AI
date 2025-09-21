import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import streamlit as st
from datetime import datetime


def send_feedback_email(student_email, student_name, job_title, score, missing_skills, suggestions, company_name="Our Company"):
    """Send feedback email to student with analysis results"""
    
    # Email configuration
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    
    # Check if email credentials are configured
    if not email_address or not email_password:
        st.error("Email credentials not configured. Please set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables.")
        return False, "Email credentials not configured"
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = student_email
        msg['Subject'] = f"Application Feedback for {job_title} Position"
        
        # Format missing skills
        if missing_skills and len(missing_skills) > 0:
            if isinstance(missing_skills, list):
                missing_skills_str = ", ".join(missing_skills[:5])
            else:
                missing_skills_str = str(missing_skills)
        else:
            missing_skills_str = "No specific skills missing"
        
        # Email body
        body = f"""
        Dear {student_name},
        
        Thank you for applying to the {job_title} position at {company_name}.
        
        We've completed our initial review of your application and would like to share some feedback:
        
        - Your application score: {score}/100
        - Key areas for improvement: {missing_skills_str}
        
        Suggestions for improvement:
        {suggestions}
        
        We encourage you to continue developing your skills and applying for future opportunities.
        
        Best regards,
        The {company_name} Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        
        # Log the email sending
        print(f"✅ Email sent to {student_email} at {datetime.now()}")
        return True, "Feedback email sent successfully"
    
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

# New function to send email from the application interface
def send_email_from_ui(recipient_email, subject, body):
    """Send email from the UI with proper error handling"""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    
    if not email_address or not email_password:
        return False, "Email credentials not configured"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        
        return True, "Email sent successfully"
    
    except Exception as e:
        return False, f"Email sent successfully: {str(e)}"