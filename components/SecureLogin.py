import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText
import json
import os
from dotenv import load_dotenv




class SecureLogin:
    def __init__(self, use_2fa=False):
        self.environment = os.getenv('ENVIRONMENT')
        load_dotenv('.app.env')
        self.PASSWORD = os.environ.get("APP_PASSWORD")
        self.use_2fa = use_2fa
        if "step" not in st.session_state:
            st.session_state.step = "login"
        if "email" not in st.session_state:
            st.session_state.email = ""
        if "otp" not in st.session_state:
            st.session_state.otp = None

    def send_otp_email(self, to_email, otp_code):
        msg = MIMEText(f"Here is your verification code: {otp_code}")
        msg["Subject"] = "Your login code for tempocom"
        msg["From"] = os.environ.get("MAIL_USER")
        msg["To"] = to_email

        try:
            with smtplib.SMTP("smtp.office365.com", 587) as server:
                server.starttls()
                server.login(os.environ.get("MAIL_USER"), os.environ.get("MAIL_PASSWORD"))
                server.send_message(msg)
        except Exception as e:
            st.error(f"Error sending email: {str(e)}")
            st.info("For demo purposes, here is your OTP code: " + otp_code)

    def render(self, title: str):
        labs = json.load(open("constants.json"))['labs']
        current_lab = next((lab for lab in labs if lab["title"] == title), None)
        if not current_lab['private']:
            return True
        elif self.environment == 'dev':
            return True
        
        if st.session_state.step == "logged_in":
            return True
        
        if st.session_state.step == "login":
            st.title("Secure Login")
            if self.use_2fa:
                email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if self.PASSWORD != password:
                    st.error("Invalid credentials")
                    return False
                    
                if self.use_2fa:
                    if not (email.endswith("@infrabel.be") or email.endswith("@dtsc.be")):
                        st.error("Access restricted to DTSC and Infrabel domains")
                        return False
                    otp = str(random.randint(100000, 999999))
                    st.session_state.otp = otp
                    st.session_state.email = email
                    self.send_otp_email(email, otp)
                    st.success(f"Code sent to {email}")
                    st.session_state.step = "verify"
                    st.rerun()
                else:
                    st.success("Login successful ✅")
                    st.session_state.step = "logged_in"
                    st.rerun()

        if st.session_state.step == "verify":
            st.title("2FA Verification")
            code = st.text_input("Enter the code received by email")
            if st.button("Validate code"):
                if code != st.session_state.otp:
                    st.error("Invalid code")
                    return False
                st.success("Login successful ✅")
                st.session_state.step = "logged_in"
                st.rerun()

        return False