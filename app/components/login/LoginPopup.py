import streamlit as st

class LoginPopup:
    def __init__(self):
        self.login_popup = st.sidebar.empty()

    def render(self):
        self.login_popup.write("Login")
        if self.login_popup.button("Login"):
            self.login_popup.write("Login successful")
        else:
            self.login_popup.write("Login failed")