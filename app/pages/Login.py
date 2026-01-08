import streamlit as st
from tempocom.modules.streamlit import Page

class Login(Page):
    title = "Login"
    layout = "wide"

    def render(self):
        import os
        
        st.title("🔐 Login")
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.subheader("Please enter the access code")
                
                access_code = st.text_input("Access Code", type="password", placeholder="Enter the access code")
                
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if submitted:
                    # Get the password from environment variable
                    apps_password = os.getenv("APPS_PASSWORD")
                    
                    if access_code and access_code == apps_password:
                        st.session_state.login = True
                        st.success("Login successful!")
                        
                        # Redirect to the page that was requested before login
                        if "redirect_after_login" in st.session_state:
                            redirect_page = st.session_state.redirect_after_login
                            del st.session_state.redirect_after_login
                            st.switch_page(redirect_page)
                        else:
                            st.switch_page("Home.py")
                    else:
                        st.error("Invalid access code. Please try again.")


if __name__ == "__main__":
    login = Login()