import streamlit as st
import os

def app_card(app: dict):

    card_id = f"card_{app['title'].replace(' ', '_')}"
    is_logged_in = st.session_state.get("login", False)
    
    with st.container(border=True):
        with st.container(border=True, height=200):
            image_url = f"{os.getenv('ASSETS_PATH')}/{app['image_path']}"
            st.image(image_url, use_container_width=True)
            
            # Add CSS to prevent scrolling
            st.markdown("""
                <style>
                div[data-testid="stVerticalBlock"] > div:has(img) {
                    overflow: hidden !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        st.subheader(app['title'])
        st.caption(app['subtitle'])

        if not app["available"]:
            if st.button("COMMING SOON", key=f"btn_card_{card_id}", use_container_width=True, disabled=True, type="tertiary"):
                pass
        
        elif app["redirect"] and (not app["private"] or is_logged_in):
            if st.button("Open the App", key=f"btn_card_{card_id}", use_container_width=True, type="primary"):
                st.switch_page(app["redirect"])
        

        elif app["private"] and not is_logged_in:
            if st.button("Login to access this app 🔐", key=f"btn_card_{card_id}", use_container_width=True, type="secondary"):
                st.session_state.redirect_after_login = app["redirect"]
                st.switch_page("pages/Login.py")
