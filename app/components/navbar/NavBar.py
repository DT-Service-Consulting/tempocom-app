import streamlit as st
from pathlib import Path
import os


def NavBar():
    css_path = Path(__file__).with_name("navbar.css")
    css = css_path.read_text(encoding="utf-8")

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    col1, col2, col3,col4,col5,col6 = st.columns([1, 2, 1, 1, 1,1])
    
    with col1:
        st.image(f"{os.getenv("ASSETS_PATH")}/logo.png", width=100)
    with col2:
        nav_cols = st.columns(4)
        with nav_cols[0]:
            if st.button("Home", type="tertiary"):
                st.switch_page("Home.py")

    with col3:pass
    with col4:pass
    with col5:pass
    with col6:
        with nav_cols[1]:
            if st.button("The project", type="tertiary"):
                st.switch_page("pages/About.py")
        
        with nav_cols[2]:
            if st.button("About us", type="tertiary"):
                import webbrowser
                webbrowser.open_new_tab("https://brain.dtsc.be")
        
        with nav_cols[3]:
            if st.button("Github", type="tertiary"):
                import webbrowser
                webbrowser.open_new_tab("https://github.com/DT-Service-Consulting/tempocom-app")
    
    with col2:
        pass
       

    #st.markdown("<div class=\"navbar-spacer\"></div>", unsafe_allow_html=True)
    #st.divider()
